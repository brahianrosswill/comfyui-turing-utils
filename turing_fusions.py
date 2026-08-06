"""Generic sm75 fusions; model-specific installation lives in adapters."""

from __future__ import annotations

import functools
import operator
from collections.abc import Sequence

import torch


_W8_LAYOUT = "TensorWiseINT8Layout"
_W4_LAYOUT = "TensorCoreConvRotW4A4Layout"


def convrot_weight_kind(weight: torch.Tensor) -> str | None:
    params = getattr(weight, "_params", None)
    if getattr(params, "orig_dtype", None) is not torch.bfloat16:
        return None
    if getattr(params, "transposed", False) or getattr(params, "convrot_groupsize", None) != 256:
        return None
    layout = getattr(weight, "_layout_cls", None)
    if layout == _W8_LAYOUT and getattr(params, "convrot", False):
        return "w8a8"
    if layout == _W4_LAYOUT and getattr(params, "quant_group_size", None) == 64:
        linear_dtype = getattr(params, "linear_dtype", None)
        return f"w4a{linear_dtype[-1]}" if linear_dtype in {"int4", "int8"} else None
    return None


def is_turing_convrot_linear(linear: torch.nn.Module) -> bool:
    return convrot_weight_kind(getattr(linear, "weight", None)) is not None


def turing_linear_input_act(linear: torch.nn.Module, x: torch.Tensor, input_act: str):
    """Fold an activation into any supported Turing ConvRot input quantizer."""
    import comfy.model_management
    import comfy.ops
    import comfy.quant_ops

    try:
        from .turing_ops import convrot_w4a4_linear, int8_linear, is_supported_turing_device
    except ImportError:
        from turing_ops import convrot_w4a4_linear, int8_linear, is_supported_turing_device

    if (
        x.dtype != torch.bfloat16
        or comfy.model_management.in_training
        or not is_supported_turing_device(x.device)
        or not is_turing_convrot_linear(linear)
    ):
        return comfy.ops.linear_input_act(linear, x, input_act)

    weight, bias, offload_stream = comfy.ops.cast_bias_weight(
        linear,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=True,
    )
    try:
        kind = convrot_weight_kind(weight)
        if kind is None:
            activated = comfy.ops.INPUT_ACT_EAGER[input_act](x)
            return torch.nn.functional.linear(activated, weight, bias)
        if kind == "w8a8":
            qdata, scale = comfy.quant_ops.TensorWiseINT8Layout.get_plain_tensors(weight)
            return int8_linear(
                x,
                qdata,
                scale,
                bias=bias,
                out_dtype=x.dtype,
                convrot=True,
                convrot_groupsize=weight._params.convrot_groupsize,
                input_act=input_act,
            )

        qdata, scale = comfy.quant_ops.TensorCoreConvRotW4A4Layout.get_plain_tensors(weight)
        params = weight._params
        return convrot_w4a4_linear(
            x,
            qdata,
            scale,
            bias=bias,
            convrot_groupsize=params.convrot_groupsize,
            quant_group_size=params.quant_group_size,
            linear_dtype=params.linear_dtype,
            input_act=input_act,
        )
    finally:
        comfy.ops.uncast_bias_weight(linear, weight, bias, offload_stream)


def _normalized_segments(
    segments: Sequence[tuple[int, int, int]],
    rows: int,
    parameter_rows: int,
) -> tuple[int, ...]:
    if rows <= 0 or parameter_rows <= 0:
        raise ValueError("segmented RMSNorm+AdaLN dimensions must be positive")
    flat: list[int] = []
    cursor = 0
    for segment in segments:
        if len(segment) != 3:
            raise ValueError("each modulation segment must contain start, stop, and row")
        try:
            start, stop, modulation_row = (operator.index(value) for value in segment)
        except TypeError as exc:
            raise ValueError("modulation segment values must be integers") from exc
        if start != cursor or stop <= start:
            raise ValueError("modulation segments must cover the input contiguously")
        if modulation_row < 0 or modulation_row >= parameter_rows:
            raise ValueError("modulation segment row is outside the scale/shift table")
        flat.extend((start, stop, modulation_row))
        cursor = stop
    if cursor != rows:
        raise ValueError("modulation segments must cover every input row")
    return tuple(flat)


@functools.lru_cache(maxsize=32)
def _cached_segment_table(flat: tuple[int, ...], device_index: int) -> torch.Tensor:
    return torch.tensor(flat, dtype=torch.int32, device=torch.device("cuda", device_index)).view(-1, 3)


def _segment_table(
    segments: Sequence[tuple[int, int, int]],
    rows: int,
    parameter_rows: int,
    device: torch.device,
) -> torch.Tensor:
    flat = _normalized_segments(segments, rows, parameter_rows)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _cached_segment_table(flat, index)


def segmented_rms_adaln(
    norm: torch.nn.Module,
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    segments: Sequence[tuple[int, int, int]],
) -> torch.Tensor:
    """Run the bundled affine RMSNorm plus segmented AdaLN operator."""
    import comfy.ops

    comfy.ops.run_every_op()
    weight, bias, offload_stream = comfy.ops.cast_bias_weight(norm, x, offloadable=True)
    try:
        if bias is not None:
            raise RuntimeError("segmented RMSNorm+AdaLN does not support a biased norm")
        if weight is None:
            weight = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
        scale = scale.to(device=x.device, dtype=x.dtype)
        shift = shift.to(device=x.device, dtype=x.dtype)
        table = _segment_table(segments, x.shape[0], scale.shape[0], x.device)
        try:
            from svdint4 import turing_segmented_rms_adaln
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Turing RMSNorm+AdaLN fusion requires an updated svdint4-kernel; reinstall the kernel package"
            ) from exc
        return turing_segmented_rms_adaln(x, weight, scale, shift, table, float(norm.eps))
    finally:
        comfy.ops.uncast_bias_weight(norm, weight, bias, offload_stream)
