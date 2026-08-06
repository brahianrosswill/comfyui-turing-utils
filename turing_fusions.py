from __future__ import annotations

import functools
import logging
import operator
import sys
import types
from collections import Counter
from collections.abc import Sequence

import torch


LOG = logging.getLogger("comfyui-svdint4")
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _w8a8_weight_signature(
    linear: torch.nn.Module,
) -> tuple[str, str, bool, bool, int | None, str]:
    weight = getattr(linear, "weight", None)
    params = getattr(weight, "_params", None)
    layout = getattr(weight, "_layout_cls", type(weight).__name__)
    orig_dtype = getattr(params, "orig_dtype", getattr(weight, "dtype", None))
    return (
        str(layout),
        str(orig_dtype),
        bool(getattr(params, "transposed", False)),
        bool(getattr(params, "convrot", False)),
        getattr(params, "convrot_groupsize", None),
        str(getattr(linear, "quant_format", None)),
    )


def _is_turing_w8a8_weight(weight: torch.Tensor) -> bool:
    params = getattr(weight, "_params", None)
    return (
        getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout"
        and getattr(params, "orig_dtype", None) == torch.bfloat16
        and not getattr(params, "transposed", False)
        and getattr(params, "convrot", False)
        and getattr(params, "convrot_groupsize", None) == 256
    )


def _is_turing_w8a8_linear(linear: torch.nn.Module) -> bool:
    return _is_turing_w8a8_weight(getattr(linear, "weight", None))


def _format_counts(values) -> str:
    counts = Counter(values)
    return ",".join(
        f"{value}:{count}"
        for value, count in sorted(counts.items(), key=lambda item: str(item[0]))
    )


def _audit_turing_fc2(blocks: Sequence[torch.nn.Module]) -> int:
    linears = [
        block.mlp.fc2
        for block in blocks
        if hasattr(block.mlp, "fc2") and hasattr(block.mlp.fc2, "weight")
    ]
    if not linears:
        return 0

    signatures = [_w8a8_weight_signature(linear) for linear in linears]
    eligible = sum(_is_turing_w8a8_linear(linear) for linear in linears)
    LOG.info(
        "Turing fc2 audit: blocks=%d eligible_w8a8=%d layouts=[%s] "
        "orig_dtypes=[%s] transposed=[%s] convrot=[%s] quant_formats=[%s]",
        len(linears),
        eligible,
        _format_counts(signature[0] for signature in signatures),
        _format_counts(signature[1] for signature in signatures),
        _format_counts(signature[2] for signature in signatures),
        _format_counts(signature[3] for signature in signatures),
        _format_counts(signature[5] for signature in signatures),
    )
    LOG.info(
        "Turing fc2 ConvRot group sizes: [%s]",
        _format_counts(signature[4] for signature in signatures),
    )
    if eligible != len(linears):
        LOG.warning(
            "Only %d/%d Turing MLP fc2 layers can use fused W8A8 SwiGLU. "
            "The checkpoint must store blocks.*.mlp.fc2 as TensorWiseINT8Layout "
            "with orig_dtype=torch.bfloat16, transposed=false, convrot=true, and "
            "convrot_groupsize=256; "
            "SVDInt4 will not silently quantize dense checkpoint weights at runtime.",
            eligible,
            len(linears),
        )
    return eligible


def turing_linear_input_act(linear: torch.nn.Module, x: torch.Tensor, input_act: str):
    """Route eligible Turing W8A8 input activations directly to the bundled backend."""
    import comfy.model_management
    import comfy.ops
    import comfy.quant_ops

    if (
        x.dtype != torch.bfloat16
        or comfy.model_management.in_training
        or not _is_turing_w8a8_linear(linear)
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
        if not _is_turing_w8a8_weight(weight):
            activated = comfy.ops.INPUT_ACT_EAGER[input_act](x)
            return torch.nn.functional.linear(activated, weight, bias)

        qdata, scale = comfy.quant_ops.TensorWiseINT8Layout.get_plain_tensors(weight)
        try:
            from .turing_ops import int8_linear
        except ImportError:
            from turing_ops import int8_linear
        return int8_linear(
            x,
            qdata,
            scale,
            bias=bias,
            out_dtype=x.dtype,
            convrot=True,
            convrot_groupsize=getattr(weight._params, "convrot_groupsize", 256),
            input_act=input_act,
        )
    finally:
        comfy.ops.uncast_bias_weight(linear, weight, bias, offload_stream)


def _turing_mlp_forward(self, x: torch.Tensor):
    if x.dtype != torch.bfloat16 or not _is_turing_w8a8_linear(self.fc2):
        return self._svdint4_original_forward(x)
    return turing_linear_input_act(self.fc2, self.fc1(x), "swiglu")


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
    device = torch.device("cuda", device_index)
    return torch.tensor(flat, dtype=torch.int32, device=device).view(-1, 3)


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
    """Run the generic bundled affine RMSNorm + segmented AdaLN operator."""
    import comfy.ops

    comfy.ops.run_every_op()
    weight, bias, offload_stream = comfy.ops.cast_bias_weight(
        norm, x, offloadable=True
    )
    try:
        if weight is None:
            weight = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
        scale = scale.to(device=x.device, dtype=x.dtype)
        shift = shift.to(device=x.device, dtype=x.dtype)
        table = _segment_table(
            segments,
            x.shape[0],
            scale.shape[0],
            x.device,
        )
        try:
            from svdint4 import turing_segmented_rms_adaln
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Turing RMSNorm+AdaLN fusion requires an updated svdint4-kernel; "
                "reinstall the kernel package"
            ) from exc
        return turing_segmented_rms_adaln(
            x,
            weight,
            scale,
            shift,
            table,
            float(norm.eps),
        )
    finally:
        comfy.ops.uncast_bias_weight(norm, weight, bias, offload_stream)


def _can_fuse(block: torch.nn.Module, x: torch.Tensor, t_emb: torch.Tensor) -> bool:
    if x.device.type != "cuda" or x.dtype not in _SUPPORTED_DTYPES or x.ndim != 2:
        return False
    index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    if index != block._svdint4_turing_device_index:
        return False
    return not (torch.is_grad_enabled() and (x.requires_grad or t_emb.requires_grad))


def _fused_block_forward(
    self,
    x,
    t_emb,
    mod_segments,
    rope_freqs,
    transformer_options={},
):
    if not _can_fuse(self, x, t_emb):
        return self._svdint4_original_forward(
            x,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options=transformer_options,
        )

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
    h = segmented_rms_adaln(self.norm1, x, shift_msa, scale_msa, mod_segments)
    model_module = sys.modules[type(self).__module__]
    x = model_module._mod_gate(
        x,
        gate_msa,
        self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options),
        mod_segments,
    )
    h = segmented_rms_adaln(self.norm2, x, shift_mlp, scale_mlp, mod_segments)
    return model_module._mod_gate(x, gate_mlp, self.mlp(h), mod_segments)


def _compatible_block(block: torch.nn.Module) -> bool:
    return all(
        hasattr(block, name)
        for name in ("norm1", "norm2", "adaln_proj", "attn", "mlp")
    )


def apply_turing_fusions(model, device: torch.device) -> int:
    """Enable bundled generic fusions on compatible blocks of an sm75 model."""
    try:
        from .turing_ops import is_supported_turing_device
    except ImportError:
        from turing_ops import is_supported_turing_device

    if not is_supported_turing_device(device):
        return 0

    try:
        from comfy.ldm.minimax.model import DiTBlock
    except ImportError:
        return 0

    root = getattr(model, "model", model)
    candidates = [
        block
        for block in root.modules()
        if isinstance(block, DiTBlock)
        and _compatible_block(block)
        and not hasattr(block, "_svdint4_original_forward")
    ]
    if not candidates:
        return 0

    eligible_fc2 = _audit_turing_fc2(candidates)

    try:
        from svdint4 import turing_segmented_rms_adaln
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Turing RMSNorm+AdaLN fusion requires an updated svdint4-kernel; "
            "reinstall the kernel package"
        ) from exc
    if not callable(turing_segmented_rms_adaln):
        raise RuntimeError("svdint4-kernel does not provide segmented RMSNorm+AdaLN")

    index = device.index if device.index is not None else torch.cuda.current_device()
    for block in candidates:
        if (
            _is_turing_w8a8_linear(getattr(block.mlp, "fc2", block.mlp))
            and not hasattr(block.mlp, "_svdint4_original_forward")
        ):
            block.mlp._svdint4_original_forward = block.mlp.forward
            block.mlp.forward = types.MethodType(_turing_mlp_forward, block.mlp)
        block._svdint4_original_forward = block.forward
        block._svdint4_turing_device_index = index
        block.forward = types.MethodType(_fused_block_forward, block)
    LOG.info("Enabled fused segmented RMSNorm+AdaLN on %d Turing blocks", len(candidates))
    if eligible_fc2:
        LOG.info(
            "Enabled direct fused W8A8 SwiGLU dispatch on %d Turing MLP fc2 layers",
            eligible_fc2,
        )
    return len(candidates)
