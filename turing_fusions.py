from __future__ import annotations

import functools
import logging
import operator
import sys
import types
from collections.abc import Sequence

import torch


LOG = logging.getLogger("comfyui-svdint4")
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


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
        block._svdint4_original_forward = block.forward
        block._svdint4_turing_device_index = index
        block.forward = types.MethodType(_fused_block_forward, block)
    LOG.info("Enabled fused segmented RMSNorm+AdaLN on %d Turing blocks", len(candidates))
    return len(candidates)
