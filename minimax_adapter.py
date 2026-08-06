"""Optional MiniMax integration for generic Turing fusions."""

from __future__ import annotations

import inspect
import logging
import types
from collections import Counter
from collections.abc import Sequence

import torch

try:
    from .turing_fusions import (
        convrot_weight_kind,
        is_turing_convrot_linear,
        segmented_rms_adaln,
        turing_linear_input_act,
    )
    from .turing_ops import is_supported_turing_device
except ImportError:
    from turing_fusions import (
        convrot_weight_kind,
        is_turing_convrot_linear,
        segmented_rms_adaln,
        turing_linear_input_act,
    )
    from turing_ops import is_supported_turing_device


LOG = logging.getLogger("comfyui-svdint4")
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BLOCK_FORWARD_PARAMETERS = (
    "x",
    "t_emb",
    "mod_segments",
    "rope_freqs",
    "transformer_options",
)


def _format_counts(values) -> str:
    counts = Counter(values)
    return ",".join(
        f"{value}:{count}"
        for value, count in sorted(counts.items(), key=lambda item: str(item[0]))
    )


def _audit_fc2(blocks: Sequence[torch.nn.Module]) -> int:
    linears = []
    for block in blocks:
        mlp = getattr(block, "mlp", None)
        if hasattr(mlp, "fc2"):
            linears.append(mlp.fc2)
    if not linears:
        return 0
    kinds = [
        convrot_weight_kind(getattr(linear, "weight", None)) or "other"
        for linear in linears
    ]
    eligible = sum(kind != "other" for kind in kinds)
    LOG.info(
        "MiniMax Turing fc2 dispatch: blocks=%d eligible=%d formats=[%s]",
        len(linears),
        eligible,
        _format_counts(kinds),
    )
    return eligible


def _compatible_block_forward(block_type: type[torch.nn.Module]) -> bool:
    parameters = tuple(inspect.signature(block_type.forward).parameters)
    return parameters == ("self", *_BLOCK_FORWARD_PARAMETERS)


def _can_fuse(x: torch.Tensor, t_emb: torch.Tensor, device_index: int) -> bool:
    if x.device.type != "cuda" or x.dtype not in _SUPPORTED_DTYPES or x.ndim != 2:
        return False
    index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    if index != device_index:
        return False
    return not (torch.is_grad_enabled() and (x.requires_grad or t_emb.requires_grad))


def _make_mlp_forward(mlp: torch.nn.Module):
    original = mlp.forward

    def forward(self, x: torch.Tensor):
        if x.dtype != torch.bfloat16 or not is_turing_convrot_linear(self.fc2):
            return original(x)
        return turing_linear_input_act(self.fc2, self.fc1(x), "swiglu")

    return types.MethodType(forward, mlp)


def _make_block_forward(block: torch.nn.Module, device_index: int, mod_gate):
    original = block.forward

    def forward(
        self,
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options={},
    ):
        if not _can_fuse(x, t_emb, device_index):
            return original(
                x,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
            )

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = segmented_rms_adaln(self.norm1, x, shift_msa, scale_msa, mod_segments)
        x = mod_gate(
            x,
            gate_msa,
            self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options),
            mod_segments,
        )
        h = segmented_rms_adaln(self.norm2, x, shift_mlp, scale_mlp, mod_segments)
        return mod_gate(x, gate_mlp, self.mlp(h), mod_segments)

    return types.MethodType(forward, block)


def apply_minimax_adapter(model, device: torch.device) -> int:
    """Install MiniMax-only forward substitutions through the ModelPatcher."""
    if not is_supported_turing_device(device):
        return 0
    if not hasattr(model, "add_object_patch"):
        raise RuntimeError("MiniMax Turing integration requires a ComfyUI ModelPatcher")

    try:
        from comfy.ldm.minimax.model import DiTBlock, _mod_gate
    except ImportError:
        return 0
    if not _compatible_block_forward(DiTBlock):
        LOG.warning("MiniMax Turing fusions are disabled because the DiTBlock forward contract changed")
        return 0

    root = getattr(model, "model", model)
    candidates = [
        (name, block)
        for name, block in root.named_modules()
        if name and isinstance(block, DiTBlock)
    ]
    if not candidates:
        return 0

    eligible_fc2 = _audit_fc2([block for _, block in candidates])
    index = device.index if device.index is not None else torch.cuda.current_device()
    block_fusions = 0
    mlp_fusions = 0
    try:
        from svdint4 import turing_segmented_rms_adaln
    except (ImportError, AttributeError):
        turing_segmented_rms_adaln = None

    for name, block in candidates:
        if hasattr(block.mlp, "fc2") and is_turing_convrot_linear(block.mlp.fc2):
            model.add_object_patch(f"{name}.mlp.forward", _make_mlp_forward(block.mlp))
            mlp_fusions += 1
        if callable(turing_segmented_rms_adaln):
            model.add_object_patch(f"{name}.forward", _make_block_forward(block, index, _mod_gate))
            block_fusions += 1

    if block_fusions:
        LOG.info("Enabled MiniMax segmented RMSNorm+AdaLN on %d Turing blocks", block_fusions)
    if mlp_fusions:
        LOG.info("Enabled MiniMax fused ConvRot SwiGLU on %d Turing MLP layers", mlp_fusions)
    if eligible_fc2 and mlp_fusions != eligible_fc2:
        raise RuntimeError("MiniMax Turing fc2 adapter did not patch every eligible layer")
    return max(block_fusions, mlp_fusions)
