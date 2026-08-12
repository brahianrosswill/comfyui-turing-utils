"""MiniMax H3 memory planning and Turing block fusions."""

from __future__ import annotations

import contextvars
import inspect
import logging
import math
import types
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ...kernel_api import load_kernel_package
from .layout import (
    ATTENTION_LAYOUT_KEY,
    RUNTIME_CONTEXT_ATTR,
    RUNTIME_OUTER_WRAPPER_KEY,
    ensure_minimax_attention_layout_provider,
    make_minimax_runtime_context_wrapper,
    mark_forward_as_minimax_layout_provider,
    minimax_temporal_topology,
    publish_minimax_attention_layout,
)
from ...quantization.fusions import (
    convrot_weight_kind,
    is_turing_convrot_linear,
    segmented_rms_adaln,
    turing_linear_input_act,
)
from ...quantization.dispatch import is_supported_turing_device, turing_int8_workspace_bytes


LOG = logging.getLogger("comfyui-turing-utils")
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BLOCK_FORWARD_PARAMETERS = (
    "x",
    "t_emb",
    "mod_segments",
    "rope_freqs",
    "transformer_options",
)
_MEMORY_SHAPE_KEY = "turing_utils_minimax_packed_sequence"
_MEMORY_CONTEXT_ATTR = RUNTIME_CONTEXT_ATTR
_MEMORY_ADAPTER_ATTR = "_turing_utils_minimax_memory_adapter"
_OUTER_SAMPLE_WRAPPER_KEY = RUNTIME_OUTER_WRAPPER_KEY
_ATTENTION_LAYOUT_KEY = ATTENTION_LAYOUT_KEY


class _MiniMaxMemoryShape(list):
    """BaseModel-compatible synthetic shape carrying exact H3 row metadata."""

    def __init__(
        self,
        equivalent_area: int,
        *,
        full_rows: int,
        target_rows: int,
        target_visual_rows: int,
        target_audio_rows: int,
        visual_condition_rows: int,
        audio_condition_rows: int,
        hidden_size: int,
        video_row_width: int,
        audio_row_width: int,
    ):
        super().__init__((1, 1, max(int(equivalent_area), 0)))
        self.equivalent_area = max(int(equivalent_area), 0)
        self.full_rows = int(full_rows)
        self.target_rows = int(target_rows)
        self.target_visual_rows = int(target_visual_rows)
        self.target_audio_rows = int(target_audio_rows)
        self.visual_condition_rows = int(visual_condition_rows)
        self.audio_condition_rows = int(audio_condition_rows)
        self.hidden_size = int(hidden_size)
        self.video_row_width = int(video_row_width)
        self.audio_row_width = int(audio_row_width)

    def explicit_condition_bytes(self, dtype_size: int) -> int:
        """Known incremental buffers not allowed to fall below the heuristic."""
        condition_rows = max(self.full_rows - self.target_rows, 0)
        packed_hidden = condition_rows * self.hidden_size * int(dtype_size)

        visual_fp32 = self.visual_condition_rows * self.video_row_width * 4
        if self.visual_condition_rows:
            # With visual conditions H3 keeps the individual condition rows and
            # also assembles one target+condition FP32 row buffer.
            visual_fp32 += (
                self.target_visual_rows + self.visual_condition_rows
            ) * self.video_row_width * 4

        audio_fp32 = self.audio_condition_rows * self.audio_row_width * 4
        if self.audio_condition_rows:
            audio_fp32 += (
                self.target_audio_rows + self.audio_condition_rows
            ) * self.audio_row_width * 4
        return packed_hidden + visual_fp32 + audio_fp32


class _MiniMaxMemoryCond:
    """Lightweight model condition exposing the synthetic shape at runtime."""

    def __init__(self, plan: _MiniMaxMemoryShape):
        self.cond = plan

    def process_cond(self, batch_size, **kwargs):
        return self

    def can_concat(self, other):
        return (
            isinstance(other, _MiniMaxMemoryCond)
            and self.cond.full_rows == other.cond.full_rows
            and self.cond.target_rows == other.cond.target_rows
            and self.cond.equivalent_area == other.cond.equivalent_area
            and self.cond.visual_condition_rows
            == other.cond.visual_condition_rows
            and self.cond.audio_condition_rows == other.cond.audio_condition_rows
        )

    def concat(self, others):
        return self.cond

    def size(self):
        return self.cond


def _minimax_memory_shape(kwargs, latent_shapes, diffusion_model):
    """Return an H3 memory proxy whose row count matches PackedLayout."""
    if not isinstance(latent_shapes, (list, tuple)) or len(latent_shapes) < 2:
        return None
    video_shape = tuple(int(value) for value in latent_shapes[0])
    audio_shape = tuple(int(value) for value in latent_shapes[1])
    if len(video_shape) != 5 or len(audio_shape) != 4:
        return None
    if video_shape[0] != 1 or audio_shape[0] != 1 or audio_shape[2] != 2:
        return None

    patch_size = tuple(int(value) for value in diffusion_model.patch_size)
    # PackedLayout and H3's modality rows currently define this exact patch
    # contract. If upstream changes it, do not estimate with different rules.
    if patch_size != (1, 2, 2):
        return None
    pt, ph, pw = patch_size
    latent_t = math.ceil(video_shape[2] / pt)
    latent_h = math.ceil(video_shape[3] / ph) * ph
    latent_w = math.ceil(video_shape[4] / pw) * pw
    frame_rows = (latent_h // ph) * (latent_w // pw)
    target_visual_rows = latent_t * frame_rows
    target_audio_rows = int(audio_shape[2]) * int(audio_shape[3])
    target_rows = target_visual_rows + target_audio_rows
    if target_rows <= 0:
        return None

    cross_attn = kwargs.get("cross_attn")
    text_rows = (
        int(cross_attn.shape[1])
        if torch.is_tensor(cross_attn) and cross_attn.ndim >= 2
        else 0
    )

    keyframes = kwargs.get("minimax_keyframes") or ()
    visual_condition_rows = len(keyframes) * frame_rows
    audio_condition_rows = 0
    for ref in kwargs.get("minimax_refs") or ():
        kind = ref.get("kind")
        if kind == "image":
            visual_condition_rows += (
                int(ref["latent_h"]) // ph
            ) * (int(ref["latent_w"]) // pw)
        elif kind == "audio":
            audio_condition_rows += int(ref.get("ref_audio_t", 0)) * 2
        elif kind in ("video", "video_audio"):
            visual_condition_rows += (
                int(ref["latent_t"])
                * (int(ref["latent_h"]) // ph)
                * (int(ref["latent_w"]) // pw)
            )
            audio_condition_rows += int(ref.get("ref_audio_t", 0)) * 2

    condition_rows = text_rows + visual_condition_rows + audio_condition_rows
    full_rows = target_rows + condition_rows
    target_area = math.prod(video_shape[1:]) + math.prod(audio_shape[1:])
    equivalent_area = math.ceil(target_area * condition_rows / target_rows)
    return _MiniMaxMemoryShape(
        equivalent_area,
        full_rows=full_rows,
        target_rows=target_rows,
        target_visual_rows=target_visual_rows,
        target_audio_rows=target_audio_rows,
        visual_condition_rows=visual_condition_rows,
        audio_condition_rows=audio_condition_rows,
        hidden_size=int(diffusion_model.hidden_size),
        video_row_width=int(diffusion_model.latents_dim) * math.prod(patch_size),
        audio_row_width=int(diffusion_model.audio_latents_dim),
    )


def _make_extra_conds_shapes(base_model, diffusion_model):
    original = base_model.extra_conds_shapes

    def extra_conds_shapes(self, **kwargs):
        out = dict(original(**kwargs))
        context = getattr(self, _MEMORY_CONTEXT_ATTR, None)
        latent_shapes = kwargs.get("latent_shapes")
        if latent_shapes is None and isinstance(context, dict):
            latent_shapes = context.get("latent_shapes")
        shape = _minimax_memory_shape(kwargs, latent_shapes, diffusion_model)
        if shape is not None:
            out[_MEMORY_SHAPE_KEY] = shape
        return out

    return types.MethodType(extra_conds_shapes, base_model)


def _make_extra_conds(base_model, diffusion_model):
    original = base_model.extra_conds

    def extra_conds(self, **kwargs):
        out = original(**kwargs)
        latent_shapes = kwargs.get("latent_shapes")
        plan = _minimax_memory_shape(kwargs, latent_shapes, diffusion_model)
        if plan is not None:
            out[_MEMORY_SHAPE_KEY] = _MiniMaxMemoryCond(plan)
        return out

    return types.MethodType(extra_conds, base_model)


def _dtype_size(dtype) -> int:
    try:
        return torch.empty((), dtype=dtype).element_size()
    except (TypeError, RuntimeError):
        return 2


def _make_memory_required(base_model, w8_output_channels: tuple[int, ...]):
    original = base_model.memory_required

    def memory_required(self, input_shape, cond_shapes={}):
        required = original(input_shape, cond_shapes=cond_shapes)
        plans = [
            shape
            for shape in cond_shapes.get(_MEMORY_SHAPE_KEY, ())
            if isinstance(shape, _MiniMaxMemoryShape)
        ]
        if not plans:
            return required

        dtype_size = _dtype_size(self.get_dtype_inference())
        heuristic_extra = sum(
            original([1, 1, plan.equivalent_area], cond_shapes={})
            for plan in plans
        )
        explicit_extra = sum(
            plan.explicit_condition_bytes(dtype_size) for plan in plans
        )
        required += max(explicit_extra - heuristic_extra, 0)

        if w8_output_channels:
            rows = max(plan.full_rows for plan in plans)
            required += max(
                turing_int8_workspace_bytes(rows, output_channels)
                for output_channels in w8_output_channels
            )
        return required

    return types.MethodType(memory_required, base_model)


_make_outer_sample_wrapper = make_minimax_runtime_context_wrapper


def _w8_output_channels(root: torch.nn.Module) -> tuple[int, ...]:
    outputs = set()
    for module in root.modules():
        weight = getattr(module, "weight", None)
        if convrot_weight_kind(weight) == "w8a8" and getattr(weight, "ndim", 0) == 2:
            outputs.add(int(weight.shape[0]))
    return tuple(sorted(outputs))


def _install_memory_planning(model, base_model, diffusion_model) -> bool:
    if getattr(base_model, _MEMORY_ADAPTER_ATTR, False):
        return False
    if not all(
        callable(getattr(base_model, name, None))
        for name in (
            "extra_conds",
            "extra_conds_shapes",
            "memory_required",
            "get_dtype_inference",
        )
    ) or not hasattr(model, "add_wrapper_with_key"):
        return False

    base_model.extra_conds = _make_extra_conds(base_model, diffusion_model)
    base_model.extra_conds_shapes = _make_extra_conds_shapes(
        base_model, diffusion_model
    )
    factors = tuple(getattr(base_model, "memory_usage_factor_conds", ()))
    if _MEMORY_SHAPE_KEY not in factors:
        base_model.memory_usage_factor_conds = (*factors, _MEMORY_SHAPE_KEY)
    outputs = _w8_output_channels(base_model)
    base_model.memory_required = _make_memory_required(base_model, outputs)

    setattr(base_model, _MEMORY_ADAPTER_ATTR, True)
    LOG.info(
        "Enabled MiniMax packed-sequence VRAM planning: W8 outputs=[%s]",
        ",".join(map(str, outputs)) or "none",
    )
    return True


class _RuntimeDispatchAudit:
    """Report the first full MiniMax block pass without CUDA timing events."""

    def __init__(self, expected_blocks: int, expected_mlps: int):
        self.expected = {"block": expected_blocks, "mlp": expected_mlps}
        self.counts = {"block": Counter(), "mlp": Counter()}
        self.dtypes = {"block": Counter(), "mlp": Counter()}
        self.shapes = {"block": Counter(), "mlp": Counter()}
        self.reasons = {"block": Counter(), "mlp": Counter()}
        self.logged_phases: set[str] = set()

    def record(
        self,
        phase: str,
        fused: bool,
        x: torch.Tensor,
        reason: str | None = None,
    ) -> None:
        if phase in self.logged_phases or self.expected[phase] == 0:
            return
        self.counts[phase]["fused" if fused else "fallback"] += 1
        self.dtypes[phase][str(x.dtype)] += 1
        self.shapes[phase][str(tuple(x.shape))] += 1
        if reason is not None:
            self.reasons[phase][reason] += 1

        calls = self.counts[phase]["fused"] + self.counts[phase]["fallback"]
        if calls < self.expected[phase]:
            return

        log = LOG.warning if self.counts[phase]["fallback"] else LOG.info
        log(
            "MiniMax Turing runtime dispatch: phase=%s fused=%d fallback=%d "
            "dtypes=[%s] shapes=[%s] reasons=[%s]",
            phase,
            self.counts[phase]["fused"],
            self.counts[phase]["fallback"],
            _format_counts(self.dtypes[phase].elements()),
            _format_counts(self.shapes[phase].elements()),
            _format_counts(self.reasons[phase].elements()) or "none",
        )
        self.logged_phases.add(phase)


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


def _block_fusion_blocker(
    x: torch.Tensor,
    t_emb: torch.Tensor,
    device_index: int,
) -> str | None:
    if x.device.type != "cuda" or x.dtype not in _SUPPORTED_DTYPES or x.ndim != 2:
        return f"input={x.device.type}/{x.dtype}/ndim{x.ndim}"
    index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    if index != device_index:
        return f"device_index={index},expected={device_index}"
    if torch.is_grad_enabled() and (x.requires_grad or t_emb.requires_grad):
        return "grad_enabled"
    return None


def _make_mlp_forward(mlp: torch.nn.Module, audit: _RuntimeDispatchAudit):
    original = mlp.forward

    def forward(self, x: torch.Tensor):
        blocker = None
        if x.dtype != torch.bfloat16:
            blocker = f"dtype={x.dtype}"
        elif not is_turing_convrot_linear(self.fc2):
            blocker = "fc2_not_turing_convrot"
        audit.record("mlp", blocker is None, x, blocker)
        if blocker is not None:
            return original(x)
        return turing_linear_input_act(self.fc2, self.fc1(x), "swiglu")

    return types.MethodType(forward, mlp)


_minimax_temporal_topology = minimax_temporal_topology


def _make_block_forward(
    block: torch.nn.Module,
    device_index: int,
    mod_gate,
    audit: _RuntimeDispatchAudit,
    layer_index: int = 0,
    layer_count: int = 0,
    base_model=None,
    diffusion_model=None,
):
    original = block.forward

    def forward(
        self,
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options={},
    ):
        publish_minimax_attention_layout(
            transformer_options,
            mod_segments,
            layer_index=layer_index,
            layer_count=layer_count,
            base_model=base_model,
            diffusion_model=diffusion_model,
        )
        blocker = _block_fusion_blocker(x, t_emb, device_index)
        audit.record("block", blocker is None, x, blocker)
        if blocker is not None:
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

    return mark_forward_as_minimax_layout_provider(types.MethodType(forward, block))

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

    layout_status = ensure_minimax_attention_layout_provider(model)
    if layout_status.required and not layout_status.installed:
        LOG.warning(
            "MiniMax H3 attention layout provider could not be installed: %s",
            layout_status.reason,
        )

    diffusion_model = getattr(root, "diffusion_model", None)
    if diffusion_model is not None:
        _install_memory_planning(model, root, diffusion_model)

    eligible_fc2 = _audit_fc2([block for _, block in candidates])
    index = device.index if device.index is not None else torch.cuda.current_device()
    block_fusions = 0
    mlp_fusions = 0
    try:
        turing_segmented_rms_adaln = getattr(
            load_kernel_package(), "turing_segmented_rms_adaln"
        )
    except (ImportError, OSError, AttributeError):
        turing_segmented_rms_adaln = None

    expected_blocks = len(candidates) if callable(turing_segmented_rms_adaln) else 0
    audit = _RuntimeDispatchAudit(expected_blocks, eligible_fc2)

    for layer_index, (name, block) in enumerate(candidates):
        if hasattr(block.mlp, "fc2") and is_turing_convrot_linear(block.mlp.fc2):
            model.add_object_patch(
                f"{name}.mlp.forward",
                _make_mlp_forward(block.mlp, audit),
            )
            mlp_fusions += 1
        if callable(turing_segmented_rms_adaln):
            model.add_object_patch(
                f"{name}.forward",
                _make_block_forward(
                    block,
                    index,
                    _mod_gate,
                    audit,
                    layer_index,
                    len(candidates),
                    root,
                    diffusion_model,
                ),
            )
            block_fusions += 1

    if block_fusions:
        LOG.info("Enabled MiniMax segmented RMSNorm+AdaLN on %d Turing blocks", block_fusions)
    if mlp_fusions:
        LOG.info("Enabled MiniMax fused ConvRot SwiGLU on %d Turing MLP layers", mlp_fusions)
    if eligible_fc2 and mlp_fusions != eligible_fc2:
        raise RuntimeError("MiniMax Turing fc2 adapter did not patch every eligible layer")
    return max(block_fusions, mlp_fusions)
