"""MiniMax H3 memory planning and capability-based CUDA fusions."""

from __future__ import annotations

import dataclasses
import inspect
import logging
import math
import weakref
from collections import Counter
from collections.abc import Sequence

import torch

from ..methods import OriginalMethod, weak_method
from ..memory import install_memory_hooks, scan_quantized_workspaces
from ...attention.integration import AttentionSiteStatus, execute_projected_attention
from ...attention.protocol import (
    QKTransformSpec,
    RMSNormSpec,
    RotaryEmbeddingSpec,
    prepared_attention_executor,
)
from ...attention.stable import (
    fused_qk_preprocessing_available,
    precompute_turing_k_anchor,
    prequantize_turing_qk,
    reusable_k_anchor_available,
)
from ...attention.tuning import attention_kernel_tuning
from ...hardware import is_supported_attention_device
from ...kernel_api import load_kernel_package
from ...profiling import CUDA_PHASE_PROFILER
from .activation_policy import (
    balanced_saturation_size,
    decide_activation_chunks,
    decide_attention_heads,
    decide_ffn_channels,
    ensure_dynamic_vram_headroom,
    estimate_attention_lifecycle_peak,
)
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
    convrot_linear_input_act_from_weight,
    convrot_w8_output_slice,
    convrot_w8_plain_tensors,
    convrot_weight_kind,
    fused_convrot_linear_input_act,
    is_turing_convrot_linear,
    segmented_rms_adaln,
)
from ...quantization.dispatch import (
    int8_linear_from_quantized,
    quantize_convrot_int8_activation,
    quantize_convrot_swiglu_activation,
    quantize_convrot_swiglu_with_scale,
    convrot_swiglu_channel_sharding_available,
    turing_int8_workspace_bytes,
)


LOG = logging.getLogger("comfyui-turing-utils")
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BLOCK_FORWARD_PARAMETERS = (
    "x",
    "t_emb",
    "mod_segments",
    "rope_freqs",
    "transformer_options",
)
_ATTENTION_FORWARD_PARAMETERS = (
    "x",
    "rope_freqs",
    "transformer_options",
)
_PREPARED_ATTENTION_FORWARD_ATTR = "_turing_utils_minimax_prepared_attention"
_MEMORY_SHAPE_KEY = "turing_utils_minimax_packed_sequence"
_MEMORY_CONTEXT_ATTR = RUNTIME_CONTEXT_ATTR
_MEMORY_ADAPTER_ATTR = "_turing_utils_minimax_memory_adapter"
_OUTER_SAMPLE_WRAPPER_KEY = RUNTIME_OUTER_WRAPPER_KEY
_ATTENTION_LAYOUT_KEY = ATTENTION_LAYOUT_KEY
_STREAMED_QKV_EXECUTOR_ATTR = "turing_utils_streamed_qkv_executor"


def _runtime_activation_plan(base_model):
    if base_model is None:
        return None
    try:
        context = getattr(base_model, RUNTIME_CONTEXT_ATTR, None)
    except ReferenceError:
        return None
    return context.get("activation_plan") if isinstance(context, dict) else None


def _weak_model_reference(base_model):
    if base_model is None:
        return None
    try:
        return weakref.proxy(base_model)
    except TypeError:
        # Test doubles and a few third-party wrappers are not weak-referenceable.
        return base_model


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


@dataclasses.dataclass(frozen=True, slots=True)
class _MiniMaxActivationProfile:
    hidden_size: int
    heads: int
    head_dim: int
    expanded_size: int

    def conservative_floor_bytes(self, rows: int, element_size: int) -> int:
        """Peak of the first safe streamed tiers, not a sum of serial ops."""
        rows = int(rows)
        element_size = int(element_size)
        block_heads = 256 // math.gcd(256, self.head_dim)
        group = balanced_saturation_size(
            self.heads,
            alignment=block_heads,
            minimum=math.ceil(self.heads / 4),
        )
        attention = estimate_attention_lifecycle_peak(
            rows=rows,
            heads=self.heads,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
            element_size=element_size,
            head_group=group,
            compact_qk=True,
            cache_quantized_input=False,
            quantized_value=True,
        )

        qkv_scales = ((rows + 63) // 64) * self.heads * 5 * 4
        qkv_persistent = (
            rows
            * (
                2 * self.heads * self.head_dim
                + self.heads * self.head_dim * element_size
            )
            + qkv_scales
        )
        qkv_tile = min(rows, 16_384) * (
            3 * self.heads * self.head_dim * element_size
            + self.hidden_size
            + 4
        )
        qkv = qkv_persistent + qkv_tile

        mlp = rows * self.hidden_size * element_size + min(rows, 16_384) * (
            2 * self.expanded_size * element_size
            + self.expanded_size
            + self.hidden_size * element_size
        )
        return max(attention, qkv, mlp)


def _activation_profile(diffusion_model) -> _MiniMaxActivationProfile | None:
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        return None
    block = blocks[0]
    attention = getattr(block, "attn", None)
    mlp = getattr(block, "mlp", None)
    fc2 = getattr(mlp, "fc2", None)
    try:
        profile = _MiniMaxActivationProfile(
            hidden_size=int(getattr(diffusion_model, "hidden_size")),
            heads=int(getattr(attention, "heads")),
            head_dim=int(getattr(attention, "head_dim")),
            expanded_size=int(getattr(fc2, "in_features")),
        )
    except (AttributeError, TypeError, ValueError):
        return None
    if min(dataclasses.astuple(profile)) <= 0:
        return None
    return profile


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
    original = OriginalMethod.capture(base_model.extra_conds_shapes, base_model)

    def extra_conds_shapes(self, **kwargs):
        out = dict(original(self, **kwargs))
        context = getattr(self, _MEMORY_CONTEXT_ATTR, None)
        latent_shapes = kwargs.get("latent_shapes")
        if latent_shapes is None and isinstance(context, dict):
            latent_shapes = context.get("latent_shapes")
        shape = _minimax_memory_shape(kwargs, latent_shapes, diffusion_model)
        if shape is not None:
            out[_MEMORY_SHAPE_KEY] = shape
        return out

    return weak_method(extra_conds_shapes, base_model)


def _make_extra_conds(base_model, diffusion_model):
    original = OriginalMethod.capture(base_model.extra_conds, base_model)

    def extra_conds(self, **kwargs):
        out = original(self, **kwargs)
        latent_shapes = kwargs.get("latent_shapes")
        plan = _minimax_memory_shape(kwargs, latent_shapes, diffusion_model)
        if plan is not None:
            out[_MEMORY_SHAPE_KEY] = _MiniMaxMemoryCond(plan)
        return out

    return weak_method(extra_conds, base_model)


def _dtype_size(dtype) -> int:
    try:
        return torch.empty((), dtype=dtype).element_size()
    except (TypeError, RuntimeError):
        return 2


def _make_memory_required(
    base_model,
    w8_output_channels: tuple[int, ...],
    fixed_workspaces: tuple[int, ...],
    activation_profile: _MiniMaxActivationProfile | None = None,
):
    original = OriginalMethod.capture(base_model.memory_required, base_model)

    def memory_required(self, input_shape, cond_shapes={}):
        required = original(self, input_shape, cond_shapes=cond_shapes)
        plans = [
            shape
            for shape in cond_shapes.get(_MEMORY_SHAPE_KEY, ())
            if isinstance(shape, _MiniMaxMemoryShape)
        ]
        if not plans:
            return required

        dtype_size = _dtype_size(self.get_dtype_inference())
        heuristic_extra = sum(
            original(self, [1, 1, plan.equivalent_area], cond_shapes={})
            for plan in plans
        )
        explicit_extra = sum(
            plan.explicit_condition_bytes(dtype_size) for plan in plans
        )
        required += max(explicit_extra - heuristic_extra, 0)

        rows = max(plan.full_rows for plan in plans)
        transient_workspaces = [
            turing_int8_workspace_bytes(rows, output_channels)
            for output_channels in w8_output_channels
        ]
        transient_workspaces.extend(fixed_workspaces)
        if activation_profile is not None:
            transient_workspaces.append(
                activation_profile.conservative_floor_bytes(rows, dtype_size)
            )
        if transient_workspaces:
            # Linear layers execute serially, so only the largest transient
            # workspace is live at once. Summing them would over-reserve VRAM.
            required += max(transient_workspaces)
        return required

    return weak_method(memory_required, base_model)


_make_outer_sample_wrapper = make_minimax_runtime_context_wrapper


def _linear_workspace_requirements(
    root: torch.nn.Module,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    profile = scan_quantized_workspaces(root, convrot_weight_kind)
    return profile.w8_output_channels, profile.fixed_workspaces


def _install_memory_planning(model, base_model, diffusion_model) -> bool:
    outputs, fixed_workspaces = _linear_workspace_requirements(base_model)
    activation_profile = _activation_profile(diffusion_model)
    installed = install_memory_hooks(
        base_model,
        marker=_MEMORY_ADAPTER_ATTR,
        condition_key=_MEMORY_SHAPE_KEY,
        extra_conds=_make_extra_conds(base_model, diffusion_model),
        extra_conds_shapes=_make_extra_conds_shapes(base_model, diffusion_model),
        memory_required=_make_memory_required(
            base_model,
            outputs,
            fixed_workspaces,
            activation_profile,
        ),
        required_methods=("get_dtype_inference",),
    )
    if not installed:
        return False
    LOG.info(
        "Enabled MiniMax packed-sequence VRAM planning: W8 outputs=[%s] "
        "fixed_workspaces=[%s MiB] activation_profile=%s",
        ",".join(map(str, outputs)) or "none",
        ",".join(f"{value / 1024**2:.1f}" for value in fixed_workspaces) or "none",
        (
            f"H{activation_profile.hidden_size}/A{activation_profile.heads}x"
            f"{activation_profile.head_dim}/F{activation_profile.expanded_size}"
            if activation_profile is not None
            else "none"
        ),
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
            "MiniMax fused runtime dispatch: phase=%s fused=%d fallback=%d "
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
        "MiniMax fused fc2 dispatch: blocks=%d eligible=%d formats=[%s]",
        len(linears),
        eligible,
        _format_counts(kinds),
    )
    return eligible


def _compatible_block_forward(block_type: type[torch.nn.Module]) -> bool:
    parameters = tuple(inspect.signature(block_type.forward).parameters)
    return parameters == ("self", *_BLOCK_FORWARD_PARAMETERS)


def _compatible_attention_forward(attention_type: type[torch.nn.Module]) -> bool:
    parameters = tuple(inspect.signature(attention_type.forward).parameters)
    return parameters == ("self", *_ATTENTION_FORWARD_PARAMETERS)


def _qk_transform(attention, x, rope_freqs) -> QKTransformSpec:
    import comfy.model_management

    query_norm = comfy.model_management.cast_to(
        attention.q_norm.weight, device=x.device, dtype=x.dtype
    )
    key_norm = comfy.model_management.cast_to(
        attention.k_norm.weight, device=x.device, dtype=x.dtype
    )
    rot_dim = int(rope_freqs.shape[-3] * 2) if rope_freqs is not None else 0
    return QKTransformSpec(
        query_norm=RMSNormSpec(
            query_norm, float(attention.q_norm.eps), "head"
        ),
        key_norm=RMSNormSpec(
            key_norm, float(attention.k_norm.eps), "head"
        ),
        rotary=RotaryEmbeddingSpec(
            rope_freqs,
            rot_dim,
            "split_half" if rope_freqs is not None else "none",
        ),
    )


def _linear_with_cast_weight(linear, x, weight, bias):
    pre_quant_scale = getattr(linear, "pre_quant_scale", None)
    if pre_quant_scale is not None:
        import comfy.model_management

        x = x * comfy.model_management.cast_to_device(
            pre_quant_scale, x.device, x.dtype
        )
    function = getattr(linear, "_forward", None)
    return (
        function(x, weight, bias)
        if callable(function)
        else torch.nn.functional.linear(x, weight, bias)
    )


def _slice_qk_transform(
    transform: QKTransformSpec,
    start: int,
    stop: int,
    full_rows: int,
) -> QKTransformSpec:
    freqs = transform.freqs
    if torch.is_tensor(freqs) and freqs.ndim >= 2 and freqs.shape[1] == full_rows:
        rotary = dataclasses.replace(transform.rotary, freqs=freqs[:, start:stop])
        return dataclasses.replace(transform, rotary=rotary)
    return transform


def _sample_qk_transform(
    transform: QKTransformSpec,
    indices: torch.Tensor,
    full_rows: int,
) -> QKTransformSpec:
    freqs = transform.freqs
    if torch.is_tensor(freqs) and freqs.ndim >= 2 and freqs.shape[1] == full_rows:
        rotary = dataclasses.replace(
            transform.rotary,
            freqs=freqs.index_select(1, indices),
        )
        return dataclasses.replace(transform, rotary=rotary)
    return transform


def _global_k_anchor(
    attention,
    x: torch.Tensor,
    weight,
    bias,
    transform: QKTransformSpec,
    inner: int,
    heads: int,
    head_dim: int,
):
    sequence = int(x.shape[0])
    sample_rows = [index * (sequence - 1) // 8 for index in range(9)]
    sample_indices = torch.tensor(
        sample_rows, dtype=torch.long, device=x.device
    )
    sampled = x.index_select(0, sample_indices)
    projected = _linear_with_cast_weight(
        attention.qkv_proj, sampled, weight, bias
    )
    key = projected[:, inner : 2 * inner]
    key = key.view(9, heads, head_dim).transpose(0, 1).unsqueeze(0)
    sample_transform = _sample_qk_transform(
        transform, sample_indices, sequence
    )
    anchor_indices, anchor_values = precompute_turing_k_anchor(
        key, sample_transform
    )
    lookup = sample_indices.to(dtype=torch.int32)
    selected = anchor_indices.clamp_min(0).to(dtype=torch.long)
    anchor_indices = torch.where(
        anchor_indices >= 0,
        lookup[selected],
        anchor_indices,
    )
    return anchor_indices, anchor_values


def _qkv_input_tile(linear, x: torch.Tensor) -> torch.Tensor:
    pre_quant_scale = getattr(linear, "pre_quant_scale", None)
    if pre_quant_scale is None:
        return x
    import comfy.model_management

    scale = comfy.model_management.cast_to_device(
        pre_quant_scale, x.device, x.dtype
    )
    return x * scale


def _quantize_qkv_rows(linear, x: torch.Tensor):
    return quantize_convrot_int8_activation(
        _qkv_input_tile(linear, x), 256
    )


def _cache_quantized_qkv_input(linear, x: torch.Tensor, chunk_rows: int):
    qactivation = torch.empty_like(x, dtype=torch.int8)
    activation_scale = torch.empty(
        (x.shape[0],), dtype=torch.float32, device=x.device
    )
    for start in range(0, x.shape[0], chunk_rows):
        stop = min(start + chunk_rows, x.shape[0])
        tile, scale = _quantize_qkv_rows(linear, x[start:stop])
        qactivation[start:stop].copy_(tile)
        activation_scale[start:stop].copy_(scale.reshape(-1))
        del tile, scale
    return qactivation, activation_scale


def _w8_qkv_component(
    qactivation: torch.Tensor,
    activation_scale: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    component: int,
    head_start: int,
    head_stop: int,
    heads: int,
    head_dim: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    inner = heads * head_dim
    start = component * inner + head_start * head_dim
    stop = component * inner + head_stop * head_dim
    return convrot_w8_output_slice(
        qactivation,
        activation_scale,
        qweight,
        weight_scale,
        bias,
        start,
        stop,
        output_dtype,
    )


def _head_group_k_anchor(
    attention,
    x: torch.Tensor,
    transform: QKTransformSpec,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    head_start: int,
    head_stop: int,
    quantized_input,
):
    sequence = int(x.shape[0])
    sample_rows = [index * (sequence - 1) // 8 for index in range(9)]
    sample_indices = torch.tensor(
        sample_rows, dtype=torch.long, device=x.device
    )
    if quantized_input is None:
        qactivation, activation_scale = _quantize_qkv_rows(
            attention.qkv_proj, x.index_select(0, sample_indices)
        )
    else:
        cached_activation, cached_scale = quantized_input
        qactivation = cached_activation.index_select(0, sample_indices)
        activation_scale = cached_scale.index_select(0, sample_indices)
    key = _w8_qkv_component(
        qactivation,
        activation_scale,
        qweight,
        weight_scale,
        bias,
        component=1,
        head_start=head_start,
        head_stop=head_stop,
        heads=int(attention.heads),
        head_dim=int(attention.head_dim),
        output_dtype=x.dtype,
    )
    group = head_stop - head_start
    key = key.view(9, group, attention.head_dim).transpose(0, 1).unsqueeze(0)
    sample_transform = _sample_qk_transform(
        transform, sample_indices, sequence
    )
    anchor_indices, anchor_values = precompute_turing_k_anchor(
        key, sample_transform
    )
    lookup = sample_indices.to(dtype=torch.int32)
    selected = anchor_indices.clamp_min(0).to(dtype=torch.long)
    anchor_indices = torch.where(
        anchor_indices >= 0,
        lookup[selected],
        anchor_indices,
    )
    return anchor_indices, anchor_values


def _stream_qkv_head_group(
    attention,
    x: torch.Tensor,
    transform: QKTransformSpec,
    transformer_options: dict,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    head_start: int,
    head_stop: int,
    chunk_rows: int,
    quantized_input,
):
    """Project one full-sequence head group into compact Q/K/V storage."""
    sequence = int(x.shape[0])
    group = head_stop - head_start
    head_dim = int(attention.head_dim)
    q_int8 = torch.empty(
        (1, group, sequence, head_dim), dtype=torch.int8, device=x.device
    )
    k_int8 = torch.empty_like(q_int8)
    q_scale = torch.empty(
        (1, group, ((sequence + 63) // 64) * 4),
        dtype=torch.float32,
        device=x.device,
    )
    k_scale = torch.empty(
        (1, group, (sequence + 63) // 64),
        dtype=torch.float32,
        device=x.device,
    )
    value = torch.empty(
        (1, group, sequence, head_dim), dtype=x.dtype, device=x.device
    )
    tuning = attention_kernel_tuning(transformer_options)
    k_anchor = (
        _head_group_k_anchor(
            attention,
            x,
            transform,
            qweight,
            weight_scale,
            bias,
            head_start,
            head_stop,
            quantized_input,
        )
        if tuning.rotate_qk and tuning.stabilize_k
        else None
    )
    qk_type = None
    route_original_basis = False
    for start in range(0, sequence, chunk_rows):
        stop = min(start + chunk_rows, sequence)
        if quantized_input is None:
            qactivation, activation_scale = _quantize_qkv_rows(
                attention.qkv_proj, x[start:stop]
            )
        else:
            cached_activation, cached_scale = quantized_input
            qactivation = cached_activation[start:stop]
            activation_scale = cached_scale[start:stop]
        query = _w8_qkv_component(
            qactivation,
            activation_scale,
            qweight,
            weight_scale,
            bias,
            component=0,
            head_start=head_start,
            head_stop=head_stop,
            heads=int(attention.heads),
            head_dim=head_dim,
            output_dtype=x.dtype,
        )
        key = _w8_qkv_component(
            qactivation,
            activation_scale,
            qweight,
            weight_scale,
            bias,
            component=1,
            head_start=head_start,
            head_stop=head_stop,
            heads=int(attention.heads),
            head_dim=head_dim,
            output_dtype=x.dtype,
        )
        rows = stop - start
        query = query.view(rows, group, head_dim).transpose(0, 1).unsqueeze(0)
        key = key.view(rows, group, head_dim).transpose(0, 1).unsqueeze(0)
        tile_transform = _slice_qk_transform(
            transform, start, stop, sequence
        )
        q_scale_start = (start // 64) * 4
        k_scale_start = start // 64
        tile_q_scale = q_scale[
            :,
            :,
            q_scale_start : q_scale_start + ((rows + 63) // 64) * 4,
        ]
        tile_k_scale = k_scale[
            :,
            :,
            k_scale_start : k_scale_start + (rows + 63) // 64,
        ]
        tile_qk = prequantize_turing_qk(
            query,
            key,
            tile_transform,
            kernel="sol",
            transformer_options=transformer_options,
            k_anchor=k_anchor,
            qk_output=(
                q_int8[:, :, start:stop],
                tile_q_scale,
                k_int8[:, :, start:stop],
                tile_k_scale,
            ),
        )
        value_tile = _w8_qkv_component(
            qactivation,
            activation_scale,
            qweight,
            weight_scale,
            bias,
            component=2,
            head_start=head_start,
            head_stop=head_stop,
            heads=int(attention.heads),
            head_dim=head_dim,
            output_dtype=x.dtype,
        )
        value[:, :, start:stop].copy_(
            value_tile.view(rows, group, head_dim).transpose(0, 1).unsqueeze(0)
        )
        qk_type = type(tile_qk)
        route_original_basis = bool(tile_qk.route_original_basis)
        del query, key, value_tile, tile_qk
        if quantized_input is None:
            del qactivation, activation_scale

    if qk_type is None:
        raise RuntimeError("head-sharded H3 QKV projection produced no tiles")
    qk = qk_type(
        query_int8=q_int8,
        query_scale=q_scale,
        key_int8=k_int8,
        key_scale=k_scale,
        tensor_layout="HND",
        input_dtype=x.dtype,
        original_head_dim=head_dim,
        route_original_basis=route_original_basis,
    )
    return qk, value


def _project_qkv_head_group(
    attention,
    x: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    head_start: int,
    head_stop: int,
    chunk_rows: int,
    quantized_input,
):
    sequence = int(x.shape[0])
    group = head_stop - head_start
    head_dim = int(attention.head_dim)
    outputs = [
        torch.empty(
            (sequence, group, head_dim), dtype=x.dtype, device=x.device
        )
        for _ in range(3)
    ]
    for start in range(0, sequence, chunk_rows):
        stop = min(start + chunk_rows, sequence)
        if quantized_input is None:
            qactivation, activation_scale = _quantize_qkv_rows(
                attention.qkv_proj, x[start:stop]
            )
        else:
            cached_activation, cached_scale = quantized_input
            qactivation = cached_activation[start:stop]
            activation_scale = cached_scale[start:stop]
        rows = stop - start
        for component, destination in enumerate(outputs):
            tile = _w8_qkv_component(
                qactivation,
                activation_scale,
                qweight,
                weight_scale,
                bias,
                component=component,
                head_start=head_start,
                head_stop=head_stop,
                heads=int(attention.heads),
                head_dim=head_dim,
                output_dtype=x.dtype,
            )
            destination[start:stop].copy_(tile.view(rows, group, head_dim))
            del tile
        if quantized_input is None:
            del qactivation, activation_scale
    return tuple(outputs)


def _apply_minimax_qk_transform(attention, query, key, rope_freqs):
    import comfy.model_management
    import comfy.quant_ops

    sequence, group, head_dim = query.shape
    if rope_freqs is not None:
        query = query.view(1, sequence, group, head_dim)
        key = key.view(1, sequence, group, head_dim)
        query_weight = comfy.model_management.cast_to(
            attention.q_norm.weight, device=query.device
        )
        key_weight = comfy.model_management.cast_to(
            attention.k_norm.weight, device=key.device
        )
        rot_dim = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            query, key = comfy.quant_ops.ck.rms_rope_split_half(
                query,
                key,
                rope_freqs,
                query_weight,
                key_weight,
                epsilon=attention.q_norm.eps,
                rot_dim=rot_dim,
            )
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                query,
                key,
                rope_freqs,
                query_weight,
                key_weight,
                epsilon=attention.q_norm.eps,
                rot_dim=rot_dim,
            )
        return query[0], key[0]
    return attention.q_norm(query), attention.k_norm(key)


def _head_sharded_attention(
    attention,
    x: torch.Tensor,
    rope_freqs,
    transform: QKTransformSpec,
    transformer_options: dict,
    attention_container,
    executor,
    head_group: int,
    cache_quantized_input: bool,
):
    import comfy.ops
    from comfy.ldm.modules.attention import optimized_attention

    comfy.ops.run_every_op()
    weight, bias, offload_stream = comfy.ops.cast_bias_weight(
        attention.qkv_proj,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=True,
    )
    try:
        plain = convrot_w8_plain_tensors(weight)
        if plain is None:
            return None
        qweight, weight_scale = plain
        quantized_input = (
            _cache_quantized_qkv_input(attention.qkv_proj, x, 16_384)
            if cache_quantized_input
            else None
        )
        sequence = int(x.shape[0])
        heads = int(attention.heads)
        head_dim = int(attention.head_dim)
        output = torch.empty(
            (sequence, heads * head_dim), dtype=x.dtype, device=x.device
        )
        streamed_executor = (
            getattr(executor, _STREAMED_QKV_EXECUTOR_ATTR, None)
            if executor is not None
            else None
        )
        compact = callable(streamed_executor) and reusable_k_anchor_available()
        for head_start in range(0, heads, head_group):
            head_stop = min(head_start + head_group, heads)
            group = head_stop - head_start
            if compact:
                qk, value = _stream_qkv_head_group(
                    attention,
                    x,
                    transform,
                    transformer_options,
                    qweight,
                    weight_scale,
                    bias,
                    head_start,
                    head_stop,
                    16_384,
                    quantized_input,
                )
                outcome = streamed_executor(
                    qk,
                    value,
                    heads=group,
                    qk_transform=transform,
                    transformer_options=transformer_options,
                )
                del qk, value
                if not outcome.supported:
                    return None
                group_output = outcome.output.squeeze(0)
            else:
                query, key, value = _project_qkv_head_group(
                    attention,
                    x,
                    qweight,
                    weight_scale,
                    bias,
                    head_start,
                    head_stop,
                    16_384,
                    quantized_input,
                )
                query, key = _apply_minimax_qk_transform(
                    attention, query, key, rope_freqs
                )
                group_output = optimized_attention(
                    attention_container(query.transpose(0, 1).unsqueeze(0)),
                    attention_container(key.transpose(0, 1).unsqueeze(0)),
                    attention_container(value.transpose(0, 1).unsqueeze(0)),
                    group,
                    mask=None,
                    skip_reshape=True,
                    transformer_options=transformer_options,
                ).squeeze(0)
                del query, key, value
            output[
                :, head_start * head_dim : head_stop * head_dim
            ].copy_(group_output)
            del group_output
        return output
    finally:
        comfy.ops.uncast_bias_weight(
            attention.qkv_proj, weight, bias, offload_stream
        )


def _stream_qkv_projection(
    attention,
    x: torch.Tensor,
    transform: QKTransformSpec,
    transformer_options: dict,
    chunk_rows: int,
):
    """Project H3 QKV by rows while retaining only Q/K INT8 and V BF16."""
    import comfy.ops

    sequence = int(x.shape[0])
    heads = int(attention.heads)
    head_dim = int(attention.head_dim)
    inner = heads * head_dim
    q_int8 = torch.empty(
        (1, heads, sequence, head_dim), dtype=torch.int8, device=x.device
    )
    k_int8 = torch.empty_like(q_int8)
    q_scale = torch.empty(
        (1, heads, ((sequence + 63) // 64) * 4),
        dtype=torch.float32,
        device=x.device,
    )
    k_scale = torch.empty(
        (1, heads, (sequence + 63) // 64),
        dtype=torch.float32,
        device=x.device,
    )
    value = torch.empty(
        (1, heads, sequence, head_dim), dtype=x.dtype, device=x.device
    )

    comfy.ops.run_every_op()
    original_w8 = convrot_weight_kind(attention.qkv_proj.weight) == "w8a8"
    weight, bias, offload_stream = comfy.ops.cast_bias_weight(
        attention.qkv_proj,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=original_w8,
    )
    qk_type = None
    route_original_basis = False
    try:
        plain = convrot_w8_plain_tensors(weight) if original_w8 else None
        tuning = attention_kernel_tuning(transformer_options)
        if tuning.rotate_qk and tuning.stabilize_k:
            if plain is not None:
                qweight, weight_scale = plain
                k_anchor = _head_group_k_anchor(
                    attention,
                    x,
                    transform,
                    qweight,
                    weight_scale,
                    bias,
                    0,
                    heads,
                    None,
                )
            else:
                k_anchor = _global_k_anchor(
                    attention,
                    x,
                    weight,
                    bias,
                    transform,
                    inner,
                    heads,
                    head_dim,
                )
        else:
            k_anchor = None
        for start in range(0, sequence, chunk_rows):
            stop = min(start + chunk_rows, sequence)
            if plain is None:
                projected = _linear_with_cast_weight(
                    attention.qkv_proj, x[start:stop], weight, bias
                )
            else:
                qactivation, activation_scale = _quantize_qkv_rows(
                    attention.qkv_proj, x[start:stop]
                )
                projected = convrot_w8_output_slice(
                    qactivation,
                    activation_scale,
                    qweight,
                    weight_scale,
                    bias,
                    0,
                    3 * inner,
                    x.dtype,
                )
                del qactivation, activation_scale
            query, key, value_tile = projected.split(inner, dim=-1)
            tile_rows = stop - start
            query = query.view(tile_rows, heads, head_dim).transpose(0, 1).unsqueeze(0)
            key = key.view(tile_rows, heads, head_dim).transpose(0, 1).unsqueeze(0)
            value_tile = (
                value_tile.view(tile_rows, heads, head_dim)
                .transpose(0, 1)
                .unsqueeze(0)
            )
            tile_transform = _slice_qk_transform(
                transform, start, stop, sequence
            )
            q_scale_start = (start // 64) * 4
            k_scale_start = start // 64
            tile_q_scale = q_scale[
                :,
                :,
                q_scale_start : q_scale_start + ((tile_rows + 63) // 64) * 4,
            ]
            tile_k_scale = k_scale[
                :,
                :,
                k_scale_start : k_scale_start + (tile_rows + 63) // 64,
            ]
            tile_qk = prequantize_turing_qk(
                query,
                key,
                tile_transform,
                kernel="sol",
                transformer_options=transformer_options,
                k_anchor=k_anchor,
                qk_output=(
                    q_int8[:, :, start:stop],
                    tile_q_scale,
                    k_int8[:, :, start:stop],
                    tile_k_scale,
                ),
            )
            value[:, :, start:stop].copy_(value_tile)
            qk_type = type(tile_qk)
            route_original_basis = bool(tile_qk.route_original_basis)
            del projected, query, key, value_tile, tile_qk
    finally:
        comfy.ops.uncast_bias_weight(
            attention.qkv_proj, weight, bias, offload_stream
        )

    if qk_type is None:
        raise RuntimeError("streamed H3 QKV projection produced no tiles")
    qk = qk_type(
        query_int8=q_int8,
        query_scale=q_scale,
        key_int8=k_int8,
        key_scale=k_scale,
        tensor_layout="HND",
        input_dtype=x.dtype,
        original_head_dim=head_dim,
        route_original_basis=route_original_basis,
    )
    return qk, value


def _make_attention_forward(
    attention,
    attention_container,
    original=None,
    base_model=None,
):
    original = OriginalMethod.capture(
        attention.forward if original is None else original,
        attention,
    )
    base_model = _weak_model_reference(base_model)

    def forward(self, x, rope_freqs=None, transformer_options={}):
        executor = prepared_attention_executor(transformer_options)
        if (
            x.ndim != 2
            or x.shape[0] < 64
            or x.dtype not in (torch.float16, torch.bfloat16)
            or self.head_dim not in (64, 128)
            or (
                torch.is_grad_enabled()
                and (
                    x.requires_grad
                    or any(parameter.requires_grad for parameter in self.parameters())
                )
            )
        ):
            return original(
                self,
                x,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options,
            )

        profiling = CUDA_PHASE_PROFILER.enabled
        transform = _qk_transform(self, x, rope_freqs)
        streamed_executor = (
            getattr(executor, _STREAMED_QKV_EXECUTOR_ATTR, None)
            if executor is not None
            else None
        )
        # Row streaming relies on the v0.30 direct-output ABI even when
        # K-anchor stabilization is disabled. Older kernels keep the complete
        # projection path instead of failing after committing partial output.
        stream_abi_available = reusable_k_anchor_available()
        qkv_is_w8 = convrot_weight_kind(self.qkv_proj.weight) == "w8a8"
        runtime_plan = _runtime_activation_plan(base_model)
        head_decision = None
        if qkv_is_w8:
            head_decision = decide_attention_heads(
                x,
                heads=int(self.heads),
                head_dim=int(self.head_dim),
                compact_qk=bool(
                    callable(streamed_executor) and stream_abi_available
                ),
                quantized_input=True,
                quantized_value=bool(callable(streamed_executor)),
                runtime_plan=runtime_plan,
                base_model=base_model,
            )
            if head_decision.sharded:
                if base_model is not None:
                    ensure_dynamic_vram_headroom(
                        base_model,
                        x.device,
                        rows=int(x.shape[0]),
                        operation="attention_heads",
                        estimated_peak_bytes=head_decision.estimated_peak_bytes,
                        runtime_plan=runtime_plan,
                    )
                output = _head_sharded_attention(
                    self,
                    x,
                    rope_freqs,
                    transform,
                    transformer_options,
                    attention_container,
                    executor,
                    head_decision.head_group,
                    head_decision.cache_quantized_input,
                )
                if output is not None:
                    if profiling:
                        output = CUDA_PHASE_PROFILER.call(
                            "minimax.out_projection", self.out_proj, output
                        )
                        CUDA_PHASE_PROFILER.complete_attention(
                            (1, self.heads, x.shape[0], self.head_dim)
                        )
                        return output
                    return self.out_proj(output)

            if base_model is not None:
                # Automatic tiers already fit immediately usable memory, so
                # this is normally a no-op. It remains useful for an explicit
                # throughput override, and may evict only inactive-model
                # VBARs; the current diffusion weights are never targeted.
                ensure_dynamic_vram_headroom(
                    base_model,
                    x.device,
                    rows=int(x.shape[0]),
                    operation="attention_execute",
                    estimated_peak_bytes=head_decision.estimated_peak_bytes,
                    runtime_plan=runtime_plan,
                )

        if executor is None:
            return original(
                self,
                x,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options,
            )
        if callable(streamed_executor) and stream_abi_available:
            decision = decide_activation_chunks(
                x,
                operation="qkv",
                hidden_size=int(x.shape[-1]),
                expanded_size=int(self.heads * self.head_dim),
                heads=int(self.heads),
                runtime_plan=runtime_plan,
                base_model=base_model,
            )
            if decision.streamed:
                if base_model is not None:
                    ensure_dynamic_vram_headroom(
                        base_model,
                        x.device,
                        rows=int(x.shape[0]),
                        operation="qkv",
                        estimated_peak_bytes=decision.streamed_peak_bytes,
                        runtime_plan=runtime_plan,
                    )
                qk, value = _stream_qkv_projection(
                    self,
                    x,
                    transform,
                    transformer_options,
                    decision.chunk_rows,
                )
                outcome = streamed_executor(
                    qk,
                    value,
                    heads=self.heads,
                    qk_transform=transform,
                    transformer_options=transformer_options,
                )
                del qk, value
                if not outcome.supported:
                    raise RuntimeError(
                        "streamed H3 QKV executor rejected a committed projection: "
                        f"{outcome.reason}"
                    )
                output = outcome.output.squeeze(0)
                if profiling:
                    output = CUDA_PHASE_PROFILER.call(
                        "minimax.out_projection", self.out_proj, output
                    )
                    CUDA_PHASE_PROFILER.complete_attention(
                        (1, self.heads, x.shape[0], self.head_dim)
                    )
                    return output
                return self.out_proj(output)

        if profiling:
            qkv = CUDA_PHASE_PROFILER.call(
                "minimax.qkv_projection", self.qkv_proj, x
            )
        else:
            qkv = self.qkv_proj(x)
        sequence = x.shape[0]
        inner = self.heads * self.head_dim
        query, key, value = qkv.split(inner, dim=-1)
        query = query.view(sequence, self.heads, self.head_dim)
        key = key.view(sequence, self.heads, self.head_dim)
        value = value.view(sequence, self.heads, self.head_dim)
        del qkv
        outcome = execute_projected_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            heads=self.heads,
            qk_transform=transform,
            transformer_options=transformer_options,
            container_factory=attention_container,
        )
        if not outcome.supported:
            del query, key, value
            return original(
                self,
                x,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options,
            )
        output = outcome.output.squeeze(0)
        if profiling:
            output = CUDA_PHASE_PROFILER.call(
                "minimax.out_projection", self.out_proj, output
            )
            CUDA_PHASE_PROFILER.complete_attention(
                (1, self.heads, sequence, self.head_dim)
            )
            return output
        return self.out_proj(output)

    setattr(forward, _PREPARED_ATTENTION_FORWARD_ATTR, True)
    return weak_method(forward, attention)


def _has_prepared_attention_forward(forward) -> bool:
    function = getattr(forward, "__func__", forward)
    return bool(getattr(function, _PREPARED_ATTENTION_FORWARD_ATTR, False))


def install_minimax_attention_sites(model, device: torch.device) -> AttentionSiteStatus:
    """Install only the model-side H3 handoff to generic attention backends."""
    try:
        from comfy.ldm.minimax.model import Attention, DiTBlock
        from comfy.ldm.modules.attention import AttentionTensorContainer
    except ImportError:
        return AttentionSiteStatus(None, 0, "minimax_unavailable")
    root = getattr(model, "model", model)
    if not callable(getattr(root, "named_modules", None)):
        return AttentionSiteStatus(None, 0, "not_minimax_h3")
    candidates = [
        (name, block)
        for name, block in root.named_modules()
        if name and isinstance(block, DiTBlock)
    ]
    if not candidates:
        return AttentionSiteStatus(None, 0, "not_minimax_h3")
    if not is_supported_attention_device(device):
        return AttentionSiteStatus("minimax_h3", 0, "not_supported_tensor_core")
    if not callable(getattr(model, "add_object_patch", None)):
        return AttentionSiteStatus("minimax_h3", 0, "model_patcher_api_unavailable")
    if not fused_qk_preprocessing_available():
        return AttentionSiteStatus("minimax_h3", 0, "fused_qk_unavailable")
    if not _compatible_attention_forward(Attention):
        return AttentionSiteStatus("minimax_h3", 0, "attention_contract_changed")

    object_patches = getattr(model, "object_patches", {})
    installed = 0
    for name, block in candidates:
        if type(block.attn) is not Attention:
            continue
        key = f"{name}.attn.forward"
        current = object_patches.get(key, block.attn.forward)
        if _has_prepared_attention_forward(current):
            continue
        model.add_object_patch(
            key,
            _make_attention_forward(
                block.attn,
                AttentionTensorContainer,
                current,
                root,
            ),
        )
        installed += 1
    return AttentionSiteStatus("minimax_h3", installed, None)


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


def _stream_mlp(mlp: torch.nn.Module, x: torch.Tensor, chunk_rows: int):
    """Evaluate an H3 SwiGLU MLP in row tiles with one weight cast per layer."""
    import comfy.ops

    comfy.ops.run_every_op()
    fc1_weight, fc1_bias, fc1_stream = comfy.ops.cast_bias_weight(
        mlp.fc1,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=False,
    )
    fc2_weight = fc2_bias = fc2_stream = None
    try:
        fc2_weight, fc2_bias, fc2_stream = comfy.ops.cast_bias_weight(
            mlp.fc2,
            x,
            offloadable=True,
            compute_dtype=x.dtype,
            want_requant=True,
        )
        output_features = int(getattr(mlp.fc2, "out_features", x.shape[-1]))
        output = torch.empty(
            (x.shape[0], output_features), dtype=x.dtype, device=x.device
        )
        for start in range(0, x.shape[0], chunk_rows):
            stop = min(start + chunk_rows, x.shape[0])
            expanded = _linear_with_cast_weight(
                mlp.fc1, x[start:stop], fc1_weight, fc1_bias
            )
            tile = convrot_linear_input_act_from_weight(
                fc2_weight, fc2_bias, expanded, "swiglu"
            )
            output[start:stop].copy_(tile)
            del expanded, tile
        return output
    finally:
        if fc2_weight is not None:
            comfy.ops.uncast_bias_weight(
                mlp.fc2, fc2_weight, fc2_bias, fc2_stream
            )
        comfy.ops.uncast_bias_weight(
            mlp.fc1, fc1_weight, fc1_bias, fc1_stream
        )


def _ffn_expanded_shard(
    mlp,
    qactivation: torch.Tensor,
    activation_scale: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    expanded_size: int,
    start: int,
    stop: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Project one aligned gate/up interval without a full fc1 output."""
    width = stop - start
    expanded = torch.empty(
        (qactivation.shape[0], 2 * width),
        dtype=output_dtype,
        device=qactivation.device,
    )
    gate = convrot_w8_output_slice(
        qactivation,
        activation_scale,
        qweight,
        weight_scale,
        bias,
        start,
        stop,
        output_dtype,
    )
    expanded[:, :width].copy_(gate)
    del gate
    up = convrot_w8_output_slice(
        qactivation,
        activation_scale,
        qweight,
        weight_scale,
        bias,
        expanded_size + start,
        expanded_size + stop,
        output_dtype,
    )
    expanded[:, width:].copy_(up)
    del up
    return expanded


def _stream_mlp_channels(
    mlp: torch.nn.Module,
    x: torch.Tensor,
    *,
    chunk_rows: int,
    chunk_channels: int,
):
    """Exact two-pass H3 FFN evaluation over ConvRot-aligned channels.

    The first pass finds the same whole-row activation scale as the unsharded
    fused SwiGLU quantizer.  The second pass recomputes each aligned interval,
    uses that common scale, and writes directly into the final compact A8 row.
    The unchanged fused fc2 consumes that row once, so no shard boundary enters
    the contraction or its BF16 epilogue.
    """
    import comfy.ops

    if chunk_channels <= 0 or chunk_channels % 256:
        return None
    if getattr(mlp.fc2, "pre_quant_scale", None) is not None:
        # This is not used by MiniMax H3 today.  Falling back avoids changing
        # the ordering of a future SmoothQuant-style pre-scale.
        return None

    comfy.ops.run_every_op()
    fc1_weight, fc1_bias, fc1_stream = comfy.ops.cast_bias_weight(
        mlp.fc1,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=True,
    )
    fc2_weight = fc2_bias = fc2_stream = None
    try:
        fc2_weight, fc2_bias, fc2_stream = comfy.ops.cast_bias_weight(
            mlp.fc2,
            x,
            offloadable=True,
            compute_dtype=x.dtype,
            want_requant=True,
        )
        fc1_plain = convrot_w8_plain_tensors(fc1_weight)
        fc2_plain = convrot_w8_plain_tensors(fc2_weight)
        if fc1_plain is None or fc2_plain is None or fc2_bias is not None:
            return None
        fc1_qweight, fc1_weight_scale = fc1_plain
        fc2_qweight, fc2_weight_scale = fc2_plain
        expanded_size = int(getattr(mlp.fc2, "in_features", 0))
        if (
            expanded_size <= 0
            or expanded_size % 256
            or fc1_qweight.shape[0] != 2 * expanded_size
            or fc2_qweight.shape[1] != expanded_size
        ):
            return None

        output_features = int(fc2_qweight.shape[0])
        output = torch.empty(
            (x.shape[0], output_features), dtype=x.dtype, device=x.device
        )
        for row_start in range(0, x.shape[0], chunk_rows):
            row_stop = min(row_start + chunk_rows, x.shape[0])
            qactivation, input_scale = _quantize_qkv_rows(
                mlp.fc1, x[row_start:row_stop]
            )

            whole_row_scale = None
            for start in range(0, expanded_size, chunk_channels):
                stop = min(start + chunk_channels, expanded_size)
                expanded = _ffn_expanded_shard(
                    mlp,
                    qactivation,
                    input_scale,
                    fc1_qweight,
                    fc1_weight_scale,
                    fc1_bias,
                    expanded_size,
                    start,
                    stop,
                    x.dtype,
                )
                local_quantized, local_scale = (
                    quantize_convrot_swiglu_activation(expanded, 256)
                )
                del expanded, local_quantized
                if whole_row_scale is None:
                    whole_row_scale = local_scale
                else:
                    torch.maximum(
                        whole_row_scale, local_scale, out=whole_row_scale
                    )
                    del local_scale

            activated = torch.empty(
                (row_stop - row_start, expanded_size),
                dtype=torch.int8,
                device=x.device,
            )
            for start in range(0, expanded_size, chunk_channels):
                stop = min(start + chunk_channels, expanded_size)
                expanded = _ffn_expanded_shard(
                    mlp,
                    qactivation,
                    input_scale,
                    fc1_qweight,
                    fc1_weight_scale,
                    fc1_bias,
                    expanded_size,
                    start,
                    stop,
                    x.dtype,
                )
                quantized = quantize_convrot_swiglu_with_scale(
                    expanded, whole_row_scale, 256
                )
                del expanded
                activated[:, start:stop].copy_(quantized)
                del quantized

            tile = int8_linear_from_quantized(
                activated,
                whole_row_scale,
                fc2_qweight,
                fc2_weight_scale,
                bias=None,
                out_dtype=x.dtype,
            )
            output[row_start:row_stop].copy_(tile)
            del (
                activated,
                input_scale,
                qactivation,
                tile,
                whole_row_scale,
            )
        return output
    finally:
        if fc2_weight is not None:
            comfy.ops.uncast_bias_weight(
                mlp.fc2, fc2_weight, fc2_bias, fc2_stream
            )
        comfy.ops.uncast_bias_weight(
            mlp.fc1, fc1_weight, fc1_bias, fc1_stream
        )


def _make_mlp_forward(
    mlp: torch.nn.Module,
    audit: _RuntimeDispatchAudit,
    base_model=None,
):
    original = OriginalMethod.capture(mlp.forward, mlp)
    base_model = _weak_model_reference(base_model)

    def forward(self, x: torch.Tensor):
        blocker = None
        if x.dtype != torch.bfloat16:
            blocker = f"dtype={x.dtype}"
        elif not is_turing_convrot_linear(self.fc2):
            blocker = "fc2_not_turing_convrot"
        elif not is_supported_attention_device(x.device):
            blocker = f"device={x.device}"
        audit.record("mlp", blocker is None, x, blocker)
        if blocker is not None:
            return original(self, x)
        expanded_size = int(getattr(self.fc2, "in_features", 0))
        runtime_plan = _runtime_activation_plan(base_model)
        decision = decide_activation_chunks(
            x,
            operation="mlp",
            hidden_size=int(x.shape[-1]),
            expanded_size=expanded_size,
            runtime_plan=runtime_plan,
            base_model=base_model,
        )
        if convrot_swiglu_channel_sharding_available():
            channel_decision = decide_ffn_channels(
                x,
                expanded_size=expanded_size,
                chunk_rows=decision.chunk_rows,
                runtime_plan=runtime_plan,
                base_model=base_model,
            )
            if channel_decision.sharded:
                if base_model is not None:
                    ensure_dynamic_vram_headroom(
                        base_model,
                        x.device,
                        rows=int(x.shape[0]),
                        operation="ffn_channels",
                        estimated_peak_bytes=channel_decision.estimated_peak_bytes,
                        runtime_plan=runtime_plan,
                    )
                output = _stream_mlp_channels(
                    self,
                    x,
                    chunk_rows=channel_decision.chunk_rows,
                    chunk_channels=channel_decision.chunk_channels,
                )
                if output is not None:
                    return output
        if base_model is not None:
            ensure_dynamic_vram_headroom(
                base_model,
                x.device,
                rows=int(x.shape[0]),
                operation="mlp",
                estimated_peak_bytes=decision.streamed_peak_bytes,
                runtime_plan=runtime_plan,
            )
        if decision.streamed:
            return _stream_mlp(self, x, decision.chunk_rows)
        return fused_convrot_linear_input_act(
            self.fc2, self.fc1(x), "swiglu"
        )

    return weak_method(forward, mlp)


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
    original = OriginalMethod.capture(block.forward, block)
    if base_model is not None:
        base_model = weakref.proxy(base_model)
    if diffusion_model is not None:
        diffusion_model = weakref.proxy(diffusion_model)

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
                self,
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

    return mark_forward_as_minimax_layout_provider(weak_method(forward, block))

def apply_minimax_adapter(model, device: torch.device) -> int:
    """Install MiniMax-only forward substitutions through the ModelPatcher."""
    if not is_supported_attention_device(device):
        return 0
    if not hasattr(model, "add_object_patch"):
        raise RuntimeError("MiniMax CUDA integration requires a ComfyUI ModelPatcher")

    try:
        from comfy.ldm.minimax.model import DiTBlock, _mod_gate
    except ImportError:
        return 0
    if not _compatible_block_forward(DiTBlock):
        LOG.warning("MiniMax CUDA fusions are disabled because the DiTBlock forward contract changed")
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
    attention_fusions = install_minimax_attention_sites(model, device).installed
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
                _make_mlp_forward(block.mlp, audit, root),
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
        LOG.info("Enabled MiniMax segmented RMSNorm+AdaLN on %d CUDA blocks", block_fusions)
    if mlp_fusions:
        LOG.info("Enabled MiniMax fused/streamed ConvRot SwiGLU on %d MLP layers", mlp_fusions)
    if attention_fusions:
        LOG.info(
            "Enabled MiniMax fused Q/K RMSNorm+RoPE+INT8 preprocessing on %d attention layers",
            attention_fusions,
        )
    if eligible_fc2 and mlp_fusions != eligible_fc2:
        raise RuntimeError("MiniMax fused fc2 adapter did not patch every eligible layer")
    return max(block_fusions, mlp_fusions, attention_fusions)
