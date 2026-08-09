"""Optional MiniMax integration for generic Turing fusions."""

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

try:
    from .turing_fusions import (
        convrot_weight_kind,
        is_turing_convrot_linear,
        segmented_rms_adaln,
        turing_linear_input_act,
    )
    from .turing_ops import is_supported_turing_device, turing_int8_workspace_bytes
except ImportError:
    from turing_fusions import (
        convrot_weight_kind,
        is_turing_convrot_linear,
        segmented_rms_adaln,
        turing_linear_input_act,
    )
    from turing_ops import is_supported_turing_device, turing_int8_workspace_bytes


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
_MEMORY_CONTEXT_ATTR = "_turing_utils_minimax_memory_context"
_MEMORY_ADAPTER_ATTR = "_turing_utils_minimax_memory_adapter"
_OUTER_SAMPLE_WRAPPER_KEY = "turing_utils_minimax_memory_context"
_ATTENTION_LAYOUT_KEY = "turing_utils_attention_layout"
_PROGRESSIVE_OUTER_WRAPPER_KEY = "turing_utils_h3_progressive_resolution_steps"
_PROGRESSIVE_COND_WRAPPER_KEY = "turing_utils_h3_progressive_resolution_cond"


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


def _make_outer_sample_wrapper(base_model):
    def outer_sample_wrapper(executor, *args, **kwargs):
        latent_shapes = kwargs.get("latent_shapes")
        if latent_shapes is None and len(args) > 8:
            latent_shapes = args[8]
        previous = getattr(base_model, _MEMORY_CONTEXT_ATTR, None)
        setattr(base_model, _MEMORY_CONTEXT_ATTR, {"latent_shapes": latent_shapes})
        try:
            return executor(*args, **kwargs)
        finally:
            if previous is None:
                try:
                    delattr(base_model, _MEMORY_CONTEXT_ATTR)
                except AttributeError:
                    pass
            else:
                setattr(base_model, _MEMORY_CONTEXT_ATTR, previous)

    return outer_sample_wrapper


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

    import comfy.patcher_extension

    model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        _OUTER_SAMPLE_WRAPPER_KEY,
        _make_outer_sample_wrapper(base_model),
    )
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


def _minimax_temporal_topology(base_model, diffusion_model, mod_segments):
    """Describe only the contiguous target-video tail; the sparse kernel stays generic."""
    if base_model is None or diffusion_model is None or not mod_segments:
        return {}
    context = getattr(base_model, _MEMORY_CONTEXT_ATTR, None)
    latent_shapes = context.get("latent_shapes") if isinstance(context, dict) else None
    if not isinstance(latent_shapes, (list, tuple)) or not latent_shapes:
        return {}
    video_shape = tuple(int(value) for value in latent_shapes[0])
    if len(video_shape) != 5:
        return {}
    patch_size = tuple(int(value) for value in diffusion_model.patch_size)
    if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
        return {}
    pt, ph, pw = patch_size
    frames = math.ceil(video_shape[2] / pt)
    spatial_tokens_height = math.ceil(video_shape[3] / ph)
    spatial_tokens_width = math.ceil(video_shape[4] / pw)
    tokens_per_frame = spatial_tokens_height * spatial_tokens_width
    topology_tokens = frames * tokens_per_frame
    topology_start = int(mod_segments[-1][0])
    if int(mod_segments[-1][1]) - topology_start != topology_tokens:
        return {}
    return {
        "topology_start_tokens": topology_start,
        "topology_tokens": topology_tokens,
        "tokens_per_frame": tokens_per_frame,
        "spatial_tokens_height": spatial_tokens_height,
        "spatial_tokens_width": spatial_tokens_width,
    }


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
        if mod_segments:
            # H3 packs target audio immediately before target video.  Keep the
            # complete non-video prefix exact: target-audio queries need global
            # video context, and video queries need exact audio keys for stable
            # joint generation.  Sparsifying target audio into 64-token
            # centroids produces broadband noise even when the video remains
            # visually plausible.
            dense_prefix_tokens = int(mod_segments[-1][0])
            layout = transformer_options.get(_ATTENTION_LAYOUT_KEY)
            expected_layout = {
                "dense_prefix_tokens": dense_prefix_tokens,
                "layer_index": layer_index,
                "layer_count": layer_count,
                **_minimax_temporal_topology(
                    base_model,
                    diffusion_model,
                    mod_segments,
                ),
            }
            if not isinstance(layout, dict) or any(
                layout.get(key) != value for key, value in expected_layout.items()
            ):
                transformer_options[_ATTENTION_LAYOUT_KEY] = {
                    **(layout if isinstance(layout, dict) else {}),
                    **expected_layout,
                }
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

    return types.MethodType(forward, block)


@dataclass(frozen=True)
class _H3ProgressiveResolutionConfig:
    low_short_edge: int
    low_resolution_steps: int
    medium_short_edge: int
    medium_resolution_steps: int
    input_downscale: str
    output_upscale: str
    visual_condition_policy: str
    debug: bool = False


def _h3_latent_shapes(conds):
    """Find the processed H3 packed-stream shapes in a conditioning batch."""
    for cond_list in conds:
        if cond_list is None:
            continue
        for cond in cond_list:
            model_conds = cond.get("model_conds", {})
            shape_cond = model_conds.get("latent_shapes")
            shapes = getattr(shape_cond, "cond", None)
            if not isinstance(shapes, (list, tuple)) or len(shapes) < 2:
                continue
            video_shape = tuple(int(value) for value in shapes[0])
            audio_shape = tuple(int(value) for value in shapes[1])
            if (
                len(video_shape) == 5
                and len(audio_shape) == 4
                and video_shape[1] == 24
                and audio_shape[1] == 32
                and audio_shape[2] == 2
            ):
                return list(shapes)
    return None


def _h3_progressive_target_hw(video_shape, low_short_edge: int) -> tuple[int, int]:
    """Return an aspect-preserving H3 latent size aligned to 32-pixel canvas units."""
    final_h, final_w = int(video_shape[-2]), int(video_shape[-1])
    final_short_pixels = min(final_h, final_w) * 16
    if low_short_edge <= 0 or low_short_edge >= final_short_pixels:
        return final_h, final_w

    scale = float(low_short_edge) / float(final_short_pixels)

    def aligned(value: int) -> int:
        # H3 consumes 2x2 latent patches, corresponding to 32x32 pixel units.
        return min(value, max(2, int(round(value * scale / 2.0)) * 2))

    return aligned(final_h), aligned(final_w)


def _resize_h3_video(video: torch.Tensor, height: int, width: int, method: str) -> torch.Tensor:
    if tuple(video.shape[-2:]) == (int(height), int(width)):
        return video
    if method == "h3_rope_bilinear":
        return _resize_h3_video_patch_grid(video, height, width, "rope_bilinear")
    if method == "h3_rope_nearest":
        return _resize_h3_video_patch_grid(video, height, width, "rope_nearest")
    if method == "h3_patch_area":
        return _resize_h3_video_patch_grid(video, height, width, "area")
    import comfy.utils

    return comfy.utils.common_upscale(
        video,
        int(width),
        int(height),
        method,
        "disabled",
    )


def _resize_h3_video_patch_grid(
    video: torch.Tensor,
    height: int,
    width: int,
    method: str,
) -> torch.Tensor:
    """Resize H3 video latents without mixing the model's four 2x2 patch phases.

    H3 patchifies every spatial 2x2 group into the feature order
    ``[channel, patch_y, patch_x]``.  Treating the latent as an ordinary image
    blends those four feature phases together.  This helper instead resizes a
    ``channels * 4`` feature map on the DiT patch grid.  RoPE modes sample that
    grid at the exact area-normalized coordinates used by H3's positional
    encoding rather than PyTorch's half-pixel convention.
    """
    if video.ndim != 5:
        raise ValueError(f"Expected H3 video latent [B,C,T,H,W], got {tuple(video.shape)}")
    height = int(height)
    width = int(width)
    if height < 1 or width < 1:
        raise ValueError(f"H3 latent resize requires positive dimensions, got {height}x{width}")
    if tuple(video.shape[-2:]) == (height, width):
        return video

    batch, channels, frames, source_h, source_w = video.shape
    # Match H3's circular pad-to-patch behavior for uncommon odd latent sizes.
    padded = video
    if source_h % 2:
        padded = torch.cat((padded, padded[..., :1, :]), dim=-2)
    if source_w % 2:
        padded = torch.cat((padded, padded[..., :1]), dim=-1)
    padded_h, padded_w = int(padded.shape[-2]), int(padded.shape[-1])
    target_h = (height + 1) // 2 * 2
    target_w = (width + 1) // 2 * 2
    source_patch_h, source_patch_w = padded_h // 2, padded_w // 2
    target_patch_h, target_patch_w = target_h // 2, target_w // 2

    # [B,C,T,H,2,W,2] -> [B*T,C*4,H,W], matching patchify_video.
    patch_features = (
        padded.reshape(
            batch,
            channels,
            frames,
            source_patch_h,
            2,
            source_patch_w,
            2,
        )
        .permute(0, 2, 1, 4, 6, 3, 5)
        .reshape(batch * frames, channels * 4, source_patch_h, source_patch_w)
    )
    original_dtype = patch_features.dtype
    work = patch_features.float()

    if method == "area":
        resized = torch.nn.functional.interpolate(
            work,
            size=(target_patch_h, target_patch_w),
            mode="area",
        )
    elif method in ("rope_bilinear", "rope_nearest"):
        def rope_axis(dim: int, other: int, device) -> torch.Tensor:
            count = dim // 2
            sqrt_area = math.sqrt(float(dim * other))
            ratio = float(dim) / sqrt_area
            return (
                torch.arange(count, device=device, dtype=torch.float32)
                * (ratio / float(count))
                + (1.0 - ratio) / 2.0
            ) * 32.0

        source_y = rope_axis(padded_h, padded_w, work.device)
        source_x = rope_axis(padded_w, padded_h, work.device)
        target_y = rope_axis(target_h, target_w, work.device)
        target_x = rope_axis(target_w, target_h, work.device)

        def normalized_source_indices(
            target_axis: torch.Tensor,
            source_axis: torch.Tensor,
        ) -> torch.Tensor:
            if source_axis.numel() <= 1:
                return torch.zeros_like(target_axis)
            step = source_axis[1] - source_axis[0]
            indices = (target_axis - source_axis[0]) / step
            return indices * (2.0 / float(source_axis.numel() - 1)) - 1.0

        grid_y = normalized_source_indices(target_y, source_y)
        grid_x = normalized_source_indices(target_x, source_x)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        grid = grid.expand(batch * frames, -1, -1, -1)
        resized = torch.nn.functional.grid_sample(
            work,
            grid,
            mode="bilinear" if method == "rope_bilinear" else "nearest",
            padding_mode="border",
            align_corners=True,
        )
    else:
        raise ValueError(f"Unsupported H3 patch-grid resize method: {method}")

    # [B*T,C*4,H,W] -> [B,C,T,H,2,W,2], then discard only pad cells.
    restored = (
        resized.to(original_dtype)
        .reshape(batch, frames, channels, 2, 2, target_patch_h, target_patch_w)
        .permute(0, 2, 1, 5, 3, 6, 4)
        .reshape(batch, channels, frames, target_h, target_w)
    )
    return restored[..., :height, :width]


def _downsample_h3_video(
    video: torch.Tensor,
    height: int,
    width: int,
    config: _H3ProgressiveResolutionConfig,
    state: dict,
) -> torch.Tensor:
    if config.input_downscale not in ("sigma_blend", "h3_rope_sigma_blend"):
        return _resize_h3_video(video, height, width, config.input_downscale)

    # Nearest-exact preserves the variance of the already-sampled high-resolution
    # noise. Area filtering preserves the emerging low-frequency composition. The
    # deterministic blend avoids injecting a second, unrelated noise trajectory.
    if config.input_downscale == "h3_rope_sigma_blend":
        nearest_method = "h3_rope_nearest"
        area_method = "h3_patch_area"
    else:
        nearest_method = "nearest-exact"
        area_method = "area"
    nearest = _resize_h3_video(video, height, width, nearest_method)
    area = _resize_h3_video(video, height, width, area_method)
    progressive_steps = max(int(state.get("progressive_steps", 1)), 1)
    step = max(int(state.get("step", 0)), 0)
    progressive_sigmas = state.get("progressive_sigmas", ())
    if len(progressive_sigmas) >= 2:
        sigma_start = float(progressive_sigmas[0])
        sigma_end = float(progressive_sigmas[-1])
        sigma = float(progressive_sigmas[min(step, len(progressive_sigmas) - 1)])
        denominator = sigma_start - sigma_end
        nearest_weight = (
            (sigma - sigma_end) / denominator
            if abs(denominator) > 1e-12
            else 0.0
        )
        nearest_weight = min(max(nearest_weight, 0.0), 1.0)
    else:
        nearest_weight = max(0.0, 1.0 - float(step) / float(progressive_steps))
    return torch.lerp(area, nearest, nearest_weight)


def _resize_h3_keyframe_payload(
    payload,
    height: int,
    width: int,
    cache: dict,
    method: str = "area",
):
    if not isinstance(payload, dict) or not payload.get("keyframes"):
        return payload

    resized_keyframes = []
    for keyframe in payload["keyframes"]:
        if not isinstance(keyframe, dict) or not torch.is_tensor(keyframe.get("latent")):
            resized_keyframes.append(keyframe)
            continue
        latent = keyframe["latent"]
        cache_key = (id(latent), int(height), int(width), method)
        resized = cache.get(cache_key)
        if resized is None:
            resized = _resize_h3_video(latent, height, width, method)
            cache[cache_key] = resized
        new_keyframe = keyframe.copy()
        new_keyframe["latent"] = resized
        resized_keyframes.append(new_keyframe)

    new_payload = payload.copy()
    new_payload["keyframes"] = resized_keyframes
    original_cond_latents = list(payload.get("cond_video_latents", ()))
    resized_latents = [
        keyframe["latent"]
        for keyframe in resized_keyframes
        if isinstance(keyframe, dict) and torch.is_tensor(keyframe.get("latent"))
    ]
    # extra_conds orders keyframes before independent reference latents.
    new_payload["cond_video_latents"] = resized_latents + original_cond_latents[
        len(payload["keyframes"]):
    ]
    # Target and condition geometry both changed; force a fresh lightweight layout.
    new_payload.pop("layout", None)
    return new_payload


def _resize_h3_memory_condition(
    memory_cond,
    high_shapes,
    low_shapes,
    *,
    resized_keyframes: int = 0,
):
    if not isinstance(memory_cond, _MiniMaxMemoryCond):
        return memory_cond
    old = memory_cond.cond
    video_shape = tuple(int(value) for value in low_shapes[0])
    audio_shape = tuple(int(value) for value in low_shapes[1])
    latent_t = int(video_shape[2])
    frame_rows = math.ceil(video_shape[3] / 2) * math.ceil(video_shape[4] / 2)
    target_visual_rows = latent_t * frame_rows
    target_audio_rows = int(audio_shape[2]) * int(audio_shape[3])
    target_rows = target_visual_rows + target_audio_rows

    old_video_shape = tuple(int(value) for value in high_shapes[0])
    old_frame_rows = math.ceil(old_video_shape[3] / 2) * math.ceil(old_video_shape[4] / 2)
    visual_condition_rows = int(old.visual_condition_rows) + int(resized_keyframes) * (
        frame_rows - old_frame_rows
    )
    condition_rows = max(int(old.full_rows) - int(old.target_rows), 0)
    condition_rows += visual_condition_rows - int(old.visual_condition_rows)
    full_rows = target_rows + condition_rows
    target_area = math.prod(video_shape[1:]) + math.prod(audio_shape[1:])
    equivalent_area = math.ceil(target_area * condition_rows / max(target_rows, 1))
    return _MiniMaxMemoryCond(
        _MiniMaxMemoryShape(
            equivalent_area,
            full_rows=full_rows,
            target_rows=target_rows,
            target_visual_rows=target_visual_rows,
            target_audio_rows=target_audio_rows,
            visual_condition_rows=visual_condition_rows,
            audio_condition_rows=int(old.audio_condition_rows),
            hidden_size=int(old.hidden_size),
            video_row_width=int(old.video_row_width),
            audio_row_width=int(old.audio_row_width),
        )
    )


def _patch_h3_conds_for_shapes(
    conds,
    high_shapes,
    low_shapes,
    config: _H3ProgressiveResolutionConfig,
    state: dict,
):
    import comfy.conds

    shape_cond = comfy.conds.CONDConstant(low_shapes)
    output = []
    payload_memo = {}
    for cond_list in conds:
        if cond_list is None:
            output.append(None)
            continue
        patched_list = []
        for cond in cond_list:
            patched_cond = cond.copy()
            model_conds = cond.get("model_conds")
            if not isinstance(model_conds, dict):
                patched_list.append(patched_cond)
                continue
            patched_model_conds = model_conds.copy()
            if "latent_shapes" in patched_model_conds:
                patched_model_conds["latent_shapes"] = shape_cond
            keyframe_count = 0
            if config.visual_condition_policy == "resize_keyframes":
                payload_cond = patched_model_conds.get("minimax_payload")
                payload = getattr(payload_cond, "cond", None)
                if isinstance(payload, dict) and payload.get("keyframes"):
                    keyframe_count = len(payload["keyframes"])
                    memo_key = id(payload_cond)
                    patched_payload_cond = payload_memo.get(memo_key)
                    if patched_payload_cond is None:
                        patched_payload = _resize_h3_keyframe_payload(
                            payload,
                            int(low_shapes[0][-2]),
                            int(low_shapes[0][-1]),
                            state.setdefault("condition_cache", {}),
                            (
                                "h3_patch_area"
                                if config.input_downscale.startswith("h3_")
                                else "area"
                            ),
                        )
                        patched_payload_cond = comfy.conds.CONDConstant(patched_payload)
                        payload_memo[memo_key] = patched_payload_cond
                    patched_model_conds["minimax_payload"] = patched_payload_cond
            if _MEMORY_SHAPE_KEY in patched_model_conds:
                patched_model_conds[_MEMORY_SHAPE_KEY] = _resize_h3_memory_condition(
                    patched_model_conds[_MEMORY_SHAPE_KEY],
                    high_shapes,
                    low_shapes,
                    resized_keyframes=keyframe_count,
                )
            patched_cond["model_conds"] = patched_model_conds
            patched_list.append(patched_cond)
        output.append(patched_list)
    return output


def _h3_conds_support_progressive_resize(conds) -> bool:
    # Spatial areas, masks, and controls need their own shape transformations.
    # The official H3 text/keyframe/reference paths do not use these fields.
    unsupported = ("area", "mask", "control", "gligen")
    return all(
        not any(key in cond for key in unsupported)
        for cond_list in conds
        if cond_list is not None
        for cond in cond_list
    )


def _make_h3_progressive_wrappers(config: _H3ProgressiveResolutionConfig):
    runtime = contextvars.ContextVar(
        "turing_utils_h3_progressive_resolution_runtime",
        default=None,
    )

    def outer_sample_wrapper(executor, *args, **kwargs):
        sigmas = kwargs.get("sigmas")
        if sigmas is None and len(args) > 3:
            sigmas = args[3]
        total_steps = max(int(getattr(sigmas, "shape", (0,))[-1]) - 1, 0)
        low_steps = min(max(int(config.low_resolution_steps), 0), total_steps)
        medium_steps = min(
            max(int(config.medium_resolution_steps), 0),
            total_steps - low_steps,
        )
        progressive_steps = low_steps + medium_steps
        if progressive_steps <= 0:
            return executor(*args, **kwargs)

        state = {
            "step": 0,
            "low_steps": low_steps,
            "medium_steps": medium_steps,
            "progressive_steps": progressive_steps,
            "total_steps": total_steps,
            "progressive_sigmas": (
                sigmas[:progressive_steps + 1].detach().to("cpu", torch.float32).tolist()
                if torch.is_tensor(sigmas)
                else ()
            ),
            "condition_cache": {},
            "logged_stages": set(),
            "fallback_logged": False,
        }
        callback = kwargs.get("callback")
        callback_in_args = len(args) > 5
        if callback_in_args:
            callback = args[5]

        def progressive_callback(step, x0, x, callback_total_steps):
            state["step"] = max(int(state["step"]), int(step) + 1)
            if callback is not None:
                return callback(step, x0, x, callback_total_steps)
            return None

        if callback_in_args:
            args_list = list(args)
            args_list[5] = progressive_callback
            args = tuple(args_list)
        else:
            kwargs = kwargs.copy()
            kwargs["callback"] = progressive_callback

        token = runtime.set(state)
        try:
            return executor(*args, **kwargs)
        finally:
            runtime.reset(token)

    def calc_cond_batch_wrapper(executor, model, conds, x_in, timestep, model_options):
        state = runtime.get()
        if state is None or int(state["step"]) >= int(state["progressive_steps"]):
            return executor(model, conds, x_in, timestep, model_options)
        if not _h3_conds_support_progressive_resize(conds):
            if config.debug and not state["fallback_logged"]:
                LOG.warning(
                    "H3 progressive resolution skipped a staged step because spatial areas, masks, or controls are attached"
                )
                state["fallback_logged"] = True
            return executor(model, conds, x_in, timestep, model_options)

        high_shapes = _h3_latent_shapes(conds)
        if high_shapes is None:
            return executor(model, conds, x_in, timestep, model_options)
        video_shape = high_shapes[0]
        if int(state["step"]) < int(state["low_steps"]):
            stage_name = "low"
            stage_short_edge = config.low_short_edge
        else:
            stage_name = "medium"
            stage_short_edge = config.medium_short_edge
        low_h, low_w = _h3_progressive_target_hw(video_shape, stage_short_edge)
        if (low_h, low_w) == tuple(video_shape[-2:]):
            return executor(model, conds, x_in, timestep, model_options)

        import comfy.utils

        high_streams = list(comfy.utils.unpack_latents(x_in, high_shapes))
        low_video = _downsample_h3_video(
            high_streams[0],
            low_h,
            low_w,
            config,
            state,
        )
        low_streams = [low_video, *high_streams[1:]]
        low_x, low_shapes = comfy.utils.pack_latents(low_streams)
        low_conds = _patch_h3_conds_for_shapes(
            conds,
            high_shapes,
            low_shapes,
            config,
            state,
        )

        if config.debug and stage_name not in state["logged_stages"]:
            LOG.warning(
                "Experimental H3 progressive resolution active: stage=%s step_range=%d:%d total_processed=%d/%d video_latent=%sx%s -> %sx%s input=%s output=%s",
                stage_name,
                0 if stage_name == "low" else state["low_steps"],
                state["low_steps"] if stage_name == "low" else state["progressive_steps"],
                state["progressive_steps"],
                state["total_steps"],
                video_shape[-1],
                video_shape[-2],
                low_w,
                low_h,
                config.input_downscale,
                config.output_upscale,
            )
            state["logged_stages"].add(stage_name)

        previous_memory_context = (
            getattr(model, _MEMORY_CONTEXT_ATTR, None) if model is not None else None
        )
        if model is not None:
            setattr(model, _MEMORY_CONTEXT_ATTR, {"latent_shapes": low_shapes})
        try:
            low_outputs = executor(model, low_conds, low_x, timestep, model_options)
        finally:
            if model is None:
                pass
            elif previous_memory_context is None:
                try:
                    delattr(model, _MEMORY_CONTEXT_ATTR)
                except AttributeError:
                    pass
            else:
                setattr(model, _MEMORY_CONTEXT_ATTR, previous_memory_context)
        high_outputs = []
        for low_output in low_outputs:
            output_streams = list(comfy.utils.unpack_latents(low_output, low_shapes))
            output_streams[0] = _resize_h3_video(
                output_streams[0],
                int(video_shape[-2]),
                int(video_shape[-1]),
                config.output_upscale,
            )
            packed_output, _ = comfy.utils.pack_latents(output_streams)
            high_outputs.append(packed_output)
        return high_outputs

    return outer_sample_wrapper, calc_cond_batch_wrapper


def apply_h3_progressive_resolution_patch(
    model,
    *,
    low_short_edge: int = 480,
    low_resolution_steps: int = 2,
    medium_short_edge: int = 720,
    medium_resolution_steps: int = 0,
    input_downscale: str = "sigma_blend",
    output_upscale: str = "bilinear",
    visual_condition_policy: str = "resize_keyframes",
    debug: bool = False,
):
    """Run early H3 DiT evaluations at lower spatial resolution while keeping one final-resolution sampler state."""
    input_methods = (
        "sigma_blend",
        "h3_rope_sigma_blend",
        "h3_rope_bilinear",
        "h3_rope_nearest",
        "nearest-exact",
        "area",
    )
    output_methods = (
        "bilinear",
        "h3_rope_bilinear",
        "h3_rope_nearest",
        "bicubic",
        "nearest-exact",
    )
    condition_policies = ("resize_keyframes", "keep_original")
    if input_downscale not in input_methods:
        raise ValueError(f"Unsupported input_downscale: {input_downscale}")
    if output_upscale not in output_methods:
        raise ValueError(f"Unsupported output_upscale: {output_upscale}")
    if visual_condition_policy not in condition_policies:
        raise ValueError(f"Unsupported visual_condition_policy: {visual_condition_policy}")
    if int(low_short_edge) < 32:
        raise ValueError("low_short_edge must be at least 32 pixels")
    if int(medium_short_edge) < 32:
        raise ValueError("medium_short_edge must be at least 32 pixels")
    if int(low_resolution_steps) < 0:
        raise ValueError("low_resolution_steps must be non-negative")
    if int(medium_resolution_steps) < 0:
        raise ValueError("medium_resolution_steps must be non-negative")

    import comfy.patcher_extension

    config = _H3ProgressiveResolutionConfig(
        low_short_edge=int(low_short_edge),
        low_resolution_steps=int(low_resolution_steps),
        medium_short_edge=int(medium_short_edge),
        medium_resolution_steps=int(medium_resolution_steps),
        input_downscale=input_downscale,
        output_upscale=output_upscale,
        visual_condition_policy=visual_condition_policy,
        debug=bool(debug),
    )
    outer_wrapper, cond_wrapper = _make_h3_progressive_wrappers(config)
    patched = model.clone()
    if callable(getattr(patched, "remove_wrappers_with_key", None)):
        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            _PROGRESSIVE_OUTER_WRAPPER_KEY,
        )
        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
            _PROGRESSIVE_COND_WRAPPER_KEY,
        )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        _PROGRESSIVE_OUTER_WRAPPER_KEY,
        outer_wrapper,
    )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
        _PROGRESSIVE_COND_WRAPPER_KEY,
        cond_wrapper,
    )
    LOG.info(
        "Enabled experimental H3 progressive resolution: low_edge=%d low_steps=%d medium_edge=%d medium_steps=%d input=%s output=%s visual_conditions=%s",
        config.low_short_edge,
        config.low_resolution_steps,
        config.medium_short_edge,
        config.medium_resolution_steps,
        config.input_downscale,
        config.output_upscale,
        config.visual_condition_policy,
    )
    return patched


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

    diffusion_model = getattr(root, "diffusion_model", None)
    if diffusion_model is not None:
        _install_memory_planning(model, root, diffusion_model)

    eligible_fc2 = _audit_fc2([block for _, block in candidates])
    index = device.index if device.index is not None else torch.cuda.current_device()
    block_fusions = 0
    mlp_fusions = 0
    try:
        from comfyui_turing_utils_kernel import turing_segmented_rms_adaln
    except (ImportError, AttributeError):
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
