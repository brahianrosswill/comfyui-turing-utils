"""Loader-independent MiniMax H3 attention layout publication.

Sparse attention sees only the flattened packed token sequence.  This module
bridges the official H3 runtime information (nested latent shapes and block
``mod_segments``) into a small, per-forward ``transformer_options`` contract.
It deliberately contains no quantization, Turing, or custom-loader policy.
"""

from __future__ import annotations

import inspect
import logging
import math
import types
from dataclasses import dataclass


LOG = logging.getLogger("comfyui-turing-utils")
ATTENTION_LAYOUT_KEY = "turing_utils_attention_layout"
ATTENTION_LAYOUT_REQUIREMENT_KEY = "turing_utils_attention_layout_required"
MINIMAX_H3_LAYOUT_KIND = "minimax_h3"
RUNTIME_CONTEXT_ATTR = "_turing_utils_minimax_runtime_context"
RUNTIME_PROVIDER_ATTR = "_turing_utils_minimax_layout_provider"
RUNTIME_OUTER_WRAPPER_KEY = "turing_utils_minimax_runtime_layout"
_FORWARD_PROVIDER_ATTR = "_turing_utils_minimax_layout_forward"
_BLOCK_FORWARD_PARAMETERS = (
    "x",
    "t_emb",
    "mod_segments",
    "rope_freqs",
    "transformer_options",
)
_LAYOUT_FIELDS = {
    "provider",
    "dense_prefix_tokens",
    "topology_start_tokens",
    "topology_tokens",
    "tokens_per_frame",
    "spatial_tokens_height",
    "spatial_tokens_width",
    "layer_index",
    "layer_count",
}


@dataclass(frozen=True)
class LayoutProviderStatus:
    model_kind: str | None
    installed: bool
    reason: str | None = None

    @property
    def required(self) -> bool:
        return self.model_kind is not None


def _root_and_diffusion_model(model):
    root = getattr(model, "model", model)
    return root, getattr(root, "diffusion_model", None)


def _is_minimax_h3_diffusion_model(diffusion_model) -> bool:
    if diffusion_model is None:
        return False
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
    except ImportError:
        return False
    return isinstance(diffusion_model, MiniMaxH3Model)


def _compatible_block_forward(forward) -> bool:
    try:
        parameters = tuple(inspect.signature(forward).parameters)
    except (TypeError, ValueError):
        return False
    if parameters and parameters[0] == "self":
        parameters = parameters[1:]
    return parameters == _BLOCK_FORWARD_PARAMETERS


def make_minimax_runtime_context_wrapper(base_model):
    """Expose the current sampler-owned nested shapes for packed H3 blocks."""

    def outer_sample_wrapper(executor, *args, **kwargs):
        latent_shapes = kwargs.get("latent_shapes")
        if latent_shapes is None and len(args) > 8:
            latent_shapes = args[8]
        previous = getattr(base_model, RUNTIME_CONTEXT_ATTR, None)
        setattr(base_model, RUNTIME_CONTEXT_ATTR, {"latent_shapes": latent_shapes})
        try:
            return executor(*args, **kwargs)
        finally:
            if previous is None:
                try:
                    delattr(base_model, RUNTIME_CONTEXT_ATTR)
                except AttributeError:
                    pass
            else:
                setattr(base_model, RUNTIME_CONTEXT_ATTR, previous)

    return outer_sample_wrapper


def _runtime_latent_shapes(base_model):
    context = getattr(base_model, RUNTIME_CONTEXT_ATTR, None)
    latent_shapes = context.get("latent_shapes") if isinstance(context, dict) else None
    if latent_shapes is None:
        # ComfyUI's sampler also publishes this directly on BaseModel.  Keeping
        # it as a fallback makes the provider tolerant of wrapper ordering.
        latent_shapes = getattr(base_model, "latent_shapes", None)
    return latent_shapes


def minimax_temporal_topology(base_model, diffusion_model, mod_segments):
    """Describe the contiguous target-video tail of the current H3 sequence."""
    if base_model is None or diffusion_model is None or not mod_segments:
        return {}
    latent_shapes = _runtime_latent_shapes(base_model)
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


def publish_minimax_attention_layout(
    transformer_options,
    mod_segments,
    *,
    layer_index: int,
    layer_count: int,
    base_model,
    diffusion_model,
) -> bool:
    """Publish exact H3 packed-sequence semantics for one transformer block."""
    if not isinstance(transformer_options, dict) or not mod_segments:
        return False
    expected_layout = {
        "provider": MINIMAX_H3_LAYOUT_KIND,
        # H3's current PackedLayout ends in target audio then target video.
        # Keeping the entire non-video prefix dense preserves text, references,
        # and joint audio/video generation.
        "dense_prefix_tokens": int(mod_segments[-1][0]),
        "layer_index": int(layer_index),
        "layer_count": int(layer_count),
        **minimax_temporal_topology(base_model, diffusion_model, mod_segments),
    }
    layout = transformer_options.get(ATTENTION_LAYOUT_KEY)
    # Never retain topology from an earlier prompt/window when current shapes
    # cannot be validated. Unknown extension fields may compose, but every
    # provider-owned field is rebuilt atomically for this exact block call.
    preserved = (
        {key: value for key, value in layout.items() if key not in _LAYOUT_FIELDS}
        if isinstance(layout, dict)
        else {}
    )
    updated = {**preserved, **expected_layout}
    if layout != updated:
        transformer_options[ATTENTION_LAYOUT_KEY] = updated
    return "topology_tokens" in expected_layout


def has_complete_minimax_attention_layout(
    transformer_options,
    sequence_length: int | None = None,
) -> bool:
    if not isinstance(transformer_options, dict):
        return False
    layout = transformer_options.get(ATTENTION_LAYOUT_KEY)
    if not isinstance(layout, dict) or layout.get("provider") != MINIMAX_H3_LAYOUT_KIND:
        return False
    names = (
        "dense_prefix_tokens",
        "topology_start_tokens",
        "topology_tokens",
        "tokens_per_frame",
        "spatial_tokens_height",
        "spatial_tokens_width",
        "layer_index",
        "layer_count",
    )
    values = {name: layout.get(name) for name in names}
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in values.values()
    ):
        return False
    prefix = values["dense_prefix_tokens"]
    start = values["topology_start_tokens"]
    tokens = values["topology_tokens"]
    frame_tokens = values["tokens_per_frame"]
    height = values["spatial_tokens_height"]
    width = values["spatial_tokens_width"]
    layer_index = values["layer_index"]
    layer_count = values["layer_count"]
    if (
        prefix < 0
        or start != prefix
        or tokens <= 0
        or frame_tokens <= 0
        or tokens % frame_tokens != 0
        or height <= 0
        or width <= 0
        or height * width != frame_tokens
        or layer_count <= 0
        or layer_index < 0
        or layer_index >= layer_count
    ):
        return False
    return sequence_length is None or start + tokens == int(sequence_length)


def _forward_has_provider(forward) -> bool:
    function = getattr(forward, "__func__", forward)
    return bool(getattr(function, _FORWARD_PROVIDER_ATTR, False))


def _make_layout_forward(
    block,
    original,
    *,
    layer_index: int,
    layer_count: int,
    base_model,
    diffusion_model,
):
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
        return original(
            x,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options=transformer_options,
        )

    setattr(forward, _FORWARD_PROVIDER_ATTR, True)
    return types.MethodType(forward, block)


def mark_forward_as_minimax_layout_provider(forward):
    """Mark a fused replacement that already calls publish_minimax_attention_layout."""
    function = getattr(forward, "__func__", forward)
    setattr(function, _FORWARD_PROVIDER_ATTR, True)
    return forward


def ensure_minimax_attention_layout_provider(model) -> LayoutProviderStatus:
    """Install the H3 layout bridge on any standard ComfyUI ModelPatcher."""
    base_model, diffusion_model = _root_and_diffusion_model(model)
    if not _is_minimax_h3_diffusion_model(diffusion_model):
        return LayoutProviderStatus(None, False, "not_minimax_h3")
    if not callable(getattr(model, "add_object_patch", None)) or not callable(
        getattr(model, "add_wrapper_with_key", None)
    ):
        return LayoutProviderStatus(
            MINIMAX_H3_LAYOUT_KIND, False, "model_patcher_api_unavailable"
        )
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        return LayoutProviderStatus(MINIMAX_H3_LAYOUT_KIND, False, "blocks_unavailable")
    if any(not _compatible_block_forward(block.forward) for block in blocks):
        return LayoutProviderStatus(
            MINIMAX_H3_LAYOUT_KIND, False, "block_forward_contract_changed"
        )

    try:
        import comfy.patcher_extension
    except ImportError:
        return LayoutProviderStatus(
            MINIMAX_H3_LAYOUT_KIND, False, "patcher_extension_unavailable"
        )

    model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        RUNTIME_OUTER_WRAPPER_KEY,
        make_minimax_runtime_context_wrapper(base_model),
    )
    object_patches = getattr(model, "object_patches", {})
    for layer_index, block in enumerate(blocks):
        key = f"diffusion_model.blocks.{layer_index}.forward"
        current = object_patches.get(key, block.forward)
        if _forward_has_provider(current):
            continue
        model.add_object_patch(
            key,
            _make_layout_forward(
                block,
                current,
                layer_index=layer_index,
                layer_count=len(blocks),
                base_model=base_model,
                diffusion_model=diffusion_model,
            ),
        )

    setattr(base_model, RUNTIME_PROVIDER_ATTR, True)
    LOG.info(
        "Enabled loader-independent MiniMax H3 attention layout provider on %d blocks",
        len(blocks),
    )
    return LayoutProviderStatus(MINIMAX_H3_LAYOUT_KIND, True)
