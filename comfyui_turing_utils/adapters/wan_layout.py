"""Semantic self-attention layout publication for Wan and Bernini models.

The provider describes only the flattened self-attention sequence.  It does
not depend on quantization or a Turing device, so sparse attention can use it
with an official ComfyUI loader as well as with this plugin's Wan adapter.
"""

from __future__ import annotations

import inspect
import logging
import math

import torch

from .methods import OriginalMethod, weak_method
from ..attention.layout import (
    ATTENTION_LAYOUT_KEY,
    AttentionSegment,
    AttentionSemanticLayout,
    AttentionTopology,
    LayoutProviderStatus,
)


LOG = logging.getLogger("comfyui-turing-utils")
WAN_LAYOUT_KIND = "wan_self_attention"
_FORWARD_ORIG_PATCH_KEY = "diffusion_model.forward_orig"
_FORWARD_PROVIDER_ATTR = "_turing_utils_wan_layout_forward"
_FORWARD_ORIG_PARAMETERS = (
    "x",
    "t",
    "context",
    "clip_fea",
    "freqs",
    "transformer_options",
)
_LAYOUT_FIELDS = {
    "protocol_version",
    "provider",
    "query_segments",
    "key_segments",
    "topologies",
    "dense_prefix_tokens",
    "topology_start_tokens",
    "topology_tokens",
    "tokens_per_frame",
    "spatial_tokens_height",
    "spatial_tokens_width",
    "segments",
    "layer_index",
    "layer_count",
}


def _root_and_diffusion_model(model):
    root = getattr(model, "model", model)
    return root, getattr(root, "diffusion_model", None)


def _is_supported_wan_model(diffusion_model) -> bool:
    if diffusion_model is None:
        return False
    try:
        from comfy.ldm.wan.model import WanModel
    except ImportError:
        return False
    # Several Wan subclasses use a materially different token order.  Inheriting
    # the base implementation is the compatibility boundary used by Bernini.
    return isinstance(diffusion_model, WanModel) and (
        type(diffusion_model).forward_orig is WanModel.forward_orig
    )


def _compatible_forward_orig(forward) -> bool:
    try:
        parameters = tuple(inspect.signature(forward).parameters)
    except (TypeError, ValueError):
        return False
    if parameters and parameters[0] == "self":
        parameters = parameters[1:]
    return parameters[: len(_FORWARD_ORIG_PARAMETERS)] == _FORWARD_ORIG_PARAMETERS


def _grid_from_video(latent, patch_size) -> tuple[int, int, int] | None:
    if not torch.is_tensor(latent) or latent.ndim != 5:
        return None
    sizes = tuple(int(value) for value in latent.shape[-3:])
    patches = tuple(int(value) for value in patch_size)
    if len(patches) != 3 or any(value <= 0 for value in patches):
        return None
    return tuple(math.ceil(size / patch) for size, patch in zip(sizes, patches))


def _image_tokens(latent, patch_size) -> int | None:
    if not torch.is_tensor(latent) or latent.ndim != 4:
        return None
    height, width = (int(value) for value in latent.shape[-2:])
    patch_height, patch_width = (int(value) for value in patch_size[-2:])
    if min(height, width, patch_height, patch_width) <= 0:
        return None
    tokens_height = height // patch_height
    tokens_width = width // patch_width
    return tokens_height * tokens_width if min(tokens_height, tokens_width) > 0 else None


def _append_video_reference(
    segments: list[AttentionSegment],
    topologies: list[AttentionTopology],
    *,
    start: int,
    grid: tuple[int, int, int],
    topology_id: str,
) -> int:
    frames, height, width = grid
    frame_tokens = height * width
    stop = start + frames * frame_tokens
    topologies.append(
        AttentionTopology(
            topology_id,
            "video_grid",
            start,
            stop,
            frame_tokens,
            height,
            width,
        )
    )
    first_stop = start + frame_tokens
    segments.append(
        AttentionSegment.for_role(
            start,
            first_stop,
            "reference_video_anchor",
            topology_id=topology_id,
        )
    )
    if frames > 2:
        segments.append(
            AttentionSegment.for_role(
                first_stop,
                stop - frame_tokens,
                "reference_video",
                topology_id=topology_id,
            )
        )
    if frames > 1:
        segments.append(
            AttentionSegment.for_role(
                stop - frame_tokens,
                stop,
                "reference_video_anchor",
                topology_id=topology_id,
            )
        )
    return stop


def build_wan_attention_layout(
    diffusion_model,
    x,
    transformer_options,
    kwargs,
) -> AttentionSemanticLayout | None:
    """Build the exact token order used by ``WanModel.forward_orig``."""
    patch_size = tuple(int(value) for value in diffusion_model.patch_size)
    target_grid = _grid_from_video(x, patch_size)
    if target_grid is None:
        return None

    segments: list[AttentionSegment] = []
    topologies: list[AttentionTopology] = []
    cursor = 0

    reference = kwargs.get("reference_latent")
    if getattr(diffusion_model, "ref_conv", None) is not None and reference is not None:
        count = _image_tokens(reference, patch_size)
        if count is None:
            return None
        segments.append(
            AttentionSegment.for_role(cursor, cursor + count, "reference_image")
        )
        cursor += count

    frames, height, width = target_grid
    target_tokens = frames * height * width
    target_start = cursor
    cursor += target_tokens
    segments.append(
        AttentionSegment.for_role(
            target_start,
            cursor,
            "target_video",
            topology_id="target_video",
        )
    )
    topologies.append(
        AttentionTopology(
            "target_video",
            "video_grid",
            target_start,
            cursor,
            height * width,
            height,
            width,
        )
    )

    context_latents = kwargs.get("context_latents")
    if context_latents is not None:
        if not isinstance(context_latents, (tuple, list)):
            return None
        for index, latent in enumerate(context_latents):
            grid = _grid_from_video(latent, patch_size)
            if grid is None:
                return None
            if grid[0] == 1:
                count = grid[1] * grid[2]
                segments.append(
                    AttentionSegment.for_role(
                        cursor,
                        cursor + count,
                        "reference_image",
                    )
                )
                cursor += count
            else:
                cursor = _append_video_reference(
                    segments,
                    topologies,
                    start=cursor,
                    grid=grid,
                    topology_id=f"context_video_{index}",
                )

    layer_count = len(getattr(diffusion_model, "blocks", ()))
    if layer_count <= 0:
        return None
    layout = AttentionSemanticLayout(
        provider=WAN_LAYOUT_KIND,
        query_segments=tuple(segments),
        key_segments=tuple(segments),
        topologies=tuple(topologies),
        layer_index=0,
        layer_count=layer_count,
    )
    return layout if layout.validate(cursor, cursor) is None else None


def publish_wan_attention_layout(
    diffusion_model,
    x,
    transformer_options,
    kwargs,
) -> bool:
    if not isinstance(transformer_options, dict):
        return False
    semantic = build_wan_attention_layout(
        diffusion_model,
        x,
        transformer_options,
        kwargs,
    )
    previous = transformer_options.get(ATTENTION_LAYOUT_KEY)
    preserved = (
        {key: value for key, value in previous.items() if key not in _LAYOUT_FIELDS}
        if isinstance(previous, dict)
        else {}
    )
    if semantic is None:
        # An incomplete sequence description must never inherit a valid layout
        # from an earlier prompt or context window.
        transformer_options[ATTENTION_LAYOUT_KEY] = {
            **preserved,
            "provider": WAN_LAYOUT_KIND,
        }
        return False
    transformer_options[ATTENTION_LAYOUT_KEY] = semantic.to_wire(
        extensions=preserved
    )
    return True


def _forward_has_provider(forward) -> bool:
    function = getattr(forward, "__func__", forward)
    return bool(getattr(function, _FORWARD_PROVIDER_ATTR, False))


def _make_layout_forward(diffusion_model, original):
    original = OriginalMethod.capture(original, diffusion_model)

    def forward_orig(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        publish_wan_attention_layout(
            self,
            x,
            transformer_options,
            kwargs,
        )
        return original(
            self,
            x,
            t,
            context,
            clip_fea=clip_fea,
            freqs=freqs,
            transformer_options=transformer_options,
            **kwargs,
        )

    setattr(forward_orig, _FORWARD_PROVIDER_ATTR, True)
    return weak_method(forward_orig, diffusion_model)


def ensure_wan_attention_layout_provider(model) -> LayoutProviderStatus:
    """Install a loader-independent layout provider for base Wan/Bernini."""
    _, diffusion_model = _root_and_diffusion_model(model)
    if not _is_supported_wan_model(diffusion_model):
        return LayoutProviderStatus(None, False, "not_compatible_wan")
    if not callable(getattr(model, "add_object_patch", None)):
        return LayoutProviderStatus(
            WAN_LAYOUT_KIND,
            False,
            "model_patcher_api_unavailable",
        )
    if not _compatible_forward_orig(diffusion_model.forward_orig):
        return LayoutProviderStatus(
            WAN_LAYOUT_KIND,
            False,
            "forward_orig_contract_changed",
        )
    object_patches = getattr(model, "object_patches", {})
    current = object_patches.get(
        _FORWARD_ORIG_PATCH_KEY,
        diffusion_model.forward_orig,
    )
    if not _forward_has_provider(current):
        model.add_object_patch(
            _FORWARD_ORIG_PATCH_KEY,
            _make_layout_forward(diffusion_model, current),
        )
    LOG.info("Enabled loader-independent Wan/Bernini attention layout provider")
    return LayoutProviderStatus(WAN_LAYOUT_KIND, True)


__all__ = [
    "WAN_LAYOUT_KIND",
    "build_wan_attention_layout",
    "ensure_wan_attention_layout_provider",
    "publish_wan_attention_layout",
]
