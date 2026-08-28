"""MiniMax H3 native-frame image attention with static Sol residual routing."""

from __future__ import annotations

from ...attention.orchestration import install_attention_strategy
from ...attention.patches import attention_base_runtime, make_sparse_attention_override
from ...attention.stable import LOG
from .layout import (
    H3_IMAGE_SOL_LAYOUT_KEY,
    H3_IMAGE_SOL_LAYOUTS,
    H3_IMAGE_SOL_STRATEGY,
    MINIMAX_H3_LAYOUT_KIND,
    is_minimax_h3_model,
)


H3_IMAGE_SOL_ROUTING_THRESHOLD = 1_000_000.0
H3_IMAGE_SOL_SKIPPED_RESIDUAL = "1x64"


def apply_h3_image_sol_attention(
    model,
    *,
    temporal_layout: str = "dense_anchor_grid",
    sparse_reference_image: bool = False,
    sparse_reference_video: bool = True,
    sparse_reference_audio: bool = False,
    dense_prefix_steps: int = 1,
    dense_suffix_steps: int = 0,
    dense_prefix_layers: int = 2,
    dense_suffix_layers: int = 0,
    debug_route_density: bool = False,
):
    temporal_layout = str(temporal_layout).strip().lower()
    if temporal_layout not in H3_IMAGE_SOL_LAYOUTS:
        raise ValueError(
            "temporal_layout must be dense_window or dense_anchor_grid"
        )
    if not is_minimax_h3_model(model):
        raise ValueError("Configure H3 Image Sol Attention requires MiniMax H3")
    runtime = attention_base_runtime(model, use_w8a8=None)
    override = make_sparse_attention_override(
        model.load_device,
        routing_threshold=H3_IMAGE_SOL_ROUTING_THRESHOLD,
        prefix_policy="auto",
        manual_prefix_tokens=0,
        skipped_residual=H3_IMAGE_SOL_SKIPPED_RESIDUAL,
        sparse_reference_image=bool(sparse_reference_image),
        sparse_reference_video=bool(sparse_reference_video),
        sparse_reference_audio=bool(sparse_reference_audio),
        dense_prefix_steps=int(dense_prefix_steps),
        dense_suffix_steps=int(dense_suffix_steps),
        dense_prefix_layers=int(dense_prefix_layers),
        dense_suffix_layers=int(dense_suffix_layers),
        debug_route_density=bool(debug_route_density),
        use_w8a8=None,
        dense_backend=runtime.dense_backend,
        dense_override=runtime.dense_override,
    )
    installed = install_attention_strategy(
        model,
        override,
        strategy="H3 image Sol",
        backend=H3_IMAGE_SOL_STRATEGY,
        implementation=f"h3_image_sol:{temporal_layout}",
        runtime_config=runtime,
    )
    if installed.layout.model_kind != MINIMAX_H3_LAYOUT_KIND:
        raise RuntimeError("MiniMax H3 attention layout provider was not selected")
    if not installed.layout.installed:
        raise RuntimeError(
            "Configure H3 Image Sol Attention requires the MiniMax H3 "
            f"attention layout provider: {installed.layout.reason}"
        )
    transformer_options = installed.model.model_options.setdefault(
        "transformer_options", {}
    )
    transformer_options[H3_IMAGE_SOL_LAYOUT_KEY] = temporal_layout
    LOG.info(
        "H3 image Sol enabled: temporal_layout=%s residual=1x64 "
        "sparse_reference=(image=%s,video=%s,audio=%s) "
        "dense_prefix_steps=%d dense_suffix_steps=%d "
        "dense_prefix_layers=%d dense_suffix_layers=%d dense_backend=%s",
        temporal_layout,
        bool(sparse_reference_image),
        bool(sparse_reference_video),
        bool(sparse_reference_audio),
        int(dense_prefix_steps),
        int(dense_suffix_steps),
        int(dense_prefix_layers),
        int(dense_suffix_layers),
        runtime.dense_backend,
    )
    return installed.model


__all__ = [
    "H3_IMAGE_SOL_ROUTING_THRESHOLD",
    "H3_IMAGE_SOL_SKIPPED_RESIDUAL",
    "apply_h3_image_sol_attention",
]
