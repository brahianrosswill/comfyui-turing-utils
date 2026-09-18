"""H3 encode entry point; ComfyUI owns the complete VAE execution lifecycle."""

from __future__ import annotations

from ...log import get_logger
from .video_vae import _tile_progress, _vae_operator_scope, require_h3_video_vae


LOG = get_logger("minimax.vae")


def encode_video(vae, pixels):
    model = require_h3_video_vae(vae)
    LOG.info(
        "H3 VAE encode: native ComfyUI lifecycle; "
        "tile progress counts submitted forwards"
    )
    with (
        _vae_operator_scope(),
        _tile_progress(model.quant_conv, "H3 VAE Encode Tiles (submitted)"),
    ):
        # Keep input cropping, compute/output dtype, transfers, tile geometry,
        # weight residency and OOM recovery exactly as the official VAE.
        return vae.encode(pixels)


__all__ = ["encode_video"]
