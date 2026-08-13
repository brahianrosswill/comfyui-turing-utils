"""Dedicated MiniMax H3 video VAE nodes."""

from __future__ import annotations

import comfy.model_management

from ..adapters.minimax.video_vae import (
    DECODER_TILING_MODES,
    DECODER_TILE_SIZES,
    TILES_PER_BATCH,
    decode_video,
    encode_video,
    require_h3_video_vae,
)


class MiniMaxH3VideoVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tiles_per_batch": (
                    TILES_PER_BATCH,
                    {
                        "default": "auto",
                        "tooltip": "Number of independent 256px tiles evaluated together on the batch dimension. auto chooses up to 4 tiles within ComfyUI's current memory budget.",
                    },
                ),
                "decoder_tile_size": (
                    DECODER_TILE_SIZES,
                    {
                        "default": "256",
                        "tooltip": "256 is the quality-stable H3 tile geometry. Larger values are explicit experiments that preserve 256px spatial RoPE spacing but still change transformer context and can change the image.",
                    },
                ),
                "attention": (
                    ["sdpa", "sage", "w8a8"],
                    {
                        "default": "sdpa",
                        "tooltip": "Decoder attention only. On Turing, BF16 SDPA inputs are consumed through containers and computed as FP16 to avoid the slow math fallback. W8A8 refers to QK attention, not VAE weight quantization.",
                    },
                ),
                "decoder_tiling": (
                    DECODER_TILING_MODES,
                    {
                        "default": "official",
                        "tooltip": "official keeps independent 256px windows. shared_overlap is experimental: it shares image-token QKV/MLP work across overlapping windows while preserving each window's local 256px attention and RoPE.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "Turing Utils/MiniMax H3"
    DESCRIPTION = (
        "Experimental MiniMax H3 video decoder with automatic W8A8 SwiGLU "
        "fusion, FP32 pixel double buffering, and dynamic weight retention "
        "across all tiles in one invocation."
    )

    def decode(
        self,
        samples,
        vae,
        tiles_per_batch,
        decoder_tile_size,
        attention,
        decoder_tiling="official",
    ):
        require_h3_video_vae(vae)
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        with comfy.model_management.cuda_device_context(vae.device):
            images = decode_video(
                vae,
                latent,
                tiles_per_batch,
                attention,
                decoder_tile_size,
                decoder_tiling,
            )
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        return (images,)


class MiniMaxH3VideoVAEEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
                "tiles_per_batch": (
                    TILES_PER_BATCH,
                    {
                        "default": "auto",
                        "tooltip": "Number of independent official 256px tiles evaluated together on the batch dimension. auto chooses up to 2 tiles within ComfyUI's current memory budget.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "Turing Utils/MiniMax H3"
    DESCRIPTION = (
        "Experimental MiniMax H3 video encoder with FP32 pixel double "
        "buffering and dynamic weight retention across all tiles. Output "
        "latents always use ComfyUI-compatible FP32 storage."
    )

    def encode(
        self,
        pixels,
        vae,
        tiles_per_batch,
    ):
        require_h3_video_vae(vae)
        with comfy.model_management.cuda_device_context(vae.device):
            latent = encode_video(
                vae,
                pixels,
                tiles_per_batch,
            )
        return ({"samples": latent},)
