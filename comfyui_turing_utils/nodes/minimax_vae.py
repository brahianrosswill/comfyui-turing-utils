"""Dedicated MiniMax H3 video VAE nodes."""

from __future__ import annotations

import comfy.model_management

from ..adapters.minimax.video_vae import (
    TILE_PRESETS,
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
                "tile_preset": (
                    TILE_PRESETS,
                    {
                        "default": "auto",
                        "tooltip": "Internal spatial tile edge in output pixels. auto chooses an aligned 256-480 tile for the current frame size and ComfyUI's current memory budget; overlap is fixed at 64 pixels.",
                    },
                ),
                "attention": (
                    ["sdpa", "sage", "w8a8"],
                    {
                        "default": "sdpa",
                        "tooltip": "Decoder attention only. On Turing, BF16 SDPA inputs are consumed through containers and computed as FP16 to avoid the slow math fallback. W8A8 refers to QK attention, not VAE weight quantization.",
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
        tile_preset,
        attention,
    ):
        require_h3_video_vae(vae)
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        with comfy.model_management.cuda_device_context(vae.device):
            images = decode_video(
                vae,
                latent,
                tile_preset,
                attention,
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
                "tile_preset": (
                    TILE_PRESETS,
                    {
                        "default": "auto",
                        "tooltip": "Internal spatial tile edge in input pixels. auto chooses an aligned 256-480 tile for the current frame size and ComfyUI's current memory budget; overlap is fixed at 64 pixels.",
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
        tile_preset,
    ):
        require_h3_video_vae(vae)
        with comfy.model_management.cuda_device_context(vae.device):
            latent = encode_video(
                vae,
                pixels,
                tile_preset,
            )
        return ({"samples": latent},)
