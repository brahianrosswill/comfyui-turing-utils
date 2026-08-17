"""Dedicated MiniMax H3 video VAE nodes."""

from __future__ import annotations

import comfy.model_management

from ..adapters.minimax.video_vae import (
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
                "attention": (
                    ["sdpa", "sage", "w8a8"],
                    {
                        "default": "sdpa",
                        "tooltip": "Decoder attention only. On Turing, BF16 SDPA inputs are consumed through containers and computed as FP16 to avoid the slow math fallback. W8A8 refers to QK attention, not VAE weight quantization.",
                    },
                ),
            },
            "optional": {
                "overlap_query_threshold": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Experimental speed/quality control. Window-query memberships below this normalized cosine weight are skipped and the survivors are renormalized. Zero preserves the stable full-overlap path.",
                    },
                ),
                "final_full_overlap_blocks": (
                    "INT",
                    {
                        "default": 36,
                        "min": 0,
                        "max": 36,
                        "step": 1,
                        "tooltip": "Number of final decoder Transformer blocks that always keep every overlapping window contribution. Earlier blocks may prune low-weight overlap queries using overlap_query_threshold.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "Turing Utils/MiniMax H3"
    DESCRIPTION = (
        "MiniMax H3 video decoder with automatic W8A8 SwiGLU "
        "fusion, asynchronous pixel double buffering, ComfyUI-managed "
        "block-level dynamic-weight prefetch, and output storage matching "
        "ComfyUI's VAE intermediate dtype."
    )

    def decode(
        self,
        samples,
        vae,
        attention,
        overlap_query_threshold=0.0,
        final_full_overlap_blocks=36,
    ):
        require_h3_video_vae(vae)
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        with comfy.model_management.cuda_device_context(vae.device):
            images = decode_video(
                vae,
                latent,
                attention,
                overlap_query_threshold=overlap_query_threshold,
                final_full_overlap_blocks=final_full_overlap_blocks,
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
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "Turing Utils/MiniMax H3"
    DESCRIPTION = (
        "MiniMax H3 video encoder with asynchronous pixel buffering, automatic "
        "tile batching, ComfyUI-managed block-level dynamic-weight prefetch, "
        "and output storage matching ComfyUI's VAE intermediate dtype."
    )

    def encode(
        self,
        pixels,
        vae,
    ):
        require_h3_video_vae(vae)
        with comfy.model_management.cuda_device_context(vae.device):
            latent = encode_video(
                vae,
                pixels,
            )
        return ({"samples": latent},)
