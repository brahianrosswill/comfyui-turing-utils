"""Dedicated MiniMax H3 video VAE nodes."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.nested_tensor

from ..adapters.minimax.video_vae import (
    decode_video,
    encode_video,
    require_h3_video_vae,
    upscale_latent_via_pixels,
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
        "Experimental MiniMax H3 video decoder with automatic W8A8 SwiGLU "
        "fusion, FP32 pixel double buffering, and dynamic weight retention "
        "across all tiles in one invocation."
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
        "Experimental MiniMax H3 video encoder with FP32 pixel double "
        "buffering, FP16 round-trip fast path, automatic tile batching, and "
        "dynamic weight retention across all tiles. Output latents always use "
        "ComfyUI-compatible FP32 storage."
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


def _resize_noise_mask(mask, latent):
    if mask is None or tuple(mask.shape[-2:]) == tuple(latent.shape[-2:]):
        return mask
    original_shape = mask.shape
    resized = F.interpolate(
        mask.reshape(-1, 1, *original_shape[-2:]).float(),
        size=latent.shape[-2:],
        mode="nearest-exact",
    )
    return resized.reshape(*original_shape[:-2], *latent.shape[-2:]).to(
        device=mask.device,
        dtype=mask.dtype,
    )


class MiniMaxH3LatentPixelUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "width": (
                    "INT",
                    {"default": 1280, "min": 32, "max": 8192, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 736, "min": 32, "max": 8192, "step": 32},
                ),
                "upscale_method": (
                    ["bicubic", "bilinear", "nearest-exact", "rtx_vsr"],
                    {
                        "default": "bicubic",
                        "tooltip": "Resize decoded RGB frames on the GPU. RTX VSR is optional and requires NVIDIA's nvidia-vfx package.",
                    },
                ),
                "rtx_vsr_quality": (
                    [
                        "high",
                        "ultra",
                        "medium",
                        "high_bitrate_high",
                        "high_bitrate_ultra",
                    ],
                    {
                        "default": "high",
                        "tooltip": "Used only when upscale_method is rtx_vsr. Standard modes also suppress compression-like noise; high-bitrate modes preserve more source texture.",
                    },
                ),
                "attention": (
                    ["sdpa", "sage", "w8a8"],
                    {
                        "default": "sdpa",
                        "tooltip": "Attention backend for the decode half of the pixel round trip.",
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
                    },
                ),
                "final_full_overlap_blocks": (
                    "INT",
                    {
                        "default": 36,
                        "min": 0,
                        "max": 36,
                        "step": 1,
                        "tooltip": "Final decoder Transformer blocks that retain all overlapping window contributions.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "Turing Utils/MiniMax H3"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Stream finalized MiniMax H3 decoder chunks through complete-frame GPU "
        "resize directly into an FP16 target store, then re-encode it while "
        "retaining VAE runtime state. Native H3 audio latents pass through unchanged."
    )

    def upscale(
        self,
        samples,
        vae,
        width,
        height,
        upscale_method,
        rtx_vsr_quality,
        attention,
        overlap_query_threshold=0.0,
        final_full_overlap_blocks=36,
    ):
        require_h3_video_vae(vae)
        if not isinstance(samples, dict) or "samples" not in samples:
            raise ValueError("samples must be a LATENT dictionary")
        container = samples["samples"]
        audio = None
        if getattr(container, "is_nested", False):
            streams = list(container.unbind())
            if len(streams) != 2:
                raise ValueError(
                    f"Expected exactly two H3 video/audio latent streams, got {len(streams)}"
                )
            video, audio = streams
        else:
            video = container
        if not torch.is_tensor(video) or video.ndim != 5 or video.shape[1] != 24:
            shape = tuple(video.shape) if hasattr(video, "shape") else type(video).__name__
            raise ValueError(f"Expected H3 video latent [B,24,T,H,W], got {shape}")

        with comfy.model_management.cuda_device_context(vae.device):
            resized_video = upscale_latent_via_pixels(
                vae,
                video,
                width,
                height,
                upscale_method,
                rtx_vsr_quality,
                attention,
                overlap_query_threshold,
                final_full_overlap_blocks,
            )

        output = samples.copy()
        output["samples"] = (
            comfy.nested_tensor.NestedTensor((resized_video, audio))
            if audio is not None
            else resized_video
        )
        noise_mask = samples.get("noise_mask")
        if noise_mask is not None:
            if audio is not None:
                if not getattr(noise_mask, "is_nested", False):
                    raise ValueError(
                        "An H3 AV latent noise_mask must contain video and audio streams"
                    )
                mask_streams = list(noise_mask.unbind())
                if len(mask_streams) != 2:
                    raise ValueError(
                        f"Expected exactly two H3 noise-mask streams, got {len(mask_streams)}"
                    )
                output["noise_mask"] = comfy.nested_tensor.NestedTensor(
                    (_resize_noise_mask(mask_streams[0], resized_video), mask_streams[1])
                )
            else:
                if getattr(noise_mask, "is_nested", False):
                    raise ValueError(
                        "A standalone H3 video latent cannot use a nested noise_mask"
                    )
                output["noise_mask"] = _resize_noise_mask(noise_mask, resized_video)
        return (output,)
