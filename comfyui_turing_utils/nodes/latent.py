"""Video latent masks and compositing with explicit temporal mapping."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from comfy_api.latest import io


# Image frames per latent after the first frame; H3 uses a repeating block instead.
VIDEO_MASK_SPECS = {
    "wan": 4,
    "minimax": None,
    "ltxv": 8,
    "hunyuan_video": 4,
    "hunyuan_video_15": 4,
    "mochi": 6,
}


def _image_frame_groups(model_type: str, latent_frames: int) -> tuple[int, ...]:
    temporal_stride = VIDEO_MASK_SPECS[model_type]
    if temporal_stride is not None:
        return (1,) + (temporal_stride,) * (latent_frames - 1)
    if latent_frames == 1:
        return (1,)
    if latent_frames < 2 or (latent_frames - 2) % 5:
        raise ValueError(
            f"MiniMax H3 image-frame mapping requires 1 or 5*n+2 latent time positions; "
            f"got {latent_frames}. Supply exactly {latent_frames} masks for direct mapping."
        )
    return (1, 4, 4, 4, 4) * ((latent_frames - 2) // 5) + (1, 4)


def _video_samples(samples):
    video = samples["samples"]
    if getattr(video, "is_nested", False):
        raise ValueError(
            "Video mask nodes accept standalone video only. "
            "Use a Separate AV Latent node before this node, then concatenate afterward."
        )
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise ValueError("Expected a video latent shaped [B,C,T,H,W], not an image or audio latent.")
    if any(size < 1 for size in video.shape):
        raise ValueError("Video latent dimensions must be non-empty.")
    return video


def _validate_mask_values(mask):
    if mask.is_complex() or any(size < 1 for size in mask.shape):
        raise ValueError("MASK must contain non-empty, real-valued frames.")
    if not torch.isfinite(mask).all() or mask.amin() < 0 or mask.amax() > 1:
        raise ValueError("MASK values must be finite and within [0,1]; 0 preserves and 1 redraws.")


def _map_video_mask(video, mask, model_type, *, hard=False):
    if model_type not in VIDEO_MASK_SPECS:
        raise ValueError(f"Unknown video mask type: {model_type!r}.")
    if not isinstance(mask, torch.Tensor) or getattr(mask, "is_nested", False) or mask.ndim not in (3, 4):
        raise ValueError("Expected MASK shaped [frames,H,W] or [B,frames,H,W].")
    _validate_mask_values(mask)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)

    batch, _, latent_frames, height, width = video.shape
    mask_batch, mask_frames, mask_height, mask_width = mask.shape
    if mask_batch not in (1, batch):
        raise ValueError(f"MASK batch must be 1 or match video batch {batch}; got {mask_batch}.")

    groups = None
    if mask_frames != latent_frames:
        groups = _image_frame_groups(model_type, latent_frames)
        expected_frames = sum(groups)
        if mask_frames != expected_frames:
            raise ValueError(
                f"Received {mask_frames} mask frames for {model_type} latent T={latent_frames}. "
                f"Expected {latent_frames} latent-frame masks or {expected_frames} image-frame masks. "
                "No temporal interpolation, padding, or truncation is performed."
            )

    # Hard coverage is decided before float32 conversion, including tiny positive inputs.
    mask = (mask > 0).float() if hard else mask.float()
    # Spatial and temporal maxima commute; shrink first to avoid large temporal intermediates.
    if (mask_height, mask_width) != (height, width):
        mask = F.adaptive_max_pool2d(
            mask.reshape(-1, 1, mask_height, mask_width), (height, width)
        ).reshape(mask_batch, mask_frames, height, width)
    if groups is not None:
        mask = torch.stack([part.amax(dim=1) for part in mask.split(groups, dim=1)], dim=1)
    return mask.unsqueeze(1).expand(batch, 1, latent_frames, height, width).clone()


class SetVideoLatentNoiseMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsSetVideoLatentNoiseMask",
            display_name="Set Video Latent Noise Mask",
            category="Turing Utils/video",
            description=(
                "Set a video-only noise mask. Matching mask/latent time counts map directly; "
                "otherwise merge image-frame masks by the selected VAE's temporal groups. "
                "Uses maximum/union in time and space. Mismatched counts are errors, "
                "never temporally interpolated, padded, or truncated."
            ),
            inputs=[
                io.Latent.Input("samples", tooltip="Standalone video latent [B,C,T,H,W]. Separate audio/video latents first."),
                io.Mask.Input("mask", tooltip="[frames,H,W], or [B,frames,H,W] for separate video batches. A single mask sequence is shared across the video batch. 0 preserves, 1 redraws."),
                io.Combo.Input("type", options=list(VIDEO_MASK_SPECS), default="minimax", tooltip=(
                    "Used only when frame counts differ. wan (2.1/2.2), hunyuan_video and "
                    "hunyuan_video_15: first frame then groups of 4; ltxv (LTX-Video/LTX-2 video): "
                    "groups of 8; mochi: groups of 6. minimax (H3 video): [1,4,4,4,4]*n+[1,4]."
                )),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, samples, mask, type):
        video = _video_samples(samples)
        output = samples.copy()
        output["noise_mask"] = _map_video_mask(video, mask, type)
        return io.NodeOutput(output)


class VideoLatentCompositeMasked(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsVideoLatentCompositeMasked",
            display_name="Video Latent Composite Masked",
            category="Turing Utils/video",
            description=(
                "Composite matching standalone video latents using hard coverage. Union the replacement "
                "latent's inherited noise_mask with the optional mask; every nonzero value means replace, "
                "not opacity. The same binary region is saved as noise_mask. With neither mask, replace all. "
                "The original latent's mask is ignored. No resizing or feathering of latent samples."
            ),
            inputs=[
                io.Latent.Input("original_latent", tooltip="Clean original high-resolution video [B,C,T,H,W]. Mask=0 preserves these samples. Its metadata is retained, but its noise_mask is replaced."),
                io.Latent.Input("replacement_latent", tooltip="Clean replacement video, e.g. upscaled first-pass denoised_output. Shape must match original_latent. Its noise_mask is inherited; mask=1 uses these samples."),
                io.Mask.Input("mask", optional=True, tooltip="Additional [frames,H,W] or [B,frames,H,W] mask. Its nonzero coverage is unioned with the inherited mask; even small positive values mean full replacement."),
                io.Combo.Input("type", options=list(VIDEO_MASK_SPECS), default="minimax", tooltip="Same temporal mapping as Set Video Latent Noise Mask. Only used to map additional image-frame masks, never to retime inherited latent masks."),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, original_latent, replacement_latent, type="minimax", mask=None):
        original = _video_samples(original_latent)
        replacement = _video_samples(replacement_latent)
        if original.shape != replacement.shape:
            raise ValueError(
                "original_latent and replacement_latent must have identical [B,C,T,H,W] shapes; "
                f"got {tuple(original.shape)} and {tuple(replacement.shape)}. Align them before compositing."
            )
        if type not in VIDEO_MASK_SPECS:
            raise ValueError(f"Unknown video mask type: {type!r}.")

        batch, channels, frames, height, width = original.shape
        coverage = None
        inherited = replacement_latent.get("noise_mask")
        if inherited is not None:
            if not isinstance(inherited, torch.Tensor) or getattr(inherited, "is_nested", False) or inherited.ndim != 5:
                raise ValueError("replacement_latent noise_mask must be standalone [B,1 or C,T,H,W]. Use Set Video Latent Noise Mask first.")
            _validate_mask_values(inherited)
            if inherited.shape[0] not in (1, batch) or inherited.shape[1] not in (1, channels) or inherited.shape[2:] != original.shape[2:]:
                raise ValueError("Inherited noise_mask must match replacement_latent T/H/W, with batch 1 or B and channels 1 or C.")
            coverage = (inherited.amax(dim=1, keepdim=True) > 0).to(original.device)
        if mask is not None:
            additional = _map_video_mask(original, mask, type, hard=True).to(device=original.device, dtype=torch.bool)
            coverage = additional if coverage is None else coverage | additional
        if coverage is None:
            coverage = torch.ones((batch, 1, frames, height, width), device=original.device, dtype=torch.bool)

        output = original_latent.copy()
        output["samples"] = torch.where(coverage, replacement.to(original), original)
        output["noise_mask"] = coverage.expand(batch, 1, frames, height, width).to(torch.float32)
        return io.NodeOutput(output)
