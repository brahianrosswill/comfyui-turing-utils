"""Video noise masks with explicit image-to-latent temporal mapping."""

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
        video = samples["samples"]
        if getattr(video, "is_nested", False):
            raise ValueError(
                "Set Video Latent Noise Mask accepts standalone video only. "
                "Use a Separate AV Latent node before this node, then concatenate afterward."
            )
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError("Expected a video latent shaped [B,C,T,H,W], not an image or audio latent.")
        if any(size < 1 for size in video.shape):
            raise ValueError("Video latent dimensions must be non-empty.")
        if type not in VIDEO_MASK_SPECS:
            raise ValueError(f"Unknown video mask type: {type!r}.")
        if not isinstance(mask, torch.Tensor) or mask.ndim not in (3, 4):
            raise ValueError("Expected MASK shaped [frames,H,W] or [B,frames,H,W].")
        if mask.is_complex() or any(size < 1 for size in mask.shape):
            raise ValueError("MASK must contain non-empty, real-valued frames.")
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)

        batch, _, latent_frames, height, width = video.shape
        mask_batch, mask_frames, mask_height, mask_width = mask.shape
        if mask_batch not in (1, batch):
            raise ValueError(f"MASK batch must be 1 or match video batch {batch}; got {mask_batch}.")

        groups = None
        if mask_frames != latent_frames:
            groups = _image_frame_groups(type, latent_frames)
            expected_frames = sum(groups)
            if mask_frames != expected_frames:
                raise ValueError(
                    f"Received {mask_frames} mask frames for {type} latent T={latent_frames}. "
                    f"Expected {latent_frames} latent-frame masks or {expected_frames} image-frame masks. "
                    "No temporal interpolation, padding, or truncation is performed."
                )

        mask = mask.float()
        if not torch.isfinite(mask).all() or mask.amin() < 0 or mask.amax() > 1:
            raise ValueError("MASK values must be finite and within [0,1]; 0 preserves and 1 redraws.")

        # Spatial and temporal maxima commute; shrink first to avoid large temporal intermediates.
        if (mask_height, mask_width) != (height, width):
            mask = F.adaptive_max_pool2d(
                mask.reshape(-1, 1, mask_height, mask_width), (height, width)
            ).reshape(mask_batch, mask_frames, height, width)
        if groups is not None:
            mask = torch.stack([part.amax(dim=1) for part in mask.split(groups, dim=1)], dim=1)

        output = samples.copy()
        output["noise_mask"] = mask.unsqueeze(1).expand(batch, 1, latent_frames, height, width).clone()
        return io.NodeOutput(output)
