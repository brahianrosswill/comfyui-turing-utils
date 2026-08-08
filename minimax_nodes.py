from __future__ import annotations

from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

try:
    from .minimax_adapter import apply_h3_progressive_resolution_patch
    from .reference_nodes import (
        AudioReferenceSet,
        AudioReferences,
        ImageReferenceSet,
        ImageReferences,
        VideoReferenceSet,
        VideoReferences,
    )
    from .wan_nodes import WanVideoFramesPadding, _pad_first_dim
except ImportError:
    from minimax_adapter import apply_h3_progressive_resolution_patch
    from reference_nodes import (
        AudioReferenceSet,
        AudioReferences,
        ImageReferenceSet,
        ImageReferences,
        VideoReferenceSet,
        VideoReferences,
    )
    from wan_nodes import WanVideoFramesPadding, _pad_first_dim


def _is_h3_frame_count(frame_count: int) -> bool:
    return frame_count >= 5 and frame_count % 17 == 5


def _ceil_h3_frame_count(frame_count: int) -> int:
    frame_count = max(int(frame_count), 5)
    remainder = (frame_count - 5) % 17
    return frame_count if remainder == 0 else frame_count + 17 - remainder


class MiniMaxH3VideoFramesPadding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_frame_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Use 0 to round up to MiniMax H3's 17*n+5 frame grid.",
                    },
                ),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height", "length", "input_length")
    FUNCTION = "pad"
    CATEGORY = "Turing Utils/video"
    TITLE = "MiniMax H3 Video Frames Padding"

    def pad(self, image, target_frame_count: int, mask=None):
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError(f"Expected IMAGE shaped [frames, height, width, channels], got {tuple(image.shape)}")
        frame_count = int(image.shape[0])
        if frame_count < 1:
            raise ValueError("MiniMax H3 Video Frames Padding requires at least one input frame")
        height = int(image.shape[1])
        width = int(image.shape[2])
        mask = WanVideoFramesPadding._validate_mask(mask, frame_count, height, width)

        if target_frame_count == 0:
            output_length = _ceil_h3_frame_count(frame_count)
        else:
            output_length = int(target_frame_count)
            if not _is_h3_frame_count(output_length):
                raise ValueError(f"target_frame_count must be 0 or 17*n+5; got {target_frame_count}")
            if output_length < frame_count:
                raise ValueError(
                    f"target_frame_count={output_length} is shorter than the input video "
                    f"({frame_count} frames). Trim upstream if needed."
                )

        pad_count = output_length - frame_count
        image = _pad_first_dim(image, pad_count)
        if mask is not None:
            mask = _pad_first_dim(mask, pad_count)
        return (image, mask, width, height, output_length, frame_count)


class MiniMaxH3ReferenceConditionHub(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsMiniMaxH3ReferenceConditionHub",
            display_name="MiniMax H3 Reference Condition (Hub)",
            category="Turing Utils/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Combo.Input("audio_binding", options=["standalone", "pair_by_index"], default="standalone"),
                ImageReferences.Input("image_references", optional=True),
                VideoReferences.Input("video_references", optional=True),
                AudioReferences.Input("audio_references", optional=True),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length, ref_image_size="match",
                audio_binding="standalone", image_references=None, video_references=None, audio_references=None):
        images = image_references.materialize() if isinstance(image_references, ImageReferenceSet) else ()
        videos = video_references.materialize() if isinstance(video_references, VideoReferenceSet) else ()
        audios = audio_references.items if isinstance(audio_references, AudioReferenceSet) else ()

        ref_images = {f"ref_image_{index}": image for index, image in enumerate(images)}
        ref_videos = {f"ref_video_{index}": video for index, video in enumerate(videos)}
        ref_video_audios = {}
        standalone_audios = audios
        if audio_binding == "pair_by_index":
            paired = min(len(videos), len(audios))
            ref_video_audios = {f"ref_video_audio_{index}": audios[index] for index in range(paired)}
            standalone_audios = audios[paired:]
        ref_audios = {f"ref_audio_{index}": audio for index, audio in enumerate(standalone_audios)}

        return MiniMaxH3ReferenceToVideo.execute(
            clip=clip,
            vae=vae,
            audio_vae=audio_vae,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            ref_image_size=ref_image_size,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
        )


class MiniMaxH3ProgressiveResolutionPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "low_short_edge": (
                    "INT",
                    {
                        "default": 480,
                        "min": 32,
                        "max": 16384,
                        "step": 32,
                        "tooltip": "Early H3 DiT calls use this pixel short edge while the sampler keeps the final-resolution latent.",
                    },
                ),
                "low_resolution_steps": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of initial model evaluations to run at the low-stage short edge.",
                    },
                ),
                "input_downscale": (
                    ["sigma_blend", "nearest-exact", "area"],
                    {
                        "default": "sigma_blend",
                        "tooltip": "Sigma blend transitions from noise-preserving nearest sampling toward area filtering across the combined low and medium stages.",
                    },
                ),
                "output_upscale": (
                    ["bilinear", "bicubic", "nearest-exact"],
                    {
                        "default": "bilinear",
                        "tooltip": "Interpolation used to return each staged denoised video prediction to the sampler's final resolution.",
                    },
                ),
                "visual_condition_policy": (
                    ["resize_keyframes", "keep_original"],
                    {
                        "default": "resize_keyframes",
                        "tooltip": "Resize already-encoded first/last-frame latents for staged calls, or retain their final-resolution condition tokens.",
                    },
                ),
            },
            "optional": {
                "debug": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Log the resolved latent geometry once for each active stage per sampling run.",
                    },
                ),
                "medium_short_edge": (
                    "INT",
                    {
                        "default": 720,
                        "min": 32,
                        "max": 16384,
                        "step": 32,
                        "tooltip": "After the low stage, H3 DiT calls use this pixel short edge. Stage names describe order; this value may be smaller than low_short_edge.",
                    },
                ),
                "medium_resolution_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of model evaluations after the low stage to run at medium_short_edge. Zero disables this stage.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Turing Utils/patches"
    TITLE = "Patch H3 Progressive Resolution (Experimental)"
    EXPERIMENTAL = True

    def patch(
        self,
        model,
        low_short_edge: int = 480,
        low_resolution_steps: int = 2,
        input_downscale: str = "sigma_blend",
        output_upscale: str = "bilinear",
        visual_condition_policy: str = "resize_keyframes",
        debug: bool = False,
        medium_short_edge: int = 720,
        medium_resolution_steps: int = 0,
    ):
        return (
            apply_h3_progressive_resolution_patch(
                model,
                low_short_edge=low_short_edge,
                low_resolution_steps=low_resolution_steps,
                medium_short_edge=medium_short_edge,
                medium_resolution_steps=medium_resolution_steps,
                input_downscale=input_downscale,
                output_upscale=output_upscale,
                visual_condition_policy=visual_condition_policy,
                debug=debug,
            ),
        )
