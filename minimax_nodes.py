from __future__ import annotations

import comfy.nested_tensor
import comfy.utils
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


class MiniMaxH3LatentResize:
    SEARCH_ALIASES = ["resize h3 latent", "upscale h3 latent", "scale h3 latent"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "width": (
                    "INT",
                    {
                        "default": 1344,
                        "min": 32,
                        "max": 16384,
                        "step": 32,
                        "tooltip": "Exact output width in pixels. MiniMax H3 requires multiples of 32.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 768,
                        "min": 32,
                        "max": 16384,
                        "step": 32,
                        "tooltip": "Exact output height in pixels. Width and height are scaled independently without cropping or padding.",
                    },
                ),
                "resize_method": (
                    ["bilinear", "bicubic", "bislerp", "nearest-exact", "area"],
                    {
                        "default": "bilinear",
                        "tooltip": "Spatial interpolation for the video latent. The audio latent is never resized.",
                    },
                ),
                "resize_keyframes": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When conditioning is connected, resize its first/last H3 keyframe latents to the same spatial grid.",
                    },
                ),
            },
            "optional": {
                "conditioning": (
                    "CONDITIONING",
                    {
                        "tooltip": "Optional H3 conditioning to resize alongside the target latent. Independent references are left unchanged.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING")
    RETURN_NAMES = ("samples", "conditioning")
    FUNCTION = "resize"
    CATEGORY = "Turing Utils/latent"
    TITLE = "Resize MiniMax H3 AV Latent (Experimental)"
    EXPERIMENTAL = True

    @staticmethod
    def _streams(samples):
        if not isinstance(samples, dict) or "samples" not in samples:
            raise ValueError("Resize MiniMax H3 AV Latent requires a LATENT dictionary")
        nested = samples["samples"]
        if not getattr(nested, "is_nested", False):
            raise ValueError("Expected a MiniMax H3 nested video/audio latent")
        streams = list(nested.unbind())
        if len(streams) != 2:
            raise ValueError(f"Expected exactly two H3 latent streams, got {len(streams)}")
        video, audio = streams
        if video.ndim != 5 or int(video.shape[1]) != 24:
            raise ValueError(
                f"Expected H3 video latent [B,24,T,H,W], got {tuple(video.shape)}"
            )
        if audio.ndim != 4 or tuple(audio.shape[1:3]) != (32, 2):
            raise ValueError(
                f"Expected H3 audio latent [B,32,2,T], got {tuple(audio.shape)}"
            )
        if int(video.shape[0]) != int(audio.shape[0]):
            raise ValueError("H3 video and audio latent batch sizes must match")
        return video, audio

    @staticmethod
    def _resize_conditioning_keyframes(conditioning, latent_width, latent_height, resize_method):
        if conditioning is None:
            return None
        if not isinstance(conditioning, (list, tuple)):
            raise ValueError("Expected ComfyUI CONDITIONING to be a list or tuple")

        output = []
        cache = {}
        for entry in conditioning:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not isinstance(entry[1], dict):
                output.append(entry)
                continue
            metadata = entry[1]
            keyframes = metadata.get("minimax_keyframes")
            if not keyframes:
                output.append(entry)
                continue

            resized_keyframes = []
            for keyframe in keyframes:
                if not isinstance(keyframe, dict) or not hasattr(keyframe.get("latent"), "shape"):
                    resized_keyframes.append(keyframe)
                    continue
                latent = keyframe["latent"]
                if latent.ndim != 5 or int(latent.shape[1]) != 24:
                    raise ValueError(
                        "Expected H3 keyframe latent [B,24,T,H,W], got "
                        f"{tuple(latent.shape)}"
                    )
                cache_key = (id(latent), int(latent_height), int(latent_width), str(resize_method))
                resized = cache.get(cache_key)
                if resized is None:
                    resized = latent
                    if tuple(latent.shape[-2:]) != (latent_height, latent_width):
                        resized = comfy.utils.common_upscale(
                            latent,
                            latent_width,
                            latent_height,
                            resize_method,
                            "disabled",
                        )
                    cache[cache_key] = resized
                updated_keyframe = keyframe.copy()
                updated_keyframe["latent"] = resized
                resized_keyframes.append(updated_keyframe)

            updated_metadata = metadata.copy()
            updated_metadata["minimax_keyframes"] = resized_keyframes
            updated_entry = list(entry)
            updated_entry[1] = updated_metadata
            output.append(tuple(updated_entry) if isinstance(entry, tuple) else updated_entry)
        return tuple(output) if isinstance(conditioning, tuple) else output

    def resize(
        self,
        samples,
        width: int,
        height: int,
        resize_method: str = "bilinear",
        resize_keyframes: bool = True,
        conditioning=None,
    ):
        width = int(width)
        height = int(height)
        if width < 32 or height < 32 or width % 32 or height % 32:
            raise ValueError(
                f"MiniMax H3 output width and height must be positive multiples of 32; got {width}x{height}"
            )
        if resize_method not in ("bilinear", "bicubic", "bislerp", "nearest-exact", "area"):
            raise ValueError(f"Unsupported H3 latent resize method: {resize_method}")
        if "noise_mask" in samples:
            raise ValueError(
                "Resize MiniMax H3 AV Latent does not support attached noise masks; resize or remove the mask before this node"
            )

        video, audio = self._streams(samples)
        latent_width = width // 16
        latent_height = height // 16
        if tuple(video.shape[-2:]) == (latent_height, latent_width):
            resized_video = video
        else:
            resized_video = comfy.utils.common_upscale(
                video,
                latent_width,
                latent_height,
                resize_method,
                "disabled",
            )

        output = samples.copy()
        output["samples"] = comfy.nested_tensor.NestedTensor((resized_video, audio))
        if conditioning is not None and bool(resize_keyframes):
            conditioning = self._resize_conditioning_keyframes(
                conditioning,
                latent_width,
                latent_height,
                resize_method,
            )
        return (output, conditioning)


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
                        "tooltip": "Interpolation used to transfer each staged flow prediction to the sampler's final resolution.",
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
