"""Composable MiniMax H3 multimodal reference-conditioning nodes."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import torch
import torchaudio

import comfy.utils
import node_helpers
from comfy_api.latest import io


H3_MODEL_FPS = 24.0
H3_QWEN_VIDEO_FPS = 2.0
H3_PIXEL_ALIGNMENT = 32
H3_SPATIAL_DOWNSCALE = 16

H3KeyframesReferenceType = io.Custom("TURING_UTILS_H3_KEYFRAMES_REFERENCE")
H3ImageReferenceType = io.Custom("TURING_UTILS_H3_IMAGE_REFERENCE")
H3VideoReferenceType = io.Custom("TURING_UTILS_H3_VIDEO_REFERENCE")
H3AudioReferenceType = io.Custom("TURING_UTILS_H3_AUDIO_REFERENCE")
H3SemanticReferenceType = io.Custom("TURING_UTILS_H3_SEMANTIC_REFERENCE")


@dataclass(frozen=True)
class H3ReferenceManifest:
    keyframe_anchors: tuple[str, ...] = ()
    image_count: int = 0
    video_audio: tuple[bool, ...] = ()
    audio_count: int = 0


@dataclass(frozen=True)
class H3KeyframesReferenceData:
    items: tuple[dict, ...]


@dataclass(frozen=True)
class H3ImageReferenceData:
    items: tuple[dict, ...]


@dataclass(frozen=True)
class H3VideoReferenceData:
    items: tuple[dict, ...]


@dataclass(frozen=True)
class H3AudioReferenceData:
    items: tuple[dict, ...]


@dataclass(frozen=True)
class H3SemanticReferenceData:
    conditioning: object
    manifest: H3ReferenceManifest


def _samples(latent):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("latent must be a LATENT dictionary")
    return latent["samples"]


def _video_latent(latent) -> torch.Tensor:
    samples = _samples(latent)
    if getattr(samples, "is_nested", False):
        streams = list(samples.unbind())
        if not streams:
            raise ValueError("H3 AV latent contains no video stream")
        samples = streams[0]
    if not torch.is_tensor(samples) or samples.ndim != 5 or int(samples.shape[1]) != 24:
        shape = (
            tuple(samples.shape)
            if hasattr(samples, "shape")
            else type(samples).__name__
        )
        raise ValueError(f"Expected H3 video latent [B,24,T,H,W], got {shape}")
    return samples


def _frame_count_from_latent_t(latent_t: int) -> int:
    latent_t = int(latent_t)
    if latent_t == 1:
        return 1
    if latent_t >= 2 and (latent_t - 2) % 5 == 0:
        return ((latent_t - 2) // 5) * 17 + 5
    raise ValueError(
        f"H3 video latent T={latent_t} is outside the canonical 5*k+2 temporal grid"
    )


def h3_latent_info(latent) -> tuple[int, int, int, float]:
    video = _video_latent(latent)
    return (
        int(video.shape[-1]) * H3_SPATIAL_DOWNSCALE,
        int(video.shape[-2]) * H3_SPATIAL_DOWNSCALE,
        _frame_count_from_latent_t(int(video.shape[2])),
        H3_MODEL_FPS,
    )


def _validate_pixels(image: torch.Tensor, name: str) -> torch.Tensor:
    if (
        not torch.is_tensor(image)
        or image.ndim != 4
        or int(image.shape[0]) < 1
        or int(image.shape[-1]) < 3
    ):
        shape = tuple(image.shape) if hasattr(image, "shape") else type(image).__name__
        raise ValueError(
            f"{name} must be IMAGE frames [N,H,W,C] with at least three channels; got {shape}"
        )
    return image[..., :3]


def _crop_to_alignment(image: torch.Tensor, name: str) -> torch.Tensor:
    image = _validate_pixels(image, name)
    height, width = int(image.shape[1]), int(image.shape[2])
    aligned_width = width - width % H3_PIXEL_ALIGNMENT
    aligned_height = height - height % H3_PIXEL_ALIGNMENT
    if aligned_width < H3_PIXEL_ALIGNMENT or aligned_height < H3_PIXEL_ALIGNMENT:
        raise ValueError(
            f"{name} must be at least {H3_PIXEL_ALIGNMENT}x{H3_PIXEL_ALIGNMENT}; "
            f"got {width}x{height}"
        )
    left = (width - aligned_width) // 2
    top = (height - aligned_height) // 2
    return image[:, top : top + aligned_height, left : left + aligned_width]


def _align_keyframe_pixels(image: torch.Tensor, latent, name: str) -> torch.Tensor:
    image = _validate_pixels(image, name)
    if latent is None:
        return _crop_to_alignment(image, name)
    width, height, _, _ = h3_latent_info(latent)
    return comfy.utils.common_upscale(
        image.movedim(-1, 1), width, height, "lanczos", "center"
    ).movedim(1, -1)


def _align_reference_pixels(
    image: torch.Tensor,
    latent,
    name: str,
    megapixels: float,
) -> torch.Tensor:
    image = _validate_pixels(image, name)
    height, width = int(image.shape[1]), int(image.shape[2])
    if latent is None:
        megapixels = float(megapixels)
        if not math.isfinite(megapixels) or megapixels <= 0.0:
            raise ValueError("megapixels must be finite and positive")
        target_area = megapixels * 1_000_000.0
    else:
        target_width, target_height, _, _ = h3_latent_info(latent)
        target_area = target_width * target_height
    scale = min(1.0, math.sqrt(target_area / (width * height)))
    aligned_width = max(
        H3_PIXEL_ALIGNMENT,
        round(width * scale / H3_PIXEL_ALIGNMENT) * H3_PIXEL_ALIGNMENT,
    )
    aligned_height = max(
        H3_PIXEL_ALIGNMENT,
        round(height * scale / H3_PIXEL_ALIGNMENT) * H3_PIXEL_ALIGNMENT,
    )
    return comfy.utils.common_upscale(
        image.movedim(-1, 1),
        aligned_width,
        aligned_height,
        "lanczos",
        "disabled",
    ).movedim(1, -1)


def _validate_visual_latent(value, name: str) -> torch.Tensor:
    if (
        not torch.is_tensor(value)
        or value.ndim != 5
        or int(value.shape[0]) != 1
        or int(value.shape[1]) != 24
    ):
        shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        raise ValueError(f"{name} VAE must produce [1,24,T,H,W], got {shape}")
    return value


def _encode_visual(vae, pixels: torch.Tensor, name: str) -> torch.Tensor:
    return _validate_visual_latent(vae.encode(pixels), name)


def _encode_audio(audio_vae, audio, name: str) -> torch.Tensor:
    if (
        not isinstance(audio, dict)
        or "waveform" not in audio
        or "sample_rate" not in audio
    ):
        raise ValueError(f"{name} must be an AUDIO dictionary")
    waveform = audio["waveform"]
    if not torch.is_tensor(waveform) or waveform.ndim != 3:
        shape = (
            tuple(waveform.shape)
            if hasattr(waveform, "shape")
            else type(waveform).__name__
        )
        raise ValueError(f"{name} waveform must be [B,C,L], got {shape}")
    source_rate = int(audio["sample_rate"])
    target_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if source_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, target_rate)
    value = audio_vae.encode(waveform[:1].movedim(1, -1))
    if (
        not torch.is_tensor(value)
        or value.ndim != 4
        or tuple(value.shape[1:3]) != (32, 2)
    ):
        shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        raise ValueError(f"{name} Audio VAE must produce [1,32,2,T], got {shape}")
    return value


def _natural_key(value: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", str(value))
    return (
        str(value)[: match.start()] if match else str(value),
        int(match.group(1)) if match else -1,
    )


def _dynamic_entries(values) -> tuple[tuple[str, object], ...]:
    if not isinstance(values, dict):
        return ()
    return tuple(
        (str(key), value)
        for key, value in sorted(
            values.items(), key=lambda item: _natural_key(str(item[0]))
        )
        if value is not None
    )


def _dynamic_suffix(name: str) -> int:
    match = re.search(r"(\d+)$", str(name))
    return int(match.group(1)) if match else -1


def _trim_h3_reference_video(frames: torch.Tensor) -> torch.Tensor:
    frame_count = int(frames.shape[0])
    if frame_count < 5:
        raise ValueError(
            "H3 reference videos require at least five input frames at 24 FPS"
        )
    valid_count = frame_count - (frame_count - 5) % 17
    return frames[:valid_count]


def _manifest(
    keyframes_reference: H3KeyframesReferenceData | None,
    image_reference: H3ImageReferenceData | None,
    video_reference: H3VideoReferenceData | None,
    audio_reference: H3AudioReferenceData | None,
) -> H3ReferenceManifest:
    return H3ReferenceManifest(
        keyframe_anchors=tuple(item["anchor"] for item in keyframes_reference.items)
        if keyframes_reference
        else (),
        image_count=len(image_reference.items) if image_reference else 0,
        video_audio=tuple(
            item["audio_latent"] is not None for item in video_reference.items
        )
        if video_reference
        else (),
        audio_count=len(audio_reference.items) if audio_reference else 0,
    )


def _reference_presentation(
    image_reference: H3ImageReferenceData | None,
    video_reference: H3VideoReferenceData | None,
    audio_reference: H3AudioReferenceData | None,
) -> list[dict]:
    presentation = []
    if image_reference:
        presentation.extend(
            {"type": "image", "data": item["image"]} for item in image_reference.items
        )
    if video_reference:
        for item in video_reference.items:
            if item["audio_latent"] is not None:
                presentation.append({"type": "audio"})
            presentation.append(
                {
                    "type": "video",
                    "data": item["qwen_frames"],
                    "timestamps": item["timestamps"],
                }
            )
    if audio_reference:
        presentation.extend({"type": "audio"} for _ in audio_reference.items)
    return presentation


def _keyframes_reference(value):
    if value is None:
        return None
    if not isinstance(value, H3KeyframesReferenceData):
        raise ValueError("keyframes_reference must come from H3 Keyframes")
    return value


def _tokenize_semantic(clip, prompt: str, keyframes_reference, presentation):
    images = (
        [item["image"] for item in keyframes_reference.items]
        if keyframes_reference
        else []
    )
    if not images:
        return (
            clip.tokenize(prompt, minimax_ref_items=presentation)
            if presentation
            else clip.tokenize(prompt)
        )
    if not presentation:
        return clip.tokenize(prompt, images=images)

    # Upstream exposes the official FL2VA and Ref2VA presentations as exclusive
    # arguments. Their raw H3 token streams are composable before Qwen encoding.
    keyframe_tokens = clip.tokenize("", images=images)
    reference_tokens = clip.tokenize(prompt, minimax_ref_items=presentation)
    key = "qwen3vl_32b"
    try:
        keyframe_rows = keyframe_tokens[key]
        reference_rows = reference_tokens[key]
        if len(keyframe_rows) != 1 or len(reference_rows) != 1:
            raise ValueError
        return {key: [keyframe_rows[0] + reference_rows[0]]}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "clip is not a compatible MiniMax H3 Qwen3-VL encoder"
        ) from error


def _reference_blocks(
    image_reference: H3ImageReferenceData | None,
    video_reference: H3VideoReferenceData | None,
    audio_reference: H3AudioReferenceData | None,
) -> list[dict]:
    refs = []
    if image_reference:
        refs.extend(
            {
                "kind": "image",
                "latent_h": int(item["latent"].shape[-2]),
                "latent_w": int(item["latent"].shape[-1]),
                "latent": item["latent"],
            }
            for item in image_reference.items
        )
    if video_reference:
        refs.extend(
            {
                "kind": "video_audio" if item["audio_latent"] is not None else "video",
                "latent_t": int(item["latent"].shape[2]),
                "latent_h": int(item["latent"].shape[-2]),
                "latent_w": int(item["latent"].shape[-1]),
                "ref_audio_t": int(item["audio_latent"].shape[-1])
                if item["audio_latent"] is not None
                else 0,
                "latent": item["latent"],
                "audio_latent": item["audio_latent"],
            }
            for item in video_reference.items
        )
    if audio_reference:
        refs.extend(
            {
                "kind": "audio",
                "ref_audio_t": int(item["audio_latent"].shape[-1]),
                "audio_latent": item["audio_latent"],
            }
            for item in audio_reference.items
        )
    return refs


class H3Keyframes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3Keyframes",
            display_name="H3 Keyframes",
            category="Turing Utils/conditioning/minimax",
            description=(
                "Encode H3 keyframe anchors as one ordered reference. With an optional "
                "latent, images are cover-resized and cropped to its decoded pixel canvas."
            ),
            inputs=[
                io.Vae.Input("vae"),
                io.Latent.Input("latent", optional=True),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[H3KeyframesReferenceType.Output("keyframes_reference")],
        )

    @classmethod
    def execute(
        cls, vae, latent=None, first_frame=None, last_frame=None
    ) -> io.NodeOutput:
        def prepare(value, role):
            if value is None:
                return None
            pixels = _align_keyframe_pixels(value[:1], latent, role)
            return {
                "image": pixels,
                "latent": _encode_visual(vae, pixels, role),
            }

        items = []
        for anchor, value in (("first", first_frame), ("last", last_frame)):
            item = prepare(value, f"{anchor}_frame")
            if item is not None:
                item["anchor"] = anchor
                items.append(item)
        return io.NodeOutput(H3KeyframesReferenceData(tuple(items)))


class H3ImageReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3ImageReference",
            display_name="H3 Image Reference",
            category="Turing Utils/conditioning/minimax",
            description=(
                "Encode dynamic H3 reference images without cropping or upscaling. "
                "A latent applies match-area sizing; without one, the short edge is "
                "limited by the megapixel budget."
            ),
            inputs=[
                io.Vae.Input("vae"),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    tooltip="Maximum source area when latent is not connected; smaller images are not enlarged.",
                ),
                io.Latent.Input("latent", optional=True),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image"),
                        prefix="image_",
                        min=0,
                        max=32,
                    ),
                ),
            ],
            outputs=[H3ImageReferenceType.Output("image_reference")],
        )

    @classmethod
    def execute(cls, vae, megapixels=1.0, latent=None, images=None) -> io.NodeOutput:
        items = []
        for name, image in _dynamic_entries(images):
            pixels = _align_reference_pixels(image[:1], latent, name, float(megapixels))
            items.append(
                {
                    "image": pixels,
                    "latent": _encode_visual(vae, pixels, name),
                }
            )
        return io.NodeOutput(H3ImageReferenceData(tuple(items)))


class H3VideoReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3VideoReference",
            display_name="H3 Video Reference",
            category="Turing Utils/conditioning/minimax",
            description=(
                "Encode dynamic H3 reference videos that were resampled to 24 FPS "
                "upstream, plus index-paired soundtracks. Qwen receives a 2 FPS view; "
                "the DiT receives the full VAE latent. A latent applies match-area "
                "sizing; without one, the megapixel budget limits source area."
            ),
            inputs=[
                io.Vae.Input("video_vae"),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    tooltip="Maximum source area when latent is not connected; smaller videos are not enlarged.",
                ),
                io.Vae.Input("audio_vae", optional=True),
                io.Latent.Input("latent", optional=True),
                io.Autogrow.Input(
                    "videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "video",
                            tooltip="Consecutive frames already resampled to 24 FPS",
                        ),
                        prefix="video_",
                        min=0,
                        max=16,
                    ),
                ),
                io.Autogrow.Input(
                    "video_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("video_audio"),
                        prefix="video_audio_",
                        min=0,
                        max=16,
                    ),
                ),
            ],
            outputs=[H3VideoReferenceType.Output("video_reference")],
        )

    @classmethod
    def execute(
        cls,
        video_vae,
        megapixels=1.0,
        audio_vae=None,
        latent=None,
        videos=None,
        video_audios=None,
    ) -> io.NodeOutput:
        soundtracks = {
            _dynamic_suffix(name): value
            for name, value in _dynamic_entries(video_audios)
        }
        items = []
        for name, video in _dynamic_entries(videos):
            frames = _validate_pixels(video, name)
            frames = _trim_h3_reference_video(frames)
            frames = _align_reference_pixels(frames, latent, name, float(megapixels))
            visual_latent = _encode_visual(video_vae, frames, name)
            soundtrack = soundtracks.get(_dynamic_suffix(name))
            audio_latent = None
            if soundtrack is not None:
                if audio_vae is None:
                    raise ValueError(
                        f"{name} has a soundtrack but audio_vae is not connected"
                    )
                audio_latent = _encode_audio(
                    audio_vae, soundtrack, f"{name} soundtrack"
                )
            sample_stride = int(H3_MODEL_FPS / H3_QWEN_VIDEO_FPS)
            indices = torch.arange(
                0, int(frames.shape[0]), sample_stride, device=frames.device
            )
            qwen_frames = frames.index_select(0, indices)
            items.append(
                {
                    "latent": visual_latent,
                    "audio_latent": audio_latent,
                    "qwen_frames": qwen_frames,
                    "timestamps": [
                        index / H3_QWEN_VIDEO_FPS
                        for index in range(int(qwen_frames.shape[0]))
                    ],
                }
            )
        return io.NodeOutput(H3VideoReferenceData(tuple(items)))


class H3AudioReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3AudioReference",
            display_name="H3 Audio Reference",
            category="Turing Utils/conditioning/minimax",
            description="Encode a dynamic set of standalone H3 reference audio clips.",
            inputs=[
                io.Vae.Input("audio_vae"),
                io.Autogrow.Input(
                    "audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("audio"),
                        prefix="audio_",
                        min=0,
                        max=16,
                    ),
                ),
            ],
            outputs=[H3AudioReferenceType.Output("audio_reference")],
        )

    @classmethod
    def execute(cls, audio_vae, audios=None) -> io.NodeOutput:
        items = tuple(
            {"audio_latent": _encode_audio(audio_vae, audio, name)}
            for name, audio in _dynamic_entries(audios)
        )
        return io.NodeOutput(H3AudioReferenceData(items))


class H3SemanticReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3SemanticReference",
            display_name="H3 Semantic Reference",
            category="Turing Utils/conditioning/minimax",
            description=(
                "Run one exact Qwen3-VL multimodal encode over the prompt and selected "
                "H3 references. The resulting semantic reference can be reused with "
                "structure-equivalent DiT references at another resolution."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                H3KeyframesReferenceType.Input("keyframes_reference", optional=True),
                H3ImageReferenceType.Input("image_reference", optional=True),
                H3VideoReferenceType.Input("video_reference", optional=True),
                H3AudioReferenceType.Input("audio_reference", optional=True),
            ],
            outputs=[H3SemanticReferenceType.Output("semantic_reference")],
        )

    @classmethod
    def execute(
        cls,
        clip,
        prompt: str,
        keyframes_reference=None,
        image_reference=None,
        video_reference=None,
        audio_reference=None,
    ) -> io.NodeOutput:
        keyframes_reference = _keyframes_reference(keyframes_reference)
        manifest = _manifest(
            keyframes_reference,
            image_reference,
            video_reference,
            audio_reference,
        )
        presentation = _reference_presentation(
            image_reference, video_reference, audio_reference
        )
        tokens = _tokenize_semantic(clip, prompt, keyframes_reference, presentation)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(H3SemanticReferenceData(conditioning, manifest))


class H3BuildConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3BuildConditioning",
            display_name="H3 Build Conditioning",
            category="Turing Utils/conditioning/minimax",
            description=(
                "Combine a reusable Qwen semantic reference with current-resolution "
                "first-last-frame, image, video, and audio VAE references."
            ),
            inputs=[
                H3SemanticReferenceType.Input("semantic_reference"),
                io.Latent.Input("latent"),
                H3KeyframesReferenceType.Input("keyframes_reference", optional=True),
                H3ImageReferenceType.Input("image_reference", optional=True),
                H3VideoReferenceType.Input("video_reference", optional=True),
                H3AudioReferenceType.Input("audio_reference", optional=True),
            ],
            outputs=[io.Conditioning.Output("conditioning")],
        )

    @classmethod
    def execute(
        cls,
        semantic_reference,
        latent,
        keyframes_reference=None,
        image_reference=None,
        video_reference=None,
        audio_reference=None,
    ) -> io.NodeOutput:
        if not isinstance(semantic_reference, H3SemanticReferenceData):
            raise ValueError("semantic_reference must come from H3 Semantic Reference")
        keyframes_reference = _keyframes_reference(keyframes_reference)
        manifest = _manifest(
            keyframes_reference,
            image_reference,
            video_reference,
            audio_reference,
        )
        if manifest != semantic_reference.manifest:
            raise ValueError(
                "Semantic and DiT H3 reference structures differ: "
                f"semantic={semantic_reference.manifest}, dit={manifest}"
            )

        target_video = _video_latent(latent)
        frame_count = _frame_count_from_latent_t(int(target_video.shape[2]))
        keyframes = []
        for item in keyframes_reference.items if keyframes_reference else ():
            anchor = item["anchor"]
            role = f"{anchor}_frame"
            frame_index = 0 if anchor == "first" else frame_count - 1
            visual = _validate_visual_latent(item["latent"], role)
            if int(visual.shape[2]) != 1 or tuple(visual.shape[-2:]) != tuple(
                target_video.shape[-2:]
            ):
                raise ValueError(
                    f"{role} latent {tuple(visual.shape)} does not match target H3 "
                    f"spatial grid {tuple(target_video.shape[-2:])}; connect the same "
                    "latent to H3 Keyframes or resize upstream"
                )
            keyframes.append({"resolved_frame_index": frame_index, "latent": visual})

        refs = _reference_blocks(image_reference, video_reference, audio_reference)
        values = {"minimax_frame_count": frame_count}
        if keyframes:
            values["minimax_keyframes"] = keyframes
        if refs:
            values["minimax_refs"] = refs
        conditioning = node_helpers.conditioning_set_values(
            semantic_reference.conditioning, values
        )
        return io.NodeOutput(conditioning)


class H3LatentInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3LatentInfo",
            display_name="H3 Latent Info",
            category="Turing Utils/latent",
            description=(
                "Read the decoded pixel width, height, canonical frame count, and "
                "model FPS from an H3 video or nested AV latent without decoding it."
            ),
            inputs=[io.Latent.Input("latent")],
            outputs=[
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Int.Output("length"),
                io.Float.Output("fps"),
            ],
        )

    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        return io.NodeOutput(*h3_latent_info(latent))


__all__ = [
    "H3AudioReference",
    "H3AudioReferenceData",
    "H3BuildConditioning",
    "H3Keyframes",
    "H3KeyframesReferenceData",
    "H3ImageReference",
    "H3ImageReferenceData",
    "H3LatentInfo",
    "H3ReferenceManifest",
    "H3SemanticReference",
    "H3SemanticReferenceData",
    "H3VideoReference",
    "H3VideoReferenceData",
    "h3_latent_info",
]
