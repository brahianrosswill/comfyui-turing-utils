from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import comfy.utils
from comfy_api.latest import io


ImageReferences = io.Custom("TURING_REFERENCE_IMAGES")
VideoReferences = io.Custom("TURING_REFERENCE_VIDEOS")
AudioReferences = io.Custom("TURING_REFERENCE_AUDIOS")


@dataclass(frozen=True)
class SpatialOptions:
    enabled: bool = False
    width: int = 0
    height: int = 0
    mode: str = "fill"
    method: str = "lanczos"
    crop_position: str = "center"
    divisible_by: int = 1
    pad_color: str = "0, 0, 0"


@dataclass(frozen=True)
class VideoOptions:
    spatial: SpatialOptions = SpatialOptions()
    align_frames: bool = False
    frame_count_mode: str = "minimum"
    frame_count: int = 0
    short_video_fill: str = "repeat_last"
    trim_position: str = "head"


@dataclass(frozen=True)
class ImageReferenceSet:
    items: tuple[torch.Tensor, ...] = ()
    options: SpatialOptions = SpatialOptions()

    def materialize(self) -> tuple[torch.Tensor, ...]:
        return tuple(_spatial_transform(image, self.options) for image in self.items)


@dataclass(frozen=True)
class VideoReferenceSet:
    items: tuple[torch.Tensor, ...] = ()
    options: VideoOptions = VideoOptions()

    def materialize(self) -> tuple[torch.Tensor, ...]:
        videos = [_spatial_transform(video, self.options.spatial) for video in self.items]
        if not self.options.align_frames or not videos:
            return tuple(videos)

        lengths = [int(video.shape[0]) for video in videos]
        if self.options.frame_count_mode == "minimum":
            target = min(lengths)
        elif self.options.frame_count_mode == "maximum":
            target = max(lengths)
        else:
            target = int(self.options.frame_count)
            if target < 1:
                raise ValueError("specified frame alignment requires frame_count >= 1")
        return tuple(_align_video_frames(video, target, self.options) for video in videos)


@dataclass(frozen=True)
class AudioReferenceSet:
    items: tuple[dict, ...] = ()


def _validate_image(image: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[-1] < 3:
        shape = tuple(image.shape) if torch.is_tensor(image) else type(image).__name__
        raise ValueError(f"{name} must be an IMAGE tensor shaped [frames, height, width, channels], got {shape}")
    if image.shape[0] < 1 or image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError(f"{name} must not be empty")
    return image[..., :3]


def _resolve_target_size(width: int, height: int, source_width: int, source_height: int, divisible_by: int) -> tuple[int, int]:
    width = int(width)
    height = int(height)
    if width <= 0 and height <= 0:
        width, height = source_width, source_height
    elif width <= 0:
        width = max(round(source_width * height / source_height), 1)
    elif height <= 0:
        height = max(round(source_height * width / source_width), 1)

    divisible_by = max(int(divisible_by), 1)
    if divisible_by > 1:
        width = max(divisible_by, width - width % divisible_by)
        height = max(divisible_by, height - height % divisible_by)
    return width, height


def _crop_offsets(extra_width: int, extra_height: int, position: str) -> tuple[int, int]:
    if position == "top":
        return extra_width // 2, 0
    if position == "bottom":
        return extra_width // 2, extra_height
    if position == "left":
        return 0, extra_height // 2
    if position == "right":
        return extra_width, extra_height // 2
    return extra_width // 2, extra_height // 2


def _parse_pad_color(value: str, channels: int, device, dtype) -> torch.Tensor:
    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise ValueError("pad_color must contain comma-separated numeric values") from error
    if not values:
        values = [0.0]
    if len(values) == 1:
        values *= channels
    if len(values) < channels:
        raise ValueError(f"pad_color needs either one value or at least {channels} values")
    values = values[:channels]
    if max(abs(item) for item in values) > 1.0:
        values = [item / 255.0 for item in values]
    return torch.tensor(values, device=device, dtype=dtype).reshape(1, channels, 1, 1)


def _resize_image(image: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height, method, "disabled").movedim(1, -1)


def _conservative_resize_mask(mask: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    source_height, source_width = int(mask.shape[-2]), int(mask.shape[-1])
    pooled_height = min(source_height, height)
    pooled_width = min(source_width, width)
    if (pooled_height, pooled_width) != (source_height, source_width):
        mask = F.adaptive_max_pool2d(mask, (pooled_height, pooled_width))
    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask, size=(height, width), mode="nearest-exact")
    return F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)


def _spatial_transform(image: torch.Tensor, options: SpatialOptions, mask: torch.Tensor | None = None):
    image = _validate_image(image, "reference")
    if mask is not None:
        if mask.ndim == 2 and image.shape[0] == 1:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3 or tuple(mask.shape) != tuple(image.shape[:3]):
            raise ValueError(
                "MASK must match IMAGE before resize; expected "
                f"{tuple(image.shape[:3])}, got {tuple(mask.shape)}"
            )

    if not options.enabled:
        return (image, mask) if mask is not None else image

    source_height, source_width = int(image.shape[1]), int(image.shape[2])
    target_width, target_height = _resolve_target_size(
        options.width, options.height, source_width, source_height, options.divisible_by
    )
    if options.mode == "stretch":
        output = _resize_image(image, target_width, target_height, options.method)
        if mask is not None:
            mask = _conservative_resize_mask(mask, target_width, target_height).squeeze(1)
        return (output, mask) if mask is not None else output

    scale = min(target_width / source_width, target_height / source_height)
    if options.mode == "fill":
        scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(round(source_width * scale), 1)
    resized_height = max(round(source_height * scale), 1)
    output = _resize_image(image, resized_width, resized_height, options.method)
    if mask is not None:
        mask = _conservative_resize_mask(mask, resized_width, resized_height).squeeze(1)

    if options.mode == "keep_aspect":
        return (output, mask) if mask is not None else output

    if options.mode == "fill":
        x, y = _crop_offsets(resized_width - target_width, resized_height - target_height, options.crop_position)
        output = output[:, y:y + target_height, x:x + target_width]
        if mask is not None:
            mask = mask[:, y:y + target_height, x:x + target_width]
        return (output, mask) if mask is not None else output

    x, y = _crop_offsets(target_width - resized_width, target_height - resized_height, options.crop_position)
    channels = int(output.shape[-1])
    color = _parse_pad_color(options.pad_color, channels, output.device, output.dtype)
    canvas = color.movedim(1, -1).expand(output.shape[0], target_height, target_width, channels).clone()
    canvas[:, y:y + resized_height, x:x + resized_width] = output
    if mask is not None:
        mask_canvas = mask.new_zeros((mask.shape[0], target_height, target_width))
        mask_canvas[:, y:y + resized_height, x:x + resized_width] = mask
        mask = mask_canvas
    return (canvas, mask) if mask is not None else canvas


def _align_video_frames(video: torch.Tensor, target: int, options: VideoOptions) -> torch.Tensor:
    length = int(video.shape[0])
    if length == target:
        return video
    if length > target:
        if options.trim_position == "tail":
            start = length - target
        elif options.trim_position == "center":
            start = (length - target) // 2
        else:
            start = 0
        return video[start:start + target]

    pad_count = target - length
    if options.short_video_fill == "black":
        padding = video.new_zeros((pad_count, *video.shape[1:]))
    else:
        padding = video[-1:].repeat(pad_count, 1, 1, 1)
    return torch.cat((video, padding), dim=0)


def _spatial_inputs() -> list:
    return [
        io.Boolean.Input("resize_enabled", default=False),
        io.Int.Input("width", default=0, min=0, max=16384, step=1),
        io.Int.Input("height", default=0, min=0, max=16384, step=1),
        io.Combo.Input("resize_mode", options=["fill", "fit", "stretch", "keep_aspect"], default="fill"),
        io.Combo.Input("upscale_method", options=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], default="lanczos"),
        io.Combo.Input("crop_position", options=["center", "top", "bottom", "left", "right"], default="center"),
        io.Int.Input("divisible_by", default=1, min=1, max=512, step=1, advanced=True),
        io.String.Input("pad_color", default="0, 0, 0", advanced=True),
    ]


def _spatial_options(resize_enabled, width, height, resize_mode, upscale_method, crop_position, divisible_by, pad_color):
    return SpatialOptions(
        enabled=bool(resize_enabled),
        width=int(width),
        height=int(height),
        mode=str(resize_mode),
        method=str(upscale_method),
        crop_position=str(crop_position),
        divisible_by=int(divisible_by),
        pad_color=str(pad_color),
    )


class ReferenceImageHub(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(io.Image.Input("image"), prefix="image_", min=0, max=64)
        return io.Schema(
            node_id="TuringUtilsReferenceImageHub",
            display_name="Reference Image Hub",
            category="Turing Utils/references",
            inputs=[
                io.Autogrow.Input("images", template=template, optional=True),
                ImageReferences.Input("previous", optional=True),
                *_spatial_inputs(),
            ],
            outputs=[ImageReferences.Output("references")],
        )

    @classmethod
    def execute(cls, images=None, previous=None, **kwargs):
        items = list(previous.items) if isinstance(previous, ImageReferenceSet) else []
        for image in (images or {}).values():
            image = _validate_image(image, "image reference")
            items.extend(image[index:index + 1].clone() for index in range(image.shape[0]))
        options = _spatial_options(**kwargs)
        return io.NodeOutput(ImageReferenceSet(tuple(items), options))


class ReferenceVideoHub(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(io.Image.Input("video"), prefix="video_", min=0, max=32)
        return io.Schema(
            node_id="TuringUtilsReferenceVideoHub",
            display_name="Reference Video Hub",
            category="Turing Utils/references",
            inputs=[
                io.Autogrow.Input("videos", template=template, optional=True),
                VideoReferences.Input("previous", optional=True),
                *_spatial_inputs(),
                io.Boolean.Input("frame_align_enabled", default=False),
                io.Combo.Input("frame_count_mode", options=["minimum", "maximum", "specified"], default="minimum"),
                io.Int.Input("frame_count", default=0, min=0, max=16385, step=1),
                io.Combo.Input("short_video_fill", options=["repeat_last", "black"], default="repeat_last"),
                io.Combo.Input("trim_position", options=["head", "center", "tail"], default="head", advanced=True),
            ],
            outputs=[VideoReferences.Output("references")],
        )

    @classmethod
    def execute(cls, videos=None, previous=None, frame_align_enabled=False, frame_count_mode="minimum", frame_count=0,
                short_video_fill="repeat_last", trim_position="head", **kwargs):
        items = list(previous.items) if isinstance(previous, VideoReferenceSet) else []
        for video in (videos or {}).values():
            items.append(_validate_image(video, "video reference"))
        options = VideoOptions(
            spatial=_spatial_options(**kwargs),
            align_frames=bool(frame_align_enabled),
            frame_count_mode=str(frame_count_mode),
            frame_count=int(frame_count),
            short_video_fill=str(short_video_fill),
            trim_position=str(trim_position),
        )
        return io.NodeOutput(VideoReferenceSet(tuple(items), options))


class ReferenceAudioHub(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(io.Audio.Input("audio"), prefix="audio_", min=0, max=32)
        return io.Schema(
            node_id="TuringUtilsReferenceAudioHub",
            display_name="Reference Audio Hub",
            category="Turing Utils/references",
            inputs=[
                io.Autogrow.Input("audios", template=template, optional=True),
                AudioReferences.Input("previous", optional=True),
            ],
            outputs=[AudioReferences.Output("references")],
        )

    @classmethod
    def execute(cls, audios=None, previous=None):
        items = list(previous.items) if isinstance(previous, AudioReferenceSet) else []
        items.extend((audios or {}).values())
        return io.NodeOutput(AudioReferenceSet(tuple(items)))
