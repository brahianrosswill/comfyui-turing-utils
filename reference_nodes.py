from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils
from comfy_api.latest import io


ImageReferences = io.Custom("TURING_REFERENCE_IMAGES")
VideoReferences = io.Custom("TURING_REFERENCE_VIDEOS")
AudioReferences = io.Custom("TURING_REFERENCE_AUDIOS")


@dataclass(frozen=True)
class SpatialOptions:
    width: int = 0
    height: int = 0
    upscale_method: str = "nearest-exact"
    keep_proportion: str = "stretch"
    pad_color: str = "0, 0, 0"
    crop_position: str = "center"
    divisible_by: int = 2
    device: str = "cpu"


@dataclass(frozen=True)
class VideoOptions:
    spatial: SpatialOptions = SpatialOptions()
    frame_count: int = 0
    short_video_fill: str = "repeat_last"


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
        target = int(self.options.frame_count)
        if target <= 0 or not videos:
            return tuple(videos)
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
    except ValueError:
        try:
            from PIL import ImageColor

            values = list(ImageColor.getrgb(value.strip()))
        except (ImportError, ValueError) as error:
            raise ValueError(
                "pad_color must be a color name, hex color, or comma-separated numeric values"
            ) from error
    if not values:
        values = [0.0]
    if len(values) == 1:
        values *= channels
    if len(values) < channels:
        raise ValueError(f"pad_color needs either one value or at least {channels} values")
    values = values[:channels]
    if max(abs(item) for item in values) > 1.0:
        values = [item / 255.0 for item in values]
    return torch.tensor(values, device=device, dtype=dtype)


def _resize_image(image: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
    if method == "nvidia_rtx_vsr":
        try:
            import nvvfx
        except ImportError as error:
            raise ImportError(
                "NVIDIA RTX Video Super Resolution requires the optional nvidia-vfx package"
            ) from error

        output_width = max(8, round(width / 8) * 8)
        output_height = max(8, round(height / 8) * 8)
        frames = image.movedim(-1, 1).cuda().contiguous()
        resized = []
        with nvvfx.VideoSuperRes(nvvfx.effects.QualityLevel.ULTRA) as effect:
            effect.output_width = output_width
            effect.output_height = output_height
            effect.load()
            for frame in frames:
                resized.append(torch.from_dlpack(effect.run(frame).image).clone())
        return torch.stack(resized, dim=0).movedim(1, -1)
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


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return image
    original_dtype = image.dtype
    work = image.float()
    radius = max(1, int(3.0 * sigma))
    points = torch.arange(-radius, radius + 1, device=work.device, dtype=work.dtype)
    kernel = torch.exp(-(points * points) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = int(work.shape[1])
    horizontal = kernel.reshape(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.reshape(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    work = F.conv2d(work, horizontal, padding=(0, radius), groups=channels)
    work = F.conv2d(work, vertical, padding=(radius, 0), groups=channels)
    return work.to(original_dtype)


def _pad_resized_image(
    image: torch.Tensor,
    pad_left: int,
    pad_right: int,
    pad_top: int,
    pad_bottom: int,
    mode: str,
    color: str,
) -> torch.Tensor:
    if not any((pad_left, pad_right, pad_top, pad_bottom)):
        return image

    batch, height, width, channels = image.shape
    padded_height = height + pad_top + pad_bottom
    padded_width = width + pad_left + pad_right
    if mode == "pad_edge_pixel":
        return F.pad(
            image.movedim(-1, 1),
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="replicate",
        ).movedim(1, -1)

    if mode == "pillarbox_blur":
        scale = max(padded_width / width, padded_height / height)
        background_width = max(round(width * scale), 1)
        background_height = max(round(height * scale), 1)
        background = _resize_image(image, background_width, background_height, "bilinear")
        x, y = _crop_offsets(
            background_width - padded_width,
            background_height - padded_height,
            "center",
        )
        background = background[:, y:y + padded_height, x:x + padded_width]
        background = _gaussian_blur(
            background.movedim(-1, 1),
            max(1.0, 0.006 * min(padded_height, padded_width)),
        ).movedim(1, -1)
        if channels >= 3:
            rgb = background[..., :3]
            luma = 0.2126 * rgb[..., :1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]
            background[..., :3] = rgb * 0.8 + luma * 0.2
        canvas = (background * 0.35).clamp_(0.0, 1.0)
    else:
        fill = _parse_pad_color(color, channels, image.device, image.dtype)
        canvas = fill.reshape(1, 1, 1, channels).expand(
            batch, padded_height, padded_width, channels
        ).clone()
        if mode == "pad_edge":
            canvas[:, :pad_top] = image[:, :1].mean(dim=2, keepdim=True)
            canvas[:, pad_top + height:] = image[:, -1:].mean(dim=2, keepdim=True)
            canvas[:, :, :pad_left] = image[:, :, :1].mean(dim=1, keepdim=True)
            canvas[:, :, pad_left + width:] = image[:, :, -1:].mean(dim=1, keepdim=True)

    canvas[:, pad_top:pad_top + height, pad_left:pad_left + width] = image
    return canvas


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

    source_height, source_width = int(image.shape[1]), int(image.shape[2])
    width = int(options.width)
    height = int(options.height)
    if width <= 0 or height <= 0:
        return (image, mask) if mask is not None else image

    mode = str(options.keep_proportion)
    supported_modes = {
        "stretch", "resize", "pad", "pad_edge", "pad_edge_pixel",
        "crop", "pillarbox_blur", "total_pixels",
    }
    if mode not in supported_modes:
        raise ValueError(f"unsupported keep_proportion mode: {mode}")
    if str(options.device) not in ("cpu", "gpu"):
        raise ValueError(f"device must be cpu or gpu, got {options.device}")
    if options.device == "gpu" and options.upscale_method == "lanczos":
        raise ValueError("Lanczos is not supported on the GPU")
    target_device = (
        comfy.model_management.get_torch_device()
        if options.device == "gpu"
        else torch.device("cpu")
    )
    image = image.to(target_device)
    if mask is not None:
        mask = mask.to(target_device)

    pad_left = pad_right = pad_top = pad_bottom = 0
    proportional = mode in ("resize", "total_pixels", "pad", "pad_edge", "pad_edge_pixel", "pillarbox_blur")
    if proportional:
        if mode == "total_pixels":
            total_pixels = width * height
            if total_pixels <= 0:
                raise ValueError("total_pixels mode requires positive width and height")
            aspect_ratio = source_width / source_height
            resized_height = int((total_pixels / aspect_ratio) ** 0.5)
            resized_width = int((total_pixels * aspect_ratio) ** 0.5)
        elif width == 0 and height == 0:
            resized_width, resized_height = source_width, source_height
        elif width == 0:
            resized_height = height
            resized_width = round(source_width * height / source_height)
        elif height == 0:
            resized_width = width
            resized_height = round(source_height * width / source_width)
        else:
            ratio = min(width / source_width, height / source_height)
            resized_width = round(source_width * ratio)
            resized_height = round(source_height * ratio)

        if mode in ("pad", "pad_edge", "pad_edge_pixel", "pillarbox_blur"):
            target_width = resized_width if width == 0 else width
            target_height = resized_height if height == 0 else height
            extra_width = target_width - resized_width
            extra_height = target_height - resized_height
            pad_left, pad_top = _crop_offsets(extra_width, extra_height, options.crop_position)
            pad_right = extra_width - pad_left
            pad_bottom = extra_height - pad_top
    else:
        resized_width = source_width if width == 0 else width
        resized_height = source_height if height == 0 else height

    divisible_by = int(options.divisible_by)
    if divisible_by > 1:
        resized_width -= resized_width % divisible_by
        resized_height -= resized_height % divisible_by
    if resized_width < 1 or resized_height < 1:
        raise ValueError(
            f"resize produced an invalid {resized_width}x{resized_height} target; "
            "increase width/height or lower divisible_by"
        )

    if mode == "crop":
        old_aspect = source_width / source_height
        new_aspect = resized_width / resized_height
        if old_aspect > new_aspect:
            crop_width = round(source_height * new_aspect)
            crop_height = source_height
        else:
            crop_width = source_width
            crop_height = round(source_width / new_aspect)
        x, y = _crop_offsets(source_width - crop_width, source_height - crop_height, options.crop_position)
        image = image[:, y:y + crop_height, x:x + crop_width]
        if mask is not None:
            mask = mask[:, y:y + crop_height, x:x + crop_width]

    output = _resize_image(image, resized_width, resized_height, options.upscale_method)
    if mask is not None:
        mask = _conservative_resize_mask(mask, resized_width, resized_height).squeeze(1)
    if mode in ("pad", "pad_edge", "pad_edge_pixel", "pillarbox_blur"):
        output = _pad_resized_image(
            output,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            mode,
            options.pad_color,
        )
        if mask is not None and any((pad_left, pad_right, pad_top, pad_bottom)):
            mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

    output = output.cpu()
    if mask is not None:
        mask = mask.cpu()
    return (output, mask) if mask is not None else output


def _align_video_frames(video: torch.Tensor, target: int, options: VideoOptions) -> torch.Tensor:
    length = int(video.shape[0])
    if length == target:
        return video
    if length > target:
        return video[:target]

    pad_count = target - length
    if options.short_video_fill == "black":
        padding = video.new_zeros((pad_count, *video.shape[1:]))
    else:
        padding = video[-1:].repeat(pad_count, 1, 1, 1)
    return torch.cat((video, padding), dim=0)


def _spatial_inputs() -> list:
    return [
        io.Int.Input(
            "width",
            default=0,
            min=0,
            max=16384,
            step=1,
            tooltip="Set width or height to 0 to disable spatial resizing.",
        ),
        io.Int.Input(
            "height",
            default=0,
            min=0,
            max=16384,
            step=1,
            tooltip="Set width or height to 0 to disable spatial resizing.",
        ),
        io.Combo.Input(
            "upscale_method",
            options=["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "nvidia_rtx_vsr"],
            default="nearest-exact",
        ),
        io.Combo.Input(
            "keep_proportion",
            options=["stretch", "resize", "pad", "pad_edge", "pad_edge_pixel", "crop", "pillarbox_blur", "total_pixels"],
            default="stretch",
        ),
        io.String.Input("pad_color", default="0, 0, 0", tooltip="Color to use for padding."),
        io.Combo.Input("crop_position", options=["center", "top", "bottom", "left", "right"], default="center"),
        io.Int.Input("divisible_by", default=2, min=0, max=512, step=1),
        io.Combo.Input("device", options=["cpu", "gpu"], default="cpu", optional=True),
    ]


def _spatial_options(
    width,
    height,
    upscale_method,
    keep_proportion,
    pad_color,
    crop_position,
    divisible_by,
    device="cpu",
):
    return SpatialOptions(
        width=int(width),
        height=int(height),
        upscale_method=str(upscale_method),
        keep_proportion=str(keep_proportion),
        pad_color=str(pad_color),
        crop_position=str(crop_position),
        divisible_by=int(divisible_by),
        device=str(device),
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
                ImageReferences.Input("previous", optional=True),
                io.Autogrow.Input("images", template=template, optional=True),
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
                VideoReferences.Input("previous", optional=True),
                io.Autogrow.Input("videos", template=template, optional=True),
                *_spatial_inputs(),
                io.Int.Input(
                    "frame_count",
                    default=0,
                    min=0,
                    max=16385,
                    step=1,
                    tooltip="0 keeps every video's original length; positive values trim or pad at the end.",
                ),
                io.Combo.Input("short_video_fill", options=["repeat_last", "black"], default="repeat_last"),
            ],
            outputs=[VideoReferences.Output("references")],
        )

    @classmethod
    def execute(cls, videos=None, previous=None, frame_count=0, short_video_fill="repeat_last", **kwargs):
        items = list(previous.items) if isinstance(previous, VideoReferenceSet) else []
        for video in (videos or {}).values():
            items.append(_validate_image(video, "video reference"))
        options = VideoOptions(
            spatial=_spatial_options(**kwargs),
            frame_count=int(frame_count),
            short_video_fill=str(short_video_fill),
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
                AudioReferences.Input("previous", optional=True),
                io.Autogrow.Input("audios", template=template, optional=True),
            ],
            outputs=[AudioReferences.Output("references")],
        )

    @classmethod
    def execute(cls, audios=None, previous=None):
        items = list(previous.items) if isinstance(previous, AudioReferenceSet) else []
        items.extend((audios or {}).values())
        return io.NodeOutput(AudioReferenceSet(tuple(items)))
