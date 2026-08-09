"""Thin ComfyUI nodes for optional resize and reference hubs."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils
from comfy_api.latest import io

from ..media.references import (
    AudioReferenceSet,
    AudioReferences,
    ImageReferenceSet,
    ImageReferences,
    VideoOptions,
    VideoReferenceSet,
    VideoReferences,
    _crop_offsets,
    _pad_resized_image,
    _resize_image,
    _spatial_inputs,
    _spatial_options,
    _validate_image,
)


class OptionalResizeImageV2:
    """KJ Resize Image v2-compatible transform with an optional IMAGE input."""

    UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "nvidia_rtx_vsr"]
    KEEP_PROPORTION = [
        "stretch", "resize", "pad", "pad_edge", "pad_edge_pixel",
        "crop", "pillarbox_blur", "total_pixels",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1}),
                "upscale_method": (cls.UPSCALE_METHODS,),
                "keep_proportion": (cls.KEEP_PROPORTION, {"default": "stretch"}),
                "pad_color": ("STRING", {"default": "0, 0, 0", "tooltip": "Color to use for padding."}),
                "crop_position": (
                    ["center", "top", "bottom", "left", "right"],
                    {"default": "center"},
                ),
                "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "device": (["cpu", "gpu"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK")
    RETURN_NAMES = ("IMAGE", "width", "height", "mask")
    FUNCTION = "resize"
    CATEGORY = "Turing Utils/image"
    TITLE = "Optional Resize Image v2"
    DESCRIPTION = (
        "Resize Image v2-compatible image and mask transform. If IMAGE is not connected, "
        "the IMAGE output is None so downstream optional first/last-frame inputs stay absent."
    )

    @staticmethod
    def _resize_mask(mask, width, height, method):
        expanded = mask.unsqueeze(1)
        if method == "lanczos":
            expanded = expanded.repeat(1, 3, 1, 1)
            return comfy.utils.common_upscale(
                expanded, width, height, method, "disabled"
            ).movedim(1, -1)[..., 0]
        return comfy.utils.common_upscale(
            expanded, width, height, method, "disabled"
        ).squeeze(1)

    @staticmethod
    def _pad_offsets(extra_width, extra_height, position):
        if position == "top":
            return extra_width // 2, extra_width - extra_width // 2, 0, extra_height
        if position == "bottom":
            return extra_width // 2, extra_width - extra_width // 2, extra_height, 0
        if position == "left":
            return 0, extra_width, extra_height // 2, extra_height - extra_height // 2
        if position == "right":
            return extra_width, 0, extra_height // 2, extra_height - extra_height // 2
        left = extra_width // 2
        top = extra_height // 2
        return left, extra_width - left, top, extra_height - top

    def resize(
        self,
        width,
        height,
        upscale_method,
        keep_proportion,
        pad_color,
        crop_position,
        divisible_by,
        image=None,
        mask=None,
        device="cpu",
    ):
        if image is None:
            return (None, 0, 0, None)
        if not torch.is_tensor(image) or image.ndim != 4 or int(image.shape[-1]) < 1:
            shape = tuple(image.shape) if torch.is_tensor(image) else type(image).__name__
            raise ValueError(f"image must be an IMAGE tensor [B,H,W,C], got {shape}")
        if min(int(image.shape[0]), int(image.shape[1]), int(image.shape[2])) < 1:
            raise ValueError("image must not be empty")
        batch, source_height, source_width, _ = image.shape
        source_height = int(source_height)
        source_width = int(source_width)

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim != 3:
                raise ValueError(f"Expected MASK [B,H,W], got {tuple(mask.shape)}")
            if tuple(mask.shape[-2:]) == (64, 64) and (source_height, source_width) != (64, 64):
                mask = None
            elif tuple(mask.shape[-2:]) != (source_height, source_width):
                mask = self._resize_mask(mask, source_width, source_height, "bilinear")

        if device == "gpu":
            if upscale_method == "lanczos":
                raise ValueError("Lanczos is not supported on the GPU")
            target_device = comfy.model_management.get_torch_device()
        elif device == "cpu":
            target_device = torch.device("cpu")
        else:
            raise ValueError(f"device must be cpu or gpu, got {device}")

        requested_width = int(width)
        requested_height = int(height)
        width = requested_width
        height = requested_height
        proportional = (
            keep_proportion in ("resize", "total_pixels")
            or str(keep_proportion).startswith("pad")
            or keep_proportion == "pillarbox_blur"
        )

        pad_left = pad_right = pad_top = pad_bottom = 0
        if proportional:
            if keep_proportion == "total_pixels":
                total_pixels = width * height
                if total_pixels <= 0:
                    raise ValueError("total_pixels mode requires positive width and height")
                aspect = source_width / source_height
                height = int(math.sqrt(total_pixels / aspect))
                width = int(math.sqrt(total_pixels * aspect))
            elif width == 0 and height == 0:
                width, height = source_width, source_height
            elif width == 0:
                width = round(source_width * height / source_height)
            elif height == 0:
                height = round(source_height * width / source_width)
            else:
                ratio = min(width / source_width, height / source_height)
                width = round(source_width * ratio)
                height = round(source_height * ratio)

            if str(keep_proportion).startswith("pad") or keep_proportion == "pillarbox_blur":
                canvas_width = requested_width if requested_width > 0 else width
                canvas_height = requested_height if requested_height > 0 else height
                pad_left, pad_right, pad_top, pad_bottom = self._pad_offsets(
                    canvas_width - width,
                    canvas_height - height,
                    crop_position,
                )
        else:
            width = source_width if width == 0 else width
            height = source_height if height == 0 else height

        divisible_by = int(divisible_by)
        if divisible_by > 1:
            width -= width % divisible_by
            height -= height % divisible_by
        if width < 1 or height < 1:
            raise ValueError(f"Resize Image v2 produced invalid output dimensions {width}x{height}")

        output = image.to(target_device)
        output_mask = mask.to(target_device) if mask is not None else None
        if keep_proportion == "crop":
            source_aspect = source_width / source_height
            target_aspect = width / height
            if source_aspect > target_aspect:
                crop_width = round(source_height * target_aspect)
                crop_height = source_height
            else:
                crop_width = source_width
                crop_height = round(source_width / target_aspect)
            x, y = _crop_offsets(
                source_width - crop_width,
                source_height - crop_height,
                crop_position,
            )
            output = output[:, y:y + crop_height, x:x + crop_width]
            if output_mask is not None:
                output_mask = output_mask[:, y:y + crop_height, x:x + crop_width]

        output = _resize_image(output, width, height, upscale_method)
        if output_mask is not None:
            mask_method = "bilinear" if upscale_method == "nvidia_rtx_vsr" else upscale_method
            output_mask = self._resize_mask(output_mask, width, height, mask_method)

        pad_mode = None
        if str(keep_proportion).startswith("pad") or keep_proportion == "pillarbox_blur":
            if divisible_by > 1:
                padded_width = width + pad_left + pad_right
                padded_height = height + pad_top + pad_bottom
                if padded_width % divisible_by:
                    pad_right += divisible_by - padded_width % divisible_by
                if padded_height % divisible_by:
                    pad_bottom += divisible_by - padded_height % divisible_by
            pad_mode = keep_proportion
            output = _pad_resized_image(
                output,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
                pad_mode,
                pad_color,
            )
            if output_mask is not None:
                output_mask = F.pad(
                    output_mask,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="replicate",
                )
            else:
                output_mask = torch.ones(
                    (batch, int(output.shape[1]), int(output.shape[2])),
                    dtype=output.dtype,
                    device=output.device,
                )
                output_mask[:, pad_top:pad_top + height, pad_left:pad_left + width] = 0.0

        output = output.cpu()
        if output_mask is None:
            output_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
        else:
            output_mask = output_mask.cpu()
        return (output, int(output.shape[2]), int(output.shape[1]), output_mask)


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
