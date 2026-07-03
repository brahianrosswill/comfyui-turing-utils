from __future__ import annotations

import torch


def _is_wan_frame_count(frame_count: int) -> bool:
    return frame_count >= 1 and (frame_count - 1) % 4 == 0


def _ceil_wan_frame_count(frame_count: int) -> int:
    if frame_count < 1:
        raise ValueError("Video must contain at least one frame.")
    return ((frame_count - 1 + 3) // 4) * 4 + 1


def _pad_first_dim(tensor: torch.Tensor, pad_count: int) -> torch.Tensor:
    if pad_count <= 0:
        return tensor
    tail = tensor[-1:].repeat(pad_count, *([1] * (tensor.ndim - 1)))
    return torch.cat((tensor, tail), dim=0)


class WanVideoFramesPadding:
    @classmethod
    def INPUT_TYPES(cls):
        target_frame_count = (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 16385,
                "step": 1,
                "tooltip": (
                    "Target real-frame count. Use 0 to round the input length up "
                    "to the next 4*n+1 frame count."
                ),
            },
        )
        return {
            "required": {
                "image": ("IMAGE",),
                "target_frame_count": target_frame_count,
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height", "length", "input_length")
    FUNCTION = "pad"
    CATEGORY = "SVDInt4/video"
    TITLE = "Wan Video Frames Padding"

    def pad(self, image, target_frame_count: int, mask=None):
        if image.ndim < 3:
            raise ValueError(f"Expected IMAGE tensor shaped [frames, height, width, channels], got {tuple(image.shape)}.")
        frame_count = int(image.shape[0])
        if frame_count < 1:
            raise ValueError("Wan Video Frames Padding requires at least one input frame.")
        height = int(image.shape[1])
        width = int(image.shape[2])

        mask = self._validate_mask(mask, frame_count, height, width)
        output_length = self._target_length(frame_count, target_frame_count)
        pad_count = output_length - frame_count

        image = _pad_first_dim(image, pad_count)
        if mask is not None:
            mask = _pad_first_dim(mask, pad_count)
        return (image, mask, width, height, output_length, frame_count)

    @staticmethod
    def _target_length(frame_count: int, target_frame_count: int) -> int:
        if target_frame_count == 0:
            return _ceil_wan_frame_count(frame_count)

        output_length = int(target_frame_count)
        if not _is_wan_frame_count(output_length):
            raise ValueError(f"target_frame_count must be 0 or 4*n+1; got {target_frame_count}.")
        if output_length < frame_count:
            raise ValueError(
                f"target_frame_count={output_length} is shorter than the input video "
                f"({frame_count} frames). This node only pads; trim upstream if needed."
            )
        return output_length

    @staticmethod
    def _validate_mask(mask, frame_count: int, height: int, width: int):
        if mask is None:
            return None
        if mask.ndim == 2 and frame_count == 1:
            mask = mask.unsqueeze(0)
        if mask.ndim < 3:
            raise ValueError(f"Expected MASK tensor shaped [frames, height, width], got {tuple(mask.shape)}.")
        if int(mask.shape[0]) != frame_count:
            raise ValueError(
                f"MASK frame count must match IMAGE frame count before padding; "
                f"got mask={int(mask.shape[0])}, image={frame_count}."
            )
        if int(mask.shape[-2]) != height or int(mask.shape[-1]) != width:
            raise ValueError(
                f"MASK spatial size must match IMAGE before padding; got mask="
                f"{int(mask.shape[-1])}x{int(mask.shape[-2])}, image={width}x{height}."
            )
        return mask
