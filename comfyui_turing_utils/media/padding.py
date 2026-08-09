"""Shared tensor padding helpers for video-oriented model nodes."""

from __future__ import annotations

import torch


def repeat_last_frame(tensor: torch.Tensor, count: int) -> torch.Tensor:
    """Append ``count`` copies of the final item along the first dimension."""
    count = int(count)
    if count <= 0:
        return tensor
    if tensor.shape[0] < 1:
        raise ValueError("Cannot repeat the final item of an empty tensor")
    tail = tensor[-1:].repeat(count, *([1] * (tensor.ndim - 1)))
    return torch.cat((tensor, tail), dim=0)


__all__ = ["repeat_last_frame"]
