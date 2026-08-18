"""Hardware capability predicates shared by all Turing features."""

from __future__ import annotations

import torch


_TURING_WITHOUT_TENSOR_CORES = (
    "GTX 1630",
    "GTX 1650",
    "GTX 1660",
    "T500",
    "T550",
    "T600",
    "MX450",
    "MX550",
    "CMP 30HX",
    "T1000",
    "T1200",
    "T2000",
)


def is_supported_turing_device(device: torch.device) -> bool:
    """Return whether ``device`` is exact sm75 with usable Tensor Cores."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    if torch.cuda.get_device_capability(index) != (7, 5):
        return False
    name = torch.cuda.get_device_name(index)
    return not any(model in name for model in _TURING_WITHOUT_TENSOR_CORES)


def is_supported_attention_device(device: torch.device) -> bool:
    """Return whether bundled integer attention can target ``device``."""
    if is_supported_turing_device(device):
        return True
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(index) >= (8, 0)


__all__ = ["is_supported_attention_device", "is_supported_turing_device"]
