"""Compatibility alias for :mod:`comfyui_turing_utils.adapters.bernini`."""

import sys
try:
    from .comfyui_turing_utils.adapters import bernini as _implementation
    from .comfyui_turing_utils.nodes.bernini import BerniniContextWindowsCore, BerniniInpaintCondition
except ImportError:
    from comfyui_turing_utils.adapters import bernini as _implementation
    from comfyui_turing_utils.nodes.bernini import BerniniContextWindowsCore, BerniniInpaintCondition

_implementation.BerniniContextWindowsCore = BerniniContextWindowsCore
_implementation.BerniniInpaintCondition = BerniniInpaintCondition

sys.modules[__name__] = _implementation
