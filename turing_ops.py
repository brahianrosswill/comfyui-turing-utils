"""Compatibility alias for :mod:`comfyui_turing_utils.quantization.dispatch`."""

import sys
try:
    from .comfyui_turing_utils.quantization import dispatch as _implementation
except ImportError:
    from comfyui_turing_utils.quantization import dispatch as _implementation

sys.modules[__name__] = _implementation
