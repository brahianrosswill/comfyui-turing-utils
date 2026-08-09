"""Compatibility alias for :mod:`comfyui_turing_utils.quantization.fusions`."""

import sys
try:
    from .comfyui_turing_utils.quantization import fusions as _implementation
except ImportError:
    from comfyui_turing_utils.quantization import fusions as _implementation

sys.modules[__name__] = _implementation
