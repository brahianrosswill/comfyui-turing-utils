"""Compatibility alias for :mod:`comfyui_turing_utils.precision`."""

import sys
try:
    from .comfyui_turing_utils import precision as _implementation
except ImportError:
    from comfyui_turing_utils import precision as _implementation

sys.modules[__name__] = _implementation
