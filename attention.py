"""Compatibility alias for :mod:`comfyui_turing_utils.attention.api`."""

import sys
try:
    from .comfyui_turing_utils.attention import api as _implementation
except ImportError:
    from comfyui_turing_utils.attention import api as _implementation

sys.modules[__name__] = _implementation
