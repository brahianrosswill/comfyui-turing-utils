"""Compatibility alias for :mod:`comfyui_turing_utils.nodes.attention`."""

import sys
try:
    from .comfyui_turing_utils.nodes import attention as _implementation
except ImportError:
    from comfyui_turing_utils.nodes import attention as _implementation

sys.modules[__name__] = _implementation
