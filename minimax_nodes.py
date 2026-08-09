"""Compatibility alias for :mod:`comfyui_turing_utils.nodes.minimax`."""

import sys
try:
    from .comfyui_turing_utils.nodes import minimax as _implementation
except ImportError:
    from comfyui_turing_utils.nodes import minimax as _implementation

sys.modules[__name__] = _implementation
