"""Compatibility alias for the MiniMax acceleration implementation."""

import sys
try:
    from .comfyui_turing_utils.adapters.minimax import acceleration as _implementation
except ImportError:
    from comfyui_turing_utils.adapters.minimax import acceleration as _implementation

sys.modules[__name__] = _implementation
