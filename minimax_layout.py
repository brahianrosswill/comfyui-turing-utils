"""Compatibility alias for the MiniMax attention layout provider."""

import sys
try:
    from .comfyui_turing_utils.adapters.minimax import layout as _implementation
except ImportError:
    from comfyui_turing_utils.adapters.minimax import layout as _implementation

sys.modules[__name__] = _implementation
