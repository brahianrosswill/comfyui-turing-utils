"""Compatibility alias for :mod:`comfyui_turing_utils.adapters.wan`."""

import sys
try:
    from .comfyui_turing_utils.adapters import wan as _implementation
except ImportError:
    from comfyui_turing_utils.adapters import wan as _implementation

sys.modules[__name__] = _implementation
