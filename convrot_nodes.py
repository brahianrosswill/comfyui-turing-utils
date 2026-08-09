"""Compatibility alias for :mod:`comfyui_turing_utils.quantization.convrot`."""

import sys
try:
    from .comfyui_turing_utils.quantization import convrot as _implementation
    from .comfyui_turing_utils.nodes.loaders import ConvRotCLIPLoader, ConvRotDiffusionModelLoader
except ImportError:
    from comfyui_turing_utils.quantization import convrot as _implementation
    from comfyui_turing_utils.nodes.loaders import ConvRotCLIPLoader, ConvRotDiffusionModelLoader

_implementation.ConvRotCLIPLoader = ConvRotCLIPLoader
_implementation.ConvRotDiffusionModelLoader = ConvRotDiffusionModelLoader

sys.modules[__name__] = _implementation
