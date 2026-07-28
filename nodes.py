from __future__ import annotations

from .bernini_nodes import BerniniContextWindowsCore
from .convrot_nodes import ConvRotCLIPLoader, ConvRotDiffusionModelLoader
from .svdint4_nodes import SVDInt4DiffusionModelLoader
from .wan_nodes import WanVideoFramesPadding


NODE_CLASS_MAPPINGS = {
    "SVDInt4ConvRotDiffusionModelLoader": ConvRotDiffusionModelLoader,
    "SVDInt4ConvRotCLIPLoader": ConvRotCLIPLoader,
    "SVDInt4DiffusionModelLoader": SVDInt4DiffusionModelLoader,
    "SVDInt4WanVideoFramesPadding": WanVideoFramesPadding,
    "SVDInt4BerniniContextWindowsCore": BerniniContextWindowsCore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVDInt4ConvRotDiffusionModelLoader": "Load ConvRot DiT",
    "SVDInt4ConvRotCLIPLoader": "Load ConvRot CLIP",
    "SVDInt4DiffusionModelLoader": "Load SVDInt4 DiT",
    "SVDInt4WanVideoFramesPadding": "Wan Video Frames Padding",
    "SVDInt4BerniniContextWindowsCore": "Bernini Context Windows",
}
