from __future__ import annotations

from .bernini_nodes import BerniniContextWindowsCore
from .dreamidv_nodes import DreamIDVConditioning, DreamIDVDiTLoader
from .seedvr2_nodes import SeedVR2Upscaler
from .svdint4_nodes import SVDInt4DiffusionModelLoader
from .wan_nodes import WanVideoFramesPadding


NODE_CLASS_MAPPINGS = {
    "SVDInt4DiffusionModelLoader": SVDInt4DiffusionModelLoader,
    "SVDInt4DreamIDVDiTLoader": DreamIDVDiTLoader,
    "SVDInt4DreamIDVConditioning": DreamIDVConditioning,
    "SVDInt4WanVideoFramesPadding": WanVideoFramesPadding,
    "SVDInt4BerniniContextWindowsCore": BerniniContextWindowsCore,
    "SVDInt4SeedVR2Upscaler": SeedVR2Upscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVDInt4DiffusionModelLoader": "Load SVDInt4 DiT",
    "SVDInt4DreamIDVDiTLoader": "Load DreamID-V DiT",
    "SVDInt4DreamIDVConditioning": "DreamID-V Conditioning",
    "SVDInt4WanVideoFramesPadding": "Wan Video Frames Padding",
    "SVDInt4BerniniContextWindowsCore": "Bernini Context Windows",
    "SVDInt4SeedVR2Upscaler": "SeedVR2 Upscale (EXPERIMENTAL / UNUSABLE)",
}
