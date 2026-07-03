from __future__ import annotations

from .bernini_nodes import BerniniContextWindowsCore, BerniniPadVideoLength
from .dreamidv_nodes import DreamIDVConditioning, DreamIDVDiTLoader
from .svdint4_nodes import SVDInt4DiffusionModelLoader


NODE_CLASS_MAPPINGS = {
    "SVDInt4DiffusionModelLoader": SVDInt4DiffusionModelLoader,
    "SVDInt4DreamIDVDiTLoader": DreamIDVDiTLoader,
    "SVDInt4DreamIDVConditioning": DreamIDVConditioning,
    "SVDInt4BerniniPadVideoLength": BerniniPadVideoLength,
    "SVDInt4BerniniContextWindowsCore": BerniniContextWindowsCore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVDInt4DiffusionModelLoader": "Load SVDInt4 DiT",
    "SVDInt4DreamIDVDiTLoader": "Load DreamID-V DiT",
    "SVDInt4DreamIDVConditioning": "DreamID-V Conditioning",
    "SVDInt4BerniniPadVideoLength": "Bernini Pad Video Length",
    "SVDInt4BerniniContextWindowsCore": "Bernini Context Windows",
}
