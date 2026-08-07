from __future__ import annotations

from .bernini_nodes import BerniniContextWindowsCore, BerniniInpaintCondition
from .convrot_nodes import ConvRotCLIPLoader, ConvRotDiffusionModelLoader
from .minimax_nodes import MiniMaxH3ReferenceConditionHub, MiniMaxH3VideoFramesPadding
from .reference_nodes import ReferenceAudioHub, ReferenceImageHub, ReferenceVideoHub
from .wan_nodes import WanVideoFramesPadding


NODE_CLASS_MAPPINGS = {
    "TuringUtilsConvRotDiffusionModelLoader": ConvRotDiffusionModelLoader,
    "TuringUtilsConvRotCLIPLoader": ConvRotCLIPLoader,
    "TuringUtilsWanVideoFramesPadding": WanVideoFramesPadding,
    "TuringUtilsMiniMaxH3VideoFramesPadding": MiniMaxH3VideoFramesPadding,
    "TuringUtilsBerniniContextWindowsCore": BerniniContextWindowsCore,
    "TuringUtilsBerniniInpaintCondition": BerniniInpaintCondition,
    "TuringUtilsReferenceImageHub": ReferenceImageHub,
    "TuringUtilsReferenceVideoHub": ReferenceVideoHub,
    "TuringUtilsReferenceAudioHub": ReferenceAudioHub,
    "TuringUtilsMiniMaxH3ReferenceConditionHub": MiniMaxH3ReferenceConditionHub,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TuringUtilsConvRotDiffusionModelLoader": "Load ConvRot DiT",
    "TuringUtilsConvRotCLIPLoader": "Load ConvRot CLIP",
    "TuringUtilsWanVideoFramesPadding": "Wan Video Frames Padding",
    "TuringUtilsMiniMaxH3VideoFramesPadding": "MiniMax H3 Video Frames Padding",
    "TuringUtilsBerniniContextWindowsCore": "Bernini Context Windows",
    "TuringUtilsBerniniInpaintCondition": "Bernini Inpaint Condition",
    "TuringUtilsReferenceImageHub": "Reference Image Hub",
    "TuringUtilsReferenceVideoHub": "Reference Video Hub",
    "TuringUtilsReferenceAudioHub": "Reference Audio Hub",
    "TuringUtilsMiniMaxH3ReferenceConditionHub": "MiniMax H3 Reference Condition (Hub)",
}
