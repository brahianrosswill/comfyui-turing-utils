from __future__ import annotations

from .nodes.attention import FrameSparseAttentionPatch, SolSparseAttentionPatch
from .nodes.bernini import BerniniContextWindowsCore, BerniniInpaintCondition
from .nodes.loaders import ConvRotCLIPLoader, ConvRotDiffusionModelLoader
from .nodes.minimax import (
    MiniMaxH3LatentResize,
    MiniMaxH3ProgressiveResolutionPatch,
    MiniMaxH3ReferenceConditionHub,
    MiniMaxH3VideoFramesPadding,
)
from .nodes.references import OptionalResizeImageV2, ReferenceAudioHub, ReferenceImageHub, ReferenceVideoHub
from .nodes.wan import WanVideoFramesPadding


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
    "TuringUtilsOptionalResizeImageV2": OptionalResizeImageV2,
    "TuringUtilsMiniMaxH3ReferenceConditionHub": MiniMaxH3ReferenceConditionHub,
    "TuringUtilsMiniMaxH3LatentResize": MiniMaxH3LatentResize,
    "TuringUtilsMiniMaxH3ProgressiveResolutionPatch": MiniMaxH3ProgressiveResolutionPatch,
    "TuringUtilsSolSparseAttentionPatch": SolSparseAttentionPatch,
    "TuringUtilsFrameSparseAttentionPatch": FrameSparseAttentionPatch,
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
    "TuringUtilsOptionalResizeImageV2": "Optional Resize Image v2",
    "TuringUtilsMiniMaxH3ReferenceConditionHub": "MiniMax H3 Reference Condition (Hub)",
    "TuringUtilsMiniMaxH3LatentResize": "Resize MiniMax H3 AV Latent (Experimental)",
    "TuringUtilsMiniMaxH3ProgressiveResolutionPatch": "Patch H3 Progressive Resolution (Experimental)",
    "TuringUtilsSolSparseAttentionPatch": "Patch Sol Sparse Attention (Experimental)",
    "TuringUtilsFrameSparseAttentionPatch": "Patch Sage Frame Sparse Attention (Experimental)",
}
