from __future__ import annotations

from .nodes.attention import AttentionKernelTuningPatch, SolSparseAttentionPatch
from .nodes.bernini import BerniniContextWindowsCore, BerniniInpaintCondition
from .nodes.loaders import ConvRotCLIPLoader, ConvRotDiffusionModelLoader
from .nodes.minimax import (
    H3ConcatAVLatent,
    H3SeparateAVLatent,
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
    "TuringUtilsH3ConcatAVLatent": H3ConcatAVLatent,
    "TuringUtilsH3SeparateAVLatent": H3SeparateAVLatent,
    "TuringUtilsMiniMaxH3LatentResize": MiniMaxH3LatentResize,
    "TuringUtilsMiniMaxH3ProgressiveResolutionPatch": MiniMaxH3ProgressiveResolutionPatch,
    "TuringUtilsSolSparseAttentionPatch": SolSparseAttentionPatch,
    "TuringUtilsAttentionKernelTuningPatch": AttentionKernelTuningPatch,
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
    "TuringUtilsH3ConcatAVLatent": "H3 Concat AV Latent",
    "TuringUtilsH3SeparateAVLatent": "H3 Separate AV Latent",
    "TuringUtilsMiniMaxH3LatentResize": "Resize MiniMax H3 AV Latent (Experimental)",
    "TuringUtilsMiniMaxH3ProgressiveResolutionPatch": "Patch H3 Progressive Resolution (Experimental)",
    "TuringUtilsSolSparseAttentionPatch": "Patch Sol Sparse Attention (Experimental)",
    "TuringUtilsAttentionKernelTuningPatch": "Patch Turing Attention Kernel Tuning (Experimental)",
}
