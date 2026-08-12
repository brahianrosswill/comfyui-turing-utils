from __future__ import annotations

from .nodes.attention import AttentionKernelTuningPatch, SolSparseAttentionPatch
from .nodes.bernini import BerniniContextWindowsCore, BerniniInpaintCondition
from .nodes.loaders import ConvRotCLIPLoader, ConvRotDiffusionModelLoader
from .nodes.media import VideoMotionContactSheet
from .nodes.minimax import (
    H3ConcatAVLatent,
    H3SeparateAVLatent,
    MiniMaxH3VideoFramesPadding,
)
from .nodes.wan import WanVideoFramesPadding


NODE_CLASS_MAPPINGS = {
    "TuringUtilsConvRotDiffusionModelLoader": ConvRotDiffusionModelLoader,
    "TuringUtilsConvRotCLIPLoader": ConvRotCLIPLoader,
    "TuringUtilsWanVideoFramesPadding": WanVideoFramesPadding,
    "TuringUtilsMiniMaxH3VideoFramesPadding": MiniMaxH3VideoFramesPadding,
    "TuringUtilsBerniniContextWindowsCore": BerniniContextWindowsCore,
    "TuringUtilsBerniniInpaintCondition": BerniniInpaintCondition,
    "TuringUtilsH3ConcatAVLatent": H3ConcatAVLatent,
    "TuringUtilsH3SeparateAVLatent": H3SeparateAVLatent,
    "TuringUtilsSolSparseAttentionPatch": SolSparseAttentionPatch,
    "TuringUtilsAttentionKernelTuningPatch": AttentionKernelTuningPatch,
    "TuringUtilsVideoMotionContactSheet": VideoMotionContactSheet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TuringUtilsConvRotDiffusionModelLoader": "Load ConvRot DiT",
    "TuringUtilsConvRotCLIPLoader": "Load ConvRot CLIP",
    "TuringUtilsWanVideoFramesPadding": "Wan Video Frames Padding",
    "TuringUtilsMiniMaxH3VideoFramesPadding": "MiniMax H3 Video Frames Padding",
    "TuringUtilsBerniniContextWindowsCore": "Bernini Context Windows",
    "TuringUtilsBerniniInpaintCondition": "Bernini Inpaint Condition",
    "TuringUtilsH3ConcatAVLatent": "H3 Concat AV Latent",
    "TuringUtilsH3SeparateAVLatent": "H3 Separate AV Latent",
    "TuringUtilsSolSparseAttentionPatch": "Patch Sol Sparse Attention (Experimental)",
    "TuringUtilsAttentionKernelTuningPatch": "Patch Turing Attention Kernel Tuning (Experimental)",
    "TuringUtilsVideoMotionContactSheet": "Video Motion Contact Sheet (Experimental)",
}
