from __future__ import annotations

from .nodes.attention import AttentionKernelTuningPatch, SolSparseAttentionPatch
from .nodes.bernini import BerniniContextWindowsCore, BerniniInpaintCondition
from .nodes.loaders import ConvRotCLIPLoader, ConvRotDiffusionModelLoader
from .nodes.media import VideoMotionContactSheet
from .nodes.minimax import (
    H3ConcatAVLatent,
    H3SeparateAVLatent,
    MiniMaxH3BlockCachePatch,
    MiniMaxH3VideoFramesPadding,
)
from .nodes.minimax_vae import MiniMaxH3VideoVAEDecode, MiniMaxH3VideoVAEEncode
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
    "TuringUtilsMiniMaxH3BlockCachePatch": MiniMaxH3BlockCachePatch,
    "TuringUtilsSolSparseAttentionPatch": SolSparseAttentionPatch,
    "TuringUtilsAttentionKernelTuningPatch": AttentionKernelTuningPatch,
    "TuringUtilsVideoMotionContactSheet": VideoMotionContactSheet,
    "TuringUtilsMiniMaxH3VideoVAEDecode": MiniMaxH3VideoVAEDecode,
    "TuringUtilsMiniMaxH3VideoVAEEncode": MiniMaxH3VideoVAEEncode,
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
    "TuringUtilsMiniMaxH3BlockCachePatch": "Patch MiniMax H3 Block Cache (Experimental)",
    "TuringUtilsSolSparseAttentionPatch": "Patch Sol Sparse Attention",
    "TuringUtilsAttentionKernelTuningPatch": "Patch Turing Attention Kernel Tuning (Experimental)",
    "TuringUtilsVideoMotionContactSheet": "Video Motion Contact Sheet (Experimental)",
    "TuringUtilsMiniMaxH3VideoVAEDecode": "MiniMax H3 Video VAE Decode (Experimental)",
    "TuringUtilsMiniMaxH3VideoVAEEncode": "MiniMax H3 Video VAE Encode (Experimental)",
}
