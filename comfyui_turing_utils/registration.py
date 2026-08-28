from __future__ import annotations

from .adapters.minimax.conditioning import install_combined_minimax_conditioning_support
from .nodes.attention import (
    H3StaticVirtualKV,
    LegacySlaSparseAttentionPatch,
    LegacySolSparseAttentionPatch,
    SlaSparseAttentionPatch,
    SolSparseAttentionPatch,
)
from .nodes.bernini import BerniniContextWindowsCore, BerniniInpaintCondition
from .nodes.krea2 import Krea2IdentityEditConditioning
from .nodes.loaders import ConvRotCLIPLoader, ConvRotDiffusionModelLoader
from .nodes.logic import IsInputPresent, LazyIfElse
from .nodes.media import ResizeImageIfPresent, VideoMotionContactSheet
from .nodes.minimax import (
    H3ConcatAVLatent,
    H3SeparateAVLatent,
    MiniMaxH3BlockCachePatch,
    MiniMaxH3LatentUpscale,
    MiniMaxH3LatentUpscaleModelLoader,
    MiniMaxH3VideoFramesPadding,
)
from .nodes.minimax_vae import (
    MiniMaxH3VideoVAEDecode,
    MiniMaxH3VideoVAEEncode,
)
from .nodes.minimax_references import (
    H3AudioReference,
    H3BuildConditioning,
    H3KeyframeReference,
    H3ImageReference,
    H3LatentInfo,
    H3SemanticReference,
    H3VideoReference,
)
from .nodes.wan import WanVideoFramesPadding


install_combined_minimax_conditioning_support()


NODE_CLASS_MAPPINGS = {
    "TuringUtilsConvRotDiffusionModelLoader": ConvRotDiffusionModelLoader,
    "TuringUtilsConvRotCLIPLoader": ConvRotCLIPLoader,
    "TuringUtilsWanVideoFramesPadding": WanVideoFramesPadding,
    "TuringUtilsMiniMaxH3VideoFramesPadding": MiniMaxH3VideoFramesPadding,
    "TuringUtilsBerniniContextWindowsCore": BerniniContextWindowsCore,
    "TuringUtilsBerniniInpaintCondition": BerniniInpaintCondition,
    "TuringUtilsKrea2IdentityEditConditioning": Krea2IdentityEditConditioning,
    "TuringUtilsIsInputPresent": IsInputPresent,
    "TuringUtilsLazyIfElse": LazyIfElse,
    "TuringUtilsH3ConcatAVLatent": H3ConcatAVLatent,
    "TuringUtilsH3SeparateAVLatent": H3SeparateAVLatent,
    "TuringUtilsH3LatentInfo": H3LatentInfo,
    "TuringUtilsH3KeyframeReference": H3KeyframeReference,
    "TuringUtilsH3ImageReference": H3ImageReference,
    "TuringUtilsH3VideoReference": H3VideoReference,
    "TuringUtilsH3AudioReference": H3AudioReference,
    "TuringUtilsH3SemanticReference": H3SemanticReference,
    "TuringUtilsH3BuildConditioning": H3BuildConditioning,
    "TuringUtilsMiniMaxH3LatentUpscaleModelLoader": MiniMaxH3LatentUpscaleModelLoader,
    "TuringUtilsMiniMaxH3LatentUpscale": MiniMaxH3LatentUpscale,
    "TuringUtilsMiniMaxH3BlockCachePatch": MiniMaxH3BlockCachePatch,
    # Preserve positional widget decoding for existing workflows.  New nodes
    # inherit the loader-selected dense backend and omit the old use_w8a8 flag.
    "TuringUtilsSolSparseAttentionPatch": LegacySolSparseAttentionPatch,
    "TuringUtilsSlaSparseAttentionPatch": LegacySlaSparseAttentionPatch,
    "TuringUtilsSolAttentionStrategy": SolSparseAttentionPatch,
    "TuringUtilsSlaAttentionStrategy": SlaSparseAttentionPatch,
    "TuringUtilsH3StaticVirtualKV": H3StaticVirtualKV,
    "TuringUtilsResizeImageIfPresent": ResizeImageIfPresent,
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
    "TuringUtilsKrea2IdentityEditConditioning": "Krea2 Identity Edit Conditioning",
    "TuringUtilsIsInputPresent": "Is Input Present",
    "TuringUtilsLazyIfElse": "Lazy If / Else",
    "TuringUtilsH3ConcatAVLatent": "H3 Concat AV Latent",
    "TuringUtilsH3SeparateAVLatent": "H3 Separate AV Latent",
    "TuringUtilsH3LatentInfo": "H3 Latent Info",
    "TuringUtilsH3KeyframeReference": "H3 Keyframe Reference",
    "TuringUtilsH3ImageReference": "H3 Image Reference",
    "TuringUtilsH3VideoReference": "H3 Video Reference",
    "TuringUtilsH3AudioReference": "H3 Audio Reference",
    "TuringUtilsH3SemanticReference": "H3 Semantic Reference",
    "TuringUtilsH3BuildConditioning": "H3 Build Conditioning",
    "TuringUtilsMiniMaxH3LatentUpscaleModelLoader": "Load MiniMax H3 Latent Upscaler",
    "TuringUtilsMiniMaxH3LatentUpscale": "MiniMax H3 Latent Upscale",
    "TuringUtilsMiniMaxH3BlockCachePatch": "Patch MiniMax H3 Block Cache (Experimental)",
    "TuringUtilsSolSparseAttentionPatch": "Patch Sol Sparse Attention",
    "TuringUtilsSlaSparseAttentionPatch": "Patch SLA Sparse Attention",
    "TuringUtilsSolAttentionStrategy": "Configure Sol Sparse Attention",
    "TuringUtilsSlaAttentionStrategy": "Configure SLA Sparse Attention",
    "TuringUtilsH3StaticVirtualKV": "Configure H3 Static Virtual KV",
    "TuringUtilsResizeImageIfPresent": "Resize Image If Present",
    "TuringUtilsVideoMotionContactSheet": "Video Motion Contact Sheet (Experimental)",
    "TuringUtilsMiniMaxH3VideoVAEDecode": "MiniMax H3 Video VAE Decode",
    "TuringUtilsMiniMaxH3VideoVAEEncode": "MiniMax H3 Video VAE Encode",
}
