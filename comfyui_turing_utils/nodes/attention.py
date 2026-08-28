"""ComfyUI nodes for production attention patches."""

from __future__ import annotations

from ..adapters.minimax.image_sol import apply_h3_image_sol_attention
from ..adapters.minimax.virtual_kv import apply_h3_virtual_kv
from ..attention import apply_sla_attention_patch, apply_sparse_attention_patch


class H3StaticVirtualKV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (
                    ["conservative", "fast", "residual"],
                    {
                        "default": "conservative",
                        "tooltip": "Conservative materializes an exact 7-slice virtual K/V context. Fast keeps 2 physical slices and computes all 7 logical slices exactly in bundled W8A8. Residual keeps the two real slices Dense and compresses the five virtual slices into Sol 2x32 residuals; kernel 0.41 supports inherited W8A8, Sage, and SDPA numeric paths. Unsupported or stale kernels safely fall back to exact materialization.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Turing Utils/patches"
    TITLE = "Configure H3 Static Virtual KV"

    def patch(self, model, mode: str = "conservative"):
        return (apply_h3_virtual_kv(model, mode=mode),)


class LegacySolSparseAttentionPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "routing_threshold": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -4.0,
                        "max": 4.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "Route blocks whose input-adaptive proxy score exceeds mean + threshold × standard deviation. Lower values preserve more exact blocks; 1.0 matches the official Sol policy.",
                    },
                ),
                "prefix_policy": (
                    ["auto", "none", "manual"],
                    {
                        "default": "auto",
                        "tooltip": "Auto applies the model's semantic segments and the three reference switches; none protects no modality ranges; manual protects only the leading token count below.",
                    },
                ),
                "manual_prefix_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 262144,
                        "step": 64,
                        "tooltip": "Leading Query tokens kept dense and leading K/V tokens kept exact only for manual policy. Boundaries round outward to 64-token blocks.",
                    },
                ),
                "skipped_residual": (
                    ["1x64", "2x32"],
                    {
                        "default": "1x64",
                        "tooltip": "Official-style 1x64 uses one K/V centroid per skipped block. 2x32 keeps two residual centroids for higher approximation quality without changing routing.",
                    },
                ),
                "sparse_reference_image": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Allow reference-image Query/KV interactions to use Sol routing. Disabled protects keyframes and reference images with dense Query and exact KV blocks.",
                    },
                ),
                "sparse_reference_video": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Allow long reference-video Query/KV interactions to use Sol routing instead of protecting the complete reference video.",
                    },
                ),
                "sparse_reference_audio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Allow reference-audio Query/KV interactions to use Sol routing. Disabled preserves reference audio and dialogue conditioning exactly.",
                    },
                ),
                "dense_prefix_steps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of early denoising steps that use the selected dense backend across every transformer layer (stable Sage, or W8A8 when enabled).",
                    },
                ),
                "dense_suffix_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of final denoising steps that use the selected dense backend across every transformer layer (stable Sage, or W8A8 when enabled).",
                    },
                ),
                "dense_prefix_layers": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Keep this many transformer layers at the beginning of every sparse step on the selected dense backend. If prefix + suffix reaches the model layer count, all layers use the dense backend without Sol preprocessing.",
                    },
                ),
                "dense_suffix_layers": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Keep this many transformer layers at the end of every sparse step on the selected dense backend. Requires layer-count metadata; overlap with the prefix intentionally makes all layers dense.",
                    },
                ),
            },
            "optional": {
                "use_w8a8": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use signed INT8 V and unsigned INT8 probability Tensor Cores for Sol exact blocks and protected dense steps/layers. Enabled is the default Turing fast path.",
                    },
                ),
                "debug_route_density": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Log min/mean/max route density once per denoising step. Disabled by default; enabling it adds tiny reductions and one synchronization per step.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Turing Utils/patches"
    TITLE = "Patch Sol Sparse Attention"

    def patch(
        self,
        model,
        routing_threshold: float = 1.0,
        prefix_policy: str = "auto",
        manual_prefix_tokens: int = 0,
        skipped_residual: str = "1x64",
        sparse_reference_image: bool = False,
        sparse_reference_video: bool = True,
        sparse_reference_audio: bool = False,
        dense_prefix_steps: int = 1,
        dense_suffix_steps: int = 0,
        dense_prefix_layers: int = 2,
        dense_suffix_layers: int = 0,
        use_w8a8: bool = True,
        debug_route_density: bool = False,
    ):
        return (
            apply_sparse_attention_patch(
                model,
                routing_threshold=routing_threshold,
                prefix_policy=prefix_policy,
                manual_prefix_tokens=manual_prefix_tokens,
                skipped_residual=skipped_residual,
                sparse_reference_image=sparse_reference_image,
                sparse_reference_video=sparse_reference_video,
                sparse_reference_audio=sparse_reference_audio,
                dense_prefix_steps=dense_prefix_steps,
                dense_suffix_steps=dense_suffix_steps,
                dense_prefix_layers=dense_prefix_layers,
                dense_suffix_layers=dense_suffix_layers,
                use_w8a8=use_w8a8,
                debug_route_density=debug_route_density,
            ),
        )


class LegacySlaSparseAttentionPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sparsity_ratio": (
                    "FLOAT",
                    {
                        "default": 0.85,
                        "min": 0.0,
                        "max": 0.99,
                        "step": 0.01,
                        "round": 0.01,
                        "tooltip": "Fraction of 64-token K/V blocks skipped for every 128-token Query block. 0.85 matches the public MiniMax H3 Turbo-SLA training/runtime hyperparameter and should be used with an SLA-trained LoRA. Zero dispatches directly to the dense backend.",
                    },
                ),
                "prefix_policy": (
                    ["auto", "none", "manual"],
                    {
                        "default": "auto",
                        "tooltip": "Auto applies the H3 semantic layout and reference switches. None reproduces the unprotected published SLA route. Manual protects only the leading token count.",
                    },
                ),
                "manual_prefix_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 262144,
                        "step": 64,
                        "tooltip": "Leading Query tokens kept dense and leading K/V tokens kept exact for manual policy.",
                    },
                ),
                "sparse_reference_image": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Allow image-reference and keyframe blocks to use SLA Top-K routing. Disabled keeps their Query dense and KV exact.",
                    },
                ),
                "sparse_reference_video": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Allow long reference-video blocks to use SLA Top-K routing.",
                    },
                ),
                "sparse_reference_audio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Allow reference-audio blocks to use SLA Top-K routing. Disabled protects dialogue conditioning exactly.",
                    },
                ),
                "dense_prefix_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Early denoising steps that use the selected dense backend in every layer. Set 0 to reproduce all-step SLA routing.",
                    },
                ),
                "dense_suffix_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Final denoising steps that use the selected dense backend in every layer.",
                    },
                ),
                "dense_prefix_layers": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Leading transformer layers kept dense in sparse steps. Prefix plus suffix reaching the model layer count makes the complete patch dense.",
                    },
                ),
                "dense_suffix_layers": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Trailing transformer layers kept dense in sparse steps.",
                    },
                ),
            },
            "optional": {
                "use_w8a8": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use INT8 V and probability Tensor Cores for selected SLA blocks and protected dense steps/layers.",
                    },
                ),
                "debug_route_density": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Log realized SLA route density, including exact reference blocks.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Turing Utils/patches"
    TITLE = "Patch SLA Sparse Attention"

    def patch(
        self,
        model,
        sparsity_ratio: float = 0.85,
        prefix_policy: str = "auto",
        manual_prefix_tokens: int = 0,
        sparse_reference_image: bool = False,
        sparse_reference_video: bool = True,
        sparse_reference_audio: bool = False,
        dense_prefix_steps: int = 0,
        dense_suffix_steps: int = 0,
        dense_prefix_layers: int = 0,
        dense_suffix_layers: int = 0,
        use_w8a8: bool = True,
        debug_route_density: bool = False,
    ):
        return (
            apply_sla_attention_patch(
                model,
                sparsity_ratio=sparsity_ratio,
                prefix_policy=prefix_policy,
                manual_prefix_tokens=manual_prefix_tokens,
                sparse_reference_image=sparse_reference_image,
                sparse_reference_video=sparse_reference_video,
                sparse_reference_audio=sparse_reference_audio,
                dense_prefix_steps=dense_prefix_steps,
                dense_suffix_steps=dense_suffix_steps,
                dense_prefix_layers=dense_prefix_layers,
                dense_suffix_layers=dense_suffix_layers,
                use_w8a8=use_w8a8,
                debug_route_density=debug_route_density,
            ),
        )


def _rewrite_dense_backend_tooltips(required: dict) -> dict:
    required = dict(required)
    explanations = {
        "dense_prefix_steps": "Leading steps of every sampler invocation that use the loader-selected dense backend across every transformer layer.",
        "dense_suffix_steps": "Trailing steps of every sampler invocation that use the loader-selected dense backend across every transformer layer.",
        "dense_prefix_layers": "Leading transformer layers kept on the loader-selected dense backend during sparse steps.",
        "dense_suffix_layers": "Trailing transformer layers kept on the loader-selected dense backend during sparse steps.",
    }
    for name, tooltip in explanations.items():
        kind, settings = required[name]
        settings = dict(settings)
        settings["tooltip"] = tooltip
        required[name] = (kind, settings)
    return required


class SolSparseAttentionPatch(LegacySolSparseAttentionPatch):
    """Configure Sol while inheriting the loader-selected dense backend."""

    TITLE = "Configure Sol Sparse Attention"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = LegacySolSparseAttentionPatch.INPUT_TYPES()
        required = _rewrite_dense_backend_tooltips(inputs["required"])
        return {
            "required": required,
            "optional": {
                "debug_route_density": inputs["optional"]["debug_route_density"]
            },
        }

    def patch(
        self,
        model,
        routing_threshold: float = 1.0,
        prefix_policy: str = "auto",
        manual_prefix_tokens: int = 0,
        skipped_residual: str = "1x64",
        sparse_reference_image: bool = False,
        sparse_reference_video: bool = True,
        sparse_reference_audio: bool = False,
        dense_prefix_steps: int = 1,
        dense_suffix_steps: int = 0,
        dense_prefix_layers: int = 2,
        dense_suffix_layers: int = 0,
        debug_route_density: bool = False,
    ):
        return (
            apply_sparse_attention_patch(
                model,
                routing_threshold=routing_threshold,
                prefix_policy=prefix_policy,
                manual_prefix_tokens=manual_prefix_tokens,
                skipped_residual=skipped_residual,
                sparse_reference_image=sparse_reference_image,
                sparse_reference_video=sparse_reference_video,
                sparse_reference_audio=sparse_reference_audio,
                dense_prefix_steps=dense_prefix_steps,
                dense_suffix_steps=dense_suffix_steps,
                dense_prefix_layers=dense_prefix_layers,
                dense_suffix_layers=dense_suffix_layers,
                debug_route_density=debug_route_density,
            ),
        )


class H3ImageSolAttentionPatch:
    @classmethod
    def INPUT_TYPES(cls):
        sol_inputs = SolSparseAttentionPatch.INPUT_TYPES()
        standard = sol_inputs["required"]
        return {
            "required": {
                "model": ("MODEL",),
                "temporal_layout": (
                    ["dense_anchor_grid", "dense_window"],
                    {
                        "default": "dense_anchor_grid",
                        "tooltip": "Dense anchor grid keeps every H3 17-frame chunk anchor plus the first continuation exact. Dense window keeps only the initial 1+4 latent-time window exact.",
                    },
                ),
                "sparse_reference_image": standard["sparse_reference_image"],
                "sparse_reference_video": standard["sparse_reference_video"],
                "sparse_reference_audio": standard["sparse_reference_audio"],
                "dense_prefix_steps": standard["dense_prefix_steps"],
                "dense_suffix_steps": standard["dense_suffix_steps"],
                "dense_prefix_layers": standard["dense_prefix_layers"],
                "dense_suffix_layers": standard["dense_suffix_layers"],
            },
            "optional": {
                "debug_route_density": sol_inputs["optional"][
                    "debug_route_density"
                ]
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Turing Utils/patches"
    TITLE = "Configure H3 Image Sol Attention"

    def patch(
        self,
        model,
        temporal_layout: str = "dense_anchor_grid",
        sparse_reference_image: bool = False,
        sparse_reference_video: bool = True,
        sparse_reference_audio: bool = False,
        dense_prefix_steps: int = 1,
        dense_suffix_steps: int = 0,
        dense_prefix_layers: int = 2,
        dense_suffix_layers: int = 0,
        debug_route_density: bool = False,
    ):
        return (
            apply_h3_image_sol_attention(
                model,
                temporal_layout=temporal_layout,
                sparse_reference_image=sparse_reference_image,
                sparse_reference_video=sparse_reference_video,
                sparse_reference_audio=sparse_reference_audio,
                dense_prefix_steps=dense_prefix_steps,
                dense_suffix_steps=dense_suffix_steps,
                dense_prefix_layers=dense_prefix_layers,
                dense_suffix_layers=dense_suffix_layers,
                debug_route_density=debug_route_density,
            ),
        )


class SlaSparseAttentionPatch(LegacySlaSparseAttentionPatch):
    """Configure SLA while inheriting the loader-selected dense backend."""

    TITLE = "Configure SLA Sparse Attention"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = LegacySlaSparseAttentionPatch.INPUT_TYPES()
        required = _rewrite_dense_backend_tooltips(inputs["required"])
        return {
            "required": required,
            "optional": {
                "debug_route_density": inputs["optional"]["debug_route_density"]
            },
        }

    def patch(
        self,
        model,
        sparsity_ratio: float = 0.85,
        prefix_policy: str = "auto",
        manual_prefix_tokens: int = 0,
        sparse_reference_image: bool = False,
        sparse_reference_video: bool = True,
        sparse_reference_audio: bool = False,
        dense_prefix_steps: int = 0,
        dense_suffix_steps: int = 0,
        dense_prefix_layers: int = 0,
        dense_suffix_layers: int = 0,
        debug_route_density: bool = False,
    ):
        return (
            apply_sla_attention_patch(
                model,
                sparsity_ratio=sparsity_ratio,
                prefix_policy=prefix_policy,
                manual_prefix_tokens=manual_prefix_tokens,
                sparse_reference_image=sparse_reference_image,
                sparse_reference_video=sparse_reference_video,
                sparse_reference_audio=sparse_reference_audio,
                dense_prefix_steps=dense_prefix_steps,
                dense_suffix_steps=dense_suffix_steps,
                dense_prefix_layers=dense_prefix_layers,
                dense_suffix_layers=dense_suffix_layers,
                debug_route_density=debug_route_density,
            ),
        )
