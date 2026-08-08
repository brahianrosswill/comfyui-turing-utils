from __future__ import annotations

try:
    from .attention import apply_sparse_attention_patch
except ImportError:
    from attention import apply_sparse_attention_patch


class SolSparseAttentionPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "min_sequence_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 262144,
                        "step": 64,
                        "tooltip": "Use 0 for automatic crossover selection, or keep stable Sage below this Q or K length.",
                    },
                ),
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
                        "tooltip": "Auto uses semantic token-layout metadata when the model supplies it; none protects no prefix; manual uses the token count below.",
                    },
                ),
                "manual_prefix_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 262144,
                        "step": 64,
                        "tooltip": "Leading Q/K tokens kept exact only when prefix_policy is manual. Rounded up to 64-token blocks.",
                    },
                ),
                "local_block_radius": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Always evaluate this many neighboring 64-token blocks on each side exactly.",
                    },
                ),
                "temporal_neighbor_frames": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Keep matching spatial ranges in this many adjacent frames exact when a model adapter supplies video topology. Has no effect without metadata.",
                    },
                ),
                "skipped_residual": (
                    ["2x32", "1x64"],
                    {
                        "default": "2x32",
                        "tooltip": "Approximate each skipped 64-token block with two 32-token K/V centroids (higher quality) or one 64-token centroid (lower summary cost).",
                    },
                ),
                "minimum_route_density": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": 0.01,
                        "tooltip": "Minimum adaptive exact-route density per 16-block routing tile. Forced prefix/local/temporal blocks can raise the actual density above this value.",
                    },
                ),
                "maximum_route_density": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": 0.01,
                        "tooltip": "Maximum adaptive exact-route density per 16-block routing tile. Forced semantic and local blocks are never removed.",
                    },
                ),
                "dense_warmup_ratio": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": 0.01,
                        "tooltip": "Fraction of early denoising steps that use stable dense Sage. One of four steps is dense at the 0.25 default.",
                    },
                ),
                "dense_tail_ratio": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": 0.01,
                        "tooltip": "Optional fraction of final denoising steps that use dense Sage. Keep zero unless late detail still flickers.",
                    },
                ),
                "dense_prefix_layers": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Keep the first transformer layers dense when an adapter supplies layer metadata. Official MiniMax H3 uses 2 of 50 layers.",
                    },
                ),
            },
            "optional": {
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
    TITLE = "Patch Sol Sparse Attention (Experimental)"

    def patch(
        self,
        model,
        min_sequence_tokens: int = 0,
        routing_threshold: float = 1.0,
        prefix_policy: str = "auto",
        manual_prefix_tokens: int = 0,
        local_block_radius: int = 1,
        temporal_neighbor_frames: int = 1,
        skipped_residual: str = "2x32",
        minimum_route_density: float = 0.0,
        maximum_route_density: float = 1.0,
        dense_warmup_ratio: float = 0.25,
        dense_tail_ratio: float = 0.0,
        dense_prefix_layers: int = 2,
        debug_route_density: bool = False,
    ):
        return (
            apply_sparse_attention_patch(
                model,
                min_sequence_tokens=min_sequence_tokens,
                routing_threshold=routing_threshold,
                prefix_policy=prefix_policy,
                manual_prefix_tokens=manual_prefix_tokens,
                local_block_radius=local_block_radius,
                temporal_neighbor_frames=temporal_neighbor_frames,
                skipped_residual=skipped_residual,
                minimum_route_density=minimum_route_density,
                maximum_route_density=maximum_route_density,
                dense_warmup_ratio=dense_warmup_ratio,
                dense_tail_ratio=dense_tail_ratio,
                dense_prefix_layers=dense_prefix_layers,
                debug_route_density=debug_route_density,
            ),
        )
