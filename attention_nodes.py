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
                        "default": 4096,
                        "min": 64,
                        "max": 262144,
                        "step": 64,
                        "tooltip": "Use stable dense Sage below this Q or K sequence length.",
                    },
                ),
                "dense_prefix_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 0,
                        "max": 262144,
                        "step": 64,
                        "tooltip": "Keep this leading Q/K token region exact. The kernel rounds it up to 64-token blocks.",
                    },
                ),
                "route_threshold": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "round": 0.01,
                        "tooltip": "Higher values select fewer exact blocks for more speed and greater approximation.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Turing Utils/patches"
    TITLE = "Patch Sol Sparse Attention (Experimental)"

    def patch(
        self,
        model,
        min_sequence_tokens: int = 4096,
        dense_prefix_tokens: int = 512,
        route_threshold: float = 1.0,
    ):
        return (
            apply_sparse_attention_patch(
                model,
                min_sequence_tokens=min_sequence_tokens,
                dense_prefix_tokens=dense_prefix_tokens,
                route_threshold=route_threshold,
            ),
        )
