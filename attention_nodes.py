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
                "attention_mass_recall": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.01,
                        "tooltip": "Estimated centroid attention mass evaluated with exact token attention. Higher values preserve more detail and use more compute.",
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
                "dense_warmup_ratio": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": 0.01,
                        "tooltip": "Fraction of early denoising steps that use stable dense Sage. Zero favors short-step acceleration.",
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
        min_sequence_tokens: int = 0,
        attention_mass_recall: float = 0.3,
        prefix_policy: str = "auto",
        manual_prefix_tokens: int = 0,
        local_block_radius: int = 1,
        dense_warmup_ratio: float = 0.0,
    ):
        return (
            apply_sparse_attention_patch(
                model,
                min_sequence_tokens=min_sequence_tokens,
                attention_mass_recall=attention_mass_recall,
                prefix_policy=prefix_policy,
                manual_prefix_tokens=manual_prefix_tokens,
                local_block_radius=local_block_radius,
                dense_warmup_ratio=dense_warmup_ratio,
            ),
        )
