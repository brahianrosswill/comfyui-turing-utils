"""Predictor-driven MiniMax H3 sparse attention configuration."""

from ..adapters.minimax.veda.integration import configure, predictor_choices
from ..adapters.minimax.veda.predictor import PRECISIONS


class H3VedaAttentionStrategy:
    TITLE = "Configure H3 Veda Sparse Attention"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "configure"
    CATEGORY = "Turing Utils/attention"
    DESCRIPTION = (
        "H3 only. Loads a Veda predictor bundle from models/veda, not a LoRA. "
        "Only the current predictor head group is staged on GPU. BF16 uses "
        "FP32 arithmetic with BF16 activations/output on Turing; FP32 is a "
        "separate diagnostic mode. Nearest plans may be outside training geometry."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "predictor_name": (predictor_choices(),),
            "predictor_precision": (list(PRECISIONS), {"default": "w8a8"}),
            "keep_ratio": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01}),
            "reference_keep_ratio": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1.0, "step": 0.01}),
            "plan_policy": (["nearest", "strict"],),
            "dense_prefix_steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "dense_suffix_steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "dense_prefix_layers": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "dense_suffix_layers": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "debug": ("BOOLEAN", {"default": False}),
        }}

    def configure(self, model, **kwargs):
        # Legacy experimental workflow/API inputs must not override automatic
        # scheduling. Internal benchmark entry points remain independently usable.
        kwargs.pop("execution_mode", None)
        kwargs.pop("projection_chunk_tiles", None)
        return (configure(model, **kwargs),)
