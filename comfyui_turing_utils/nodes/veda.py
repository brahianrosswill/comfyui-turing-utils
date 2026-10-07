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
        inputs = {"required": {
            "model": ("MODEL",),
            "predictor_name": (predictor_choices(),),
            "predictor_precision": (list(PRECISIONS), {"default": "w8a8"}),
            "keep_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "0 uses the predictor's trained keep ratio; 1 keeps full attention."}),
            "reference_keep_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "0 uses the predictor's trained keep ratio; 1 keeps references dense in both directions."}),
            "plan_policy": (["nearest", "strict"],),
            "dense_prefix_steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "dense_suffix_steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "dense_prefix_layers": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "dense_suffix_layers": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "debug": ("BOOLEAN", {"default": False}),
        }}
        for name, spec in inputs["required"].items():
            if name not in ("model", "predictor_name", "predictor_precision"):
                inputs["required"][name] = (spec[0], {**(spec[1] if len(spec) > 1 else {}), "advanced": True})
        return inputs

    def configure(self, model, **kwargs):
        return (configure(model, **kwargs),)
