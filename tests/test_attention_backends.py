from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import attention_backends  # noqa: E402
from comfy.ldm.modules import attention  # noqa: E402


class FakeModel:
    def __init__(self):
        self.model_options = {}


class AttentionBackendsTest(unittest.TestCase):
    def test_backend_choices_are_stable(self):
        self.assertEqual(
            attention_backends.attention_backend_choices(),
            ("default", "sdpa", "sage_attn", "flash_attn"),
        )

    def test_aliases_normalize_to_node_options(self):
        self.assertEqual(attention_backends.normalize_attention_backend("auto"), "default")
        self.assertEqual(attention_backends.normalize_attention_backend("torch-sdpa"), "sdpa")
        self.assertEqual(attention_backends.normalize_attention_backend("sage attention"), "sage_attn")
        self.assertEqual(attention_backends.normalize_attention_backend("flash-attn"), "flash_attn")

    def test_default_backend_does_not_patch_attention(self):
        model = FakeModel()
        attention_backends.apply_attention_backend(model, "default")
        self.assertEqual(model.model_options, {"transformer_options": {}})

    def test_sdpa_backend_overrides_optimized_attention(self):
        model = FakeModel()
        attention_backends.apply_attention_backend(model, "sdpa")
        transformer_options = model.model_options["transformer_options"]
        self.assertEqual(transformer_options["svdint4_attention_backend"], "sdpa")
        self.assertIn("optimized_attention_override", transformer_options)

        q = torch.randn(1, 8, 16)
        k = torch.randn(1, 8, 16)
        v = torch.randn(1, 8, 16)
        out = attention.optimized_attention(q, k, v, heads=2, transformer_options=transformer_options)
        self.assertEqual(tuple(out.shape), (1, 8, 16))


if __name__ == "__main__":
    unittest.main()
