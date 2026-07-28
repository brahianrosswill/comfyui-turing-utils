from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import attention_backends  # noqa: E402
import svdint4_nodes  # noqa: E402
from comfy.ldm.modules import attention  # noqa: E402


class FakeModel:
    def __init__(self):
        self.model_options = {}


class AttentionBackendsTest(unittest.TestCase):
    def test_backend_choices_are_stable(self):
        self.assertEqual(
            attention_backends.attention_backend_choices(),
            ("auto", "sage_attn", "flash_attn", "sdpa"),
        )

    def test_aliases_normalize_to_node_options(self):
        self.assertEqual(attention_backends.normalize_attention_backend("default"), "auto")
        self.assertEqual(attention_backends.normalize_attention_backend("torch-sdpa"), "sdpa")
        self.assertEqual(attention_backends.normalize_attention_backend("sage attention"), "sage_attn")
        self.assertEqual(attention_backends.normalize_attention_backend("flash-attn"), "flash_attn")

    def test_svdint4_node_defaults_to_auto(self):
        with mock.patch("svdint4_nodes._model_names", return_value=[]):
            patch_attention = svdint4_nodes.SVDInt4DiffusionModelLoader.INPUT_TYPES()["optional"]["patch_attention"]

        self.assertEqual(
            patch_attention[0],
            ("auto", "sage_attn", "flash_attn", "sdpa"),
        )
        self.assertEqual(patch_attention[1]["default"], "auto")

    def test_auto_prefers_sage(self):
        model = FakeModel()
        functions = {
            "sage": lambda *args, **kwargs: None,
            "flash": lambda *args, **kwargs: None,
            "pytorch": lambda *args, **kwargs: None,
        }
        with mock.patch(
            "comfy.ldm.modules.attention.get_attention_function",
            side_effect=lambda name, default: functions.get(name, default),
        ):
            attention_backends.apply_attention_backend(model, "auto")

        self.assertEqual(
            model.model_options["transformer_options"]["svdint4_attention_backend"],
            "sage_attn",
        )

    def test_auto_falls_back_to_flash(self):
        model = FakeModel()
        functions = {
            "sage": None,
            "flash": lambda *args, **kwargs: None,
            "pytorch": lambda *args, **kwargs: None,
        }
        with mock.patch(
            "comfy.ldm.modules.attention.get_attention_function",
            side_effect=lambda name, default: functions.get(name, default),
        ):
            attention_backends.apply_attention_backend(model, "auto")

        transformer_options = model.model_options["transformer_options"]
        self.assertEqual(transformer_options["svdint4_attention_backend"], "flash_attn")
        self.assertIn("optimized_attention_override", transformer_options)

    def test_auto_falls_back_to_sdpa(self):
        model = FakeModel()
        functions = {
            "sage": None,
            "flash": None,
            "pytorch": lambda *args, **kwargs: None,
        }
        with mock.patch(
            "comfy.ldm.modules.attention.get_attention_function",
            side_effect=lambda name, default: functions.get(name, default),
        ):
            attention_backends.apply_attention_backend(model, "auto")

        self.assertEqual(
            model.model_options["transformer_options"]["svdint4_attention_backend"],
            "sdpa",
        )

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
