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

    def test_external_sage_sends_fp32_qkv_to_original_attention(self):
        sage = mock.Mock()
        original = mock.Mock(return_value="original")
        with mock.patch(
            "comfy.ldm.modules.attention.get_attention_function",
            side_effect=lambda name, default: sage if name == "sage" else default,
        ):
            override = attention_backends.make_attention_override("sage_attn")

        q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
        self.assertEqual(override(original, q, q, q, 2, skip_reshape=True), "original")
        original.assert_called_once()
        sage.assert_not_called()

    def test_external_sage_sends_mixed_qkv_to_original_attention(self):
        sage = mock.Mock()
        original = mock.Mock(return_value="original")
        with mock.patch(
            "comfy.ldm.modules.attention.get_attention_function",
            side_effect=lambda name, default: sage if name == "sage" else default,
        ):
            override = attention_backends.make_attention_override("sage_attn")

        bf16 = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        fp32 = bf16.float()
        self.assertEqual(override(original, bf16, fp32, bf16, 2, skip_reshape=True), "original")
        original.assert_called_once()
        sage.assert_not_called()

    def test_sage_does_not_recast_supported_qkv_dtype(self):
        captured = {}

        def sage(q, k, v, *args, **kwargs):
            captured["q"] = q
            return q

        with mock.patch(
            "comfy.ldm.modules.attention.get_attention_function",
            side_effect=lambda name, default: sage if name == "sage" else default,
        ):
            override = attention_backends.make_attention_override("sage_attn")

        q = torch.randn(1, 2, 4, 8, dtype=torch.float16)
        out = override(lambda *args, **kwargs: None, q, q, q, 2, skip_reshape=True)

        self.assertIs(captured["q"], q)
        self.assertIs(out, q)

    def test_sage_fp32_fallback_runs_through_comfy_attention_wrapper(self):
        model = FakeModel()
        with (
            mock.patch(
                "comfy.ldm.modules.attention.get_attention_function",
                side_effect=lambda name, default: (
                    attention.attention_sage if name == "sage" else default
                ),
            ),
            mock.patch("comfy.ldm.modules.attention.sageattn") as sage,
        ):
            attention_backends.apply_attention_backend(model, "sage_attn")
            transformer_options = model.model_options["transformer_options"]
            q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
            output = attention.optimized_attention(
                q,
                q,
                q,
                heads=2,
                skip_reshape=True,
                transformer_options=transformer_options,
            )
        self.assertEqual(output.shape, (1, 4, 16))
        sage.assert_not_called()

    def test_sdpa_keeps_fp32_qkv_unchanged(self):
        captured = {}

        def sdpa(q, k, v, *args, **kwargs):
            captured["q"] = q
            return q

        with mock.patch(
            "comfy.ldm.modules.attention.get_attention_function",
            side_effect=lambda name, default: sdpa if name == "pytorch" else default,
        ):
            override = attention_backends.make_attention_override("sdpa")

        q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
        out = override(lambda *args, **kwargs: None, q, q, q, 2, skip_reshape=True)

        self.assertIs(captured["q"], q)
        self.assertIs(out, q)

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

    def test_turing_auto_uses_bundled_sageattention2(self):
        model = FakeModel()
        q = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        with (
            mock.patch("turing_ops.is_supported_turing_device", return_value=True),
            mock.patch("turing_attention.available", return_value=True),
            mock.patch("turing_attention.preflight") as preflight,
            mock.patch("turing_attention.attention", return_value=q) as kernel,
        ):
            attention_backends.apply_attention_backend(model, "auto", device=torch.device("cuda", 0))
            override = model.model_options["transformer_options"]["optimized_attention_override"]
            out = override(lambda *args, **kwargs: None, q, q, q, 2, skip_reshape=True)

        self.assertIs(out, q)
        kernel.assert_called_once()
        preflight.assert_called_once_with(torch.device("cuda", 0))
        self.assertEqual(
            model.model_options["transformer_options"]["svdint4_attention_backend"],
            "turing_sage2",
        )

    def test_turing_explicit_non_sage_backend_is_honored(self):
        flash = lambda *args, **kwargs: None
        with (
            mock.patch("turing_ops.is_supported_turing_device", return_value=True),
            mock.patch(
                "comfy.ldm.modules.attention.get_attention_function",
                side_effect=lambda name, default: flash if name == "flash" else default,
            ),
        ):
            override = attention_backends.make_attention_override("flash_attn", device=torch.device("cuda", 0))
        self.assertEqual(override.svdint4_attention_backend, "flash_attn")


if __name__ == "__main__":
    unittest.main()
