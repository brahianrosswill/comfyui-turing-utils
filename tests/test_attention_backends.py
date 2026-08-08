from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import attention as attention_backends  # noqa: E402
from comfy.ldm.modules import attention as comfy_attention  # noqa: E402


class FakeModel:
    def __init__(self):
        self.model_options = {}


class AttentionBackendsTest(unittest.TestCase):
    def test_sparse_backend_requires_the_threshold_kernel_abi(self):
        sage_module = SimpleNamespace(sparse_available=lambda: True)
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.9.0"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertFalse(attention_backends.bundled_sparse_available())
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.11.1"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertFalse(attention_backends.bundled_sparse_available())
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.12.0"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertFalse(attention_backends.bundled_sparse_available())
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.13.0"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertTrue(attention_backends.bundled_sparse_available())

    def test_frame_sparse_backend_requires_015_kernel_abi(self):
        sage_module = SimpleNamespace(frame_sparse_available=lambda: True)
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.13.0"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertFalse(attention_backends.bundled_frame_sparse_available())
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.14.0"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertFalse(attention_backends.bundled_frame_sparse_available())
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.15.0"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertTrue(attention_backends.bundled_frame_sparse_available())

    def test_backend_choices_are_stable(self):
        self.assertEqual(
            attention_backends.attention_backend_choices(),
            ("auto", "sage_attn", "flash_attn", "sdpa"),
        )

    def test_aliases_normalize_to_node_options(self):
        self.assertEqual(attention_backends.normalize_attention_backend("default"), "auto")
        self.assertEqual(attention_backends.normalize_attention_backend("torch-sdpa"), "sdpa")
        self.assertEqual(attention_backends.normalize_attention_backend("sage attention"), "sage_attn")
        self.assertEqual(attention_backends.normalize_attention_backend("sage"), "sage_attn")
        self.assertEqual(attention_backends.normalize_attention_backend("sage_"), "sage_attn")
        self.assertEqual(attention_backends.normalize_attention_backend("turing-sage"), "sage_attn")
        self.assertEqual(attention_backends.normalize_attention_backend("flash-attn"), "flash_attn")
        with self.assertRaises(ValueError):
            attention_backends.normalize_attention_backend("sage1")
        with self.assertRaises(ValueError):
            attention_backends.normalize_attention_backend("sol-sparse")

    def test_backend_registration_rejects_alias_collisions_without_partial_registration(self):
        backend = attention_backends.AttentionBackend(
            option="test_collision",
            attention_function=None,
            label="sage",
        )
        with self.assertRaisesRegex(ValueError, "alias collision"):
            attention_backends.register_attention_backend(backend)
        self.assertNotIn("test_collision", attention_backends.attention_backend_choices())

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
            model.model_options["transformer_options"]["turing_utils_attention_backend"],
            "sage_attn",
        )
        self.assertEqual(
            model.model_options["transformer_options"]["turing_utils_attention_implementation"],
            "comfy:sage",
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
        self.assertEqual(transformer_options["turing_utils_attention_backend"], "flash_attn")
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
            model.model_options["transformer_options"]["turing_utils_attention_backend"],
            "sdpa",
        )

    def test_external_sage_sends_fp32_qkv_to_pytorch_attention(self):
        sage = mock.Mock()
        pytorch = mock.Mock(return_value="pytorch")
        original = mock.Mock(return_value="original")
        with mock.patch(
            "comfy.ldm.modules.attention.get_attention_function",
            side_effect=lambda name, default: (
                sage if name == "sage" else pytorch if name == "pytorch" else default
            ),
        ):
            override = attention_backends.make_attention_override("sage_attn")
            q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
            self.assertEqual(override(original, q, q, q, 2, skip_reshape=True), "pytorch")
        pytorch.assert_called_once()
        original.assert_not_called()
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
                    comfy_attention.attention_sage if name == "sage" else
                    comfy_attention.attention_pytorch if name == "pytorch" else default
                ),
            ),
            mock.patch("comfy.ldm.modules.attention.sageattn") as sage,
        ):
            attention_backends.apply_attention_backend(model, "sage_attn")
            transformer_options = model.model_options["transformer_options"]
            q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
            output = comfy_attention.optimized_attention(
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
        self.assertEqual(transformer_options["turing_utils_attention_backend"], "sdpa")
        self.assertIn("optimized_attention_override", transformer_options)

        q = torch.randn(1, 8, 16)
        k = torch.randn(1, 8, 16)
        v = torch.randn(1, 8, 16)
        out = comfy_attention.optimized_attention(q, k, v, heads=2, transformer_options=transformer_options)
        self.assertEqual(tuple(out.shape), (1, 8, 16))

    def test_turing_auto_uses_stable_bundled_sage_baseline(self):
        model = FakeModel()
        q = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention.bundled_available", return_value=True),
            mock.patch("attention.preflight_bundled") as preflight,
            mock.patch("attention.turing_sage_attention", return_value=q) as kernel,
        ):
            attention_backends.apply_attention_backend(model, "auto", device=torch.device("cuda", 0))
            override = model.model_options["transformer_options"]["optimized_attention_override"]
            out = override(lambda *args, **kwargs: None, q, q, q, 2, skip_reshape=True)

        self.assertIs(out, q)
        kernel.assert_called_once()
        preflight.assert_called_once_with(torch.device("cuda", 0))
        self.assertEqual(
            model.model_options["transformer_options"]["turing_utils_attention_backend"],
            "sage_attn",
        )
        self.assertEqual(
            model.model_options["transformer_options"]["turing_utils_attention_implementation"],
            "bundled_turing_sage",
        )

    def test_turing_explicit_sage_attn_selects_bundled_backend(self):
        model = FakeModel()
        q = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention.bundled_available", return_value=True),
            mock.patch("attention.preflight_bundled") as preflight,
            mock.patch("attention.turing_sage_attention", return_value=q) as kernel,
        ):
            attention_backends.apply_attention_backend(
                model, "sage_attn", device=torch.device("cuda", 0)
            )
            override = model.model_options["transformer_options"]["optimized_attention_override"]
            override(lambda *args, **kwargs: None, q, q, q, 2, skip_reshape=True)

        preflight.assert_called_once_with(torch.device("cuda", 0))
        self.assertNotIn("variant", kernel.call_args.kwargs)

    def test_legacy_sage_alias_uses_external_sage_on_non_turing_device(self):
        sage = mock.Mock(return_value="sage")
        with (
            mock.patch("attention.is_supported_turing_device", return_value=False),
            mock.patch(
                "comfy.ldm.modules.attention.get_attention_function",
                side_effect=lambda name, default: sage if name == "sage" else default,
            ),
        ):
            override = attention_backends.make_attention_override(
                "sage", device=torch.device("cuda", 0)
            )
        self.assertEqual(override.turing_utils_attention_backend, "sage_attn")
        self.assertEqual(override.turing_utils_attention_implementation, "comfy:sage")

    def test_turing_explicit_non_sage_backend_is_honored(self):
        flash = lambda *args, **kwargs: None
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch(
                "comfy.ldm.modules.attention.get_attention_function",
                side_effect=lambda name, default: flash if name == "flash" else default,
            ),
        ):
            override = attention_backends.make_attention_override("flash_attn", device=torch.device("cuda", 0))
        self.assertEqual(override.turing_utils_attention_backend, "flash_attn")

    def test_sparse_override_preflights_independent_kernel(self):
        q = torch.zeros((1, 2, 256, 128), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention.bundled_sparse_available", return_value=True),
            mock.patch("attention.preflight_bundled") as stable_preflight,
            mock.patch("attention.preflight_bundled_sparse") as preflight,
            mock.patch("attention.turing_sol_sparse_attention", return_value=q) as sparse,
        ):
            override = attention_backends.make_sparse_attention_override(
                torch.device("cuda", 0),
                min_sequence_tokens=8192,
                routing_threshold=0.85,
                prefix_policy="manual",
                manual_prefix_tokens=256,
                local_block_radius=2,
                temporal_neighbor_frames=2,
                skipped_residual="1x64",
                minimum_route_density=0.2,
                maximum_route_density=0.7,
                dense_prefix_steps=2,
                dense_suffix_steps=1,
                dense_prefix_layers=3,
                dense_suffix_layers=4,
                debug_route_density=True,
            )
            output = override(
                mock.Mock(),
                q,
                q,
                q,
                2,
                skip_reshape=True,
            )

        self.assertIs(output, q)
        sparse.assert_called_once()
        stable_preflight.assert_called_once_with(torch.device("cuda", 0))
        preflight.assert_called_once_with(torch.device("cuda", 0))
        self.assertEqual(sparse.call_args.kwargs["min_sequence_tokens"], 8192)
        self.assertEqual(sparse.call_args.kwargs["routing_threshold"], 0.85)
        self.assertEqual(sparse.call_args.kwargs["prefix_policy"], "manual")
        self.assertEqual(sparse.call_args.kwargs["manual_prefix_tokens"], 256)
        self.assertEqual(sparse.call_args.kwargs["local_block_radius"], 2)
        self.assertEqual(sparse.call_args.kwargs["temporal_neighbor_frames"], 2)
        self.assertEqual(sparse.call_args.kwargs["skipped_residual"], "1x64")
        self.assertEqual(sparse.call_args.kwargs["minimum_route_density"], 0.2)
        self.assertEqual(sparse.call_args.kwargs["maximum_route_density"], 0.7)
        self.assertTrue(sparse.call_args.kwargs["debug_route_density"])
        self.assertIsInstance(sparse.call_args.kwargs["debug_route_keys"], set)
        self.assertIsInstance(sparse.call_args.kwargs["debug_route_state"], dict)
        self.assertEqual(
            override.turing_utils_attention_implementation,
            "bundled_turing_sol_sparse_experimental",
        )

    def test_sparse_override_uses_stable_sage_for_first_and_last_layers(self):
        q = torch.zeros((1, 2, 4096, 128), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention.bundled_sparse_available", return_value=True),
            mock.patch("attention.preflight_bundled"),
            mock.patch("attention.preflight_bundled_sparse"),
            mock.patch("attention.turing_sage_attention", return_value=q) as stable,
            mock.patch("attention.turing_sol_sparse_attention", return_value=q) as sparse,
        ):
            override = attention_backends.make_sparse_attention_override(
                torch.device("cuda", 0), debug_route_density=True
            )
            for layer_index in (0, 1, 49):
                override(
                    mock.Mock(),
                    q,
                    q,
                    q,
                    2,
                    skip_reshape=True,
                    transformer_options={
                        "turing_utils_attention_layout": {
                            "layer_index": layer_index,
                            "layer_count": 50,
                        }
                    },
                )

        self.assertEqual(stable.call_count, 2)
        self.assertEqual(sparse.call_count, 1)
        self.assertEqual(
            sparse.call_args.kwargs["debug_context"]["last_sparse_layer"], 48
        )

    def test_experimental_sparse_rejects_non_turing_device(self):
        with (
            mock.patch("attention.is_supported_turing_device", return_value=False),
            self.assertRaisesRegex(RuntimeError, "sm75 Turing"),
        ):
            attention_backends.make_sparse_attention_override(torch.device("cuda", 0))

    def test_frame_sparse_override_preflights_and_forwards_parameters(self):
        q = torch.zeros((1, 2, 4096, 128), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention.bundled_frame_sparse_available", return_value=True),
            mock.patch("attention.preflight_bundled") as stable_preflight,
            mock.patch("attention.preflight_bundled_frame_sparse") as preflight,
            mock.patch("attention._sparse_dense_schedule", return_value=False) as schedule,
            mock.patch("attention.turing_frame_sparse_attention", return_value=q) as sparse,
        ):
            override = attention_backends.make_frame_sparse_attention_override(
                torch.device("cuda", 0),
                prefix_policy="manual",
                manual_prefix_tokens=256,
                temporal_window_frames=3,
                global_anchor_stride=16,
                rotate_global_anchors=False,
                sink_frames=2,
                dense_prefix_steps=2,
                dense_suffix_steps=1,
                dense_prefix_layers=3,
                dense_suffix_layers=4,
                debug_route_density=True,
            )
            output = override(mock.Mock(), q, q, q, 2, skip_reshape=True)

        self.assertIs(output, q)
        self.assertNotIn("track_step", schedule.call_args.kwargs)
        stable_preflight.assert_called_once_with(torch.device("cuda", 0))
        preflight.assert_called_once_with(torch.device("cuda", 0))
        self.assertEqual(sparse.call_args.kwargs["prefix_policy"], "manual")
        self.assertEqual(sparse.call_args.kwargs["manual_prefix_tokens"], 256)
        self.assertEqual(sparse.call_args.kwargs["temporal_window_frames"], 3)
        self.assertEqual(sparse.call_args.kwargs["global_anchor_stride"], 16)
        self.assertFalse(sparse.call_args.kwargs["rotate_global_anchors"])
        self.assertEqual(sparse.call_args.kwargs["sink_frames"], 2)
        self.assertTrue(sparse.call_args.kwargs["debug_route_density"])
        self.assertEqual(
            override.turing_utils_attention_implementation,
            "bundled_turing_frame_sparse_experimental",
        )

    def test_frame_sparse_override_uses_stable_sage_for_protected_layers(self):
        q = torch.zeros((1, 2, 4096, 128), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention.bundled_frame_sparse_available", return_value=True),
            mock.patch("attention.preflight_bundled"),
            mock.patch("attention.preflight_bundled_frame_sparse"),
            mock.patch("attention.turing_sage_attention", return_value=q) as stable,
            mock.patch("attention.turing_frame_sparse_attention", return_value=q) as sparse,
        ):
            override = attention_backends.make_frame_sparse_attention_override(
                torch.device("cuda", 0)
            )
            for layer_index in (0, 1, 49):
                override(
                    mock.Mock(),
                    q,
                    q,
                    q,
                    2,
                    skip_reshape=True,
                    transformer_options={
                        "turing_utils_attention_layout": {
                            "layer_index": layer_index,
                            "layer_count": 50,
                        }
                    },
                )

        self.assertEqual(stable.call_count, 2)
        self.assertEqual(sparse.call_count, 1)


if __name__ == "__main__":
    unittest.main()
