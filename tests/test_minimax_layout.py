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

import attention  # noqa: E402
import minimax_layout  # noqa: E402


class FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_options = None

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        self.seen_options = transformer_options
        return x


class FakeMiniMaxDiffusion(torch.nn.Module):
    patch_size = (1, 2, 2)

    def __init__(self, blocks=2):
        super().__init__()
        self.blocks = torch.nn.ModuleList(FakeBlock() for _ in range(blocks))


class FakeBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = FakeMiniMaxDiffusion()
        self.latent_shapes = None


class FakePatcher:
    def __init__(self, base=None):
        self.model = base if base is not None else FakeBase()
        self.load_device = torch.device("cuda", 0)
        self.model_options = {"transformer_options": {"existing": True}}
        self.object_patches = {}
        self.wrappers = {}

    def clone(self):
        cloned = FakePatcher(self.model)
        cloned.model_options = {
            "transformer_options": self.model_options["transformer_options"].copy()
        }
        cloned.object_patches = self.object_patches.copy()
        cloned.wrappers = self.wrappers.copy()
        return cloned

    def add_object_patch(self, name, value):
        self.object_patches[name] = value

    def add_wrapper_with_key(self, wrapper_type, key, value):
        self.wrappers[(wrapper_type, key)] = value


class MiniMaxLayoutProviderTest(unittest.TestCase):
    @staticmethod
    def _latent_shapes():
        return [
            torch.Size((1, 24, 7, 8, 10)),
            torch.Size((1, 32, 2, 12)),
        ]

    def _minimax_type_patch(self):
        import comfy.ldm.minimax.model as minimax_model

        return mock.patch.object(
            minimax_model,
            "MiniMaxH3Model",
            FakeMiniMaxDiffusion,
        )

    def test_official_model_patcher_gets_runtime_layout_without_custom_loader(self):
        patcher = FakePatcher()
        with self._minimax_type_patch():
            status = minimax_layout.ensure_minimax_attention_layout_provider(patcher)

        self.assertTrue(status.installed)
        self.assertEqual(status.model_kind, minimax_layout.MINIMAX_H3_LAYOUT_KIND)
        self.assertEqual(len(patcher.object_patches), 2)
        self.assertEqual(len(patcher.wrappers), 1)

        runtime_wrapper = next(iter(patcher.wrappers.values()))
        block_forward = patcher.object_patches["diffusion_model.blocks.1.forward"]
        options = {}
        x = torch.zeros((228, 8), dtype=torch.bfloat16)

        def executor(*args, **kwargs):
            return block_forward(
                x,
                x,
                [(0, 64, 0), (64, 88, 2), (88, 228, 3)],
                None,
                transformer_options=options,
            )

        output = runtime_wrapper(
            executor,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            0,
            latent_shapes=self._latent_shapes(),
        )

        self.assertIs(output, x)
        self.assertEqual(
            options[minimax_layout.ATTENTION_LAYOUT_KEY],
            {
                "provider": "minimax_h3",
                "dense_prefix_tokens": 88,
                "layer_index": 1,
                "layer_count": 2,
                "topology_start_tokens": 88,
                "topology_tokens": 140,
                "tokens_per_frame": 20,
                "spatial_tokens_height": 4,
                "spatial_tokens_width": 5,
            },
        )
        self.assertTrue(
            minimax_layout.has_complete_minimax_attention_layout(options, 228)
        )
        self.assertFalse(
            hasattr(patcher.model, minimax_layout.RUNTIME_CONTEXT_ATTR)
        )

    def test_provider_installation_is_keyed_and_idempotent(self):
        patcher = FakePatcher()
        with self._minimax_type_patch():
            first = minimax_layout.ensure_minimax_attention_layout_provider(patcher)
            first_forwards = dict(patcher.object_patches)
            first_wrappers = dict(patcher.wrappers)
            second = minimax_layout.ensure_minimax_attention_layout_provider(patcher)

        self.assertTrue(first.installed)
        self.assertTrue(second.installed)
        self.assertEqual(set(patcher.object_patches), set(first_forwards))
        self.assertEqual(set(patcher.wrappers), set(first_wrappers))
        for key, value in first_forwards.items():
            self.assertIs(patcher.object_patches[key], value)

    def test_publisher_drops_stale_topology_when_current_shapes_do_not_validate(self):
        options = {
            minimax_layout.ATTENTION_LAYOUT_KEY: {
                "provider": "minimax_h3",
                "dense_prefix_tokens": 88,
                "topology_start_tokens": 88,
                "topology_tokens": 140,
                "tokens_per_frame": 20,
                "spatial_tokens_height": 4,
                "spatial_tokens_width": 5,
                "layer_index": 1,
                "layer_count": 2,
                "extension_field": "keep",
            }
        }
        base = FakeBase()
        base.latent_shapes = self._latent_shapes()

        published = minimax_layout.publish_minimax_attention_layout(
            options,
            [(0, 64, 0), (64, 200, 3)],
            layer_index=0,
            layer_count=2,
            base_model=base,
            diffusion_model=base.diffusion_model,
        )

        self.assertFalse(published)
        layout = options[minimax_layout.ATTENTION_LAYOUT_KEY]
        self.assertEqual(layout["extension_field"], "keep")
        self.assertEqual(layout["dense_prefix_tokens"], 64)
        self.assertNotIn("topology_tokens", layout)
        self.assertFalse(
            minimax_layout.has_complete_minimax_attention_layout(options, 200)
        )

    def test_sparse_patch_installs_layout_provider_on_official_loader_model(self):
        model = FakePatcher()
        override = object()
        with (
            self._minimax_type_patch(),
            mock.patch("attention.make_sparse_attention_override", return_value=override),
        ):
            patched = attention.apply_sparse_attention_patch(model)

        options = patched.model_options["transformer_options"]
        self.assertEqual(
            options[minimax_layout.ATTENTION_LAYOUT_REQUIREMENT_KEY],
            minimax_layout.MINIMAX_H3_LAYOUT_KIND,
        )
        self.assertIs(options["optimized_attention_override"], override)
        self.assertEqual(len(patched.object_patches), 2)
        self.assertEqual(len(patched.wrappers), 1)
        self.assertFalse(model.object_patches)
        self.assertFalse(model.wrappers)

    def test_frame_sparse_patch_uses_the_same_official_loader_provider(self):
        model = FakePatcher()
        override = SimpleNamespace(
            turing_utils_frame_sparse_settings={
                "quality_profile": "custom",
                "sparse_pattern": "frame_window",
                "temporal_window_frames": 2,
                "global_anchor_stride": 12,
                "rotate_global_anchors": True,
                "sink_frames": 1,
                "radial_spatial_radius": 1,
                "radial_max_temporal_stride": 16,
                "dense_prefix_layers": 1,
                "dense_suffix_layers": 1,
            }
        )
        with (
            self._minimax_type_patch(),
            mock.patch(
                "attention.make_frame_sparse_attention_override",
                return_value=override,
            ),
        ):
            patched = attention.apply_frame_sparse_attention_patch(model)

        options = patched.model_options["transformer_options"]
        self.assertEqual(
            options[minimax_layout.ATTENTION_LAYOUT_REQUIREMENT_KEY],
            minimax_layout.MINIMAX_H3_LAYOUT_KIND,
        )
        self.assertIs(options["optimized_attention_override"], override)
        self.assertEqual(len(patched.object_patches), 2)
        self.assertEqual(len(patched.wrappers), 1)


if __name__ == "__main__":
    unittest.main()
