from __future__ import annotations

import sys
import unittest
import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.adapters.minimax import acceleration as minimax_adapter  # noqa: E402


def _convrot_weight(kind: str):
    from comfy.quant_ops import (
        QuantizedTensor,
        TensorCoreConvRotW4A4Layout,
        TensorWiseINT8Layout,
    )

    if kind == "w8a8":
        params = TensorWiseINT8Layout.Params(
            scale=torch.ones(4, dtype=torch.float32),
            orig_dtype=torch.bfloat16,
            orig_shape=(4, 256),
            convrot=True,
            convrot_groupsize=256,
        )
        return QuantizedTensor(
            torch.zeros((4, 256), dtype=torch.int8),
            "TensorWiseINT8Layout",
            params,
        )

    linear_dtype = {"w4a4": "int4", "w4a8": "int8"}[kind]
    params = TensorCoreConvRotW4A4Layout.Params(
        scale=torch.ones((4, 4), dtype=torch.float32),
        orig_dtype=torch.bfloat16,
        orig_shape=(4, 256),
        convrot_groupsize=256,
        quant_group_size=64,
        linear_dtype=linear_dtype,
    )
    return QuantizedTensor(
        torch.zeros((4, 128), dtype=torch.uint8),
        "TensorCoreConvRotW4A4Layout",
        params,
    )


class FakePatcher:
    def __init__(self, root: torch.nn.Module):
        self.model = root
        self.object_patches = {}
        self.wrappers = {}

    def add_object_patch(self, name, value):
        self.object_patches[name] = value

    def add_wrapper_with_key(self, wrapper_type, key, value):
        self.wrappers[(wrapper_type, key)] = value


class MiniMaxAdapterTest(unittest.TestCase):
    @staticmethod
    def _diffusion_spec():
        return SimpleNamespace(
            patch_size=(1, 2, 2),
            hidden_size=5376,
            latents_dim=24,
            audio_latents_dim=32,
        )

    @staticmethod
    def _latent_shapes():
        return [
            torch.Size((1, 24, 7, 8, 10)),
            torch.Size((1, 32, 2, 12)),
        ]

    @staticmethod
    def _types(kind: str):
        class FakeMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Identity()
                self.fc2 = SimpleNamespace(weight=_convrot_weight(kind))

            def forward(self, x):
                return x

        class FakeBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = torch.nn.Identity()
                self.norm2 = torch.nn.Identity()
                self.adaln_proj = torch.nn.Identity()
                self.attn = torch.nn.Identity()
                self.mlp = FakeMLP()

            def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
                return x

        return FakeBlock, FakeMLP

    def test_all_convrot_formats_install_object_patches_without_mutating_modules(self):
        import comfy.ldm.minimax.model as minimax_model

        for kind in ("w8a8", "w4a4", "w4a8"):
            with self.subTest(kind=kind):
                FakeBlock, _ = self._types(kind)
                root = torch.nn.Module()
                root.block = FakeBlock()
                patcher = FakePatcher(root)
                original_block_forward = root.block.forward
                original_mlp_forward = root.block.mlp.forward
                with (
                    mock.patch("comfyui_turing_utils.adapters.minimax.acceleration.is_supported_turing_device", return_value=True),
                    mock.patch.object(minimax_model, "DiTBlock", FakeBlock),
                    mock.patch.dict(
                        sys.modules,
                        {"comfyui_turing_utils_kernel": SimpleNamespace(turing_segmented_rms_adaln=mock.Mock())},
                    ),
                ):
                    count = minimax_adapter.apply_minimax_adapter(
                        patcher, torch.device("cuda", 0)
                    )

                self.assertEqual(count, 1)
                self.assertEqual(
                    set(patcher.object_patches),
                    {"block.forward", "block.mlp.forward"},
                )
                self.assertIs(root.block.forward.__func__, original_block_forward.__func__)
                self.assertIs(root.block.mlp.forward.__func__, original_mlp_forward.__func__)

    def test_changed_block_contract_disables_adapter_cleanly(self):
        import comfy.ldm.minimax.model as minimax_model

        class ChangedBlock(torch.nn.Module):
            def forward(self, x, new_required_argument):
                return x

        root = torch.nn.Module()
        root.block = ChangedBlock()
        patcher = FakePatcher(root)
        with (
            mock.patch("comfyui_turing_utils.adapters.minimax.acceleration.is_supported_turing_device", return_value=True),
            mock.patch.object(minimax_model, "DiTBlock", ChangedBlock),
            self.assertLogs("comfyui-turing-utils", level="WARNING"),
        ):
            count = minimax_adapter.apply_minimax_adapter(
                patcher, torch.device("cuda", 0)
            )

        self.assertEqual(count, 0)
        self.assertFalse(patcher.object_patches)

    def test_runtime_audit_reports_a_complete_fused_window_once(self):
        audit = minimax_adapter._RuntimeDispatchAudit(expected_blocks=2, expected_mlps=2)
        x = torch.zeros((3, 256), dtype=torch.bfloat16)
        with self.assertLogs("comfyui-turing-utils", level="INFO") as captured:
            audit.record("block", True, x)
            audit.record("mlp", True, x)
            audit.record("block", True, x)
            audit.record("mlp", True, x)
            audit.record("mlp", False, x, "should_not_be_recorded")

        messages = "\n".join(captured.output)
        self.assertIn("phase=block fused=2 fallback=0", messages)
        self.assertIn("phase=mlp fused=2 fallback=0", messages)
        self.assertNotIn("should_not_be_recorded", messages)

    def test_runtime_audit_exposes_mlp_dtype_fallback(self):
        FakeBlock, _ = self._types("w8a8")
        mlp = FakeBlock().mlp
        audit = minimax_adapter._RuntimeDispatchAudit(expected_blocks=0, expected_mlps=1)
        patched = minimax_adapter._make_mlp_forward(mlp, audit)
        x = torch.zeros((3, 256), dtype=torch.float32)

        with self.assertLogs("comfyui-turing-utils", level="WARNING") as captured:
            output = patched(x)

        self.assertIs(output, x)
        self.assertIn("phase=mlp fused=0 fallback=1", "\n".join(captured.output))
        self.assertIn("dtype=torch.float32", "\n".join(captured.output))

    def test_runtime_audit_keeps_fused_w8a8_mlp_dispatch(self):
        FakeBlock, _ = self._types("w8a8")
        mlp = FakeBlock().mlp
        audit = minimax_adapter._RuntimeDispatchAudit(expected_blocks=0, expected_mlps=1)
        patched = minimax_adapter._make_mlp_forward(mlp, audit)
        x = torch.zeros((3, 256), dtype=torch.bfloat16)
        sentinel = object()

        with (
            mock.patch(
                "comfyui_turing_utils.adapters.minimax.acceleration.turing_linear_input_act",
                return_value=sentinel,
            ) as fused,
            self.assertLogs("comfyui-turing-utils", level="INFO") as captured,
        ):
            output = patched(x)

        self.assertIs(output, sentinel)
        fused.assert_called_once_with(mlp.fc2, x, "swiglu")
        self.assertIn("phase=mlp fused=1 fallback=0", "\n".join(captured.output))

    def test_block_forward_publishes_semantic_attention_prefix(self):
        FakeBlock, _ = self._types("w8a8")
        block = FakeBlock()
        audit = minimax_adapter._RuntimeDispatchAudit(expected_blocks=1, expected_mlps=0)
        patched = minimax_adapter._make_block_forward(
            block,
            0,
            mock.Mock(),
            audit,
        )
        options = {}
        x = torch.zeros((32, 256), dtype=torch.bfloat16)
        patched(
            x,
            x,
            [(0, 8, 0), (8, 12, 1), (12, 16, 2), (16, 32, 3)],
            None,
            transformer_options=options,
        )
        self.assertEqual(
            options[minimax_adapter._ATTENTION_LAYOUT_KEY],
            {
                "provider": "minimax_h3",
                "dense_prefix_tokens": 16,
                "layer_index": 0,
                "layer_count": 0,
            },
        )

    def test_memory_rows_match_packed_layout_for_multimodal_references(self):
        from comfy.ldm.minimax.model import PackedLayout

        keyframes = [
            {"resolved_frame_index": 0},
            {"resolved_frame_index": 99},
        ]
        refs = [
            {"kind": "image", "latent_h": 6, "latent_w": 8},
            {
                "kind": "video_audio",
                "latent_t": 3,
                "latent_h": 4,
                "latent_w": 6,
                "ref_audio_t": 5,
            },
            {"kind": "audio", "ref_audio_t": 7},
        ]
        kwargs = {
            "cross_attn": torch.empty(1, 11, 5120),
            "minimax_keyframes": keyframes,
            "minimax_refs": refs,
        }
        plan = minimax_adapter._minimax_memory_shape(
            kwargs, self._latent_shapes(), self._diffusion_spec()
        )
        layout = PackedLayout(
            11,
            7,
            8,
            10,
            12,
            keyframes=keyframes,
            refs=refs,
            frame_count=100,
        )

        self.assertEqual(plan.full_rows, layout.seq_len)
        self.assertEqual(plan.target_visual_rows, 7 * 4 * 5)
        self.assertEqual(plan.target_audio_rows, 24)
        self.assertEqual(plan.visual_condition_rows, 2 * 20 + 12 + 18)
        self.assertEqual(plan.audio_condition_rows, 10 + 14)
        self.assertEqual(
            plan.equivalent_area,
            math.ceil(
                (24 * 7 * 8 * 10 + 32 * 2 * 12)
                * (11 + 40 + 12 + 18 + 10 + 14)
                / (7 * 20 + 24)
            ),
        )

    def test_memory_rows_cover_zero_one_and_many_reference_inputs(self):
        spec = self._diffusion_spec()
        shapes = self._latent_shapes()
        baseline = minimax_adapter._minimax_memory_shape(
            {"cross_attn": torch.empty(1, 5, 5120)}, shapes, spec
        )
        one = minimax_adapter._minimax_memory_shape(
            {
                "cross_attn": torch.empty(1, 7, 5120),
                "minimax_refs": [
                    {"kind": "image", "latent_h": 6, "latent_w": 8}
                ],
            },
            shapes,
            spec,
        )
        many = minimax_adapter._minimax_memory_shape(
            {
                "cross_attn": torch.empty(1, 13, 5120),
                "minimax_refs": [
                    {"kind": "image", "latent_h": 6, "latent_w": 8},
                    {
                        "kind": "video",
                        "latent_t": 3,
                        "latent_h": 4,
                        "latent_w": 6,
                        "ref_audio_t": 0,
                    },
                    {"kind": "audio", "ref_audio_t": 7},
                ],
            },
            shapes,
            spec,
        )

        self.assertEqual(baseline.full_rows - baseline.target_rows, 5)
        self.assertEqual(one.full_rows - one.target_rows, 7 + 12)
        self.assertEqual(many.full_rows - many.target_rows, 13 + 12 + 18 + 14)
        self.assertLess(baseline.equivalent_area, one.equivalent_area)
        self.assertLess(one.equivalent_area, many.equivalent_area)

    def test_memory_required_uses_buffer_floor_and_all_w8_outputs(self):
        class Base:
            memory_usage_factor_conds = (minimax_adapter._MEMORY_SHAPE_KEY,)

            def get_dtype_inference(self):
                return torch.bfloat16

            def memory_required(self, input_shape, cond_shapes={}):
                shapes = [input_shape]
                shapes.extend(cond_shapes.get(minimax_adapter._MEMORY_SHAPE_KEY, ()))
                return sum(shape[0] * math.prod(shape[2:]) for shape in shapes)

        plan = minimax_adapter._minimax_memory_shape(
            {
                "cross_attn": torch.empty(1, 11, 5120),
                "minimax_refs": [
                    {"kind": "image", "latent_h": 6, "latent_w": 8}
                ],
            },
            self._latent_shapes(),
            self._diffusion_spec(),
        )
        base = Base()
        base.memory_required = minimax_adapter._make_memory_required(
            base, (2048, 4096)
        )
        target_area = 24 * 7 * 8 * 10 + 32 * 2 * 12
        with mock.patch(
            "comfyui_turing_utils.adapters.minimax.acceleration.turing_int8_workspace_bytes",
            side_effect=lambda rows, output: rows + output,
        ):
            required = base.memory_required(
                [1, 1, target_area],
                cond_shapes={minimax_adapter._MEMORY_SHAPE_KEY: [plan]},
            )

        explicit = plan.explicit_condition_bytes(2)
        expected = target_area + max(plan.equivalent_area, explicit)
        expected += plan.full_rows + 4096
        self.assertEqual(required, expected)

    def test_outer_sample_wrapper_exposes_and_restores_latent_shapes(self):
        base = SimpleNamespace()
        latent_shapes = self._latent_shapes()
        seen = []

        class Executor:
            def __call__(self, *args, **kwargs):
                seen.append(getattr(base, minimax_adapter._MEMORY_CONTEXT_ATTR))
                return "ok"

        wrapper = minimax_adapter._make_outer_sample_wrapper(base)
        result = wrapper(
            Executor(),
            torch.empty(1, 1, 16),
            None,
            None,
            None,
            None,
            None,
            False,
            0,
            latent_shapes=latent_shapes,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(seen[0]["latent_shapes"], latent_shapes)
        self.assertFalse(hasattr(base, minimax_adapter._MEMORY_CONTEXT_ATTR))

    def test_temporal_topology_matches_the_target_video_tail(self):
        base = SimpleNamespace()
        latent_shapes = self._latent_shapes()
        setattr(
            base,
            minimax_adapter._MEMORY_CONTEXT_ATTR,
            {"latent_shapes": latent_shapes},
        )
        # Seven frames at 8x10 latent resolution with a 1x2x2 patch become
        # seven contiguous 20-token frames.  Only the final target-video
        # segment is described; text, references and target audio stay prefix.
        topology = minimax_adapter._minimax_temporal_topology(
            base,
            self._diffusion_spec(),
            [(0, 64, 0), (64, 88, 2), (88, 228, 3)],
        )

        self.assertEqual(
            topology,
            {
                "topology_start_tokens": 88,
                "topology_tokens": 140,
                "tokens_per_frame": 20,
                "spatial_tokens_height": 4,
                "spatial_tokens_width": 5,
            },
        )

    def test_temporal_topology_rejects_a_mismatched_packed_segment(self):
        base = SimpleNamespace()
        setattr(
            base,
            minimax_adapter._MEMORY_CONTEXT_ATTR,
            {"latent_shapes": self._latent_shapes()},
        )

        self.assertEqual(
            minimax_adapter._minimax_temporal_topology(
                base,
                self._diffusion_spec(),
                [(0, 64, 0), (64, 200, 3)],
            ),
            {},
        )

    def test_runtime_memory_condition_reports_the_same_plan(self):
        class Base:
            def extra_conds(self, **kwargs):
                return {"existing": object()}

        base = Base()
        base.extra_conds = minimax_adapter._make_extra_conds(
            base, self._diffusion_spec()
        )
        out = base.extra_conds(
            cross_attn=torch.empty(1, 9, 5120),
            latent_shapes=self._latent_shapes(),
            minimax_refs=[
                {"kind": "image", "latent_h": 6, "latent_w": 8}
            ],
        )
        cond = out[minimax_adapter._MEMORY_SHAPE_KEY]
        expected = minimax_adapter._minimax_memory_shape(
            {
                "cross_attn": torch.empty(1, 9, 5120),
                "minimax_refs": [
                    {"kind": "image", "latent_h": 6, "latent_w": 8}
                ],
            },
            self._latent_shapes(),
            self._diffusion_spec(),
        )

        self.assertEqual(cond.size().full_rows, expected.full_rows)
        self.assertIs(cond.process_cond(1), cond)
        self.assertIn("existing", out)

    def test_memory_planning_installs_once_without_own_runtime_wrapper(self):
        class Diffusion(torch.nn.Module):
            patch_size = (1, 2, 2)
            hidden_size = 5376
            latents_dim = 24
            audio_latents_dim = 32

        class Base(torch.nn.Module):
            memory_usage_factor_conds = ("existing",)

            def __init__(self):
                super().__init__()
                self.diffusion_model = Diffusion()

            def extra_conds(self, **kwargs):
                return {}

            def extra_conds_shapes(self, **kwargs):
                return {"existing": [1, 1, 4]}

            def memory_required(self, input_shape, cond_shapes={}):
                return 100

            def get_dtype_inference(self):
                return torch.bfloat16

        base = Base()
        patcher = FakePatcher(base)
        self.assertTrue(
            minimax_adapter._install_memory_planning(
                patcher, base, base.diffusion_model
            )
        )
        self.assertFalse(
            minimax_adapter._install_memory_planning(
                patcher, base, base.diffusion_model
            )
        )
        self.assertEqual(
            base.memory_usage_factor_conds,
            ("existing", minimax_adapter._MEMORY_SHAPE_KEY),
        )
        # Runtime shape publication now belongs to the loader-independent H3
        # layout provider rather than the quantization memory planner.
        self.assertEqual(len(patcher.wrappers), 0)


if __name__ == "__main__":
    unittest.main()
