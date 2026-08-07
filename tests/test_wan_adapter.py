from __future__ import annotations

import sys
import unittest
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import wan_adapter  # noqa: E402


class WanMemoryPlanningTest(unittest.TestCase):
    def test_empty_context_has_no_synthetic_shape(self):
        self.assertIsNone(wan_adapter._context_latents_shape([]))

    def test_context_shape_aggregates_multiple_latents(self):
        latents = [
            torch.empty(1, 16, 5, 8, 8),
            torch.empty(1, 16, 3, 8, 8),
        ]
        self.assertEqual(
            wan_adapter._context_latents_shape(latents),
            [1, 16, 8 * 8 * 8],
        )

    def test_context_shape_sums_per_reference_padding(self):
        latents = [
            torch.empty(1, 16, 1, 3, 3),
            torch.empty(1, 16, 1, 3, 3),
        ]
        shape = wan_adapter._context_latents_shape(latents, (1, 2, 2))
        self.assertEqual(shape, [1, 16, 2 * 1 * 4 * 4])
        self.assertEqual(wan_adapter._shape_token_rows(shape, (1, 2, 2)), 8)

    def test_context_shape_uses_sampler_batch_hint(self):
        latents = [
            torch.empty(1, 16, 5, 8, 8),
            torch.empty(1, 16, 3, 8, 8),
        ]
        self.assertEqual(
            wan_adapter._context_latents_shape(
                latents, (1, 2, 2), estimate_batch_size=3
            ),
            [3, 16, 8 * 8 * 8],
        )

    def test_shape_token_rows_handles_spatial_and_flat_cond_shapes(self):
        self.assertEqual(
            wan_adapter._shape_token_rows([2, 16, 5, 8, 8], (1, 2, 2)),
            160,
        )
        self.assertEqual(
            wan_adapter._shape_token_rows([1, 16, 320], (1, 2, 2)),
            80,
        )

    def test_apply_installs_context_estimator_and_preserves_existing_shapes(self):
        from comfy.ldm.wan.model import WanModel

        diffusion = WanModel.__new__(WanModel)
        torch.nn.Module.__init__(diffusion)
        diffusion.patch_size = (1, 2, 2)

        class Base:
            memory_usage_factor_conds = ("existing",)
            diffusion_model = diffusion

            def extra_conds(self, **kwargs):
                return {}

            def extra_conds_shapes(self, **kwargs):
                return {"existing": [1, 1, 4]}

            def memory_required(self, input_shape, cond_shapes={}):
                return 100.0

        base = Base()
        patcher = SimpleNamespace(model=base)
        context = [torch.empty(1, 16, 3, 8, 8)]
        with (
            mock.patch("wan_adapter.is_supported_turing_device", return_value=True),
            mock.patch(
                "wan_adapter._quantized_wan_summary",
                return_value=(Counter({"w8a8": 2}), (4096,)),
            ),
            mock.patch(
                "wan_adapter._install_wan_self_attention_fusion", return_value=0
            ),
            mock.patch("wan_adapter.turing_int8_workspace_bytes", return_value=64.0),
        ):
            count = wan_adapter.apply_wan_adapter(patcher, torch.device("cuda", 0))

        self.assertEqual(count, 2)
        self.assertEqual(
            base.extra_conds_shapes(context_latents=context),
            {"existing": [1, 1, 4], "context_latents": [1, 16, 3 * 8 * 8]},
        )
        self.assertEqual(
            base.memory_usage_factor_conds, ("existing", "context_latents")
        )
        self.assertEqual(
            base.memory_required(
                [1, 16, 3, 8, 8],
                cond_shapes={"context_latents": [[1, 16, 3 * 8 * 8]]},
            ),
            1_572_964.0,
        )

    def test_workspace_uses_largest_live_accumulator_not_largest_output(self):
        class Base:
            def memory_required(self, input_shape, cond_shapes={}):
                return 0

        base = Base()
        base.memory_required = wan_adapter._make_memory_required(
            base, (1, 2, 2), (2048, 4096)
        )
        # 4096 columns cross the fixed-workspace threshold and need no global
        # accumulator, while 2048 columns still need a 40,960,000-byte buffer.
        self.assertEqual(
            base.memory_required([1, 16, 1, 100, 200]),
            5_000 * 2_048 * 4,
        )

    def test_outer_sample_wrapper_makes_context_estimate_batch_aware(self):
        base = SimpleNamespace()
        seen = []

        class Executor:
            def __call__(self, *args, **kwargs):
                seen.append(getattr(base, wan_adapter._MEMORY_CONTEXT_ATTR))
                return "ok"

        wrapper = wan_adapter._make_outer_sample_wrapper(base)
        result = wrapper(Executor(), torch.empty(3, 16, 5, 8, 8))
        self.assertEqual(result, "ok")
        self.assertEqual(seen, [{"batch_size": 3}])
        self.assertFalse(hasattr(base, wan_adapter._MEMORY_CONTEXT_ATTR))

    def test_runtime_context_condition_reports_padded_batch_shape(self):
        import comfy.conds

        latents = [
            torch.empty(1, 16, 1, 3, 3),
            torch.empty(1, 16, 1, 3, 3),
        ]

        class Base:
            def extra_conds(self, **kwargs):
                return {"context_latents": comfy.conds.CONDList(latents)}

        base = Base()
        base.extra_conds = wan_adapter._make_extra_conds(base, (1, 2, 2))
        cond = base.extra_conds()["context_latents"].process_cond(3)
        self.assertEqual(cond.size(), [1, 16, 3 * 2 * 1 * 4 * 4])

    def test_wan_attention_releases_qk_through_prequantized_sage_bridge(self):
        from comfy.ldm.wan import model as wan_model
        from comfyui_turing_utils_kernel.turing_sage import core, quant

        attention = SimpleNamespace(
            q=object(),
            k=object(),
            v=object(),
            o=torch.nn.Identity(),
            norm_q=torch.nn.Identity(),
            norm_k=torch.nn.Identity(),
            num_heads=2,
            head_dim=64,
        )
        attention.forward = lambda *_args, **_kwargs: None
        x = torch.zeros(1, 3, 128, dtype=torch.bfloat16)
        projected = tuple(torch.full_like(x, value) for value in (1, 2, 3))
        q_int8 = torch.zeros(1, 3, 2, 64, dtype=torch.int8)
        k_int8 = torch.zeros_like(q_int8)
        q_scale = torch.ones(1, 2, 4)
        k_scale = torch.ones(1, 2, 1)
        prequantized = torch.full_like(projected[0].reshape(1, 3, 2, 64), 4)

        with (
            mock.patch("wan_adapter.turing_linear_group", return_value=projected),
            mock.patch.object(wan_model, "apply_rope1", side_effect=lambda value, _freqs: value),
            mock.patch.object(wan_model, "optimized_attention") as generic_attention,
            mock.patch.object(
                quant,
                "per_warp_int8",
                return_value=(q_int8, q_scale, k_int8, k_scale),
            ) as quantize,
            mock.patch.object(
                core, "sageattn_prequantized", return_value=prequantized
            ) as sage,
        ):
            patched = wan_adapter._make_self_attention_forward(attention, wan_model)
            output = patched(
                x,
                None,
                transformer_options={
                    "turing_utils_attention_implementation": "bundled_turing_sage"
                },
            )

        self.assertEqual(output.shape, x.shape)
        quantize.assert_called_once()
        sage.assert_called_once()
        self.assertEqual(sage.call_args.args[4].shape, (1, 3, 2, 64))
        generic_attention.assert_not_called()

    def test_wan_install_only_patches_self_attention(self):
        from comfy.ldm.wan import model as wan_model

        attention = wan_model.WanSelfAttention.__new__(wan_model.WanSelfAttention)
        torch.nn.Module.__init__(attention)
        attention.q = torch.nn.Linear(1, 1)
        attention.k = torch.nn.Linear(1, 1)
        attention.v = torch.nn.Linear(1, 1)
        block = wan_model.WanAttentionBlock.__new__(wan_model.WanAttentionBlock)
        torch.nn.Module.__init__(block)

        diffusion = SimpleNamespace(
            named_modules=lambda: (
                ("blocks.0", block),
                ("blocks.0.self_attn", attention),
            )
        )
        patches = {}
        patcher = SimpleNamespace(
            add_object_patch=lambda name, value: patches.__setitem__(name, value)
        )

        with (
            mock.patch("wan_adapter.convrot_weight_kind", return_value="w4a4"),
            mock.patch(
                "wan_adapter._make_self_attention_forward", return_value="patched"
            ),
        ):
            count = wan_adapter._install_wan_self_attention_fusion(
                patcher, diffusion
            )

        self.assertEqual(count, 1)
        self.assertEqual(
            patches,
            {"diffusion_model.blocks.0.self_attn.forward": "patched"},
        )


if __name__ == "__main__":
    unittest.main()
