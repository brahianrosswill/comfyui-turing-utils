from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import turing_fusions


class SegmentValidationTest(unittest.TestCase):
    def test_accepts_nonuniform_contiguous_segments(self):
        self.assertEqual(
            turing_fusions._normalized_segments(
                [(0, 1, 2), (1, 7, 0), (7, 19, 1)], 19, 3
            ),
            (0, 1, 2, 1, 7, 0, 7, 19, 1),
        )

    def test_rejects_gaps_and_invalid_modulation_rows(self):
        with self.assertRaisesRegex(ValueError, "contiguously"):
            turing_fusions._normalized_segments([(0, 2, 0), (3, 4, 0)], 4, 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            turing_fusions._normalized_segments([(0, 4, 1)], 4, 1)


class FusionDispatchTest(unittest.TestCase):
    def test_segmented_norm_preserves_cast_weight_lifecycle(self):
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        weight = torch.ones(8, dtype=torch.bfloat16)
        scale = torch.zeros(2, 8, dtype=torch.bfloat16)
        shift = torch.zeros(2, 8, dtype=torch.bfloat16)
        table = torch.tensor([[0, 2, 0], [2, 4, 1]], dtype=torch.int32)
        output = torch.empty_like(x)
        kernel = mock.Mock(return_value=output)
        norm = SimpleNamespace(eps=1.0e-5)

        with (
            mock.patch(
                "comfy.ops.cast_bias_weight",
                return_value=(weight, None, (None, None, None)),
            ) as cast,
            mock.patch("comfy.ops.uncast_bias_weight") as uncast,
            mock.patch.object(turing_fusions, "_segment_table", return_value=table),
            mock.patch.dict(
                sys.modules,
                {"svdint4": SimpleNamespace(turing_segmented_rms_adaln=kernel)},
            ),
        ):
            result = turing_fusions.segmented_rms_adaln(
                norm, x, shift, scale, [(0, 2, 0), (2, 4, 1)]
            )

        self.assertIs(result, output)
        cast.assert_called_once_with(norm, x, offloadable=True)
        uncast.assert_called_once_with(norm, weight, None, (None, None, None))
        kernel.assert_called_once()
        self.assertIs(kernel.call_args.args[0], x)
        self.assertIs(kernel.call_args.args[1], weight)
        self.assertIs(kernel.call_args.args[4], table)

    def test_apply_patches_only_compatible_turing_blocks(self):
        import comfy.ldm.minimax.model as minimax_model
        import turing_ops

        class FakeBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = torch.nn.Identity()
                self.norm2 = torch.nn.Identity()
                self.adaln_proj = torch.nn.Identity()
                self.attn = torch.nn.Identity()
                self.mlp = torch.nn.Identity()

            def forward(self, value):
                return value

        block = FakeBlock()
        model = SimpleNamespace(model=torch.nn.Sequential(block))
        kernel = mock.Mock()
        with (
            mock.patch.object(turing_ops, "is_supported_turing_device", return_value=True),
            mock.patch.object(minimax_model, "DiTBlock", FakeBlock),
            mock.patch.dict(
                sys.modules,
                {"svdint4": SimpleNamespace(turing_segmented_rms_adaln=kernel)},
            ),
        ):
            count = turing_fusions.apply_turing_fusions(model, torch.device("cuda", 0))

        self.assertEqual(count, 1)
        self.assertTrue(callable(block._svdint4_original_forward))
        self.assertIs(block.forward.__func__, turing_fusions._fused_block_forward)
        self.assertEqual(block._svdint4_turing_device_index, 0)


if __name__ == "__main__":
    unittest.main()
