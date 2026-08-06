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
    @staticmethod
    def _w8a8_weight(out_features=4, in_features=8):
        from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

        qdata = torch.zeros((out_features, in_features), dtype=torch.int8)
        params = TensorWiseINT8Layout.Params(
            scale=torch.ones(out_features, dtype=torch.float32),
            orig_dtype=torch.bfloat16,
            orig_shape=(out_features, in_features),
            convrot=True,
            convrot_groupsize=256,
        )
        return QuantizedTensor(qdata, "TensorWiseINT8Layout", params)

    def test_fc2_audit_reports_actual_loaded_weight_contract(self):
        good = SimpleNamespace(
            mlp=SimpleNamespace(
                fc2=SimpleNamespace(
                    weight=self._w8a8_weight(),
                    quant_format="int8_tensorwise",
                )
            )
        )
        dense = SimpleNamespace(
            mlp=SimpleNamespace(
                fc2=SimpleNamespace(
                    weight=torch.zeros(4, 8, dtype=torch.bfloat16)
                )
            )
        )

        with self.assertLogs("comfyui-svdint4", level="INFO") as logs:
            eligible = turing_fusions._audit_turing_fc2([good, dense])

        self.assertEqual(eligible, 1)
        output = "\n".join(logs.output)
        self.assertIn("eligible_w8a8=1", output)
        self.assertIn("TensorWiseINT8Layout:1", output)
        self.assertIn("torch.bfloat16:2", output)
        self.assertIn("transposed=[False:2]", output)
        self.assertIn("convrot=[False:1,True:1]", output)
        self.assertIn("ConvRot group sizes: [256:1,None:1]", output)

    def test_direct_turing_input_act_preserves_cast_lifecycle(self):
        weight = self._w8a8_weight()
        linear = SimpleNamespace(weight=weight)
        x = torch.zeros((2, 16), dtype=torch.bfloat16)
        output = torch.empty((2, 4), dtype=torch.bfloat16)
        offload = (None, None, None)

        with (
            mock.patch(
                "comfy.ops.cast_bias_weight",
                return_value=(weight, None, offload),
            ) as cast,
            mock.patch("comfy.ops.uncast_bias_weight") as uncast,
            mock.patch("turing_ops.int8_linear", return_value=output) as int8_linear,
        ):
            result = turing_fusions.turing_linear_input_act(linear, x, "swiglu")

        self.assertIs(result, output)
        cast.assert_called_once_with(
            linear,
            x,
            offloadable=True,
            compute_dtype=torch.bfloat16,
            want_requant=True,
        )
        uncast.assert_called_once_with(linear, weight, None, offload)
        self.assertEqual(int8_linear.call_args.kwargs["input_act"], "swiglu")
        self.assertTrue(int8_linear.call_args.kwargs["convrot"])

    def test_direct_turing_input_act_keeps_dense_fallback(self):
        linear = SimpleNamespace(weight=torch.zeros(4, 8, dtype=torch.bfloat16))
        x = torch.zeros((2, 16), dtype=torch.bfloat16)
        output = torch.empty((2, 4), dtype=torch.bfloat16)

        with mock.patch("comfy.ops.linear_input_act", return_value=output) as fallback:
            result = turing_fusions.turing_linear_input_act(linear, x, "swiglu")

        self.assertIs(result, output)
        fallback.assert_called_once_with(linear, x, "swiglu")

    def test_direct_turing_input_act_handles_dequantized_cast(self):
        quantized = self._w8a8_weight()
        dense = torch.zeros((4, 8), dtype=torch.bfloat16)
        linear = SimpleNamespace(weight=quantized)
        x = torch.zeros((2, 16), dtype=torch.bfloat16)
        offload = (None, None, None)

        with (
            mock.patch(
                "comfy.ops.cast_bias_weight",
                return_value=(dense, None, offload),
            ),
            mock.patch("comfy.ops.uncast_bias_weight") as uncast,
            mock.patch("turing_ops.int8_linear") as int8_linear,
        ):
            result = turing_fusions.turing_linear_input_act(linear, x, "swiglu")

        self.assertEqual(result.shape, (2, 4))
        self.assertEqual(result.dtype, torch.bfloat16)
        int8_linear.assert_not_called()
        uncast.assert_called_once_with(linear, dense, None, offload)

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

    def test_apply_installs_direct_dispatch_only_for_eligible_fc2(self):
        import comfy.ldm.minimax.model as minimax_model
        import turing_ops

        class FakeMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Identity()
                self.fc2 = SimpleNamespace(
                    weight=FusionDispatchTest._w8a8_weight(),
                    quant_format="int8_tensorwise",
                )

            def forward(self, value):
                return value

        class FakeBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = torch.nn.Identity()
                self.norm2 = torch.nn.Identity()
                self.adaln_proj = torch.nn.Identity()
                self.attn = torch.nn.Identity()
                self.mlp = FakeMLP()

            def forward(self, value):
                return value

        block = FakeBlock()
        model = SimpleNamespace(model=torch.nn.Sequential(block))
        with (
            mock.patch.object(
                turing_ops,
                "is_supported_turing_device",
                return_value=True,
            ),
            mock.patch.object(minimax_model, "DiTBlock", FakeBlock),
            mock.patch.dict(
                sys.modules,
                {"svdint4": SimpleNamespace(turing_segmented_rms_adaln=mock.Mock())},
            ),
        ):
            count = turing_fusions.apply_turing_fusions(model, torch.device("cuda", 0))

        self.assertEqual(count, 1)
        self.assertTrue(callable(block.mlp._svdint4_original_forward))
        self.assertIs(block.mlp.forward.__func__, turing_fusions._turing_mlp_forward)


if __name__ == "__main__":
    unittest.main()
