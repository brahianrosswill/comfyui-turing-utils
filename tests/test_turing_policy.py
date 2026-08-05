from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

import bf16_policy  # noqa: E402
import turing_ops  # noqa: E402
from comfy_kitchen.backends import cuda as kitchen_cuda  # noqa: E402


SUMMARY = SimpleNamespace(w4a4=1, w4a8=1, w8a8=1)
NO_CONVROT = SimpleNamespace(w4a4=0, w4a8=0, w8a8=0)
BF16_CONFIG = SimpleNamespace(supported_inference_dtypes=[torch.bfloat16, torch.float32])


class BF16PolicyTest(unittest.TestCase):
    def test_any_model_declaring_bf16_uses_it_on_non_turing_cuda(self):
        with (
            mock.patch("bf16_policy._explicit_dtype_override", return_value=False),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(8, 6)),
        ):
            dtype = bf16_policy.select_compute_dtype(
                BF16_CONFIG, torch.device("cuda", 0), NO_CONVROT
            )
        self.assertIs(dtype, torch.bfloat16)

    def test_model_without_declared_bf16_keeps_comfyui_policy(self):
        config = SimpleNamespace(supported_inference_dtypes=[torch.float16, torch.float32])
        with mock.patch("bf16_policy._explicit_dtype_override", return_value=False):
            dtype = bf16_policy.select_compute_dtype(config, torch.device("cuda", 0), NO_CONVROT)
        self.assertIsNone(dtype)

    def test_explicit_comfyui_dtype_override_wins(self):
        with mock.patch("bf16_policy._explicit_dtype_override", return_value=True):
            dtype = bf16_policy.select_compute_dtype(
                BF16_CONFIG, torch.device("cuda", 0), NO_CONVROT
            )
        self.assertIsNone(dtype)

    def test_supported_turing_preflights_generic_kernels(self):
        with (
            mock.patch("bf16_policy._explicit_dtype_override", return_value=False),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)),
            mock.patch("bf16_policy.is_supported_turing_device", return_value=True),
            mock.patch("bf16_policy._preflight_turing") as preflight,
        ):
            dtype = bf16_policy.select_compute_dtype(
                BF16_CONFIG, torch.device("cuda", 1), SUMMARY, "auto"
            )
        self.assertIs(dtype, torch.bfloat16)
        preflight.assert_called_once_with(SUMMARY, torch.device("cuda", 1), "auto")

    def test_turing_preflight_failure_does_not_silently_fallback_to_fp32(self):
        with (
            mock.patch("bf16_policy._explicit_dtype_override", return_value=False),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)),
            mock.patch("bf16_policy.is_supported_turing_device", return_value=True),
            mock.patch("bf16_policy._preflight_turing", side_effect=RuntimeError("attention self-test")),
            self.assertRaisesRegex(RuntimeError, "attention self-test"),
        ):
            bf16_policy.select_compute_dtype(
                BF16_CONFIG, torch.device("cuda", 0), SUMMARY, "auto"
            )

    def test_w8a8_alone_registers_turing_backend_before_preflight(self):
        summary = SimpleNamespace(w4a4=0, w4a8=0, w8a8=1)
        with (
            mock.patch(
                "comfy_kitchen.list_backends",
                return_value={
                    "cuda": {
                        "available": True,
                        "disabled": False,
                        "capabilities": ("int8_linear",),
                    }
                },
            ),
            mock.patch("bf16_policy.register_backend", return_value=True) as register,
            mock.patch("bf16_policy.backend_available", return_value=True),
            mock.patch("bf16_policy.preflight_kitchen") as preflight,
        ):
            bf16_policy._preflight_turing(summary, torch.device("cuda", 0), "sdpa")
        register.assert_called_once_with()
        preflight.assert_called_once_with(torch.device("cuda", 0), False, True)

    def test_gtx16_keeps_comfyui_fallback(self):
        with (
            mock.patch("bf16_policy._explicit_dtype_override", return_value=False),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)),
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce GTX 1660 Ti"),
            mock.patch("bf16_policy.is_supported_turing_device", return_value=False),
        ):
            dtype = bf16_policy.select_compute_dtype(
                BF16_CONFIG, torch.device("cuda", 0), NO_CONVROT
            )
        self.assertIsNone(dtype)

    def test_device_check_uses_requested_tensor_device(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", side_effect=lambda index: (7, 5) if index == 1 else (8, 6)),
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA T4"),
        ):
            self.assertTrue(turing_ops.is_supported_turing_device(torch.device("cuda", 1)))
            self.assertFalse(turing_ops.is_supported_turing_device(torch.device("cuda", 0)))

    def test_gtx_16_series_is_not_treated_as_supported_turing(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)),
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce GTX 1660 Ti"),
        ):
            self.assertFalse(turing_ops.is_supported_turing_device(torch.device("cuda", 0)))

    def test_w4a8_preflight_reports_missing_independent_kernel(self):
        with (
            mock.patch.object(turing_ops, "is_supported_turing_device", return_value=True),
            mock.patch.object(turing_ops, "_kernel_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "does not provide Turing W4A8"),
        ):
            turing_ops.preflight_w4a8(torch.device("cuda", 0))

    def test_int8_activation_uses_staged_bf16_rotation_above_48k_shared_limit(self):
        x = torch.empty((3, 5376), dtype=torch.bfloat16)
        with (
            mock.patch.object(kitchen_cuda, "quantize_int8_rowwise_convrot64") as fused,
            mock.patch.object(kitchen_cuda, "quantize_int8_convrot_staged", return_value=("q", "s")) as staged,
        ):
            result = turing_ops._quantize_turing_int8_activation(x, 256)
        self.assertEqual(result, ("q", "s"))
        staged.assert_called_once_with(x, 256)
        fused.assert_not_called()

    def test_int8_activation_keeps_small_rotation_under_48k(self):
        x = torch.empty((3, 256), dtype=torch.bfloat16)
        with (
            mock.patch.object(
                kitchen_cuda,
                "quantize_int8_rowwise_convrot64",
                return_value=("q", "s"),
            ) as fused,
            mock.patch.object(kitchen_cuda, "quantize_int8_convrot_staged") as staged,
        ):
            result = turing_ops._quantize_turing_int8_activation(x, 256)
        self.assertEqual(result, ("q", "s"))
        fused.assert_called_once_with(x, 256)
        staged.assert_not_called()

    def test_w8a8_uses_shared_staged_quantizer(self):
        x = torch.ones((2, 10752), dtype=torch.bfloat16)
        weight = torch.zeros((8, 5376), dtype=torch.int8)
        weight_scale = torch.ones((), dtype=torch.float32)
        qactivation = torch.zeros((2, 5376), dtype=torch.int8)
        activation_scale = torch.ones((2, 1), dtype=torch.float32)
        gemm_output = torch.zeros((2, 8), dtype=torch.float32)
        expected_output = gemm_output.to(torch.bfloat16)
        with (
            mock.patch.object(turing_ops, "is_supported_turing_device", return_value=True),
            mock.patch.object(
                turing_ops,
                "_quantize_turing_int8_activation",
                return_value=(qactivation, activation_scale),
            ) as quantize,
            mock.patch.object(
                kitchen_cuda,
                "_int4_linear_via_int8_values",
                return_value=gemm_output,
            ) as linear,
        ):
            result = turing_ops.int8_linear(
                x,
                weight,
                weight_scale,
                out_dtype=torch.bfloat16,
                convrot=True,
                convrot_groupsize=256,
                input_act="swiglu",
            )
        self.assertTrue(torch.equal(result, expected_output))
        self.assertIs(result.dtype, torch.bfloat16)
        activated = quantize.call_args.args[0]
        self.assertEqual(activated.shape, (2, 5376))
        self.assertTrue(
            torch.allclose(
                activated.float(),
                torch.full_like(activated.float(), 0.7310586),
                atol=1.0e-3,
                rtol=0.0,
            )
        )
        linear.assert_called_once()
        self.assertEqual(linear.call_args.args[3].shape, (8,))
        self.assertIs(linear.call_args.args[5], torch.float32)

    def test_w4a8_linear_uses_shared_staged_quantizer(self):
        x = torch.empty((2, 5376), dtype=torch.bfloat16)
        qweight = torch.empty((8, 2688), dtype=torch.int8)
        wscales = torch.ones(8, dtype=torch.float32)
        qactivation = torch.empty((2, 5376), dtype=torch.int8)
        activation_scale = torch.ones((2, 1), dtype=torch.float32)
        output = torch.zeros((2, 8), dtype=torch.bfloat16)
        linear = mock.Mock(return_value=output)
        with (
            mock.patch.object(turing_ops, "is_supported_turing_device", return_value=True),
            mock.patch.object(
                turing_ops,
                "_quantize_turing_int8_activation",
                return_value=(qactivation, activation_scale),
            ) as quantize,
            mock.patch.dict(sys.modules, {"svdint4": SimpleNamespace(turing_w4a8_linear=linear)}),
        ):
            result = turing_ops.convrot_w4a4_linear(
                x,
                qweight,
                wscales,
                convrot_groupsize=256,
                quant_group_size=64,
                linear_dtype="int8",
            )
        self.assertTrue(torch.equal(result, output))
        quantize.assert_called_once()
        self.assertEqual(quantize.call_args.args[0].data_ptr(), x.data_ptr())
        self.assertEqual(quantize.call_args.args[1], 256)
        linear.assert_called_once()
        self.assertIs(linear.call_args.args[0], qactivation)
        self.assertIs(linear.call_args.args[1], qweight)
        self.assertIs(linear.call_args.args[2], activation_scale)

    def test_w4a4_uses_grouped_rotation_when_fused_shared_memory_is_full(self):
        x = torch.empty((3, 16384), dtype=torch.bfloat16)
        rotated = torch.empty_like(x)
        with (
            mock.patch.object(kitchen_cuda, "quantize_int4_rowwise_convrot64") as fused,
            mock.patch.object(kitchen_cuda, "rotate_int8_convrot_weight", return_value=rotated) as rotate,
            mock.patch.object(kitchen_cuda, "quantize_int4_rowwise", return_value=("q", "s")) as quantize,
        ):
            result = turing_ops._quantize_turing_int4_activation(x, 256)
        self.assertEqual(result, ("q", "s"))
        fused.assert_not_called()
        rotate.assert_called_once_with(x, 256)
        quantize.assert_called_once_with(rotated)

    def test_w4a4_keeps_h3_bf16_rotation_fused_when_it_fits(self):
        x = torch.empty((3, 14336), dtype=torch.bfloat16)
        with (
            mock.patch.object(
                kitchen_cuda,
                "quantize_int4_rowwise_convrot64",
                return_value=("q", "s"),
            ) as fused,
            mock.patch.object(kitchen_cuda, "rotate_int8_convrot_weight") as rotate,
        ):
            result = turing_ops._quantize_turing_int4_activation(x, 256)
        self.assertEqual(result, ("q", "s"))
        fused.assert_called_once_with(x, 256)
        rotate.assert_not_called()

    def test_w4a4_linear_uses_int4_staged_helper(self):
        x = torch.empty((2, 256), dtype=torch.bfloat16)
        qweight = torch.empty((8, 128), dtype=torch.int8)
        wscales = torch.ones(8, dtype=torch.float32)
        qactivation = torch.empty((2, 128), dtype=torch.int8)
        activation_scale = torch.ones((2, 1), dtype=torch.float32)
        output = torch.zeros((2, 8), dtype=torch.bfloat16)
        with (
            mock.patch.object(turing_ops, "is_supported_turing_device", return_value=True),
            mock.patch.object(
                turing_ops,
                "_quantize_turing_int4_activation",
                return_value=(qactivation, activation_scale),
            ) as quantize,
            mock.patch.object(kitchen_cuda, "int4_linear", return_value=output) as linear,
        ):
            result = turing_ops.convrot_w4a4_linear(
                x,
                qweight,
                wscales,
                convrot_groupsize=256,
                quant_group_size=64,
                linear_dtype="int4",
            )
        self.assertTrue(torch.equal(result, output))
        quantize.assert_called_once()
        self.assertEqual(quantize.call_args.args[0].data_ptr(), x.data_ptr())
        self.assertEqual(quantize.call_args.args[1], 256)
        linear.assert_called_once()

    def test_non_turing_w8a8_delegates_to_kitchen(self):
        x = torch.empty((2, 256), dtype=torch.bfloat16)
        weight = torch.empty((8, 256), dtype=torch.int8)
        weight_scale = torch.ones((), dtype=torch.float32)
        with (
            mock.patch.object(turing_ops, "is_supported_turing_device", return_value=False),
            mock.patch.object(kitchen_cuda, "int8_linear", return_value="official") as official,
        ):
            result = turing_ops.int8_linear(
                x,
                weight,
                weight_scale,
                out_dtype=torch.bfloat16,
                convrot=True,
                convrot_groupsize=256,
            )
        self.assertEqual(result, "official")
        official.assert_called_once()

    def test_non_turing_w4a4_delegates_to_kitchen(self):
        x = torch.empty((2, 256), dtype=torch.bfloat16)
        qweight = torch.empty((8, 128), dtype=torch.int8)
        wscales = torch.ones(8, dtype=torch.float32)
        with (
            mock.patch.object(turing_ops, "is_supported_turing_device", return_value=False),
            mock.patch.object(
                kitchen_cuda,
                "convrot_w4a4_linear",
                return_value="official",
            ) as official,
        ):
            result = turing_ops.convrot_w4a4_linear(
                x,
                qweight,
                wscales,
                convrot_groupsize=256,
                quant_group_size=64,
                linear_dtype="int4",
            )
        self.assertEqual(result, "official")
        official.assert_called_once()

    def test_nonstandard_w4a8_group_size_delegates_to_kitchen(self):
        x = torch.empty((1, 256), dtype=torch.bfloat16)
        qweight = torch.empty((8, 128), dtype=torch.int8)
        wscales = torch.ones(8, dtype=torch.float32)
        with mock.patch.object(
            kitchen_cuda,
            "convrot_w4a4_linear",
            return_value="official",
        ) as official:
            result = turing_ops.convrot_w4a4_linear(
                x,
                qweight,
                wscales,
                convrot_groupsize=256,
                quant_group_size=128,
                linear_dtype="int8",
            )
        self.assertEqual(result, "official")
        official.assert_called_once_with(
            x,
            qweight,
            wscales,
            bias=None,
            convrot_groupsize=256,
            quant_group_size=128,
            linear_dtype="int8",
        )


if __name__ == "__main__":
    unittest.main()
