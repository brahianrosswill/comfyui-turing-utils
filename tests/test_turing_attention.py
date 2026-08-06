from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "kernel"))
sys.path.insert(0, str(PLUGIN_ROOT))

import attention as turing_attention  # noqa: E402


class TuringAttentionContractTest(unittest.TestCase):
    def test_fp32_uses_bf16_kernel_and_restores_fp32_output(self):
        q = torch.zeros((1, 1, 32, 128), dtype=torch.float32)
        kernel_output = torch.ones_like(q, dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sageattn", return_value=kernel_output) as sage,
        ):
            output = turing_attention.turing_sage_attention(
                mock.Mock(), q, q, q, 1,
                skip_reshape=True, skip_output_reshape=True
            )
        q_arg, k_arg, v_arg = sage.call_args.args[:3]
        self.assertEqual(q_arg.dtype, torch.bfloat16)
        self.assertEqual(k_arg.dtype, torch.bfloat16)
        self.assertEqual(v_arg.dtype, torch.bfloat16)
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, kernel_output.float())

    def test_unsupported_fp32_head_dimension_falls_back_without_casting(self):
        q = torch.zeros((1, 1, 32, 256), dtype=torch.float32)
        original = mock.Mock(return_value=q)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sageattn") as sage,
        ):
            output = turing_attention.turing_sage_attention(original, q, q, q, 1, skip_reshape=True)
        self.assertIs(output, q)
        original.assert_called_once()
        self.assertIs(original.call_args.args[0], q)
        sage.assert_not_called()

    def test_mask_uses_original_attention(self):
        q = torch.zeros((1, 1, 32, 128), dtype=torch.bfloat16)
        original = mock.Mock(return_value="fallback")
        with mock.patch("attention.is_supported_turing_device", return_value=True):
            output = turing_attention.turing_sage_attention(
                original, q, q, q, 1, mask=torch.ones(1), skip_reshape=True
            )
        self.assertEqual(output, "fallback")
        original.assert_called_once()

    def test_fp16_hnd_input_and_output_are_preserved(self):
        q = torch.zeros((1, 4, 32, 64), dtype=torch.float16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sageattn", return_value=q) as sage,
        ):
            output = turing_attention.turing_sage_attention(
                mock.Mock(), q, q[:, :2], q[:, :2], 4,
                skip_reshape=True, skip_output_reshape=True, enable_gqa=True
            )
        self.assertIs(output, q)
        self.assertEqual(sage.call_args.kwargs["tensor_layout"], "HND")

    def test_causal_and_scale_options_are_forwarded(self):
        q = torch.zeros((1, 4, 32, 64), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sageattn", return_value=q) as sage,
        ):
            turing_attention.turing_sage_attention(
                mock.Mock(),
                q,
                q,
                q,
                4,
                skip_reshape=True,
                is_causal=True,
                scale=0.125,
            )
        self.assertTrue(sage.call_args.kwargs["is_causal"])
        self.assertEqual(sage.call_args.kwargs["sm_scale"], 0.125)
        self.assertTrue(sage.call_args.kwargs["smooth_q"])
        self.assertTrue(sage.call_args.kwargs["smooth_k"])
        self.assertEqual(sage.call_args.kwargs["variant"], "sage2")

    def test_bf16_unreshaped_gqa_keeps_compact_kv_heads(self):
        q = torch.zeros((1, 32, 4 * 64), dtype=torch.bfloat16)
        k = torch.zeros((1, 16, 2 * 64), dtype=torch.bfloat16)
        v = torch.zeros_like(k)
        kernel_output = torch.zeros((1, 32, 4, 64), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sageattn", return_value=kernel_output) as sage,
        ):
            output = turing_attention.turing_sage_attention(
                mock.Mock(), q, k, v, 4, enable_gqa=True
            )
        q_arg, k_arg, v_arg = sage.call_args.args[:3]
        self.assertEqual(q_arg.shape, (1, 32, 4, 64))
        self.assertEqual(k_arg.shape, (1, 16, 2, 64))
        self.assertEqual(v_arg.shape, (1, 16, 2, 64))
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(sage.call_args.kwargs["tensor_layout"], "NHD")

    def test_unsupported_head_dimension_uses_original_attention(self):
        q = torch.zeros((1, 1, 32, 256), dtype=torch.bfloat16)
        original = mock.Mock(return_value="fallback")
        with mock.patch("attention.is_supported_turing_device", return_value=True):
            output = turing_attention.turing_sage_attention(original, q, q, q, 1, skip_reshape=True)
        self.assertEqual(output, "fallback")


if __name__ == "__main__":
    unittest.main()
