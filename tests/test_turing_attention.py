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

    def test_causal_and_scale_options_are_forwarded_to_stable_sage(self):
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
        self.assertFalse(sage.call_args.kwargs["smooth_k"])
        self.assertNotIn("variant", sage.call_args.kwargs)

    def test_distinct_sequence_shapes_each_report_bundled_dispatch_once(self):
        q_short = torch.zeros((1, 4, 32, 64), dtype=torch.bfloat16)
        q_long = torch.zeros((1, 4, 96, 64), dtype=torch.bfloat16)
        turing_attention._LOGGED_TURING_KERNELS.clear()
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("attention._sageattn", side_effect=lambda q, *args, **kwargs: q),
            self.assertLogs("comfyui-turing-utils", level="INFO") as captured,
        ):
            turing_attention.turing_sage_attention(
                mock.Mock(), q_short, q_short, q_short, 4,
                skip_reshape=True, skip_output_reshape=True
            )
            turing_attention.turing_sage_attention(
                mock.Mock(), q_long, q_long, q_long, 4,
                skip_reshape=True, skip_output_reshape=True
            )
            turing_attention.turing_sage_attention(
                mock.Mock(), q_long, q_long, q_long, 4,
                skip_reshape=True, skip_output_reshape=True
            )

        dispatch_logs = [line for line in captured.output if "Bundled Turing Sage active" in line]
        self.assertEqual(len(dispatch_logs), 2)
        self.assertTrue(any("32" in line for line in dispatch_logs))
        self.assertTrue(any("96" in line for line in dispatch_logs))

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

    def test_experimental_sparse_uses_kernel_for_generic_long_attention(self):
        q = torch.zeros((1, 4, 4096, 128), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sol_sparse_sageattn", return_value=q) as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(),
                q,
                q,
                q,
                4,
                skip_reshape=True,
                skip_output_reshape=True,
            )
        self.assertIs(output, q)
        self.assertEqual(sparse.call_args.kwargs["prefix_tokens"], 512)
        self.assertEqual(sparse.call_args.kwargs["tau"], 1.0)

    def test_experimental_sparse_keeps_short_calls_dense(self):
        q = torch.zeros((1, 4, 256, 128), dtype=torch.bfloat16)
        baseline = mock.Mock(return_value=q)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention.turing_sage_attention", baseline),
            mock.patch("attention._sol_sparse_sageattn") as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(), q, q, q, 4,
                skip_reshape=True,
            )
        self.assertIs(output, q)
        baseline.assert_called_once()
        sparse.assert_not_called()

    def test_experimental_sparse_unreshaped_gqa_restores_output_layout(self):
        q = torch.zeros((1, 4096, 4 * 128), dtype=torch.float16)
        k = torch.zeros((1, 4096, 2 * 128), dtype=torch.float16)
        v = torch.zeros_like(k)
        kernel_output = torch.zeros((1, 4, 4096, 128), dtype=torch.float16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sol_sparse_sageattn", return_value=kernel_output) as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(), q, k, v, 4, enable_gqa=True
            )
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(sparse.call_args.args[1].shape, (1, 2, 4096, 128))

    def test_experimental_sparse_fallback_preserves_original_layout(self):
        q = torch.zeros((1, 4096, 4 * 64), dtype=torch.bfloat16)
        baseline = mock.Mock(return_value=q)
        with (
            mock.patch("attention.turing_sage_attention", baseline),
            mock.patch("attention._sol_sparse_sageattn") as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(), q, q, q, 4
            )
        self.assertIs(output, q)
        baseline.assert_called_once()
        self.assertIs(baseline.call_args.args[1], q)
        sparse.assert_not_called()

    def test_experimental_sparse_forwards_patch_parameters(self):
        q = torch.zeros((1, 4, 8192, 128), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sol_sparse_sageattn", return_value=q) as sparse,
        ):
            turing_attention.turing_sol_sparse_attention(
                mock.Mock(),
                q,
                q,
                q,
                4,
                skip_reshape=True,
                dense_prefix_tokens=192,
                route_threshold=1.75,
                min_sequence_tokens=8000,
            )
        self.assertEqual(sparse.call_args.kwargs["prefix_tokens"], 192)
        self.assertEqual(sparse.call_args.kwargs["tau"], 1.75)

    def test_experimental_sparse_fp32_uses_bf16_boundary_and_restores_output(self):
        q = torch.zeros((1, 1, 4096, 128), dtype=torch.float32)
        kernel_output = torch.ones_like(q, dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sol_sparse_sageattn", return_value=kernel_output) as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(),
                q,
                q,
                q,
                1,
                skip_reshape=True,
                skip_output_reshape=True,
            )
        self.assertTrue(all(value.dtype == torch.bfloat16 for value in sparse.call_args.args[:3]))
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, kernel_output.float())


if __name__ == "__main__":
    unittest.main()
