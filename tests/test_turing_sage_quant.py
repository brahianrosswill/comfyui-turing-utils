from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "kernel"))

from comfyui_turing_utils_kernel import turing_sage  # noqa: E402
from comfyui_turing_utils_kernel.turing_sage import quant  # noqa: E402


class TuringSageQuantContractTest(unittest.TestCase):
    def test_compiled_attention_facades_have_fake_tensor_contracts(self):
        q = torch.empty((1, 4, 129, 128), dtype=torch.bfloat16, device="meta")
        k = torch.empty((1, 2, 151, 128), dtype=torch.bfloat16, device="meta")
        v = torch.empty_like(k)

        stable = turing_sage.sageattn_compiled(q, k, v)
        w8a8 = turing_sage.w8a8attn_compiled(q, k, v)

        self.assertEqual(stable.shape, q.shape)
        self.assertEqual(stable.dtype, q.dtype)
        self.assertEqual(stable.device.type, "meta")
        self.assertEqual(w8a8.shape, q.shape)
        self.assertEqual(w8a8.dtype, q.dtype)
        self.assertEqual(w8a8.device.type, "meta")

    def test_compiled_attention_facade_is_a_fullgraph_leaf(self):
        q = torch.empty((1, 4, 129, 128), dtype=torch.bfloat16, device="meta")
        k = torch.empty((1, 2, 151, 128), dtype=torch.bfloat16, device="meta")
        v = torch.empty_like(k)

        compiled = torch.compile(
            lambda query, key, value: turing_sage.sageattn_compiled(
                query, key, value
            ),
            backend="eager",
            fullgraph=True,
        )
        output = compiled(q, k, v)
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(output.device.type, "meta")

    def test_split_quantizers_preserve_production_scale_contract(self):
        q = torch.empty((2, 4, 129, 128), dtype=torch.bfloat16)
        k = torch.empty((2, 2, 151, 128), dtype=torch.bfloat16)
        with (
            mock.patch.object(quant._fused, "quant_per_warp_int8_cuda") as quant_q,
            mock.patch.object(quant._fused, "quant_per_block_int8_cuda") as quant_k,
        ):
            q_int8, q_scale = quant.quantize_query_per_warp(q)
            k_int8, k_scale = quant.quantize_key_per_block(k)

        self.assertEqual(q_int8.shape, q.shape)
        self.assertEqual(k_int8.shape, k.shape)
        self.assertEqual(q_scale.shape, (2, 4, 12))
        self.assertEqual(k_scale.shape, (2, 2, 3))
        quant_q.assert_called_once_with(q, q_int8, q_scale, 64, 16, 1)
        quant_k.assert_called_once_with(k, k_int8, k_scale, 64, 1)

    def test_production_per_warp_scale_shapes(self):
        q = torch.empty((2, 4, 65, 64), dtype=torch.float16)
        k = torch.empty((2, 2, 73, 64), dtype=torch.float16)
        with (
            mock.patch.object(quant._fused, "quant_per_warp_int8_cuda") as quant_q,
            mock.patch.object(quant._fused, "quant_per_block_int8_cuda") as quant_k,
        ):
            q_int8, q_scale, k_int8, k_scale = quant.per_warp_int8(q, k)

        self.assertEqual(q_int8.shape, q.shape)
        self.assertEqual(k_int8.shape, k.shape)
        self.assertEqual(q_scale.shape, (2, 4, 8))
        self.assertEqual(k_scale.shape, (2, 2, 2))
        quant_q.assert_called_once()
        quant_k.assert_called_once()

    def test_hadamard_quantizer_fuses_qk_and_preserves_scale_contract(self):
        q = torch.empty((2, 4, 129, 128), dtype=torch.bfloat16)
        k = torch.empty((2, 2, 151, 128), dtype=torch.bfloat16)
        with mock.patch.object(
            quant._fused, "quant_qk_per_warp_int8_rotated_anchored_cuda"
        ) as fused:
            q_int8, q_scale, k_int8, k_scale = quant.per_warp_int8_hadamard(q, k)

        self.assertEqual(q_int8.shape, q.shape)
        self.assertEqual(k_int8.shape, k.shape)
        self.assertEqual(q_scale.shape, (2, 4, 12))
        self.assertEqual(k_scale.shape, (2, 2, 3))
        args = fused.call_args.args
        for actual, expected in zip(
            args[:6], (q, k, q_int8, k_int8, q_scale, k_scale)
        ):
            self.assertIs(actual, expected)
        self.assertEqual(args[6].shape, (2, 2))
        self.assertEqual(args[6].dtype, torch.int32)
        self.assertEqual(args[7:], (64, 16, 64, 1))

    def test_hadamard_quantizer_can_disable_k_stabilization(self):
        q = torch.empty((1, 2, 65, 64), dtype=torch.float16)
        k = torch.empty((1, 1, 73, 64), dtype=torch.float16)
        with mock.patch.object(
            quant._fused, "quant_qk_per_warp_int8_rotated_cuda"
        ) as fused:
            quant.per_warp_int8_hadamard(q, k, stabilize_k=False)
        fused.assert_called_once()

    def test_production_quant_module_has_no_experimental_api(self):
        self.assertFalse(hasattr(quant, "per_thread_int4"))
        self.assertFalse(hasattr(quant, "per_thread_int4_fused"))
        self.assertFalse(hasattr(quant, "sage2_score_correction"))
        self.assertFalse(hasattr(quant, "per_block_int8"))

    def test_public_sage_api_exposes_only_stable_and_current_sparse_entrypoints(self):
        self.assertEqual(
            turing_sage.__all__,
            [
                "available",
                "prequantize_sageattn",
                "prequantize_sol_sageattn",
                "preflight",
                "preflight_sparse",
                "preflight_w8a8",
                "run_attention_correctness_gate",
                "sageattn",
                "sageattn_compiled",
                "sageattn_from_prequantized",
                "sageattn_varlen",
                "sageattn_varlen_compiled",
                "sol_sparse_sageattn",
                "sol_sparse_sageattn_compiled",
                "sol_sparse_sageattn_from_prequantized",
                "split_prequantization_available",
                "sparse_available",
                "w8a8attn",
                "w8a8attn_compiled",
                "w8a8_available",
            ],
        )
        self.assertFalse(hasattr(turing_sage, "sageattn_sage1"))
        self.assertFalse(hasattr(turing_sage, "sageattn_sage2"))
        self.assertFalse(hasattr(turing_sage, "sage_"))


if __name__ == "__main__":
    unittest.main()
