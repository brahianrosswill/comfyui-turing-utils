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
                "preflight",
                "preflight_sparse",
                "sageattn",
                "sageattn_varlen",
                "sol_sparse_sageattn",
                "sparse_available",
            ],
        )
        self.assertFalse(hasattr(turing_sage, "sageattn_sage1"))
        self.assertFalse(hasattr(turing_sage, "sageattn_sage2"))
        self.assertFalse(hasattr(turing_sage, "sage_"))


if __name__ == "__main__":
    unittest.main()
