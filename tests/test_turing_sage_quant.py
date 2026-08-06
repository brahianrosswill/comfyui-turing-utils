from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "kernel"))

from svdint4.turing_sage import quant  # noqa: E402


class TuringSageQuantContractTest(unittest.TestCase):
    def test_sage1_per_block_scale_shapes_and_fused_k_smoothing(self):
        q = torch.empty((2, 4, 65, 64), dtype=torch.float16)
        k = torch.empty((2, 2, 73, 64), dtype=torch.float16)
        km = torch.empty((2, 2, 1, 64), dtype=torch.float16)
        with (
            mock.patch.object(quant._fused, "quant_per_block_int8_cuda") as quant_block,
            mock.patch.object(
                quant._fused, "quant_per_block_int8_fuse_sub_mean_cuda"
            ) as quant_smooth,
        ):
            q_int8, q_scale, k_int8, k_scale = quant.per_block_int8(q, k, km)

        self.assertEqual(q_int8.shape, q.shape)
        self.assertEqual(k_int8.shape, k.shape)
        self.assertEqual(q_scale.shape, (2, 4, 2))
        self.assertEqual(k_scale.shape, (2, 2, 2))
        quant_block.assert_called_once()
        quant_smooth.assert_called_once()

    def test_sage2_int4_is_packed_and_uses_sm75_per_thread_scale_layout(self):
        q = torch.empty((2, 4, 65, 128), dtype=torch.bfloat16)
        k = torch.empty((2, 2, 73, 128), dtype=torch.bfloat16)
        q_mean = torch.empty((2, 4, 2, 128), dtype=torch.float32)
        k_mean = torch.empty((2, 2, 1, 128), dtype=torch.float32)
        with (
            mock.patch.object(
                quant, "token_block_mean", side_effect=(q_mean, k_mean)
            ) as means,
            mock.patch.object(
                quant._fused, "quant_query_per_thread_int4_cuda"
            ) as quant_q,
            mock.patch.object(
                quant._fused, "quant_key_per_thread_int4_cuda"
            ) as quant_k,
        ):
            values = quant.per_thread_int4(q, k, smooth_q=True, smooth_k=True)

        q_int4, q_scale, k_int4, k_scale, returned_q_mean, returned_k_mean = values
        self.assertEqual(q_int4.shape, (2, 4, 65, 64))
        self.assertEqual(k_int4.shape, (2, 2, 73, 64))
        self.assertEqual(q_scale.shape, (2, 4, 64))  # 2 CTAs * 4 Q warps * 8 groups
        self.assertEqual(k_scale.shape, (2, 2, 8))  # 2 CTAs * 4 K groups
        self.assertIs(returned_q_mean, q_mean)
        self.assertIs(returned_k_mean, k_mean)
        self.assertEqual(means.call_args_list[0].args[1], 64)
        self.assertEqual(means.call_args_list[1].args[1], 73)
        quant_q.assert_called_once()
        quant_k.assert_called_once()


if __name__ == "__main__":
    unittest.main()
