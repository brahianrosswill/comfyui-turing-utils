import sys
from pathlib import Path
import unittest

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "kernel"))

import comfyui_turing_utils_kernel as kernel  # noqa: E402


class KernelCustomOpContractTest(unittest.TestCase):
    def test_convrot_fake_contracts(self):
        x = torch.empty((3, 512), dtype=torch.bfloat16, device="meta")
        expectations = {
            "turing_swiglu_int8_convrot_quantize": (3, 256),
            "turing_swiglu_int4_convrot_quantize": (3, 128),
            "turing_gelu_int8_convrot_quantize": (3, 512),
            "turing_gelu_int4_convrot_quantize": (3, 256),
            "turing_bf16_gelu_int8_convrot_quantize": (3, 512),
            "turing_bf16_gelu_int4_convrot_quantize": (3, 256),
        }
        for name, shape in expectations.items():
            quantized, scale = getattr(kernel, name)(x)
            self.assertEqual(quantized.shape, shape)
            self.assertEqual(quantized.dtype, torch.int8)
            self.assertEqual(scale.shape, (3, 1))
            self.assertEqual(scale.dtype, torch.float32)
            self.assertEqual(scale.device.type, "meta")

        for bits, shape in ((8, (3, 256)), (4, (3, 128))):
            quantized, scale = getattr(
                kernel, f"turing_bf16_int{bits}_convrot_quantize"
            )(x, swiglu=True)
            self.assertEqual(quantized.shape, shape)
            self.assertEqual(scale.shape, (3, 1))

    def test_linear_epilogue_and_norm_fake_contracts(self):
        activation = torch.empty((7, 128), dtype=torch.int8, device="meta")
        weight = torch.empty((64, 64), dtype=torch.int8, device="meta")
        row_scale = torch.empty((7,), dtype=torch.float32, device="meta")
        column_scale = torch.empty((64,), dtype=torch.float32, device="meta")
        linear = kernel.turing_w4a8_linear(
            activation, weight, row_scale, column_scale
        )
        self.assertEqual(linear.shape, (7, 64))
        self.assertEqual(linear.dtype, torch.bfloat16)

        accumulator = torch.empty((7, 80), dtype=torch.int32, device="meta")
        epilogue = kernel.turing_dequantize_int8_bf16(
            accumulator, row_scale, column_scale, 64
        )
        self.assertEqual(epilogue.shape, (7, 64))
        self.assertEqual(epilogue.dtype, torch.bfloat16)

        x2 = torch.empty((7, 128), dtype=torch.bfloat16, device="meta")
        segments = torch.empty((2, 3), dtype=torch.int32, device="meta")
        normalized = kernel.turing_segmented_rms_adaln(
            x2,
            torch.empty((128,), dtype=x2.dtype, device="meta"),
            torch.empty((2, 128), dtype=x2.dtype, device="meta"),
            torch.empty((2, 128), dtype=x2.dtype, device="meta"),
            segments,
        )
        self.assertEqual(normalized.shape, x2.shape)
        self.assertEqual(normalized.dtype, x2.dtype)

    def test_sol_and_varlen_are_fullgraph_leaves(self):
        q = torch.empty((1, 4, 129, 128), dtype=torch.bfloat16, device="meta")
        k = torch.empty((1, 2, 151, 128), dtype=torch.bfloat16, device="meta")
        compiled = torch.compile(
            lambda query, key, value: kernel.turing_sage.sol_sparse_sageattn_compiled(
                query,
                key,
                value,
                dense_query_ranges=((0, 64),),
                exact_kv_ranges=((0, 64),),
            ),
            backend="eager",
            fullgraph=True,
        )
        self.assertEqual(compiled(q, k, k).shape, q.shape)

        qv = torch.empty((129, 4, 128), dtype=q.dtype, device="meta")
        kv = torch.empty((151, 2, 128), dtype=q.dtype, device="meta")
        cu = torch.empty((2,), dtype=torch.int32, device="meta")
        output = kernel.turing_sage.sageattn_varlen_compiled(
            qv, kv, kv, cu, cu, 129, 151
        )
        self.assertEqual(output.shape, qv.shape)
        w8a8_output = kernel.turing_sage.w8a8attn_varlen_compiled(
            qv, kv, kv, cu, cu, 129, 151, is_causal=True
        )
        self.assertEqual(w8a8_output.shape, qv.shape)


if __name__ == "__main__":
    unittest.main()
