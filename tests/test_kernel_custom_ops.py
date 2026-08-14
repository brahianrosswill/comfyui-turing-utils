import sys
from pathlib import Path
import unittest

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "kernel"))

import comfyui_turing_utils_kernel as kernel  # noqa: E402


class KernelCustomOpContractTest(unittest.TestCase):
    def test_overlap_blend_fake_contract_is_a_fullgraph_leaf(self):
        values = torch.empty((2, 5, 17, 64), dtype=torch.float16, device="meta")
        local_indices = torch.empty((31, 5), dtype=torch.int32, device="meta")
        weights = torch.empty((31, 5), dtype=torch.float32, device="meta")
        compiled = torch.compile(
            lambda value, local, weight: kernel.turing_sage.overlap_blend_compiled(
                value, local, weight
            ),
            backend="eager",
            fullgraph=True,
        )
        output = compiled(values, local_indices, weights)
        self.assertEqual(output.shape, (2, 31, 64))
        self.assertEqual(output.dtype, torch.float16)
        self.assertEqual(output.device.type, "meta")

    @unittest.skipUnless(
        torch.cuda.is_available() and kernel.turing_sage.overlap_blend_available(),
        "the rebuilt CUDA overlap epilogue is required",
    )
    def test_overlap_blend_matches_ordered_fp32_reference(self):
        torch.manual_seed(47)
        batch, windows, tokens, channels, global_tokens = 2, 5, 17, 96, 31
        local_indices = torch.full(
            (global_tokens, windows), -1, dtype=torch.int32, device="cuda"
        )
        weights = torch.zeros(
            (global_tokens, windows), dtype=torch.float32, device="cuda"
        )
        for global_index in range(global_tokens):
            owners = [global_index % windows, (global_index + 2) % windows]
            local_indices[global_index, owners[0]] = global_index % tokens
            local_indices[global_index, owners[1]] = (global_index * 3) % tokens
            weights[global_index, owners[0]] = 0.35
            weights[global_index, owners[1]] = 0.65

        for dtype, tolerance in (
            (torch.float16, 2**-10),
            (torch.bfloat16, 2**-7),
        ):
            base = torch.randn(
                batch,
                windows,
                tokens + 3,
                channels,
                dtype=dtype,
                device="cuda",
            )
            values = base[:, :, :tokens]
            expected = torch.zeros(
                batch, global_tokens, channels, dtype=torch.float32, device="cuda"
            )
            for global_index in range(global_tokens):
                for window in range(windows):
                    local = int(local_indices[global_index, window])
                    if local >= 0:
                        expected[:, global_index].add_(
                            values[:, window, local].float()
                            * weights[global_index, window]
                        )
            expected = expected.to(dtype)
            first = kernel.turing_sage.overlap_blend_compiled(
                values, local_indices, weights
            )
            second = kernel.turing_sage.overlap_blend_compiled(
                values, local_indices, weights
            )
            torch.testing.assert_close(first, expected, rtol=0.0, atol=tolerance)
            self.assertTrue(torch.equal(first, second))

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

        codebook_linear = kernel.turing_codebook_w4a8_linear(
            activation,
            weight,
            row_scale,
            torch.empty((64, 8), dtype=torch.uint8, device="meta"),
            column_scale,
            torch.empty((16,), dtype=torch.float32, device="meta"),
        )
        self.assertEqual(codebook_linear.shape, (7, 64))
        self.assertEqual(codebook_linear.dtype, torch.bfloat16)

        long_codebook_linear = kernel.turing_codebook_w4a8_linear(
            torch.empty((9000, 128), dtype=torch.int8, device="meta"),
            weight,
            torch.empty((9000,), dtype=torch.float32, device="meta"),
            torch.empty((64, 8), dtype=torch.uint8, device="meta"),
            column_scale,
            torch.empty((16,), dtype=torch.float32, device="meta"),
            chunk_rows=-1,
        )
        self.assertEqual(long_codebook_linear.shape, (9000, 64))

        int8_linear = kernel.turing_int8_linear(
            activation,
            torch.empty((64, 128), dtype=torch.int8, device="meta"),
            row_scale,
            column_scale,
        )
        self.assertEqual(int8_linear.shape, (7, 64))
        self.assertEqual(int8_linear.dtype, torch.bfloat16)

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
