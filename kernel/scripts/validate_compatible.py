#!/usr/bin/env python3
"""Numerical and optional timing checks for Turing kernels on a compatible GPU."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Always validate the extension built beside this script, not an older
# svdint4 wheel that may already be installed in the active environment.
KERNEL_ROOT = Path(__file__).resolve().parents[1]
kernel_root = str(KERNEL_ROOT)
if kernel_root in sys.path:
    sys.path.remove(kernel_root)
sys.path.insert(0, kernel_root)

import torch

import svdint4


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, **kwargs) -> None:
    try:
        torch.testing.assert_close(actual.float(), expected.float(), **kwargs)
    except AssertionError as exc:
        raise RuntimeError(f"{name} numerical validation failed") from exc


def _assert_quantized_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    mean_atol: float,
    max_atol: float,
    bias_atol: float,
) -> None:
    """Check approximate INT4 output without unstable near-zero relative error."""
    difference = actual.float() - expected.float()
    if not torch.isfinite(difference).all():
        raise RuntimeError(f"{name} quantized validation produced non-finite values")
    mean_error = difference.abs().mean().item()
    max_error = difference.abs().max().item()
    bias = difference.mean().abs().item()
    if mean_error > mean_atol or max_error > max_atol or bias > bias_atol:
        raise RuntimeError(
            f"{name} quantized validation failed: mean_abs={mean_error:.6g}, "
            f"max_abs={max_error:.6g}, bias={bias:.6g}"
        )


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    low = values[:, 0::2].to(torch.int32) & 0x0F
    high = values[:, 1::2].to(torch.int32) & 0x0F
    return (low | high << 4).to(torch.int8)


def _unpack_int4(values: torch.Tensor) -> torch.Tensor:
    packed = values.to(torch.int32)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack((low, high), dim=-1).flatten(-2)


def validate_convrot(device: torch.device) -> None:
    for hidden in (256, 5376, 7168, 14336):
        generator = torch.Generator(device=device).manual_seed(4100 + hidden)
        x = torch.randn((3, hidden * 2), generator=generator, device=device, dtype=torch.bfloat16)
        for bits in (8, 4):
            staged = getattr(svdint4, f"turing_swiglu_int{bits}_convrot_quantize")
            rowbuffer = getattr(svdint4, f"turing_bf16_int{bits}_convrot_quantize")
            staged_q, staged_scale = staged(x, 256)
            row_q, row_scale = rowbuffer(x, 256, swiglu=True)
            if not torch.equal(staged_q, row_q):
                raise RuntimeError(f"ConvRot INT{bits} packed mismatch for K={hidden}")
            if bits == 4:
                unpacked = _unpack_int4(row_q)
                if unpacked.min() < -7 or unpacked.max() > 7:
                    raise RuntimeError(
                        f"ConvRot INT4 range mismatch for K={hidden}: expected [-7, 7]"
                    )
            _assert_close(
                f"ConvRot INT{bits} scale K={hidden}",
                row_scale,
                staged_scale,
                rtol=1.0e-6,
                atol=5.0e-8,
            )

    # Match comfy-kitchen's deterministic INT4 activation contract at the
    # numerically sensitive lower boundary: symmetric [-7, 7] and a 1e-10
    # scale floor.  Comparing only the two local implementations would not
    # catch a shared packing or scale bug.
    zero = torch.zeros((2, 512), device=device, dtype=torch.bfloat16)
    for implementation in (
        lambda: svdint4.turing_swiglu_int4_convrot_quantize(zero, 256),
        lambda: svdint4.turing_bf16_int4_convrot_quantize(zero, 256, swiglu=True),
    ):
        packed, scale = implementation()
        unpacked = _unpack_int4(packed)
        if torch.count_nonzero(unpacked):
            raise RuntimeError("zero ConvRot INT4 input must quantize to zero")
        expected_scale = torch.full_like(scale, 1.0e-10)
        if not torch.equal(scale, expected_scale):
            raise RuntimeError("ConvRot INT4 scale floor must match comfy-kitchen (1e-10)")
        if unpacked.min() < -7 or unpacked.max() > 7:
            raise RuntimeError("ConvRot INT4 activations must use the symmetric [-7, 7] range")


def validate_w4a8(device: torch.device) -> None:
    m, n, k = 7, 13, 64
    activation = ((torch.arange(m * k, device=device) % 23) - 11).to(torch.int8).reshape(m, k)
    weight_values = ((torch.arange(n * k, device=device) % 15) - 7).to(torch.int8).reshape(n, k)
    activation_scale = torch.linspace(0.01, 0.03, m, device=device)
    weight_scale = torch.linspace(0.02, 0.06, n, device=device)
    bias = torch.linspace(-0.2, 0.2, n, dtype=torch.bfloat16, device=device)
    output = svdint4.turing_w4a8_linear(
        activation,
        _pack_int4(weight_values),
        activation_scale,
        weight_scale,
        bias,
    )
    reference = (
        activation.float() @ weight_values.float().t()
    ) * activation_scale[:, None] * weight_scale[None, :] + bias.float()
    _assert_close("packed W4A8", output, reference, rtol=0.01, atol=0.01)


def validate_segmented_norm(device: torch.device) -> None:
    rows, hidden = 19, 384
    segments = torch.tensor(((0, 1, 2), (1, 7, 0), (7, rows, 1)), dtype=torch.int32, device=device)
    row_ids = torch.cat(
        (
            torch.full((1,), 2, device=device, dtype=torch.long),
            torch.full((6,), 0, device=device, dtype=torch.long),
            torch.full((rows - 7,), 1, device=device, dtype=torch.long),
        )
    )
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = torch.randn((rows, hidden), device=device, dtype=dtype)
        weight = torch.randn((hidden,), device=device, dtype=dtype)
        scale = torch.randn((3, hidden), device=device, dtype=dtype) * 0.1
        shift = torch.randn((3, hidden), device=device, dtype=dtype) * 0.1
        output = svdint4.turing_segmented_rms_adaln(
            x, weight, scale, shift, segments, 1.0e-5
        )
        norm = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1.0e-5)
        reference = norm * weight.float()
        reference = reference * (1.0 + scale.float()[row_ids]) + shift.float()[row_ids]
        _assert_close(f"segmented norm {dtype}", output, reference, rtol=0.01, atol=0.02)


def validate_sage(device: torch.device) -> None:
    from svdint4.turing_sage import (
        sageattn_hybrid,
        sageattn_sage1,
        sageattn_sage2,
        sageattn_varlen,
    )
    from svdint4.turing_sage.quant import (
        per_thread_int4,
        per_thread_int4_fused,
        sage2_score_correction,
    )

    variants = (
        ("sage_", sageattn_hybrid, {"smooth_k": False}, 0.08, 0.06),
        ("sage1", sageattn_sage1, {"smooth_k": True}, 0.08, 0.06),
        ("sage2", sageattn_sage2, {"smooth_q": True, "smooth_k": True}, 0.15, 0.12),
    )
    for name, implementation, options, rtol, atol in variants:
        for dtype in (torch.float16, torch.bfloat16):
            for head_dim, is_causal in ((32, False), (64, True), (96, False), (128, False)):
                q = torch.randn((1, 4, 129, head_dim), device=device, dtype=dtype) * 0.4
                k = torch.randn((1, 2, 151, head_dim), device=device, dtype=dtype) * 0.4
                v = torch.randn_like(k)
                output = implementation(
                    q, k, v, tensor_layout="HND", is_causal=is_causal, **options
                )
                reference = torch.nn.functional.scaled_dot_product_attention(
                    q.float(), k.float(), v.float(), enable_gqa=True, is_causal=is_causal
                )
                case_name = f"{name} fixed {dtype} D={head_dim} causal={is_causal}"
                if name == "sage2":
                    _assert_quantized_close(
                        case_name, output, reference,
                        mean_atol=0.01, max_atol=0.20, bias_atol=0.002,
                    )
                else:
                    _assert_close(case_name, output, reference, rtol=rtol, atol=atol)

            q_nhd = q.transpose(1, 2).contiguous()
            k_nhd = k.transpose(1, 2).contiguous()
            v_nhd = v.transpose(1, 2).contiguous()
            output_nhd = implementation(
                q_nhd, k_nhd, v_nhd, tensor_layout="NHD", **options
            )
            reference_nhd = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), enable_gqa=True
            ).transpose(1, 2)
            if name == "sage2":
                _assert_quantized_close(
                    f"{name} NHD {dtype}", output_nhd, reference_nhd,
                    mean_atol=0.01, max_atol=0.20, bias_atol=0.002,
                )
            else:
                _assert_close(
                    f"{name} NHD {dtype}", output_nhd, reference_nhd,
                    rtol=rtol, atol=atol,
                )

    # The fused preprocessor must preserve the official packed code layout
    # exactly; score correction is allowed only the FP16-input/FP32-accum MMA
    # rounding that its reference models here.
    q = torch.randn((2, 8, 137, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn((2, 2, 151, 128), device=device, dtype=torch.bfloat16)
    legacy_prepared = per_thread_int4(q, k, "HND", True, True)
    fused_prepared = per_thread_int4_fused(q, k, "HND", True, True)
    for index, (legacy, fused) in enumerate(zip(legacy_prepared, fused_prepared)):
        if not torch.equal(legacy, fused):
            raise RuntimeError(f"Sage2 fused preprocessing mismatch at output {index}")
    q_mean, k_mean = fused_prepared[4], fused_prepared[5]
    correction = sage2_score_correction(q_mean, k, k_mean, "HND", True)
    centered_k = (k.float() - k_mean[:, :, 0, None, :]).half().float()
    centered_k = torch.repeat_interleave(centered_k, q.size(1) // k.size(1), dim=1)
    correction_reference = torch.einsum(
        "bhqd,bhkd->bhqk", q_mean.half().float(), centered_k
    )
    _assert_close(
        "Sage2 FP16-TC score correction", correction, correction_reference,
        rtol=1.0e-5, atol=5.0e-6,
    )

    # Force one Q block per correction workspace launch and verify that the
    # absolute Q indexing is identical, including causal masking and LSE.
    import svdint4.turing_sage.core as sage_core
    original_workspace = sage_core._SAGE2_CORRECTION_WORKSPACE_BYTES
    try:
        sage_core._SAGE2_CORRECTION_WORKSPACE_BYTES = 1 << 30
        full_output, full_lse = sageattn_sage2(
            q[:1], k[:1], k[:1], is_causal=True, return_lse=True
        )
        sage_core._SAGE2_CORRECTION_WORKSPACE_BYTES = 1
        chunked_output, chunked_lse = sageattn_sage2(
            q[:1], k[:1], k[:1], is_causal=True, return_lse=True
        )
    finally:
        sage_core._SAGE2_CORRECTION_WORKSPACE_BYTES = original_workspace
    if not torch.equal(full_output, chunked_output) or not torch.equal(
        full_lse, chunked_lse
    ):
        raise RuntimeError("Sage2 bounded correction workspace changed output or LSE")

    # Exercise MQA, all PV policies retained by the hybrid, smooth_v, and LSE.
    q = torch.randn((1, 4, 65, 64), device=device, dtype=torch.bfloat16)
    k = torch.randn((1, 1, 73, 64), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), enable_gqa=True
    )
    for pv_accum_dtype, smooth_v in (
        ("fp16", False),
        ("fp16", True),
        ("fp16+fp32", False),
    ):
        output = sageattn_hybrid(
            q,
            k,
            v,
            tensor_layout="HND",
            smooth_k=True,
            smooth_v=smooth_v,
            pv_accum_dtype=pv_accum_dtype,
        )
        _assert_close(
            f"Sage MQA PV={pv_accum_dtype} smooth_v={smooth_v}",
            output,
            reference,
            rtol=0.12,
            atol=0.10,
        )

    for name, implementation, options, rtol, atol in variants:
        output, lse = implementation(
            q,
            k,
            v,
            tensor_layout="HND",
            return_lse=True,
            smooth_v=False,
            **options,
        )
        if name == "sage2":
            _assert_quantized_close(
                f"{name} MQA", output, reference,
                mean_atol=0.035, max_atol=0.70, bias_atol=0.003,
            )
        else:
            _assert_close(
                f"{name} MQA", output, reference,
                rtol=max(rtol, 0.15), atol=max(atol, 0.12),
            )
        if lse.shape != q.shape[:-1] or not torch.isfinite(lse).all():
            raise RuntimeError(
                f"{name} return_lse expected {q.shape[:-1]}, got {lse.shape}"
            )
        score_key = torch.repeat_interleave(k.float(), q.size(1) // k.size(1), dim=1)
        reference_lse = torch.logsumexp(
            torch.matmul(q.float(), score_key.transpose(-2, -1)) * (q.size(-1) ** -0.5),
            dim=-1,
        )
        if name == "sage2":
            _assert_quantized_close(
                f"{name} LSE", lse, reference_lse,
                mean_atol=0.04, max_atol=0.35, bias_atol=0.03,
            )
        else:
            _assert_close(
                f"{name} LSE", lse, reference_lse, rtol=0.02, atol=0.02
            )

    # The short-varlen fused kernel and long-varlen INT8 kernel are distinct.
    short_q_lengths = (33, 129)
    short_k_lengths = (47, 151)
    short_cu_q = torch.tensor(
        (0, short_q_lengths[0], sum(short_q_lengths)), dtype=torch.int32, device=device
    )
    short_cu_k = torch.tensor(
        (0, short_k_lengths[0], sum(short_k_lengths)), dtype=torch.int32, device=device
    )
    short_q = torch.randn((sum(short_q_lengths), 4, 96), device=device, dtype=torch.bfloat16)
    short_k = torch.randn((sum(short_k_lengths), 2, 96), device=device, dtype=torch.bfloat16)
    short_v = torch.randn_like(short_k)
    for variant in ("sage_", "sage1", "sage2"):
        short_output = sageattn_varlen(
            short_q, short_k, short_v, short_cu_q, short_cu_k,
            max(short_q_lengths), max(short_k_lengths), variant=variant,
            smooth_k=variant != "sage_", smooth_q=True,
        )
        _validate_varlen_batches(
            f"{variant} short varlen", short_output, short_q, short_k, short_v,
            short_cu_q, short_cu_k, short_q_lengths, short_k_lengths,
            rtol=0.15 if variant == "sage2" else 0.08,
            atol=0.12 if variant == "sage2" else 0.06,
            quantized=variant == "sage2",
        )

    q_lengths = (65, 513)
    k_lengths = (73, 601)
    cu_q = torch.tensor((0, q_lengths[0], sum(q_lengths)), dtype=torch.int32, device=device)
    cu_k = torch.tensor((0, k_lengths[0], sum(k_lengths)), dtype=torch.int32, device=device)
    q = torch.randn((sum(q_lengths), 4, 64), device=device, dtype=torch.bfloat16)
    k = torch.randn((sum(k_lengths), 2, 64), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    output = sageattn_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(q_lengths),
        max(k_lengths),
        smooth_k=False,
        variant="sage_",
    )
    _validate_varlen_batches(
        "Sage long varlen",
        output,
        q,
        k,
        v,
        cu_q,
        cu_k,
        q_lengths,
        k_lengths,
    )

    # Pure sequence-long FP16 PV accumulation can silently overflow when V has
    # a DC bias. H3 exercises this regime even though zero-mean random tests do
    # not. Sage1/Sage2 must remain finite without allocating a smoothed V copy.
    q = torch.randn((1, 4, 64, 128), device=device, dtype=torch.bfloat16) * 0.5
    k = torch.randn((1, 2, 8192, 128), device=device, dtype=torch.bfloat16) * 0.5
    v = torch.randn_like(k) + 16
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), enable_gqa=True
    )
    for name, implementation, options in (
        ("sage1", sageattn_sage1, {"smooth_k": True}),
        ("sage2", sageattn_sage2, {"smooth_q": True, "smooth_k": True}),
    ):
        output = implementation(q, k, v, tensor_layout="HND", **options)
        if not torch.isfinite(output).all():
            raise RuntimeError(f"{name} long-sequence biased-V output is non-finite")
        _assert_quantized_close(
            f"{name} long-sequence biased V",
            output,
            reference,
            mean_atol=0.08,
            max_atol=0.30,
            bias_atol=0.05,
        )


def _validate_varlen_batches(
    name: str,
    output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    rtol: float = 0.08,
    atol: float = 0.06,
    quantized: bool = False,
) -> None:
    for batch, (q_len, k_len) in enumerate(zip(q_lengths, k_lengths)):
        q_start = int(cu_q[batch])
        k_start = int(cu_k[batch])
        reference = torch.nn.functional.scaled_dot_product_attention(
            q[q_start:q_start + q_len].transpose(0, 1).unsqueeze(0).float(),
            k[k_start:k_start + k_len].transpose(0, 1).unsqueeze(0).float(),
            v[k_start:k_start + k_len].transpose(0, 1).unsqueeze(0).float(),
            enable_gqa=True,
        ).squeeze(0).transpose(0, 1)
        actual = output[q_start:q_start + q_len]
        if quantized:
            _assert_quantized_close(
                f"{name} batch {batch}", actual, reference,
                mean_atol=0.035, max_atol=0.70, bias_atol=0.003,
            )
        else:
            _assert_close(
                f"{name} batch {batch}", actual, reference,
                rtol=rtol, atol=atol,
            )


def _elapsed_ms(function, iterations: int) -> float:
    for _ in range(5):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def benchmark_convrot(device: torch.device, iterations: int) -> None:
    rows = 2048
    for hidden in (5376, 7168, 14336):
        x = torch.randn((rows, hidden * 2), device=device, dtype=torch.bfloat16)
        for bits in (8, 4):
            staged = getattr(svdint4, f"turing_swiglu_int{bits}_convrot_quantize")
            rowbuffer = getattr(svdint4, f"turing_bf16_int{bits}_convrot_quantize")
            staged_ms = _elapsed_ms(lambda: staged(x, 256), iterations)
            row_ms = _elapsed_ms(lambda: rowbuffer(x, 256, swiglu=True), iterations)
            print(
                f"M={rows} K={hidden} INT{bits}: staged={staged_ms:.3f} ms "
                f"rowbuffer={row_ms:.3f} ms speedup={staged_ms / row_ms:.2f}x"
            )


def benchmark_sage(device: torch.device, iterations: int) -> None:
    from svdint4.turing_sage import sageattn_hybrid, sageattn_sage1, sageattn_sage2

    batch, q_heads, kv_heads, sequence, head_dim = 1, 8, 4, 2048, 128
    variants = (
        ("sage_", sageattn_hybrid, {"smooth_k": False}),
        ("sage1", sageattn_sage1, {"smooth_k": True}),
        ("sage2", sageattn_sage2, {"smooth_q": True, "smooth_k": True}),
    )
    for dtype in (torch.float16, torch.bfloat16):
        q = torch.randn((batch, q_heads, sequence, head_dim), device=device, dtype=dtype)
        k = torch.randn((batch, kv_heads, sequence, head_dim), device=device, dtype=dtype)
        v = torch.randn_like(k)
        for name, implementation, options in variants:
            elapsed = _elapsed_ms(
                lambda implementation=implementation, options=options: implementation(
                    q, k, v, tensor_layout="HND", **options
                ),
                iterations,
            )
            print(
                f"{name} HND N={sequence} Hq={q_heads} Hkv={kv_heads} "
                f"D={head_dim} {dtype}: {elapsed:.3f} ms"
            )
        if dtype is torch.bfloat16:
            cast_ms = _elapsed_ms(lambda: v.to(torch.float16), iterations)
            print(f"legacy full BF16->FP16 V conversion alone: {cast_ms:.3f} ms")


def benchmark_sage1_breakdown(device: torch.device, iterations: int) -> None:
    """Isolate the three policy differences between Sage1 and ``sage_``."""
    from svdint4.turing_sage.core import sageattn_qk_int8_pv_fp16_cuda

    batch, q_heads, kv_heads, sequence, head_dim = 1, 8, 4, 2048, 128
    q = torch.randn(
        (batch, q_heads, sequence, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (batch, kv_heads, sequence, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    configurations = (
        ("sage_ baseline", "per_warp", False, "fp32"),
        ("per-block Q/K", "per_block", False, "fp32"),
        ("+ K smoothing", "per_block", True, "fp32"),
        ("+ mixed PV (sage1)", "per_block", True, "fp16+fp32"),
        ("old unsafe FP16 PV", "per_block", True, "fp16"),
    )
    print("sage1 vs sage_ breakdown (smooth_v is disabled in every case)")
    for label, granularity, smooth_k, accumulator in configurations:
        elapsed = _elapsed_ms(
            lambda granularity=granularity, smooth_k=smooth_k,
            accumulator=accumulator: sageattn_qk_int8_pv_fp16_cuda(
                q,
                k,
                v,
                tensor_layout="HND",
                qk_quant_gran=granularity,
                smooth_k=smooth_k,
                smooth_v=False,
                pv_accum_dtype=accumulator,
            ),
            iterations,
        )
        print(f"  {label:<21} {elapsed:.3f} ms")


def benchmark_sage2_breakdown(device: torch.device, iterations: int) -> None:
    """Separate official-layout preprocessing and score-correction policies."""
    from svdint4.turing_sage import sm75_compile
    from svdint4.turing_sage.quant import (
        per_thread_int4,
        per_thread_int4_fused,
        sage2_score_correction,
        token_block_mean,
    )

    batch, q_heads, kv_heads, sequence, head_dim = 1, 8, 4, 2048, 128
    q = torch.randn(
        (batch, q_heads, sequence, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (batch, kv_heads, sequence, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    configurations = (
        ("none", False, False),
        ("K", False, True),
        ("Q", True, False),
        ("Q+K", True, True),
    )
    print("sage2 breakdown (A40 timings are compute_75 compatibility data)")
    q_mean_ms = _elapsed_ms(
        lambda: token_block_mean(q, 64, tensor_layout="HND"), iterations
    )
    k_mean_ms = _elapsed_ms(
        lambda: token_block_mean(k, sequence, tensor_layout="HND"), iterations
    )
    print(f"  means only: Q-block={q_mean_ms:.3f} ms K-global={k_mean_ms:.3f} ms")

    for label, smooth_q, smooth_k in configurations:
        legacy_quant_ms = _elapsed_ms(
            lambda smooth_q=smooth_q, smooth_k=smooth_k: per_thread_int4(
                q,
                k,
                tensor_layout="HND",
                smooth_q=smooth_q,
                smooth_k=smooth_k,
            ),
            iterations,
        )
        quant_ms = _elapsed_ms(
            lambda smooth_q=smooth_q, smooth_k=smooth_k: per_thread_int4_fused(
                q, k, tensor_layout="HND", smooth_q=smooth_q, smooth_k=smooth_k
            ),
            iterations,
        )
        prepared = per_thread_int4_fused(
            q,
            k,
            tensor_layout="HND",
            smooth_q=smooth_q,
            smooth_k=smooth_k,
        )
        q_int4, q_scale, k_int4, k_scale, q_mean, k_mean = prepared
        output = torch.empty_like(q)

        if smooth_q:
            correction_ms = _elapsed_ms(
                lambda: sage2_score_correction(
                    q_mean, k, k_mean, "HND", smooth_k
                ),
                iterations,
            )
            correction = sage2_score_correction(q_mean, k, k_mean, "HND", smooth_k)
            attention_only = lambda: sm75_compile.qk_int4_sv_f16_accum_f16_f32_precomputed_attn(
                q_int4, k_int4, v, output, q_scale, k_scale, correction,
                1, 0, head_dim**-0.5, 0, 0, q_mean.size(2),
            )
        else:
            correction_ms = 0.0
            attention_only = lambda: sm75_compile.qk_int4_sv_f16_accum_f16_f32_attn(
                q_int4, k_int4, v, output, q_scale, k_scale, k, q_mean,
                k_mean, 1, 0, head_dim**-0.5, 0, 0, int(smooth_k),
            )

        attention_ms = _elapsed_ms(attention_only, iterations)
        print(
            f"  smooth={label:<3} legacy-pre={legacy_quant_ms:.3f} ms "
            f"fused-pre={quant_ms:.3f} ms correction={correction_ms:.3f} ms "
            f"attention={attention_ms:.3f} ms "
            f"total={quant_ms + correction_ms + attention_ms:.3f} ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("validation requires a CUDA device")
    capability = torch.cuda.get_device_capability(device)
    if capability < (7, 5):
        raise RuntimeError(f"validation requires sm75 or newer, got sm{capability[0]}{capability[1]}")
    print(
        f"kernel={Path(svdint4.__file__).resolve()} version={svdint4.__version__}"
    )
    print(f"device={torch.cuda.get_device_name(device)} capability=sm{capability[0]}{capability[1]}")
    torch.manual_seed(20260806)
    with torch.inference_mode(), torch.cuda.device(device):
        validate_convrot(device)
        validate_w4a8(device)
        validate_segmented_norm(device)
        validate_sage(device)
        torch.cuda.synchronize(device)
        print("numerical validation passed")
        if args.benchmark:
            benchmark_convrot(device, args.iterations)
            benchmark_sage(device, args.iterations)
            benchmark_sage1_breakdown(device, args.iterations)
            benchmark_sage2_breakdown(device, args.iterations)


if __name__ == "__main__":
    main()
