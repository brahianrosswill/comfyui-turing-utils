#!/usr/bin/env python3
"""Numerical and optional timing checks for production Turing kernels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
                    raise RuntimeError(f"ConvRot INT4 range mismatch for K={hidden}")
            _assert_close(
                f"ConvRot INT{bits} scale K={hidden}",
                row_scale,
                staged_scale,
                rtol=1.0e-6,
                atol=5.0e-8,
            )

    zero = torch.zeros((2, 512), device=device, dtype=torch.bfloat16)
    for implementation in (
        lambda: svdint4.turing_swiglu_int4_convrot_quantize(zero, 256),
        lambda: svdint4.turing_bf16_int4_convrot_quantize(zero, 256, swiglu=True),
    ):
        packed, scale = implementation()
        if torch.count_nonzero(_unpack_int4(packed)):
            raise RuntimeError("zero ConvRot INT4 input must quantize to zero")
        if not torch.equal(scale, torch.full_like(scale, 1.0e-10)):
            raise RuntimeError("ConvRot INT4 scale floor must match comfy-kitchen")


def validate_w4a8(device: torch.device) -> None:
    m, n, k = 7, 13, 64
    activation = ((torch.arange(m * k, device=device) % 23) - 11).to(torch.int8).reshape(m, k)
    weight = ((torch.arange(n * k, device=device) % 15) - 7).to(torch.int8).reshape(n, k)
    activation_scale = torch.linspace(0.01, 0.03, m, device=device)
    weight_scale = torch.linspace(0.02, 0.06, n, device=device)
    bias = torch.linspace(-0.2, 0.2, n, dtype=torch.bfloat16, device=device)
    output = svdint4.turing_w4a8_linear(
        activation, _pack_int4(weight), activation_scale, weight_scale, bias
    )
    reference = (
        activation.float() @ weight.float().t()
    ) * activation_scale[:, None] * weight_scale[None, :] + bias.float()
    _assert_close("packed W4A8", output, reference, rtol=0.01, atol=0.01)


def validate_segmented_norm(device: torch.device) -> None:
    rows, hidden = 19, 384
    segments = torch.tensor(
        ((0, 1, 2), (1, 7, 0), (7, rows, 1)), dtype=torch.int32, device=device
    )
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
        norm = x.float() * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + 1.0e-5
        )
        reference = norm * weight.float()
        reference = reference * (1.0 + scale.float()[row_ids]) + shift.float()[row_ids]
        _assert_close(f"segmented norm {dtype}", output, reference, rtol=0.01, atol=0.02)


def _validate_varlen_batches(
    output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
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
        _assert_close(
            f"Sage varlen batch {batch}",
            output[q_start:q_start + q_len],
            reference,
            rtol=0.08,
            atol=0.06,
        )


def validate_sage(device: torch.device) -> None:
    from svdint4.turing_sage import sageattn, sageattn_varlen

    for dtype in (torch.float16, torch.bfloat16):
        for head_dim, is_causal in ((32, False), (64, True), (96, False), (128, False)):
            q = torch.randn((1, 4, 129, head_dim), device=device, dtype=dtype) * 0.4
            k = torch.randn((1, 2, 151, head_dim), device=device, dtype=dtype) * 0.4
            v = torch.randn_like(k)
            output = sageattn(q, k, v, tensor_layout="HND", is_causal=is_causal)
            reference = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), enable_gqa=True, is_causal=is_causal
            )
            _assert_close(
                f"Sage fixed {dtype} D={head_dim} causal={is_causal}",
                output,
                reference,
                rtol=0.08,
                atol=0.06,
            )

        output_nhd = sageattn(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            tensor_layout="NHD",
        )
        reference_nhd = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), enable_gqa=True
        ).transpose(1, 2)
        _assert_close(f"Sage NHD {dtype}", output_nhd, reference_nhd, rtol=0.08, atol=0.06)

    q = torch.randn((1, 4, 65, 64), device=device, dtype=torch.bfloat16)
    k = torch.randn((1, 1, 73, 64), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    output, lse = sageattn(q, k, v, return_lse=True)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), enable_gqa=True
    )
    _assert_close("Sage MQA", output, reference, rtol=0.15, atol=0.12)
    score_key = torch.repeat_interleave(k.float(), q.size(1) // k.size(1), dim=1)
    reference_lse = torch.logsumexp(
        torch.matmul(q.float(), score_key.transpose(-2, -1)) * q.size(-1) ** -0.5,
        dim=-1,
    )
    _assert_close("Sage LSE", lse, reference_lse, rtol=0.02, atol=0.02)

    q_lengths, k_lengths = (65, 513), (73, 601)
    cu_q = torch.tensor((0, q_lengths[0], sum(q_lengths)), dtype=torch.int32, device=device)
    cu_k = torch.tensor((0, k_lengths[0], sum(k_lengths)), dtype=torch.int32, device=device)
    q = torch.randn((sum(q_lengths), 4, 64), device=device, dtype=torch.bfloat16)
    k = torch.randn((sum(k_lengths), 2, 64), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    output = sageattn_varlen(
        q, k, v, cu_q, cu_k, max(q_lengths), max(k_lengths)
    )
    _validate_varlen_batches(output, q, k, v, cu_q, cu_k, q_lengths, k_lengths)


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


def benchmark_sage(device: torch.device, iterations: int) -> None:
    from svdint4.turing_sage import sageattn

    for dtype in (torch.float16, torch.bfloat16):
        q = torch.randn((1, 8, 2048, 128), device=device, dtype=dtype)
        k = torch.randn((1, 4, 2048, 128), device=device, dtype=dtype)
        v = torch.randn_like(k)
        elapsed = _elapsed_ms(lambda: sageattn(q, k, v), iterations)
        print(f"sage HND N=2048 Hq=8 Hkv=4 D=128 {dtype}: {elapsed:.3f} ms")


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
    print(f"kernel={Path(svdint4.__file__).resolve()} version={svdint4.__version__}")
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
            benchmark_sage(device, args.iterations)


if __name__ == "__main__":
    main()
