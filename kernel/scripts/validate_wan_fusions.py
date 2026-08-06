#!/usr/bin/env python3
"""Numerical and A40-compatible microbenchmark for Wan Turing fusions."""

from __future__ import annotations

import argparse
import torch
import torch.nn.functional as functional

import comfy_kitchen
from comfyui_turing_utils_kernel import (
    turing_bf16_int4_convrot_quantize,
    turing_bf16_int8_convrot_quantize,
    turing_bf16_gelu_int4_convrot_quantize,
    turing_bf16_gelu_int8_convrot_quantize,
    turing_layer_norm_adaln,
)


def _time_cuda(function, repetitions: int) -> float:
    for _ in range(5):
        function()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repetitions):
        function()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / repetitions


def _expanded_modulation(value: torch.Tensor, sequence: int) -> torch.Tensor:
    if value.shape[1] == 1:
        return value
    repeats = sequence // value.shape[1]
    if repeats == 1:
        return value
    if repeats * value.shape[1] != sequence:
        repeats += 1
    return torch.repeat_interleave(value, repeats, dim=1)[:, :sequence]


def validate(device: torch.device, rows: int, hidden: int, repetitions: int) -> None:
    generator = torch.Generator(device=device).manual_seed(1234)
    x = torch.randn(rows, hidden, generator=generator, device=device, dtype=torch.bfloat16)
    fused8 = lambda: turing_bf16_gelu_int8_convrot_quantize(x, 256)
    eager8 = lambda: turing_bf16_int8_convrot_quantize(
        functional.gelu(x, approximate="tanh"), 256
    )
    fused4 = lambda: turing_bf16_gelu_int4_convrot_quantize(x, 256)
    eager4 = lambda: turing_bf16_int4_convrot_quantize(
        functional.gelu(x, approximate="tanh"), 256
    )
    int8_equal = 1.0
    int4_equal = 1.0
    scale8_error = 0.0
    scale4_error = 0.0
    for gelu_input in (x, x * 0.125, x * 8.0, x + 3.0):
        q8, s8 = turing_bf16_gelu_int8_convrot_quantize(gelu_input, 256)
        q8_ref, s8_ref = turing_bf16_int8_convrot_quantize(
            functional.gelu(gelu_input, approximate="tanh"), 256
        )
        int8_equal = min(int8_equal, float((q8 == q8_ref).float().mean()))
        scale8_error = max(
            scale8_error, float((s8.float() - s8_ref.float()).abs().max())
        )

        q4, s4 = turing_bf16_gelu_int4_convrot_quantize(gelu_input, 256)
        q4_ref, s4_ref = turing_bf16_int4_convrot_quantize(
            functional.gelu(gelu_input, approximate="tanh"), 256
        )
        int4_equal = min(int4_equal, float((q4 == q4_ref).float().mean()))
        scale4_error = max(
            scale4_error, float((s4.float() - s4_ref.float()).abs().max())
        )

    batch, sequence, modulation_steps = 2, 17, 4
    norm_x = torch.randn(
        batch, sequence, hidden, generator=generator, device=device, dtype=torch.bfloat16
    )
    scale = torch.randn(
        batch, modulation_steps, hidden,
        generator=generator, device=device, dtype=torch.bfloat16,
    ) * 0.1
    shift = torch.randn(
        batch, modulation_steps, hidden,
        generator=generator, device=device, dtype=torch.bfloat16,
    ) * 0.1
    fused_norm = lambda: turing_layer_norm_adaln(norm_x, scale, shift, 1.0e-6)

    def eager_norm():
        normalized = functional.layer_norm(
            norm_x, (hidden,), eps=1.0e-6
        )
        return torch.addcmul(
            _expanded_modulation(shift, sequence),
            normalized,
            1 + _expanded_modulation(scale, sequence),
        )

    norm_errors = []
    norm_cases = (norm_x, norm_x * 0.125, norm_x + 100.0, norm_x + 1000.0)
    for norm_input in norm_cases:
        normalized = turing_layer_norm_adaln(norm_input, scale, shift, 1.0e-6)
        reference = torch.addcmul(
            _expanded_modulation(shift, sequence),
            functional.layer_norm(norm_input, (hidden,), eps=1.0e-6),
            1 + _expanded_modulation(scale, sequence),
        )
        if not torch.allclose(
            normalized.float(), reference.float(), rtol=0.006, atol=0.002
        ):
            raise RuntimeError(
                "fused LayerNorm+AdaLN exceeds the BF16 validation tolerance"
            )
        norm_errors.append((normalized.float() - reference.float()).abs())
    norm_max_error = max(float(error.max()) for error in norm_errors)
    norm_mean_error = max(float(error.mean()) for error in norm_errors)

    print(
        f"device={torch.cuda.get_device_name(device)} capability={torch.cuda.get_device_capability(device)} "
        f"kitchen={getattr(comfy_kitchen, '__version__', 'unknown')}"
    )
    print(
        f"gelu_int8 code_equal={int8_equal:.6f} scale_max_abs={scale8_error:.8g} "
        f"fused_ms={_time_cuda(fused8, repetitions):.4f} "
        f"eager_ms={_time_cuda(eager8, repetitions):.4f}"
    )
    print(
        f"gelu_int4 packed_equal={int4_equal:.6f} scale_max_abs={scale4_error:.8g} "
        f"fused_ms={_time_cuda(fused4, repetitions):.4f} "
        f"eager_ms={_time_cuda(eager4, repetitions):.4f}"
    )
    print(
        f"layernorm_adaln cases={len(norm_cases)} max_abs={norm_max_error:.8g} "
        f"max_mean_abs={norm_mean_error:.8g} "
        f"fused_ms={_time_cuda(fused_norm, repetitions):.4f} "
        f"eager_ms={_time_cuda(eager_norm, repetitions):.4f}"
    )

    if int8_equal < 0.999 or int4_equal < 0.999:
        raise RuntimeError("fused GELU quantization does not match the eager Kitchen path")


def evaluate_patch_embedding(device: torch.device, repetitions: int) -> None:
    """Compare the existing FP32 Wan boundary with a prospective FP16 path."""
    generator = torch.Generator(device=device).manual_seed(5678)
    x = torch.randn(
        1, 16, 2, 60, 104, generator=generator, device=device, dtype=torch.float32
    )
    weight = torch.randn(
        5120, 16, 1, 2, 2,
        generator=generator, device=device, dtype=torch.float32,
    ) * 0.02
    bias = torch.randn(
        5120, generator=generator, device=device, dtype=torch.float32
    ) * 0.02
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        fp32 = lambda: functional.conv3d(x, weight, bias, stride=(1, 2, 2)).to(
            torch.bfloat16
        )
        x16, weight16, bias16 = x.half(), weight.half(), bias.half()
        fp16 = lambda: functional.conv3d(
            x16, weight16, bias16, stride=(1, 2, 2)
        ).to(torch.bfloat16)
        reference = fp32()
        candidate = fp16()
        error = (reference.float() - candidate.float()).abs()
        print(
            f"patch_embedding fp32_ms={_time_cuda(fp32, repetitions):.4f} "
            f"fp16_ms={_time_cuda(fp16, repetitions):.4f} "
            f"max_abs={float(error.max()):.8g} mean_abs={float(error.mean()):.8g}"
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=13824)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--patch-embedding", action="store_true")
    args = parser.parse_args()
    validate(torch.device(args.device), args.rows, args.hidden, args.repetitions)
    if args.patch_embedding:
        evaluate_patch_embedding(torch.device(args.device), min(args.repetitions, 20))


if __name__ == "__main__":
    main()
