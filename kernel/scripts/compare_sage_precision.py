#!/usr/bin/env python3
"""Compare official Sage quantizers with the bundled SM75 INT8/INT4 variants.

The current official SageAttention package exposes INT8 QK kernels even though
the SageAttention2 paper studies per-thread INT4.  The package still ships the
official Triton INT4 quantizer source, so this script invokes that quantizer
directly and reconstructs its mathematical attention reference.  It separates
quantization error, local-vs-official quantizer differences, and the SM75
online-softmax/FP16-PV error.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

KERNEL_ROOT = Path(__file__).resolve().parents[1]
kernel_root = str(KERNEL_ROOT)
if kernel_root in sys.path:
    sys.path.remove(kernel_root)
sys.path.insert(0, kernel_root)

import torch

from svdint4.turing_sage import sageattn_hybrid, sageattn_sage1, sageattn_sage2
from svdint4.turing_sage.quant import per_thread_int4


@dataclass
class ErrorStats:
    absolute_sum: float = 0.0
    signed_sum: float = 0.0
    squared_sum: float = 0.0
    count: int = 0
    maximum: float = 0.0
    cases: int = 0
    nonfinite_cases: int = 0

    def add(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        allow_nonfinite: bool = False,
    ) -> bool:
        self.cases += 1
        difference = actual.float() - expected.float()
        if not torch.isfinite(difference).all():
            self.nonfinite_cases += 1
            if not allow_nonfinite:
                raise RuntimeError("precision comparison produced non-finite values")
            return False
        self.absolute_sum += difference.abs().sum().item()
        self.signed_sum += difference.sum().item()
        self.squared_sum += difference.square().sum().item()
        self.count += difference.numel()
        self.maximum = max(self.maximum, difference.abs().max().item())
        return True

    def values(self) -> tuple[float, float, float, float]:
        if self.count == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        return (
            self.absolute_sum / self.count,
            self.maximum,
            abs(self.signed_sum / self.count),
            (self.squared_sum / self.count) ** 0.5,
        )


def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    values = packed.to(torch.int32)
    low = values & 0x0F
    high = (values >> 4) & 0x0F
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack((low, high), dim=-1).flatten(-2)


def _reconstruct_local_int4(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    smooth_q: bool,
    smooth_k: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    packed_q, q_scale, packed_k, k_scale, q_mean, k_mean = per_thread_int4(
        q, k, tensor_layout="HND", smooth_q=smooth_q, smooth_k=smooth_k
    )
    q_values = _unpack_int4(packed_q).float()
    k_values = _unpack_int4(packed_k).float()

    q_tokens = torch.arange(q.size(2), device=q.device)
    q_in_block = q_tokens.remainder(64)
    q_scale_index = (
        q_tokens.div(64, rounding_mode="floor") * 32
        + q_in_block.div(16, rounding_mode="floor") * 8
        + q_in_block.remainder(8)
    )
    q_mean_index = q_tokens.div(64, rounding_mode="floor")
    q_reconstructed = q_values * q_scale[:, :, q_scale_index].unsqueeze(-1)
    if smooth_q:
        q_reconstructed = q_reconstructed + q_mean[:, :, q_mean_index]

    k_tokens = torch.arange(k.size(2), device=k.device)
    k_scale_index = (
        k_tokens.div(64, rounding_mode="floor") * 4
        + k_tokens.remainder(8).div(2, rounding_mode="floor")
    )
    k_reconstructed = k_values * k_scale[:, :, k_scale_index].unsqueeze(-1)
    if smooth_k:
        k_reconstructed = k_reconstructed + k_mean
    return q_reconstructed, k_reconstructed, q_mean, k_mean, q_values, k_values


def _exact_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v.float(),
        is_causal=is_causal,
        enable_gqa=q.size(1) != k.size(1),
    )


def _paper_int4_attention(
    q_reconstructed: torch.Tensor,
    k_reconstructed: torch.Tensor,
    q_original: torch.Tensor,
    k_original: torch.Tensor,
    q_mean: torch.Tensor,
    k_mean: torch.Tensor,
    v: torch.Tensor,
    *,
    smooth_q: bool,
    smooth_k: bool,
    is_causal: bool,
) -> torch.Tensor:
    """Evaluate Sage2's INT4 QK plus exact Q-mean score correction.

    With Q smoothing, Sage2 does not simply attend with reconstructed INT4 Q
    and K.  Its non-row-constant correction multiplies the Q-block mean by the
    original centered K, preserving substantially more accuracy when Q has a
    large blockwise bias.  K-mean terms omitted here are row constants and
    therefore cancel in softmax.
    """
    if not smooth_q:
        return _exact_attention(
            q_reconstructed, k_reconstructed, v, is_causal=is_causal
        )

    q_mean_tokens = torch.repeat_interleave(q_mean, 64, dim=2)[:, :, : q_original.size(2)]
    q_centered_quantized = q_reconstructed - q_mean_tokens
    if smooth_k:
        k_centered_quantized = k_reconstructed - k_mean
        k_centered_original = k_original.float() - k_mean
    else:
        k_centered_quantized = k_reconstructed
        k_centered_original = k_original.float()

    groups = q_reconstructed.size(1) // k_reconstructed.size(1)
    k_centered_quantized = torch.repeat_interleave(
        k_centered_quantized, groups, dim=1
    )
    k_centered_original = torch.repeat_interleave(k_centered_original, groups, dim=1)
    value = torch.repeat_interleave(v.float(), groups, dim=1)
    scores = torch.matmul(
        q_centered_quantized, k_centered_quantized.transpose(-2, -1)
    )
    scores.add_(torch.matmul(q_mean_tokens, k_centered_original.transpose(-2, -1)))
    scores.mul_(q_original.size(-1) ** -0.5)
    if is_causal:
        mask = torch.ones(
            (q_original.size(2), k_original.size(2)),
            dtype=torch.bool,
            device=q_original.device,
        ).tril()
        scores.masked_fill_(~mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), value)


def _official_int4_quantizers():
    """Return the official, shipped-but-not-public Triton INT4 quantizers."""
    try:
        from sageattention.triton.quant_per_thread import (
            quant_key_per_thread_int4_kernel,
            quant_query_per_thread_int4_kernel,
        )
    except (ImportError, OSError):
        return None
    return quant_query_per_thread_int4_kernel, quant_key_per_thread_int4_kernel


def _official_int4_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_mean: torch.Tensor,
    k_mean: torch.Tensor,
    *,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the official source INT4 quantizers and exact FP32 attention.

    SageAttention does not expose these kernels through its public attention
    API.  Centering uses the exact means returned by the bundled quantizer so
    the reported local-vs-official delta isolates quantization and rounding.
    """
    quantizers = _official_int4_quantizers()
    if quantizers is None:
        raise RuntimeError("official SageAttention INT4 quantizer source is unavailable")
    quant_query, quant_key = quantizers

    q_centered = q.float() - torch.repeat_interleave(q_mean, 64, dim=2)[:, :, : q.size(2)]
    k_centered = k.float() - k_mean
    q_values = torch.empty_like(q, dtype=torch.int8)
    k_values = torch.empty_like(k, dtype=torch.int8)
    q_blocks = (q.size(2) + 63) // 64
    k_blocks = (k.size(2) + 63) // 64
    q_scale = torch.empty(
        (q.size(0), q.size(1), q_blocks * 32), device=q.device, dtype=torch.float32
    )
    k_scale = torch.empty(
        (k.size(0), k.size(1), k_blocks * 4), device=k.device, dtype=torch.float32
    )
    q_grid = (q_blocks * 32, q.size(1), q.size(0))
    k_grid = (k_blocks * 4, k.size(1), k.size(0))
    quant_query[q_grid](
        q_centered,
        q_values,
        q_scale,
        q.size(2),
        q_centered.stride(0), q_centered.stride(1), q_centered.stride(2),
        q_values.stride(0), q_values.stride(1), q_values.stride(2),
        q_scale.stride(0), q_scale.stride(1),
        C=q.size(3), BLK=16,
    )
    quant_key[k_grid](
        k_centered,
        k_values,
        k_scale,
        k.size(2),
        k_centered.stride(0), k_centered.stride(1), k_centered.stride(2),
        k_values.stride(0), k_values.stride(1), k_values.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        C=k.size(3), BLK=64,
    )

    q_tokens = torch.arange(q.size(2), device=q.device)
    q_scale_index = (
        q_tokens.div(64, rounding_mode="floor") * 32
        + q_tokens.remainder(64).div(16, rounding_mode="floor") * 8
        + q_tokens.remainder(8)
    )
    k_tokens = torch.arange(k.size(2), device=k.device)
    k_scale_index = (
        k_tokens.div(64, rounding_mode="floor") * 4
        + k_tokens.remainder(8).div(2, rounding_mode="floor")
    )
    q_reconstructed = (
        q_values.float() * q_scale[:, :, q_scale_index].unsqueeze(-1)
        + torch.repeat_interleave(q_mean, 64, dim=2)[:, :, : q.size(2)]
    )
    k_reconstructed = k_values.float() * k_scale[:, :, k_scale_index].unsqueeze(-1) + k_mean
    return (
        _paper_int4_attention(
            q_reconstructed,
            k_reconstructed,
            q,
            k,
            q_mean,
            k_mean,
            v,
            smooth_q=True,
            smooth_k=True,
            is_causal=is_causal,
        ),
        q_values,
        k_values,
    )


def _official_int8_function():
    try:
        import sageattention
    except (ImportError, OSError):
        return None, None
    implementation = getattr(sageattention, "sageattn_qk_int8_pv_fp16_triton", None)
    if implementation is None:
        return None, None
    try:
        package_version = version("sageattention")
    except PackageNotFoundError:
        package_version = "unknown"
    return implementation, package_version


def _apply_profile(
    q: torch.Tensor,
    k: torch.Tensor,
    profile: str,
    input_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if profile == "gaussian":
        return q, k
    if profile != "biased":
        raise ValueError(f"unknown input profile: {profile}")

    # Smooth K targets a sequence-invariant per-channel K bias. Sage2's Q
    # smoothing additionally targets a per-64-token Q-block channel bias. Use
    # both here so the ablation measures the intended outlier regime rather
    # than only an approximately zero-mean Gaussian.
    dimensions = torch.arange(q.size(-1), device=q.device, dtype=torch.float32)
    k_bias = torch.cos(dimensions * 0.13) * (2.0 * input_scale)
    q_bias = torch.sin(dimensions * 0.17) * (1.5 * input_scale)
    q_blocks = torch.arange(q.size(2), device=q.device).div(64, rounding_mode="floor")
    q_block_sign = q_blocks.remainder(3).float() - 1.0
    q = (
        q.float()
        + q_block_sign.view(1, 1, -1, 1) * q_bias.view(1, 1, 1, -1)
    ).to(q.dtype)
    k = (k.float() + k_bias.view(1, 1, 1, -1)).to(k.dtype)
    return q, k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--sequence", type=int, default=257)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--profile",
        choices=("gaussian", "biased"),
        default="gaussian",
        help="biased adds the channel offsets targeted by Q/K smoothing",
    )
    args = parser.parse_args()
    if args.seeds <= 0 or args.sequence < 64:
        raise ValueError("seeds must be positive and sequence must be at least 64")

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    official_int8, official_version = _official_int8_function()
    official_int4 = _official_int4_quantizers() is not None
    methods = {
        "local-sage_-int8-per-warp-fp32": ErrorStats(),
        "local-sage1-int8": ErrorStats(),
        "local-sage1-int8-no-smooth": ErrorStats(),
        "local-paper-int4-math": ErrorStats(),
        "local-int4-math-no-smooth": ErrorStats(),
        "local-int4-math-smooth-k": ErrorStats(),
        "local-int4-math-smooth-q": ErrorStats(),
        "local-sage2-int4": ErrorStats(),
    }
    if official_int8 is not None:
        methods[f"official-{official_version}-int8"] = ErrorStats()
        methods[f"official-{official_version}-int8-no-smooth"] = ErrorStats()
    if official_int4:
        methods[f"official-{official_version}-int4-quant-math"] = ErrorStats()
    kernel_delta = ErrorStats()
    quantizer_math_delta = ErrorStats()
    quantizer_value_mismatches = 0
    quantizer_value_count = 0
    official_nonfinite_examples: list[str] = []

    with torch.inference_mode(), torch.cuda.device(device):
        for seed in range(args.seeds):
            for head_dim in (64, 128):
                for input_scale in (0.4, 1.0):
                    for is_causal in (False, True):
                        torch.manual_seed(seed * 1009 + head_dim * 7 + int(input_scale * 10))
                        q = torch.randn(
                            (1, 4, args.sequence, head_dim), device=device, dtype=dtype
                        ) * input_scale
                        k = torch.randn(
                            (1, 2, args.sequence, head_dim), device=device, dtype=dtype
                        ) * input_scale
                        v = torch.randn_like(k)
                        q, k = _apply_profile(q, k, args.profile, input_scale)
                        reference = torch.nn.functional.scaled_dot_product_attention(
                            q.float(), k.float(), v.float(),
                            is_causal=is_causal, enable_gqa=True,
                        )
                        local_sage1 = sageattn_sage1(q, k, v, is_causal=is_causal)
                        local_hybrid = sageattn_hybrid(
                            q, k, v, is_causal=is_causal, smooth_k=False
                        )
                        local_sage1_no_smooth = sageattn_sage1(
                            q, k, v, is_causal=is_causal, smooth_k=False
                        )
                        (
                            q_int4_math,
                            k_int4_math,
                            q_mean,
                            k_mean,
                            bundled_q_values,
                            bundled_k_values,
                        ) = _reconstruct_local_int4(q, k, smooth_q=True, smooth_k=True)
                        int4_math = _paper_int4_attention(
                            q_int4_math,
                            k_int4_math,
                            q,
                            k,
                            q_mean,
                            k_mean,
                            v,
                            smooth_q=True,
                            smooth_k=True,
                            is_causal=is_causal,
                        )
                        int4_ablations = {}
                        for label, smooth_q, smooth_k in (
                            ("local-int4-math-no-smooth", False, False),
                            ("local-int4-math-smooth-k", False, True),
                            ("local-int4-math-smooth-q", True, False),
                        ):
                            (
                                q_ablated,
                                k_ablated,
                                q_mean_ablated,
                                k_mean_ablated,
                                *_,
                            ) = _reconstruct_local_int4(
                                q, k, smooth_q=smooth_q, smooth_k=smooth_k
                            )
                            int4_ablations[label] = _paper_int4_attention(
                                q_ablated,
                                k_ablated,
                                q,
                                k,
                                q_mean_ablated,
                                k_mean_ablated,
                                v,
                                smooth_q=smooth_q,
                                smooth_k=smooth_k,
                                is_causal=is_causal,
                            )
                        local_sage2 = sageattn_sage2(q, k, v, is_causal=is_causal)
                        methods["local-sage_-int8-per-warp-fp32"].add(
                            local_hybrid, reference
                        )
                        methods["local-sage1-int8"].add(local_sage1, reference)
                        methods["local-sage1-int8-no-smooth"].add(
                            local_sage1_no_smooth, reference
                        )
                        methods["local-paper-int4-math"].add(int4_math, reference)
                        for label, ablated in int4_ablations.items():
                            methods[label].add(ablated, reference)
                        methods["local-sage2-int4"].add(local_sage2, reference)
                        kernel_delta.add(local_sage2, int4_math)
                        if official_int8 is not None:
                            official_output = official_int8(
                                q.clone(), k.clone(), v.clone(),
                                tensor_layout="HND", is_causal=is_causal,
                                smooth_k=True,
                            )
                            finite = methods[f"official-{official_version}-int8"].add(
                                official_output, reference, allow_nonfinite=True
                            )
                            if not finite and len(official_nonfinite_examples) < 8:
                                official_nonfinite_examples.append(
                                    f"smooth_k=true seed={seed} D={head_dim} "
                                    f"scale={input_scale} causal={is_causal}"
                                )
                            official_output_no_smooth = official_int8(
                                q.clone(), k.clone(), v.clone(),
                                tensor_layout="HND", is_causal=is_causal,
                                smooth_k=False,
                            )
                            finite = methods[f"official-{official_version}-int8-no-smooth"].add(
                                official_output_no_smooth,
                                reference,
                                allow_nonfinite=True,
                            )
                            if not finite and len(official_nonfinite_examples) < 8:
                                official_nonfinite_examples.append(
                                    f"smooth_k=false seed={seed} D={head_dim} "
                                    f"scale={input_scale} causal={is_causal}"
                                )
                        if official_int4:
                            official_int4_math, official_q_values, official_k_values = (
                                _official_int4_reference(
                                    q, k, v, q_mean, k_mean, is_causal=is_causal
                                )
                            )
                            methods[f"official-{official_version}-int4-quant-math"].add(
                                official_int4_math, reference
                            )
                            quantizer_math_delta.add(int4_math, official_int4_math)
                            # Reconstructed values include scales/means; compare
                            # the raw codes separately to expose rounding deltas.
                            quantizer_value_mismatches += int(
                                (bundled_q_values != official_q_values).sum().item()
                                + (bundled_k_values != official_k_values).sum().item()
                            )
                            quantizer_value_count += bundled_q_values.numel() + bundled_k_values.numel()
        torch.cuda.synchronize(device)

    print(
        f"device={torch.cuda.get_device_name(device)} dtype={dtype} "
        f"seeds={args.seeds} sequence={args.sequence} profile={args.profile}"
    )
    if official_int8 is None:
        print("official SageAttention INT8 comparison skipped: package/API unavailable")
    if not official_int4:
        print("official SageAttention INT4 source comparison skipped: quantizer unavailable")
    print(
        f"{'method':<42} {'MAE':>11} {'max_abs':>11} {'bias':>11} "
        f"{'RMSE':>11} {'nonfinite':>10}"
    )
    for name, stats in methods.items():
        mae, maximum, bias, rmse = stats.values()
        print(
            f"{name:<42} {mae:11.6g} {maximum:11.6g} {bias:11.6g} "
            f"{rmse:11.6g} {stats.nonfinite_cases:>4}/{stats.cases:<4}"
        )
    mae, maximum, bias, rmse = kernel_delta.values()
    print(f"{'local-sage2 vs INT4 math':<42} {mae:11.6g} {maximum:11.6g} {bias:11.6g} {rmse:11.6g}")
    if official_int4:
        mae, maximum, bias, rmse = quantizer_math_delta.values()
        print(f"{'local vs official INT4 math':<42} {mae:11.6g} {maximum:11.6g} {bias:11.6g} {rmse:11.6g}")
        mismatch_rate = quantizer_value_mismatches / max(1, quantizer_value_count)
        print(
            "INT4 raw-code mismatch: "
            f"{quantizer_value_mismatches}/{quantizer_value_count} ({mismatch_rate:.6%})"
        )
    if official_nonfinite_examples:
        print("official INT8 non-finite examples:")
        for example in official_nonfinite_examples:
            print(f"  {example}")


if __name__ == "__main__":
    main()
