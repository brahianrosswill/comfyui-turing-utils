#!/usr/bin/env python3
"""Compare bundled kernels with Kitchen and other available backends.

The report deliberately separates prequantized contraction time from the
end-to-end path.  Mixing those scopes was the source of several misleading
historical comparisons: a fast GEMM can still lose after activation rotation,
quantization, weight decode, output conversion, or an avoidable workspace.

Each device loads the matching cubin from a multi-architecture build. Some
portable ConvRot CUTLASS operators retain their SM75-compatible schedule on
newer GPUs, while attention selects newer instruction paths at compile time;
always compare both end-to-end and contraction-only scopes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
from typing import Callable


KERNEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KERNEL_ROOT))

import torch

import comfyui_turing_utils_kernel as turing


@dataclass(frozen=True)
class Measurement:
    name: str
    scope: str
    milliseconds: float


def _append_optional(
    measurements: list[Measurement],
    name: str,
    scope: str,
    function: Callable[[], torch.Tensor],
    warmup: int,
    repeats: int,
) -> bool:
    try:
        elapsed = _elapsed_ms(function, warmup, repeats)
    except Exception as error:
        print(f"{name} ({scope}) unavailable: {error}")
        return False
    measurements.append(Measurement(name, scope, elapsed))
    return True


def _elapsed_ms(function: Callable[[], torch.Tensor], warmup: int, repeats: int) -> float:
    result = None
    for _ in range(warmup):
        result = function()
    del result
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            result = function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    del result
    return statistics.median(samples)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = torch.linalg.vector_norm(actual.float() - expected.float())
    norm = torch.linalg.vector_norm(expected.float()).clamp_min(1.0e-12)
    return float(delta / norm)


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            actual.float().reshape(1, -1), expected.float().reshape(1, -1)
        ).item()
    )


def _pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    low = values[:, 0::2].to(torch.int32) & 0x0F
    high = values[:, 1::2].to(torch.int32) & 0x0F
    return (low | (high << 4)).to(torch.int8)


def _make_codebook_weight(
    n: int,
    k: int,
    device: torch.device,
    generator: torch.Generator,
    group_size: int = 16,
):
    codes = torch.randint(
        0, 16, (n, k), device=device, dtype=torch.int32, generator=generator
    )
    packed = (
        (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
    ).to(torch.int8)
    # A monotonic symmetric table is sufficient for arithmetic and timing
    # comparisons; format-quality tests belong to checkpoint calibration.
    codebook = torch.tensor(
        (-1.0, -0.72, -0.54, -0.42, -0.32, -0.23, -0.15, -0.07,
          0.0, 0.07, 0.15, 0.23, 0.32, 0.42, 0.54, 1.0),
        device=device,
        dtype=torch.float32,
    )
    group_scale = (
        torch.rand(
            (n, k // group_size), device=device, generator=generator
        )
        * 72.0
        + 8.0
    ).to(torch.float8_e4m3fn)
    decoded = (
        codebook[codes]
        * group_scale.float().repeat_interleave(group_size, dim=1)
    ).round().clamp(-127, 127).to(torch.int8)
    return packed, group_scale, codebook, decoded


def _print_table(title: str, measurements: list[Measurement]) -> None:
    print(f"\n{title}")
    print(f"{'scope':<18} {'implementation':<40} {'ms':>11} {'vs fastest':>11}")
    by_scope: dict[str, list[Measurement]] = {}
    for measurement in measurements:
        by_scope.setdefault(measurement.scope, []).append(measurement)
    for scope, values in by_scope.items():
        fastest = min(value.milliseconds for value in values)
        for value in values:
            print(
                f"{scope:<18} {value.name:<40} "
                f"{value.milliseconds:>11.3f} {value.milliseconds / fastest:>10.3f}x"
            )


def _kitchen_codebook_core(
    kitchen_cuda,
    activation: torch.Tensor,
    packed: torch.Tensor,
    activation_scale: torch.Tensor,
    group_scale: torch.Tensor,
    channel_scale: torch.Tensor,
    codebook: torch.Tensor,
):
    m, _ = activation.shape
    n, k_half = packed.shape
    k = k_half * 2
    chunk_rows = kitchen_cuda._int4_int8_weight_chunk_cols(m, n)

    def run():
        # Match the public bundled operator: output and bounded decode
        # workspace are owned by one call rather than retained by the harness.
        workspace = torch.empty(
            (min(chunk_rows, n), k), dtype=torch.int8, device=activation.device
        )
        output = torch.empty(
            (m, n), dtype=torch.bfloat16, device=activation.device
        )
        stream = torch.cuda.current_stream(activation.device).cuda_stream
        used = kitchen_cuda._C.w4a8_codebook_gemm_chunked(
            kitchen_cuda._wrap_for_dlpack(activation),
            kitchen_cuda._wrap_for_dlpack(packed),
            kitchen_cuda._wrap_for_dlpack(group_scale.view(torch.uint8)),
            kitchen_cuda._wrap_for_dlpack(codebook),
            kitchen_cuda._wrap_for_dlpack(channel_scale),
            kitchen_cuda._wrap_for_dlpack(activation_scale),
            None,
            kitchen_cuda._wrap_for_dlpack(workspace),
            kitchen_cuda._wrap_for_dlpack(output),
            16,
            chunk_rows,
            kitchen_cuda.DTYPE_TO_CODE[torch.bfloat16],
            stream,
        )
        if not used:
            raise RuntimeError("Kitchen rejected the grouped-codebook benchmark shape")
        return output

    return run


def compare_w4_format_quality(device: torch.device) -> None:
    """Compare checkpoint-format error independently of kernel arithmetic."""
    try:
        from comfy_kitchen.backends import eager as kitchen_eager
    except ImportError:
        print("Kitchen eager backend unavailable; skipping W4 format quality comparison")
        return

    generator = torch.Generator(device=device).manual_seed(5199)
    weight = torch.randn(
        (512, 5376), device=device, dtype=torch.bfloat16, generator=generator
    ) * 0.02
    legacy_q, legacy_scale = kitchen_eager.quantize_convrot_w4a4_weight(
        weight,
        convrot_groupsize=256,
        quant_group_size=64,
        stochastic_rounding=0,
    )
    legacy = kitchen_eager.dequantize_convrot_w4a4_weight(
        legacy_q,
        legacy_scale,
        convrot_groupsize=256,
        quant_group_size=64,
        output_dtype=torch.float32,
    )
    qdata, s_rel, s_channel, correction, codebook = (
        kitchen_eager.quantize_w4a8_int8_weight(
            weight,
            group_size=16,
            convrot_groupsize=256,
            symmetric=True,
            scale_dtype=torch.float8_e4m3fn,
            codebook=True,
            stochastic_rounding=0,
        )
    )
    grouped = kitchen_eager.dequantize_w4a8_int8_weight(
        qdata,
        s_rel,
        s_channel,
        codebook=codebook,
        correction=correction,
        group_size=16,
        convrot_groupsize=256,
        output_dtype=torch.float32,
    )
    reference = weight.float()
    print("\nW4 checkpoint-format quality (Gaussian 512x5376)")
    print(f"{'format':<34} {'relative L2':>14} {'cosine':>14} {'max abs':>14}")
    for name, reconstructed in (
        ("legacy ConvRot signed W4", legacy),
        ("grouped-codebook W4 g16", grouped),
    ):
        print(
            f"{name:<34} {_relative_l2(reconstructed, reference):>14.7f} "
            f"{_cosine(reconstructed, reference):>14.9f} "
            f"{float((reconstructed - reference).abs().max()):>14.7f}"
        )


def benchmark_linear(
    device: torch.device,
    rows: tuple[int, ...],
    warmup: int,
    repeats: int,
) -> None:
    try:
        from comfy_kitchen.backends import cuda as kitchen_cuda
    except ImportError:
        kitchen_cuda = None
    if kitchen_cuda is not None:
        try:
            smoke_x = torch.zeros((8, 256), dtype=torch.bfloat16, device=device)
            smoke_w = torch.zeros((8, 256), dtype=torch.int8, device=device)
            kitchen_cuda.int8_linear(
                smoke_x,
                smoke_w,
                torch.ones((8,), dtype=torch.float32, device=device),
                out_dtype=torch.bfloat16,
                convrot=True,
                convrot_groupsize=256,
            )
            torch.cuda.synchronize(device)
        except Exception as error:
            print(f"Kitchen CUDA linear comparison unavailable: {error}")
            kitchen_cuda = None

    compare_w4_format_quality(device)

    shapes = (
        ("qkv", 21504, 5376, None),
        ("fc1", 28672, 5376, None),
        # H3 fc2 consumes the activated half of fc1's [gate | up] output.
        # Benchmarking a plain BF16 [M, 28672] tensor both misses the SwiGLU
        # fusion and selects a row-buffer path that production never uses.
        ("fc2", 5376, 14336, "swiglu"),
    )
    for m in rows:
        for shape_name, n, k, input_act in shapes:
            generator = torch.Generator(device=device).manual_seed(5200 + m + n + k)
            activation = torch.randint(
                -127, 128, (m, k), device=device, dtype=torch.int8, generator=generator
            )
            activation_scale = torch.rand(
                (m,), device=device, generator=generator
            ) * 0.01 + 1.0e-4
            channel_scale = torch.rand(
                (n,), device=device, generator=generator
            ) * 0.02 + 1.0e-4
            packed_codebook, group_scale, codebook, decoded = _make_codebook_weight(
                n, k, device, generator
            )
            signed_w4 = torch.randint(
                -8, 8, (n, k), device=device, dtype=torch.int8, generator=generator
            )
            packed_signed = _pack_signed_int4(signed_w4)

            measurements = [
                Measurement(
                    "bundled raw W8A8",
                    "prequantized",
                    _elapsed_ms(
                        lambda: turing.turing_int8_linear(
                            activation, decoded, activation_scale, channel_scale
                        ),
                        warmup,
                        repeats,
                    ),
                ),
                Measurement(
                    "bundled legacy packed W4A8",
                    "prequantized",
                    _elapsed_ms(
                        lambda: turing.turing_w4a8_linear(
                            activation, packed_signed, activation_scale, channel_scale
                        ),
                        warmup,
                        repeats,
                    ),
                ),
                Measurement(
                    "bundled grouped-codebook W4A8 inline/auto",
                    "prequantized",
                    _elapsed_ms(
                        lambda: turing.turing_codebook_w4a8_linear(
                            activation,
                            packed_codebook,
                            activation_scale,
                            group_scale,
                            channel_scale,
                            codebook,
                        ),
                        warmup,
                        repeats,
                    ),
                ),
            ]
            if m > 8192:
                measurements.append(
                    Measurement(
                        "bundled grouped-codebook W4A8 staged",
                        "prequantized",
                        _elapsed_ms(
                            lambda: turing.turing_codebook_w4a8_linear(
                                activation,
                                packed_codebook,
                                activation_scale,
                                group_scale,
                                channel_scale,
                                codebook,
                                chunk_rows=4096,
                            ),
                            warmup,
                            repeats,
                        ),
                    )
                )

            # End-to-end bundled scope includes BF16 ConvRot activation
            # quantization, weight decode, contraction and BF16 output.  It
            # intentionally does not depend on Kitchen being usable.
            input_columns = k * (2 if input_act == "swiglu" else 1)
            x = torch.randn(
                (m, input_columns),
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )

            def bundled_quantize():
                if input_act == "swiglu":
                    return turing.turing_swiglu_int8_convrot_quantize(x, 256)
                return turing.turing_bf16_int8_convrot_quantize(
                    x, 256, swiglu=False
                )

            def bundled_codebook_e2e():
                qx, sx = bundled_quantize()
                return turing.turing_codebook_w4a8_linear(
                    qx, packed_codebook, sx, group_scale, channel_scale, codebook
                )

            def bundled_codebook_staged_e2e():
                qx, sx = bundled_quantize()
                return turing.turing_codebook_w4a8_linear(
                    qx,
                    packed_codebook,
                    sx,
                    group_scale,
                    channel_scale,
                    codebook,
                    chunk_rows=4096,
                )

            def bundled_w8_e2e():
                qx, sx = bundled_quantize()
                return turing.turing_int8_linear(qx, decoded, sx, channel_scale)

            def kitchen_codebook_e2e():
                kitchen_x = x
                if input_act == "swiglu":
                    gate, up = x.chunk(2, dim=-1)
                    kitchen_x = torch.nn.functional.silu(gate) * up
                return kitchen_cuda.w4a8_int8_linear(
                    kitchen_x,
                    packed_codebook,
                    group_scale,
                    channel_scale,
                    codebook=codebook,
                    group_size=16,
                    convrot_groupsize=256,
                    out_dtype=torch.bfloat16,
                )

            _append_optional(
                measurements,
                "bundled grouped-codebook W4A8 inline/auto",
                "end-to-end",
                bundled_codebook_e2e,
                warmup,
                repeats,
            )
            if m > 8192:
                _append_optional(
                    measurements,
                    "bundled grouped-codebook W4A8 staged",
                    "end-to-end",
                    bundled_codebook_staged_e2e,
                    warmup,
                    repeats,
                )
            _append_optional(
                measurements,
                "bundled W8A8",
                "end-to-end",
                bundled_w8_e2e,
                warmup,
                repeats,
            )

            if kitchen_cuda is not None:
                measurements.extend(
                    (
                        Measurement(
                            "Kitchen raw W8A8",
                            "prequantized",
                            _elapsed_ms(
                                lambda: kitchen_cuda._int4_linear_via_int8_values(
                                    activation,
                                    decoded,
                                    activation_scale,
                                    channel_scale,
                                    None,
                                    torch.bfloat16,
                                ),
                                warmup,
                                repeats,
                            ),
                        ),
                        Measurement(
                            "Kitchen legacy chunked W4A8",
                            "prequantized",
                            _elapsed_ms(
                                lambda: kitchen_cuda._int4_weight_int8_act_gemm_dequant_chunked(
                                    activation,
                                    packed_signed,
                                    activation_scale,
                                    channel_scale,
                                    None,
                                    torch.bfloat16,
                                ),
                                warmup,
                                repeats,
                            ),
                        ),
                        Measurement(
                            "Kitchen grouped-codebook W4A8",
                            "prequantized",
                            _elapsed_ms(
                                _kitchen_codebook_core(
                                    kitchen_cuda,
                                    activation,
                                    packed_codebook,
                                    activation_scale,
                                    group_scale,
                                    channel_scale,
                                    codebook,
                                ),
                                warmup,
                                repeats,
                            ),
                        ),
                    )
                )

                measurements.extend(
                    (
                        Measurement(
                            "Kitchen grouped-codebook W4A8",
                            "end-to-end",
                            _elapsed_ms(kitchen_codebook_e2e, warmup, repeats),
                        ),
                        Measurement(
                            "Kitchen W8A8",
                            "end-to-end",
                            _elapsed_ms(
                                lambda: kitchen_cuda.int8_linear(
                                    x,
                                    decoded,
                                    channel_scale,
                                    out_dtype=torch.bfloat16,
                                    convrot=True,
                                    convrot_groupsize=256,
                                    input_act=input_act,
                                ),
                                warmup,
                                repeats,
                            ),
                        ),
                    )
                )

            # A small output sample is enough to catch scale order, decode and
            # epilogue regressions without allocating a floating H3 weight.
            sample_rows = min(m, 8)
            actual = turing.turing_codebook_w4a8_linear(
                activation[:sample_rows],
                packed_codebook,
                activation_scale[:sample_rows],
                group_scale,
                channel_scale,
                codebook,
            )
            reference = (
                activation[:sample_rows].float() @ decoded.float().t()
            ) * activation_scale[:sample_rows, None] * channel_scale[None, :]
            print(
                f"\nlinear shape={shape_name} M={m} N={n} K={k} "
                f"codebook arithmetic rel_l2={_relative_l2(actual, reference):.6g} "
                f"cosine={_cosine(actual, reference):.8f}"
            )
            _print_table("backend timing", measurements)
            del (
                activation,
                activation_scale,
                channel_scale,
                packed_codebook,
                group_scale,
                codebook,
                decoded,
                signed_w4,
                packed_signed,
                x,
                actual,
                reference,
            )
            torch.cuda.empty_cache()


def benchmark_attention(
    device: torch.device,
    sequences: tuple[int, ...],
    heads: int,
    kv_heads: int,
    head_dim: int,
    warmup: int,
    repeats: int,
) -> None:
    try:
        import comfy_kitchen
    except ImportError:
        comfy_kitchen = None
    from comfyui_turing_utils_kernel.turing_sage import (
        prequantize_sageattn,
        sageattn,
        sageattn_from_prequantized,
        w8a8attn,
    )

    external_sage = None
    try:
        from sageattention import sageattn as external_sage
    except ImportError:
        pass

    kitchen_available = comfy_kitchen is not None

    for sequence in sequences:
        generator = torch.Generator(device=device).manual_seed(6100 + sequence)
        q = torch.randn(
            (1, heads, sequence, head_dim),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        k = torch.randn(
            (1, kv_heads, sequence, head_dim),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        v = torch.randn(
            k.shape, device=device, dtype=torch.bfloat16, generator=generator
        )
        measurements = [
            Measurement(
                "bundled Sage",
                "end-to-end",
                _elapsed_ms(lambda: sageattn(q, k, v), warmup, repeats),
            ),
            Measurement(
                "bundled W8A8",
                "end-to-end",
                _elapsed_ms(lambda: w8a8attn(q, k, v), warmup, repeats),
            ),
            Measurement(
                "PyTorch SDPA",
                "end-to-end",
                _elapsed_ms(
                    lambda: torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, enable_gqa=True
                    ),
                    warmup,
                    repeats,
                ),
            ),
        ]
        if kitchen_available:
            try:
                elapsed = _elapsed_ms(
                    lambda: comfy_kitchen.int8_attention(q, k, v), warmup, repeats
                )
            except Exception as error:
                print(f"Kitchen INT8 attention comparison unavailable: {error}")
                kitchen_available = False
            else:
                measurements.append(
                    Measurement("Kitchen INT8 attention", "end-to-end", elapsed)
                )
        if external_sage is not None:
            try:
                elapsed = _elapsed_ms(
                    lambda: external_sage(q, k, v, tensor_layout="HND"),
                    warmup,
                    repeats,
                )
            except Exception as error:
                print(f"external SageAttention unavailable for this shape: {error}")
            else:
                measurements.append(
                    Measurement("external SageAttention", "end-to-end", elapsed)
                )

        bundled_state = __import__(
            "comfyui_turing_utils_kernel.turing_sage", fromlist=["prequantize_sol_sageattn"]
        ).prequantize_sol_sageattn(q, k, v, use_w8a8=True, force_dense=True)
        sage_state = prequantize_sageattn(q, k, v)
        from comfyui_turing_utils_kernel.turing_sage import (
            sol_sparse_sageattn_from_prequantized,
        )

        measurements.append(
            Measurement(
                "bundled Sage",
                "prequantized",
                _elapsed_ms(
                    lambda: sageattn_from_prequantized(sage_state),
                    warmup,
                    repeats,
                ),
            )
        )
        measurements.append(
            Measurement(
                "bundled W8A8",
                "prequantized",
                _elapsed_ms(
                    lambda: sol_sparse_sageattn_from_prequantized(bundled_state),
                    warmup,
                    repeats,
                ),
            )
        )
        kitchen_state = None
        if kitchen_available:
            kitchen_state = comfy_kitchen.prequantize_int8_attention(q, k, v)
            measurements.append(
                Measurement(
                    "Kitchen INT8 attention",
                    "prequantized",
                    _elapsed_ms(
                        lambda: comfy_kitchen.int8_attention_from_prequantized(
                            kitchen_state
                        ),
                        warmup,
                        repeats,
                    ),
                )
            )

        # Use SDPA as a diagnostic reference, not as a quality oracle for the
        # two intentionally different INT8 quantizers.
        reference = torch.nn.functional.scaled_dot_product_attention(
            q[:, :, :128].float(),
            k[:, :, :128].float(),
            v[:, :, :128].float(),
            enable_gqa=True,
        )
        bundled = w8a8attn(q[:, :, :128], k[:, :, :128], v[:, :, :128])
        kitchen = None
        if kitchen_available:
            kitchen = comfy_kitchen.int8_attention(
                q[:, :, :128], k[:, :, :128], v[:, :, :128]
            )
        print(
            f"\nattention B=1 Hq={heads} Hkv={kv_heads} "
            f"N={sequence} D={head_dim} "
            f"bundled rel_l2={_relative_l2(bundled, reference):.6g}"
            + (
                f" Kitchen rel_l2={_relative_l2(kitchen, reference):.6g}"
                if kitchen is not None
                else ""
            )
        )
        _print_table("backend timing", measurements)
        del q, k, v, bundled_state, sage_state, reference, bundled
        if kitchen_state is not None:
            del kitchen_state
        if kitchen is not None:
            del kitchen
        torch.cuda.empty_cache()


def benchmark_preprocessing(
    device: torch.device,
    rows: tuple[int, ...],
    warmup: int,
    repeats: int,
) -> None:
    """Compare the bandwidth-bound operators surrounding H3 contractions."""
    try:
        from comfy_kitchen.backends import cuda as kitchen_cuda
    except ImportError:
        kitchen_cuda = None

    for m in rows:
        generator = torch.Generator(device=device).manual_seed(7300 + m)
        for name, raw_k, input_act in (
            ("H3 qkv/fc1 ConvRot A8", 5376, None),
            ("H3 fc2 fused SwiGLU+ConvRot A8", 28672, "swiglu"),
            ("Wan tanh-GELU+ConvRot A8", 5120, "gelu_tanh"),
        ):
            x = torch.randn(
                (m, raw_k),
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            if input_act == "swiglu":
                bundled = lambda: turing.turing_bf16_int8_convrot_quantize(
                    x, 256, swiglu=True
                )
                bundled_int4 = lambda: turing.turing_bf16_int4_convrot_quantize(
                    x, 256, swiglu=True
                )
                staged = lambda: turing.turing_swiglu_int8_convrot_quantize(x, 256)
                staged_int4 = lambda: turing.turing_swiglu_int4_convrot_quantize(x, 256)
            elif input_act == "gelu_tanh":
                bundled = lambda: turing.turing_bf16_gelu_int8_convrot_quantize(
                    x, 256
                )
                bundled_int4 = lambda: turing.turing_bf16_gelu_int4_convrot_quantize(
                    x, 256
                )
            else:
                bundled = lambda: turing.turing_bf16_int8_convrot_quantize(
                    x, 256, swiglu=False
                )
                bundled_int4 = lambda: turing.turing_bf16_int4_convrot_quantize(
                    x, 256, swiglu=False
                )

            def activated_input():
                if input_act == "swiglu":
                    gate, up = x.chunk(2, dim=-1)
                    return torch.nn.functional.silu(gate) * up
                if input_act == "gelu_tanh":
                    return torch.nn.functional.gelu(x, approximate="tanh")
                return x

            def kitchen_staged():
                return kitchen_cuda.quantize_int8_convrot_staged(
                    activated_input(), 256
                )

            def kitchen_int4():
                return kitchen_cuda.quantize_int4_rowwise_convrot64(
                    activated_input(), 256
                )

            measurements = [
                Measurement(
                    "bundled native",
                    "end-to-end",
                    _elapsed_ms(bundled, warmup, repeats),
                )
            ]
            if input_act == "swiglu":
                _append_optional(
                    measurements,
                    "bundled staged compatibility path",
                    "end-to-end",
                    staged,
                    warmup,
                    repeats,
                )
            if input_act == "swiglu" and raw_k == 28672:
                # Forced geometry remains a resource-validation aid; the
                # registered custom op uses its production selector.
                core = getattr(turing, "_C", None)
                if core is not None:
                    for threads in (512, 768, 1024):
                        _append_optional(
                            measurements,
                            f"bundled forced {threads} threads",
                            "geometry",
                            lambda threads=threads: core.turing_bf16_int8_convrot_quantize(
                                x, 256, True, threads
                            ),
                            warmup,
                            repeats,
                        )
            if kitchen_cuda is not None:
                _append_optional(
                    measurements,
                    "Kitchen convrot64 candidate",
                    "end-to-end",
                    lambda: kitchen_cuda.quantize_int8_rowwise_convrot64(
                        x,
                        256,
                        input_act=input_act,
                    ),
                    warmup,
                    repeats,
                )
                _append_optional(
                    measurements,
                    "Kitchen staged fallback",
                    "end-to-end",
                    kitchen_staged,
                    warmup,
                    repeats,
                )
            _print_table(f"{name} M={m} raw_K={raw_k}", measurements)

            int4_measurements = [
                Measurement(
                    "bundled native",
                    "end-to-end",
                    _elapsed_ms(bundled_int4, warmup, repeats),
                )
            ]
            if input_act == "swiglu":
                _append_optional(
                    int4_measurements,
                    "bundled staged compatibility path",
                    "end-to-end",
                    staged_int4,
                    warmup,
                    repeats,
                )
            if kitchen_cuda is not None:
                _append_optional(
                    int4_measurements,
                    "Kitchen convrot64",
                    "end-to-end",
                    kitchen_int4,
                    warmup,
                    repeats,
                )
            _print_table(
                f"{name.replace('A8', 'A4')} M={m} raw_K={raw_k}",
                int4_measurements,
            )
            del x
            torch.cuda.empty_cache()

        hidden = 5376
        x = torch.randn(
            (m, hidden),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        weight = torch.ones(hidden, device=device, dtype=torch.bfloat16)
        scale = torch.randn(
            (1, hidden),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ) * 0.02
        shift = torch.randn(
            (1, hidden),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ) * 0.02
        segments = torch.tensor(((0, m, 0),), device=device, dtype=torch.int32)
        measurements = [
            Measurement(
                "bundled segmented RMSNorm+AdaLN",
                "end-to-end",
                _elapsed_ms(
                    lambda: turing.turing_segmented_rms_adaln(
                        x, weight, scale, shift, segments, 1.0e-6
                    ),
                    warmup,
                    repeats,
                ),
            )
        ]
        if kitchen_cuda is not None:
            _append_optional(
                measurements,
                "Kitchen RMS+AdaLN (unit weight)",
                "end-to-end",
                lambda: kitchen_cuda.rms_adaln(x, scale[0], shift[0], 1.0e-6),
                warmup,
                repeats,
            )
        _print_table(f"H3 RMSNorm+AdaLN M={m} K={hidden}", measurements)

        layer_measurements = [
            Measurement(
                "bundled LayerNorm+AdaLN",
                "end-to-end",
                _elapsed_ms(
                    lambda: turing.turing_layer_norm_adaln(
                        x.unsqueeze(0),
                        scale.unsqueeze(0),
                        shift.unsqueeze(0),
                        1.0e-6,
                    ),
                    warmup,
                    repeats,
                ),
            )
        ]
        if kitchen_cuda is not None:
            _append_optional(
                layer_measurements,
                "Kitchen LayerNorm+AdaLN",
                "end-to-end",
                lambda: kitchen_cuda.adaln(x, scale[0], shift[0], 1.0e-6),
                warmup,
                repeats,
            )
        _print_table(f"LayerNorm+AdaLN M={m} K={hidden}", layer_measurements)

        accumulator = torch.randint(
            -4096,
            4097,
            (m, hidden),
            device=device,
            dtype=torch.int32,
            generator=generator,
        )
        row_scale = torch.rand((m,), device=device, generator=generator)
        column_scale = torch.rand((hidden,), device=device, generator=generator)
        epilogue_measurements = [
            Measurement(
                "bundled vectorized BF16 epilogue",
                "end-to-end",
                _elapsed_ms(
                    lambda: turing.turing_dequantize_int8_bf16(
                        accumulator, row_scale, column_scale
                    ),
                    warmup,
                    repeats,
                ),
            ),
            Measurement(
                "PyTorch eager",
                "end-to-end",
                _elapsed_ms(
                    lambda: (
                        accumulator.float()
                        * row_scale[:, None]
                        * column_scale[None, :]
                    ).to(torch.bfloat16),
                    warmup,
                    repeats,
                ),
            ),
        ]
        _print_table(f"INT32 scale+BF16 epilogue M={m} N={hidden}", epilogue_measurements)
        del (
            x,
            weight,
            scale,
            shift,
            segments,
            accumulator,
            row_scale,
            column_scale,
        )
        torch.cuda.empty_cache()


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--suite",
        choices=("linear", "preprocess", "attention", "quality", "all"),
        default="all",
    )
    parser.add_argument("--rows", type=_parse_ints, default=(4096, 8192))
    parser.add_argument("--sequences", type=_parse_ints, default=(4096, 8192))
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--kv-heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("benchmark requires a CUDA device")
    capability = torch.cuda.get_device_capability(device)
    properties = torch.cuda.get_device_properties(device)
    print(
        f"kernel={Path(turing.__file__).resolve()} version={turing.__version__}\n"
        f"torch={torch.__version__} cuda={torch.version.cuda}\n"
        f"device={properties.name} capability=sm{capability[0]}{capability[1]} "
        f"shared_per_block={properties.shared_memory_per_block}"
    )
    if capability != (7, 5):
        print(
            "NOTE: native cubin selected; portable linear schedules and "
            "architecture-specialized attention are reported separately."
        )
    with torch.inference_mode(), torch.cuda.device(device):
        if args.suite == "quality":
            compare_w4_format_quality(device)
        if args.suite in ("linear", "all"):
            benchmark_linear(
                device,
                args.rows,
                args.warmup,
                args.repeats,
            )
        if args.suite in ("preprocess", "all"):
            benchmark_preprocessing(device, args.rows, args.warmup, args.repeats)
        if args.suite in ("attention", "all"):
            if args.heads <= 0 or args.kv_heads <= 0 or args.heads % args.kv_heads:
                raise ValueError("heads must be positive and divisible by kv-heads")
            benchmark_attention(
                device,
                args.sequences,
                args.heads,
                args.kv_heads,
                args.head_dim,
                args.warmup,
                args.repeats,
            )


if __name__ == "__main__":
    main()
