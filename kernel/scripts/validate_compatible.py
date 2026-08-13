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

import comfyui_turing_utils_kernel


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
            staged = getattr(comfyui_turing_utils_kernel, f"turing_swiglu_int{bits}_convrot_quantize")
            rowbuffer = getattr(comfyui_turing_utils_kernel, f"turing_bf16_int{bits}_convrot_quantize")
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
        lambda: comfyui_turing_utils_kernel.turing_swiglu_int4_convrot_quantize(zero, 256),
        lambda: comfyui_turing_utils_kernel.turing_bf16_int4_convrot_quantize(zero, 256, swiglu=True),
    ):
        packed, scale = implementation()
        if torch.count_nonzero(_unpack_int4(packed)):
            raise RuntimeError("zero ConvRot INT4 input must quantize to zero")
        if not torch.equal(scale, torch.full_like(scale, 1.0e-10)):
            raise RuntimeError("ConvRot INT4 scale floor must match comfy-kitchen")


def validate_w4a8(device: torch.device) -> None:
    # The first two cases exercise the compatibility path. The remaining
    # cases cover every production Tensor Core tile and predicated edge tiles.
    cases = (
        (7, 13, 64),
        (3, 16, 12),
        (1, 8, 16),
        (17, 72, 80),
        (33, 128, 128),
        (129, 264, 256),
        (513, 520, 64),
        (8193, 8, 16),
    )
    for index, (m, n, k) in enumerate(cases):
        generator = torch.Generator(device=device).manual_seed(4300 + index)
        activation = torch.randint(
            -128, 128, (m, k), generator=generator, device=device, dtype=torch.int8
        )
        weight = torch.randint(
            -8, 8, (n, k), generator=generator, device=device, dtype=torch.int8
        )
        activation_scale = torch.rand((m,), generator=generator, device=device) * 0.02 + 0.001
        weight_scale = torch.rand((n,), generator=generator, device=device) * 0.03 + 0.001
        bias = None
        if index % 2 == 0:
            bias = torch.randn(
                (n,), generator=generator, device=device, dtype=torch.bfloat16
            ) * 0.1
        output = comfyui_turing_utils_kernel.turing_w4a8_linear(
            activation, _pack_int4(weight), activation_scale, weight_scale, bias
        )
        reference = (
            activation.float() @ weight.float().t()
        ) * activation_scale[:, None] * weight_scale[None, :]
        if bias is not None:
            reference = reference + bias.float()
        _assert_close(
            f"packed W4A8 M={m} N={n} K={k}",
            output,
            reference,
            rtol=0.01,
            atol=0.01,
        )

    # Kijai's current MiniMax-H3 W4A8 files use Kitchen's symmetric grouped
    # codebook layout: packed 4-bit indices, E4M3 relative group scales, and
    # one FP32 channel scale.  Validate the exact decode contract separately
    # from the legacy signed-nibble format above.
    generator = torch.Generator(device=device).manual_seed(4388)
    m, n, k, group_size = 7, 24, 256, 16
    activation = torch.randint(
        -128, 128, (m, k), generator=generator, device=device, dtype=torch.int8
    )
    codes = torch.randint(
        0, 16, (n, k), generator=generator, device=device, dtype=torch.int32
    )
    packed_codes = (
        (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
    ).to(torch.int8)
    codebook = torch.linspace(-1.0, 1.0, 16, device=device, dtype=torch.float32)
    group_scale = (
        torch.rand(
            (n, k // group_size), generator=generator, device=device
        )
        * 64.0
        + 4.0
    ).to(torch.float8_e4m3fn)
    activation_scale = torch.rand((m,), generator=generator, device=device) * 0.01
    channel_scale = torch.rand((n,), generator=generator, device=device) * 0.02
    bias = torch.randn((n,), generator=generator, device=device, dtype=torch.bfloat16)
    output = comfyui_turing_utils_kernel.turing_codebook_w4a8_linear(
        activation,
        packed_codes,
        activation_scale,
        group_scale,
        channel_scale,
        codebook,
        bias,
        group_size,
    )
    decoded = (
        codebook[codes]
        * group_scale.float().repeat_interleave(group_size, dim=1)
    ).round().clamp(-127, 127)
    reference = (
        activation.float() @ decoded.float().t()
    ) * activation_scale[:, None] * channel_scale[None, :] + bias.float()
    _assert_close(
        "grouped-codebook W4A8",
        output,
        reference,
        rtol=0.01,
        atol=0.02,
    )

    # The long-sequence policy decodes packed W4 directly while filling the
    # CUTLASS shared tile. It must stay bit exact with the staged decoder,
    # including a predicated N edge.
    long_activation = torch.randint(
        -128,
        128,
        (8193, k),
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    long_scale = torch.rand(
        (8193,), generator=generator, device=device
    ) * 0.01
    inline = comfyui_turing_utils_kernel.turing_codebook_w4a8_linear(
        long_activation,
        packed_codes,
        long_scale,
        group_scale,
        channel_scale,
        codebook,
        bias,
        group_size,
    )
    staged = comfyui_turing_utils_kernel.turing_codebook_w4a8_linear(
        long_activation,
        packed_codes,
        long_scale,
        group_scale,
        channel_scale,
        codebook,
        bias,
        group_size,
        chunk_rows=n,
    )
    if not torch.equal(inline, staged):
        raise RuntimeError("inline grouped-codebook W4A8 must be bit exact with staged decode")


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
        output = comfyui_turing_utils_kernel.turing_segmented_rms_adaln(
            x, weight, scale, shift, segments, 1.0e-5
        )
        norm = x.float() * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + 1.0e-5
        )
        reference = norm * weight.float()
        reference = reference * (1.0 + scale.float()[row_ids]) + shift.float()[row_ids]
        _assert_close(f"segmented norm {dtype}", output, reference, rtol=0.01, atol=0.02)


def _reference_qk_preprocessing(
    value: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    *,
    epsilon: float,
    rot_dim: int,
    norm_scope: str,
    split_half: bool,
) -> torch.Tensor:
    batch, heads, sequence, head_dim = value.shape
    if norm_scope == "head":
        rrms = torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + epsilon)
        normalized = (value.float() * rrms * weight.float()).to(value.dtype)
    else:
        rows = value.transpose(1, 2).reshape(batch, sequence, heads * head_dim)
        rrms = torch.rsqrt(rows.float().square().mean(dim=-1, keepdim=True) + epsilon)
        normalized = (rows.float() * rrms * weight.float()).to(value.dtype)
        normalized = normalized.view(batch, sequence, heads, head_dim).transpose(1, 2)
    if rot_dim == 0:
        return normalized

    pairs = rot_dim // 2
    prefix = normalized[..., :rot_dim]
    if split_half:
        first, second = prefix[..., :pairs], prefix[..., pairs:]
    else:
        paired = prefix.reshape(batch, heads, sequence, pairs, 2)
        first, second = paired[..., 0], paired[..., 1]
    matrix = freqs[:, :sequence, 0]
    out0 = (
        matrix[..., 0, 0].unsqueeze(1) * first
        + matrix[..., 0, 1].unsqueeze(1) * second
    ).to(value.dtype)
    out1 = (
        matrix[..., 1, 0].unsqueeze(1) * first
        + matrix[..., 1, 1].unsqueeze(1) * second
    ).to(value.dtype)
    rotated = (
        torch.cat((out0, out1), dim=-1)
        if split_half
        else torch.stack((out0, out1), dim=-1).reshape_as(prefix)
    )
    if rot_dim == head_dim:
        return rotated
    return torch.cat((rotated, normalized[..., rot_dim:]), dim=-1)


def validate_qk_preprocessing(device: torch.device) -> None:
    from comfyui_turing_utils_kernel.turing_sage.core import prequantize_rms_rope_qk
    from comfyui_turing_utils_kernel.turing_sage.quant import (
        per_warp_int8,
        per_warp_int8_hadamard,
    )

    for dtype in (torch.float16, torch.bfloat16):
        for head_dim, norm_scope, split_half in (
            (64, "head", True),
            (128, "head", True),
            (64, "row", False),
            (128, "row", False),
        ):
            batch, heads, sequence = 1, 3, 129
            generator = torch.Generator(device=device).manual_seed(
                4700 + head_dim + (100 if norm_scope == "row" else 0)
            )
            query = torch.randn(
                (batch, heads, sequence, head_dim),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            key = torch.randn(
                query.shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            norm_size = head_dim if norm_scope == "head" else heads * head_dim
            query_norm = torch.randn(
                norm_size, generator=generator, device=device, dtype=dtype
            )
            key_norm = torch.randn(
                norm_size, generator=generator, device=device, dtype=dtype
            )
            rot_dim = head_dim if norm_scope == "row" else head_dim - 32
            freqs = torch.randn(
                (batch, sequence, 1, rot_dim // 2, 2, 2),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            reference_query = _reference_qk_preprocessing(
                query,
                query_norm,
                freqs,
                epsilon=1.0e-6,
                rot_dim=rot_dim,
                norm_scope=norm_scope,
                split_half=split_half,
            )
            reference_key = _reference_qk_preprocessing(
                key,
                key_norm,
                freqs,
                epsilon=1.0e-6,
                rot_dim=rot_dim,
                norm_scope=norm_scope,
                split_half=split_half,
            )
            reference = per_warp_int8(reference_query, reference_key)
            fused = prequantize_rms_rope_qk(
                query,
                key,
                query_norm,
                key_norm,
                freqs,
                epsilon=1.0e-6,
                rot_dim=rot_dim,
                norm_scope=norm_scope,
                split_half=split_half,
            )
            actual = (
                fused.query_int8,
                fused.query_scale,
                fused.key_int8,
                fused.key_scale,
            )
            for index, (expected, result) in enumerate(zip(reference, actual)):
                if expected.dtype == torch.int8:
                    max_lsb = int(
                        (expected.to(torch.int16) - result.to(torch.int16))
                        .abs()
                        .max()
                        .item()
                    )
                    if max_lsb > 2:
                        raise RuntimeError(
                            f"fused Q/K preprocessing INT8 mismatch: {dtype=} "
                            f"{head_dim=} {norm_scope=} tensor={index} max_lsb={max_lsb}"
                        )
                else:
                    _assert_close(
                        "fused Q/K preprocessing scale",
                        result,
                        expected,
                        rtol=0.015,
                        atol=0.0011,
                    )

            rotated_reference = per_warp_int8_hadamard(
                reference_query, reference_key, stabilize_k=True
            )
            rotated = prequantize_rms_rope_qk(
                query,
                key,
                query_norm,
                key_norm,
                freqs,
                epsilon=1.0e-6,
                rot_dim=rot_dim,
                norm_scope=norm_scope,
                split_half=split_half,
                rotate_qk=True,
                stabilize_k=True,
            )
            for expected, result in zip(
                rotated_reference,
                (rotated.query_int8, rotated.query_scale, rotated.key_int8, rotated.key_scale),
            ):
                if expected.dtype == torch.int8:
                    if int((expected.to(torch.int16) - result.to(torch.int16)).abs().max()) > 2:
                        raise RuntimeError("rotated fused Q/K preprocessing exceeds 2 INT8 LSB")
                else:
                    _assert_close(
                        "rotated fused Q/K preprocessing scale",
                        result,
                        expected,
                        rtol=0.015,
                        atol=0.0011,
                    )


def _validate_varlen_batches(
    output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    *,
    is_causal: bool = False,
    name: str = "Sage",
    rtol: float = 0.08,
    atol: float = 0.06,
) -> None:
    for batch, (q_len, k_len) in enumerate(zip(q_lengths, k_lengths)):
        q_start = int(cu_q[batch])
        k_start = int(cu_k[batch])
        reference = torch.nn.functional.scaled_dot_product_attention(
            q[q_start:q_start + q_len].transpose(0, 1).unsqueeze(0).float(),
            k[k_start:k_start + k_len].transpose(0, 1).unsqueeze(0).float(),
            v[k_start:k_start + k_len].transpose(0, 1).unsqueeze(0).float(),
            enable_gqa=True,
            is_causal=is_causal,
        ).squeeze(0).transpose(0, 1)
        _assert_close(
            f"{name} varlen causal={is_causal} batch {batch}",
            output[q_start:q_start + q_len],
            reference,
            rtol=rtol,
            atol=atol,
        )


def validate_sage(device: torch.device) -> None:
    from comfyui_turing_utils_kernel.turing_sage import sageattn, sageattn_varlen
    from comfyui_turing_utils_kernel.turing_sage.core import sageattn_prequantized
    from comfyui_turing_utils_kernel.turing_sage.quant import per_warp_int8

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
        q_nhd = q.transpose(1, 2).contiguous()
        k_nhd = k.transpose(1, 2).contiguous()
        v_nhd = v.transpose(1, 2).contiguous()
        q_int8, q_scale, k_int8, k_scale = per_warp_int8(
            q_nhd, k_nhd, tensor_layout="NHD"
        )
        prequantized = sageattn_prequantized(
            q_int8, q_scale, k_int8, k_scale, v_nhd, tensor_layout="NHD"
        )
        if not torch.equal(prequantized, output_nhd):
            raise RuntimeError(f"prequantized Sage bridge mismatch for {dtype}")

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


def validate_w8a8(device: torch.device) -> None:
    """Validate the production dense W8A8 fixed and packed-varlen contracts."""
    from comfyui_turing_utils_kernel.turing_sage import w8a8attn, w8a8attn_varlen

    for dtype in (torch.float16, torch.bfloat16):
        for head_dim in (64, 96, 128):
            q = torch.randn((1, 4, 129, head_dim), device=device, dtype=dtype) * 0.4
            k = torch.randn((1, 2, 151, head_dim), device=device, dtype=dtype) * 0.4
            v = torch.randn_like(k)
            for is_causal in (False, True):
                output = w8a8attn(q, k, v, is_causal=is_causal)
                reference = torch.nn.functional.scaled_dot_product_attention(
                    q.float(),
                    k.float(),
                    v.float(),
                    enable_gqa=True,
                    is_causal=is_causal,
                )
                _assert_close(
                    f"W8A8 fixed {dtype} D={head_dim} causal={is_causal}",
                    output,
                    reference,
                    rtol=0.12,
                    atol=0.09,
                )

            output_nhd = w8a8attn(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                tensor_layout="NHD",
            )
            reference_nhd = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), enable_gqa=True
            ).transpose(1, 2)
            _assert_close(
                f"W8A8 NHD {dtype} D={head_dim}",
                output_nhd,
                reference_nhd,
                rtol=0.12,
                atol=0.09,
            )

            q_lengths, k_lengths = (65, 129), (73, 151)
            cu_q = torch.tensor(
                (0, q_lengths[0], sum(q_lengths)), dtype=torch.int32, device=device
            )
            cu_k = torch.tensor(
                (0, k_lengths[0], sum(k_lengths)), dtype=torch.int32, device=device
            )
            packed_q = torch.randn(
                (sum(q_lengths), 4, head_dim), device=device, dtype=dtype
            ) * 0.4
            packed_k = torch.randn(
                (sum(k_lengths), 2, head_dim), device=device, dtype=dtype
            ) * 0.4
            packed_v = torch.randn_like(packed_k)
            for is_causal in (False, True):
                packed_output = w8a8attn_varlen(
                    packed_q,
                    packed_k,
                    packed_v,
                    cu_q,
                    cu_k,
                    max(q_lengths),
                    max(k_lengths),
                    is_causal=is_causal,
                )
                _validate_varlen_batches(
                    packed_output,
                    packed_q,
                    packed_k,
                    packed_v,
                    cu_q,
                    cu_k,
                    q_lengths,
                    k_lengths,
                    is_causal=is_causal,
                    name=f"W8A8 D={head_dim}",
                    rtol=0.12,
                    atol=0.09,
                )


def _inverse_route_hadamard(value: torch.Tensor) -> torch.Tensor:
    head_dim = value.size(-1)
    result = value.float()
    span = 1
    while span < head_dim:
        shaped = result.reshape(*result.shape[:-1], -1, 2, span)
        left = shaped[..., 0, :]
        right = shaped[..., 1, :]
        result = torch.cat((left + right, left - right), dim=-1).reshape_as(result)
        span *= 2
    result = result * (head_dim ** -0.5)
    words = (0x1035997B, 0x8087F5EE, 0xEE2E4E1A, 0x71132418)
    signs = torch.tensor(
        [
            -1.0 if ((words[channel >> 5] >> (channel & 31)) & 1) == 0 else 1.0
            for channel in range(head_dim)
        ],
        device=result.device,
    )
    return result * signs


def _expected_int8_sol_route_count(
    q_int8: torch.Tensor,
    q_scale: torch.Tensor,
    k_int8: torch.Tensor,
    k_scale: torch.Tensor,
    threshold_sigma: float,
) -> int:
    """Reconstruct the fused kernel's INT8-consistent 1x64 routing policy."""
    _, query_heads, query_length, head_dim = q_int8.shape
    _, kv_heads, key_length, _ = k_int8.shape
    query_blocks = (query_length + 63) // 64
    key_blocks = (key_length + 63) // 64
    heads_per_kv = query_heads // kv_heads

    route_key_centroids = torch.empty(
        (kv_heads, key_blocks, head_dim), device=q_int8.device, dtype=torch.float32
    )
    score_key_centroids = torch.empty_like(route_key_centroids)
    for kv_head in range(kv_heads):
        for key_block in range(key_blocks):
            start = key_block * 64
            stop = min(start + 64, key_length)
            centroid = (
                k_int8[0, kv_head, start:stop].float().sum(dim=0)
                * k_scale[0, kv_head, key_block]
                / (stop - start)
            )
            # The proxy Tensor Core operand stays in the exact Hadamard score
            # domain; diagonal threshold statistics use its inverse-transformed
            # pre-Hadamard centroid. Both are rounded to the kernel's FP16
            # summary storage before use.
            score_key_centroids[kv_head, key_block] = centroid.half().float()
            route_key_centroids[kv_head, key_block] = (
                _inverse_route_hadamard(centroid).half().float()
            )

    key_means = route_key_centroids.mean(dim=1)
    key_variances = route_key_centroids.square().mean(dim=1) - key_means.square()
    key_variances.clamp_min_(0.0)

    selected = 0
    for query_head in range(query_heads):
        kv_head = query_head // heads_per_kv
        for query_block in range(query_blocks):
            start = query_block * 64
            stop = min(start + 64, query_length)
            rows = q_int8[0, query_head, start:stop].float()
            row_scales = q_scale[
                0,
                query_head,
                query_block * 4 : query_block * 4 + 4,
            ].repeat_interleave(16)[: stop - start]
            dequantized = rows * row_scales[:, None]
            query_mean = _inverse_route_hadamard(dequantized.mean(dim=0))
            threshold = (
                torch.dot(query_mean, key_means[kv_head])
                + threshold_sigma
                * torch.sqrt(
                    torch.dot(query_mean.square(), key_variances[kv_head]) + 1.0e-6
                )
            )
            # The correction/routing MMA consumes Q after the same explicit
            # FP16 conversion used in shared memory by the CUDA kernel.
            score_query_mean = dequantized.half().float().mean(dim=0)
            proxy_scores = score_key_centroids[kv_head] @ score_query_mean
            block_indices = torch.arange(key_blocks, device=q_int8.device)
            route = (block_indices - query_block).abs() <= 1
            route |= proxy_scores > threshold
            selected += int(route.sum().item())
    return selected


def validate_sparse(device: torch.device) -> None:
    from comfyui_turing_utils_kernel.turing_sage import (
        prequantize_sol_sageattn,
        run_attention_correctness_gate,
        sageattn,
        sol_sparse_sageattn,
        sol_sparse_sageattn_from_prequantized,
        w8a8attn,
    )
    from comfyui_turing_utils_kernel.turing_sage.quant import (
        per_warp_int8_hadamard,
    )

    for head_dim in (1, 32, 63, 64, 65, 96, 127, 128):
        q = torch.randn((1, 4, 129, head_dim), device=device, dtype=torch.bfloat16)
        k = torch.randn((1, 2, 151, head_dim), device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), enable_gqa=True
        )
        dense_state = prequantize_sol_sageattn(
            q, k, v, use_w8a8=True, force_dense=True
        )
        expected_kernel_dim = 64 if head_dim <= 64 else 128
        if (
            dense_state.query_int8.size(-1) != expected_kernel_dim
            or dense_state.key_int8.size(-1) != expected_kernel_dim
            or dense_state.value_int8.size(2) != expected_kernel_dim
            or dense_state.value_scale.size(-1) != expected_kernel_dim
        ):
            raise RuntimeError(
                f"W8A8 D={head_dim} used the wrong native kernel dimension"
            )
        dense_w8a8 = sol_sparse_sageattn_from_prequantized(dense_state)
        _assert_close(
            f"W8A8 padded D={head_dim}",
            dense_w8a8,
            reference,
            rtol=0.15,
            atol=0.12,
        )
        exact_sol = sol_sparse_sageattn(
            q, k, v, threshold_sigma=-1000.0, use_w8a8=True
        )
        _assert_close(
            f"Sol W8A8 padded D={head_dim}",
            exact_sol,
            dense_w8a8,
            rtol=0.02,
            atol=0.01,
        )

    for dtype in (torch.float16, torch.bfloat16):
        for query_length, key_length in ((129, 151), (151, 129)):
            q = torch.randn((1, 4, query_length, 128), device=device, dtype=dtype)
            k = torch.randn((1, 2, key_length, 128), device=device, dtype=dtype)
            v = torch.randn_like(k)
            output, selected, possible = sol_sparse_sageattn(
                q, k, v, threshold_sigma=-1000.0, return_stats=True
            )
            reference = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), enable_gqa=True
            )
            _assert_close(
                f"Sol exact route {dtype} Q={query_length} K={key_length}",
                output,
                reference,
                rtol=0.08,
                atol=0.06,
            )
            if int(selected.item()) != possible:
                raise RuntimeError("threshold=-1000 must select every sparse Q/K block")

            dense_w8a8 = w8a8attn(q, k, v)
            _assert_close(
                f"W8A8 dense {dtype} Q={query_length} K={key_length}",
                dense_w8a8,
                reference,
                rtol=0.12,
                atol=0.09,
            )
            sol_w8a8, selected_w8a8, possible_w8a8 = sol_sparse_sageattn(
                q,
                k,
                v,
                threshold_sigma=-1000.0,
                return_stats=True,
                use_w8a8=True,
            )
            if query_length == 129:
                for use_w8a8 in (False, True):
                    gate = run_attention_correctness_gate(
                        q, k, v, use_w8a8=use_w8a8
                    )
                    print(
                        f"attention correctness {gate.candidate}->{gate.reference} "
                        f"{dtype}: max_abs={gate.max_abs:.6g} "
                        f"relative_l2={gate.relative_l2:.6g} "
                        f"cosine={gate.cosine:.7f}"
                    )
            _assert_close(
                f"Sol W8A8 exact route {dtype} Q={query_length} K={key_length}",
                sol_w8a8,
                dense_w8a8,
                rtol=0.02,
                atol=0.01,
            )
            if int(selected_w8a8.item()) != possible_w8a8:
                raise RuntimeError(
                    "Sol W8A8 threshold=-1000 must select every sparse Q/K block"
                )

    # Verify the CUDA route against an independently reconstructed policy in
    # the exact quantized score domain. This catches accidental use of the
    # original BF16/FP16 tensors or a scale/partial-block indexing mismatch.
    generator = torch.Generator(device=device).manual_seed(20260812)
    q = torch.randn(
        (1, 4, 1025, 128), generator=generator, device=device, dtype=torch.bfloat16
    ) * 0.4
    k = torch.randn(
        (1, 2, 1025, 128), generator=generator, device=device, dtype=torch.bfloat16
    ) * 0.4
    v = torch.randn(
        (1, 2, 1025, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    q_int8, q_scale, k_int8, k_scale = per_warp_int8_hadamard(q, k)
    # H^-1(mean(HK)) must reconstruct the pre-Hadamard K centroid. This is the
    # defining equivalence behind routing before Hadamard while retaining the
    # post-Hadamard exact QK/PV representation.
    for key_block in range((k.size(2) + 63) // 64):
        start = key_block * 64
        stop = min(start + 64, k.size(2))
        rotated_centroid = (
            k_int8[0, 0, start:stop].float().mean(dim=0)
            * k_scale[0, 0, key_block]
        )
        reconstructed = _inverse_route_hadamard(rotated_centroid)
        original = k[0, 0, start:stop].float().mean(dim=0)
        relative_l2 = (
            torch.linalg.vector_norm(reconstructed - original)
            / torch.linalg.vector_norm(original).clamp_min(1.0e-12)
        )
        if float(relative_l2) > 0.02:
            raise RuntimeError(
                "pre-Hadamard centroid reconstruction regressed: "
                f"block={key_block} relative_l2={float(relative_l2):.6f}"
            )
    expected_selected = _expected_int8_sol_route_count(
        q_int8, q_scale, k_int8, k_scale, threshold_sigma=1.0
    )
    selected_profiles = []
    for residual_subblocks in (1, 2):
        _, selected, possible = sol_sparse_sageattn(
            q,
            k,
            v,
            threshold_sigma=1.0,
            residual_subblocks=residual_subblocks,
            return_stats=True,
        )
        actual_selected = int(selected.item())
        if actual_selected != expected_selected:
            raise RuntimeError(
                "INT8-consistent Sol route mismatch: "
                f"profile={residual_subblocks} selected={actual_selected}, "
                f"expected={expected_selected}"
            )
        selected_profiles.append(actual_selected)
    print(
        "Sol INT8 route oracle: "
        f"selected={expected_selected}/{possible}, profiles={selected_profiles}"
    )

    # Non-aligned protected spans round outward to blocks. Those Query blocks
    # use exact Sage; the same K/V blocks are exact sinks for every sparse Query.
    sequence = 321
    protected = ((0, 77), (130, 181))
    q = torch.randn((1, 4, sequence, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn((1, 2, sequence, 128), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    output = sol_sparse_sageattn(
        q,
        k,
        v,
        dense_query_ranges=protected,
        exact_kv_ranges=protected,
        threshold_sigma=-1000.0,
    )
    dense = sageattn(q, k, v)
    _assert_close("Sol protected modality ranges", output, dense, rtol=0.025, atol=0.02)

    sequence = 1025
    blocks = (sequence + 63) // 64

    def correlated(heads: int) -> torch.Tensor:
        centers = torch.randn((1, heads, blocks, 128), device=device)
        values = centers.repeat_interleave(64, dim=2)[:, :, :sequence]
        values = values + torch.randn_like(values) * 0.1
        values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1.0e-6)
        return values.to(torch.bfloat16)

    q = correlated(4)
    k = correlated(2)
    v_centers = torch.randn((1, 2, blocks, 128), device=device)
    v = v_centers.repeat_interleave(64, dim=2)[:, :, :sequence]
    v = (v + torch.randn_like(v) * 0.1).to(torch.bfloat16)
    output, selected, possible = sol_sparse_sageattn(
        q,
        k,
        v,
        dense_query_ranges=((0, 128),),
        exact_kv_ranges=((0, 128),),
        threshold_sigma=1.0,
        return_stats=True,
    )
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), enable_gqa=True
    )
    _assert_close("Sol correlated sequence", output, reference, rtol=0.02, atol=0.02)
    density = int(selected.item()) / possible
    if not 0.1 < density < 0.9:
        raise RuntimeError(f"Sol route density is implausible: {density:.3f}")

    # Exercise the default mixed skipped/exact path. W8A8 quantizes only exact
    # P@V blocks; skipped blocks intentionally retain the FP16 centroid
    # approximation. Their outputs therefore need not be bitwise close, but a
    # probability-domain mismatch would amplify this delta by roughly 2^8.
    residual_fp16 = sol_sparse_sageattn(
        q,
        k,
        v,
        exact_kv_ranges=((0, 128),),
        threshold_sigma=1.0,
        residual_subblocks=1,
    )
    residual_w8a8 = sol_sparse_sageattn(
        q,
        k,
        v,
        exact_kv_ranges=((0, 128),),
        threshold_sigma=1.0,
        residual_subblocks=1,
        use_w8a8=True,
    )
    residual_delta = (
        residual_fp16.float() - residual_w8a8.float()
    ).abs().mean().item()
    if residual_delta > 0.02:
        raise RuntimeError(
            "Sol W8A8 skipped/exact online-softmax domains diverged: "
            f"mean_abs={residual_delta:.6g}"
        )

    dense_w8a8 = w8a8attn(q, k, v)
    protected_dense_w8a8 = sol_sparse_sageattn(
        q,
        k,
        v,
        dense_query_ranges=((0, sequence),),
        use_w8a8=True,
    )
    if not torch.equal(dense_w8a8, protected_dense_w8a8):
        raise RuntimeError(
            "fully protected Sol W8A8 queries did not use the route-free dense kernel"
        )

    # 2x32 changes skipped-block reconstruction only; routing and exact blocks
    # are deliberately identical to the official-style 1x64 path.
    sequence = 2048
    blocks = sequence // 64
    half_centers = torch.randn((1, 2, blocks, 2, 128), device=device)
    k = half_centers.repeat_interleave(32, dim=3).reshape(1, 2, sequence, 128)
    k = (k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + 1.0e-6)).to(
        torch.bfloat16
    )
    q = k.repeat_interleave(2, dim=1)
    value_centers = torch.randn((1, 2, blocks, 2, 128), device=device)
    v = value_centers.repeat_interleave(32, dim=3).reshape(
        1, 2, sequence, 128
    ).to(torch.bfloat16)
    dense = sageattn(q, k, v)
    residual_64 = sol_sparse_sageattn(
        q, k, v, threshold_sigma=1000.0, residual_subblocks=1
    )
    residual_32 = sol_sparse_sageattn(
        q, k, v, threshold_sigma=1000.0, residual_subblocks=2
    )
    error_64 = (residual_64.float() - dense.float()).square().mean().item()
    error_32 = (residual_32.float() - dense.float()).square().mean().item()
    if error_32 > error_64 * 1.01 + 1.0e-7:
        raise RuntimeError(
            "2x32 skipped residual regressed against 1x64: "
            f"mse={error_32:.6g} vs {error_64:.6g}"
        )
    print(
        "Sol skipped residual quality: "
        f"2x32 mse={error_32:.6g}, 1x64 mse={error_64:.6g}"
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


def benchmark_sage(device: torch.device, iterations: int) -> None:
    from comfyui_turing_utils_kernel.turing_sage import sageattn

    for dtype in (torch.float16, torch.bfloat16):
        q = torch.randn((1, 8, 2048, 128), device=device, dtype=dtype)
        k = torch.randn((1, 4, 2048, 128), device=device, dtype=dtype)
        v = torch.randn_like(k)
        elapsed = _elapsed_ms(lambda: sageattn(q, k, v), iterations)
        print(f"sage HND N=2048 Hq=8 Hkv=4 D=128 {dtype}: {elapsed:.3f} ms")


def benchmark_sparse(device: torch.device, iterations: int) -> None:
    from comfyui_turing_utils_kernel.turing_sage import (
        prequantize_sol_sageattn,
        sageattn,
        sol_sparse_sageattn,
        sol_sparse_sageattn_from_prequantized,
        w8a8attn,
    )

    for sequence in (4096, 8192, 16384):
        q = torch.randn((1, 4, sequence, 128), device=device, dtype=torch.float16) * 0.2
        k = torch.randn_like(q) * 0.2
        v = torch.randn_like(q) * 0.2
        sparse = lambda: sol_sparse_sageattn(
            q, k, v, threshold_sigma=1.0, residual_subblocks=2
        )
        sparse_64 = lambda: sol_sparse_sageattn(
            q, k, v, threshold_sigma=1.0, residual_subblocks=1
        )
        sparse_w8a8 = lambda: sol_sparse_sageattn(
            q,
            k,
            v,
            threshold_sigma=1.0,
            residual_subblocks=1,
            use_w8a8=True,
        )
        dense = lambda: sageattn(q, k, v)
        dense_w8a8 = lambda: w8a8attn(q, k, v)
        sparse_ms = _elapsed_ms(sparse, iterations)
        sparse_64_ms = _elapsed_ms(sparse_64, iterations)
        sparse_w8a8_ms = _elapsed_ms(sparse_w8a8, iterations)
        dense_ms = _elapsed_ms(dense, iterations)
        dense_w8a8_ms = _elapsed_ms(dense_w8a8, iterations)
        _, selected_tensor, possible = sol_sparse_sageattn(
            q, k, v, threshold_sigma=1.0, return_stats=True
        )
        density = int(selected_tensor.item()) / possible
        print(
            f"sparse HND N={sequence} H=4 D=128 threshold=1.0 density={density:.3f}: "
            f"2x32 {sparse_ms:.3f} ms, 1x64 {sparse_64_ms:.3f} ms, "
            f"W8A8 1x64 {sparse_w8a8_ms:.3f} ms, "
            f"sage {dense_ms:.3f} ms, dense W8A8 {dense_w8a8_ms:.3f} ms, "
            f"{dense_ms / sparse_w8a8_ms:.3f}x sparse-W8A8 speedup"
        )

    # Compare the native D64 CTA with the previous behavior using the exact
    # same inputs, zero-padded to D128 while retaining the D64 softmax scale.
    # This is directional A40 validation, not a substitute for sm75 profiling.
    q64 = torch.randn((1, 8, 4096, 64), device=device, dtype=torch.bfloat16)
    k64 = torch.randn((1, 4, 4096, 64), device=device, dtype=torch.bfloat16)
    v64 = torch.randn_like(k64)
    for label, q, k, v in (
        ("native D64", q64, k64, v64),
        (
            "D64 padded to D128",
            torch.nn.functional.pad(q64, (0, 64)),
            torch.nn.functional.pad(k64, (0, 64)),
            torch.nn.functional.pad(v64, (0, 64)),
        ),
    ):
        dense_state = prequantize_sol_sageattn(
            q, k, v, use_w8a8=True, force_dense=True, sm_scale=64**-0.5
        )
        sparse_state = prequantize_sol_sageattn(
            q, k, v, use_w8a8=True, threshold_sigma=1.0, sm_scale=64**-0.5
        )
        dense_ms = _elapsed_ms(
            lambda: sol_sparse_sageattn_from_prequantized(dense_state), iterations
        )
        sparse_ms = _elapsed_ms(
            lambda: sol_sparse_sageattn_from_prequantized(sparse_state), iterations
        )
        print(
            f"native-head check {label} "
            f"kernel D={dense_state.query_int8.size(-1)}: "
            f"dense W8A8 core {dense_ms:.3f} ms, Sol-W8A8 core {sparse_ms:.3f} ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--experimental-sparse", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("validation requires a CUDA device")
    capability = torch.cuda.get_device_capability(device)
    if capability < (7, 5):
        raise RuntimeError(f"validation requires sm75 or newer, got sm{capability[0]}{capability[1]}")
    print(f"kernel={Path(comfyui_turing_utils_kernel.__file__).resolve()} version={comfyui_turing_utils_kernel.__version__}")
    print(f"device={torch.cuda.get_device_name(device)} capability=sm{capability[0]}{capability[1]}")
    torch.manual_seed(20260806)
    with torch.inference_mode(), torch.cuda.device(device):
        validate_convrot(device)
        validate_w4a8(device)
        validate_segmented_norm(device)
        validate_qk_preprocessing(device)
        validate_sage(device)
        validate_w8a8(device)
        if args.experimental_sparse:
            validate_sparse(device)
        torch.cuda.synchronize(device)
        print("numerical validation passed")
        if args.benchmark:
            benchmark_sage(device, args.iterations)
            if args.experimental_sparse:
                benchmark_sparse(device, args.iterations)


if __name__ == "__main__":
    main()
