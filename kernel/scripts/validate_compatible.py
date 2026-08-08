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


def validate_sparse(device: torch.device) -> None:
    from comfyui_turing_utils_kernel.turing_sage import (
        sageattn,
        sol_sparse_route_selected,
        sol_sparse_sageattn,
    )

    for dtype in (torch.float16, torch.bfloat16):
        for query_length, key_length in ((129, 151), (151, 129)):
            q = torch.randn((1, 4, query_length, 128), device=device, dtype=dtype)
            k = torch.randn((1, 2, key_length, 128), device=device, dtype=dtype)
            v = torch.randn_like(k)
            output, route = sol_sparse_sageattn(
                q, k, v, threshold_sigma=-1000.0, return_route=True
            )
            reference = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), enable_gqa=True
            )
            _assert_close(
                f"experimental sparse exact route {dtype} Q={query_length} K={key_length}",
                output,
                reference,
                rtol=0.08,
                atol=0.06,
            )
            expected_query_blocks = (query_length + 63) // 64
            if (
                route.dtype != torch.int32
                or route.shape[:3] != (1, 4, expected_query_blocks)
            ):
                raise RuntimeError("experimental sparse route shape is incompatible")

    # A non-block-aligned semantic prefix is evaluated by stable Sage while
    # only target-video Query blocks enter sparse routing. K quantization is
    # shared by both paths, including GQA head mapping.
    sequence = 321
    prefix = 77
    q = torch.randn((1, 4, sequence, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn((1, 2, sequence, 128), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    output, route = sol_sparse_sageattn(
        q,
        k,
        v,
        prefix_tokens=prefix,
        threshold_sigma=-1000.0,
        return_route=True,
    )
    dense = sageattn(q, k, v)
    _assert_close(
        "experimental sparse split prefix full route",
        output,
        dense,
        rtol=0.02,
        atol=0.01,
    )
    if route.shape[:3] != (1, 4, (sequence - prefix + 63) // 64):
        raise RuntimeError("sparse route must contain target-video Query blocks only")

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
    output, route = sol_sparse_sageattn(
        q, k, v, prefix_tokens=128, threshold_sigma=1.0, return_route=True
    )
    sage_reference = sageattn(q, k, v)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), enable_gqa=True
    )
    _assert_close(
        "experimental sparse correlated sequence",
        output,
        reference,
        rtol=0.02,
        atol=0.02,
    )
    _assert_close(
        "experimental sparse correlated sequence against stable Sage",
        output,
        sage_reference,
        rtol=0.02,
        atol=0.02,
    )
    selected = sum(
        (int(word) & 0xFFFF).bit_count() for word in route.cpu().reshape(-1)
    )
    selected_cuda = sol_sparse_route_selected(route)
    if selected_cuda != selected:
        raise RuntimeError(
            f"experimental sparse route popcount mismatch: {selected_cuda} != {selected}"
        )
    possible = route.shape[0] * route.shape[1] * route.shape[2] * blocks
    density = selected / possible
    if not 0.1 < density < 0.9:
        raise RuntimeError(f"experimental sparse route density is implausible: {density:.3f}")

    # The budget is applied independently to every 16-key routing tile. With
    # no semantic or temporal forcing and 16 exact key blocks, 0.25 selects
    # exactly four keys for every Query/head row (the overlapping self block is
    # included in that budget rather than added on top).
    budget_sequence = 1024
    q = torch.randn((1, 4, budget_sequence, 128), device=device, dtype=torch.float16)
    k = torch.randn((1, 2, budget_sequence, 128), device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    _, route = sol_sparse_sageattn(
        q,
        k,
        v,
        threshold_sigma=1000.0,
        local_block_radius=0,
        minimum_route_density=0.25,
        maximum_route_density=0.25,
        return_route=True,
    )
    expected = 4 * 16 * 4
    selected = sol_sparse_route_selected(route)
    if selected != expected:
        raise RuntimeError(
            f"experimental sparse route budget selected {selected}, expected {expected}"
        )

    # Two 32-token centroids must preserve a deliberately bimodal skipped
    # block at least as well as one 64-token centroid. Exact/local routing is
    # unchanged, so this isolates only the skipped-residual approximation.
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
        q,
        k,
        v,
        threshold_sigma=1000.0,
        local_block_radius=0,
        residual_subblocks=1,
    )
    residual_32 = sol_sparse_sageattn(
        q,
        k,
        v,
        threshold_sigma=1000.0,
        local_block_radius=0,
        residual_subblocks=2,
    )
    error_64 = (residual_64.float() - dense.float()).square().mean().item()
    error_32 = (residual_32.float() - dense.float()).square().mean().item()
    if error_32 > error_64 * 1.01 + 1.0e-7:
        raise RuntimeError(
            "2x32 skipped residual regressed against 1x64: "
            f"mse={error_32:.6g} vs {error_64:.6g}"
        )
    print(
        "sparse skipped residual quality: "
        f"2x32 mse={error_32:.6g}, 1x64 mse={error_64:.6g}"
    )


def validate_frame_sparse(device: torch.device) -> None:
    from comfyui_turing_utils_kernel.turing_sage import (
        frame_sparse_sageattn,
        sageattn,
    )
    from comfyui_turing_utils_kernel.turing_sage.core import (
        _frame_sparse_schedule_cpu,
    )

    # A full schedule must reduce to the production dense Sage path for both
    # storage dtypes, including a non-aligned semantic Query prefix and GQA.
    for dtype in (torch.float16, torch.bfloat16):
        prefix = 77
        topology_tokens = 4 * 64
        sequence = prefix + topology_tokens
        q = torch.randn((1, 4, sequence, 128), device=device, dtype=dtype) * 0.3
        k = torch.randn((1, 2, sequence, 128), device=device, dtype=dtype) * 0.3
        v = torch.randn_like(k) * 0.3
        output, density = frame_sparse_sageattn(
            q,
            k,
            v,
            prefix_tokens=prefix,
            topology_start_tokens=prefix,
            topology_tokens=topology_tokens,
            tokens_per_frame=64,
            temporal_window_frames=4,
            global_anchor_stride=1,
            sink_frames=1,
            return_schedule_density=True,
        )
        dense = sageattn(q, k, v)
        if density != 1.0:
            raise RuntimeError(f"frame-sparse dense route has density {density}, expected 1")
        _assert_close(
            f"frame-sparse full route {dtype}",
            output,
            dense,
            rtol=0.02,
            atol=0.01,
        )

    # Compare a genuinely sparse, frame-misaligned route with an explicit
    # FP32 block mask. Prefix Query rows remain globally dense; video Query
    # rows see exactly the CSR-selected 64-token blocks.
    prefix = 77
    tokens_per_frame = 65
    topology_tokens = 7 * tokens_per_frame
    sequence = prefix + topology_tokens
    schedule_kwargs = {
        "prefix_tokens": prefix,
        "topology_start_tokens": prefix,
        "topology_tokens": topology_tokens,
        "tokens_per_frame": tokens_per_frame,
        "temporal_window_frames": 1,
        "global_anchor_stride": 4,
        "global_anchor_offset": 2,
        "sink_frames": 1,
    }
    q = torch.randn((1, 4, sequence, 128), device=device, dtype=torch.bfloat16) * 0.3
    k = torch.randn((1, 2, sequence, 128), device=device, dtype=torch.bfloat16) * 0.3
    v = torch.randn_like(k) * 0.3
    output, density = frame_sparse_sageattn(
        q,
        k,
        v,
        return_schedule_density=True,
        **schedule_kwargs,
    )
    row_offsets, key_blocks, expected_density = _frame_sparse_schedule_cpu(
        key_length=sequence,
        **schedule_kwargs,
    )
    mask = torch.zeros(
        (topology_tokens, sequence), device=device, dtype=torch.bool
    )
    for query_block, (start, end) in enumerate(
        zip(row_offsets, row_offsets[1:])
    ):
        query_start = query_block * 64
        query_end = min(query_start + 64, topology_tokens)
        for key_block in key_blocks[start:end]:
            key_start = key_block * 64
            key_end = min(key_start + 64, sequence)
            mask[query_start:query_end, key_start:key_end] = True
    reference_prefix = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, :prefix].float(),
        k.float(),
        v.float(),
        enable_gqa=True,
    )
    reference_video = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, prefix:].float(),
        k.float(),
        v.float(),
        attn_mask=mask,
        enable_gqa=True,
    )
    reference = torch.cat((reference_prefix, reference_video), dim=2)
    if density != expected_density or not 0.0 < density < 1.0:
        raise RuntimeError(
            f"frame-sparse schedule density is incompatible: {density}"
        )
    _assert_close(
        "frame-sparse explicit CSR mask",
        output,
        reference,
        rtol=0.08,
        atol=0.06,
    )

    # Exercise the static radial policy separately.  The exact 2D topology is
    # deliberately non-square so row-major spatial addressing and partial
    # 8x8 tiles are both covered.
    prefix = 64
    spatial_height, spatial_width = 12, 16
    tokens_per_frame = spatial_height * spatial_width
    topology_tokens = 6 * tokens_per_frame
    sequence = prefix + topology_tokens
    radial_kwargs = {
        "prefix_tokens": prefix,
        "topology_start_tokens": prefix,
        "topology_tokens": topology_tokens,
        "tokens_per_frame": tokens_per_frame,
        "temporal_window_frames": 0,
        "global_anchor_stride": 0,
        "global_anchor_offset": 1,
        "sink_frames": 1,
        "sparse_pattern": "radial",
        "spatial_tokens_height": spatial_height,
        "spatial_tokens_width": spatial_width,
        "radial_spatial_radius": 0,
        "radial_max_temporal_stride": 8,
    }
    q = torch.randn((1, 4, sequence, 128), device=device, dtype=torch.bfloat16) * 0.3
    k = torch.randn((1, 2, sequence, 128), device=device, dtype=torch.bfloat16) * 0.3
    v = torch.randn_like(k) * 0.3
    output, density = frame_sparse_sageattn(
        q, k, v, return_schedule_density=True, **radial_kwargs
    )
    row_offsets, key_blocks, expected_density = _frame_sparse_schedule_cpu(
        key_length=sequence, **radial_kwargs
    )
    mask = torch.zeros(
        (topology_tokens, sequence), device=device, dtype=torch.bool
    )
    for query_block, (start, end) in enumerate(zip(row_offsets, row_offsets[1:])):
        query_start = query_block * 64
        query_end = min(query_start + 64, topology_tokens)
        for key_block in key_blocks[start:end]:
            key_start = key_block * 64
            key_end = min(key_start + 64, sequence)
            mask[query_start:query_end, key_start:key_end] = True
    reference_prefix = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, :prefix].float(), k.float(), v.float(), enable_gqa=True
    )
    reference_video = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, prefix:].float(),
        k.float(),
        v.float(),
        attn_mask=mask,
        enable_gqa=True,
    )
    reference = torch.cat((reference_prefix, reference_video), dim=2)
    if density != expected_density or not 0.0 < density < 1.0:
        raise RuntimeError(f"radial frame-sparse density is incompatible: {density}")
    _assert_close(
        "radial frame-sparse explicit CSR mask",
        output,
        reference,
        rtol=0.08,
        atol=0.06,
    )

    # A deterministic, temporally correlated proxy reports approximation
    # error against dense Sage.  It is a screening metric rather than a visual
    # quality gate; real H3 clips are still required before changing defaults.
    prefix = 64
    spatial_height, spatial_width, frames = 16, 16, 16
    tokens_per_frame = spatial_height * spatial_width
    topology_tokens = frames * tokens_per_frame
    sequence = prefix + topology_tokens
    generator = torch.Generator(device=device).manual_seed(20260809)
    prefix_q = torch.randn(
        (1, 2, prefix, 128), generator=generator, device=device, dtype=torch.float32
    )
    spatial_q = torch.randn(
        (1, 2, 1, tokens_per_frame, 128),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    temporal_q = torch.randn(
        (1, 2, frames, 1, 128),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ) * 0.08
    video_q = (spatial_q + temporal_q).reshape(1, 2, topology_tokens, 128)
    q = torch.cat((prefix_q, video_q), dim=2).to(torch.bfloat16)
    k = (q.float() * 0.91 + torch.randn(
        q.shape, generator=generator, device=device, dtype=torch.float32
    ) * 0.04).to(torch.bfloat16)
    v = (q.float() * 0.73 + torch.randn(
        q.shape, generator=generator, device=device, dtype=torch.float32
    ) * 0.06).to(torch.bfloat16)
    dense = sageattn(q, k, v)
    proxy_policies = {
        "conservative": dict(
            sparse_pattern="frame_window",
            temporal_window_frames=3,
            global_anchor_stride=8,
            sink_frames=2,
            radial_spatial_radius=1,
            radial_max_temporal_stride=8,
        ),
        "balanced": dict(
            sparse_pattern="radial",
            temporal_window_frames=2,
            global_anchor_stride=0,
            sink_frames=1,
            radial_spatial_radius=0,
            radial_max_temporal_stride=16,
        ),
        "fast": dict(
            sparse_pattern="radial",
            temporal_window_frames=1,
            global_anchor_stride=0,
            sink_frames=1,
            radial_spatial_radius=0,
            radial_max_temporal_stride=32,
        ),
    }
    for name, policy in proxy_policies.items():
        sparse, density = frame_sparse_sageattn(
            q,
            k,
            v,
            prefix_tokens=prefix,
            topology_start_tokens=prefix,
            topology_tokens=topology_tokens,
            tokens_per_frame=tokens_per_frame,
            spatial_tokens_height=spatial_height,
            spatial_tokens_width=spatial_width,
            return_schedule_density=True,
            **policy,
        )
        error = sparse.float() - dense.float()
        nrmse = (
            error.square().mean().sqrt()
            / dense.float().square().mean().sqrt().clamp_min(1.0e-8)
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            sparse.float().flatten(), dense.float().flatten(), dim=0
        ).item()
        print(
            f"frame sparse proxy {name}: density={density:.4f} "
            f"nrmse={nrmse:.6f} cosine={cosine:.6f}"
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
        sageattn,
        sol_sparse_route_selected,
        sol_sparse_sageattn,
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
        dense = lambda: sageattn(q, k, v)
        sparse_ms = _elapsed_ms(sparse, iterations)
        sparse_64_ms = _elapsed_ms(sparse_64, iterations)
        dense_ms = _elapsed_ms(dense, iterations)
        _, route = sol_sparse_sageattn(
            q, k, v, threshold_sigma=1.0, return_route=True
        )
        blocks = (sequence + 63) // 64
        selected = sol_sparse_route_selected(route)
        density = selected / (4 * blocks * blocks)
        print(
            f"sparse HND N={sequence} H=4 D=128 threshold=1.0 density={density:.3f}: "
            f"2x32 {sparse_ms:.3f} ms, 1x64 {sparse_64_ms:.3f} ms, "
            f"sage {dense_ms:.3f} ms, {dense_ms / sparse_ms:.3f}x"
        )


def benchmark_frame_sparse(device: torch.device, iterations: int) -> None:
    from comfyui_turing_utils_kernel.turing_sage import (
        frame_sparse_sageattn,
        sageattn,
    )

    cases = (
        ("480p-like", 46773, 3438, 43335, 405, 15, 27),
        ("720p-like", 100483, 2043, 98440, 920, 23, 40),
    )
    results: dict[tuple[str, str], tuple[float, float, float]] = {}
    for name, sequence, prefix, topology_tokens, tokens_per_frame, height, width in cases:
        q = torch.randn(
            (1, 8, sequence, 128), device=device, dtype=torch.bfloat16
        ) * 0.2
        k = torch.randn_like(q) * 0.2
        v = torch.randn_like(q) * 0.2

        dense_ms = _elapsed_ms(lambda: sageattn(q, k, v), iterations)
        for pattern in ("frame_window", "radial"):
            kwargs = dict(
                prefix_tokens=prefix,
                topology_start_tokens=prefix,
                topology_tokens=topology_tokens,
                tokens_per_frame=tokens_per_frame,
                temporal_window_frames=2,
                global_anchor_stride=12 if pattern == "frame_window" else 0,
                sink_frames=1,
                sparse_pattern=pattern,
                spatial_tokens_height=height,
                spatial_tokens_width=width,
                radial_spatial_radius=0,
                radial_max_temporal_stride=16,
            )

            def sparse():
                return frame_sparse_sageattn(q, k, v, **kwargs)

            _, density = frame_sparse_sageattn(
                q, k, v, return_schedule_density=True, **kwargs
            )
            sparse_ms = _elapsed_ms(sparse, iterations)
            results[(name, pattern)] = (sparse_ms, dense_ms, density)
            print(
                f"frame sparse {name} {pattern} H=8 D=128 density={density:.4f}: "
                f"{sparse_ms:.3f} ms, sage {dense_ms:.3f} ms, "
                f"{dense_ms / sparse_ms:.3f}x"
            )
        del q, k, v
        torch.cuda.empty_cache()

    dense_480 = results[("480p-like", "frame_window")][1]
    for pattern in ("frame_window", "radial"):
        sparse_720 = results[("720p-like", pattern)][0]
        print(
            f"frame sparse 720p-like {pattern} / dense Sage 480p-like: "
            f"{sparse_720 / dense_480:.3f}x attention time"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--experimental-sparse", action="store_true")
    parser.add_argument("--experimental-frame-sparse", action="store_true")
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
        validate_sage(device)
        if args.experimental_sparse:
            validate_sparse(device)
        if args.experimental_frame_sparse:
            validate_frame_sparse(device)
        torch.cuda.synchronize(device)
        print("numerical validation passed")
        if args.benchmark:
            benchmark_sage(device, args.iterations)
            if args.experimental_sparse:
                benchmark_sparse(device, args.iterations)
            if args.experimental_frame_sparse:
                benchmark_frame_sparse(device, args.iterations)


if __name__ == "__main__":
    main()
