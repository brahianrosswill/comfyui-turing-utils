import warnings
from functools import wraps
from typing import Any, Optional

import torch

from .. import _sage_fused_sm75 as _fused
from . import sm75_compile
from .quant import (
    per_block_int8,
    per_thread_int4_fused,
    sage2_score_correction,
    per_warp_int8,
    per_warp_int8_varlen,
    sub_mean,
)


_SAGE2_CORRECTION_WORKSPACE_BYTES = 128 * 1024 * 1024


def _on_input_device(function):
    @wraps(function)
    def wrapped(q: torch.Tensor, *args, **kwargs):
        with torch.cuda.device(q.device):
            return function(q, *args, **kwargs)

    return wrapped


def _validate_fixed_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str) -> None:
    if tensor_layout not in {"HND", "NHD"}:
        raise ValueError(f"Unsupported tensor_layout: {tensor_layout}")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("fixed-length Q/K/V must be four-dimensional")
    head_dim = q.size(-1)
    if k.size(-1) != head_dim or v.size(-1) != head_dim:
        raise ValueError("Q/K/V head dimensions must match")
    head_axis = 1 if tensor_layout == "HND" else 2
    seq_axis = 2 if tensor_layout == "HND" else 1
    if q.size(0) != k.size(0) or q.size(0) != v.size(0):
        raise ValueError("Q/K/V batch sizes must match")
    if k.size(head_axis) != v.size(head_axis) or k.size(seq_axis) != v.size(seq_axis):
        raise ValueError("K/V head counts and sequence lengths must match")
    if k.size(head_axis) == 0 or q.size(head_axis) % k.size(head_axis) != 0:
        raise ValueError("the Q head count must be divisible by the KV head count")
    if q.size(seq_axis) == 0 or k.size(seq_axis) == 0:
        raise ValueError("empty Q/K sequences are not supported")


def _short_sequence_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    sm_scale: Optional[float],
    return_lse: bool,
):
    """Use a bounded exact path below the SM75 kernel's 64-token CTA.

    The vendored CTA kernel is optimized for large video sequences. Its
    predicated single-tile path is not deterministic under CUDA memcheck when
    either logical sequence is shorter than one CTA. The exact FP32 path is
    bounded to fewer than 4096 scores per head and cannot cause the large-N
    SDPA allocation that motivated the bundled implementation.
    """
    sequence_axis = 2 if tensor_layout == "HND" else 1
    if q.size(sequence_axis) >= 64 and k.size(sequence_axis) >= 64:
        return None

    q_hnd = q if tensor_layout == "HND" else q.transpose(1, 2)
    k_hnd = k if tensor_layout == "HND" else k.transpose(1, 2)
    v_hnd = v if tensor_layout == "HND" else v.transpose(1, 2)
    q_heads = q_hnd.size(1)
    kv_heads = k_hnd.size(1)
    scale = float(sm_scale) if sm_scale is not None else q.size(-1) ** -0.5
    output_hnd = torch.nn.functional.scaled_dot_product_attention(
        q_hnd.float(),
        k_hnd.float(),
        v_hnd.float(),
        is_causal=is_causal,
        enable_gqa=q_heads != kv_heads,
        scale=scale,
    ).to(q.dtype)
    output = output_hnd if tensor_layout == "HND" else output_hnd.transpose(1, 2)
    if not return_lse:
        return output, None

    key_for_q = torch.repeat_interleave(k_hnd.float(), q_heads // kv_heads, dim=1)
    scores = torch.matmul(q_hnd.float(), key_for_q.transpose(-2, -1)) * scale
    if is_causal:
        causal_mask = torch.ones(
            (q_hnd.size(2), k_hnd.size(2)), dtype=torch.bool, device=q.device
        ).tril()
        scores.masked_fill_(~causal_mask, float("-inf"))
    return output, torch.logsumexp(scores, dim=-1)


@_on_input_device
def sageattn_hybrid(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    smooth_k: bool = False,
    pv_accum_dtype: str = "fp32",
    **kwargs: Any,
):
    return sageattn_qk_int8_pv_fp16_cuda(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran="per_warp",
        sm_scale=sm_scale,
        return_lse=return_lse,
        smooth_k=smooth_k,
        pv_accum_dtype=pv_accum_dtype,
    )


@_on_input_device
def sageattn_sage1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    smooth_k: bool = True,
    smooth_v: bool = False,
    **kwargs: Any,
):
    """Turing SageAttention1 with stable mixed-precision PV accumulation.

    Each 64-token PV tile uses FP16 Tensor Core MMA and is immediately folded
    into an FP32 running accumulator.  Keeping the full sequence accumulator
    in FP16 is unsafe for long video sequences whose V tensor has a DC bias.
    """
    return sageattn_qk_int8_pv_fp16_cuda(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran="per_block",
        sm_scale=sm_scale,
        return_lse=return_lse,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        pv_accum_dtype="fp16+fp32",
    )


@_on_input_device
def sageattn_sage2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    smooth_q: bool = True,
    smooth_k: bool = True,
    smooth_v: bool = False,
    **kwargs: Any,
):
    """Turing Sage2 adaptation: packed INT4 Q/K and stable mixed-precision PV.

    The official Sage2 FP8 PV path is unavailable on SM75. This adaptation
    keeps the official per-thread INT4 and Q/K smoothing strategy, while the
    probability/value MMA uses native Turing FP16 within each 64-token tile,
    then folds the tile result into an FP32 running accumulator.
    """
    dtype = q.dtype
    if not q.is_cuda:
        raise ValueError("Input tensors must be on CUDA")
    if dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Sage2 Q/K/V must be float16 or bfloat16")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("Q/K/V must have matching dtypes")
    _validate_fixed_qkv(q, k, v, tensor_layout)

    short_result = _short_sequence_attention(
        q, k, v, tensor_layout, is_causal, sm_scale, return_lse
    )
    if short_result is not None:
        return short_result if return_lse else short_result[0]

    tensor_layout_id = 0 if tensor_layout == "NHD" else 1
    head_dim_og = q.size(-1)
    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif 64 < head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    q_int4, q_scale, k_int4, k_scale, q_mean, k_mean = per_thread_int4_fused(
        q,
        k,
        tensor_layout=tensor_layout,
        smooth_q=smooth_q,
        smooth_k=smooth_k,
    )
    if smooth_v:
        value_for_kernel, value_mean = sub_mean(v, tensor_layout=tensor_layout)
    else:
        value_for_kernel, value_mean = v, None
    output = torch.empty_like(q)
    if smooth_q:
        batch = q.size(0)
        q_heads = q_mean.size(1)
        q_blocks = q_mean.size(2)
        kv_tokens = k.size(2 if tensor_layout == "HND" else 1)
        bytes_per_q_block = batch * q_heads * kv_tokens * 4
        correction_blocks = max(
            1, _SAGE2_CORRECTION_WORKSPACE_BYTES // bytes_per_q_block
        )
        # The correction GEMM computes 16 Q-block means per WMMA tile. Keep
        # full tiles when the bounded workspace permits it.
        if correction_blocks >= 16:
            correction_blocks = (correction_blocks // 16) * 16
        correction_blocks = min(q_blocks, correction_blocks)
        if return_lse:
            q_tokens = q.size(2 if tensor_layout == "HND" else 1)
            lse = torch.empty(
                (batch, q_heads, q_tokens), device=q.device, dtype=torch.float32
            )
        else:
            lse = None

        for q_block_start in range(0, q_blocks, correction_blocks):
            q_block_count = min(correction_blocks, q_blocks - q_block_start)
            q_mean_chunk = q_mean[
                :, :, q_block_start : q_block_start + q_block_count
            ].contiguous()
            score_correction = sage2_score_correction(
                q_mean_chunk,
                k,
                k_mean,
                tensor_layout=tensor_layout,
                smooth_k=smooth_k,
            )
            chunk_lse = sm75_compile.qk_int4_sv_f16_accum_f16_f32_precomputed_attn(
                q_int4,
                k_int4,
                value_for_kernel,
                output,
                q_scale,
                k_scale,
                score_correction,
                tensor_layout_id,
                int(is_causal),
                sm_scale,
                int(return_lse),
                q_block_start,
                q_block_count,
            )
            if return_lse:
                token_start = q_block_start * 64
                token_end = min((q_block_start + q_block_count) * 64, q_tokens)
                lse[:, :, token_start:token_end].copy_(
                    chunk_lse[:, :, token_start:token_end]
                )
    else:
        lse = sm75_compile.qk_int4_sv_f16_accum_f16_f32_attn(
            q_int4,
            k_int4,
            value_for_kernel,
            output,
            q_scale,
            k_scale,
            k,
            q_mean,
            k_mean,
            tensor_layout_id,
            int(is_causal),
            sm_scale,
            int(return_lse),
            0,
            int(smooth_k),
        )
    if value_mean is not None:
        head_axis = 1 if tensor_layout == "HND" else 2
        sequence_axis = 2 if tensor_layout == "HND" else 1
        q_heads = output.size(head_axis)
        expanded_mean = torch.repeat_interleave(
            value_mean, q_heads // value_mean.size(1), dim=1
        ).unsqueeze(sequence_axis)
        output.add_(expanded_mean)
    output = output[..., :head_dim_og]

    if return_lse:
        lse = lse / 1.44269504
        if smooth_k:
            q_hnd = q if tensor_layout == "HND" else q.transpose(1, 2)
            q_heads = q_hnd.size(1)
            kv_heads = k_mean.size(1)
            mean_for_q = torch.repeat_interleave(
                k_mean[:, :, 0], q_heads // kv_heads, dim=1
            )
            lse = lse + torch.sum(q_hnd.float() * mean_for_q.unsqueeze(2), dim=-1) * sm_scale
        return output, lse
    return output


@_on_input_device
def _sageattn_varlen_hybrid(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    head_dim_og = q.size(-1)
    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."
    assert cu_seqlens_q.is_contiguous() and cu_seqlens_k.is_contiguous(), "cu_seqlens_q and cu_seqlens_k must be contiguous."

    if smooth_k:
        km = k.mean(dim=0, keepdim=True)
        k = k - km

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    if max_seqlen_q >= 512:
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_varlen(
            q,
            k,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            BLKQ=64,
            WARPQ=16,
            BLKK=64,
        )
        o = torch.empty_like(q)
        sm75_compile.qk_int8_sv_f16_varlen_accum_f32_attn(
            q_int8,
            k_int8,
            v.contiguous(),
            o,
            q_scale,
            k_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            int(is_causal),
            sm_scale,
        )
        return o[..., :head_dim_og]

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)
    _fused.varlen_attention_fwd_cuda(q, k, v, cu_seqlens_q, cu_seqlens_k, o, max_seqlen_q, sm_scale, int(is_causal))
    return o[..., :head_dim_og]


def sageattn(*args, **kwargs):
    """Default bundled entry point: the stable direct-FP32 ``sage_`` path."""
    return sageattn_hybrid(*args, **kwargs)


sage_ = sageattn_hybrid


@_on_input_device
def sageattn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = False,
    smooth_q: bool = True,
    variant: str = "sage_",
    **kwargs: Any,
) -> torch.Tensor:
    """Variable-length facade with sequence-local smoothing statistics."""
    if variant == "sage_":
        return _sageattn_varlen_hybrid(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            is_causal=is_causal,
            sm_scale=sm_scale,
            smooth_k=smooth_k,
        )
    if variant not in {"sage1", "sage2"}:
        raise ValueError(f"Unknown bundled Turing Sage variant: {variant}")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("varlen Q/K/V must have shape [total_tokens, heads, head_dim]")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("Q/K/V must have matching dtypes")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        raise ValueError("cu_seqlens_q/cu_seqlens_k must be one-dimensional")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("Q and KV cumulative length arrays must have the same batch size")

    q_offsets = cu_seqlens_q.detach().cpu().tolist()
    k_offsets = cu_seqlens_k.detach().cpu().tolist()
    output = torch.empty_like(q)
    implementation = sageattn_sage1 if variant == "sage1" else sageattn_sage2
    for batch in range(len(q_offsets) - 1):
        q_start, q_end = int(q_offsets[batch]), int(q_offsets[batch + 1])
        k_start, k_end = int(k_offsets[batch]), int(k_offsets[batch + 1])
        q_len, k_len = q_end - q_start, k_end - k_start
        if q_len <= 0 or k_len <= 0:
            raise ValueError("empty sequences are not supported")
        if q_len > max_seqlen_q or k_len > max_seqlen_k:
            raise ValueError("actual sequence length exceeds declared max_seqlen")
        q_fixed = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_fixed = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        v_fixed = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
        variant_kwargs = {"smooth_k": smooth_k}
        if variant == "sage2":
            variant_kwargs["smooth_q"] = smooth_q
        batch_output = implementation(
            q_fixed,
            k_fixed,
            v_fixed,
            tensor_layout="HND",
            is_causal=is_causal,
            sm_scale=sm_scale,
            **variant_kwargs,
        )
        output[q_start:q_end].copy_(batch_output.squeeze(0).transpose(0, 1))
    return output


def sageattn_qk_int8_pv_fp16_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_warp",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    if qk_quant_gran not in {"per_block", "per_warp"}:
        raise ValueError("sm75 INT8 backend supports per_block or per_warp Q/K quantization")
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    _validate_fixed_qkv(q, k, v, tensor_layout)

    short_result = _short_sequence_attention(
        q, k, v, tensor_layout, is_causal, sm_scale, return_lse
    )
    if short_result is not None:
        return short_result if return_lse else short_result[0]

    tensor_layout_id = 0 if tensor_layout == "NHD" else 1
    is_causal_id = 1 if is_causal else 0
    qk_quant_gran_id = {"per_block": 1, "per_warp": 2}[qk_quant_gran]
    return_lse_id = 1 if return_lse else 0

    head_dim_og = q.size(-1)
    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if tensor_layout_id == 0 else 2
    nh_dim = 2 if tensor_layout_id == 0 else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if qk_quant_gran == "per_block":
        q_int8, q_scale, k_int8, k_scale = per_block_int8(
            q, k, km, tensor_layout=tensor_layout, BLKQ=64, BLKK=64
        )
    else:
        q_int8, q_scale, k_int8, k_scale = per_warp_int8(
            q,
            k,
            km,
            tensor_layout=tensor_layout,
            BLKQ=64,
            WARPQ=16,
            BLKK=64,
            fuse_qk=(km is None and (is_causal or (tensor_layout == "HND" and q.size(-1) == 64))),
        )

    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    if pv_accum_dtype in ["fp32", "fp16+fp32"] and smooth_v:
        warnings.warn(f"pv_accum_dtype is {pv_accum_dtype}, smooth_v will be ignored.")
        smooth_v = False

    if pv_accum_dtype == "fp32":
        v = v.contiguous()
        lse = sm75_compile.qk_int8_sv_f16_accum_f32_attn(
            q_int8, k_int8, v, o, q_scale, k_scale, tensor_layout_id, is_causal_id, qk_quant_gran_id, sm_scale, return_lse_id
        )
    elif pv_accum_dtype == "fp16":
        if smooth_v:
            smoothed_v, vm = sub_mean(v, tensor_layout=tensor_layout)
            lse = sm75_compile.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(
                q_int8, k_int8, smoothed_v, o, q_scale, k_scale, vm, tensor_layout_id, is_causal_id, qk_quant_gran_id, sm_scale, return_lse_id
            )
        else:
            v = v.contiguous()
            lse = sm75_compile.qk_int8_sv_f16_accum_f16_attn(
                q_int8, k_int8, v, o, q_scale, k_scale, tensor_layout_id, is_causal_id, qk_quant_gran_id, sm_scale, return_lse_id
            )
    elif pv_accum_dtype == "fp16+fp32":
        v = v.contiguous()
        lse = sm75_compile.qk_int8_sv_f16_accum_f16_attn_inst_buf(
            q_int8, k_int8, v, o, q_scale, k_scale, tensor_layout_id, is_causal_id, qk_quant_gran_id, sm_scale, return_lse_id
        )
    else:
        raise ValueError(f"Unsupported pv_accum_dtype: {pv_accum_dtype}")

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    return o
