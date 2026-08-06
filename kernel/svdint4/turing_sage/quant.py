import torch
from typing import Optional

from .. import _sage_fused_sm75 as _fused


def token_block_mean(
    value: torch.Tensor,
    block_size: int,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """Return FP32 token means without materializing an FP32 copy of the input."""
    if tensor_layout == "HND":
        batch, heads, tokens, head_dim = value.shape
    elif tensor_layout == "NHD":
        batch, tokens, heads, head_dim = value.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    blocks = (tokens + block_size - 1) // block_size
    output = torch.empty(
        (batch, heads, blocks, head_dim), device=value.device, dtype=torch.float32
    )
    _fused.token_block_mean_cuda(
        value, output, block_size, 0 if tensor_layout == "NHD" else 1
    )
    return output


def per_thread_int4(
    q: torch.Tensor,
    k: torch.Tensor,
    tensor_layout: str = "HND",
    smooth_q: bool = True,
    smooth_k: bool = True,
):
    """Packed per-thread INT4 quantization used by the Turing Sage2 path."""
    if tensor_layout == "HND":
        batch, q_heads, q_tokens, head_dim = q.shape
        _, kv_heads, kv_tokens, _ = k.shape
    elif tensor_layout == "NHD":
        batch, q_tokens, q_heads, head_dim = q.shape
        _, kv_tokens, kv_heads, _ = k.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    if head_dim % 2:
        raise ValueError("INT4 Q/K require an even head dimension")

    q_shape = list(q.shape)
    k_shape = list(k.shape)
    q_shape[-1] //= 2
    k_shape[-1] //= 2
    q_int4 = torch.empty(q_shape, dtype=torch.int8, device=q.device)
    k_int4 = torch.empty(k_shape, dtype=torch.int8, device=k.device)
    q_blocks = (q_tokens + 63) // 64
    k_blocks = (kv_tokens + 63) // 64
    q_scale = torch.empty((batch, q_heads, q_blocks * 32), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((batch, kv_heads, k_blocks * 4), device=k.device, dtype=torch.float32)

    empty_mean = torch.empty((0,), device=q.device, dtype=torch.float32)
    q_mean = token_block_mean(q, 64, tensor_layout) if smooth_q else empty_mean
    k_mean = token_block_mean(k, kv_tokens, tensor_layout) if smooth_k else empty_mean
    layout_id = 0 if tensor_layout == "NHD" else 1
    _fused.quant_query_per_thread_int4_cuda(
        q, q_mean, q_int4, q_scale, layout_id, smooth_q
    )
    _fused.quant_key_per_thread_int4_cuda(
        k, k_mean, k_int4, k_scale, layout_id, smooth_k
    )
    return q_int4, q_scale, k_int4, k_scale, q_mean, k_mean


def per_thread_int4_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    tensor_layout: str = "HND",
    smooth_q: bool = True,
    smooth_k: bool = True,
):
    """Official-layout INT4 preprocessing with fused smoothing where local.

    Q block means are produced in the same CTA that quantizes the corresponding
    64-token block. Centered K quantization uses one CTA per 64-token block;
    its sequence-wide mean remains a separate reduction because it crosses all
    K blocks.
    """
    if tensor_layout == "HND":
        batch, q_heads, q_tokens, head_dim = q.shape
        _, kv_heads, kv_tokens, _ = k.shape
    elif tensor_layout == "NHD":
        batch, q_tokens, q_heads, head_dim = q.shape
        _, kv_tokens, kv_heads, _ = k.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    if head_dim % 2:
        raise ValueError("INT4 Q/K require an even head dimension")

    q_shape = list(q.shape)
    k_shape = list(k.shape)
    q_shape[-1] //= 2
    k_shape[-1] //= 2
    q_int4 = torch.empty(q_shape, dtype=torch.int8, device=q.device)
    k_int4 = torch.empty(k_shape, dtype=torch.int8, device=k.device)
    q_blocks = (q_tokens + 63) // 64
    k_blocks = (kv_tokens + 63) // 64
    q_scale = torch.empty(
        (batch, q_heads, q_blocks * 32), device=q.device, dtype=torch.float32
    )
    k_scale = torch.empty(
        (batch, kv_heads, k_blocks * 4), device=k.device, dtype=torch.float32
    )
    empty_mean = torch.empty((0,), device=q.device, dtype=torch.float32)
    layout_id = 0 if tensor_layout == "NHD" else 1

    if smooth_q:
        q_mean = torch.empty(
            (batch, q_heads, q_blocks, head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        _fused.quant_query_per_thread_int4_fused_cuda(
            q, q_mean, q_int4, q_scale, layout_id
        )
    else:
        q_mean = empty_mean
        _fused.quant_query_per_thread_int4_cuda(
            q, q_mean, q_int4, q_scale, layout_id, False
        )

    if smooth_k:
        k_mean = token_block_mean(k, kv_tokens, tensor_layout)
        _fused.quant_key_per_thread_int4_fused_cuda(
            k, k_mean, k_int4, k_scale, layout_id
        )
    else:
        k_mean = empty_mean
        _fused.quant_key_per_thread_int4_cuda(
            k, k_mean, k_int4, k_scale, layout_id, False
        )
    return q_int4, q_scale, k_int4, k_scale, q_mean, k_mean


def sage2_score_correction(
    q_mean: torch.Tensor,
    k: torch.Tensor,
    k_mean: torch.Tensor,
    tensor_layout: str = "HND",
    smooth_k: bool = True,
) -> torch.Tensor:
    """Compute the Q-smoothing score correction with FP16 TC/FP32 accumulation.

    The result has one correction row per 64-token Q block. It is deliberately
    kept separate from the packed INT4 score kernel so the correction path can
    be validated and profiled independently from the unchanged INT4 MMA.
    """
    if tensor_layout == "HND":
        batch, kv_heads, kv_tokens, _ = k.shape
    elif tensor_layout == "NHD":
        batch, kv_tokens, kv_heads, _ = k.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    if q_mean.ndim != 4 or q_mean.size(0) != batch:
        raise ValueError("q_mean must have shape [B, Hq, Qblocks, D]")
    if q_mean.size(1) % kv_heads:
        raise ValueError("Q heads must be divisible by KV heads")
    correction = torch.empty(
        (batch, q_mean.size(1), q_mean.size(2), kv_tokens),
        device=k.device,
        dtype=torch.float32,
    )
    empty_mean = torch.empty((0,), device=k.device, dtype=torch.float32)
    _fused.sage2_score_correction_cuda(
        q_mean,
        k,
        k_mean if smooth_k else empty_mean,
        correction,
        0 if tensor_layout == "NHD" else 1,
        smooth_k,
    )
    return correction


def per_block_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    km: Optional[torch.Tensor] = None,
    BLKQ: int = 64,
    BLKK: int = 64,
    tensor_layout: str = "HND",
):
    """Quantize Q and K with one symmetric INT8 scale per 64-token CTA."""
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, _ = q.shape
        _, h_kv, kv_len, _ = k.shape
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, _ = q.shape
        _, kv_len, h_kv, _ = k.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    tensor_layout_id = 0 if tensor_layout == "NHD" else 1
    q_scale = torch.empty(
        (b, h_qo, (qo_len + BLKQ - 1) // BLKQ),
        device=q.device,
        dtype=torch.float32,
    )
    k_scale = torch.empty(
        (b, h_kv, (kv_len + BLKK - 1) // BLKK),
        device=k.device,
        dtype=torch.float32,
    )

    _fused.quant_per_block_int8_cuda(q, q_int8, q_scale, BLKQ, tensor_layout_id)
    if km is not None:
        km = km.squeeze(1) if tensor_layout_id == 0 else km.squeeze(2)
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(
            k, km, k_int8, k_scale, BLKK, tensor_layout_id
        )
    else:
        _fused.quant_per_block_int8_cuda(k, k_int8, k_scale, BLKK, tensor_layout_id)

    return q_int8, q_scale, k_int8, k_scale


def per_warp_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    km: Optional[torch.Tensor] = None,
    BLKQ: int = 64,
    WARPQ: int = 16,
    BLKK: int = 64,
    tensor_layout: str = "HND",
    fuse_qk: bool = False,
):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, _ = q.shape
        _, h_kv, kv_len, _ = k.shape
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, _ = q.shape
        _, kv_len, h_kv, _ = k.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    tensor_layout_id = 0 if tensor_layout == "NHD" else 1
    q_scale = torch.empty(
        (b, h_qo, ((qo_len + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ)),
        device=q.device,
        dtype=torch.float32,
    )
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32)

    if km is None and fuse_qk:
        _fused.quant_qk_per_warp_int8_cuda(q, k, q_int8, k_int8, q_scale, k_scale, BLKQ, WARPQ, BLKK, tensor_layout_id)
    else:
        _fused.quant_per_warp_int8_cuda(q, q_int8, q_scale, BLKQ, WARPQ, tensor_layout_id)
        if km is not None:
            km = km.squeeze(1) if tensor_layout_id == 0 else km.squeeze(2)
            _fused.quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, k_scale, BLKK, tensor_layout_id)
        else:
            _fused.quant_per_block_int8_cuda(k, k_int8, k_scale, BLKK, tensor_layout_id)

    return q_int8, q_scale, k_int8, k_scale


def per_warp_int8_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    BLKQ: int = 64,
    WARPQ: int = 16,
    BLKK: int = 64,
):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    _, h_qo, _ = q.shape
    _, h_kv, _ = k.shape
    batch = cu_seqlens_q.numel() - 1

    q_scale = torch.empty(
        (batch, h_qo, ((max_seqlen_q + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ)),
        device=q.device,
        dtype=torch.float32,
    )
    k_scale = torch.empty(
        (batch, h_kv, (max_seqlen_k + BLKK - 1) // BLKK),
        device=k.device,
        dtype=torch.float32,
    )

    _fused.quant_per_warp_int8_varlen_cuda(q, cu_seqlens_q, q_int8, q_scale, max_seqlen_q, BLKQ, WARPQ)
    _fused.quant_per_warp_int8_varlen_cuda(k, cu_seqlens_k, k_int8, k_scale, max_seqlen_k, BLKK, BLKK)

    return q_int8, q_scale, k_int8, k_scale


def sub_mean(v: torch.Tensor, tensor_layout: str = "HND"):
    tensor_layout_id = 0 if tensor_layout == "NHD" else 1
    vm = v.mean(dim=1 if tensor_layout_id == 0 else 2)
    v_smoothed = torch.empty(v.shape, dtype=torch.float16, device=v.device)
    _fused.sub_mean_cuda(v, vm, v_smoothed, tensor_layout_id)
    return v_smoothed, vm
