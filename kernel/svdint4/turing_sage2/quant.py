import torch
from typing import Optional

from .. import _sage_fused_sm75 as _fused


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
