import torch

from .. import _sage_fused_sm75 as _fused


def rms_rope_per_warp_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    freqs: torch.Tensor | None,
    *,
    key_freqs: torch.Tensor | None = None,
    epsilon: float,
    rot_dim: int,
    tensor_layout: str,
    norm_scope: str,
    split_half: bool,
    rotate_qk: bool,
    stabilize_k: bool,
    anchor_indices: torch.Tensor | None = None,
    anchor_values: torch.Tensor | None = None,
    return_anchor: bool = False,
    output_buffers: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ] | None = None,
):
    """Fuse Q/K normalization and RoPE into the production INT8 contract."""
    if tensor_layout == "HND":
        batch, q_heads, q_tokens, head_dim = q.shape
        _, k_heads, k_tokens, _ = k.shape
        layout_id = 1
    elif tensor_layout == "NHD":
        batch, q_tokens, q_heads, head_dim = q.shape
        _, k_tokens, k_heads, _ = k.shape
        layout_id = 0
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    if head_dim not in (64, 128):
        raise ValueError("fused RMSNorm+RoPE Q/K quantization requires head_dim 64 or 128")
    if norm_scope not in {"head", "row"}:
        raise ValueError("norm_scope must be head or row")

    q_scale_shape = (batch, q_heads, ((q_tokens + 63) // 64) * 4)
    k_scale_shape = (batch, k_heads, (k_tokens + 63) // 64)
    if output_buffers is None:
        q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
        k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
        q_scale = torch.empty(
            q_scale_shape, dtype=torch.float32, device=q.device
        )
        k_scale = torch.empty(
            k_scale_shape, dtype=torch.float32, device=k.device
        )
    else:
        q_int8, q_scale, k_int8, k_scale = output_buffers
        expected = (
            (q_int8, q.shape, torch.int8),
            (q_scale, q_scale_shape, torch.float32),
            (k_int8, k.shape, torch.int8),
            (k_scale, k_scale_shape, torch.float32),
        )
        if any(
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != q.device
            for tensor, shape, dtype in expected
        ):
            raise ValueError(
                "Q/K preprocessing output buffer has an incompatible shape, dtype, or device"
            )
    if norm_scope == "row":
        q_rrms = torch.empty((batch, q_tokens), dtype=torch.float32, device=q.device)
        k_rrms = torch.empty((batch, k_tokens), dtype=torch.float32, device=k.device)
        norm_scope_id = 1
    else:
        q_rrms = torch.empty(0, dtype=torch.float32, device=q.device)
        k_rrms = torch.empty(0, dtype=torch.float32, device=k.device)
        norm_scope_id = 0
    detect_anchor = bool(
        stabilize_k
        and rotate_qk
        and anchor_indices is None
        and anchor_values is None
    )
    if stabilize_k and rotate_qk:
        if (anchor_indices is None) != (anchor_values is None):
            raise ValueError("precomputed K anchor requires indices and values")
        if anchor_indices is None:
            anchor_indices = torch.empty(
                (batch, k_heads), dtype=torch.int32, device=k.device
            )
            anchor_values = torch.empty(
                (batch, k_heads, head_dim), dtype=torch.float32, device=k.device
            )
        elif (
            anchor_indices.shape != (batch, k_heads)
            or anchor_indices.dtype != torch.int32
            or anchor_indices.device != k.device
            or anchor_values.shape != (batch, k_heads, head_dim)
            or anchor_values.dtype != torch.float32
            or anchor_values.device != k.device
        ):
            raise ValueError("precomputed K anchor has an incompatible shape, dtype, or device")
        anchor_indices = anchor_indices.contiguous()
        anchor_values = anchor_values.contiguous()
    else:
        if anchor_indices is not None or anchor_values is not None:
            raise ValueError("precomputed K anchor requires rotated K stabilization")
        anchor_indices = torch.empty(0, dtype=torch.int32, device=k.device)
        anchor_values = torch.empty(0, dtype=torch.float32, device=k.device)
    if freqs is None:
        freqs = torch.empty(0, dtype=q.dtype, device=q.device)
    if key_freqs is None:
        key_freqs = freqs

    _fused.quant_qk_rms_rope_int8_cuda(
        q,
        k,
        q_int8,
        k_int8,
        q_scale,
        k_scale,
        q_norm,
        k_norm,
        freqs,
        key_freqs,
        q_rrms,
        k_rrms,
        anchor_indices,
        anchor_values,
        float(epsilon),
        int(rot_dim),
        64,
        16,
        64,
        layout_id,
        norm_scope_id,
        bool(split_half),
        bool(rotate_qk),
        detect_anchor,
    )
    if return_anchor:
        return q_int8, q_scale, k_int8, k_scale, anchor_indices, anchor_values
    return q_int8, q_scale, k_int8, k_scale


def quantize_query_per_warp(
    q: torch.Tensor,
    BLKQ: int = 64,
    WARPQ: int = 16,
    tensor_layout: str = "HND",
):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    if tensor_layout == "HND":
        batch, q_heads, q_tokens, _ = q.shape
    elif tensor_layout == "NHD":
        batch, q_tokens, q_heads, _ = q.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    q_scale = torch.empty(
        (batch, q_heads, ((q_tokens + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ)),
        device=q.device,
        dtype=torch.float32,
    )
    layout_id = 0 if tensor_layout == "NHD" else 1
    _fused.quant_per_warp_int8_cuda(q, q_int8, q_scale, BLKQ, WARPQ, layout_id)
    return q_int8, q_scale


def quantize_key_per_block(
    k: torch.Tensor,
    BLKK: int = 64,
    tensor_layout: str = "HND",
):
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    if tensor_layout == "HND":
        batch, kv_heads, kv_tokens, _ = k.shape
    elif tensor_layout == "NHD":
        batch, kv_tokens, kv_heads, _ = k.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    k_scale = torch.empty(
        (batch, kv_heads, (kv_tokens + BLKK - 1) // BLKK),
        device=k.device,
        dtype=torch.float32,
    )
    layout_id = 0 if tensor_layout == "NHD" else 1
    _fused.quant_per_block_int8_cuda(k, k_int8, k_scale, BLKK, layout_id)
    return k_int8, k_scale


def per_warp_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    BLKQ: int = 64,
    WARPQ: int = 16,
    BLKK: int = 64,
    tensor_layout: str = "HND",
    fuse_qk: bool = False,
):
    """Quantize Q per warp and K per block for the stable SM75 backend."""
    if tensor_layout == "HND":
        batch, q_heads, q_tokens, _ = q.shape
        _, kv_heads, kv_tokens, _ = k.shape
    elif tensor_layout == "NHD":
        batch, q_tokens, q_heads, _ = q.shape
        _, kv_tokens, kv_heads, _ = k.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    layout_id = 0 if tensor_layout == "NHD" else 1
    if fuse_qk:
        q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
        k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
        q_scale = torch.empty(
            (batch, q_heads, ((q_tokens + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ)),
            device=q.device,
            dtype=torch.float32,
        )
        k_scale = torch.empty(
            (batch, kv_heads, (kv_tokens + BLKK - 1) // BLKK),
            device=q.device,
            dtype=torch.float32,
        )
        _fused.quant_qk_per_warp_int8_cuda(
            q, k, q_int8, k_int8, q_scale, k_scale, BLKQ, WARPQ, BLKK, layout_id
        )
    else:
        q_int8, q_scale = quantize_query_per_warp(q, BLKQ, WARPQ, tensor_layout)
        k_int8, k_scale = quantize_key_per_block(k, BLKK, tensor_layout)
    return q_int8, q_scale, k_int8, k_scale


def per_warp_int8_hadamard(
    q: torch.Tensor,
    k: torch.Tensor,
    BLKQ: int = 64,
    WARPQ: int = 16,
    BLKK: int = 64,
    tensor_layout: str = "HND",
    stabilize_k: bool = True,
):
    """Fuse the shared randomized Hadamard transform into Q/K quantization."""
    if tensor_layout == "HND":
        batch, q_heads, q_tokens, _ = q.shape
        _, kv_heads, kv_tokens, _ = k.shape
    elif tensor_layout == "NHD":
        batch, q_tokens, q_heads, _ = q.shape
        _, kv_tokens, kv_heads, _ = k.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    q_scale = torch.empty(
        (batch, q_heads, ((q_tokens + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ)),
        device=q.device,
        dtype=torch.float32,
    )
    k_scale = torch.empty(
        (batch, kv_heads, (kv_tokens + BLKK - 1) // BLKK),
        device=q.device,
        dtype=torch.float32,
    )
    layout_id = 0 if tensor_layout == "NHD" else 1
    if stabilize_k:
        anchor_indices = torch.empty(
            (batch, kv_heads), dtype=torch.int32, device=q.device
        )
        _fused.quant_qk_per_warp_int8_rotated_anchored_cuda(
            q,
            k,
            q_int8,
            k_int8,
            q_scale,
            k_scale,
            anchor_indices,
            BLKQ,
            WARPQ,
            BLKK,
            layout_id,
        )
    else:
        _fused.quant_qk_per_warp_int8_rotated_cuda(
            q,
            k,
            q_int8,
            k_int8,
            q_scale,
            k_scale,
            BLKQ,
            WARPQ,
            BLKK,
            layout_id,
        )
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
    rotate_qk: bool = False,
):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    batch = cu_seqlens_q.numel() - 1
    q_scale = torch.empty(
        (batch, q.size(1), ((max_seqlen_q + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ)),
        device=q.device,
        dtype=torch.float32,
    )
    k_scale = torch.empty(
        (batch, k.size(1), (max_seqlen_k + BLKK - 1) // BLKK),
        device=q.device,
        dtype=torch.float32,
    )
    _fused.quant_per_warp_int8_varlen_cuda(
        q,
        cu_seqlens_q,
        q_int8,
        q_scale,
        max_seqlen_q,
        BLKQ,
        WARPQ,
        bool(rotate_qk),
    )
    _fused.quant_per_warp_int8_varlen_cuda(
        k,
        cu_seqlens_k,
        k_int8,
        k_scale,
        max_seqlen_k,
        BLKK,
        BLKK,
        bool(rotate_qk),
    )
    return q_int8, q_scale, k_int8, k_scale
