from __future__ import annotations

import torch


@torch.library.custom_op("turing_utils::overlap_blend", mutates_args=())
def overlap_blend_op(
    window_values: torch.Tensor,
    local_indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    from .core import overlap_blend

    return overlap_blend(window_values, local_indices, weights)


@overlap_blend_op.register_fake
def _overlap_blend_fake(window_values, local_indices, weights):
    return torch.empty(
        (
            window_values.shape[0],
            local_indices.shape[0],
            window_values.shape[-1],
        ),
        dtype=window_values.dtype,
        device=window_values.device,
    )


@torch.library.custom_op(
    "turing_utils::overlap_accumulate",
    mutates_args=("output",),
)
def overlap_accumulate_op(
    window_values: torch.Tensor,
    local_indices: torch.Tensor,
    weights: torch.Tensor,
    output_indices: torch.Tensor,
    output: torch.Tensor,
) -> None:
    from .core import overlap_accumulate

    overlap_accumulate(
        window_values,
        local_indices,
        weights,
        output_indices,
        output,
    )


@overlap_accumulate_op.register_fake
def _overlap_accumulate_fake(
    window_values,
    local_indices,
    weights,
    output_indices,
    output,
):
    return None


@torch.library.custom_op("turing_utils::qk_rms_rope_int8", mutates_args=())
def qk_rms_rope_int8(
    query: torch.Tensor,
    key: torch.Tensor,
    query_norm: torch.Tensor,
    key_norm: torch.Tensor,
    freqs: torch.Tensor,
    key_freqs: torch.Tensor,
    epsilon: float,
    rot_dim: int,
    tensor_layout: str,
    norm_scope: str,
    split_half: bool,
    rotate_qk: bool,
    stabilize_k: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from .quant import rms_rope_per_warp_int8

    return rms_rope_per_warp_int8(
        query,
        key,
        query_norm,
        key_norm,
        freqs if freqs.numel() else None,
        key_freqs=key_freqs if key_freqs.numel() else None,
        epsilon=epsilon,
        rot_dim=rot_dim,
        tensor_layout=tensor_layout,
        norm_scope=norm_scope,
        split_half=split_half,
        rotate_qk=rotate_qk,
        stabilize_k=stabilize_k,
    )


@qk_rms_rope_int8.register_fake
def _qk_rms_rope_int8_fake(
    query,
    key,
    query_norm,
    key_norm,
    freqs,
    key_freqs,
    epsilon,
    rot_dim,
    tensor_layout,
    norm_scope,
    split_half,
    rotate_qk,
    stabilize_k,
):
    if tensor_layout == "HND":
        batch, query_heads, query_tokens, _ = query.shape
        _, key_heads, key_tokens, _ = key.shape
    elif tensor_layout == "NHD":
        batch, query_tokens, query_heads, _ = query.shape
        _, key_tokens, key_heads, _ = key.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    query_scale = torch.empty(
        (batch, query_heads, ((query_tokens + 63) // 64) * 4),
        dtype=torch.float32,
        device=query.device,
    )
    key_scale = torch.empty(
        (batch, key_heads, (key_tokens + 63) // 64),
        dtype=torch.float32,
        device=key.device,
    )
    return (
        torch.empty_like(query, dtype=torch.int8),
        query_scale,
        torch.empty_like(key, dtype=torch.int8),
        key_scale,
    )


@torch.library.custom_op("turing_utils::qk_rms_rope_anchor", mutates_args=())
def qk_rms_rope_anchor(
    key: torch.Tensor,
    key_norm: torch.Tensor,
    freqs: torch.Tensor,
    epsilon: float,
    rot_dim: int,
    tensor_layout: str,
    norm_scope: str,
    split_half: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .quant import rms_rope_per_warp_int8

    result = rms_rope_per_warp_int8(
        key,
        key,
        key_norm,
        key_norm,
        freqs if freqs.numel() else None,
        epsilon=epsilon,
        rot_dim=rot_dim,
        tensor_layout=tensor_layout,
        norm_scope=norm_scope,
        split_half=split_half,
        rotate_qk=True,
        stabilize_k=True,
        return_anchor=True,
    )
    return result[4], result[5]


@qk_rms_rope_anchor.register_fake
def _qk_rms_rope_anchor_fake(
    key,
    key_norm,
    freqs,
    epsilon,
    rot_dim,
    tensor_layout,
    norm_scope,
    split_half,
):
    if tensor_layout == "HND":
        batch, heads, _, head_dim = key.shape
    elif tensor_layout == "NHD":
        batch, _, heads, head_dim = key.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    return (
        torch.empty((batch, heads), dtype=torch.int32, device=key.device),
        torch.empty(
            (batch, heads, head_dim), dtype=torch.float32, device=key.device
        ),
    )


@torch.library.custom_op(
    "turing_utils::qk_rms_rope_int8_anchored", mutates_args=()
)
def qk_rms_rope_int8_anchored(
    query: torch.Tensor,
    key: torch.Tensor,
    query_norm: torch.Tensor,
    key_norm: torch.Tensor,
    freqs: torch.Tensor,
    key_freqs: torch.Tensor,
    anchor_indices: torch.Tensor,
    anchor_values: torch.Tensor,
    epsilon: float,
    rot_dim: int,
    tensor_layout: str,
    norm_scope: str,
    split_half: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from .quant import rms_rope_per_warp_int8

    return rms_rope_per_warp_int8(
        query,
        key,
        query_norm,
        key_norm,
        freqs if freqs.numel() else None,
        key_freqs=key_freqs if key_freqs.numel() else None,
        epsilon=epsilon,
        rot_dim=rot_dim,
        tensor_layout=tensor_layout,
        norm_scope=norm_scope,
        split_half=split_half,
        rotate_qk=True,
        stabilize_k=True,
        anchor_indices=anchor_indices,
        anchor_values=anchor_values,
    )


@qk_rms_rope_int8_anchored.register_fake
def _qk_rms_rope_int8_anchored_fake(
    query,
    key,
    query_norm,
    key_norm,
    freqs,
    key_freqs,
    anchor_indices,
    anchor_values,
    epsilon,
    rot_dim,
    tensor_layout,
    norm_scope,
    split_half,
):
    return _qk_rms_rope_int8_fake(
        query,
        key,
        query_norm,
        key_norm,
        freqs,
        key_freqs,
        epsilon,
        rot_dim,
        tensor_layout,
        norm_scope,
        split_half,
        True,
        True,
    )


@torch.library.custom_op(
    "turing_utils::qk_rms_rope_int8_out",
    mutates_args=("query_output", "query_scale", "key_output", "key_scale"),
)
def qk_rms_rope_int8_out(
    query: torch.Tensor,
    key: torch.Tensor,
    query_norm: torch.Tensor,
    key_norm: torch.Tensor,
    freqs: torch.Tensor,
    key_freqs: torch.Tensor,
    anchor_indices: torch.Tensor,
    anchor_values: torch.Tensor,
    query_output: torch.Tensor,
    query_scale: torch.Tensor,
    key_output: torch.Tensor,
    key_scale: torch.Tensor,
    epsilon: float,
    rot_dim: int,
    tensor_layout: str,
    norm_scope: str,
    split_half: bool,
    rotate_qk: bool,
    stabilize_k: bool,
) -> None:
    from .quant import rms_rope_per_warp_int8

    rms_rope_per_warp_int8(
        query,
        key,
        query_norm,
        key_norm,
        freqs if freqs.numel() else None,
        key_freqs=key_freqs if key_freqs.numel() else None,
        epsilon=epsilon,
        rot_dim=rot_dim,
        tensor_layout=tensor_layout,
        norm_scope=norm_scope,
        split_half=split_half,
        rotate_qk=rotate_qk,
        stabilize_k=stabilize_k,
        anchor_indices=anchor_indices if anchor_indices.numel() else None,
        anchor_values=anchor_values if anchor_values.numel() else None,
        output_buffers=(query_output, query_scale, key_output, key_scale),
    )


@qk_rms_rope_int8_out.register_fake
def _qk_rms_rope_int8_out_fake(
    query,
    key,
    query_norm,
    key_norm,
    freqs,
    key_freqs,
    anchor_indices,
    anchor_values,
    query_output,
    query_scale,
    key_output,
    key_scale,
    epsilon,
    rot_dim,
    tensor_layout,
    norm_scope,
    split_half,
    rotate_qk,
    stabilize_k,
):
    return None


@torch.library.custom_op("turing_utils::sage_attention", mutates_args=())
def sage_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    from .core import sageattn

    return sageattn(
        query,
        key,
        value,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        sm_scale=sm_scale if sm_scale > 0.0 else None,
        smooth_k=False,
    )


@sage_attention.register_fake
def _sage_attention_fake(
    query,
    key,
    value,
    tensor_layout,
    is_causal,
    sm_scale,
):
    return torch.empty_like(query)


@torch.library.custom_op("turing_utils::w8a8_attention", mutates_args=())
def w8a8_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    sm_scale: float,
    key_tile_tokens: int,
    rotate_qk: bool,
    stabilize_k: bool,
) -> torch.Tensor:
    from .core import w8a8attn

    return w8a8attn(
        query,
        key,
        value,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        sm_scale=sm_scale if sm_scale > 0.0 else None,
        key_tile_tokens=key_tile_tokens,
        rotate_qk=rotate_qk,
        stabilize_k=stabilize_k,
    )


@w8a8_attention.register_fake
def _w8a8_attention_fake(
    query,
    key,
    value,
    tensor_layout,
    is_causal,
    sm_scale,
    key_tile_tokens,
    rotate_qk,
    stabilize_k,
):
    return torch.empty_like(query)


@torch.library.custom_op("turing_utils::sage_attention_varlen", mutates_args=())
def sage_attention_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    from .core import sageattn_varlen

    return sageattn_varlen(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal=is_causal,
        sm_scale=sm_scale if sm_scale > 0.0 else None,
        smooth_k=False,
    )


@sage_attention_varlen.register_fake
def _sage_attention_varlen_fake(
    query,
    key,
    value,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    is_causal,
    sm_scale,
):
    return torch.empty_like(query)


@torch.library.custom_op("turing_utils::w8a8_attention_varlen", mutates_args=())
def w8a8_attention_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool,
    sm_scale: float,
    rotate_qk: bool,
) -> torch.Tensor:
    from .core import w8a8attn_varlen

    return w8a8attn_varlen(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal=is_causal,
        sm_scale=sm_scale if sm_scale > 0.0 else None,
        rotate_qk=rotate_qk,
        stabilize_k=False,
    )


@w8a8_attention_varlen.register_fake
def _w8a8_attention_varlen_fake(
    query,
    key,
    value,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    is_causal,
    sm_scale,
    rotate_qk,
):
    return torch.empty_like(query)


@torch.library.custom_op("turing_utils::sol_attention", mutates_args=())
def sol_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dense_query_starts: list[int],
    dense_query_stops: list[int],
    exact_kv_starts: list[int],
    exact_kv_stops: list[int],
    threshold_sigma: float,
    residual_subblocks: int,
    use_w8a8: bool,
    sm_scale: float,
    key_tile_tokens: int,
    rotate_qk: bool,
    stabilize_k: bool,
) -> torch.Tensor:
    from .core import sol_sparse_sageattn

    if len(dense_query_starts) != len(dense_query_stops):
        raise ValueError("dense Query range starts/stops must have equal length")
    if len(exact_kv_starts) != len(exact_kv_stops):
        raise ValueError("exact KV range starts/stops must have equal length")
    return sol_sparse_sageattn(
        query,
        key,
        value,
        tensor_layout="HND",
        sm_scale=sm_scale if sm_scale > 0.0 else None,
        dense_query_ranges=tuple(zip(dense_query_starts, dense_query_stops)),
        exact_kv_ranges=tuple(zip(exact_kv_starts, exact_kv_stops)),
        threshold_sigma=threshold_sigma,
        residual_subblocks=residual_subblocks,
        use_w8a8=use_w8a8,
        key_tile_tokens=key_tile_tokens,
        rotate_qk=rotate_qk,
        stabilize_k=stabilize_k,
    )


@sol_attention.register_fake
def _sol_attention_fake(
    query,
    key,
    value,
    dense_query_starts,
    dense_query_stops,
    exact_kv_starts,
    exact_kv_stops,
    threshold_sigma,
    residual_subblocks,
    use_w8a8,
    sm_scale,
    key_tile_tokens,
    rotate_qk,
    stabilize_k,
):
    return torch.empty_like(query)


@torch.library.custom_op("turing_utils::sla_attention", mutates_args=())
def sla_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dense_query_starts: list[int],
    dense_query_stops: list[int],
    exact_kv_starts: list[int],
    exact_kv_stops: list[int],
    sparsity_ratio: float,
    use_w8a8: bool,
    sm_scale: float,
    key_tile_tokens: int,
    rotate_qk: bool,
    stabilize_k: bool,
) -> torch.Tensor:
    from .core import sla_sparse_sageattn

    if len(dense_query_starts) != len(dense_query_stops):
        raise ValueError("dense Query range starts/stops must have equal length")
    if len(exact_kv_starts) != len(exact_kv_stops):
        raise ValueError("exact KV range starts/stops must have equal length")
    return sla_sparse_sageattn(
        query,
        key,
        value,
        tensor_layout="HND",
        sm_scale=sm_scale if sm_scale > 0.0 else None,
        dense_query_ranges=tuple(zip(dense_query_starts, dense_query_stops)),
        exact_kv_ranges=tuple(zip(exact_kv_starts, exact_kv_stops)),
        sparsity_ratio=sparsity_ratio,
        use_w8a8=use_w8a8,
        key_tile_tokens=key_tile_tokens,
        rotate_qk=rotate_qk,
        stabilize_k=stabilize_k,
    )


@sla_attention.register_fake
def _sla_attention_fake(
    query,
    key,
    value,
    dense_query_starts,
    dense_query_stops,
    exact_kv_starts,
    exact_kv_stops,
    sparsity_ratio,
    use_w8a8,
    sm_scale,
    key_tile_tokens,
    rotate_qk,
    stabilize_k,
):
    return torch.empty_like(query)
