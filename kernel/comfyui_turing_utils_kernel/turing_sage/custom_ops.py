from __future__ import annotations

import torch


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
        sm_scale=sm_scale if sm_scale > 0.0 else None,
        key_tile_tokens=key_tile_tokens,
        rotate_qk=rotate_qk,
        stabilize_k=stabilize_k,
    )


@w8a8_attention.register_fake
def _w8a8_attention_fake(
    query, key, value, tensor_layout, sm_scale, key_tile_tokens, rotate_qk, stabilize_k
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
