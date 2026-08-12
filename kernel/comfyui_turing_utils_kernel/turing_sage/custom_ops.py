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
) -> torch.Tensor:
    from .core import w8a8attn

    return w8a8attn(
        query,
        key,
        value,
        tensor_layout=tensor_layout,
        sm_scale=sm_scale if sm_scale > 0.0 else None,
    )


@w8a8_attention.register_fake
def _w8a8_attention_fake(query, key, value, tensor_layout, sm_scale):
    return torch.empty_like(query)
