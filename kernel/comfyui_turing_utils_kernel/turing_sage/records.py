"""Immutable tensor contracts exchanged by the attention preparation stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True, slots=True)
class PrequantizedSageAttention:
    query_int8: torch.Tensor
    query_scale: torch.Tensor
    key_int8: torch.Tensor
    key_scale: torch.Tensor
    value: torch.Tensor
    tensor_layout: str
    original_head_dim: int
    is_causal: bool
    sm_scale: float


@dataclass(frozen=True, slots=True)
class PrequantizedQK:
    query_int8: torch.Tensor
    query_scale: torch.Tensor
    key_int8: torch.Tensor
    key_scale: torch.Tensor
    tensor_layout: str
    input_dtype: torch.dtype
    original_head_dim: int
    route_original_basis: bool = False


@dataclass(frozen=True, slots=True)
class PrequantizedSolAttention:
    query_int8: torch.Tensor
    query_scale: torch.Tensor
    key_int8: torch.Tensor
    key_scale: torch.Tensor
    value: Optional[torch.Tensor]
    value_int8: torch.Tensor
    value_scale: torch.Tensor
    summaries: tuple[torch.Tensor, ...]
    value_source_indices: Optional[torch.Tensor]
    sparse_query_blocks: torch.Tensor
    exact_kv_blocks: torch.Tensor
    output_dtype: torch.dtype
    sm_scale: float
    threshold_sigma: float
    residual_subblocks: int
    possible_blocks: int
    use_w8a8: bool
    force_dense: bool
    original_head_dim: int
    key_tile_tokens: int
    is_causal: bool
    route_original_basis: bool


@dataclass(frozen=True, slots=True)
class PrequantizedSlaAttention:
    query_int8: torch.Tensor
    query_scale: torch.Tensor
    key_int8: torch.Tensor
    key_scale: torch.Tensor
    value: Optional[torch.Tensor]
    value_int8: torch.Tensor
    value_scale: torch.Tensor
    route_words: torch.Tensor
    sparse_query_blocks: torch.Tensor
    output_dtype: torch.dtype
    sm_scale: float
    sparsity_ratio: float
    possible_blocks: int
    use_w8a8: bool
    original_head_dim: int
    key_tile_tokens: int


__all__ = [
    "PrequantizedQK",
    "PrequantizedSageAttention",
    "PrequantizedSlaAttention",
    "PrequantizedSolAttention",
]
