"""Cached, kernel-independent Sol and SLA route policy construction."""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock

import torch


SOL_POLICY_CACHE_LIMIT = 64
SOL_POLICY_CACHE: OrderedDict[
    tuple, tuple[torch.Tensor, torch.Tensor, int]
] = OrderedDict()
SOL_POLICY_CACHE_LOCK = Lock()


def normalize_token_ranges(
    ranges,
    sequence_length: int,
) -> tuple[tuple[int, int], ...]:
    normalized = []
    for item in ranges or ():
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("Sol policy ranges must contain (start, stop) pairs")
        start, stop = (int(item[0]), int(item[1]))
        if start < 0 or stop <= start or stop > sequence_length:
            raise ValueError("Sol policy range is outside the attention sequence")
        normalized.append((start, stop))
    normalized.sort()
    for previous, current in zip(normalized, normalized[1:]):
        if current[0] < previous[1]:
            raise ValueError("Sol policy ranges must not overlap")
    return tuple(normalized)


def sol_block_policy(
    device: torch.device,
    query_length: int,
    key_length: int,
    dense_query_ranges,
    exact_kv_ranges,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    dense_ranges = normalize_token_ranges(dense_query_ranges, query_length)
    exact_ranges = normalize_token_ranges(exact_kv_ranges, key_length)
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    cache_key = (
        device.type,
        device_index,
        query_length,
        key_length,
        dense_ranges,
        exact_ranges,
    )
    with SOL_POLICY_CACHE_LOCK:
        cached = SOL_POLICY_CACHE.get(cache_key)
        if cached is not None:
            SOL_POLICY_CACHE.move_to_end(cache_key)
            return cached

    query_blocks = (query_length + 63) // 64
    key_blocks = (key_length + 63) // 64
    sparse_query = torch.ones(query_blocks, dtype=torch.uint8)
    exact_kv = torch.zeros(key_blocks, dtype=torch.uint8)
    for start, stop in dense_ranges:
        sparse_query[start // 64 : (stop + 63) // 64] = 0
    for start, stop in exact_ranges:
        exact_kv[start // 64 : (stop + 63) // 64] = 1
    sparse_count = int(sparse_query.sum().item())
    policy = (sparse_query.to(device), exact_kv.to(device), sparse_count)
    with SOL_POLICY_CACHE_LOCK:
        existing = SOL_POLICY_CACHE.get(cache_key)
        if existing is not None:
            return existing
        SOL_POLICY_CACHE[cache_key] = policy
        while len(SOL_POLICY_CACHE) > SOL_POLICY_CACHE_LIMIT:
            SOL_POLICY_CACHE.popitem(last=False)
    return policy


def sla_fixed_topk_indices(
    query_summary: torch.Tensor,
    key_summary: torch.Tensor,
    sparsity_ratio: float,
) -> torch.Tensor:
    """Select the fixed SLA K budget for every 128-token Query block."""
    if query_summary.ndim != 4 or key_summary.ndim != 4:
        raise ValueError("SLA Q/K summaries must be four-dimensional")
    if query_summary.size(0) != key_summary.size(0):
        raise ValueError("SLA Q/K summary batch sizes must match")
    if query_summary.size(-1) != key_summary.size(-1):
        raise ValueError("SLA Q/K summary head dimensions must match")
    query_heads = query_summary.size(1)
    key_heads = key_summary.size(1)
    if key_heads <= 0 or query_heads % key_heads:
        raise ValueError("SLA Query heads must be divisible by KV heads")
    key_blocks = key_summary.size(2)
    keep_blocks = min(
        key_blocks,
        max(1, int((1.0 - float(sparsity_ratio)) * key_blocks)),
    )
    groups = query_heads // key_heads
    grouped_query = query_summary.reshape(
        query_summary.size(0),
        key_heads,
        groups,
        query_summary.size(2),
        query_summary.size(3),
    )
    # Smooth-K shifts every score in a Query row by one constant, leaving
    # Top-K exactly invariant. Omitting it avoids two temporary tensors.
    scores = torch.matmul(
        grouped_query,
        key_summary.unsqueeze(2).transpose(-1, -2),
    ).reshape(
        query_summary.size(0),
        query_heads,
        query_summary.size(2),
        key_blocks,
    )
    return torch.topk(
        scores,
        keep_blocks,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices.to(torch.int32).contiguous()


__all__ = [
    "SOL_POLICY_CACHE",
    "normalize_token_ranges",
    "sla_fixed_topk_indices",
    "sol_block_policy",
]
