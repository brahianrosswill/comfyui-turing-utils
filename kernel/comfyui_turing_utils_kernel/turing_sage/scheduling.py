"""Static scheduling model for SM75 attention experiments.

This module deliberately does not dispatch a kernel.  It is a release gate for
persistent-queue and split-K proposals: a proposal must demonstrate that the
existing Query grid cannot saturate the GPU and that its reduction workspace is
bounded before it is allowed into the runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass


QUERY_TILE_TOKENS = 64


@dataclass(frozen=True, slots=True)
class AttentionScheduleEstimate:
    query_ctas: int
    resident_ctas: int
    waves: int
    tail_utilization: float
    route_imbalance: float
    persistent_queue_useful: bool
    split_k_candidate: bool
    split_k_workspace_bytes: int
    reason: str


def estimate_attention_schedule(
    *,
    query_tokens: int,
    key_tokens: int,
    batch: int,
    query_heads: int,
    head_dim: int,
    sm_count: int,
    ctas_per_sm: int = 2,
    route_density_min: float = 1.0,
    route_density_max: float = 1.0,
    split_k: int = 2,
) -> AttentionScheduleEstimate:
    values = (query_tokens, key_tokens, batch, query_heads, head_dim, sm_count, ctas_per_sm)
    if any(int(value) <= 0 for value in values):
        raise ValueError("attention schedule dimensions must be positive")
    if split_k < 1:
        raise ValueError("split_k must be positive")
    if not 0.0 <= route_density_min <= route_density_max <= 1.0:
        raise ValueError("route densities must satisfy 0 <= min <= max <= 1")

    query_blocks = (int(query_tokens) + QUERY_TILE_TOKENS - 1) // QUERY_TILE_TOKENS
    query_ctas = int(batch) * int(query_heads) * query_blocks
    resident_ctas = int(sm_count) * int(ctas_per_sm)
    waves = (query_ctas + resident_ctas - 1) // resident_ctas
    tail = query_ctas - (waves - 1) * resident_ctas
    tail_utilization = tail / resident_ctas
    density_floor = max(float(route_density_min), 1.0 / max(1, (key_tokens + 63) // 64))
    route_imbalance = float(route_density_max) / density_floor

    # CUDA already assigns ordinary CTAs to the next available SM. A software
    # atomic queue is only interesting for a very small grid with extreme task
    # variance; it does not improve the many-wave H3 self-attention grid.
    persistent_useful = query_ctas < resident_ctas and route_imbalance >= 2.0

    # Split-K creates FP32 partial output plus row max/denominator for every
    # Query/head/split. This full-grid number is intentionally conservative and
    # makes the memory cost visible before any kernel work starts.
    workspace_elements = (
        int(batch)
        * int(query_heads)
        * int(query_tokens)
        * int(split_k)
        * (int(head_dim) + 2)
    )
    workspace_bytes = workspace_elements * 4
    split_candidate = query_ctas < resident_ctas and key_tokens > 4 * QUERY_TILE_TOKENS

    if split_candidate:
        reason = "short-Q/long-K underfills the Query grid; bounded split-K merits a separate experiment"
    elif waves >= 4:
        reason = "many Query waves already saturate hardware CTA scheduling; persistent/split-K adds overhead"
    else:
        reason = "the Query grid is adequate and does not justify split-K reduction state"
    return AttentionScheduleEstimate(
        query_ctas=query_ctas,
        resident_ctas=resident_ctas,
        waves=waves,
        tail_utilization=tail_utilization,
        route_imbalance=route_imbalance,
        persistent_queue_useful=persistent_useful,
        split_k_candidate=split_candidate,
        split_k_workspace_bytes=workspace_bytes,
        reason=reason,
    )
