#!/usr/bin/env python3
"""Report whether persistent or split-K scheduling is justified."""

from __future__ import annotations

import argparse

from comfyui_turing_utils_kernel.turing_sage.scheduling import estimate_attention_schedule


def _format_bytes(value: int) -> str:
    return f"{value / (1024 ** 3):.3f} GiB"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sm-count", type=int, default=72, help="Target GPU SM count")
    parser.add_argument("--ctas-per-sm", type=int, default=2)
    args = parser.parse_args()
    cases = (
        ("H3-480p-like", 52842, 52842, 56, 0.237, 0.322),
        ("H3-720p-like", 100483, 100483, 56, 0.145, 0.322),
        ("short-Q/long-K", 128, 52842, 56, 1.0, 1.0),
    )
    for name, q, k, heads, density_min, density_max in cases:
        result = estimate_attention_schedule(
            query_tokens=q,
            key_tokens=k,
            batch=1,
            query_heads=heads,
            head_dim=128,
            sm_count=args.sm_count,
            ctas_per_sm=args.ctas_per_sm,
            route_density_min=density_min,
            route_density_max=density_max,
            split_k=2,
        )
        print(
            f"{name}: Q={q} K={k} CTAs={result.query_ctas} waves={result.waves} "
            f"tail={result.tail_utilization:.3f} imbalance={result.route_imbalance:.2f}x "
            f"persistent={result.persistent_queue_useful} splitK={result.split_k_candidate} "
            f"splitK_workspace={_format_bytes(result.split_k_workspace_bytes)}\n"
            f"  {result.reason}"
        )


if __name__ == "__main__":
    main()
