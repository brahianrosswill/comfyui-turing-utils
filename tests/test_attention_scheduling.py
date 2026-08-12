from __future__ import annotations

import sys
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "kernel"))

from comfyui_turing_utils_kernel.turing_sage.scheduling import (  # noqa: E402
    estimate_attention_schedule,
)


class AttentionSchedulingTest(unittest.TestCase):
    def test_h3_many_wave_grid_rejects_persistent_and_split_k(self):
        result = estimate_attention_schedule(
            query_tokens=52842,
            key_tokens=52842,
            batch=1,
            query_heads=56,
            head_dim=128,
            sm_count=72,
            route_density_min=0.237,
            route_density_max=0.322,
        )
        self.assertGreater(result.waves, 300)
        self.assertFalse(result.persistent_queue_useful)
        self.assertFalse(result.split_k_candidate)
        self.assertGreater(result.split_k_workspace_bytes, 2 * 1024**3)

    def test_short_query_long_key_is_split_k_candidate(self):
        result = estimate_attention_schedule(
            query_tokens=64,
            key_tokens=52842,
            batch=1,
            query_heads=56,
            head_dim=128,
            sm_count=72,
        )
        self.assertTrue(result.split_k_candidate)
        self.assertFalse(result.persistent_queue_useful)

    def test_extreme_sparse_variance_can_trigger_queue_experiment(self):
        result = estimate_attention_schedule(
            query_tokens=64,
            key_tokens=8192,
            batch=1,
            query_heads=8,
            head_dim=64,
            sm_count=72,
            route_density_min=0.1,
            route_density_max=0.9,
        )
        self.assertTrue(result.persistent_queue_useful)


if __name__ == "__main__":
    unittest.main()
