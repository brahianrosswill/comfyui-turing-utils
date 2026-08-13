from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.adapters.minimax.block_cache import (  # noqa: E402
    MiniMaxH3BlockCache,
    MiniMaxH3BlockCacheGroup,
    _relative_change,
    _sample_rms,
    _turbo_ranges,
)


def _options(schedule, sigma, **extra):
    return {
        "sample_sigmas": torch.tensor(schedule, dtype=torch.float32),
        "sigmas": torch.tensor([sigma], dtype=torch.float32),
        **extra,
    }


def _layout():
    return SimpleNamespace(
        segments=((0, 1, "audio"), (1, 2, "video"))
    )


class MiniMaxH3BlockCacheTest(unittest.TestCase):
    def test_recovered_scalar_helpers_match_reference_behavior(self):
        hidden = torch.arange(64 * 32, dtype=torch.float32).reshape(64, 32)
        expected = torch.sqrt(hidden[0:64:1, ::16].square().mean() + 1e-12)
        self.assertEqual(_sample_rms(hidden, (0, 64)), float(expected.item()))
        self.assertEqual(_sample_rms(hidden, (4, 4)), 0.0)
        self.assertEqual(_relative_change(0.0, 0.0), 1.0)
        self.assertEqual(_relative_change(1.0, 0.0), 99_999_999.0)
        self.assertEqual(_turbo_ranges(0), (0, 0))
        self.assertEqual(_turbo_ranges(1), (0, 1))
        self.assertEqual(_turbo_ranges(2), (1, 1))
        self.assertEqual(_turbo_ranges(50), (6, 44))

    def test_standard_cache_reproduces_skip_and_mcs_trajectory(self):
        cache = MiniMaxH3BlockCache(0.08, 0.1, 0.9, 2, "gpu", 50)
        cache.set_cache_ranges(_layout())
        schedule = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

        first = torch.tensor([[10.0], [20.0]])
        self.assertFalse(cache.prepare_middle(first, _options(schedule, 1.0)))
        cache.store_middle(first + torch.tensor([[1.0], [2.0]]))

        second = torch.tensor([[10.1], [20.1]])
        self.assertTrue(cache.prepare_middle(second, _options(schedule, 0.8)))
        self.assertTrue(torch.equal(second, torch.tensor([[11.1], [22.1]])))

        third = torch.tensor([[10.2], [20.2]])
        self.assertTrue(cache.prepare_middle(third, _options(schedule, 0.6)))
        fourth = torch.tensor([[10.3], [20.3]])
        self.assertFalse(cache.prepare_middle(fourth, _options(schedule, 0.4)))
        self.assertEqual(cache.reject_counts["mcs"], 1)
        self.assertEqual(cache.cache_hits, 2)
        self.assertEqual(cache.skipped_blocks, 100)

    def test_residual_forecast_matches_reference_factor(self):
        cache = MiniMaxH3BlockCache(0.5, 0.0, 1.0, 1, "gpu", 50)
        cache.set_cache_ranges(_layout())
        schedule = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

        first = torch.tensor([[10.0], [20.0]])
        self.assertFalse(cache.prepare_middle(first, _options(schedule, 1.0)))
        cache.store_middle(first + torch.tensor([[1.0], [2.0]]))
        second = torch.tensor([[10.1], [20.1]])
        self.assertTrue(cache.prepare_middle(second, _options(schedule, 0.9)))

        third = torch.tensor([[10.2], [20.2]])
        self.assertFalse(cache.prepare_middle(third, _options(schedule, 0.8)))
        cache.store_middle(third + torch.tensor([[1.1], [2.2]]))
        self.assertAlmostEqual(cache.residual_delta_score, 0.1, places=6)

        fourth = torch.tensor([[10.3], [20.3]])
        self.assertTrue(cache.prepare_middle(fourth, _options(schedule, 0.7)))
        expected = torch.tensor([[11.4175], [22.5350]])
        self.assertTrue(torch.allclose(fourth, expected, atol=1e-6, rtol=0.0))

    def test_turbo_state_uses_middle_blocks_and_segment_scales(self):
        cache = MiniMaxH3BlockCache(0.3, 0.2, 0.8, 1, "gpu", 50, True)
        cache.set_cache_ranges(_layout())
        schedule = [1.0, 0.75, 0.5, 0.25, 0.0]
        first = torch.tensor([[10.0], [20.0]])
        self.assertFalse(cache.prepare_middle(first, _options(schedule, 1.0)))
        cache.store_middle(first + torch.tensor([[1.0], [2.0]]))

        second = torch.tensor([[10.1], [20.1]])
        self.assertTrue(cache.prepare_middle(second, _options(schedule, 0.75)))
        self.assertEqual((cache.skip_start, cache.skip_end), (6, 44))
        self.assertTrue(
            torch.allclose(
                second,
                torch.tensor([[11.11], [22.11]]),
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_group_separates_sampler_branches_and_short_trajectory(self):
        group = MiniMaxH3BlockCacheGroup(0.08, 0.1, 0.9, 2, "auto", 50, True)
        short = {
            "sample_sigmas": torch.ones(5),
            "uuids": ["positive"],
        }
        state = group.state_for(short)
        self.assertTrue(state.turbo_mode)
        self.assertEqual(state.threshold, 0.3)
        self.assertEqual(state.max_consecutive_skips, 1)
        self.assertIs(state, group.state_for(short))
        self.assertIsNot(
            state,
            group.state_for(
                {"sample_sigmas": torch.ones(5), "uuids": ["negative"]}
            ),
        )

        long = group.state_for({"sample_sigmas": torch.ones(12)})
        self.assertFalse(long.turbo_mode)
        self.assertEqual(long.threshold, 0.08)

    def test_discontinuous_sigma_clears_cached_residual(self):
        cache = MiniMaxH3BlockCache(1.0, 0.0, 1.0, 5, "gpu", 50)
        cache.set_cache_ranges(_layout())
        schedule = [1.0, 0.8, 0.6, 0.4, 0.0]
        first = torch.tensor([[10.0], [20.0]])
        cache.prepare_middle(first, _options(schedule, 1.0))
        cache.store_middle(first + 0.1)
        second = torch.tensor([[10.01], [20.01]])
        self.assertTrue(cache.prepare_middle(second, _options(schedule, 0.8)))

        restarted = torch.tensor([[10.02], [20.02]])
        self.assertFalse(cache.prepare_middle(restarted, _options(schedule, 0.8)))
        self.assertEqual(cache.reject_counts["discontinuity"], 1)
        self.assertIsNone(cache.residual)
        self.assertTrue(math.isfinite(cache.last_sigma))

    def test_forced_full_run_matches_reference_rejection_reason(self):
        cache = MiniMaxH3BlockCache(1.0, 0.0, 1.0, 5, "gpu", 50)
        hidden = torch.tensor([[10.0], [20.0]])
        schedule = [1.0, 0.8, 0.6, 0.0]

        self.assertFalse(
            cache.prepare_middle(
                hidden,
                _options(schedule, 1.0),
                force_full=True,
            )
        )
        self.assertEqual(cache.reject_counts["patch_overlap"], 1)


if __name__ == "__main__":
    unittest.main()
