from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import turing_profile


class _FakeEvent:
    clock = 0.0

    def __init__(self, enable_timing=True):
        self.timestamp = None

    def record(self):
        self.timestamp = type(self).clock
        type(self).clock += 1.0

    def synchronize(self):
        return None

    def elapsed_time(self, other):
        return other.timestamp - self.timestamp


class TuringProfileTest(unittest.TestCase):
    def setUp(self):
        turing_profile._reset_for_tests()
        _FakeEvent.clock = 0.0

    def test_two_windows_report_then_disable_automatically(self):
        value = SimpleNamespace(
            device=torch.device("cuda", 0),
            dtype=torch.bfloat16,
        )
        with (
            mock.patch.object(turing_profile, "BLOCKS_PER_WINDOW", 2),
            mock.patch.object(turing_profile, "WINDOW_COUNT", 2),
            mock.patch.object(turing_profile.torch.cuda, "Event", _FakeEvent),
            mock.patch.object(turing_profile.LOG, "warning") as warning,
        ):
            for _ in range(4):
                with turing_profile.profile_block(value):
                    with turing_profile.cuda_region("phase.attention", value):
                        with turing_profile.cuda_region(
                            "detail.attention.sage2", value
                        ):
                            pass

        state = turing_profile._PROFILES[0]
        self.assertEqual(state.windows, 2)
        self.assertFalse(state.enabled)
        self.assertEqual(state.samples, [])
        self.assertTrue(
            any(
                call.args
                and "capture complete" in str(call.args[0])
                for call in warning.call_args_list
            )
        )

    def test_regions_are_noops_outside_an_active_block(self):
        value = SimpleNamespace(
            device=torch.device("cuda", 0),
            dtype=torch.bfloat16,
        )
        with mock.patch.object(turing_profile.torch.cuda, "Event", _FakeEvent):
            with turing_profile.cuda_region("detail.unused", value):
                pass
        self.assertEqual(turing_profile._PROFILES[0].samples, [])


if __name__ == "__main__":
    unittest.main()
