from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "kernel"))

with mock.patch.dict(
    sys.modules,
    {
        "comfyui_turing_utils_kernel._C": SimpleNamespace(),
        "comfyui_turing_utils_kernel._sage_fused_sm75": SimpleNamespace(),
        "comfyui_turing_utils_kernel._sage_qattn_sm75": SimpleNamespace(),
    },
):
    from comfyui_turing_utils_kernel.turing_sage.core import (  # noqa: E402
        _frame_sparse_schedule_cpu,
    )


class FrameSparseScheduleTest(unittest.TestCase):
    def test_dense_frame_policy_selects_every_key_block(self):
        row_offsets, key_blocks, density = _frame_sparse_schedule_cpu(
            key_length=384,
            topology_start_tokens=64,
            topology_tokens=320,
            tokens_per_frame=64,
            prefix_tokens=64,
            temporal_window_frames=4,
            global_anchor_stride=1,
            global_anchor_offset=0,
            sink_frames=1,
        )
        self.assertEqual(row_offsets, [0, 6, 12, 18, 24, 30])
        for row in range(5):
            self.assertEqual(key_blocks[row * 6 : (row + 1) * 6], list(range(6)))
        self.assertEqual(density, 1.0)

    def test_non_aligned_frames_keep_prefix_local_and_rotated_anchors(self):
        row_offsets, key_blocks, density = _frame_sparse_schedule_cpu(
            key_length=984,
            topology_start_tokens=64,
            topology_tokens=920,
            tokens_per_frame=184,
            prefix_tokens=64,
            temporal_window_frames=0,
            global_anchor_stride=3,
            global_anchor_offset=1,
            sink_frames=1,
        )
        self.assertEqual(len(row_offsets), (920 + 63) // 64 + 1)
        self.assertGreater(density, 0.0)
        self.assertLess(density, 1.0)
        for start, end in zip(row_offsets, row_offsets[1:]):
            row = key_blocks[start:end]
            self.assertEqual(row, sorted(set(row)))
            self.assertIn(0, row)
            # Sink frame 0 and rotated anchor frame 1 straddle these blocks.
            self.assertTrue({1, 2}.issubset(row))
            self.assertTrue({3, 4, 5, 6}.intersection(row))

    def test_invalid_partial_frame_topology_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete token frames"):
            _frame_sparse_schedule_cpu(
                key_length=1000,
                topology_start_tokens=64,
                topology_tokens=936,
                tokens_per_frame=184,
                prefix_tokens=64,
                temporal_window_frames=1,
                global_anchor_stride=4,
                global_anchor_offset=0,
                sink_frames=1,
            )

    def test_radial_schedule_keeps_near_frames_and_samples_distant_spatial_tiles(self):
        row_offsets, key_blocks, density = _frame_sparse_schedule_cpu(
            key_length=1088,
            topology_start_tokens=64,
            topology_tokens=1024,
            tokens_per_frame=256,
            prefix_tokens=64,
            temporal_window_frames=0,
            global_anchor_stride=0,
            global_anchor_offset=0,
            sink_frames=0,
            sparse_pattern="radial",
            spatial_tokens_height=16,
            spatial_tokens_width=16,
            radial_spatial_radius=0,
            radial_max_temporal_stride=16,
        )
        first_row = key_blocks[row_offsets[0] : row_offsets[1]]
        self.assertTrue(set(range(0, 7)).issubset(first_row))
        self.assertTrue({9, 10}.issubset(first_row))
        self.assertTrue({7, 8}.isdisjoint(first_row))
        self.assertTrue(set(range(13, 17)).isdisjoint(first_row))
        self.assertGreater(density, 0.0)
        self.assertLess(density, 1.0)

    def test_radial_schedule_requires_exact_spatial_topology(self):
        with self.assertRaisesRegex(ValueError, "exact spatial token"):
            _frame_sparse_schedule_cpu(
                key_length=1088,
                topology_start_tokens=64,
                topology_tokens=1024,
                tokens_per_frame=256,
                prefix_tokens=64,
                temporal_window_frames=1,
                global_anchor_stride=0,
                global_anchor_offset=0,
                sink_frames=0,
                sparse_pattern="radial",
                spatial_tokens_height=15,
                spatial_tokens_width=16,
            )


if __name__ == "__main__":
    unittest.main()
