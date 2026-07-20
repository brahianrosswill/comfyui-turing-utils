from __future__ import annotations

import random
import sys
import unittest
from collections import Counter
from pathlib import Path


CALIBRATION_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "calibration"
sys.path.insert(0, str(CALIBRATION_ROOT))

import build_bernini_manifest as manifest  # noqa: E402
import collect_bernini_stats as collector  # noqa: E402


class CalibrationManifestTest(unittest.TestCase):
    def test_release_quotas_and_window_constraints(self):
        rows = manifest.build_rows(
            1024,
            split="calib",
            videos=["source.mp4", "reference.mp4"],
            images=[f"image-{index}.png" for index in range(5)],
            prompts=[{"prompt": "test", "source": "unit", "task_type": "v2v"}],
            rng=random.Random(1234),
        )
        self.assertEqual(Counter(row["resolution_bucket"] for row in rows), {"480": 256, "540": 192, "720": 320, "1080": 256})
        self.assertEqual(Counter(row["frame_bucket"] for row in rows), manifest.FRAME_QUOTAS)
        self.assertEqual(Counter(row["conditioning_signature"] for row in rows), manifest.CONDITIONING_QUOTAS)
        self.assertEqual(Counter(row["window_role"] for row in rows), manifest.WINDOW_ROLE_QUOTAS)
        self.assertEqual(Counter(row["motion_bucket"] for row in rows), manifest.MOTION_QUOTAS)
        self.assertEqual(Counter(row["aspect_bucket"] for row in rows), {"landscape": 640, "portrait": 256, "square": 128})
        for bucket, count in {"480": 256, "540": 192, "720": 320, "1080": 256}.items():
            variants = Counter(row["size_variant"] for row in rows if row["resolution_bucket"] == bucket)
            self.assertEqual(variants["nearby_aligned"], round(count * 0.2))
            self.assertEqual(variants["standard"], count - round(count * 0.2))
        for row in rows:
            if row["window_role"] in {"single_full", "short_video"}:
                self.assertLessEqual(row["frame_bucket"], row["context_window_size"])
        collector.validate_manifest(rows)

    def test_validation_resolution_quota(self):
        rows = manifest.build_rows(
            256,
            split="validation",
            videos=["source.mp4", "reference.mp4"],
            images=[f"image-{index}.png" for index in range(5)],
            prompts=[{"prompt": "test", "source": "unit", "task_type": "v2v"}],
            rng=random.Random(5678),
        )
        self.assertEqual(Counter(row["resolution_bucket"] for row in rows), manifest.VALIDATION_RESOLUTION_QUOTAS)


if __name__ == "__main__":
    unittest.main()
