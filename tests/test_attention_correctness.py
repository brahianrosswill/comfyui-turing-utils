import sys
from pathlib import Path
import unittest

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = PLUGIN_ROOT / "kernel"
sys.path.insert(0, str(KERNEL_ROOT))

from comfyui_turing_utils_kernel.turing_sage.correctness import (  # noqa: E402
    attention_error_metrics,
    require_attention_correctness,
)


class AttentionCorrectnessGateTest(unittest.TestCase):
    def test_metrics_accept_identical_cpu_tensors(self):
        reference = torch.tensor(((1.0, -2.0), (0.5, 4.0)))
        result = attention_error_metrics(reference.clone(), reference)
        self.assertTrue(result.finite)
        self.assertEqual(result.max_abs, 0.0)
        self.assertEqual(result.relative_l2, 0.0)
        self.assertAlmostEqual(result.cosine, 1.0, places=6)
        require_attention_correctness(
            result, max_abs=0.0, relative_l2=0.0, cosine=0.999
        )

    def test_gate_reports_every_failed_bound(self):
        result = attention_error_metrics(
            torch.tensor((1.0, float("nan"))),
            torch.tensor((0.0, 1.0)),
            candidate_name="sol",
            reference_name="sage",
            selected_blocks=3,
            possible_blocks=4,
        )
        with self.assertRaisesRegex(AssertionError, "non-finite output") as raised:
            require_attention_correctness(
                result, max_abs=0.1, relative_l2=0.1, cosine=0.99
            )
        self.assertIn("selected 3/4", str(raised.exception))

    def test_shape_mismatch_fails_before_reduction(self):
        with self.assertRaisesRegex(AssertionError, "shape"):
            attention_error_metrics(torch.zeros(2), torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
