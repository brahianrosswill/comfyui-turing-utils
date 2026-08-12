from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.profiling import CudaPhaseProfiler  # noqa: E402


class CudaPhaseProfilerTest(unittest.TestCase):
    def test_disabled_profiler_creates_no_cuda_events(self):
        profiler = CudaPhaseProfiler(0)
        function = mock.Mock(return_value="output")
        with mock.patch("torch.cuda.Event") as event:
            output = profiler.call("phase", function, 1, keyword=2)

        self.assertEqual(output, "output")
        function.assert_called_once_with(1, keyword=2)
        event.assert_not_called()
        self.assertFalse(profiler.records)


if __name__ == "__main__":
    unittest.main()
