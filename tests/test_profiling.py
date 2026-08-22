from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils import profiling  # noqa: E402
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

    def test_runtime_metadata_distinguishes_native_cubin_from_ptx_fallback(self):
        with (
            mock.patch.object(
                profiling,
                "attention_kernel_architectures",
                return_value=("sm75+ptx", "sm86"),
            ),
            mock.patch.object(profiling, "attention_runtime_profile_schema", return_value=1),
            mock.patch.object(profiling, "kernel_version", return_value="0.31.0"),
            mock.patch.object(profiling.torch.cuda, "current_device", return_value=0),
            mock.patch.object(
                profiling.torch.cuda,
                "get_device_capability",
                return_value=(8, 6),
            ),
            mock.patch.object(
                profiling.torch.cuda,
                "get_device_name",
                return_value="NVIDIA GeForce RTX 3070",
            ),
        ):
            result = profiling._runtime_profile_metadata()

        self.assertEqual(result["device_sm"], "sm86")
        self.assertEqual(result["compiled_attention"], "sm75+ptx,sm86")
        self.assertTrue(result["native_arch"])
        self.assertEqual(result["profile_schema"], 1)


if __name__ == "__main__":
    unittest.main()
