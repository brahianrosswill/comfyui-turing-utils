from __future__ import annotations

import os
import platform
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SETUP_PATH = PLUGIN_ROOT / "kernel" / "setup.py"


class KernelSetupTest(unittest.TestCase):
    def _run_windows_setup(self, conda_prefix: Path):
        def extension(*, name, **kwargs):
            return SimpleNamespace(name=name, kwargs=kwargs)

        environment = {
            "CONDA_PREFIX": str(conda_prefix),
            "SVDINT4_ARCH_LIST": "8.6",
        }
        with (
            mock.patch.object(platform, "system", return_value="Windows"),
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch("torch.utils.cpp_extension.CUDA_HOME", None),
            mock.patch("torch.utils.cpp_extension.CUDAExtension", side_effect=extension),
            mock.patch("torch.utils.cpp_extension.BuildExtension", object()),
            mock.patch("shutil.which", return_value=None),
            mock.patch("setuptools.setup") as setup,
        ):
            runpy.run_path(str(SETUP_PATH), run_name="__svdint4_windows_setup_test__")
        return setup.call_args.kwargs["ext_modules"]

    def test_windows_conda_target_specific_cccl_path_is_added(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            cccl = prefix / "Library" / "include" / "targets" / "x64"
            (cccl / "nv").mkdir(parents=True)
            (cccl / "nv" / "target").touch()

            extensions = self._run_windows_setup(prefix)

        self.assertEqual([extension.name for extension in extensions], ["svdint4._C"])
        self.assertIn(str(cccl.resolve()), extensions[0].kwargs["include_dirs"])

    def test_windows_missing_cccl_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp_dir, self.assertRaisesRegex(
            RuntimeError, "cuda-cccl=12.8.90"
        ):
            self._run_windows_setup(Path(temp_dir))


if __name__ == "__main__":
    unittest.main()
