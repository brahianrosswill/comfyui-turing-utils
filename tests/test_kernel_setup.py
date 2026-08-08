from __future__ import annotations

import os
import platform
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SETUP_PATH = PLUGIN_ROOT / "kernel" / "setup.py"


class KernelSetupTest(unittest.TestCase):
    @staticmethod
    def _extension(*, name, **kwargs):
        return SimpleNamespace(name=name, kwargs=kwargs)

    def _run_windows_setup(self, conda_prefix: Path):
        environment = {
            "CONDA_PREFIX": str(conda_prefix),
            "COMFYUI_TURING_UTILS_ARCH_LIST": "8.6",
        }
        with (
            mock.patch.object(platform, "system", return_value="Windows"),
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch("torch.utils.cpp_extension.CUDA_HOME", None),
            mock.patch("torch.utils.cpp_extension.CUDAExtension", side_effect=self._extension),
            mock.patch("torch.utils.cpp_extension.BuildExtension", object()),
            mock.patch("shutil.which", return_value=None),
            mock.patch("setuptools.setup") as setup,
        ):
            runpy.run_path(str(SETUP_PATH), run_name="__turing_utils_windows_setup_test__")
        return setup.call_args.kwargs["ext_modules"]

    @staticmethod
    def _make_cutlass_headers(include_dir: Path) -> None:
        for relative in (
            "cutlass/cutlass.h",
            "cutlass/gemm/device/gemm_universal_adapter.h",
            "cutlass/epilogue/threadblock/fusion/visitors.hpp",
            "cute/tensor.hpp",
        ):
            header = include_dir / relative
            header.parent.mkdir(parents=True, exist_ok=True)
            header.touch()

    def test_sage_compute75_ptx_is_opt_in_and_keeps_sm75(self):
        environment = {
            "COMFYUI_TURING_UTILS_ARCH_LIST": "7.5+PTX",
        }
        with (
            mock.patch.object(platform, "system", return_value="Linux"),
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch("torch.utils.cpp_extension.CUDAExtension", side_effect=self._extension),
            mock.patch("torch.utils.cpp_extension.BuildExtension", object()),
            mock.patch("setuptools.setup") as setup,
        ):
            runpy.run_path(str(SETUP_PATH), run_name="__turing_utils_linux_setup_test__")

        extensions = setup.call_args.kwargs["ext_modules"]
        self.assertEqual(
            [extension.name for extension in extensions],
            ["comfyui_turing_utils_kernel._C", "comfyui_turing_utils_kernel._sage_qattn_sm75", "comfyui_turing_utils_kernel._sage_fused_sm75"],
        )
        flags = extensions[1].kwargs["extra_compile_args"]["nvcc"]
        self.assertIn("arch=compute_75,code=sm_75", flags)
        self.assertIn("arch=compute_75,code=compute_75", flags)
        self.assertNotIn("arch=compute_86,code=sm_86", flags)
        self.assertNotIn("-DCOMFYUI_TURING_UTILS_EXPERIMENTAL_SAGE_VARIANTS", flags)
        self.assertIn("csrc/turing/sage/sol_sparse_cuda_sm75.cu", extensions[1].kwargs["sources"])
        self.assertEqual(setup.call_args.kwargs["version"], "0.10.0")
        self.assertEqual(set(setup.call_args.kwargs["packages"]), {
            "comfyui_turing_utils_kernel",
            "comfyui_turing_utils_kernel.turing_sage",
        })

    def test_sparse_source_does_not_require_optional_cuda_library_headers(self):
        source = (
            PLUGIN_ROOT / "kernel" / "csrc" / "turing" / "sage" / "sol_sparse_cuda_sm75.cu"
        ).read_text(encoding="utf-8")
        self.assertIn('#include "torch_compat.h"', source)
        self.assertNotIn("ATen/cuda/CUDAContext", source)
        self.assertNotIn("CUDAGuard", source)
        self.assertNotIn("cusparse", source.lower())

    def test_windows_conda_target_specific_cccl_path_is_added(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            cccl = prefix / "Library" / "include" / "targets" / "x64"
            (cccl / "nv").mkdir(parents=True)
            (cccl / "nv" / "target").touch()
            conda_include = prefix / "Library" / "include"
            self._make_cutlass_headers(conda_include)

            extensions = self._run_windows_setup(prefix)

        self.assertEqual([extension.name for extension in extensions], ["comfyui_turing_utils_kernel._C"])
        self.assertEqual(extensions[0].kwargs["include_dirs"][1], str(conda_include.resolve()))
        self.assertIn(str(cccl.resolve()), extensions[0].kwargs["include_dirs"])
        self.assertIn("/std:c++20", extensions[0].kwargs["extra_compile_args"]["cxx"])
        self.assertIn("-std=c++20", extensions[0].kwargs["extra_compile_args"]["nvcc"])

    def test_nvidia_cutlass_python_package_is_auto_detected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            site_root = Path(temp_dir)
            cutlass_include = site_root / "cutlass_library" / "source" / "include"
            self._make_cutlass_headers(cutlass_include)
            environment = {"COMFYUI_TURING_UTILS_ARCH_LIST": "8.6"}
            with (
                mock.patch.object(platform, "system", return_value="Linux"),
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(sys, "path", [str(site_root), *sys.path]),
                mock.patch("torch.utils.cpp_extension.CUDAExtension", side_effect=self._extension),
                mock.patch("torch.utils.cpp_extension.BuildExtension", object()),
                mock.patch("setuptools.setup") as setup,
            ):
                runpy.run_path(str(SETUP_PATH), run_name="__turing_utils_cutlass_package_test__")

        core = setup.call_args.kwargs["ext_modules"][0]
        self.assertEqual(core.kwargs["include_dirs"][1], str(cutlass_include.resolve()))

    def test_windows_uses_cached_portable_cutlass_headers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            prefix = temp_root / "conda"
            cccl = prefix / "Library" / "include" / "targets" / "x64"
            (cccl / "nv").mkdir(parents=True)
            (cccl / "nv" / "target").touch()
            cutlass_include = (
                temp_root / "comfyui-turing-utils-cutlass-4.2.0.0" / "include"
            )
            self._make_cutlass_headers(cutlass_include)

            with (
                mock.patch.object(sys, "path", [str(temp_root / "empty-site")]),
                mock.patch("tempfile.gettempdir", return_value=temp_dir),
            ):
                extensions = self._run_windows_setup(prefix)

        core = extensions[0]
        self.assertEqual(core.kwargs["include_dirs"][1], str(cutlass_include.resolve()))
        self.assertIn(str(cccl.resolve()), core.kwargs["include_dirs"])

    def test_windows_missing_cccl_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp_dir, self.assertRaisesRegex(
            RuntimeError, "cuda-cccl=12.8.90"
        ):
            self._run_windows_setup(Path(temp_dir))


if __name__ == "__main__":
    unittest.main()
