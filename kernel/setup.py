import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    # Keep MSVC diagnostics in English so PyTorch's compiler probe can decode
    # cl.exe output reliably on non-English Windows installations.
    os.environ.setdefault("VSLANG", "1033")

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


def _windows_cccl_include_dirs() -> list[str]:
    """Locate the target-specific CCCL headers omitted from some Conda nvcc paths."""
    if not IS_WINDOWS:
        return []

    candidates: list[Path] = []
    override = os.environ.get("SVDINT4_CCCL_INCLUDE_DIR")
    if override:
        candidates.append(Path(override))

    prefixes = [Path(sys.prefix)]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefixes.insert(0, Path(conda_prefix))
    for prefix in prefixes:
        candidates.extend(
            [
                prefix / "Library" / "include" / "targets" / "x64",
                prefix / "Library" / "include" / "cccl",
                prefix / "Library" / "include",
                prefix / "include" / "cccl",
                prefix / "include",
            ]
        )

    cuda_roots: list[Path] = []
    for value in (CUDA_HOME, os.environ.get("CUDA_PATH")):
        if value:
            cuda_roots.append(Path(value))
    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_roots.append(Path(nvcc).resolve().parent.parent)
    for cuda_root in cuda_roots:
        candidates.extend(
            [
                cuda_root / "include" / "targets" / "x64",
                cuda_root / "include" / "cccl",
                cuda_root / "include",
                cuda_root / "targets" / "x64" / "include",
            ]
        )

    for entry in sys.path:
        if entry:
            candidates.append(Path(entry) / "nvidia" / "cuda_cccl" / "include")

    found: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "nv" / "target").is_file():
            found.append(str(candidate.resolve()))
    if found:
        return found

    raise RuntimeError(
        "CUDA CCCL headers are missing or undiscoverable: nv/target was not found. "
        "For a CUDA 12.8 Conda environment on Windows, install them with "
        "`conda install -c nvidia cuda-cccl=12.8.90`, or set "
        "SVDINT4_CCCL_INCLUDE_DIR to the directory containing nv/target."
    )


def _arch_list() -> str:
    value = os.environ.get("SVDINT4_ARCH_LIST", "7.5;8.0;8.6;8.9")
    arches = []
    for raw in value.replace(",", ";").split(";"):
        arch = raw.strip()
        if not arch:
            continue
        if arch in {"75", "80", "86", "89"}:
            arch = f"{arch[0]}.{arch[1]}"
        arches.append(arch)
    return ";".join(arches)


ARCH_LIST = _arch_list()
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", ARCH_LIST)
CCCL_INCLUDE_DIRS = _windows_cccl_include_dirs()

COMMON_DEFINES = [
    "-DENABLE_BF16=1",
    "-DSVDINT4_MINIMAL=1",
]

HOST_CXX_STANDARD = os.environ.get(
    "SVDINT4_HOST_CXX_STANDARD",
    os.environ.get("SVDINT4_CXX_STANDARD", "c++20" if IS_WINDOWS else "c++2a"),
)
NVCC_CXX_STANDARD = os.environ.get("SVDINT4_NVCC_CXX_STANDARD", "c++20")

NVCC_FLAGS = [
    *COMMON_DEFINES,
    f"-std={NVCC_CXX_STANDARD}",
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--ptxas-options=--allow-expensive-optimizations=true",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_HALF2_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
]

if IS_WINDOWS:
    NVCC_FLAGS.extend(
        [
            "--use-local-env",
            "-Xcompiler",
            "/MD",
            "-Xcompiler",
            "/O2",
            "-Xcompiler",
            "/EHsc",
            "-Xcompiler",
            "/bigobj",
            "-Xcompiler",
            "/FS",
            "-Xcompiler",
            "/DNOMINMAX",
            "-Xcompiler",
            "/DWIN32_LEAN_AND_MEAN",
        ]
    )

CUDAHOSTCXX = os.environ.get("SVDINT4_CUDAHOSTCXX") or os.environ.get("CUDAHOSTCXX")
if CUDAHOSTCXX:
    NVCC_FLAGS.extend(["-ccbin", CUDAHOSTCXX])

if IS_WINDOWS:
    std_flag = HOST_CXX_STANDARD if HOST_CXX_STANDARD.startswith("/std:") else f"/std:{HOST_CXX_STANDARD}"
    CXX_FLAGS = [
        "/DENABLE_BF16=1",
        "/DSVDINT4_MINIMAL=1",
        std_flag,
        "/O2",
        "/EHsc",
        "/MD",
        "/bigobj",
        "/FS",
        "/DNOMINMAX",
        "/DWIN32_LEAN_AND_MEAN",
        "/wd4251",
        "/wd4275",
        "/wd4819",
    ]
else:
    CXX_FLAGS = [
        *COMMON_DEFINES,
        f"-std={HOST_CXX_STANDARD}",
        "-O3",
        "-fvisibility=hidden",
    ]


core_ext = CUDAExtension(
    name="svdint4._C",
    sources=[
        "csrc/bindings.cpp",
        "csrc/svdint4/dispatcher.cu",
        "csrc/svdint4/gemm_instantiations.cu",
        "csrc/turing/bf16_epilogue.cu",
        "csrc/turing/convrot_quant.cu",
        "csrc/turing/segmented_rms_adaln.cu",
        "csrc/turing/w4a8.cu",
    ],
    include_dirs=[
        str((ROOT / "csrc").resolve()),
        *CCCL_INCLUDE_DIRS,
    ],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)


def _includes_sm75() -> bool:
    return any(arch.split("+")[0] == "7.5" for arch in ARCH_LIST.split(";") if arch)


def _includes_sm75_ptx() -> bool:
    return any(arch.upper() == "7.5+PTX" for arch in ARCH_LIST.split(";") if arch)


ext_modules = [core_ext]
if _includes_sm75():
    sm75_nvcc_flags = [
        *NVCC_FLAGS,
        "--use_fast_math",
        "-gencode",
        "arch=compute_75,code=sm_75",
    ]
    if _includes_sm75_ptx():
        # A newer validation GPU JITs compute_75 PTX, so compilation cannot
        # introduce Ampere instructions or BF16 hardware paths.
        sm75_nvcc_flags.extend(
            [
                "-gencode",
                "arch=compute_75,code=compute_75",
            ]
        )
    sage_include_dirs = [
        str((ROOT / "csrc" / "turing").resolve()),
        str((ROOT / "csrc" / "turing" / "sage").resolve()),
        *CCCL_INCLUDE_DIRS,
    ]
    ext_modules.extend(
        [
            CUDAExtension(
                name="svdint4._sage_qattn_sm75",
                sources=[
                    "csrc/turing/sage/pybind_sm75.cpp",
                    "csrc/turing/sage/qk_int_sv_f16_cuda_sm75.cu",
                    "csrc/turing/sage/qk_int_sv_f16_varlen_cuda_sm75.cu",
                ],
                include_dirs=sage_include_dirs,
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": sm75_nvcc_flags},
            ),
            CUDAExtension(
                name="svdint4._sage_fused_sm75",
                sources=[
                    "csrc/turing/sage/pybind_fused.cpp",
                    "csrc/turing/sage/fused.cu",
                ],
                include_dirs=sage_include_dirs,
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": sm75_nvcc_flags},
            ),
        ]
    )


setup(
    name="svdint4-kernel",
    version="0.6.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
