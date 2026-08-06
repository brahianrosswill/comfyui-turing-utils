# svdint4-kernel

CUDA/PyTorch extension used by ComfyUI SVDInt4.

In addition to the SVDQuant path, version 0.6.2 provides exact-sm75 packed W4A8,
W8/W4 staged and BF16 row-buffer activation fusions, and a self-contained
Sage attention backend. Bundled `sage` uses per-warp INT8 Q/K and direct FP32
probability/value accumulation without the standalone `sageattention` package.
Unstable Sage1/Sage2 adaptations are excluded from the production package and
default CUDA build.

Most users should install it from the parent `comfyui-svdint4` directory:

```bash
python -m pip install -v --no-build-isolation -e ./kernel
```

The pip distribution is named `svdint4-kernel`; the Python package is imported
as `svdint4`.

## Source layout

```text
csrc/
  bindings.cpp, kernel_api.h             public binding declarations
  tensor_bridge.h                        tensor/stream bridge used by CUDA entry points
  svdint4/                                SVDInt4 and SVD correction kernels
  turing/
    convrot_quant.cu                      staged/row-buffer W8 and W4 ConvRot quantizers
    segmented_rms_adaln.cu                affine segmented RMSNorm + AdaLN
    w4a8.cu                               packed Turing W4A8 kernel
    sage/                                 bundled production SM75 Sage kernel
svdint4/
  ops.py                                  stable SVDInt4/W4A8 Python API
  turing_sage/                            lazy production Sage Python facade
experiments/
  turing_sage_variants/                   source-only research checkpoint guide
```

`tensor_bridge.h` is the small tensor/stream ABI bridge shared by the CUDA
entry points; it is not a Python runtime layer or a separate subsystem.
The two bundled Sage extensions are built only when `SVDINT4_ARCH_LIST`
contains `7.5`. Release builds generate only sm75 Sage cubins. For compatible
GPU validation, `SVDINT4_ARCH_LIST="7.5+PTX"` also embeds compute_75 PTX and
deliberately selects the Turing CTA schedule after the newer GPU JITs that PTX.
This prevents compilation from introducing newer-architecture instructions
and does not change runtime dispatch in the ComfyUI plugin. The
extensions are never JIT-compiled by the plugin.

## Requirements

- Python 3.10 or newer
- PyTorch with CUDA
- CUDA toolkit with `nvcc`
- C++20-capable compiler
- NVIDIA GPU, `sm_75` or newer

## Build

From this `kernel/` directory:

```bash
python -m pip install -v --no-build-isolation -e .
```

By default the extension builds for `sm_75`, `sm_80`, `sm_86`, and `sm_89`.
Override this with:

```bash
SVDINT4_ARCH_LIST="8.0;8.6" \
python -m pip install -v --no-build-isolation -e .
```

To build bundled Sage for an A40 compatibility run while retaining the
required Turing target:

```bash
SVDINT4_ARCH_LIST="7.5+PTX" \
python -m pip install -v --no-build-isolation -e .
```

Then run the numerical matrix, optionally with ConvRot timings:

```bash
python scripts/validate_compatible.py --device cuda:0 --benchmark
```

The removed Sage1/Sage2 experiments, precision comparator and benchmark
breakdowns are reproducible from the checkpoint documented in
`experiments/turing_sage_variants/README.md`; they are intentionally not
installed by this package.

If `nvcc` selects the wrong host compiler:

```bash
CXX=/path/to/g++ SVDINT4_CUDAHOSTCXX=/path/to/g++ \
python -m pip install -v --no-build-isolation -e .
```

On Windows, run from an x64 Visual Studio Developer shell. The binding avoids
heavy `torch/extension.h` and `ATen/cuda/CUDAContext.h` headers so split CUDA
environments do not need cuSPARSE development headers just to compile the
Python extension wrapper.

NVIDIA's CUDA 12.8 Conda packages place CCCL under
`%CONDA_PREFIX%\Library\include\targets\x64`. The build adds that directory
automatically when it contains `nv\target`. If the header is absent, install
the matching package before rebuilding:

```bat
conda install -c nvidia cuda-cccl=12.8.90
```

For a custom location, set `SVDINT4_CCCL_INCLUDE_DIR` to the directory that
directly contains `nv\target`.

If PyTorch prints `Error checking compiler version for cl`, MSVC is likely
emitting localized diagnostics that PyTorch cannot decode with the active
Windows code page. The build script sets `VSLANG=1033` automatically on Windows.
You can also set it explicitly before installing:

```powershell
$env:VSLANG = "1033"
python -m pip install -v --no-build-isolation -e .
```

## Check

```bash
python - <<'PY'
import torch
import svdint4
from svdint4.ops import svd_int4_linear

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("svdint4:", svdint4.__file__)
print("kernel api:", callable(svd_int4_linear))
print("Turing Sage family:", svdint4.turing_sage.available())
print("Turing W4A8 api:", callable(svdint4.turing_w4a8_linear))
print("Turing fused SwiGLU api:", callable(svdint4.turing_swiglu_int8_convrot_quantize))
print("Turing fused W4 SwiGLU api:", callable(svdint4.turing_swiglu_int4_convrot_quantize))
print("Turing RMSNorm+AdaLN api:", callable(svdint4.turing_segmented_rms_adaln))
PY
```

## License

Apache-2.0. See `LICENSE`, `NOTICE`, and `LICENSES/`.
