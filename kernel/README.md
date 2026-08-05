# svdint4-kernel

CUDA/PyTorch extension used by ComfyUI SVDInt4.

In addition to the SVDQuant path, version 0.3 provides exact-sm75 packed W4A8
and a self-contained SageAttention2 backend. The attention extensions support
FP16/BF16, GQA, HND/NHD, different Q/KV lengths, causal and variable-length
execution without the standalone `sageattention` package.

Most users should install it from the parent `comfyui-svdint4` directory:

```bash
python -m pip install -v --no-build-isolation -e ./kernel
```

The pip distribution is named `svdint4-kernel`; the Python package is imported
as `svdint4`.

## Source layout

```text
csrc/
  bindings.cpp, kernel_api.h, runtime.h   shared binding/ABI support
  svdint4/                                SVDInt4 and SVD correction kernels
  turing/
    w4a8.cu                               packed Turing W4A8 kernel
    sage2/                                bundled SM75 SageAttention2 kernels
svdint4/
  ops.py                                  stable SVDInt4/W4A8 Python API
  turing_sage2/                           lazy SageAttention2 Python facade
```

`runtime.h` is only the small tensor/stream ABI header used by the original
SVDInt4 CUDA subset; it is not a Python runtime layer or a separate subsystem.
The two SageAttention2 extensions are built only when `SVDINT4_ARCH_LIST`
contains `7.5`. They are never JIT-compiled by the ComfyUI plugin.

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

If `nvcc` selects the wrong host compiler:

```bash
CXX=/path/to/g++ SVDINT4_CUDAHOSTCXX=/path/to/g++ \
python -m pip install -v --no-build-isolation -e .
```

On Windows, run from an x64 Visual Studio Developer shell. The binding avoids
heavy `torch/extension.h` and `ATen/cuda/CUDAContext.h` headers so split CUDA
environments do not need cuSPARSE development headers just to compile the
Python extension wrapper.

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
print("Turing SageAttention2:", svdint4.turing_sage2.available())
print("Turing W4A8 api:", callable(svdint4.turing_w4a8_linear))
PY
```

## License

Apache-2.0. See `LICENSE`, `NOTICE`, and `LICENSES/`.
