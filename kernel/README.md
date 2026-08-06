# svdint4-kernel

Separately installed CUDA/PyTorch extension for the ComfyUI plugin's exact-sm75
runtime. Version 0.7.0 contains packed W4A8 Tensor Core GEMM, W8/W4 ConvRot
activation quantizers, BF16 epilogues, fused normalization, and bundled Sage
attention. It contains no model-weight format or model loader.

## Install

```bash
python -m pip install -v --no-build-isolation -e ./kernel
```

The pip distribution is named `svdint4-kernel`; the Python package is imported
as `svdint4`. The extension is installed independently so Python-only custom
node updates do not compile CUDA code.

## Source layout

```text
csrc/
  bindings.cpp, kernel_api.h             public binding declarations
  tensor_bridge.h                        tensor/stream bridge used by CUDA entry points
  turing/
    convrot_quant.cu                      staged/row-buffer W8 and W4 ConvRot quantizers
    segmented_rms_adaln.cu                affine segmented RMSNorm + AdaLN
    w4a8.cu                               packed W4-to-S8 SM75 Tensor Core GEMM
    sage/                                 bundled production SM75 Sage kernel
svdint4/
  ops.py                                  stable Turing operator API
  turing_sage/                            lazy production Sage facade
experiments/
  turing_sage_variants/                   source-only research checkpoint guide
```

## Build configuration

CUTLASS headers are discovered from Conda on Linux or Windows, the CUDA
toolkit, NVIDIA's `nvidia-cutlass` package, or a configured checkout. If none
is present, the build downloads a pinned NVIDIA wheel and verifies its SHA256.

The default build targets `sm_75`, `sm_80`, `sm_86`, and `sm_89`. Override it
with `SVDINT4_ARCH_LIST`. Bundled Sage is built only when the list contains
`7.5`; `7.5+PTX` permits compatible A40 validation without introducing
Ampere-only instructions.

```bash
SVDINT4_ARCH_LIST="7.5+PTX" \
python -m pip install -v --no-build-isolation -e ./kernel
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark
```

On Windows, use an x64 Visual Studio Developer shell. CUDA 12.8 Conda users
must have the CCCL directory containing `nv/target`; the build discovers
`%CONDA_PREFIX%\Library\include\targets\x64` automatically.

## Check

```bash
python - <<'PY'
import svdint4

print("kernel:", svdint4.__file__)
print("Turing Sage:", svdint4.turing_sage.available())
print("Turing W4A8:", callable(svdint4.turing_w4a8_linear))
print("Turing SwiGLU:", callable(svdint4.turing_swiglu_int8_convrot_quantize))
print("Turing norm:", callable(svdint4.turing_segmented_rms_adaln))
PY
```

## License

Apache-2.0. See `LICENSE`, `NOTICE`, and `LICENSES/`.
