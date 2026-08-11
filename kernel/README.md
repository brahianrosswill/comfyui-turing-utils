# comfyui-turing-utils-kernel

Separately installed CUDA/PyTorch extension for the ComfyUI plugin's exact-sm75
runtime. Version 0.17.0 contains packed W4A8 Tensor Core GEMM, W8/W4 ConvRot
activation quantizers with fused SwiGLU/tanh-GELU, BF16 epilogues, fused RMSNorm
and LayerNorm modulation, bundled Sage attention, and an explicitly selected
experimental model-independent sparse attention kernel with input-adaptive
centroid threshold routing, stable-Sage INT8 QK for selected blocks, compact
dense-Query/exact-KV modality masks, INT8-consistent routing, 32 KiB attention
CTAs, and one-by-64 or two-by-32 skipped residuals on one shared route. The
Q-to-K-centroid Tensor Core pass now drives both routing and skipped-block
online-softmax correction, so no duplicate Q/K centroid scan or full global
route map is materialized. The local neighborhood is fixed to the official
+/- one block. Original V means remain dedicated to value approximation.
It also adds an experimental static frame-sparse Sage path. Cached
head-independent CSR schedules provide complete-frame windows or a radial
policy with 8x8 spatial-token locality and logarithmic temporal sampling,
without online summaries or per-head routing. The kernel retains production
INT8 QK, FP16/BF16 V, and FP32 softmax/accumulation in a 32 KiB CTA. The tested
40 KiB Q128 CTA is intentionally excluded because it regressed throughput.
It contains no model-weight format or model loader.

## Install

```bash
python -m pip install -v --no-build-isolation -e ./kernel
```

The pip distribution is named `comfyui-turing-utils-kernel`; the Python package
is imported as `comfyui_turing_utils_kernel`. The extension is installed
independently so Python-only custom-node updates do not compile CUDA code.

## Source layout

```text
csrc/
  bindings.cpp, kernel_api.h             public binding declarations
  tensor_bridge.h                        tensor/stream bridge used by CUDA entry points
  turing/
    convrot_quant.cu                      staged/row-buffer W8 and W4 ConvRot quantizers
    segmented_rms_adaln.cu                RMSNorm/LayerNorm + AdaLN kernels
    w4a8.cu                               packed W4-to-S8 SM75 Tensor Core GEMM
    sage/                                 bundled stable and experimental sparse SM75 attention
comfyui_turing_utils_kernel/
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
with `COMFYUI_TURING_UTILS_ARCH_LIST`. Bundled Sage is built only when the list
contains `7.5`; `7.5+PTX` permits compatible A40 validation without introducing
Ampere-only instructions.

```bash
COMFYUI_TURING_UTILS_ARCH_LIST="7.5+PTX" \
python -m pip install -v --no-build-isolation -e ./kernel
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark --experimental-sparse
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark --experimental-frame-sparse
python kernel/scripts/validate_wan_fusions.py --device cuda:0
```

On Windows, use an x64 Visual Studio Developer shell. CUDA 12.8 Conda users
must have the CCCL directory containing `nv/target`; the build discovers
`%CONDA_PREFIX%\Library\include\targets\x64` automatically.
The extension uses C++20 by default. This is required for the bundled CUTLASS
W4A8 templates to compile reliably with NVCC and MSVC. The host and device
standards can be overridden with `COMFYUI_TURING_UTILS_HOST_CXX_STANDARD` and
`COMFYUI_TURING_UTILS_NVCC_CXX_STANDARD` when diagnosing a toolchain issue.

## Check

```bash
python - <<'PY'
import comfyui_turing_utils_kernel

print("kernel:", comfyui_turing_utils_kernel.__file__)
print("Turing Sage:", comfyui_turing_utils_kernel.turing_sage.available())
print("Turing sparse:", comfyui_turing_utils_kernel.turing_sage.sparse_available())
print("Turing frame sparse:", comfyui_turing_utils_kernel.turing_sage.frame_sparse_available())
print("Turing W4A8:", callable(comfyui_turing_utils_kernel.turing_w4a8_linear))
print("Turing SwiGLU:", callable(comfyui_turing_utils_kernel.turing_swiglu_int8_convrot_quantize))
print("Turing norm:", callable(comfyui_turing_utils_kernel.turing_segmented_rms_adaln))
PY
```

## License

Apache-2.0. See `LICENSE`, `NOTICE`, and `LICENSES/`.
