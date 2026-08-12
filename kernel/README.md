# comfyui-turing-utils-kernel

Separately installed CUDA/PyTorch extension for the ComfyUI plugin's exact-sm75
runtime. Version 0.22.3 contains packed W4A8 Tensor Core GEMM, W8/W4 ConvRot
activation quantizers with fused SwiGLU/tanh-GELU, BF16 epilogues, fused RMSNorm
and LayerNorm modulation, bundled Sage attention, pure-INT8 W8A8 attention,
and an explicitly selected experimental model-independent sparse attention
kernel with input-adaptive
centroid threshold routing, stable-Sage INT8 QK for selected blocks, compact
dense-Query/exact-KV modality masks, INT8-consistent routing, native D64/D128
attention CTAs, and one-by-64 or two-by-32 skipped residuals on one shared route. The
Q-to-K-centroid Tensor Core pass now drives both routing and skipped-block
online-softmax correction, so no duplicate Q/K centroid scan or full global
route map is materialized. The local neighborhood is fixed to the official
+/- one block. Original V means remain dedicated to value approximation.
It contains no model-weight format or model loader.

Every stable public tensor operator is registered through
`torch.library.custom_op` with a fake/meta implementation: W4A8 GEMM, BF16
epilogue, ConvRot activation fusions, normalization fusions, fixed and varlen
Sage, dense W8A8, Sol, and fused Q/K RMSNorm+RoPE+INT8 preprocessing.
Prequantized Python state objects deliberately stay
outside this boundary because they are ComfyUI tensor-lifetime coordination,
not graph-level tensor operators.

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
    sage/                                 bundled dense/sparse attention and fused Q/K preprocessing
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
python kernel/scripts/validate_wan_fusions.py --device cuda:0
```

`--experimental-sparse` also runs the explicit correctness gate. A fully
selected Sol route is compared with stable Sage, while Sol-W8A8 is compared
with route-free W8A8. The gate checks finite output, maximum absolute error,
relative L2 error, cosine similarity, and exact selected-block coverage; it is
not imported or executed during normal inference.

On Windows, use an x64 Visual Studio Developer shell. CUDA 12.8 Conda users
must have the CCCL directory containing `nv/target`; the build discovers
`%CONDA_PREFIX%\Library\include\targets\x64` automatically.
The build selects its Windows language dialect from the actual CUDA toolkit
used by NVCC: CUDA 12 and newer use C++20, while older toolkits use C++17.
NVCC 12.0 was the first CUDA release with C++20 support, and CUTLASS's
EVT-based W4A8 epilogue instantiates more reliably under NVCC/MSVC in that
dialect. Linux stays on PyTorch's portable C++17 baseline, including with CUDA
12, so it does not unnecessarily require GCC 10+. Toolkit detection prefers
`nvcc --version` and toolkit metadata over PyTorch's compiled CUDA label. The
standards can be overridden with
`COMFYUI_TURING_UTILS_HOST_CXX_STANDARD` and
`COMFYUI_TURING_UTILS_NVCC_CXX_STANDARD` when diagnosing a toolchain issue.

Version 0.22 retains the split prequantize/execute attention ABI used by current
ComfyUI attention tensor containers. It releases the original Q/K/V storage
before allocating the output (W8A8 keeps no floating-point V copy), while the
one-call APIs remain available for older ComfyUI/plugin combinations. All
attention launches use PyTorch's current CUDA stream and can participate in
CUDA Graph capture. Sol accepts logical CTA-K64 or CTA-K128 scheduling;
CTA-K128 processes two consecutive 64-token stages while reusing the same
shared tile. Route-free dense W8A8 accepts either value for ABI compatibility
but deliberately normalizes both to its faster compile-time CTA-K64 loop.
Native D64 uses 16 KiB dynamic shared memory and
native D128 uses 32 KiB; inputs below either width pad only to the next native
specialization. Fused Hadamard Q/K rotation and adaptive K anchoring can be
disabled only through the explicit experimental tuning patch.
The new Q/K preprocessing operator accepts FP16/BF16 D64/D128 HND/NHD tensors.
Its largest D128 rotated/anchored specialization uses about 21.1 KiB static
shared memory with no local spill in the compute_75 cubin, while D64 uses about
10.6 KiB. Model semantics remain in plugin adapters rather than the kernel API.
Version 0.22.3 restores a compile-time single-stage loop for route-free dense
W8A8. This keeps the D128 dense specialization at 180 registers/thread and
removes the sparse CTA-K staging overhead without changing the public ABI.

## Check

```bash
python - <<'PY'
import comfyui_turing_utils_kernel

print("kernel:", comfyui_turing_utils_kernel.__file__)
print("Turing Sage:", comfyui_turing_utils_kernel.turing_sage.available())
print("Turing sparse:", comfyui_turing_utils_kernel.turing_sage.sparse_available())
print("Turing W8A8 attention:", comfyui_turing_utils_kernel.turing_sage.w8a8_available())
print("Fused Q/K preprocessing:", comfyui_turing_utils_kernel.turing_sage.fused_qk_preprocessing_available())
print("Turing W4A8:", callable(comfyui_turing_utils_kernel.turing_w4a8_linear))
print("Turing SwiGLU:", callable(comfyui_turing_utils_kernel.turing_swiglu_int8_convrot_quantize))
print("Turing norm:", callable(comfyui_turing_utils_kernel.turing_segmented_rms_adaln))
PY
```

## License

Apache-2.0. See `LICENSE`, `NOTICE`, and `LICENSES/`.
