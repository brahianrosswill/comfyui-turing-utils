# Turing runtime flow

The Turing path is a generic runtime capability. MiniMax H3 only supplies an
adapter that connects its block structure to those capabilities.

```text
checkpoint metadata
  -> quantization summary
  -> prepare exact-sm75 runtime and run kernel self-tests
  -> select the model-declared BF16 inference boundary
  -> ComfyUI constructs the ModelPatcher
  -> normalize ConvRot logical weight dtype without copying packed data
  -> install optional model adapter object patches
  -> install attention override
```

Diffusion and CLIP loaders prepare their own runtime. Loading a
text encoder therefore does not depend on a diffusion loader having run first.
An explicit ComfyUI dtype flag wins over automatic BF16 selection, but it does
not skip attention or quantized-kernel preflight.

## Ownership

| Module | Scope |
|---|---|
| `precision.py` | BF16 selection, Kitchen contract, exact-sm75 preflight |
| `attention.py` | generic backend selection and bundled Sage adapter |
| `turing_ops.py` | exact-sm75 Kitchen backend and W8/W4 dispatch policy |
| `turing_fusions.py` | model-independent fused Linear activation and segmented norm calls |
| `minimax_adapter.py` | MiniMax block discovery and ModelPatcher object patches only |
| `kernel/csrc/turing` | separately installed Turing kernels, including bundled Sage |

## Linear matrix

| Weight/activation | Plain input | SwiGLU input | Output |
|---|---|---|---|
| W8A8 | fused Kitchen rotation when it fits; BF16 row-buffer or staged fallback | SwiGLU is folded into the same rotation/quantization decision | requested dtype, BF16 fast epilogue where eligible |
| W4A4 | fused A4 rotation when it fits; BF16 row-buffer or grouped staged fallback | bundled staged/row-buffer SwiGLU produces packed A4 directly | original BF16 boundary |
| W4A8 | shares the W8 activation quantizer and consumes packed W4 directly | shares fused W8 SwiGLU quantization | BF16 |

The BF16 row-buffer stores one completed rotated row as BF16 and keeps only
active FHT groups in FP32. Launch selection is constrained to less than 48 KiB
total shared memory. It does not allocate the full activated or rotated BF16
intermediate. A40 compatibility validation shows identical packed INT8/INT4
values versus the staged operators across K=256, 5376, 7168, and 14336; INT4
scale differences are at most normal FP32 reduction-order roundoff.

W4A8 reads packed W4 tiles directly, expands each vector once while filling
CUTLASS's SM75 crosswise INT8 shared-memory layout, and executes the contraction
with `m8n8k16` INT8 Tensor Core instructions. Scaling, optional bias, and BF16
storage stay in the epilogue. It never creates a persistent W8 weight copy and
keeps every production tile within the default 48 KiB shared-memory limit.
Non-Tensor-Core-compatible edge dimensions retain a small DP4A compatibility
kernel so the public API does not silently narrow its accepted shapes.

For W8A8 GEMM, Kitchen's fused Turing kernel remains first choice. The no-bias
contraction fallback uses cuBLAS INT8 plus the bundled vectorized BF16
epilogue. Its INT32 workspace is intentionally retained because the measured
H3 out-projection contribution was small and large-sequence shapes can lose
occupancy when forced through the alternate fused GEMM.

## Attention matrix

On exact sm75, both `auto` and the explicit `sage_attn` option select the
bundled Sage implementation; the standalone package is not required. On other
GPUs, `sage_attn` means the independently installed SageAttention package.
Legacy serialized `sage`, `sage_`, `sage_hybrid`, and `turing_sage` values
normalize invisibly to `sage_attn` and are not displayed by either loader.

| Option | Q/K path | Smoothing | PV path |
|---|---|---|---|
| `sage_attn` on Turing | INT8, per-16-token Q-warp scales | disabled | FP16 V tiles with direct FP32 accumulation |

Integer Q/K MMA accumulates into INT32. The stable facade supports FP16 and
BF16 Q/K/V, HND/NHD, GQA, causal mode, unequal Q/KV lengths, head dimensions
through 128, and variable-length batches. BF16 V is converted tile-by-tile
while loading shared memory, so no full V conversion tensor exists. FP32 Q/K/V
use one BF16 boundary conversion and restore FP32 output.

When either logical sequence is shorter than the 64-token SM75 CTA, the facade
uses a bounded exact FP32 SDPA path. It contains fewer than 4096 scores per head
and cannot reproduce the large-sequence SDPA allocation failure.

The Sage1 and Sage2 adaptations produced severe block artefacts and black
flicker in local Turing tests. They are unstable experiments, not production
fallbacks. The loader, public package, default bindings, and default template
instantiations exclude them. Their complete checkpoint and reproduction steps
are documented in
[`kernel/experiments/turing_sage_variants`](../kernel/experiments/turing_sage_variants/README.md).

On non-Turing GPUs, installed standalone Sage has priority, followed by Flash
Attention and PyTorch SDPA. An all-FP32 call that cannot enter external Sage
uses ComfyUI's PyTorch attention implementation deterministically.

The loader log reports `sage_attn via bundled_turing_sage` when `auto` or
`sage_attn` is bound to the local sm75 implementation. The first real bundled
call for each distinct sequence shape also reports dtype, layout, and Q/K/V
shapes. Any unsupported call reports its fallback reason once, so a masked,
disabled-low-precision, or incompatible shape can no longer turn a performance
regression into an invisible backend change. MiniMax additionally emits one
`phase=block` and one `phase=mlp` runtime-dispatch line after the first complete
pass. A healthy H3 W8A8 run reports 50 fused and zero fallback calls for both
phases; these counters use no CUDA events or device synchronization.

## Validation boundary

Release builds target sm75 for bundled Sage. Static tests validate dispatch,
fallbacks, loader independence, shapes, dtypes, the 48 KiB policy, the public
symbol boundary, and exclusion of experimental packages. For compatible A40
validation, build with:

```bash
COMFYUI_TURING_UTILS_ARCH_LIST="7.5+PTX" \
python -m pip install -v --no-build-isolation -e ./kernel
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark
```

An A40 run validates numerical behavior, allocation shapes, and the absence of
Ampere-only source dependencies. It JITs compute_75 PTX and selects the same
CTA schedule used on sm75. This does not replace the final exact-sm75 occupancy
and end-to-end test.
