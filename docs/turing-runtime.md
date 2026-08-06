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

Diffusion, packed SVDInt4, and CLIP loaders prepare their own runtime. Loading a
text encoder therefore does not depend on a diffusion loader having run first.
An explicit ComfyUI dtype flag wins over automatic BF16 selection, but it does
not skip attention or quantized-kernel preflight.

## Ownership

| Module | Scope |
|---|---|
| `precision.py` | BF16 selection, Kitchen contract, exact-sm75 preflight |
| `attention.py` | generic backend selection and bundled Sage variant adapter |
| `turing_ops.py` | exact-sm75 Kitchen backend and W8/W4 dispatch policy |
| `turing_fusions.py` | model-independent fused Linear activation and segmented norm calls |
| `minimax_adapter.py` | MiniMax block discovery and ModelPatcher object patches only |
| `kernel/csrc/svdint4` | architecture-independent SVDInt4 kernels |
| `kernel/csrc/turing` | separately installed Turing kernels, including the bundled Sage family |

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

W4A8 currently unpacks packed W4 tiles into INT8 shared memory and uses the
sm75 DP4A compatibility kernel. It never creates a persistent W8 weight copy.
A tensor-core version would need a different tile/permutation and register
schedule after unpacking; A40 timing cannot establish the resulting sm75
occupancy. The compatibility kernel is therefore retained until an exact-sm75
profile can justify replacing it.

For W8A8 GEMM, Kitchen's fused Turing kernel remains first choice. The no-bias
contraction fallback uses cuBLAS INT8 plus the bundled vectorized BF16
epilogue. Its INT32 workspace is intentionally retained: the measured H3
out-projection contribution was small, while forcing the alternate fused GEMM
for every large-sequence shape can lower occupancy. This policy should only be
changed from an exact-sm75 profile.

## Attention matrix

On exact sm75, `auto` and `sage_attn` select bundled `sage2`; the standalone
package is not required. The loader also exposes `sage1` and the temporary
`sage_` accuracy baseline:

| Option | Q/K path | Smoothing | PV path |
|---|---|---|---|
| `sage2` | packed INT4, per-thread scales | blockwise Q mean + global K mean with fused score correction | FP16 inputs and FP16 accumulation |
| `sage1` | INT8, one scale per 64-token CTA | global K mean fused into K quantization | FP16 inputs and FP16 accumulation |
| `sage_` | INT8, per-16-token Q-warp scales | disabled by default to preserve the former behavior | FP16 inputs and FP32 accumulation |

Integer QK MMA accumulates into INT32 in every variant; “FP16/FP32
accumulation” above refers to the probability/value operation. The `sage2`
path is a Turing adaptation of SageAttention2 rather than a bit-identical copy:
SM75 has no official FP8 PV tensor-core path, so it uses native FP16 PV. Its
SM75 INT4 MMA is decomposed into Turing `m8n8k32.s4` instructions.

`sage2` follows the official outlier-smoothing identity. It subtracts a mean
from each 64-token Q block and a global sequence mean from K during
quantization, then computes the non-row-constant Q-mean correction inside each
attention CTA. It never creates an N-by-N correction tensor. At D=128, packed
Q/K, the FP16 V tile, FP32 Q/K means, and 64 correction values total about
25.25 KiB shared memory, below the 48 KiB policy. Optional V smoothing is
available for comparisons but defaults off because it necessarily creates a
full smoothed FP16 V buffer and adds the mean back to output.

FP16 and BF16 Q/K/V support HND/NHD, GQA, causal mode, unequal Q/KV sequence
lengths, head dimensions through 128, and variable-length batches. The normal
unsmoothed BF16 V path converts to FP16 tile-by-tile while loading shared
memory, so no full V conversion tensor exists. Sage1/Sage2 varlen smoothing is
sequence-local; the facade invokes the fixed kernel per packed sequence rather
than mixing statistics across a batch. FP32 Q/K/V use a single BF16 boundary
conversion and restore FP32 output.

When either logical sequence is shorter than the 64-token SM75 CTA, the facade
uses an exact FP32 SDPA boundary path. This avoids the vendored kernel's
non-deterministic single-tile tail under CUDA memcheck. The fallback is bounded
to fewer than 4096 scores per head, so it cannot reproduce the large-sequence
SDPA memory failure that the bundled kernels are intended to avoid.

On non-Turing GPUs, the installed standalone Sage backend has priority, then
Flash Attention, then PyTorch SDPA. An all-FP32 call that cannot enter external
Sage uses the ComfyUI PyTorch attention implementation deterministically.

## Validation boundary

Release builds target sm75 for bundled Sage. Static tests validate dispatch,
fallbacks, loader independence, shapes, dtypes, and the 48 KiB calculations.
For compatible A40 validation, build with:

```bash
SVDINT4_ARCH_LIST="7.5+PTX" \
python -m pip install -v --no-build-isolation -e ./kernel
```

An A40 run validates numerical behavior, memory allocation shape, and the
absence of Ampere-only source dependencies. The A40 JITs compute_75 PTX rather
than loading a native sm86 cubin, and the compatibility build forces the same
CTA schedule selected on sm75. It does not replace the final
exact-sm75 occupancy and end-to-end test.

`kernel/scripts/compare_sage_precision.py` evaluates identical inputs against
the installed official SageAttention INT8 API, bundled Sage1, and bundled
Sage2. Although the current public attention API exposes INT8 QK, the package
still ships its official per-thread INT4 Triton quantizer source. The script
invokes those source kernels directly, compares their raw INT4 codes with the
bundled packed quantizer, and reconstructs Sage2's exact Q-mean score
correction in FP32. This separates INT4 quantization loss, local-vs-official
quantizer differences, and the additional SM75 online-softmax/FP16-PV loss.

Run it once with the default Gaussian profile and once with `--profile biased`.
The latter adds a sequence-invariant channel bias to K and a per-64-token block
bias to Q, then reports no-smoothing, K-only, Q-only, and Q+K ablations. This
tests the regime the smoothing identities are designed to improve instead of
drawing conclusions only from approximately zero-mean random inputs. The
official source-level INT4 result is deliberately not described as a public
official INT4 attention API.

One BF16 A40/compute_75 diagnostic run with SageAttention 2.2.0, ten seeds,
sequence length 257, D=64/128, causal/non-causal cases, and two input scales
produced the following aggregate MAE against FP32 SDPA:

| Input profile | Bundled Sage1 INT8 | Official public INT8 | INT4 no smooth | INT4 Q+K smooth | Bundled Sage2 |
|---|---:|---:|---:|---:|---:|
| Gaussian | 0.000729 | 0.000746 | 0.010711 | 0.010604 | 0.010608 |
| Biased | 0.001110 | 0.001156 | 0.021953 | 0.012704 | 0.012708 |

For Gaussian inputs, all 11,842,560 local and official-source INT4 codes were
identical. The biased run differed on two codes (0.000017%); the complete
local-vs-official INT4 mathematical output MAE was 1.15e-7. The bundled Sage2
kernel differed from its FP32 INT4 mathematical reference by 1.45e-4 and
1.73e-4 respectively, isolating the smaller online-softmax/FP16-PV component
from the much larger INT4 quantization loss. One of 80 biased causal cases from
the official public INT8 Triton API returned a non-finite output; the comparator
records and reports such cases instead of dropping them. These numbers are
compatibility diagnostics, not exact-sm75 throughput or release-quality model
validation.
