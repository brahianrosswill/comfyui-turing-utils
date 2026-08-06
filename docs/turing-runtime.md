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

On exact sm75, `auto` and `sage_attn` select bundled `sage_`; the standalone
package is not required. This direct-FP32 path is the fastest and most stable
current default. The loader also exposes experimental `sage1` and `sage2`:

| Option | Q/K path | Smoothing | PV path |
|---|---|---|---|
| `sage_` | INT8, per-16-token Q-warp scales | disabled | FP16 MMA with direct FP32 accumulation |
| `sage1` | INT8, one scale per 64-token CTA | global K mean fused into K quantization | FP16 MMA per 64-token tile, then FP32 running accumulator |
| `sage2` | packed INT4, per-thread scales | blockwise Q mean + global K mean with fused score correction | FP16 MMA per 64-token tile, then FP32 running accumulator |

Integer QK MMA accumulates into INT32 in every variant; the mixed accumulation
above refers to the probability/value operation. The `sage2`
path is a Turing adaptation of SageAttention2 rather than a bit-identical copy:
SM75 has no official FP8 PV tensor-core path, so it uses native FP16 PV. Its
SM75 INT4 MMA is decomposed into Turing `m8n8k32.s4` instructions.

The current standalone SageAttention CUDA API also defaults its FP16-V path to
FP32 PV accumulation. Its fully FP16 mode is an explicit faster option whose
own API documentation warns about numerical instability and suggests V
smoothing for biased inputs. The former bundled Sage1 selected that unsafe
mode unconditionally; that is why it could fail on H3 even when the normal
standalone path did not. The bundled stable modes do not require `smooth_v`.

`sage2` follows the official outlier-smoothing identity. It subtracts a mean
from each 64-token Q block and a global sequence mean from K during
quantization. Q smoothing and packed quantization share one CTA per block;
centered K quantization likewise uses one CTA per block after the global mean
reduction. The non-row-constant Q-mean correction is computed with native SM75
FP16 Tensor Core MMA and FP32 accumulation, then added to the dequantized score
inside the attention CTA. The INT4 score MMA itself is unchanged.

The correction matrix is only 1/64 of a full score matrix, but it would still
be unbounded for H3 video sequences. The runtime therefore processes Q blocks
in chunks with a 128 MiB maximum FP32 correction workspace. Attention uses
absolute Q-block indices across chunks, including causal masking and LSE. At
D=128 the attention CTA uses about 24.25 KiB shared memory, below the 48 KiB
policy. Optional V smoothing remains
available only for explicit comparisons. The loader never enables it because
it creates a full smoothed FP16 V buffer. Normal Sage1/Sage2 instead avoid
long-sequence FP16 overflow by flushing every tile into FP32 without allocating
a converted or smoothed V copy.

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

For N=2048, Hq=8, Hkv=4, D=128, and BF16 input, repeated A40 compute_75
runs measured 0.256–0.264 ms for `sage_`, 0.281–0.298 ms for Sage1, and
0.379–0.401 ms for Sage2. In the 30-iteration validation run the Q+K Sage2
path spent about 0.126 ms in fused preprocessing, 0.020 ms in Tensor Core
correction, and 0.249 ms in the attention kernel; the
remaining time is allocation/dispatch. The previous scalar in-CTA correction
alone took about 0.405 ms, before preprocessing. At N=8192 the same run
measured 3.284 ms, 3.394 ms, and 4.095 ms respectively.

These results explain both sides of Sage2 on Turing. Packed INT4 QK is not the
regression, and the correction redesign removes the former near-2x slowdown.
However, SM75 has no FP8 PV Tensor Core path, while exact score preservation
still requires a correction workspace and an FP32 score addition. Sage2 is
therefore now a meaningful experimental path but is not assumed faster than
Sage1 or `sage_` on Turing. Exact-sm75 profiling remains the release gate for
later architecture-specific scheduling.

The corresponding Sage1 ablation measured 0.231 ms for the `sage_` per-warp,
no-smoothing, direct-FP32 baseline; 0.233 ms after changing only Q/K to
per-block; 0.251 ms after adding global K smoothing; and 0.257 ms after adding
the FP16-tile/FP32-running PV buffer used by stable Sage1. The removed
sequence-long FP16 accumulator measured 0.240 ms on this random input, but it
is not a valid production option: with N=16384 and a V bias of 16 it overflowed
while both stable accumulators remained finite. Thus Sage1 did not literally
become `sage_`, but stabilization necessarily removed most of the old unsafe
FP16 speed advantage. Its remaining differences are per-block scales and K
smoothing; the latter improves the biased-profile MAE shown below.

`kernel/scripts/compare_sage_precision.py` evaluates identical inputs against
the installed official SageAttention INT8 API, bundled Sage1, and bundled
Sage2. Although the current public attention API exposes INT8 QK, the package
still ships its official per-thread INT4 Triton quantizer source. The script
invokes those source kernels directly, compares their raw INT4 codes with the
bundled packed quantizer, and reconstructs Sage2's exact Q-mean score
correction in FP32. This separates INT4 quantization loss, local-vs-official
quantizer differences, and the additional SM75 online-softmax/mixed-PV loss.

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

| Input profile | `sage_` | Bundled Sage1 | Official INT8 | INT4 no smooth | INT4 Q+K smooth | Bundled Sage2 |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian | 0.000694 | 0.000726 | 0.000746 | 0.010711 | 0.010604 | 0.010607 |
| Biased | 0.001329 | 0.001107 | 0.001139 | 0.021953 | 0.012704 | 0.012707 |

For Gaussian inputs, all 11,842,560 local and official-source INT4 codes were
identical. The biased run differed on two codes (0.000017%); the complete
local-vs-official INT4 mathematical output MAE was 1.15e-7. The bundled Sage2
kernel differed from its FP32 INT4 mathematical reference by 1.37e-4 and
1.63e-4 respectively, isolating the smaller online-softmax/mixed-PV component
from the much larger INT4 quantization loss. One of 80 biased causal cases from
the official public INT8 Triton API returned a non-finite output; the comparator
records and reports such cases instead of dropping them. These numbers are
compatibility diagnostics, not exact-sm75 throughput or release-quality model
validation.
