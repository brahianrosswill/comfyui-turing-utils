# Turing runtime flow

The Turing path is a generic runtime capability. Model adapters only connect
MiniMax or Wan/Bernini block structure to those capabilities.

```text
checkpoint metadata
  -> quantization summary
  -> prepare exact-sm75 runtime and run kernel self-tests
  -> replace an unsupported-FP16 model's sm75 FP32 fallback with BF16
  -> ComfyUI constructs the ModelPatcher
  -> when BF16 was selected, normalize ConvRot logical dtype without copying packed data
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
| `comfyui_turing_utils/precision.py` | BF16 selection, Kitchen contract, exact-sm75 preflight |
| `comfyui_turing_utils/attention/` | prepared-attention protocol, stable Sage, sparse policies, semantic layout, and patches |
| `comfyui_turing_utils/nodes/attention.py` | model-independent production Sol patch UI and explicit kernel tuning |
| `comfyui_turing_utils/quantization/` | exact-sm75 W8/W4 dispatch, ConvRot loading, and generic fusions |
| `comfyui_turing_utils/adapters/minimax/` | MiniMax layout, packed-sequence planning, and fusions |
| `comfyui_turing_utils/adapters/wan.py` | Wan/Bernini context-aware planning and Q/K preprocessing hooks |
| `comfyui_turing_utils/adapters/wan_layout.py` | Wan/Bernini semantic attention-layout provider |
| `comfyui_turing_utils/kernel_api.py` | sole lazy boundary to the independently installed kernel package |
| `kernel/csrc/turing` | separately installed Turing kernels, including bundled Sage |

## Linear matrix

| Weight/activation | Plain input | Fused activation input | Output |
|---|---|---|---|
| W8A8 | fused Kitchen rotation when it fits; BF16 row-buffer or staged fallback | SwiGLU and tanh-GELU are folded into the same rotation/quantization decision | requested dtype, BF16 fast epilogue where eligible |
| W4A4 | fused A4 rotation when it fits; BF16 row-buffer or grouped staged fallback | bundled staged/row-buffer SwiGLU or tanh-GELU produces packed A4 directly | original BF16 boundary |
| Legacy W4A8 | shares the W8 activation quantizer and consumes signed packed W4 directly | shares fused W8 SwiGLU/tanh-GELU quantization | BF16 |
| Grouped-codebook W4A8 | shares the W8 activation quantizer; long sequences decode packed g16 codebook values directly into the W8A8 shared tile, with a bounded staged fallback | shares fused W8 SwiGLU/tanh-GELU quantization | BF16 |

The BF16 row-buffer stores one completed rotated row as BF16 and keeps only
active FHT groups in FP32. Launch selection uses the device's opt-in dynamic
shared-memory limit rather than a fixed 48 KiB policy. Among the 512/768/1024
thread geometries that fit, dispatch estimates resident CTAs from the device's
shared-memory and thread limits and selects the geometry with the most active
warps for the actual row count. Resident CTA count is an input to that choice,
not a fixed acceptance rule. It does not allocate the full activated or rotated BF16
intermediate. A40 compatibility validation shows identical packed INT8/INT4
values versus the staged operators across K=256, 5376, 7168, and 14336; INT4
scale differences are at most normal FP32 reduction-order roundoff.

For the real H3 fc2 boundary (`M=4096`, raw SwiGLU width 28672, contracted
width 14336), an A40 JITing compute_75 PTX measured 0.589 ms for the row-buffer
A8 path versus 0.967 ms for the former staged path, and 0.530 versus 0.888 ms
for A4. Forced 512/768/1024 A8 geometries measured 0.599/0.775/0.767 ms and
produced identical packed values and scales. A40 selects 512 because its
100 KiB shared-memory pool can keep two such CTAs resident; exact Turing has a
different 64 KiB residency calculation, so its final geometry and throughput
still require an exact-sm75 measurement.

W4A8 reads packed W4 tiles directly, expands each vector once while filling
CUTLASS's SM75 crosswise INT8 shared-memory layout, and executes the contraction
with `m8n8k16` INT8 Tensor Core instructions. Scaling, optional bias, and BF16
storage stay in the epilogue. It never creates a persistent W8 weight copy and
rejects only tiles that exceed the actual per-device launch limit.
Legacy W4A8 edge dimensions no longer fall back to a full-matrix DP4A kernel.
K values divisible by four use relaxed, predicated global iterators while the
contraction remains `m8n8k16` Tensor Core MMA. For N tails, the aligned output
channels use the normal long-sequence kernel and only the final one-to-seven
packed rows are zero-padded into an eight-row temporary before a small Tensor
Core launch. No persistent expanded weight or full padded-weight copy is made.
On an A40 JITing compute-75 PTX, `M=4096,N=5376,K=5372` fell from 49.52 ms on
the former DP4A fallback to 1.21 ms on the predicated Tensor Core path. This is
a directional result; exact-sm75 acceptance still requires real Turing.

Since kernel 0.24, the runtime also accepts the symmetric `asym_w4a8_int8`
layout used by the current MiniMax-H3 experimental W4A8 checkpoints. Unlike
the legacy signed nibble format, each stored nibble is a codebook index and
must be combined with an E4M3 per-group relative scale before INT8 MMA. The
SM75 implementation loads one packed 16-value group per thread, combines it
with the E4M3 scale and codebook in registers, and writes the decoded values
directly into CUTLASS's normal crosswise S8 shared-memory tile. The
long-sequence path therefore has no decoded-weight workspace; short sequences
and non-g16 compatibility cases keep the bounded staged implementation.
Neither path expands the checkpoint at load time or creates an MxN INT32
accumulator. Asymmetric correction files remain a Kitchen fallback and are not
silently accepted by this path.

The benchmark uses H3's actual fc2 contracted width 14336 rather than treating
the full 28672-wide fc1 output as the fc2 contraction. On an A40 running the
compute-75/PTX schedule, the latest 4,096-row prequantized W8/legacy-W4/
codebook-W4 timings were 4.38/4.78/5.54 ms for QKV, 6.21/6.40/7.33 ms for
fc1, and 2.70/3.04/3.48 ms for fc2. At 8,192 rows they were
9.95/9.85/10.54 ms, 13.32/13.22/14.05 ms, and 6.13/6.29/6.75 ms. Codebook W4
is therefore a checkpoint-size and quantization-quality feature, not a claim
of higher contraction throughput than W8A8. Inline decode still removes the
112 MiB staged workspace at long H3 sequence lengths without creating a
persistent expanded weight. A synthetic checkpoint-format comparison measured
relative L2 0.07314 and cosine 0.99732 for grouped-codebook g16, versus relative
L2 0.16035 and cosine 0.98738 for the legacy row-scaled signed W4 format. These
are direction and format tests, not final exact-sm75 or model-quality acceptance
results. Final throughput acceptance still requires an exact-sm75 run.

The inline and raw-W8 long-sequence kernels use the same 256-thread CTA and
shared-memory tile. Both 128x256 and 256x128 policies are compiled. The SM75
cubin reports 244/248 registers for inline decode and 208/210 for raw W8,
zero local-memory spill, and one resident CTA for each long policy; inline
decode does not lower CTA density. The staged fallback uses 4,096-output-channel chunks,
matching Kitchen's production chunk policy, and is bounded at 112 MiB for H3
fc2 and 21 MiB for qkv/fc1. MiniMax and Wan planning remain conservative for
short or non-g16 inputs and reserve only the largest mutually exclusive staged
workspace.

The staged codebook decoder packs all 16 output bytes in registers. Its exact
SM75 image uses 28 registers, 64 bytes of static shared storage, and zero stack
or local memory; the previous addressable temporary arrays required a 16-byte
per-thread stack frame. The compatibility path remains bit exact with inline
g16 decode, including predicated N edges.

Exact-sm75 raw W8, signed W4, and long-sequence codebook W4 dispatch perform a
one-time per-device/per-MNK tile microbenchmark and cache the result for the
process. It measures both long policies instead of assuming that more resident
CTAs or one fixed shared-memory budget predicts throughput. CUDA Graph capture
and non-sm75 compatibility runs use the static heuristic and never synchronize
for tuning. The tuning cost is paid once per distinct contraction shape and is
small compared with model initialization. On A40 compute_75/PTX, the existing
long-sequence 128x256 policy remained best for H3-like raw W8/W4, while
codebook W4 at M=10,000, N=K=5,376 measured 4.38 ms for 256x128 versus 6.27 ms
for 128x256. These numbers justify device measurement but do not select the
Turing winner. `benchmark_backends.py --suite linear --tile-sweep` exposes all
compiled policies for acceptance runs.

The stable Sage main loop was also rebuilt from historical commit `4255f3c`
in an isolated worktree and compared with the current compute-75 image on the
same A40 tensors. Old/current timings were 0.7685/0.7677 ms at N=4096,
2.8179/2.8197 ms at N=8192, and 10.9173/10.9179 ms at N=16384. The maximum
difference is about 0.1%, so an observed whole-workflow regression should be
profiled in projection, Q/K/V preparation, model patching, or VRAM movement;
it is not explained by a slower bundled stable-Sage CUDA main loop.

For W8A8 GEMM, Kitchen's fused Turing kernel remains first choice. The no-bias
contraction fallback uses cuBLAS INT8 plus the bundled vectorized BF16
epilogue. Once a full MxN INT32 accumulator would reach 64 MiB, dispatch tries
the fixed-workspace fused path even for contraction/square layers. If Kitchen
cannot serve that shape, it falls back without narrowing the accepted input.

Wan projections, block normalization, attention dispatch, and feed-forward
execution stay on ComfyUI's native path; current ComfyUI folds tanh-GELU through
`linear_input_act` without replacing the block forward. The adapter only adds
context-aware memory planning and therefore does not alter Wan model numerics.
Bernini context latents are included in ComfyUI's memory estimate; context
windows budget both the context tokens and the possible causal anchor without
changing the conditioning used at runtime.

MiniMax planning receives the sampler's original nested video/audio shapes
before ComfyUI flattens them. It mirrors `PackedLayout` row accounting for text,
first/last keyframes, reference images, reference video, reference audio, and
the target streams. The normal ComfyUI heuristic is scaled by the complete
packed sequence, with a lower bound from the known FP32 condition-row buffers
and BF16 packed hidden state. W8A8 planning checks every distinct output width
and reserves the largest live INT32 accumulator; it does not assume that the
widest layer owns the largest workspace after fixed-workspace dispatch.

Wan/Bernini reference shapes are padded per stream before aggregation. The
outer sampling wrapper also supplies the real sampler batch, so repeated
context conditions are represented during initial model loading as well as in
the per-step batching check.

Wan patch embedding deliberately retains ComfyUI's FP32 convolution boundary.
An A40 test with TF32 disabled found the prospective FP16 path slower and
observed a non-zero BF16 output delta, so it is not installed as a speculative
Turing optimization. The reproducible comparison is available through
`validate_wan_fusions.py --patch-embedding`.

## Attention matrix

The loader exposes exactly `w8a8`, `sage`, and `sdpa`, with W8A8 selected by
default. On exact sm75, W8A8 and Sage use the bundled implementations. On other
GPUs, W8A8 delegates to Comfy Kitchen and Sage delegates to ComfyUI's registered
SageAttention function. `auto` is no longer accepted. Legacy serialized
`sage_attn`, `sage_`, `sage_hybrid`, and `turing_sage` values normalize
invisibly to `sage` and are not displayed by the loader.

| Option | Q/K path | Smoothing | PV path |
|---|---|---|---|
| `sage` on Turing | INT8, per-16-token Q-warp scales | disabled | FP16 V tiles with direct FP32 accumulation |
| `w8a8` on Turing | stable-Sage INT8 score domain | disabled | channel-wise signed INT8 V and unsigned INT8 probabilities, INT32 Tensor Core PV, FP32 online state |
| `Patch Sol Sparse Attention` | fused 64-token centroid routing; selected tiles reuse stable Sage INT8 QK | input-adaptive `mean + tau * std` threshold | exact FP16 V tiles plus skipped-block V centroids, FP32 online accumulation |

Integer Q/K MMA accumulates into INT32. The stable facade supports FP16 and
BF16 Q/K/V, HND/NHD, GQA, causal mode, unequal Q/KV lengths, head dimensions
through 128, and variable-length batches. BF16 V is converted tile-by-tile
while loading shared memory, so no full V conversion tensor exists. FP32 Q/K/V
use one BF16 boundary conversion and restore FP32 output. Explicit SDPA instead
consumes BF16 containers and converts Q, K, then V to FP16 before PyTorch
dispatch on exact sm75, avoiding the BF16 math implementation while bounding
overlapping storage; its output is restored to BF16.

When either logical sequence is shorter than the 64-token SM75 CTA, the facade
uses a bounded exact FP32 SDPA path. It contains fewer than 4096 scores per head
and cannot reproduce the large-sequence SDPA allocation failure.

The sparse backend is installed only by the independent
`Patch Sol Sparse Attention` node and is never a loader option. Keeping a
separate node is intentional: its routing, modality, step, layer, and residual
quality parameters do not belong in the loader.
Dispatch depends only on the attention call: matching
FP16, BF16, or FP32 Q/K/V; head dimensions 1--128; unmasked non-causal attention;
and both Q and K meeting the configurable minimum sequence length. HND and
ComfyUI's unreshaped layout, GQA, unequal Q/K lengths, and incomplete final
blocks are supported. Other calls use bundled stable Sage without model-family,
sampling-step checks. A model adapter may publish semantic layer and topology
metadata; unknown models remain fully generic.

The bundled `w8a8` backend and Sol's `use_w8a8` option require kernel 0.23.0.
They are specialized for exact sm75 and head dimensions 1--128. Dense W8A8
supports fixed HND/NHD and native packed-varlen inputs, GQA, unequal Q/K
lengths, and an upper-left causal diagonal; arbitrary masks remain unsupported.
Sol remains fixed-shape, unmasked, and non-causal. The W8A8 path
keeps the same Q64 tile shape as Sol: native D64 uses 16 KiB shared memory and
native D128 uses 32 KiB. V is quantized once per call into a channel-major,
16-token-permuted signed-INT8 tensor;
softmax probabilities are packed to unsigned INT8; PV uses SM75 U8xS8 Tensor
Core MMA and the output remains FP32 until normalization and dtype writeback.
The route-free dense specialization omits centroid summaries and route state.
Short calls can lose to stable Sage because the extra V scan is not amortized,
so exact-sm75 users should still compare W8A8 and Sage for short sequences.

For packed varlen, Q/K/output and cumulative sequence metadata remain compact;
the implementation does not pad every sequence to the batch maximum or launch
one attention kernel per sequence. Q/K Hadamard rotation is fused into their
packed quantizers. V uses a per-sequence, per-head, per-channel signed-INT8
scale and pads each sequence internally by at most 63 tokens, preserving
aligned 128-bit V tile reads without a `batch * max_length` allocation.
Adaptive K-anchor subtraction is intentionally unavailable in this contract
because finding an anchor would add a separate per-sequence scan.

Kernel 0.20.0 provides the split prequantize/execute ABI used by current ComfyUI's
`AttentionTensorContainer`. Q/K quantization, optional V quantization, and Sol
correction summaries are completed before allocating the output. The original
Q/K/V tensors are then released; stable Sage and FP16-PV sparse paths retain
only the contiguous V buffer required by the main kernel, while W8A8 retains
only quantized V. Older kernels remain supported through the one-call ABI.
All bundled attention kernels launch on PyTorch's current CUDA stream, which
also makes the graph-leaf dense Sage/W8A8 operations safe for CUDA Graph
capture. It also exposes logical CTA-K64/128 scheduling and fused
Hadamard/adaptive-anchor quality controls through a separate experimental
tuning patch. Logical K128 processes two K64 stages using the same shared tile.

Kernel 0.22 extends that lifetime contract upstream into model-owned Q/K
preprocessing. MiniMax H3 passes raw projected Q/K together with per-head
RMSNorm and partial split-half RoPE semantics; Wan and Bernini pass whole-row
RMSNorm and full interleaved RoPE semantics. One custom op reads each raw Q/K
tile once, keeps it in CTA shared memory, and emits the exact INT8/scales layout
consumed by dense Sage, dense W8A8, Sol FP16-PV, and Sol W8A8 for supported H3
and Wan/Bernini self-attention calls. It therefore
avoids materializing normalized/rotated BF16 Q/K. Protected Sol steps and layers
use the corresponding dense finalizer without repeating preprocessing.
The handoff is installed through a model-neutral attention-site registry, so
an explicit Sol patch on an official ComfyUI-loaded H3 or Bernini model can use
the same fused path as the ConvRot loader. Capability rejection occurs before
Q/K/V ownership transfers; a rejecting backend is forbidden from consuming a
tensor and the model safely retains its original attention path.

The compute_75 cubin reports at most 21,128 bytes static shared memory and 76
registers/thread for D128 preprocessing, or 10,632 bytes and 59
registers/thread for D64, with no local spill. An A40 PTX direction test at
BF16 B1/H8/N8192/D128 measured 0.25 ms and about 16 MiB peak allocation for the
fused path versus 2.44 ms and 112 MiB for an unfused PyTorch reference. This is
not a comparison against Kitchen's fused model op and does not replace an
exact-SM75 end-to-end profile.

Kernel 0.23 separates Sol's route-statistics basis from its exact score basis.
The selected-block QK and skipped-block correction MMA remain post-Hadamard and
therefore reuse W8A8's exact INT8 representation. Only the Q/K block centroids
used by the diagonal `mean + tau * std` threshold are inverse-transformed to
the pre-Hadamard basis. D64 needs one cross-warp butterfly and D128 needs two;
they reuse the existing route scratch. No complete Q/K tensor is reread, no
route map is materialized, and dynamic shared memory remains 16/32 KiB. One
separate FP16 K centroid is retained for `1x64`, because the post-Hadamard copy
continues to serve skipped-block correction.
At N=4096, Hq=8/Hkv=4, D128 BF16 on the A40 compute_75 path, five 500-call
runs measured -0.45% for FP16-PV and +0.19% for W8A8 versus the old rotated
threshold domain, with identical selected-block counts; both are measurement
noise rather than observable overhead. Selected-block outputs are unchanged
when every block is exact. The route words are explicitly scalarized into four
registers per lane; the resource gate verifies zero local/stack spill after
enabling the transform.

Optional phase timing is process-local and disabled unless
`COMFYUI_TURING_UTILS_PROFILE_CALLS` is a positive integer. The disabled path
creates no CUDA events and performs no synchronization.

The node keeps the measured 4096-token crossover internally; shorter calls use
stable Sage. `routing_threshold=1.0` matches the official mean-plus-one-standard-
deviation policy. Lower values preserve more exact blocks. The local safeguard
is fixed to +/- one 64-token block and is no longer exposed. Density bounds and
frame-distance temporal protection were removed from the complete Python/CUDA
path.

`skipped_residual=1x64` is the official-style fast default. `2x32` changes only
the skipped-block reconstruction; it deliberately shares the identical route.
`dense_prefix_steps=1`, `dense_suffix_steps=0`, `dense_prefix_layers=2`, and
`dense_suffix_layers=0` match the default protection policy. Every dense step or
layer calls the selected protected backend directly: route-free bundled W8A8
by default, or stable bundled Sage when `use_w8a8` is disabled. If the prefix
and suffix layer counts sum to at least the runtime layer count, every valid
layer takes this direct dense path and no Sol summaries or routing are built.

The common layout contract contains contiguous semantic segments. MiniMax H3's
adapter publishes text, keyframe/reference image, reference-video first/last
latent-frame anchors, reference-video interior, reference audio, target audio,
and target video ranges from the runtime `PackedLayout`.
The three reference switches independently decide whether those reference
Query and KV blocks may be sparse. Defaults are image=false, video=true, and
audio=false. Reference-video anchors follow the image switch, while the clip
interior follows the video switch. A disabled switch makes that modality's Query block exact and its
KV block an exact sink for every sparse Query. Target video is sparse; text and
target audio remain protected. Non-aligned boundaries conservatively round
outward to complete 64-token blocks. Missing or inconsistent required H3 layout
metadata selects stable Sage.

The CUDA kernel builds no global route map. Query/key/value summaries remain
separate compact preprocessing tensors, while threshold routing executes inside
each sparse Query CTA immediately before skipped-residual and selected-block
online-softmax updates. The temporary route occupies CTA-local shared memory and
four route words per lane, then survives in registers while the normal D64
16 KiB or D128 32 KiB shared-memory tiles are reused. The kernel accepts at
most 4096 K/V blocks
(262144 tokens) per call.

Sol derives Q/K centroids from the same prequantized INT8 tensors and scales as
selected-block Sage. The K/V preprocessing scan produces one or two such K
centroids and matching original V means for skipped-block reconstruction.
Selected blocks use stable Sage's per-16-row Q and per-64-row K INT8 scales and
SM75 integer Tensor Core QK. By default PV retains the established FP16/FP32
behavior. Optional W8A8 uses signed INT8 V, unsigned INT8 probabilities, INT32
Tensor Core PV, and FP32 online state; skipped residual V centroids remain
original-value FP16. K and, when enabled, V are quantized once per call and
shared by sparse and dense Query blocks.

The Query CTA derives the route threshold from its INT8 Q tile and expands that
tile into resident FP16 shared storage for skipped-block correction. One
Q-to-K-centroid Tensor Core traversal supplies both the routing score and the
online-softmax correction, with conflict-free per-warp shared partials instead
of shared atomics. The compact route is then copied into four 32-bit registers
per lane before the arena is reused for exact K/V tiles. Keeping both the FP16
correction operand and the INT8 exact operand live would raise D128 shared
storage above 32 KiB, so the production kernel deliberately re-reads the small
INT8 Q tile instead of reducing CTA residency. Exact-Q staging is only 8 KiB
per CTA and is normally L2-hot after routing.

The official-style `1x64` and quality `2x32` residual paths, plus the 64- and
128-token exact-K staging paths, are separate compile-time specializations.
This removes runtime loop bounds and second-stage state from the default
long-sequence kernel without changing the selected route or arithmetic order.
A correctness gate requires bitwise-identical output between K64 and K128
staging. The exact compute_75 image contains 24 variants for each native head
dimension, all with zero stack/local-memory spill and unchanged 16 KiB D64 /
32 KiB D128 dynamic shared memory. Register use ranges from 138--175 for D64
and 180--255 for D128. Occupancy is reported for diagnosis but is not a
production gate; final resident-CTA throughput still requires real Turing
profiling.

With the 0.23 causal/varlen specializations, compute_75 reports zero local and
stack storage for all dense W8A8 variants. D128 uses 180 registers for fixed
non-causal, 244 for fixed causal, 245 for packed non-causal, and 246--247 for
packed causal; D64 stays at or below 175. All retain the existing 32/16 KiB
dynamic shared-memory budgets. On an A40 executing the compute_75 path, a BF16
GQA batch with Q lengths 3072/4096, K lengths 3201/4096, Hq=8, Hkv=4, D128
measured 1.145 ms packed versus 1.253 ms for two fixed calls (non-causal), and
0.700 versus 0.765 ms (causal). Packed V quantization measured 0.075 ms, down
from 1.146 ms in the discarded serial prototype. These are direction tests,
not a substitute for exact-sm75 profiling.

On an A40 JITing compute_75 PTX, four-head FP16 synthetic tests at threshold 1.0
selected 19.4%, 17.8%, and 16.6% of blocks at 4096, 8192, and 16384 tokens.
After residual and K-stage specialization, Sol-W8A8 `1x64` measured 0.235,
0.519, and 1.526 ms versus dense Sage at 0.405, 1.431, and 5.352 ms (1.72x,
2.76x, and 3.51x). `2x32` FP16-PV measured 0.232, 0.607, and 1.996 ms. An
H3-like BF16 shape with 56 heads and 52,842 tokens measured 176.3 ms at 16.2%
route density, down from about 186.8 ms before both compile-time
specializations. These are directional A40 compute_75/PTX compatibility
results, not Turing end-to-end or visual-quality measurements.

Kernel 0.21 adds native D64 dense W8A8 and Sol kernels rather than padding D64
to D128. On the same A40 compute_75/PTX directional run at BF16, N=4096,
Hq/Hkv=8/4, the prequantized dense core measured 0.409 ms for native D64 versus
0.866 ms for the same input zero-padded to D128; Sol-W8A8 measured 0.166 versus
0.257 ms. D64 uses 16 KiB shared memory and remains below the D128 register
footprint. Final speed and CTA residency still require exact-sm75 measurement.

For the new W8A8 path, an H3-like BF16 shape (`N=52,842`, 56 heads,
`threshold=1.0`, 15.9% route density) measured 715.5 ms for route-free dense
W8A8 versus 765.2 ms for stable Sage, and 220.9 ms for Sol-W8A8 versus 282.8 ms
for Sol with FP16 PV. Thus V quantization is amortized at the intended long
sequence: dense W8A8 was 1.07x faster and Sol-W8A8 was 1.28x faster than its
FP16-PV counterpart. At 4k--16k tokens the extra V scan can instead make W8A8
slower; this is the main reason to retain Sage as an explicit alternative to
the W8A8 default.

Kernel 0.22.3 removes an unintended runtime two-stage loop from route-free
dense W8A8 while retaining CTA-K64/128 staging for Sol. On the same A40
compute_75 direction check at BF16, `N=53,192`, and 56 heads, the prequantized
W8A8 core measured 696.3 ms versus 774.1 ms for stable Sage (1.11x). Before
the fix the dense core measured about 946 ms. This is still an A40 directional
test; exact-sm75 end-to-end throughput remains the acceptance criterion.

Stable Sage deliberately remains CTA-K64. A CTA-K128 experiment kept roughly
the same theoretical active-warp count, but the upstream kernel has no
cross-K-warp merge for its online-softmax state: its output cosine versus
CTA-K64 was only about 0.50, and it was 10--20% slower in A40 compute_75/PTX
tests. The kernel now rejects any future multi-K-warp instantiation at compile
time. Route-free dense W8A8 also deliberately uses one compile-time K stage;
its public 64/128 route tile setting applies to Sol routing and does not create
a slower dense runtime staging loop. The common benchmark reports both Sage
and W8A8 prequantized cores so preprocessing cannot hide this distinction.

The Sage1 and Sage2 adaptations produced severe block artefacts and black
flicker in local Turing tests. They are unstable experiments, not production
fallbacks. The loader, public package, default bindings, and default template
instantiations exclude them. Their complete checkpoint and reproduction steps
are documented in
[`kernel/experiments/turing_sage_variants`](../kernel/experiments/turing_sage_variants/README.md).

On non-Turing GPUs the selected dense backend is deterministic: W8A8 uses
Comfy Kitchen, Sage uses the registered SageAttention function, and SDPA uses
ComfyUI's PyTorch implementation. Flash Attention is not a loader option. An
all-FP32 call that cannot enter external Sage uses ComfyUI's PyTorch attention
implementation deterministically. Sol is the exception to the former
"non-Turing means no local attention" rule: kernel 0.28.0 can compile its
integer routing/exact core natively for Ampere, Ada, and Hopper, while every
protected dense step or layer still delegates to those architecture-native
dense backends.

The loader log reports `w8a8 via bundled_turing_w8a8` or
`sage via bundled_turing_sage` for local SM75 implementations. Sol logs the
resolved protected ranges, three reference switches, threshold, residual
profile, and fixed local radius. MiniMax additionally emits its fused block/MLP
dispatch counters.

`debug_route_density` is disabled by default. With kernel package 0.23.0 or
newer, the already-running sparse CTA accumulates one selected-block counter;
there is no route allocation or popcount kernel. Counts remain on-device across
layers and synchronize once for the end-of-step log. The log reports selected
and possible blocks, min/mean/max layer density, sampling step, layer range,
protected Query count, and residual profile. Debug-off adds no counter atomic,
event, synchronization, or allocation.

The final A40 compute-75/PTX regression sweep keeps preprocessing and core
attention separate. At N=4096/H56/D128, end-to-end Sage/W8A8/SDPA/external-
Sage measured 4.67/5.47/4.39/4.63 ms, while the already-prequantized bundled
cores measured 4.54/4.32 ms. At N=8192 they measured
18.68/19.65/18.04/17.10 ms end to end and 18.39/16.51 ms prequantized. This
confirms that W8A8's INT8 PV core is faster but its extra V quantization is not
amortized at short sequences; the long H3 measurements above are the intended
default workload. It also prevents a core win from being mislabeled as an
end-to-end win.

The same sweep measured fused H3-like fc2 SwiGLU ConvRot input preparation at
0.65 versus 1.04 ms staged for 4,096 rows and 1.17 versus 1.92 ms for 8,192
rows (about 1.6x). The packed BF16 epilogue measured about 5x faster than its
eager reference. These are the retained optimizations. Grouped-codebook W4 is
retained for checkpoint size/quality despite being 6--29% slower than raw W8
in the tested contractions; no production documentation claims otherwise.

## Validation boundary

Local builds detect and deduplicate all visible supported GPU capabilities. A
mixed Turing/Ampere host therefore emits both cubins automatically. An explicit
`COMFYUI_TURING_UTILS_ARCH_LIST` remains available for cross-compilation;
GPU-less builders use a conservative sm75 fallback. Static tests validate
dispatch, fallbacks, loader independence, shapes, dtypes, spill-free SM75
resources for every compiled core/attention/preprocessing family, the public
symbol boundary, and exclusion of the retired Sage1/Sage2 variants. For
compatible A40 validation, build with:

```bash
COMFYUI_TURING_UTILS_ARCH_LIST="7.5+PTX" \
python -m pip install -v --no-build-isolation -e ./kernel
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark --sol
python kernel/scripts/validate_wan_fusions.py --device cuda:0
python kernel/scripts/audit_attention_resources.py
```

An A40 run validates numerical behavior, allocation shapes, and the absence of
Ampere-only source dependencies. It JITs compute_75 PTX and selects the same
CTA schedule used on sm75. This does not replace the final exact-sm75 occupancy
and end-to-end test.

For native Ampere acceptance, use `COMFYUI_TURING_UTILS_ARCH_LIST="8.6"`.
The resulting attention cubins use the `__CUDA_ARCH__ >= 800` async-copy and
INT8 MMA paths; A40 preflight covers BF16 D64/D128 GQA for both Sol and W8A8.
The historical `_sage_*_sm75` module names remain stable ABI identifiers.
On the initial native-sm86 direction check (`N=4096`, H8, D128, BF16), dense
W8A8 measured 0.840 ms, Sol FP16-PV 0.428 ms, and Sol W8A8 0.417 ms at 19.7%
route density. Their output cosine against dense W8A8 was 0.9985 and 0.9982,
respectively. These figures verify native dispatch and arithmetic; they are not
an H3 end-to-end or visual-quality claim.
