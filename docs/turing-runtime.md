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
| `precision.py` | BF16 selection, Kitchen contract, exact-sm75 preflight |
| `attention.py` | generic backend selection and bundled Sage adapter |
| `attention_nodes.py` | model-independent experimental sparse-attention patch UI |
| `turing_ops.py` | exact-sm75 Kitchen backend and W8/W4 dispatch policy |
| `turing_fusions.py` | model-independent fused Linear activation and segmented norm calls |
| `minimax_adapter.py` | MiniMax packed-sequence memory planning and ModelPatcher object patches |
| `wan_adapter.py` | Wan/Bernini context-aware memory planning hooks |
| `kernel/csrc/turing` | separately installed Turing kernels, including bundled Sage |

## Linear matrix

| Weight/activation | Plain input | Fused activation input | Output |
|---|---|---|---|
| W8A8 | fused Kitchen rotation when it fits; BF16 row-buffer or staged fallback | SwiGLU and tanh-GELU are folded into the same rotation/quantization decision | requested dtype, BF16 fast epilogue where eligible |
| W4A4 | fused A4 rotation when it fits; BF16 row-buffer or grouped staged fallback | bundled staged/row-buffer SwiGLU or tanh-GELU produces packed A4 directly | original BF16 boundary |
| W4A8 | shares the W8 activation quantizer and consumes packed W4 directly | shares fused W8 SwiGLU/tanh-GELU quantization | BF16 |

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

On exact sm75, both `auto` and the explicit `sage_attn` option select the
bundled Sage implementation; the standalone package is not required. On other
GPUs, `sage_attn` means the independently installed SageAttention package.
Legacy serialized `sage`, `sage_`, `sage_hybrid`, and `turing_sage` values
normalize invisibly to `sage_attn` and are not displayed by either loader.

| Option | Q/K path | Smoothing | PV path |
|---|---|---|---|
| `sage_attn` on Turing | INT8, per-16-token Q-warp scales | disabled | FP16 V tiles with direct FP32 accumulation |
| `Patch Sol Sparse Attention` | fused 64-token centroid routing; selected tiles reuse stable Sage INT8 QK | input-adaptive `mean + tau * std` threshold | exact FP16 V tiles plus skipped-block V centroids, FP32 online accumulation |
| `Patch Sage Frame Sparse Attention` | cached head-independent frame schedule; selected tiles reuse stable Sage INT8 QK | fixed local/sink/rotating-anchor structure | exact selected FP16 V tiles, FP32 online accumulation |

Integer Q/K MMA accumulates into INT32. The stable facade supports FP16 and
BF16 Q/K/V, HND/NHD, GQA, causal mode, unequal Q/KV lengths, head dimensions
through 128, and variable-length batches. BF16 V is converted tile-by-tile
while loading shared memory, so no full V conversion tensor exists. FP32 Q/K/V
use one BF16 boundary conversion and restore FP32 output.

When either logical sequence is shorter than the 64-token SM75 CTA, the facade
uses a bounded exact FP32 SDPA path. It contains fewer than 4096 scores per head
and cannot reproduce the large-sequence SDPA allocation failure.

The sparse backend is installed only by the independent
`Patch Sol Sparse Attention (Experimental)` node and is never part of a loader's
`auto` priority list. Dispatch depends only on the attention call: matching
FP16, BF16, or FP32 Q/K/V; 128-dimensional heads; unmasked non-causal attention;
and both Q and K meeting the configurable minimum sequence length. HND and
ComfyUI's unreshaped layout, GQA, unequal Q/K lengths, and incomplete final
blocks are supported. Other calls use bundled stable Sage without model-family,
sampling-step checks. A model adapter may publish semantic layer and topology
metadata; unknown models remain fully generic.

The patch node keeps the measured 4096-token crossover internally; shorter
calls use stable Sage and no manual sequence-length control is exposed.
`routing_threshold=1.0` routes a block when its centroid score exceeds the
current query row's projected mean by one
standard deviation; lowering it evaluates more blocks exactly.
`minimum_route_density` and `maximum_route_density` optionally clamp the
adaptive selection count inside each 16-key routing tile. Defaults `0/1`
preserve threshold-only routing with a fast path. Prefix, local, and temporal
blocks remain forced and may exceed the requested maximum. The default
`dense_prefix_steps` and `dense_suffix_steps` are explicit counts of whole
denoising steps that use stable Sage across every layer; both default to zero.
`dense_prefix_layers=1` and `dense_suffix_layers=1` keep the first and last
transformer layers dense while only the middle layers use sparse attention.
Suffix protection requires adapter-provided layer-count metadata. Step lookup
is cached once per sampler timestep rather than synchronized in every
transformer block.

All dense schedule and protected-layer calls go directly through the bundled
stable Turing Sage backend. They do not run Sol with a 100% route and therefore
do not allocate Sol summaries or execute its routing kernels. A previously
installed model attention override is replaced by the Sol patch; unsupported
calls can still reach ComfyUI's original attention through stable Sage's normal
fallback handling.

The independent `Patch Sage Frame Sparse Attention (Experimental)` node uses the
same integer dense-step and first/last-layer safeguards, but removes Sol's
input-adaptive routing, summaries, threshold, density budgets, and skipped-block
residuals. Video Query rows attend every token in the configured neighboring
latent frames, the first `sink_frames`, periodic global anchor frames, and the
resolved semantic K/V prefix. Periodic anchors can rotate by transformer-layer
index so information propagates through different long-range connections instead
of one fixed temporal grid. Non-video Query rows always remain globally dense,
which preserves H3 audio-to-video and reference-to-target attention. A model
adapter must publish a contiguous video-tail boundary and exact tokens per latent
frame; missing or inconsistent topology selects stable Sage.

The structured schedule is cached as two compact CUDA INT32 tensors and shared by
all batches and heads. It is created once per shape and rotated anchor offset, has
no device synchronization in the steady path, and selects 64-token blocks so the
CUDA kernel can reuse the production Sage tile loaders and Tensor Core MMA. Both
the attention kernel and dense fallbacks use the bundled stable Sage backend; a
previous attention override is replaced when this explicit patch is connected.
The defaults are a two-frame radius, one anchor every 12 frames, rotating anchors,
one sink frame, zero dense prefix/suffix steps, and one dense layer at each model
boundary. The compute_75 cubin reports 176 BF16 / 183 FP16 registers per thread,
zero stack and local-memory spill, and 32 KiB dynamic shared memory; both register
and shared-memory limits permit two 128-thread CTAs per SM75 SM.

For H3-like BF16 inputs on an A40 restricted to compute_75 PTX, the default static
route measured 20.13% density at 46,773 tokens and 15.11% at 100,483 tokens. With
56 heads, sparse attention measured 187 ms and 580 ms respectively; dense Sage
measured 600 ms and 2773 ms. The 720p-like sparse attention component was 0.97x
the 480p-like dense component. This does not make the complete 720p transformer
step equal to 480p: projections and MLPs still process roughly the full token
ratio. Exact-sm75 end-to-end quality and occupancy testing remains required.

The configurable local neighborhood remains exact. Prefix policy can use exact
model-supplied semantic layout metadata, disable prefix protection, or accept a
manual token count; `auto` does not guess a prefix for an unknown model. MiniMax
publishes the actual target-video boundary, so text, reference, and target-audio
Query rows run once through stable Sage, while target-video Query rows retain
the semantic prefix as an exact K/V sink. This preserves exact cross-modal
attention in both directions without paying sparse routing and output work for
the prefix Query rows. It also publishes target-video frame geometry; the kernel can
keep matching spatial ranges in adjacent frames exact without reordering tokens.
Other blocks use the fused statistical threshold. Skipped blocks retain
approximate mass and output through two 32-token K/V centroids by default,
which better preserves within-block changes than one 64-token centroid.
`skipped_residual=1x64` keeps the previous lower-cost approximation. Selected
blocks reuse stable Sage's per-16-row Q and per-64-row K
INT8 scales and SM75 integer Tensor Core QK. PV, softmax, and output accumulation
retain the established FP16/FP32 behavior. The selected and residual layouts
overlay one 32 KiB dynamic shared-memory arena; this allows two CTAs to fit in a
64 KiB SM75 shared-memory budget when registers permit. K is quantized once and shared by the dense
prefix and sparse target paths. Routing is written directly from the centroid
Tensor Core tile, so no full proxy-score matrix is materialized.

On an A40 JITing only compute_75 PTX, four-head FP16 uniform synthetic tests at
`routing_threshold=1.0` selected 19.7%, 17.7%, and 16.7% of target blocks. The
default 2x32 residual measured 1.68x, 2.33x, and 2.71x over bundled Sage at
4096, 8192, and 16384 tokens; 1x64 was 7-14% faster. A 56-head BF16 H3-shaped
synthetic test with 46,773 total tokens, a 3,438-token semantic prefix, 405
tokens per frame, one adjacent temporal frame, and threshold 1.5 measured
231.0 ms for 2x32, 209.9 ms for 1x64, and 601.0 ms for dense Sage (2.60x and
2.86x). These are A40 compatibility measurements: they do not show the expected
SM75 occupancy gain from reducing shared memory, and they are not an H3
end-to-end or quality prediction. Exact-sm75 visual and performance testing
remains required before the backend can leave experimental status.

Routing Q/K centroids stay in the original FP16/BF16 domain. Skipped-block
softmax scores instead use Q and K centroids reconstructed from the exact INT8
tensors and scales consumed by selected-block Tensor Core MMA, so both branches
share one score domain. Original V means remain unquantized. One fused K/V scan
produces the routing K centroid, one or two score-K centroids, and matching V
means. The attention CTA reconstructs its summary Q tile from the existing INT8
Q buffer, avoiding a second original-Q read. Two centroids double only the
compact score-K and V-summary tensors; the temporary peak remains dominated by
output and the existing Q/K INT8 buffers.

The final compute_75 cubin reports 228 BF16 / 235 FP16 registers per sparse
attention thread with zero stack and local-memory spill. The launch uses 32 KiB
of dynamic shared memory. At the 46,773-token shape above, a 2x32 CUDA profile
attributed 180.5 ms to sparse attention, 41.9 ms to the dense semantic-prefix
Sage call, 3.5 ms to Q/K quantization, 3.1 ms to the fused K/V summaries, 1.2 ms
to Q summaries, and about 0.7 ms to routing and its statistics. Fusing routing
summaries into the production quantizers was deliberately not retained: the
maximum measured saving is under 2% of this call, while it would couple the
experimental sparse ABI to stable Sage quantization and still could not remove
the required V-summary scan.

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
regression into an invisible backend change. The sparse patch logs its selected
minimum length, prefix policy and resolved prefix, routing threshold, local and
temporal neighborhoods, and resolved topology. MiniMax additionally emits one
`phase=block` and one `phase=mlp` runtime-dispatch line after the first complete
pass. A healthy H3 W8A8 run reports 50 fused and zero fallback calls for both
phases; these counters use no CUDA events or device synchronization.

The patch node's optional `debug_route_density` switch is disabled by default.
When enabled with kernel package 0.13.0 or newer, it launches one tiny CUDA
popcount reduction per sparse layer, keeps the scalar results on-device, and
synchronizes once at the end of each denoising step. The warning reports total
selected/possible target-query blocks plus min/mean/max layer density, sampling
step and layer range, prefix, local radius, temporal radius, residual mode, and
route budget. It also reports
which warmup, tail, or protected-layer decisions selected stable Sage. No
counter kernel, event, synchronization, or extra route allocation is used while
the switch is off.

The independent frame-sparse patch uses a different debug path: its route is a
cached CPU-built CSR schedule, so density is already a host scalar. Enabling
its `debug_route_density` only logs that static value and never reads a CUDA
tensor or tracks a denoising step when both dense step guards are zero.

Frame-sparse parameters are grouped around two policies:

- `quality_profile=custom` preserves all explicit controls and is the default.
  `conservative`, `balanced`, and `fast` replace the sparse policy controls and
  protected layer counts as one tested set.
- `frame_window` reads all spatial tokens in local, sink, and periodic anchor
  frames. `radial` reads all nearby frames, then only matching 8x8 spatial-token
  tiles from progressively subsampled distant frames.
- `temporal_window_frames` is the complete-frame radius;
  `global_anchor_stride=0` disables full periodic anchors.
- `radial_spatial_radius` expands the distant 8x8 tile neighborhood;
  `radial_max_temporal_stride` caps distant temporal subsampling.
- `rotate_global_anchors` also rotates the radial sampling phase by transformer
  layer, even when full-frame anchors are disabled.
- dense prefix/suffix steps and layers always call bundled stable Turing Sage.
  They do not call the pre-patch ComfyUI backend or a slower private dense
  implementation.

## Validation boundary

Release builds target sm75 for bundled Sage. Static tests validate dispatch,
fallbacks, loader independence, shapes, dtypes, the 32 KiB sparse policy, the public
symbol boundary, and exclusion of the retired Sage1/Sage2 variants. For compatible A40
validation, build with:

```bash
COMFYUI_TURING_UTILS_ARCH_LIST="7.5+PTX" \
python -m pip install -v --no-build-isolation -e ./kernel
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark --experimental-sparse
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark --experimental-frame-sparse
python kernel/scripts/validate_wan_fusions.py --device cuda:0
```

An A40 run validates numerical behavior, allocation shapes, and the absence of
Ampere-only source dependencies. It JITs compute_75 PTX and selects the same
CTA schedule used on sm75. This does not replace the final exact-sm75 occupancy
and end-to-end test.
