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
| `comfyui_turing_utils/attention/` | stable Sage, sparse policies, topology contract, and patch installation |
| `comfyui_turing_utils/nodes/attention.py` | model-independent experimental sparse-attention patch UI |
| `comfyui_turing_utils/quantization/` | exact-sm75 W8/W4 dispatch, ConvRot loading, and generic fusions |
| `comfyui_turing_utils/adapters/minimax/` | MiniMax layout, packed-sequence planning, fusions, and progressive experiment |
| `comfyui_turing_utils/adapters/wan.py` | Wan/Bernini context-aware memory planning hooks |
| `comfyui_turing_utils/kernel_api.py` | sole lazy boundary to the independently installed kernel package |
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
| explicit `w8a8` on Turing | stable-Sage INT8 score domain | disabled | channel-wise signed INT8 V and unsigned INT8 probabilities, INT32 Tensor Core PV, FP32 online state |
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

The explicit `w8a8` backend and Sol's `use_w8a8` option require kernel 0.18.0.
They are specialized for exact sm75, head dimension 128, and unmasked
non-causal attention. `auto` continues to select stable Sage. The W8A8 path
keeps the same Q64/K64 and 32 KiB shared-memory shape as Sol: V is quantized
once per call into a channel-major, 16-token-permuted signed-INT8 tensor;
softmax probabilities are packed to unsigned INT8; PV uses SM75 U8xS8 Tensor
Core MMA and the output remains FP32 until normalization and dtype writeback.
The route-free dense specialization omits centroid summaries and route state.
Short calls can lose to stable Sage because the extra V scan is not amortized,
which is why W8A8 remains explicit.

The node keeps the measured 4096-token crossover internally; shorter calls use
stable Sage. `routing_threshold=1.0` matches the official mean-plus-one-standard-
deviation policy. Lower values preserve more exact blocks. The local safeguard
is fixed to +/- one 64-token block and is no longer exposed. Density bounds and
frame-distance temporal protection were removed from the complete Python/CUDA
path.

`skipped_residual=1x64` is the official-style fast default. `2x32` changes only
the skipped-block reconstruction; it deliberately shares the identical route.
`dense_prefix_steps=0`, `dense_suffix_steps=0`, `dense_prefix_layers=2`, and
`dense_suffix_layers=0` match the default protection policy. Every dense step or
layer calls the selected protected backend directly: stable bundled Sage by
default, or the route-free bundled W8A8 kernel when `use_w8a8` is enabled.

The common layout contract contains contiguous semantic segments. MiniMax H3's
adapter publishes text, keyframe/reference image, reference video, reference
audio, target audio, and target video ranges from the runtime `PackedLayout`.
The three reference switches independently decide whether those reference
Query and KV blocks may be sparse. Defaults are image=false, video=true, and
audio=false. A disabled switch makes that modality's Query block exact and its
KV block an exact sink for every sparse Query. Target video is sparse; text and
target audio remain protected. Non-aligned boundaries conservatively round
outward to complete 64-token blocks. Missing or inconsistent required H3 layout
metadata selects stable Sage.

The CUDA kernel builds no global route map. Query/key/value summaries remain
separate compact preprocessing tensors, while threshold routing executes inside
each sparse Query CTA immediately before skipped-residual and selected-block
online-softmax updates. The temporary route occupies CTA-local shared memory and
four route words per lane, then survives in registers while the normal 32 KiB
shared-memory tiles are reused. The kernel accepts at most 4096 K/V blocks
(262144 tokens) per call.

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

Sol derives Q/K centroids from the same prequantized INT8 tensors and scales as
selected-block Sage. The K/V preprocessing scan produces one or two such K
centroids and matching original V means for skipped-block reconstruction.
Selected blocks use stable Sage's per-16-row Q and per-64-row K INT8 scales and
SM75 integer Tensor Core QK. By default PV retains the established FP16/FP32
behavior. Optional W8A8 uses signed INT8 V, unsigned INT8 probabilities, INT32
Tensor Core PV, and FP32 online state; skipped residual V centroids remain
original-value FP16. K and, when enabled, V are quantized once per call and
shared by sparse and dense Query blocks.

The Query CTA loads its INT8 Q tile once, derives the route threshold directly
from it, and expands it into resident FP16 shared storage for skipped-block
correction. One Q-to-K-centroid Tensor Core traversal supplies both the routing
score and the online-softmax correction, with conflict-free per-warp shared
partials instead of shared atomics. The compact route is then copied into four
32-bit registers per lane before the arena is reused for exact K/V tiles. The
FP16-PV compute_75 cubin reports 221 BF16 / 233 FP16 registers per main thread, a
16-byte stack frame, zero local-memory spill, and 32 KiB dynamic shared memory.
Register and shared-memory limits permit two 128-thread CTAs per SM75; actual
occupancy and bank behavior still need Nsight confirmation on Turing.

The W8A8 sparse specialization reaches the SM75 compiler's 255-register limit
with a 16-byte stack frame but reports zero local-memory spill; 128 threads use
32768 registers per CTA, so the 32 KiB shared-memory and register budgets still
permit two CTAs on a 64 KiB/65536-register Turing SM. The route-free dense W8A8
specialization uses about 180 registers and no stack/local spill. These resource
figures are static compute_75 reports; resident-CTA throughput still requires a
real Turing profile.

On an A40 JITing compute_75 PTX, four-head FP16 synthetic tests at threshold 1.0
selected 20.0%, 17.6%, and 16.7% of blocks at 4096, 8192, and 16384 tokens.
Official-style `1x64` measured 0.186, 0.483, and 1.553 ms versus dense Sage at
0.388, 1.385, and 5.138 ms (2.09x, 2.87x, and 3.31x). `2x32` measured 0.209,
0.566, and 1.888 ms. An H3-like BF16 shape with 56 heads and 52,860 tokens
measured 279.8 ms at 15.9% route density versus 769.3 ms for dense Sage (2.75x
attention speedup). These are directional A40 compute_75/PTX compatibility
results, not Turing end-to-end or visual-quality measurements.

For the new W8A8 path, an H3-like BF16 shape (`N=52,842`, 56 heads,
`threshold=1.0`, 15.9% route density) measured 715.5 ms for route-free dense
W8A8 versus 765.2 ms for stable Sage, and 220.9 ms for Sol-W8A8 versus 282.8 ms
for Sol with FP16 PV. Thus V quantization is amortized at the intended long
sequence: dense W8A8 was 1.07x faster and Sol-W8A8 was 1.28x faster than its
FP16-PV counterpart. At 4k--16k tokens the extra V scan can instead make W8A8
slower; this reinforces explicit opt-in and does not justify changing `auto`.

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
`sage_attn` binds the local SM75 implementation. Sol logs the resolved protected
ranges, three reference switches, threshold, residual profile, and fixed local
radius. MiniMax additionally emits its fused block/MLP dispatch counters.

`debug_route_density` is disabled by default. With kernel package 0.17.0 or
newer, the already-running sparse CTA accumulates one selected-block counter;
there is no route allocation or popcount kernel. Counts remain on-device across
layers and synchronize once for the end-of-step log. The log reports selected
and possible blocks, min/mean/max layer density, sampling step, layer range,
protected Query count, and residual profile. Debug-off adds no counter atomic,
event, synchronization, or allocation.

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
