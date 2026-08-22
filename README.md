# ComfyUI Turing Utils

Compatibility and performance extensions for CUDA Tensor Core GPUs. The plugin
currently provides ConvRot W8A8/W4A8/W4A4 support, exact-sm75 BF16 activation
storage, bundled sm75+ W8A8 attention, exact-sm75 Sage, native sm75+ Sol and
fixed-Top-K SLA sparse attention, and focused Wan/Bernini utilities.

## Requirements

- NVIDIA GPU with CUDA support
- Turing, Ampere, Ada, or newer architecture
- Python 3.10 or newer
- PyTorch with CUDA and ComfyUI
- `comfy-kitchen>=0.2.26` for ConvRot model integration
- the independently installed `comfyui-turing-utils-kernel>=0.29.1` for SLA;
  native sm75+ W8A8/Sol requires a cubin (or PTX) for the target GPU, while
  exact-sm75 installs additionally provide bundled Sage and BF16 compatibility

Grouped-codebook `asym_w4a8_int8` checkpoints require kernel 0.24.0. Existing
W8A8, W4A4, legacy W4A8, and attention paths keep their earlier minimum.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wjie98/comfyui-turing-utils.git
cd comfyui-turing-utils
python -m pip install -v --no-build-isolation -e ./kernel
```

The kernel build detects every visible supported CUDA architecture and removes
duplicates. A machine with a 2080 Ti and a 3070 therefore builds `7.5;8.6` in
one install. Set `COMFYUI_TURING_UTILS_ARCH_LIST` only for cross-compilation or
to override the visible-device set. GPU-less build hosts fall back to `7.5`.

The custom node and CUDA package have separate installation lifecycles.
Python-only plugin updates never invoke a compiler or JIT; rebuild the kernel
only after its CUDA sources or required version change.

## Nodes

- `Load ConvRot DiT` loads ComfyUI ConvRot diffusion models. It supports W8A8,
  W4A8, and W4A4 dispatch. Its attention choices are `w8a8`, `sage`, and
  `sdpa`; W8A8 is the default.
- `Load ConvRot CLIP` loads a ConvRot text encoder independently of the DiT.
- `Bernini Inpaint Condition` starts sampling from the source-video latent,
  supports local or global repainting, and optionally adds the source as aligned
  context tokens.
- `Bernini Context Windows` applies reference-aware Wan context windows with
  selectable absolute or official relative temporal positions.
- `Wan Video Frames Padding` exposes Wan-compatible frame padding.
- `MiniMax H3 Video Frames Padding` pads to H3's `17*n+5` frame grid.
- `Resize Image If Present` resizes, crops, or pads an optional image and mask.
  With no image connected it returns no image, so one graph can safely feed
  optional first- or last-frame conditioning sockets without making a black
  placeholder frame.
- `H3 Concat AV Latent` combines standalone H3 video and audio latents into the
  model's native nested AV latent. `H3 Separate AV Latent` splits the streams
  again; both nodes preserve matching video/audio noise masks.
- `H3 Latent Info` reports the decoded pixel width, height, frame count, and
  H3's 24 FPS model rate without running the VAE.
- `H3 Keyframe Reference` dynamically adds `image_N` inputs and matching
  `keyframe_N` outputs. Every output is role-free and reusable: it can connect
  to either the independent first- or last-frame socket on the semantic/build
  nodes, including both roles across different sampling branches.
  `H3 Image/Video/Audio Reference` encode dynamic
  generic reference sets without allocating a target latent. Visual references
  use match-area sizing when a `latent` is connected and a configurable
  megapixel area budget otherwise; neither mode crops or deliberately enlarges
  the source. The default unbound reference budget is 1.0 megapixel.
  Video reference inputs must be resampled to 24 FPS by their upstream loaders.
- `H3 Semantic Reference` performs the Qwen3-VL presentation encode once from
  the prompt and reference objects. `H3 Build Conditioning` combines that
  reusable semantic result with structure-equivalent VAE references and an H3
  target latent. This permits low-resolution semantic images and separately
  encoded high-resolution DiT keyframes without rerunning Qwen.
- `Load MiniMax H3 Latent Upscaler` loads the attention-free 3D learned latent
  upscaler through ComfyUI's normal offload lifecycle. Place compatible weights
  from [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
  in `models/latent_upscale_models/`.
- `MiniMax H3 Latent Upscale` enlarges only the video stream by a continuous
  1x--4x multiplier and passes the audio stream through exactly. Its optional
  `CONDITIONING` input enlarges FL2AV first/last keyframe latents with the same
  learned model, while Ref2AV image/video/audio references retain their
  independent geometry. Without it, only the AV latent is processed. No text
  or VAE conditioning stage is rerun.
- `MiniMax H3 Video VAE Decode/Encode` provide fixed 256px H3
  tiling, full-overlap shared-core decode, a deterministic FP32 overlap
  epilogue, global multiband stitching, and ComfyUI-managed block-level weight
  prefetch. Their public tensors follow ComfyUI's configured VAE intermediate
  dtype while numerically sensitive accumulation remains FP32.
- `Patch MiniMax H3 Block Cache (Experimental)` skips stable transformer-block
  spans by reusing one exact trajectory residual. It provides conservative
  standard, 4-step, and 8-step profiles, isolates sampler branches, prefetches
  only blocks that actually execute, and follows ComfyUI's Dynamic VRAM and
  pinned-memory lifecycle. It is a Python-only patch and does not require
  rebuilding the CUDA package.
- `Video Motion Contact Sheet (Experimental)` samples an `N x N` chronological
  storyboard from a loaded `VIDEO` or decoded `IMAGE` frame batch. It can use
  uniform or motion-weighted sampling and optionally wraps each panel in
  annotated film rails so frame numbers and timestamps stay outside the image.
- `Patch Sol Sparse Attention` applies the production model-generic,
  loader-independent
  long-sequence sparse backend. It uses an input-adaptive statistical threshold,
  keeps one 64-token skipped-block centroid by default, accepts semantic
  multimodal layout metadata, and exposes integer dense-step safeguards,
  dense first/last-layer protection, the native integer W8A8 PV path by default, and an
  internal automatic short-sequence crossover.
- `Patch SLA Sparse Attention` implements the MiniMax H3 Turbo-SLA runtime as
  fixed-budget 128-query by 64-key Top-K routing. It shares Sol's semantic
  reference protection, dense step/layer scheduling, fused Q/K preprocessing,
  tensor lifetime, and optional W8A8 PV path, but deliberately does not add
  Sol's local blocks or skipped-block residual. Use it with the SLA-trained
  LoRA; `sparsity_ratio=0.85` matches the published runtime hyperparameter.
- `Patch Turing Attention Kernel Tuning (Experimental)` overrides the logical
  CTA-K schedule and the fused Hadamard/adaptive-anchor quality controls for
  dense W8A8 and Sol. Its defaults are the production policy; explicit values
  are intended for target-card profiling and do not affect stable Sage.

## MiniMax H3 automatic activation memory

The H3 adapter uses one capability-based path on Turing, Ampere, Ada, Hopper,
and newer Tensor Core GPUs. CUDA selects the cubin compiled for the installed
card; Python does not maintain a per-generation H3 algorithm. At each QKV or
FFN call, the adapter reads ComfyUI's live free/reclaimable memory and the
`--reserve-vram` ceiling:

- if the complete activation fits with safety headroom, it keeps the normal
  full-row path for maximum throughput;
- otherwise QKV projection is streamed by rows while retaining only INT8 Q/K
  and BF16 V, and the SwiGLU FFN is streamed into its final hidden output;
- if that compact state still does not fit, attention is evaluated in legal
  whole-head groups. Every group still attends over the complete sequence and
  keeps the selected backend (including explicit SDPA); ConvRot groups are
  split only on their 256-value boundary;
- at the extreme FFN floor, the intermediate width is split on the same
  256-value boundary. A first pass obtains the original whole-row scale, a
  second pass writes directly into the final compressed INT8 activation, and
  the original fused fc2 performs the complete contraction once;
- each layer's weights are cast/transferred once and reused by every row tile,
  so activation savings do not multiply Dynamic VRAM traffic.
- under AIMDO DynamicVRAM, resident VBAR pages are inspected through their
  resident/pinned flags. Only resident, unpinned pages count as reclaimable;
  inactive-model pages are released before current diffusion weights when a
  faster activation tier needs headroom. This avoids both pessimistic head
  sharding and the noisy `vbars_analyze` diagnostic path.
- the H3 video-VAE overlap accumulator uses the same sm75+ native capability
  gate, so Ampere does not lose that fused decode path.

No workflow socket or node changes are required. For a 16 GiB display card
that must leave 4 GiB to Windows and the compositor, launch ComfyUI with
`--reserve-vram 4`. The default `auto` mode then treats 12 GiB as a hard
inference ceiling even while the desktop is temporarily idle.

The policy can be diagnosed or overridden with these environment variables:

```text
COMFYUI_TURING_UTILS_H3_ACTIVATION_MODE=auto|throughput|balanced
COMFYUI_TURING_UTILS_H3_QKV_CHUNK_ROWS=16384
COMFYUI_TURING_UTILS_H3_MLP_CHUNK_ROWS=32768
COMFYUI_TURING_UTILS_H3_HEAD_GROUP=14
COMFYUI_TURING_UTILS_H3_FFN_CHUNK_CHANNELS=2048
```

Overrides are diagnostic controls; `auto` is the production default. QKV
streaming is available through bundled W8A8, Sol-W8A8, and SLA-W8A8 prepared
attention. Kernel 0.30 precomputes the adaptive K anchor from the same nine
global sequence locations and reuses it while writing every row tile directly
into the final Q/K storage. Row and head splitting therefore do not discard the
global anchor, RMSNorm, RoPE, orthogonal rotation, scale blocks, or any K/V row.

The automatic ladder is: full throughput, row streaming, compact prepared
Q/K, whole-head grouping, then two-pass FFN-channel grouping. It is selected
independently for each live operator, so a 12 GiB budget normally stops at row
streaming while a much tighter run can descend further. The final hidden output
and the chosen attention backend's irreducible state still have to fit; `auto`
cannot make an arbitrarily reference-heavy 15-second workflow fit 6 GiB.
None of these rungs requires Triton. If a Windows Kitchen build lacks its
optional fixed-workspace W8 entry point, large aligned contractions use the
bundled CUTLASS BF16-output kernel instead of allocating a full INT32 matrix.

Some Python symbols and extension filenames still contain `turing`/`sm75` for
backward ABI compatibility. They do not select a separate H3 implementation;
device-specific MMA/copy instructions are compile-time CUDA specializations.

## Turing behavior

When a model declares BF16 inference support but ComfyUI would otherwise fall
back to FP32 on exact sm75 Tensor Core GPUs, the plugin keeps activation storage
and bundled-kernel boundaries in BF16. Reductions and other precision-sensitive
internal arithmetic remain FP32. Explicit ComfyUI dtype flags still win.

The ConvRot path reuses comfy-kitchen W8A8 and W4A4 operators and supplies a
packed W4A8 SM75 Tensor Core kernel. Its row-buffer quantizers retain completed
rows in BF16 and use FP32 only for active rotation/reduction scratch. Launches
select the largest useful tile that fits the device's opt-in shared-memory
limit; shared-memory size or resident CTA count is not an acceptance target.
MiniMax-specific integration is isolated
under `comfyui_turing_utils/adapters/minimax/`, including packed-sequence VRAM
planning for text, keyframes, and multimodal references. Wan/Bernini integration
is isolated under `comfyui_turing_utils/adapters/`; it adds batch-aware,
per-reference-padded VRAM planning and, for supported Tensor Core attention calls,
the same single-owner Q/K/V lifetime and fused RMSNorm+RoPE+INT8 preprocessing
used by H3. This includes explicitly selected Sol calls; Sol remains opt-in.
Generic dtype, attention, and fused operators remain model-independent.

The bundled Sage backend accepts FP16/BF16 Q/K/V, GQA, causal attention,
unequal sequence lengths, HND/NHD layouts, and head dimensions up to 128. FP32
callers use BF16 boundary storage and receive FP32 output. On non-Turing GPUs,
the explicit `sage` choice uses ComfyUI's registered SageAttention backend.

The default `w8a8` attention backend uses the same bundled prepared-attention
path on sm75 and newer Tensor Core GPUs. Native builds select compile-time
architecture specializations; sm80+ uses asynchronous shared-memory copies and
the matching INT8 MMA implementation without Triton.
The bundled kernel retains stable Sage's INT8 Q/K score domain,
quantizes V channel-wise to signed INT8, packs online-softmax probabilities to
unsigned INT8, and evaluates both QK and PV with Turing Tensor Cores. It
supports FP16/BF16 storage, GQA, unequal sequence lengths, head dimensions
1--128, fixed HND/NHD layouts, upper-left causal masking, and native packed
varlen `[total_tokens, heads, dim]` inputs. Arbitrary masks remain unsupported.
The dense/sparse core has native D64 and D128 specializations, pads 1--63 only to
D64 and 65--127 only to D128, and retains the original softmax scale and output
width. The dense kernel uses a route-free specialization of the Sol exact-token
core; unsupported calls fall back through the pre-existing attention override.

Sol remains an independent patch rather than a loader option because its
quality/performance policy is intentionally configurable. Connect the model
through the Sol patch node to enable it explicitly. The
kernel accepts FP16/BF16/FP32 Q/K/V, GQA, head dimensions 1--128, unequal Q/K,
and unmasked non-causal sequences; incompatible or short calls use the selected
architecture-native dense backend. Automatic semantic protection requires separate Query/K layout
metadata for unequal sequences; ambiguous single-sequence metadata falls back
instead of applying the wrong ranges.

The bundled Sol core and its protected dense W8A8 path are native on sm75,
Ampere, Ada, and Hopper when those architectures are included in the kernel
build. Explicit Sage remains exact-sm75 and uses installed SageAttention on
newer GPUs. The extension filenames retain their historical `_sm75` suffix as
a Python ABI name; it no longer describes the only cubin that can be built.

Online Sol routing on Ampere or newer requires kernel package 0.28.0.
Adapter-protected Query blocks run through the
selected exact dense backend, while every sparse Query keeps protected modality
blocks as exact K/V sinks. Selected blocks reuse stable Sage's INT8 Tensor Core
QK path. Routing and exact selected-block QK both derive from the
same prequantized INT8 Q/K tensors and scales. Exact proxy/correction scores
remain in the post-Hadamard INT8 score domain, while route centroids are
inverse-transformed to the pre-Hadamard basis before diagonal threshold
statistics are formed. The orthogonal transform preserves centroid dot
products without estimating per-channel variance in the mixed basis. Each
Q-to-K-centroid Tensor Core score is reused for route selection and
skipped-block online-softmax correction, while original V means remain in the
value approximation.
Official-style `1x64` is the default; optional `2x32` improves bimodal
skipped-block fidelity without changing routing.

Kernel 0.23.0 retains the current ComfyUI attention tensor-container lifecycle
and adds adapter-owned fused Q/K preprocessing. H3 per-head RMSNorm plus
split-half RoPE, and Wan/Bernini whole-row RMSNorm plus interleaved RoPE, feed
the production INT8 Q/K representation without materializing normalized BF16
Q/K. Dense Sage, dense W8A8, Sol, and Sol-W8A8 share this path for H3 and
Wan/Bernini self-attention; protected Sol
steps/layers use the matching dense finalizer. Raw Q/K are released after the
fused preprocessing launch, and W8A8 releases raw V after V quantization. The
D128 preprocessing CTA uses at most about 21.1 KiB static shared memory; D64
uses about 10.6 KiB.

Internal CUDA phase timing is disabled by default and allocates no events. For
a bounded diagnostic run, set `COMFYUI_TURING_UTILS_PROFILE_CALLS` to the
number of attention calls to collect before one report is emitted. Kernel 0.31
also embeds the wheel's exact CUDA architecture set and, while this profiler is
enabled, reports the specialization CUDA selected for dense/Sol attention.
With DynamicVRAM, the report reuses the existing outer sampler fence instead
of synchronizing after an inner attention call, so profiling does not break
the asynchronous weight-prefetch pipeline.
`compiled_attention=[sm75,sm86] native_arch=True` proves that the wheel contains
an exact cubin for the active device. The native report additionally includes
`binary_sm`, `ptx_compute`, registers, shared/local memory, active CTAs and
occupancy; the historical `_sm75` extension filename remains only an ABI name.

Sol's `use_w8a8` switch is enabled by default. Selected exact blocks and
protected dense steps/layers use the
same signed-V/unsigned-probability Tensor Core path. Skipped-block correction
keeps original V centroids and FP32 online state, so the switch changes exact
PV throughput rather than the routing policy. Both variants keep a 64-query
tile. Native D64 uses 16 KiB dynamic shared memory, while D128 uses 32 KiB. The
automatic logical K schedule uses 64 tokens for short K and two sequential
64-token stages for K above 1024.
These are the current production geometries, not global resource limits.

Sol keeps the first denoising step and the first two transformer layers dense
by default. If `dense_prefix_layers + dense_suffix_layers` reaches or exceeds
the runtime layer count, every layer dispatches directly to the selected dense
W8A8 or Sage backend and skips all Sol preprocessing.

The `sdpa` option keeps ComfyUI's `AttentionTensorContainer` ownership path.
On exact-sm75, BF16 Q/K/V are consumed and converted one at a time to FP16 before
calling PyTorch SDPA, avoiding its slow BF16 math fallback while bounding the
period where floating input copies overlap. The output is restored to BF16.

The threshold and fixed +/- one-block local neighborhood execute inside each
attention CTA. Only compact, head-independent dense-Query and exact-KV policy
masks are stored globally; no full route map or follow-up popcount kernel is
materialized. MiniMax H3 publishes complete text, reference-image, reference-
video-anchor, reference-video-interior, reference-audio, target-audio, and
target-video spans. Reference-video first/last latent frames follow the image
sparsity switch; its interior follows the video switch. Defaults remain
image=false, video=true, audio=false. The current D64/D128 variants use 16/32
KiB respectively; larger candidates are accepted only when they improve real
SM75 latency without spilling. A40 compute_75 direction
tests validate numerical behavior and speed; final quality, occupancy, and
throughput still require an actual Turing GPU.

See [`docs/operator-support.md`](docs/operator-support.md) for the operator and
feature matrix, [`docs/turing-runtime.md`](docs/turing-runtime.md) for dispatch
and validation details, and [`docs/architecture.md`](docs/architecture.md) for the
Python/kernel layering. Experimental Sage1/Sage2 sources are not installed or
exposed by loader nodes.

## Kernel validation

```bash
COMFYUI_TURING_UTILS_ARCH_LIST="7.5+PTX" \
python -m pip install -v --no-build-isolation -e ./kernel
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark --sol
python kernel/scripts/release_gate.py --build --device cuda:0
python kernel/scripts/benchmark_backends.py --device cuda:0 --suite all
python kernel/scripts/diagnose_runtime.py --device cuda:0
python kernel/scripts/benchmark_arch_matrix.py --devices 0,1 --suite all
```

Compatible A40 runs validate numerical behavior and allocation shapes but do
not replace final exact-sm75 occupancy and end-to-end testing.
For native A40 validation, build with
`COMFYUI_TURING_UTILS_ARCH_LIST="8.6"`; this emits sm86 cubins and enables the
Ampere async-copy and INT8 MMA specializations rather than JITing compute_75.

`diagnose_runtime.py` reports hardware shared-memory limits, installed kernel
ABI features, and live allocator state as JSON. `benchmark_arch_matrix.py`
runs identical arguments serially on multiple local GPUs and writes one JSON
artifact, so a 2080 Ti and 3070 build can be compared without mixing warmups,
shapes, or backend scope. Capability checks inspect the compiled extension's
real symbols as well as its Python version, so a stale editable-build binary
is reported and safely excluded from scheduling instead of failing mid-run.

## License

Apache-2.0. See `kernel/LICENSE`, `kernel/NOTICE`, and `kernel/LICENSES/`.
