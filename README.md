# ComfyUI Turing Utils

Compatibility and performance extensions that fill gaps in ComfyUI on older
NVIDIA Turing GPUs. The plugin currently provides ConvRot W8A8/W4A8/W4A4
support, exact-sm75 BF16 activation storage, bundled Turing Sage/W8A8
attention, native sm75+ Sol sparse attention, and focused Wan/Bernini utilities.

## Requirements

- NVIDIA GPU with CUDA support
- Turing, Ampere, Ada, or newer architecture
- Python 3.10 or newer
- PyTorch with CUDA and ComfyUI
- `comfy-kitchen>=0.2.26` for ConvRot model integration
- the independently installed `comfyui-turing-utils-kernel>=0.28.0` for native
  Sol on Ampere or newer; exact-sm75 installs also provide the local dense
  attention and quantized-linear paths

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
- `H3 Concat AV Latent` combines standalone H3 video and audio latents into the
  model's native nested AV latent. `H3 Separate AV Latent` splits the streams
  again; both nodes preserve matching video/audio noise masks.
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
- `Patch Turing Attention Kernel Tuning (Experimental)` overrides the logical
  CTA-K schedule and the fused Hadamard/adaptive-anchor quality controls for
  dense W8A8 and Sol. Its defaults are the production policy; explicit values
  are intended for target-card profiling and do not affect stable Sage.

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
per-reference-padded VRAM planning and, for supported Turing attention calls,
the same single-owner Q/K/V lifetime and fused RMSNorm+RoPE+INT8 preprocessing
used by H3. This includes explicitly selected Sol calls; Sol remains opt-in.
Generic dtype, attention, and fused operators remain model-independent.

The bundled Sage backend accepts FP16/BF16 Q/K/V, GQA, causal attention,
unequal sequence lengths, HND/NHD layouts, and head dimensions up to 128. FP32
callers use BF16 boundary storage and receive FP32 output. On non-Turing GPUs,
the explicit `sage` choice uses ComfyUI's registered SageAttention backend.

The default `w8a8` attention backend uses the bundled exact-sm75 kernel on
supported Turing GPUs and Comfy Kitchen INT8 attention on newer architectures.
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

The bundled Sol core is native on sm75, Ampere, Ada, and Hopper when those
architectures are included in the kernel build. Exact-sm75 protected dense
steps/layers use bundled Sage or W8A8. On newer GPUs they delegate to the
installed SageAttention or Comfy Kitchen W8A8 backend, so Sol does not replace
an architecture-specific dense implementation with the Turing schedule. The
extension filenames retain their historical `_sm75` suffix as a Python ABI
name; it no longer describes the only cubin that can be built.

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
number of attention calls to collect before one report is emitted.

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
```

Compatible A40 runs validate numerical behavior and allocation shapes but do
not replace final exact-sm75 occupancy and end-to-end testing.
For native A40 validation, build with
`COMFYUI_TURING_UTILS_ARCH_LIST="8.6"`; this emits sm86 cubins and enables the
Ampere async-copy and INT8 MMA specializations rather than JITing compute_75.

## License

Apache-2.0. See `kernel/LICENSE`, `kernel/NOTICE`, and `kernel/LICENSES/`.
