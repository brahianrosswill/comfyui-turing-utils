# ComfyUI Turing Utils

Compatibility and performance extensions that fill gaps in ComfyUI on older
NVIDIA Turing GPUs. The plugin currently provides ConvRot W8A8/W4A8/W4A4
support, exact-sm75 BF16 activation storage, bundled Turing Sage attention, and
Wan/Bernini context-window utilities.

## Requirements

- NVIDIA GPU with CUDA support
- Turing, Ampere, Ada, or newer architecture
- Python 3.10 or newer
- PyTorch with CUDA and ComfyUI
- `comfy-kitchen>=0.2.26` for ConvRot model integration
- the independently installed `comfyui-turing-utils-kernel>=0.20.0` on exact sm75

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wjie98/comfyui-turing-utils.git
cd comfyui-turing-utils
python -m pip install -v --no-build-isolation -e ./kernel
```

The custom node and CUDA package have separate installation lifecycles.
Python-only plugin updates never invoke a compiler or JIT; rebuild the kernel
only after its CUDA sources or required version change.

## Nodes

- `Load ConvRot DiT` loads ComfyUI ConvRot diffusion models. It supports W8A8,
  W4A8, and W4A4 dispatch and selects bundled Sage automatically on exact sm75.
- `Load ConvRot CLIP` loads a ConvRot text encoder independently of the DiT.
- `Reference Image/Video/Audio Hub` collects ordered heterogeneous references.
  Hubs can be chained; image/video resizing follows KJ Resize Image v2 controls,
  zero width/height defaults disable resizing, and an optional positive video
  frame count trims or pads only at the end.
- `Optional Resize Image v2` mirrors KJ Resize Image v2's controls, defaults,
  dimensions, mask output, and resize/crop/pad behavior. Its image socket is
  optional; without an image it returns no image, allowing one node to feed an
  optional first- or last-frame condition socket without fabricating a frame.
- `Bernini Inpaint Condition` starts sampling from the source-video latent,
  supports local or global repainting, and optionally adds the source as aligned
  context tokens.
- `MiniMax H3 Reference Condition (Hub)` feeds fixed Hub inputs into H3's native
  picture/video/audio reference protocol.
- `Bernini Context Windows` applies reference-aware Wan context windows with
  selectable absolute or official relative temporal positions.
- `Wan Video Frames Padding` exposes Wan-compatible frame padding.
- `MiniMax H3 Video Frames Padding` pads to H3's `17*n+5` frame grid.
- `H3 Concat AV Latent` combines standalone H3 video and audio latents into the
  model's native nested AV latent. `H3 Separate AV Latent` splits the streams
  again; both nodes preserve matching video/audio noise masks.
- `Resize MiniMax H3 AV Latent (Experimental)` stretches only the packed H3
  video latent to an exact independently selected width and height on the
  model's 32-pixel grid, while preserving the audio latent unchanged. It is
  intended for explicitly separated low- and high-resolution sampling stages
  and performs no cropping, padding, re-noising, or sigma conversion. An
  optional conditioning input can resize first/last H3 keyframe latents onto
  the same grid; independent picture/video reference latents are not changed.
- `Patch H3 Progressive Resolution (Experimental)` keeps one final-resolution
  H3 video/audio latent but can evaluate an initial low stage and a following
  medium stage at independently configured video short edges. Stage names
  describe execution order rather than size; either target may be larger than
  the other. Audio stays untouched, and already-encoded first/last-frame latents
  can be resized and cached without running conditioning a second time.
- `Patch Sol Sparse Attention (Experimental)` applies the model-generic,
  loader-independent
  long-sequence sparse backend. It uses an input-adaptive statistical threshold,
  keeps one 64-token skipped-block centroid by default, accepts semantic
  multimodal layout metadata, and exposes integer dense-step safeguards,
  dense first/last-layer protection, an optional SM75 W8A8 PV path, and an
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
rows in BF16, use FP32 only for active rotation/reduction scratch, and stay under
the default 48 KiB shared-memory limit. MiniMax-specific integration is isolated
under `comfyui_turing_utils/adapters/minimax/`, including packed-sequence VRAM
planning for text, keyframes, and multimodal references. Wan/Bernini integration
is isolated under `comfyui_turing_utils/adapters/`; it adds only batch-aware,
per-reference-padded VRAM planning
and leaves Wan block normalization, projections, attention dispatch, and
feed-forward execution to ComfyUI.
Generic dtype, attention, and fused operators remain model-independent.

The progressive-resolution patch is explicit and is never selected by a
loader. Build H3 conditioning once at the final output size, then connect the
model through the patch before constructing the guider or KSampler. The sampler
continues to own the final-resolution noisy state. The first
`low_resolution_steps` evaluations use `low_short_edge`; the following
`medium_resolution_steps` evaluations use `medium_short_edge`, so their sum is
the complete staged interval. During those steps the patch temporarily repacks
a staged-resolution video stream with the unchanged audio stream, runs the
normal H3 conditioning path, and enlarges the denoised video prediction before
returning it to the sampler. A target at or above the final short edge is a
no-op for that stage. H3 rebuilds its packed layout from the temporary latent
shape. `sigma_blend` is the default input policy: it blends
noise-variance-preserving nearest sampling with area filtering over the complete
staged interval. The returned denoised prediction preserves the sampler's
high-resolution residual, which is algebraically equivalent to resizing H3's
flow velocity instead of repeatedly projecting the sampler state into the
staged-resolution subspace. The configured low and medium step counts remain
fully user-controlled, including all-staged configurations intended for
diagnostics. Spatial condition areas, masks, controls, and GLIGEN are
currently unsupported and make that model call fall back to full resolution.
Peak memory is still set by the later final-resolution steps, and final quality
and speed require local workflow validation.

The bundled Sage backend accepts FP16/BF16 Q/K/V, GQA, causal attention,
unequal sequence lengths, HND/NHD layouts, and head dimensions up to 128. FP32
callers use BF16 boundary storage and receive FP32 output. On non-Turing GPUs,
the plugin prefers an installed SageAttention backend and then follows ComfyUI's
normal fallback order.

The explicit experimental `w8a8` attention backend is exact-sm75-only and is
never selected by `auto`. It retains stable Sage's INT8 Q/K score domain,
quantizes V channel-wise to signed INT8, packs online-softmax probabilities to
unsigned INT8, and evaluates both QK and PV with Turing Tensor Cores. It is
currently specialized for unmasked, non-causal heads in the range 1--128 with
FP16/BF16 storage, GQA, unequal sequence lengths, and HND/NHD layouts. The
dense/sparse core pads smaller heads internally while retaining the original
softmax scale and output width. The
dense kernel uses a route-free specialization of the Sol exact-token core;
unsupported calls fall back through the pre-existing attention override.

Sparse attention is not a loader option and `auto` never selects it. Connect
the model through the experimental Sol patch node to enable it explicitly. The
kernel accepts FP16/BF16/FP32 Q/K/V, GQA, head dimensions 1--128, unequal Q/K,
and unmasked non-causal sequences; incompatible or short calls use bundled
stable Sage. Automatic semantic protection requires separate Query/K layout
metadata for unequal sequences; ambiguous single-sequence metadata falls back
instead of applying the wrong ranges.

Online Sol routing, its optional W8A8 mode, and the explicit dense W8A8 backend
require kernel package 0.20.0. Adapter-protected Query
blocks run through the selected exact dense backend, while every sparse Query
keeps protected modality blocks as exact K/V sinks. Selected blocks reuse stable Sage's INT8
Tensor Core QK path. Routing and exact selected-block QK both derive from the
same prequantized INT8 Q/K tensors and scales. Each Q-to-K-centroid Tensor Core
score is reused for route selection and skipped-block online-softmax
correction, while original V means remain in the value approximation.
Official-style `1x64` is the default; optional `2x32` improves bimodal
skipped-block fidelity without changing routing.

Kernel 0.20.0 integrates current ComfyUI's attention tensor-container
lifecycle. Supported bundled Sage, W8A8, and Sol calls quantize
before output allocation and release their original Q/K/V storage as soon as
the selected path permits. Older kernel packages continue through the
compatible one-call path, but do not receive this peak-memory improvement.

Sol's `use_w8a8` switch is disabled by default for backward-compatible quality.
When enabled, selected exact blocks and protected dense steps/layers use the
same signed-V/unsigned-probability Tensor Core path. Skipped-block correction
keeps original V centroids and FP32 online state, so the switch changes exact
PV throughput rather than the routing policy. Both variants keep a 64-query
tile and 32 KiB dynamic shared-memory budget. The automatic logical K schedule
uses 64 tokens for short K and two sequential 64-token stages for K above 1024.
The latter does not hold 128 K/V tokens in shared memory, preserving CTA density.

The threshold and fixed +/- one-block local neighborhood execute inside each
attention CTA. Only compact, head-independent dense-Query and exact-KV policy
masks are stored globally; no full route map or follow-up popcount kernel is
materialized. MiniMax H3 publishes complete text, reference-image, reference-
video-anchor, reference-video-interior, reference-audio, target-audio, and
target-video spans. Reference-video first/last latent frames follow the image
sparsity switch; its interior follows the video switch. Defaults remain
image=false, video=true, audio=false. The CTA remains at 32 KiB shared memory. A40 compute_75 direction
tests validate numerical behavior and speed; final quality, occupancy, and
throughput still require an actual Turing GPU.

See [`docs/turing-runtime.md`](docs/turing-runtime.md) for the dispatch and
validation matrix and [`docs/architecture.md`](docs/architecture.md) for the
Python/kernel layering. Experimental Sage1/Sage2 sources are not installed or
exposed by loader nodes.

## Kernel validation

```bash
COMFYUI_TURING_UTILS_ARCH_LIST="7.5+PTX" \
python -m pip install -v --no-build-isolation -e ./kernel
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark --experimental-sparse
python kernel/scripts/release_gate.py --build --device cuda:0
```

Compatible A40 runs validate numerical behavior and allocation shapes but do
not replace final exact-sm75 occupancy and end-to-end testing.

## License

Apache-2.0. See `kernel/LICENSE`, `kernel/NOTICE`, and `kernel/LICENSES/`.
