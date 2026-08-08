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
- the independently installed `comfyui-turing-utils-kernel>=0.8.0` on exact sm75

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
- `Bernini Inpaint Condition` starts sampling from the source-video latent,
  supports local or global repainting, and optionally adds the source as aligned
  context tokens.
- `MiniMax H3 Reference Condition (Hub)` feeds fixed Hub inputs into H3's native
  picture/video/audio reference protocol.
- `Bernini Context Windows` applies reference-aware Wan context windows with
  selectable absolute or official relative temporal positions.
- `Wan Video Frames Padding` exposes Wan-compatible frame padding.
- `MiniMax H3 Video Frames Padding` pads to H3's `17*n+5` frame grid.
- `Patch H3 Progressive Resolution (Experimental)` keeps one final-resolution
  H3 video/audio latent but can evaluate an initial low stage and a following
  medium stage at independently configured video short edges. Stage names
  describe execution order rather than size; either target may be larger than
  the other. Audio stays untouched, and already-encoded first/last-frame latents
  can be resized and cached without running conditioning a second time.
- `Patch Sol Sparse Attention (Experimental)` applies the model-independent
  long-sequence sparse backend. It uses an input-adaptive statistical threshold,
  keeps skipped-block centroid residuals (two 32-token centroids by default),
  accepts semantic prefix/video topology metadata, and exposes stable route
  budgets, integer dense-step safeguards, dense first/last-layer protection,
  and an internal automatic short-sequence crossover.
- `Patch Sage Frame Sparse Attention (Experimental)` applies a lower-overhead
  structured video path. Non-video queries remain on stable Sage; video queries
  attend the exact semantic prefix, complete local latent frames, fixed sink
  frames, and layer-rotated periodic anchor frames. It has no online routing or
  centroid residuals and is never selected by a loader or `auto`.

## Turing behavior

When a model declares BF16 inference support but ComfyUI would otherwise fall
back to FP32 on exact sm75 Tensor Core GPUs, the plugin keeps activation storage
and bundled-kernel boundaries in BF16. Reductions and other precision-sensitive
internal arithmetic remain FP32. Explicit ComfyUI dtype flags still win.

The ConvRot path reuses comfy-kitchen W8A8 and W4A4 operators and supplies a
packed W4A8 SM75 Tensor Core kernel. Its row-buffer quantizers retain completed
rows in BF16, use FP32 only for active rotation/reduction scratch, and stay under
the default 48 KiB shared-memory limit. MiniMax-specific integration is isolated
in `minimax_adapter.py`, including packed-sequence VRAM planning for text,
keyframes, and multimodal references. Wan/Bernini integration is isolated in
`wan_adapter.py`; it adds only batch-aware, per-reference-padded VRAM planning
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
staged interval. Spatial condition areas, masks, controls, and GLIGEN are
currently unsupported and make that model call fall back to full resolution.
Peak memory is still set by the later final-resolution steps, and final quality
and speed require local workflow validation.

The bundled Sage backend accepts FP16/BF16 Q/K/V, GQA, causal attention,
unequal sequence lengths, HND/NHD layouts, and head dimensions up to 128. FP32
callers use BF16 boundary storage and receive FP32 output. On non-Turing GPUs,
the plugin prefers an installed SageAttention backend and then follows ComfyUI's
normal fallback order.

Sparse attention is not a loader option and `auto` never selects it. Connect the
model through one of the experimental patch nodes to enable it explicitly. The
current kernels accept FP16/BF16/FP32 Q/K/V,
GQA, 128-dimensional heads, and unmasked non-causal sequences; incompatible or
short calls use bundled stable Sage. Semantic-prefix Query rows run through
stable Sage, while sparse target Query rows keep the prefix as an exact K/V
sink. Selected sparse blocks reuse stable Sage's INT8 Tensor Core QK path. Sol
routing requires kernel package 0.13.0. Routing uses original FP16/BF16 Q/K
centroids, while skipped-block score estimates are reconstructed from the same
INT8 Q/K tensors and scales as selected blocks. Original V means remain in the
value path. `skipped_residual=2x32` improves bimodal skipped-block fidelity;
`1x64` remains available for maximum speed. Optional route-density bounds clamp
adaptive choices per 16-key tile without removing forced prefix/local/temporal
blocks. The K/V summaries are produced in one fused scan; the attention kernel
reconstructs its summary Q tile from the existing INT8 Q buffer instead of
rereading the original Q tensor. The sparse CTA uses 32 KiB of shared memory so
SM75 can admit two CTAs when registers permit. Final quality/performance testing
on an actual Turing GPU remains required.

The frame-sparse ABI requires kernel package 0.15.0. It consumes a cached,
head-independent CSR schedule and evaluates only complete selected 64-token K/V
blocks with the stable Sage math path. `frame_window` selects complete local,
sink, and periodic anchor frames. `radial` keeps complete nearby frames, then
uses exact 8x8 spatial-token neighborhoods at logarithmically increasing
temporal strides. MiniMax H3 supplies the packed target-video boundary,
tokens-per-frame, and exact spatial-token height/width through its adapter;
unknown layouts fall back to stable Sage instead of guessing.

The node defaults to `custom` plus the established `frame_window` policy, so an
old workflow does not silently change. `conservative`, `balanced`, and `fast`
are coherent presets; selecting one intentionally overrides pattern, coverage,
anchors, sinks, radial controls, and protected layer counts. On an A40 executing
compute_75 code, 56-head BF16 attention measured 187/577 ms for 480p/720p-like
`frame_window` shapes and 178/442 ms for the balanced radial settings. Their
720p block densities were 15.1% and 11.4%. The radial attention component was
about 0.74x the earlier ~600 ms 480p dense Sage component, but total model-step
time still includes QKV, MLP, normalization, and activation traffic proportional
to the larger sequence. Exact-sm75 end-to-end and visual validation remains
required.

The retained CUDA CTA handles one Q64 block, uses 32 KiB dynamic shared memory,
and has zero stack/local spill. An evaluated Q128/40 KiB design kept the same
eight resident warps per SM75 but regressed the 720p-like main kernel from
73.67 ms to 85.87 ms at eight heads, so it is not shipped.

See [`docs/turing-runtime.md`](docs/turing-runtime.md) for the dispatch and
validation matrix. Experimental Sage1/Sage2 sources are not installed or
exposed by loader nodes.

## Kernel validation

```bash
COMFYUI_TURING_UTILS_ARCH_LIST="7.5+PTX" \
python -m pip install -v --no-build-isolation -e ./kernel
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark --experimental-sparse
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark --experimental-frame-sparse
```

Compatible A40 runs validate numerical behavior and allocation shapes but do
not replace final exact-sm75 occupancy and end-to-end testing.

## License

Apache-2.0. See `kernel/LICENSE`, `kernel/NOTICE`, and `kernel/LICENSES/`.
