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
- `Patch Sol Sparse Attention (Experimental)` applies the model-independent
  long-sequence sparse backend. It uses an input-adaptive statistical threshold,
  keeps skipped-block centroid residuals, accepts semantic prefix/video topology
  metadata, and exposes dense step/layer safeguards plus an automatic
  short-sequence crossover.

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

The bundled Sage backend accepts FP16/BF16 Q/K/V, GQA, causal attention,
unequal sequence lengths, HND/NHD layouts, and head dimensions up to 128. FP32
callers use BF16 boundary storage and receive FP32 output. On non-Turing GPUs,
the plugin prefers an installed SageAttention backend and then follows ComfyUI's
normal fallback order.

Sparse attention is not a loader option and `auto` never selects it. Connect the
model through `Patch Sol Sparse Attention (Experimental)` to enable it on any
compatible attention call. The current kernel accepts FP16/BF16/FP32 Q/K/V,
GQA, 128-dimensional heads, and unmasked non-causal sequences; incompatible or
short calls use bundled stable Sage. It requires kernel package 0.11.0; optional
route-density debug logging requires 0.11.1. Final
quality/performance testing on an actual Turing GPU.

See [`docs/turing-runtime.md`](docs/turing-runtime.md) for the dispatch and
validation matrix. Experimental Sage1/Sage2 sources are not installed or
exposed by loader nodes.

## Kernel validation

```bash
COMFYUI_TURING_UTILS_ARCH_LIST="7.5+PTX" \
python -m pip install -v --no-build-isolation -e ./kernel
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark --experimental-sparse
```

Compatible A40 runs validate numerical behavior and allocation shapes but do
not replace final exact-sm75 occupancy and end-to-end testing.

## License

Apache-2.0. See `kernel/LICENSE`, `kernel/NOTICE`, and `kernel/LICENSES/`.
