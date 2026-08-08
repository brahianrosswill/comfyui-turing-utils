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
| `Patch Sol Sparse Attention` | fused 64-token centroid threshold routing; selected tiles use FP16 Tensor Cores | input-adaptive `mean + tau * std` threshold | exact FP16 V tiles plus skipped-block V centroids, FP32 online accumulation |

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

`min_sequence_tokens=0` selects the measured 4096-token crossover; a positive
value remains a manual override. `routing_threshold=1.0` routes a block when
its centroid score exceeds the current query row's projected mean by one
standard deviation; lowering it evaluates more blocks exactly. The default
`dense_warmup_ratio=0.25` protects one step in a four-step workflow,
`dense_tail_ratio=0` avoids an extra dense tail by default, and
`dense_prefix_layers=2` follows the validated H3 policy when layer metadata is
available. Step lookup is cached once per sampler timestep rather than
synchronized in every transformer block.

The configurable local neighborhood remains exact. Prefix policy can use exact
model-supplied semantic layout metadata, disable prefix protection, or accept a
manual token count; `auto` does not guess a prefix for an unknown model. MiniMax
publishes the actual target-video boundary, so text, reference, and target-audio
rows plus their cross-modal attention remain exact without tying the generic
backend to H3. It also publishes target-video frame geometry; the kernel can
keep matching spatial ranges in adjacent frames exact without reordering tokens.
Other blocks use the fused statistical threshold. Skipped blocks retain
approximate mass and output through K/V centroids without the former K-variance
overweighting. The exact path uses FP16 Tensor Core QK/PV
with FP32 softmax and output accumulation and stays within the default 48 KiB
shared-memory limit. Routing is written directly from the centroid Tensor Core
tile, so no full proxy-score matrix is materialized.

On an A40 JITing only compute_75 PTX, 56-head BF16 uniform synthetic tests at
`routing_threshold=1.0` with one adjacent temporal frame selected about 22.1%,
19.1%, and 17.5% of blocks and measured 1.85x, 2.26x, and 2.37x kernel speedups
over bundled Sage at 4096, 8192, and 16384 tokens. At 16K, temporal protection
adds roughly 0.7 density points and 2.3% sparse-kernel time. These are kernel-level
compatibility results, not an H3 end-to-end or quality prediction. Exact-sm75
visual and performance testing remains required before the backend can leave
experimental status.

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

## Validation boundary

Release builds target sm75 for bundled Sage. Static tests validate dispatch,
fallbacks, loader independence, shapes, dtypes, the 48 KiB policy, the public
symbol boundary, and exclusion of the retired Sage1/Sage2 variants. For compatible A40
validation, build with:

```bash
COMFYUI_TURING_UTILS_ARCH_LIST="7.5+PTX" \
python -m pip install -v --no-build-isolation -e ./kernel
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark --experimental-sparse
python kernel/scripts/validate_wan_fusions.py --device cuda:0
```

An A40 run validates numerical behavior, allocation shapes, and the absence of
Ampere-only source dependencies. It JITs compute_75 PTX and selects the same
CTA schedule used on sm75. This does not replace the final exact-sm75 occupancy
and end-to-end test.
