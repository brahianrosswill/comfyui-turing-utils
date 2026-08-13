# MiniMax H3 Video VAE nodes

`MiniMax H3 Video VAE Encode (Experimental)` and
`MiniMax H3 Video VAE Decode (Experimental)` isolate H3-specific memory and
compute experiments from ComfyUI's global VAE settings.

The nodes validate that the supplied VAE is a `MiniMaxH3VideoVAE`; they do not
patch the loader or alter another model's behavior. Latents and finalized IMAGE
outputs remain FP32 for ComfyUI compatibility.

## Dynamic weight lifecycle

Both nodes automatically use the VAE's existing `ModelPatcher` and AIMDO/VBAR
allocations:

1. reserve memory using the maximum of ComfyUI's H3 estimate and the
   tile-aware H3 CNN/transformer workspace estimate;
2. fault the first weight stage, asynchronously when ComfyUI streams are enabled;
3. start the next stage transfer while the current stage computes when streams
   are available;
4. retain completed stages for all spatial tiles and temporal chunks in this
   invocation;
5. unpin every stage and ask the patcher to release the VBAR pages when the
   invocation ends, including exception and failed-prefetch paths.

If the complete retained set does not fit, the session releases its retained
prefix and continues with ComfyUI's synchronous dynamic streaming. This avoids
letting a partial pin consume the temporary workspace needed by later stages.
It is a runtime capacity decision, not a GPU-model whitelist.

Disabling ComfyUI asynchronous streams disables transfer/compute overlap but
does not disable cross-tile weight retention.

## Independent options

- Spatial tiling is fixed to H3's quality-stable 256px edge and 64px overlap.
  `tiles_per_batch` controls how many independent tiles execute together.
  `auto` chooses the largest useful batch predicted to fit ComfyUI's current
  available/reclaimable memory, capped at 2 for the encoder and 4 for the
  decoder based on static A40 throughput. Numeric values are strict requests
  capped only by the number of tiles in the frame.
- Decoder `decoder_tile_size` is 256 by default. Larger numeric values are
  retained only as explicit experiments. They keep the spatial RoPE increment
  of a 256px tile instead of renormalizing every larger tile back to `[-1, 1]`,
  but they still change transformer context and are not quality-equivalent to
  the official geometry. In particular, an 18-token/288px tile must either
  compress its coordinate step to stay inside `[-1, 1]` or preserve the
  16-token step and extend its endpoints to `+/-1.0625`; neither matches the
  trained 16-token/256px distribution. Encoder geometry is never changed.
- Decoder `decoder_tiling=official` preserves ComfyUI's independent-window
  equations and final linear pixel ramps exactly.
- `official_multiband` preserves every independent 256px decoder window but
  replaces the order-dependent final ramps with normalized two-dimensional
  overlap-add. Low frequencies use a broad cosine transition; high frequencies
  use a center-biased transition so color/illumination can agree without
  averaging fine texture as aggressively. It does not reduce transformer token
  work and is the seam-quality control mode.
- `shared_core` is the primary token-saving experiment. Every physical image
  token belongs to the nearest window center and computes Q, attention output,
  output projection, residual, and MLP exactly once. Its K/V context remains
  the complete 256px owner window plus that window's independent register and
  suffix states. It uses true asymmetric Q/K attention without masks or padded
  queries. Local RoPE remains the original coordinate inside the owner window.
- `shared_overlap` remains available only as the earlier comparison path. It
  blends window attention outputs after every transformer block and can visibly
  soften detail. Both shared modes require 256px windows. `tiles_per_batch`
  controls how many compatible windows enter one launch and does not change a
  mode's result.
- decoder `attention` selects SDPA, Sage, or W8A8-QK for this VAE call only.
  SDPA is the default. On Turing, container-owned BF16 Q/K/V are converted to
  FP16 before SDPA so PyTorch does not select the BF16 math fallback.

The following behaviors are automatic rather than node switches:

- VAE activations follow the dtype selected when ComfyUI loaded the VAE. This
  defaults to FP16 for H3 on supported NVIDIA devices, including Turing, while
  still respecting an explicit global `--fp32-vae` override; normalized latent
  outputs remain FP32;
- a compatible TensorWise W8A8 `w2` quantizer consumes SwiGLU directly; dense
  weights retain the eager path;
- two reusable pinned FP32 pixel buffers overlap transfers with compute;
- independent spatial tiles are concatenated on the batch dimension, never on
  the sequence dimension. Attention therefore cannot cross tile boundaries and
  no block-diagonal mask or masked recomputation is required;
- encoder copies convert directly from pinned FP32 source storage into GPU
  buffers of the selected activation dtype, avoiding two persistent FP32 GPU
  clips and a separate activation cast in the default FP16 path;
- completed dynamic weight stages remain resident across all spatial tiles and
  temporal chunks, and are released after the invocation;
- decoder attention receives single-owner Q/K/V containers. Turing Sage and
  W8A8 can additionally fuse Q/K RMSNorm, RoPE, and quantization through the
  same prepared-attention protocol used by diffusion-model adapters.

If Windows or the current RAM budget cannot provide the pinned buffers, the
nodes preserve correctness and fall back to synchronous FP32 copies.

Both nodes publish matching ComfyUI and terminal `tqdm` progress bars. Their
total is the number of spatial tiles multiplied by the number of temporal
clips/chunks. Every kernel batch advances both bars by the actual number of
tiles in that batch, including a shorter final batch. CUDA events feed the bars
from a small background consumer only after each batch really finishes; this
keeps the ETA meaningful without synchronizing the main submission stream or
disabling transfer/compute overlap.

In either shared mode a window is revisited by every decoder transformer block,
so the progress total additionally includes the decoder block count.
This reports completed window-block work rather than holding at zero until the
entire merged frame finishes.

## Experimental shared-core decoder

The official decoder repeats all work for every overlap copy. For each block,
shared-core instead:

1. keeps one global image-token state and projects its Q/K/V once;
2. assigns each token to the spatial window whose center is nearest, producing
   contiguous non-overlapping query cores;
3. gathers a full 256px K/V halo for each owner window;
4. gathers only that window's core Q rows and their matching local RoPE rows;
5. appends the independent per-window suffix queries and suffix K/V;
6. runs real `Qcore+suffix x KVwindow+suffix` attention; no attention mask or
   padded fake queries are materialized;
7. scatters every core output directly to its unique global tokens and updates
   output projection/MLP once; and
8. updates suffix residual/MLP state separately for every window.

For a 480x848 frame with five resident temporal latent tokens, the official
window grid contains 19,200 image-query tokens per block while shared-core owns
7,950. K/V still contains the complete window halos, so the score contraction
is reduced on the Q axis rather than converted into global attention. The
approximation is the hard owner transition: adjacent cores can use different
local RoPE coordinates and suffix states. It avoids the repeated hidden-state
averaging that blurred `shared_overlap`, but still requires target-checkpoint
visual testing for fixed ownership seams.

`shared_core` uses existing asymmetric support in SDPA and the bundled Turing
Sage/W8A8 kernels. It changes Python scheduling only and does not require a
kernel rebuild.

## Official multiband stitching

`official_multiband` runs the same independent decoder windows as `official`.
Each final pixel tile is split into an 8x-downsampled bilinear low-frequency
component and a residual high-frequency component. Both components use
two-dimensional cosine windows normalized over every tile that covers a pixel;
the high-frequency window is raised to the fourth power to prefer tile centers.
This handles triple/four-way coverage without depending on traversal order and
preserves identical tile outputs within floating-point rounding. Normalization
and the final overlap-add canvas use FP32 even when decoder activations use
FP16. Its extra interpolation and pixel accumulation are small relative to the
unchanged transformer work.

## Experimental shared-overlap decoder

The official decoder evaluates every overlapping 256px tile as a completely
independent sample and blends only final pixels. A physical overlap token can
therefore have two different hidden states, local RoPE coordinates, and
attention contexts. Removing one copy cannot be bitwise equivalent.

The experimental path instead performs the following at each decoder block:

1. keep one global image-token lattice for the temporal chunk;
2. apply block RMSNorm and QKV projection once to each unique image token;
3. gather projected Q/K/V into the original 256px windows;
4. append independent per-window register and suffix states;
5. apply each window's original local RoPE and execute packed window batches
   without a dense or block-diagonal mask;
6. blend overlap attention outputs back to the unique lattice in the official
   order, using the mean per-pixel ramp of each 16px output patch as that
   hidden token's blend weight;
7. apply the attention output projection, residual, and SwiGLU MLP once to each
   unique image token;
8. retain register/suffix residual and MLP state independently per window.

The approximation occurs at step 6: hidden states merge after every block
instead of only after final pixel projection. This is intended to preserve the
trained window size and local position scale while eliminating redundant
token-wise work; it still requires real-checkpoint visual validation. A frame
that fits in one tile automatically uses the exact official path because there
is no overlap to share.

## Current validation status

The custom decoder and encoder equations are tested against ComfyUI's reference
implementations, including temporal chunk reconstruction and tiled assembly.
The retained-weight lifecycle is tested for both success and allocation-failure
cleanup.

On an A40 with the complete 2.6B-parameter H3 VAE architecture and identical
zeroed FP16 weights, the custom single-tile-batch 256 path matched the official
runtime and output exactly. Direct real-checkpoint testing later showed visible
grids for non-256 tiles, even though the custom arbitrary-size assembly was
bitwise equal to ComfyUI's reference implementation with the same overridden
tile size. This isolates the problem to H3's tile-conditioned decoder behavior,
not the overlap copier, and is why automatic geometry selection was removed.

The batch-aware estimator scales the H3 CNN/transformer workspace with
`tiles_per_batch` while counting the full decoded canvas, finalized pixel copy,
and encoder double buffers only once. On the same A40 setup, decoder batch
1/2/4 took 0.673/0.325/0.313 seconds, while encoder batch 1/2/4 took
2.795/2.560/2.567 seconds with 2.37/4.36/8.33 GiB peaks. Every batched result
was bitwise equal to batch 1. Real-checkpoint throughput and peak memory still
need validation on the target Turing card.

With a fully initialized 36-layer FP16 H3 decoder on an A40, SDPA, and seven
latent temporal tokens, shared overlap changed 480x848 from 2.377 s to 1.461 s
(1.63x) and 720x1280 from 4.109 s to 2.991 s (1.37x). Peaks changed from
5.26/5.34 GiB to 5.37/5.72 GiB. Synthetic outputs were finite and invariant to
window batch size; the expected approximation delta was small for the bounded
random-weight test but is not a substitute for a real H3 checkpoint comparison.

A separate bounded-random 36-layer A40 simulation with five resident temporal
tokens at 480x848 measured warmed SDPA at 1.517 s for official independent
windows and 0.911 s for shared-core (1.67x), with approximately 5.08/5.09 GiB
allocated peaks. The per-block image-query count fell from 19,200 to 7,950 and
outputs were finite and invariant to compatible window batch grouping. This
tests scheduling and structural work reduction only; visual quality and SM75
throughput still require the real checkpoint on the target Turing card.

Dense SwiGLU routing was bitwise identical but slightly slower, so automatic
fusion is restricted to compatible W8A8 weights. Sage and SDPA did not beat the
official attention choice at the H3 VAE's relatively short per-tile sequence on
this machine.

No complete H3 VAE checkpoint was available locally, so an end-to-end real-video
comparison remains required on the target Turing card. In particular, W8A8-QK
attention requires a compatible kernel/CUDA build.
