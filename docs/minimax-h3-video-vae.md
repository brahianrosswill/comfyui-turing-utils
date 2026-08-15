# MiniMax H3 Video VAE

The two experimental MiniMax H3 VAE nodes provide a focused Turing execution
path for the official H3 video VAE. They accept the normal ComfyUI `VAE`,
`LATENT`, and `IMAGE` types and do not require the Turing Utils model loader.

## Decode

`MiniMax H3 Video VAE Decode` uses one fixed spatial policy:

- 256px windows with 64px overlap, matching the geometry expected by H3;
- full cosine partition-of-unity queries throughout every overlap by default,
  with a deterministic FP32 epilogue instead of atomic overlap reductions;
- one globally reconciled image-token state for decoder transformer blocks;
- independent 256px windows only for the final pixel projection;
- normalized multiband stitching whose frequency split is evaluated on two
  complete canvases rather than independently at every tile edge.

Two optional controls provide a shared-state speed/quality experiment:

- `overlap_query_threshold=0` retains every cosine membership and is the
  quality-preserving default. A positive value drops only very small window
  memberships, always retains at least one owner, and renormalizes survivors.
- `final_full_overlap_blocks=36` keeps the complete overlap in every decoder
  Transformer block. Lower values allow the threshold only in earlier blocks
  and restore complete overlap for the requested final block count. Hidden
  states remain globally reconciled after every block.

For the common 864x480 decode geometry, threshold `0.03` retains about 80% of
overlap queries in early blocks and forms only three query-size groups. Treat
it as an experimental starting point, not a universal recommendation.

The multiband implementation first constructs a low-frequency-priority canvas
and a high-frequency-priority canvas, then separates their bands globally. It
therefore exactly reconstructs identical overlapping content (within FP32
rounding) and avoids the paired seam lines caused by tile-local low-pass
boundary conditions.

The attention selector applies to every shared decoder Transformer block:

- `sdpa`: PyTorch SDPA, with FP16 computation for Turing BF16 inputs to avoid
  its slow math fallback;
- `sage`: bundled Turing Sage attention;
- `w8a8`: quantized QK attention when the installed kernel supports it.

## Encode

`MiniMax H3 Video VAE Encode` retains the official 256px/64px tiled encoder
geometry and ComfyUI-compatible FP32 latent output. It chooses up to sixteen
simultaneous spatial tiles from the current memory budget and retains one
encoder weight cycle across all spatial tiles and temporal clips. Generic
ComfyUI IMAGE input keeps its FP32 host representation; an existing FP16 pixel
store is transferred and normalized as FP16 on the GPU without first widening
the complete clip to FP32.

## Latent Pixel Upscale

`MiniMax H3 Latent Pixel Upscale` performs the quality-preserving two-stage
bridge in one experimental node:

1. decode the H3 video latent with the optimized decoder;
2. resize each finalized temporal pixel chunk as complete RGB frames, never as
   independent spatial tiles;
3. stream the resized chunks directly into one FP16 target store;
4. immediately encode that store back into an H3 video latent.

There is no complete source-resolution pixel store: decode finalization,
spatial resize, and FP16 target staging form one bounded pipeline. The target
store stays on the GPU when the current memory budget allows and falls back to
double-buffered CPU FP16 staging for very large outputs. The target allocation
is included in decoder auto-batch planning. Normal interpolation is batched on
the GPU. The optional `rtx_vsr` method lazily uses NVIDIA's `nvidia-vfx`
package and has no effect on installations that do not select it. RTX VSR
input/output remains GPU-resident and each SDK-owned DLPack result is cloned
before the next frame.

Target width and height must be multiples of 32 and must not be smaller than
the decoded source. A nested H3 AV latent keeps its audio samples and audio
mask unchanged; only the video latent and its spatial noise mask are replaced.
This node does not add noise or choose a second-stage sigma. Feed its output to
a separate sampler so denoise strength and scheduling remain explicit.

## Automatic execution

Both nodes choose the number of simultaneously evaluated windows internally.
The choice is bounded by current free/reclaimable device memory and conservative
activation estimates (up to sixteen windows for both decode and encode). This
control is intentionally not exposed in the node UI.

When all decode windows fit in one attention batch, kernel package 0.26.0 uses
a single deterministic SM75 overlap epilogue. It accumulates window outputs in
FP32 in fixed window order and writes the model dtype once. Smaller batches and
pruned schedules fall back to the equivalent ordered FP32 Python path.

Both paths retain prefetched weights across spatial windows and temporal chunks
when memory allows. Decode uses asynchronous FP32 pixel double buffering;
encode uses asynchronous input buffering when pinned host memory is available.
The nodes publish completed-tile progress to both ComfyUI and the terminal,
then release the VAE weights through ComfyUI's normal dynamic-memory manager.

These nodes are experimental. The fixed tiling policy is deliberate: discarded
arbitrary-size and alternate stitching strategies were removed after producing
visible grids, blur, or unnecessary duplicated token work.
