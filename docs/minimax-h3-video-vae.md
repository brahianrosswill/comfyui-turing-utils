# MiniMax H3 Video VAE

The two experimental MiniMax H3 VAE nodes provide a focused Turing execution
path for the official H3 video VAE. They accept the normal ComfyUI `VAE`,
`LATENT`, and `IMAGE` types and do not require the Turing Utils model loader.

## Decode

`MiniMax H3 Video VAE Decode` uses one fixed spatial policy:

- 256px windows with 64px overlap, matching the geometry expected by H3;
- one latent-token feather at shared attention boundaries, with deterministic
  accumulation instead of atomic overlap reductions;
- shared image-token state for the early decoder transformer blocks;
- independent 256px windows for the configurable transformer tail and final
  pixel projection;
- normalized multiband stitching whose frequency split is evaluated on two
  complete canvases rather than independently at every tile edge.

`independent_tail_blocks` controls how many final decoder transformer blocks
run independently per window. The default is 2. A larger value gives late
blocks more tile-local freedom but repeats more work. Zero keeps every
transformer block on the shared-token path; final projection and multiband
stitching remain independent so boundaries are still reconstructed correctly.

The tail is a deliberate approximation control, not a monotonic quality
slider. At the shared/tiled fork, overlapping windows initially contain the
same hidden tokens but use different local RoPE coordinates and context. Every
additional independent block lets those copies diverge further. Quantized
attention can amplify that divergence, so `2` is the tested default and `8`
may expose grids even though it is closer to doing more of the decoder in the
official per-tile form. Large values are mainly useful for diagnosis.

The multiband implementation first constructs a low-frequency-priority canvas
and a high-frequency-priority canvas, then separates their bands globally. It
therefore exactly reconstructs identical overlapping content (within FP32
rounding) and avoids the paired seam lines caused by tile-local low-pass
boundary conditions.

The attention selector applies to both the shared prefix and independent tail:

- `sdpa`: PyTorch SDPA, with FP16 computation for Turing BF16 inputs to avoid
  its slow math fallback;
- `sage`: bundled Turing Sage attention;
- `w8a8`: quantized QK attention when the installed kernel supports it.

## Encode

`MiniMax H3 Video VAE Encode` retains the official 256px/64px tiled encoder
geometry and ComfyUI-compatible FP32 latent output.

## Latent Pixel Upscale

`MiniMax H3 Latent Pixel Upscale` performs the quality-preserving two-stage
bridge in one experimental node:

1. decode the H3 video latent with the optimized decoder;
2. resize complete RGB frames, never independent pixel tiles;
3. immediately encode the resized frames back into an H3 video latent.

The intermediate pixel store is FP16 because H3's encoder itself consumes
FP16 by default. It stays on the GPU when the current memory budget allows and
falls back to CPU FP16 staging for very large targets. Normal interpolation is
batched on the GPU. The optional `rtx_vsr` method lazily uses NVIDIA's
`nvidia-vfx` package and has no effect on installations that do not select it.
RTX VSR input/output remains GPU-resident and each SDK-owned DLPack result is
cloned before the next frame.

Target width and height must be multiples of 32 and must not be smaller than
the decoded source. A nested H3 AV latent keeps its audio samples and audio
mask unchanged; only the video latent and its spatial noise mask are replaced.
This node does not add noise or choose a second-stage sigma. Feed its output to
a separate sampler so denoise strength and scheduling remain explicit.

## Automatic execution

Both nodes choose the number of simultaneously evaluated windows internally.
The choice is bounded by current free/reclaimable device memory and conservative
activation estimates (up to four windows for decode and two for encode). This
control is intentionally not exposed in the node UI.

Both paths retain prefetched weights across spatial windows and temporal chunks
when memory allows. Decode uses asynchronous FP32 pixel double buffering;
encode uses asynchronous input buffering when pinned host memory is available.
The nodes publish completed-tile progress to both ComfyUI and the terminal,
then release the VAE weights through ComfyUI's normal dynamic-memory manager.

These nodes are experimental. The fixed tiling policy is deliberate: discarded
arbitrary-size and alternate stitching strategies were removed after producing
visible grids, blur, or unnecessary duplicated token work.
