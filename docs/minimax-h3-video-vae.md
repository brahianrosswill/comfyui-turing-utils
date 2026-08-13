# MiniMax H3 Video VAE

The two experimental MiniMax H3 VAE nodes provide a focused Turing execution
path for the official H3 video VAE. They accept the normal ComfyUI `VAE`,
`LATENT`, and `IMAGE` types and do not require the Turing Utils model loader.

## Decode

`MiniMax H3 Video VAE Decode` uses one fixed spatial policy:

- 256px windows with 64px overlap, matching the geometry expected by H3;
- one latent-token feather at shared attention boundaries;
- shared image-token state for the early decoder transformer blocks;
- independent 256px windows for the configurable transformer tail and final
  pixel projection;
- normalized multiband stitching for the projected pixels.

`independent_tail_blocks` controls how many final decoder transformer blocks
run independently per window. The default is 2. A larger value gives late
blocks more tile-local freedom but repeats more work. Zero keeps every
transformer block on the shared-token path; final projection and multiband
stitching remain independent so boundaries are still reconstructed correctly.

The attention selector applies to both the shared prefix and independent tail:

- `sdpa`: PyTorch SDPA, with FP16 computation for Turing BF16 inputs to avoid
  its slow math fallback;
- `sage`: bundled Turing Sage attention;
- `w8a8`: quantized QK attention when the installed kernel supports it.

## Encode

`MiniMax H3 Video VAE Encode` retains the official 256px/64px tiled encoder
geometry and ComfyUI-compatible FP32 latent output.

## Automatic execution

Both nodes choose the number of simultaneously evaluated windows internally.
The choice is bounded by current free/reclaimable device memory and conservative
activation estimates (up to four windows for decode and two for encode). This
control is intentionally not exposed in the node UI.

Both paths retain prefetched weights across spatial windows and temporal chunks
when memory allows. Decode uses asynchronous FP32 pixel double buffering;
encode uses asynchronous input buffering when pinned host memory is available.
The nodes publish progress to both ComfyUI and the terminal, then release the
VAE weights through ComfyUI's normal dynamic-memory manager.

These nodes are experimental. The fixed tiling policy is deliberate: discarded
arbitrary-size and alternate stitching strategies were removed after producing
visible grids, blur, or unnecessary duplicated token work.
