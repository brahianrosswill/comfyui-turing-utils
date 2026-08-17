# MiniMax H3 Video VAE

The MiniMax H3 VAE nodes provide a focused Turing execution path for the
official H3 video VAE. They accept the normal ComfyUI `VAE`, `LATENT`, and
`IMAGE` types and do not require the Turing Utils model loader. The decoder is
the production path, and the encoder follows the same public ComfyUI VAE
storage contract.

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
geometry. It chooses up to sixteen simultaneous spatial tiles from the current
memory budget. Encoder stages use ComfyUI's dynamic-VBAR prefetch queue, which
returns each stage before moving to the next instead of pinning one complete
encoder weight cycle. Pixel preparation, moments, and latent normalization keep
their numerically sensitive FP32 steps; the published latent is converted only
at the boundary to `vae.vae_output_dtype()`, matching the official VAE wrapper.

## Automatic execution

Both nodes choose the number of simultaneously evaluated windows internally.
The choice is bounded by current free/reclaimable device memory and conservative
activation estimates (up to sixteen windows for both decode and encode).
This control is intentionally not exposed in the node UI.

With kernel ABI 0.27 or newer, the decoder streams each attention subbatch into
the global FP32 canvas through a deterministic SM75 accumulator. Compact
inverse maps live only for the current decode, and kernel launches retain the
same group/window order as the validated Python reduction. Older kernels and
unsupported dtypes automatically use that ordered Python fallback.

Both paths use ComfyUI's official short-lived prefetch queues. Decoder weights
are leased one Transformer block at a time and encoder weights one execution
stage at a time; unpinned VBAR pages may remain resident when memory allows,
but the plugin does not retain or evict them itself. After FP32 pixel
finalization, decode uses asynchronous output-dtype double buffering; encode
uses asynchronous input buffering when pinned host memory is available. The
nodes publish completed-tile progress to both ComfyUI and the terminal, while
model residency and cleanup remain owned by ComfyUI's dynamic-memory manager.

The decoder's fixed tiling policy is deliberate: discarded arbitrary-size and
alternate stitching strategies were removed after producing visible grids,
blur, or unnecessary duplicated token work. Positive
`overlap_query_threshold` values remain an optional experimental speed/quality
control; zero is the stable decoder default.
