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

- `tile_preset` is `auto` by default. Numeric choices (`256`, `288`, `320`,
  `384`, and `480`) are the internal tile edge in pixels. `auto` searches
  16-pixel-aligned edges from 256 through 480, rejects candidates whose
  predicted peak exceeds ComfyUI's current available/reclaimable memory, and
  minimizes repeated tile area among the remaining candidates. Spatial overlap
  remains fixed at the H3 default of 64 pixels. Numeric presets are strict and
  intentionally ignore the adaptive budget.
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

## Current validation status

The custom decoder and encoder equations are tested against ComfyUI's reference
implementations, including temporal chunk reconstruction and tiled assembly.
The retained-weight lifecycle is tested for both success and allocation-failure
cleanup.

On an A40 with the complete 2.6B-parameter H3 VAE architecture and identical
zeroed FP16 weights, the custom 256 path matched the official runtime and output
exactly. Adaptive tiles reduced one-chunk decoder time from 2.22 s to 1.12 s at
480x848 and from 4.19 s to 2.94 s at 720x1280. A two-clip encoder improved from
5.21 s to 2.42 s and from 9.74 s to 6.73 s respectively. The speed comes from
eliminating repeated tile work rather than a faster fixed-256 implementation.
The zero-weight auto outputs were also identical, but that does not establish
real-checkpoint quality: changing tile boundaries changes the context seen by
the decoder transformer and must be visually validated with real weights.

Larger tiles exchange memory for speed. The measured encoder peaks were 7058
MiB at 480x848/480px and 5094 MiB at 720x1280/400px. The tile-aware estimates
were 7412 MiB and 5360 MiB, while the 256 estimates retain additional safety
margin. The estimator includes H3 CNN/transformer workspaces, full decoded
chunk residency, finalized pixel copies, and encoder double buffers rather than
reusing ComfyUI's 256-oriented bounded-tile estimate unchanged.

With a 3 GiB activation budget, auto selected 272px at 480x848 (2355 MiB
measured, 2619 MiB estimated, 3.18 s) and 288px at 720x1280 (2777 MiB measured,
2999 MiB estimated, 7.94 s). These remained about 1.64x and 1.23x faster than
the corresponding official 256 runs while respecting the constrained budget.

Dense SwiGLU routing was bitwise identical but slightly slower, so automatic
fusion is restricted to compatible W8A8 weights. Sage and SDPA did not beat the
official attention choice at the H3 VAE's relatively short per-tile sequence on
this machine.

No complete H3 VAE checkpoint was available locally, so an end-to-end real-video
comparison remains required on the target Turing card. In particular, W8A8-QK
attention requires a compatible kernel/CUDA build.
