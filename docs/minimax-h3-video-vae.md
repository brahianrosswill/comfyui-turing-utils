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
  the official geometry. Encoder geometry is never changed.
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

Both nodes publish a ComfyUI progress bar whose total is the number of spatial
tiles multiplied by the number of temporal clips/chunks. Every kernel batch
advances it by the actual number of tiles in that batch, including a shorter
final batch. CUDA events feed the bar from a small background consumer only
after each batch really finishes; this keeps its ETA meaningful without
synchronizing the main submission stream or disabling transfer/compute overlap.

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

Dense SwiGLU routing was bitwise identical but slightly slower, so automatic
fusion is restricted to compatible W8A8 weights. Sage and SDPA did not beat the
official attention choice at the H3 VAE's relatively short per-tile sequence on
this machine.

No complete H3 VAE checkpoint was available locally, so an end-to-end real-video
comparison remains required on the target Turing card. In particular, W8A8-QK
attention requires a compatible kernel/CUDA build.
