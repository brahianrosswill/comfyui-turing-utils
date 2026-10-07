# H3 Veda sparse attention

One automatic inference path, with no selectable execution variants or legacy
workflow/ABI compatibility. H3 only, batch one, NVIDIA SM75 or newer. Predictor
bundles are not LoRAs; normal model weight patches remain independent.

## Node

Basic inputs: model, predictor file, predictor precision (W8A8 by default).
Advanced inputs: target/reference keep ratios, strict/nearest plan policy,
dense layer/step schedule and debug. A ratio of 0 uses the bundle's trained
value. Both ratios equal to 1 use the existing dense backend with no predictor
workspace. Scheduling never changes quality ratios.

SOL and Veda replace one another rather than stacking. ComfyUI's detected
block-replacing sparse patch is rejected. Native, transposed and nearest plans
are distinguished; strict mode requires a native trained geometry. Non-native
plans and unmappable references retained as dense are logged. Debug reports
forward sparse-call counts (including head shards).

## Execution and memory

- Fused gather/valid-only TripPool, INT8 batched predictor GEMM and route
  preprocessing/postprocessing. PyTorch Top-K preserves selection semantics.
- Physical HND V storage, INT8 QK / floating PV attention, padded-CTA skipping
  and fused scatter.
- Head groups and score chunks adapt to the non-evicting memory budget.
  Projected Q/K features convert to FP32 once per group, not per score chunk.
- Whole projection is preferred when it fits. Tile-streamed W8A8 QKV is only
  an internal memory-pressure fallback, preserving original-position RoPE,
  pooling before quantization and normal LoRA weight merging.
- CPU-owned predictor; only bounded head groups are staged on GPU. Transfers
  overlap preparation and record consumer streams. No persistent GPU predictor
  cache, CUDA Graph capture or online autotuning.
- Forward-local layout cache is bounded to 4 MiB. Predictor staging, projection
  scratch, input gathers and weight row packs participate in the budget.
- BF16 predictor mode rounds operands/results to BF16, using FP32 arithmetic
  on Turing. FP16 and FP32 remain supported. Predictor precision does not
  change the attention kernel's quantization contract.

## Validation

Use the owning development environment and instance-scoped caches:
`VEDA_TEST_CUDA=1 ops/test-dev.sh -q -k veda`.

Tests cover masks/padding, route quotas, precision semantics, head shards,
whole/chunked quantization and real W8 GEMM + RMSNorm + original-position RoPE.
Internal benchmark chunk controls exist to regression-test the memory fallback,
not as alternative production configurations.

`veda/reference.py` is an independent official-arithmetic oracle. Decoded
source weights round to BF16; pooled features, projections, residuals and
logits stay FP32. It is deliberately different from BF16 emulation and FP32
diagnostic mode. Disable TF32 for CUDA oracle comparisons. Route recall is
measured independently of logit error.

Existing A40 synthetic measurements found tile-streamed projection reduced
scratch but was slower when the whole path fit. This is why it is selected
only under memory pressure. Random predictor tests are not real-checkpoint
quality validation or quality-matched comparisons with SOL. Actual 2080 Ti,
Windows and end-to-end video quality/performance validation remain outstanding.
