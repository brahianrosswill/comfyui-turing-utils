# H3 Veda preparation

Veda uses its own predictor bundle, not a LoRA. Predictor precision does not
change the INT8-QK / floating-PV attention contract. On SM75, BF16 predictor
operations round operands/results to BF16 but compute using FP32. FP32 remains
a separate diagnostic mode.

## Preparation and lifetime

- Fused Q/K/V gather + valid-only TripPool, FP32 sums and extrema. Physical
  HND storage eliminates a full V transpose/copy before attention.
- Batched INT8 predictor GEMM; BF16 projection rounding precedes residual
  addition, matching the individual-head implementation.
- Fused score masking/diagonal preparation, quota/validity postprocessing,
  selected-tile route packing and union with global keys. Top-K sorting itself
  still uses PyTorch, preserving tie behavior. Quotas and forced diagonals are
  unchanged, including FP64 fractional quotas.
- Whole 64-row query CTAs containing only padding are zeroed and skipped.
  A dedicated scatter kernel restores valid output rows only.
- Floating tile tensors are released before the attention output allocation.
- Optional tile-aligned compact preparation retains INT8 Q/K, floating V and
  predictor features instead of full floating tile Q/K. Chunk boundaries are
  multiples of 128 and preserve the 16/64-token quantization groups.
- Compact preparation is selected only when the non-evicting budget permits
  fewer head groups than whole preparation, or a full head cannot fit.
- Route scoring can grow from 128 to at most 256 rows after selecting the head
  group, only when its extra scratch fits the same non-evicting budget. It never
  shrinks the selected head group to accommodate a larger score block. The
  benchmark's `--score-chunk-rows 128` retains the prior scheduling baseline;
  `0` selects automatically. This changes launch granularity, not keep ratios.
- Predictor weights use pinned host staging on a forward-local upload stream.
  Copies overlap tile preparation; event waits and allocator stream recording
  protect consumption/lifetime. For multi-shape calls only, a budget-slack
  check permits staging this call's heads once (<=8 MiB) rather than repeatedly
  staging each shape. This allocation dies with the call, not the forward.
- The forward-local GPU layout LRU has a total byte limit of 4 MiB.
  There is no cross-forward GPU predictor/layout cache or CUDA Graph capture.

This does **not** re-enable SOL's compact QKV projection. That path quantizes
before Veda's tile ordering and uses a different rotation/stabilization
contract. The default caller still supplies floating post-RoPE Q/K/V. A separate
experimental Veda-specific projected path is now available below.

## Experimental execution modes

Execution modes and projection chunk sizes are internal implementation/benchmark
controls, not node inputs. Legacy node arguments for these controls are ignored
so old experimental workflows cannot accidentally force an expensive path.

The node enables conservative automatic scheduling: use full projection when it
fits; within W8 head-sharded execution, switch to projected QKV only when the
estimated full group exceeds the live non-evicting budget and projected scratch
fits. Select the largest safe chunk from 64/32/16/8/4/1 tiles, reserving 64 MiB
and accounting for input gathers, GEMM scratch and INT8 weight-row packing.
Existing score/tile/head grouping remains budget-aware. No online benchmark or
CUDA Graph capture is performed. Heterogeneous batching is not selected merely
because memory is free: its modest synthetic gain and higher residency are not
yet sufficient evidence for a general automatic crossover policy.

- `auto`: the conservative memory-pressure policy above; `serial`: benchmark
  baseline without automatic projected-QKV selection.
- `heterogeneous`: retain at most three compact groups, then execute their
  attention in one CUDA launch using per-head pointer/length descriptors. Q/K/V
  and routes are **not** repacked or padded to the largest shape. Each head keeps
  its own tile counts, dense-global query policy, scales and route word stride.
  Live budget checks flush groups early or use serial attention when necessary.
- `compact_qkv`: for merged plain ConvRot W8A8 QKV weights, generate only each
  tile chunk's real rows. Original source indices select input activations and
  RoPE frequencies. Pool floating post-RoPE Q/K before Sage quantization; retain
  only INT8 Q/K, floating V, scales and predictor features across chunks. This
  is a streamed pipeline, **not** a single fused GEMM/RoPE/pooling kernel.
- `compact_qkv_heterogeneous`: combine the two experimental paths.

Compact projection honors standard weight casting/LoRA merging and releases
weights through the existing `finally` path. Non-contiguous heads require one
bounded INT8 weight-row pack per shape group, reused across chunks and released
before attention. No expanded FP16/BF16 QKV weight copy or persistent GPU cache
is introduced. The callback scratch estimate includes this pack, input gather,
quantization and GEMM temporaries. CPU row plans are pinned and reused only for
the current forward; small index uploads are asynchronous. Dense scheduled
layers/steps keep their original path; unsupported QKV weight formats retain
ordinary projection. Internal forced-projection benchmarks default to 16 tiles;
64 trades more scratch for fewer launches. The node chooses automatically.

The initial padded heterogeneous prototype remains a low-level comparison path,
not the node's selected implementation. It raised peak scratch from ~222 to
~683 MiB and was not a useful speed optimization. Ragged descriptors remove
that repacking, but retaining several groups still increases peak memory.

Real W8 GEMM + RMSNorm + original-position RoPE tests compare complete and
projected paths, including cached/uncached quantized inputs, scalar/per-output
weight scales, nonzero head-shard starts and interleaved shapes. These are random
weights, not official predictor/video quality validation. An A40 synthetic
20,044-row, 14-head, D128, hidden7168 run with cached input and 64-tile chunks:

| Layout | Full QKV + Veda | Projected QKV + Veda | Peak scratch full / projected |
| --- | ---: | ---: | ---: |
| One shape | 16.38 ms | 17.21 ms | 698 / 440 MiB |
| Three interleaved shapes | 22.30 ms | 27.94 ms | 428 / 227 MiB |

Thus compact projection currently offers a memory tradeoff, not an unconditional
speedup. Its remaining overhead is fragmented GEMM/gather/normalization dispatch
and repeated input work across shape groups. No production/2080 Ti claim follows
from this benchmark. The benchmark supports `--projected-qkv`,
`--cache-qkv-input`, `--projection-chunk-tiles`, and `--heterogeneous`.

The ragged specialization compiles to 172 registers/thread for FP16 and 168 for
BF16 on SM75, with no stack/local spills and unchanged 32 KiB dynamic shared
memory. Its resource bound remains two CTAs per SM. SM86 uses 168 registers.
After eliminating duplicate intra-group budget observations (the next shape
already rechecks live memory), the three-shape attention benchmark measured
~9.81 ms W8A8 versus ~10.26 ms serial, with ~405 versus ~222 MiB peak scratch.
The explicit batching mode therefore still trades memory for a modest gain;
it is not the default for memory-constrained Turing deployments.

## Validation and performance scope

`VEDA_TEST_CUDA=1 ops/test-dev.sh -q -k veda` covers partial tiles, strided
inputs, precision semantics, head shards, route packing and exact equality
of whole/chunked quantized tensors.

`kernel/scripts/benchmark_veda.py` uses random predictor weights, not the
published model. Run in the owning environment with instance-specific cache,
temporary-directory and CUDA settings. `--available-mib` simulates the budget
decision only; it is not an actual VRAM-pressure measurement.

A40, 20,044 tokens, 14 heads, D128, grid 37x30x18, tile 4x4x8:

| Configuration | Median latency | Additional peak allocation |
| --- | ---: | ---: |
| Original W8A8 Veda preparation | ~13.8 ms | Not recorded |
| Fused whole preparation | ~8.5 ms | ~598 MiB |
| Forced 32-tile compact preparation | ~9.5 ms | ~393 MiB |
| HND + fused selection + padding-CTA skip (current) | ~6.31 ms | ~492 MiB |

The first-round preparation under a synthetic 560 MiB available-budget report
with automatic compact
preparation measured ~9.2 ms versus ~11.1 ms for forced whole preparation.
Under 450 MiB, both choices still require two groups, so auto keeps the
whole path (~11.3 ms) rather than the slower compact path (~13.9 ms).
These historical budget thresholds are not current tuning constants: the HND
version accounts for the removed V copy and re-evaluates head grouping.

The current single-shape synthetic case compared with SOL at ~7.32 ms. A
three-shape interleaved plan initially measured ~13 ms and dropped to ~10.8 ms
after bounded LRU caching and call-local predictor staging. Multi-shape calls
still execute shape groups serially. The single-shape result must not be
generalized to arbitrary trained plans. These are not quality-matched SOL
comparisons or end-to-end H3 benchmarks.

Budget-aware score chunks measured 6.22–6.25 ms versus 6.30–6.31 ms for the
single-shape W8A8 case, and 10.37–10.38 ms versus 10.66–11.08 ms for the
three-shape case (two alternating runs, 30 repetitions each). Additional peak
allocation stayed at ~492 and ~222 MiB respectively. This is a small scheduling
gain, not a solution to heterogeneous-shape serialization, and no CUDA kernel
resources or CTA occupancy changed in this round.

Further staging experiments were rejected: packing four predictor transfers
into one did not reduce transfer latency (~0.185 versus ~0.174 ms for 14 W8A8
heads), and CPU head reordering added ~0.29 ms in that microbenchmark. Neither
experimental path is enabled. Heterogeneous tile shapes still execute serially;
the remaining large optimization requires a heterogeneous-layout kernel or
projection-level compact preparation, not merely fewer Python copy calls.

SM75 compiled resources: gather/TripPool 50 registers/thread, scatter 26
(FP64 diagnostic variant 28), route packing 16, predictor epilogue 22, score
preparation 22, quota postprocessing 20; all have zero shared memory, stack
and local memory.
Gather, scatter and route kernels use 128 threads; the other preparation
kernels use 256. D128 attention remains at 182 registers/thread for FP16 and
178 for BF16, with the same 32 KiB dynamic shared memory and two-CTA SM75
resource bound. No 2080 Ti hardware measurements have been performed.
