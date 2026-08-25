#pragma once

// Included inside sol_sparse_cuda_sm75.cu's anonymous namespace. The owning
// translation unit supplies kWarps, kRouteWordBits and the CUDA headers.

template <int SelectedCapacity>
__device__ __forceinline__ int compact_route_words(
    const uint32_t *__restrict__ route_words,
    int *__restrict__ selected_count,
    uint16_t *__restrict__ selected_blocks,
    uint32_t *__restrict__ scratch,
    int active_key_blocks)
{
  // Each thread owns one ascending route word. Warp scans plus four shared
  // warp totals assign stable, non-overlapping output ranges without making a
  // single lane enumerate every selected block. Words and bits retain their
  // original ascending order, which keeps online-softmax rounding unchanged.
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int route_word_count =
      (active_key_blocks + kRouteWordBits - 1) / kRouteWordBits;
  uint32_t word = linear_thread < route_word_count
      ? route_words[linear_thread]
      : 0;
  const int final_word_bits = active_key_blocks % kRouteWordBits;
  if (linear_thread == route_word_count - 1 && final_word_bits != 0)
    word &= (1U << final_word_bits) - 1U;
  unsigned int word_count = __popc(word);
  unsigned int inclusive = word_count;
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1)
  {
    const unsigned int other = __shfl_up_sync(
        0xffffffffu, inclusive, offset);
    if (lane >= offset)
      inclusive += other;
  }
  if (lane == WARP_SIZE - 1)
    scratch[warp] = inclusive;
  __syncthreads();
  if (linear_thread == 0)
  {
    unsigned int running = 0;
#pragma unroll
    for (int warp_index = 0; warp_index < kWarps; ++warp_index)
    {
      scratch[kWarps + warp_index] = running;
      running += scratch[warp_index];
    }
    *selected_count = running <= SelectedCapacity
        ? static_cast<int>(running)
        : -static_cast<int>(running);
  }
  __syncthreads();
  const unsigned int output_start =
      scratch[kWarps + warp] + inclusive - word_count;
  const int selected = *selected_count;
  unsigned int output_index = output_start;
  while (word != 0)
  {
    const int bit = __ffs(static_cast<int>(word)) - 1;
    const int key_block = linear_thread * kRouteWordBits + bit;
    if (key_block < active_key_blocks && output_index < SelectedCapacity)
      selected_blocks[output_index++] = static_cast<uint16_t>(key_block);
    word &= word - 1;
  }
  __syncthreads();
  return selected;
}

__device__ __forceinline__ int next_shared_route_block(
    const uint32_t *__restrict__ route_words,
    int start,
    int active_key_blocks)
{
  for (int key_block = start; key_block < active_key_blocks; ++key_block)
  {
    if ((route_words[key_block / kRouteWordBits] >>
         (key_block % kRouteWordBits)) & 1U)
      return key_block;
  }
  return active_key_blocks;
}

template <int SelectedCapacity>
__device__ __forceinline__ int next_compact_route_block(
    const uint32_t *__restrict__ route_words,
    const uint16_t *__restrict__ selected_blocks,
    int selected_count,
    int selected_position,
    int start,
    int active_key_blocks)
{
  if (selected_count >= 0)
  {
    if (selected_position >= selected_count)
      return active_key_blocks;
    if (selected_position < SelectedCapacity)
    {
      const int candidate = static_cast<int>(
          selected_blocks[selected_position]);
      // A corrupted or stale compact entry must never become an out-of-range
      // global K/V prefetch. The bitmap remains authoritative and provides a
      // deterministic ascending fallback without changing softmax order.
      if (candidate >= start && candidate < active_key_blocks)
        return candidate;
    }
  }
  return next_shared_route_block(route_words, start, active_key_blocks);
}

struct RouteWords
{
  uint32_t word0;
  uint32_t word1;
  uint32_t word2;
  uint32_t word3;
};

__device__ __forceinline__ uint32_t route_word(
    const RouteWords &route_words,
    int lane_slot)
{
  switch (lane_slot)
  {
    case 0: return route_words.word0;
    case 1: return route_words.word1;
    case 2: return route_words.word2;
    default: return route_words.word3;
  }
}

__device__ __forceinline__ bool register_route_selected(
    const RouteWords &route_words,
    int key_block)
{
  const int word_index = key_block / kRouteWordBits;
  const uint32_t word = __shfl_sync(
      0xffffffff,
      route_word(route_words, word_index / WARP_SIZE),
      word_index % WARP_SIZE);
  return (word >> (key_block % kRouteWordBits)) & 1U;
}

__device__ __forceinline__ int next_register_route_block(
    const RouteWords &route_words,
    int start,
    int active_key_blocks)
{
  for (int key_block = start; key_block < active_key_blocks; ++key_block)
  {
    if (register_route_selected(route_words, key_block))
      return key_block;
  }
  return active_key_blocks;
}
