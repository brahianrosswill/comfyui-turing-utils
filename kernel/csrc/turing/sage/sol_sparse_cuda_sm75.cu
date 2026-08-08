/*
 * Experimental Sol-style sparse attention for SM75.
 *
 * One 64-token centroid per block feeds an input-adaptive mean + tau * std
 * threshold. The threshold comparison is fused into the FP16 Tensor Core
 * centroid tile, so no full proxy map is materialized. Selected blocks are
 * evaluated with the production per-warp/per-block INT8 Sage QK path. Skipped
 * blocks use one 64-token or two 32-token centroids reconstructed from those
 * same INT8 Q/K tensors, then retain their contributions in the shared FP32
 * online softmax instead of being dropped. Original FP16/BF16 summaries remain
 * isolated to routing and original V means remain isolated to approximation.
 */

#include "../utils.cuh"
#include "../math.cuh"
#include "attn_utils.cuh"
#include "torch_compat.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <type_traits>

namespace {

constexpr int kBlockTokens = 64;
constexpr int kHeadDim = 128;
constexpr int kWarps = 4;
constexpr int kRouteTile = 16;
constexpr int kSummaryTileTokens = 32;
constexpr int kHalfPacks = kHeadDim / 8;
constexpr int kInt8Packs = kHeadDim / 16;
constexpr int kTilePacks = kBlockTokens * kHalfPacks;
constexpr int kInt8TilePacks = kBlockTokens * kInt8Packs;
constexpr int kTileBytes = kBlockTokens * kHeadDim * sizeof(half);
constexpr int kInt8TileBytes = kBlockTokens * kHeadDim * sizeof(int8_t);
constexpr int kSummaryTileBytes = kSummaryTileTokens * kHeadDim * sizeof(half);
constexpr int kAttentionSharedBytes = 2 * kTileBytes;

__global__ void route_popcount_kernel(
    const int32_t *__restrict__ route,
    int64_t elements,
    unsigned long long *__restrict__ selected)
{
  __shared__ unsigned long long partial[256];
  unsigned long long count = 0;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x)
  {
    count += __popc(static_cast<uint32_t>(route[index]));
  }
  partial[threadIdx.x] = count;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
  {
    if (threadIdx.x < offset)
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicAdd(selected, partial[0]);
}

template <typename T>
__device__ __forceinline__ float scalar_to_float(T value);

template <>
__device__ __forceinline__ float scalar_to_float<half>(half value)
{
  return __half2float(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<nv_bfloat16>(nv_bfloat16 value)
{
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ b128_t pack_to_half(const T *source);

template <>
__device__ __forceinline__ b128_t pack_to_half<half>(const half *source)
{
  return *reinterpret_cast<const b128_t *>(source);
}

template <>
__device__ __forceinline__ b128_t pack_to_half<nv_bfloat16>(const nv_bfloat16 *source)
{
  return bf16_pack_to_half(source);
}

template <typename T>
__global__ void block_summary_kernel(
    const T *__restrict__ input,
    half *__restrict__ summary,
    int batch_size,
    int num_heads,
    int sequence_length,
    int padded_blocks,
    int64_t stride_batch,
    int64_t stride_head,
    int64_t stride_sequence)
{
  const int block_index = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int dimension = threadIdx.x;
  if (dimension >= kHeadDim)
    return;

  const int token_start = block_index * kBlockTokens;
  const int token_count = token_start < sequence_length
      ? min(kBlockTokens, sequence_length - token_start)
      : 0;
  float sum = 0.0f;
  const T *head_input = input + batch * stride_batch + head * stride_head;
  for (int token = 0; token < token_count; ++token)
  {
    sum += scalar_to_float(head_input[(token_start + token) * stride_sequence + dimension]);
  }
  const int64_t output_index =
      ((static_cast<int64_t>(batch) * num_heads + head) * padded_blocks + block_index) *
          kHeadDim +
      dimension;
  summary[output_index] = token_count ? __float2half_rn(sum / token_count) : __float2half_rn(0.0f);
}

template <typename T>
__global__ void kv_block_summary_kernel(
    const T *__restrict__ key,
    const int8_t *__restrict__ key_int8,
    const float *__restrict__ key_scale,
    const T *__restrict__ value,
    half *__restrict__ key_summary,
    half *__restrict__ key_score_summary,
    half *__restrict__ value_mean,
    int batch_size,
    int num_heads,
    int sequence_length,
    int padded_blocks,
    int residual_subblocks,
    int padded_residual_summaries,
    int64_t stride_batch_k,
    int64_t stride_head_k,
    int64_t stride_sequence_k,
    int64_t stride_batch_k_int8,
    int64_t stride_head_k_int8,
    int64_t stride_sequence_k_int8,
    int64_t stride_batch_v,
    int64_t stride_head_v,
    int64_t stride_sequence_v)
{
  const int block_index = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int dimension = threadIdx.x;
  if (dimension >= kHeadDim)
    return;

  const int token_start = block_index * kBlockTokens;
  const int token_count = token_start < sequence_length
      ? min(kBlockTokens, sequence_length - token_start)
      : 0;
  float key_sum = 0.0f;
  float value_sum[2] = {0.0f, 0.0f};
  int quantized_key_sum[2] = {0, 0};
  const T *head_key = key + batch * stride_batch_k + head * stride_head_k;
  const int8_t *head_key_int8 = key_int8 +
      batch * stride_batch_k_int8 + head * stride_head_k_int8;
  const T *head_value = value + batch * stride_batch_v + head * stride_head_v;
  for (int token = 0; token < token_count; ++token)
  {
    const float key_value = scalar_to_float(
        head_key[(token_start + token) * stride_sequence_k + dimension]);
    key_sum += key_value;
    const int residual_index = token / (kBlockTokens / residual_subblocks);
    quantized_key_sum[residual_index] += static_cast<int>(
        head_key_int8[(token_start + token) * stride_sequence_k_int8 + dimension]);
    value_sum[residual_index] += scalar_to_float(
        head_value[(token_start + token) * stride_sequence_v + dimension]);
  }
  const int64_t output_index =
      ((static_cast<int64_t>(batch) * num_heads + head) * padded_blocks + block_index) *
          kHeadDim +
      dimension;
  if (token_count)
  {
    const float reciprocal = 1.0f / token_count;
    const float key_mean = key_sum * reciprocal;
    const int num_key_blocks = (sequence_length + kBlockTokens - 1) / kBlockTokens;
    const float dequant_scale = key_scale[
        (static_cast<int64_t>(batch) * num_heads + head) * num_key_blocks +
        block_index];
    key_summary[output_index] = __float2half_rn(key_mean);
    const int residual_tokens = kBlockTokens / residual_subblocks;
#pragma unroll
    for (int residual_index = 0; residual_index < 2; ++residual_index)
    {
      if (residual_index >= residual_subblocks)
        break;
      const int residual_start = residual_index * residual_tokens;
      const int residual_count = max(0, min(residual_tokens, token_count - residual_start));
      const int64_t residual_output_index =
          ((static_cast<int64_t>(batch) * num_heads + head) *
               padded_residual_summaries +
           block_index * residual_subblocks + residual_index) *
              kHeadDim +
          dimension;
      if (residual_count > 0)
      {
        const float residual_reciprocal = 1.0f / residual_count;
        key_score_summary[residual_output_index] = __float2half_rn(
            static_cast<float>(quantized_key_sum[residual_index]) *
            dequant_scale * residual_reciprocal);
        value_mean[residual_output_index] = __float2half_rn(
            value_sum[residual_index] * residual_reciprocal);
      }
      else
      {
        key_score_summary[residual_output_index] = __float2half_rn(0.0f);
        value_mean[residual_output_index] = __float2half_rn(0.0f);
      }
    }
  }
  else
  {
    key_summary[output_index] = __float2half_rn(0.0f);
    if (block_index * residual_subblocks < padded_residual_summaries)
    {
#pragma unroll
      for (int residual_index = 0; residual_index < 2; ++residual_index)
      {
        if (residual_index >= residual_subblocks)
          break;
        const int64_t residual_output_index =
            ((static_cast<int64_t>(batch) * num_heads + head) *
                 padded_residual_summaries +
             block_index * residual_subblocks + residual_index) *
                kHeadDim +
            dimension;
        key_score_summary[residual_output_index] = __float2half_rn(0.0f);
        value_mean[residual_output_index] = __float2half_rn(0.0f);
      }
    }
  }
}

__device__ __forceinline__ bool forced_route_block(
    int query_block,
    int key_block,
    int prefix_blocks,
    int local_block_radius,
    int query_token_offset,
    int topology_start_tokens,
    int topology_tokens,
    int tokens_per_frame,
    int temporal_neighbor_frames);

__device__ __forceinline__ float block_reduce(
    float value,
    float *__restrict__ scratch,
    bool maximum);

__global__ void key_summary_stats_kernel(
    const half *__restrict__ key_summary,
    float *__restrict__ key_summary_mean,
    float *__restrict__ key_summary_variance,
    int num_key_blocks,
    int padded_key_blocks)
{
  const int packed_head = blockIdx.x;
  const int dimension = threadIdx.x;
  if (dimension >= kHeadDim)
    return;

  const half *head_summary =
      key_summary + static_cast<int64_t>(packed_head) * padded_key_blocks * kHeadDim;
  float sum = 0.0f;
  float square_sum = 0.0f;
  for (int key_block = 0; key_block < num_key_blocks; ++key_block)
  {
    const float value = __half2float(
        head_summary[static_cast<int64_t>(key_block) * kHeadDim + dimension]);
    sum += value;
    square_sum = fmaf(value, value, square_sum);
  }
  const float reciprocal = 1.0f / static_cast<float>(num_key_blocks);
  const float mean = sum * reciprocal;
  const int64_t output_index =
      static_cast<int64_t>(packed_head) * kHeadDim + dimension;
  key_summary_mean[output_index] = mean;
  key_summary_variance[output_index] =
      fmaxf(square_sum * reciprocal - mean * mean, 0.0f);
}

__global__ void query_threshold_kernel(
    const half *__restrict__ query_summary,
    const float *__restrict__ key_summary_mean,
    const float *__restrict__ key_summary_variance,
    float *__restrict__ thresholds,
    int num_query_heads,
    int num_kv_heads,
    int num_query_blocks,
    int padded_query_blocks,
    float threshold_sigma)
{
  __shared__ float reduction[kHeadDim];
  const int query_block = blockIdx.x;
  const int query_head = blockIdx.y;
  const int batch = blockIdx.z;
  const int kv_head = query_head / (num_query_heads / num_kv_heads);
  const int dimension = threadIdx.x;

  const half *query_block_summary = query_summary +
      ((static_cast<int64_t>(batch) * num_query_heads + query_head) *
           padded_query_blocks +
       query_block) *
          kHeadDim;
  const float *mean = key_summary_mean +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) * kHeadDim;
  const float *variance = key_summary_variance +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) * kHeadDim;
  const float q_value = __half2float(query_block_summary[dimension]);
  const float projected_mean = block_reduce(q_value * mean[dimension], reduction, false);
  const float projected_variance = block_reduce(
      q_value * q_value * variance[dimension], reduction, false);
  if (dimension == 0)
  {
    thresholds[
        (static_cast<int64_t>(batch) * num_query_heads + query_head) *
            num_query_blocks +
        query_block] = projected_mean +
        threshold_sigma * sqrtf(fmaxf(projected_variance, 0.0f) + 1.0e-6f);
  }
}

__global__ void route_threshold_fused_kernel(
    const half *__restrict__ query_summary,
    const half *__restrict__ key_summary,
    const float *__restrict__ thresholds,
    int32_t *__restrict__ route,
    int batch_size,
    int num_query_heads,
    int num_kv_heads,
    int num_query_blocks,
    int num_key_blocks,
    int padded_query_blocks,
    int padded_key_blocks,
    int route_words,
    int prefix_blocks,
    int local_block_radius,
    int query_token_offset,
    int topology_start_tokens,
    int topology_tokens,
    int tokens_per_frame,
    int temporal_neighbor_frames,
    float minimum_route_density,
    float maximum_route_density)
{
  using namespace nvcuda;
  __shared__ float scores[kRouteTile * kRouteTile];

  const int packed_head = blockIdx.z;
  const int batch = packed_head / num_query_heads;
  const int query_head = packed_head % num_query_heads;
  const int kv_head = query_head / (num_query_heads / num_kv_heads);
  const int query_start = blockIdx.y * kRouteTile;
  const int key_start = blockIdx.x * kRouteTile;

  wmma::fragment<wmma::matrix_a, kRouteTile, kRouteTile, 16, half, wmma::row_major> query_fragment;
  wmma::fragment<wmma::matrix_b, kRouteTile, kRouteTile, 16, half, wmma::col_major> key_fragment;
  wmma::fragment<wmma::accumulator, kRouteTile, kRouteTile, 16, float> score_fragment;
  wmma::fill_fragment(score_fragment, 0.0f);

  const half *query = query_summary +
      ((static_cast<int64_t>(batch) * num_query_heads + query_head) *
           padded_query_blocks +
       query_start) *
          kHeadDim;
  const half *key = key_summary +
      ((static_cast<int64_t>(batch) * num_kv_heads + kv_head) *
           padded_key_blocks +
       key_start) *
          kHeadDim;
#pragma unroll
  for (int inner = 0; inner < kHeadDim; inner += 16)
  {
    wmma::load_matrix_sync(query_fragment, query + inner, kHeadDim);
    wmma::load_matrix_sync(key_fragment, key + inner, kHeadDim);
    wmma::mma_sync(score_fragment, query_fragment, key_fragment, score_fragment);
  }
  wmma::store_matrix_sync(scores, score_fragment, kRouteTile, wmma::mem_row_major);
  __syncwarp();

  const int local_query = threadIdx.x;
  if (local_query < kRouteTile)
  {
    const int query_block = query_start + local_query;
    if (query_block < num_query_blocks)
    {
      const float threshold = thresholds[
          (static_cast<int64_t>(batch) * num_query_heads + query_head) *
              num_query_blocks +
          query_block];
      uint32_t selected_bits = 0;
      const bool budget_enabled =
          minimum_route_density > 0.0f || maximum_route_density < 1.0f;
      if (!budget_enabled)
      {
#pragma unroll
        for (int local_key = 0; local_key < kRouteTile; ++local_key)
        {
          const int key_block = key_start + local_key;
          if (key_block < num_key_blocks &&
              (forced_route_block(
                   query_block,
                   key_block,
                   prefix_blocks,
                   local_block_radius,
                   query_token_offset,
                   topology_start_tokens,
                   topology_tokens,
                   tokens_per_frame,
                   temporal_neighbor_frames) ||
               scores[local_query * kRouteTile + local_key] > threshold))
            selected_bits |= 1U << local_key;
        }
      }
      else
      {
        const int valid_keys = min(kRouteTile, num_key_blocks - key_start);
        int forced_count = 0;
        int candidate_count = 0;
#pragma unroll
        for (int local_key = 0; local_key < kRouteTile; ++local_key)
        {
          if (local_key >= valid_keys)
            break;
          const int key_block = key_start + local_key;
          const bool forced = forced_route_block(
              query_block,
              key_block,
              prefix_blocks,
              local_block_radius,
              query_token_offset,
              topology_start_tokens,
              topology_tokens,
              tokens_per_frame,
              temporal_neighbor_frames);
          forced_count += forced ? 1 : 0;
          candidate_count += !forced &&
                  scores[local_query * kRouteTile + local_key] > threshold
              ? 1
              : 0;
        }
        const int minimum_selected = static_cast<int>(
            ceilf(minimum_route_density * valid_keys - 1.0e-6f));
        const int maximum_selected = max(
            minimum_selected,
            static_cast<int>(
                floorf(maximum_route_density * valid_keys + 1.0e-6f)));
        int desired_total = forced_count + candidate_count;
        desired_total = max(minimum_selected, min(maximum_selected, desired_total));
        desired_total = max(forced_count, min(valid_keys, desired_total));
        const int desired_adaptive = desired_total - forced_count;

#pragma unroll
        for (int local_key = 0; local_key < kRouteTile; ++local_key)
        {
          if (local_key >= valid_keys)
            break;
          const int key_block = key_start + local_key;
          const bool forced = forced_route_block(
              query_block,
              key_block,
              prefix_blocks,
              local_block_radius,
              query_token_offset,
              topology_start_tokens,
              topology_tokens,
              tokens_per_frame,
              temporal_neighbor_frames);
          if (forced)
          {
            selected_bits |= 1U << local_key;
            continue;
          }
          const float score = scores[local_query * kRouteTile + local_key];
          int rank = 0;
#pragma unroll
          for (int other = 0; other < kRouteTile; ++other)
          {
            if (other >= valid_keys || other == local_key)
              continue;
            const int other_key_block = key_start + other;
            if (forced_route_block(
                    query_block,
                    other_key_block,
                    prefix_blocks,
                    local_block_radius,
                    query_token_offset,
                    topology_start_tokens,
                    topology_tokens,
                    tokens_per_frame,
                    temporal_neighbor_frames))
              continue;
            const float other_score =
                scores[local_query * kRouteTile + other];
            if (other_score > score ||
                (other_score == score && other < local_key))
              ++rank;
          }
          if (rank < desired_adaptive)
            selected_bits |= 1U << local_key;
        }
      }
      route[
          ((static_cast<int64_t>(batch) * num_query_heads + query_head) *
               num_query_blocks +
           query_block) *
              route_words +
          blockIdx.x] = static_cast<int32_t>(selected_bits);
    }
  }
}

__device__ __forceinline__ bool forced_route_block(
    int query_block,
    int key_block,
    int prefix_blocks,
    int local_block_radius,
    int query_token_offset,
    int topology_start_tokens,
    int topology_tokens,
    int tokens_per_frame,
    int temporal_neighbor_frames)
{
  if (key_block < prefix_blocks)
    return true;

  const int query_start = query_token_offset + query_block * kBlockTokens;
  const int query_end = query_start + kBlockTokens;
  const int key_start = key_block * kBlockTokens;
  const int key_end = key_start + kBlockTokens;
  const int local_tokens = local_block_radius * kBlockTokens;
  if (key_start < query_end + local_tokens &&
      key_end > query_start - local_tokens)
    return true;
  if (temporal_neighbor_frames <= 0 || topology_tokens <= 0 ||
      tokens_per_frame <= 0)
    return false;

  const int topology_end = topology_start_tokens + topology_tokens;
  if (query_start < topology_start_tokens || query_start >= topology_end)
    return false;
  const int bounded_query_end = min(query_end, topology_end);
  for (int frame_offset = 1; frame_offset <= temporal_neighbor_frames;
       ++frame_offset)
  {
    const int token_offset = frame_offset * tokens_per_frame;
    const int previous_start = query_start - token_offset;
    const int previous_end = bounded_query_end - token_offset;
    if (previous_start < topology_end && previous_end > topology_start_tokens &&
        key_start < previous_end && key_end > previous_start)
      return true;
    const int next_start = query_start + token_offset;
    const int next_end = bounded_query_end + token_offset;
    if (next_start < topology_end && next_end > topology_start_tokens &&
        key_start < next_end && key_end > next_start)
      return true;
  }
  return false;
}

__device__ __forceinline__ float block_reduce(
    float value,
    float *__restrict__ scratch,
    bool maximum)
{
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
  {
    const float other = __shfl_down_sync(0xffffffff, value, offset);
    value = maximum ? fmaxf(value, other) : value + other;
  }
  if (lane == 0)
    scratch[warp] = value;
  __syncthreads();
  if (warp == 0)
  {
    value = lane < kHeadDim / WARP_SIZE
        ? scratch[lane]
        : (maximum ? -3.402823466e+38F : 0.0f);
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
      const float other = __shfl_down_sync(0xffffffff, value, offset);
      value = maximum ? fmaxf(value, other) : value + other;
    }
    if (lane == 0)
      scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

template <int Rows, typename T>
__device__ __forceinline__ void load_half_tile(
    const T *__restrict__ source,
    int64_t stride_sequence,
    int row_start,
    int row_limit,
    const smem_t<SwizzleMode::k128B, kHalfPacks> &destination)
{
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  static_assert(Rows > 0 && Rows <= kBlockTokens && Rows % 16 == 0);
  constexpr int tile_packs = Rows * kHalfPacks;
  for (int line = linear_thread; line < tile_packs; line += kWarps * WARP_SIZE)
  {
    const int row = line / kHalfPacks;
    const int column = line % kHalfPacks;
    const uint32_t offset = destination.get_permuted_offset(row, column);
    if (row_start + row < row_limit)
    {
      destination.base[offset] =
          pack_to_half(source + static_cast<int64_t>(row_start + row) * stride_sequence + column * 8);
    }
    else
    {
      destination.base[offset] = make_uint4(0, 0, 0, 0);
    }
  }
}

__device__ __forceinline__ void load_int8_tile(
    const int8_t *__restrict__ source,
    int64_t stride_sequence,
    int row_start,
    int row_limit,
    const smem_t<SwizzleMode::k128B, kInt8Packs> &destination)
{
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  for (int line = linear_thread; line < kInt8TilePacks;
       line += kWarps * WARP_SIZE)
  {
    const int row = line / kInt8Packs;
    const int column = line % kInt8Packs;
    const uint32_t offset = destination.get_permuted_offset(row, column);
    if (row_start + row < row_limit)
    {
      destination.base[offset] = *reinterpret_cast<const b128_t *>(
          source + static_cast<int64_t>(row_start + row) * stride_sequence +
          column * 16);
    }
    else
    {
      destination.base[offset] = make_uint4(0, 0, 0, 0);
    }
  }
}

__device__ __forceinline__ void load_dequantized_int8_tile(
    const int8_t *__restrict__ source,
    int64_t stride_sequence,
    const float *__restrict__ scale,
    int row_start,
    int row_limit,
    const smem_t<SwizzleMode::k128B, kHalfPacks> &destination)
{
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  for (int line = linear_thread; line < kTilePacks;
       line += kWarps * WARP_SIZE)
  {
    const int row = line / kHalfPacks;
    const int column = line % kHalfPacks;
    const int global_row = row_start + row;
    const uint32_t offset = destination.get_permuted_offset(row, column);
    b128_t packed = make_uint4(0, 0, 0, 0);
    if (global_row < row_limit)
    {
      const float dequant_scale = scale[
          (global_row / kBlockTokens) * kWarps +
          (global_row % kBlockTokens) / (kBlockTokens / kWarps)];
      half *packed_half = reinterpret_cast<half *>(&packed);
#pragma unroll
      for (int element = 0; element < 8; ++element)
      {
        const int8_t quantized = source[
            static_cast<int64_t>(global_row) * stride_sequence +
            column * 8 + element];
        packed_half[element] = __float2half_rn(
            static_cast<float>(quantized) * dequant_scale);
      }
    }
    destination.base[offset] = packed;
  }
}

template <int KeyTiles>
__device__ __forceinline__ void compute_fp16_qk(
    const smem_t<SwizzleMode::k128B, kHalfPacks> &query,
    const smem_t<SwizzleMode::k128B, kHalfPacks> &key,
    float score[1][KeyTiles][8])
{
  static_assert(KeyTiles == 2 || KeyTiles == 4);
  uint32_t query_offset = query.get_permuted_offset(
      threadIdx.y * 16 + threadIdx.x % 16, threadIdx.x / 16);
  uint32_t key_offset = key.get_permuted_offset(
      threadIdx.x % 8 + (threadIdx.x / 16) * 8,
      (threadIdx.x / 8) % 2);

#pragma unroll
  for (int inner = 0; inner < kHeadDim / 16; ++inner)
  {
    uint32_t query_fragment[4];
    query.ldmatrix_m8n8x4(query_offset, query_fragment);
    query_offset = query.advance_offset_by_row<16>(query_offset);
    query_offset = query.advance_offset_by_column<2>(
        query_offset - 16 * kHalfPacks, inner);

#pragma unroll
    for (int key_tile = 0; key_tile < KeyTiles; ++key_tile)
    {
      uint32_t key_fragment[4];
      key.ldmatrix_m8n8x4(key_offset, key_fragment);
      key_offset = key.advance_offset_by_row<16>(key_offset);
      if (inner == 0)
      {
        mma::mma_sync_m16n16k16_row_col_f16f16f32<mma::MMAMode::kInit>(
            score[0][key_tile], query_fragment, key_fragment);
      }
      else
      {
        mma::mma_sync_m16n16k16_row_col_f16f16f32<mma::MMAMode::kInplaceUpdate>(
            score[0][key_tile], query_fragment, key_fragment);
      }
    }
    key_offset = key.advance_offset_by_column<2>(
        key_offset - KeyTiles * 16 * kHalfPacks, inner);
  }
}

__device__ __forceinline__ void compute_int8_qk(
    const smem_t<SwizzleMode::k128B, kInt8Packs> &query,
    const smem_t<SwizzleMode::k128B, kInt8Packs> &key,
    int32_t score[1][4][8])
{
  uint32_t query_offset = query.get_permuted_offset(
      threadIdx.y * 16 + threadIdx.x % 16, threadIdx.x / 16);
  uint32_t key_offset = key.get_permuted_offset(
      threadIdx.x % 8 + (threadIdx.x / 16) * 8,
      (threadIdx.x / 8) % 2);
  compute_int_qk<4, 1, 1, 4, 4,
                 SwizzleMode::k128B, kInt8Packs, DataType::kInt8>(
      query, key, score, query_offset, key_offset);
}

__device__ __forceinline__ bool route_selected(
    const int32_t *__restrict__ route_row,
    int key_block)
{
  return (static_cast<uint32_t>(route_row[key_block / kRouteTile]) >>
          (key_block % kRouteTile)) &
      1U;
}

template <typename T>
__global__ void sparse_attention_kernel(
    const int8_t *__restrict__ query_int8,
    const int8_t *__restrict__ key_int8,
    const T *__restrict__ value,
    T *__restrict__ output,
    const float *__restrict__ query_scale,
    const float *__restrict__ key_scale,
    const half *__restrict__ key_score_summary,
    const half *__restrict__ value_mean,
    const int32_t *__restrict__ route,
    int query_length,
    int key_length,
    int num_query_heads,
    int num_kv_heads,
    int num_query_blocks,
    int num_key_blocks,
    int residual_subblocks,
    int padded_residual_summaries,
    int route_words,
    int64_t stride_batch_q_int8,
    int64_t stride_head_q_int8,
    int64_t stride_sequence_q_int8,
    int64_t stride_batch_k_int8,
    int64_t stride_head_k_int8,
    int64_t stride_sequence_k_int8,
    int64_t stride_batch_v,
    int64_t stride_head_v,
    int64_t stride_sequence_v,
    int64_t stride_batch_o,
    int64_t stride_head_o,
    int64_t stride_sequence_o,
    float softmax_scale)
{
  static_assert(
      kAttentionSharedBytes == 32 * 1024,
      "SM75 sparse attention must stay within 32 KiB");
  extern __shared__ int8_t shared_bytes[];
  smem_t<SwizzleMode::k128B, kHalfPacks> shared_query(shared_bytes);
  smem_t<SwizzleMode::k128B, kHalfPacks> shared_summary_key(
      shared_bytes + kTileBytes);
  smem_t<SwizzleMode::k128B, kHalfPacks> shared_summary_value(
      shared_bytes + kTileBytes + kSummaryTileBytes);
  smem_t<SwizzleMode::k128B, kHalfPacks> shared_output(shared_bytes);
  smem_t<SwizzleMode::k128B, kInt8Packs> shared_query_int8(shared_bytes);
  smem_t<SwizzleMode::k128B, kInt8Packs> shared_key_int8(
      shared_bytes + kInt8TileBytes);
  smem_t<SwizzleMode::k128B, kHalfPacks> shared_selected_value(
      shared_bytes + 2 * kInt8TileBytes);

  const int query_block = blockIdx.x;
  const int query_head = blockIdx.y;
  const int batch = blockIdx.z;
  const int kv_head = query_head / (num_query_heads / num_kv_heads);
  const int8_t *query_int8_head_ptr = query_int8 +
      batch * stride_batch_q_int8 + query_head * stride_head_q_int8;
  const int8_t *key_int8_head_ptr = key_int8 +
      batch * stride_batch_k_int8 + kv_head * stride_head_k_int8;
  const T *value_head_ptr =
      value + batch * stride_batch_v + kv_head * stride_head_v;
  T *output_head_ptr =
      output + batch * stride_batch_o + query_head * stride_head_o;
  const half *key_score_summary_head = key_score_summary +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) *
          padded_residual_summaries * kHeadDim;
  const half *value_mean_head = value_mean +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) *
          padded_residual_summaries * kHeadDim;
  const int32_t *route_row = route +
      ((static_cast<int64_t>(batch) * num_query_heads + query_head) *
           num_query_blocks +
       query_block) *
          route_words;
  const float *query_scale_head = query_scale +
      (static_cast<int64_t>(batch) * num_query_heads + query_head) *
          num_query_blocks * kWarps;
  const float q_dequant_scale =
      query_scale_head[query_block * kWarps + threadIdx.y];
  const float *key_scale_head = key_scale +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) * num_key_blocks;

  float output_fragment[1][8][8];
  float row_max[1][2];
  float denominator[1][2];
#pragma unroll
  for (int value_tile = 0; value_tile < 8; ++value_tile)
  {
#pragma unroll
    for (int element = 0; element < 8; ++element)
      output_fragment[0][value_tile][element] = 0.0f;
  }
  row_max[0][0] = -5000000.0f;
  row_max[0][1] = -5000000.0f;
  denominator[0][0] = 1.0f;
  denominator[0][1] = 1.0f;

  load_dequantized_int8_tile(
      query_int8_head_ptr,
      stride_sequence_q_int8,
      query_scale_head,
      query_block * kBlockTokens,
      query_length,
      shared_query);
  __syncthreads();

  const float scale_log2 = softmax_scale * math::log2e;
  const uint32_t value_mma_offset = shared_summary_value.get_permuted_offset(
      threadIdx.x % 16, threadIdx.x / 16);

  // Evaluate skipped residual summaries in groups of 32. This keeps their
  // normalization mass and V contribution without revisiting K/V tokens.
  const int residual_tokens = kBlockTokens / residual_subblocks;
  const int num_residual_summaries =
      (key_length + residual_tokens - 1) / residual_tokens;
  for (int summary_start = 0; summary_start < num_residual_summaries;
       summary_start += kSummaryTileTokens)
  {
    load_half_tile<kSummaryTileTokens>(
        key_score_summary_head,
        kHeadDim,
        summary_start,
        num_residual_summaries,
        shared_summary_key);
    load_half_tile<kSummaryTileTokens>(
        value_mean_head,
        kHeadDim,
        summary_start,
        num_residual_summaries,
        shared_summary_value);
    __syncthreads();

    float score[1][2][8];
    compute_fp16_qk<2>(shared_query, shared_summary_key, score);
#pragma unroll
    for (int key_tile = 0; key_tile < 2; ++key_tile)
    {
#pragma unroll
      for (int element = 0; element < 8; ++element)
      {
        const int local_summary = 2 * (threadIdx.x % 4) + key_tile * 16 +
            8 * (element / 4) + element % 2;
        const int residual_summary = summary_start + local_summary;
        const int key_block = residual_summary / residual_subblocks;
        if (residual_summary >= num_residual_summaries ||
            route_selected(route_row, key_block))
        {
          score[0][key_tile][element] = -5000000.0f;
        }
        else
        {
          const int residual_index = residual_summary % residual_subblocks;
          const int residual_start =
              key_block * kBlockTokens + residual_index * residual_tokens;
          const int remaining = key_length - residual_start;
          const int block_length = remaining < residual_tokens
              ? remaining
              : residual_tokens;
          score[0][key_tile][element] =
              score[0][key_tile][element] * scale_log2 +
              math::ptx_log2(static_cast<float>(block_length));
        }
      }
    }
    update_mdo<1, 2, 8, false, false, true>(
        score, output_fragment, row_max, denominator, 1.0f);
    uint32_t probability[1][2][4];
    RS_32_to_16<1, 2>(score, probability);
    accumulate_d<1, 2, ComputeUnit::kTensorCore>(probability, denominator);
    uint32_t value_offset = value_mma_offset;
    compute_fp16_sv_permuted<4, 1, 1, 2, 8,
                             SwizzleMode::k128B, kHalfPacks, 4>(
        shared_summary_value,
        probability,
        output_fragment,
        denominator,
        value_offset);
    __syncthreads();
  }

  // Selected blocks retain exact token-level attention. Q/K are quantized once
  // with the production Sage per-16-row Q and per-64-row K scales, then use the
  // same SM75 INT8 Tensor Core MMA as stable Sage. V and output stay FP16/BF16
  // with FP32 online-softmax accumulation.
  load_int8_tile(
      query_int8_head_ptr,
      stride_sequence_q_int8,
      query_block * kBlockTokens,
      query_length,
      shared_query_int8);
  __syncthreads();
  for (int key_block = 0; key_block < num_key_blocks; ++key_block)
  {
    if (!route_selected(route_row, key_block))
      continue;
    load_int8_tile(
        key_int8_head_ptr,
        stride_sequence_k_int8,
        key_block * kBlockTokens,
        key_length,
        shared_key_int8);
    load_half_tile<kBlockTokens>(
        value_head_ptr,
        stride_sequence_v,
        key_block * kBlockTokens,
        key_length,
        shared_selected_value);
    __syncthreads();

    int32_t integer_score[1][4][8];
    compute_int8_qk(shared_query_int8, shared_key_int8, integer_score);
    float score[1][4][8];
#pragma unroll
    for (int key_tile = 0; key_tile < 4; ++key_tile)
    {
#pragma unroll
      for (int element = 0; element < 8; ++element)
        score[0][key_tile][element] =
            __int2float_rz(integer_score[0][key_tile][element]);
    }
    const uint32_t key_lane_base =
        key_block * kBlockTokens + 2 * (threadIdx.x % 4);
    apply_out_of_bound_mask<1, 4>(key_lane_base, score, key_length);
    update_mdo<1, 4, 8, false, false, false>(
        score,
        output_fragment,
        row_max,
        denominator,
        scale_log2 * q_dequant_scale * key_scale_head[key_block]);
    uint32_t probability[1][4][4];
    RS_32_to_16<1, 4>(score, probability);
    accumulate_d<1, 4, ComputeUnit::kTensorCore>(probability, denominator);
    uint32_t value_offset = value_mma_offset;
    compute_fp16_sv_permuted<4, 1, 1, 4, 8,
                             SwizzleMode::k128B, kHalfPacks, 4>(
        shared_selected_value,
        probability,
        output_fragment,
        denominator,
        value_offset);
    __syncthreads();
  }

  normalize_d<1, 8, ComputeUnit::kTensorCore>(
      output_fragment, row_max, denominator);

  const uint32_t output_row_base = threadIdx.y * 16 + threadIdx.x / 4;
#pragma unroll
  for (int value_tile = 0; value_tile < 8; ++value_tile)
  {
    const uint32_t output_offset = shared_output.get_permuted_offset(
        output_row_base, value_tile * 2);
    uint32_t converted[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair)
    {
      if constexpr (std::is_same<T, half>::value)
      {
        reinterpret_cast<half2 *>(converted)[pair] =
            __float22half2_rn(reinterpret_cast<float2 *>(output_fragment[0][value_tile])[pair]);
      }
      else
      {
        reinterpret_cast<nv_bfloat162 *>(converted)[pair] =
            __float22bfloat162_rn(
                reinterpret_cast<float2 *>(output_fragment[0][value_tile])[pair]);
      }
    }
    reinterpret_cast<uint32_t *>(shared_output.base + output_offset)[threadIdx.x % 4] =
        converted[0];
    reinterpret_cast<uint32_t *>(
        shared_output.base + output_offset + 8 * kHalfPacks)[threadIdx.x % 4] =
        converted[1];
    reinterpret_cast<uint32_t *>(shared_output.base + (output_offset ^ 0x1))[threadIdx.x % 4] =
        converted[2];
    reinterpret_cast<uint32_t *>(
        shared_output.base + (output_offset ^ 0x1) + 8 * kHalfPacks)[threadIdx.x % 4] =
        converted[3];
  }
  __syncthreads();

  constexpr int output_line_lanes = 8;
  constexpr int output_rows_per_warp = 4;
  T *output_lane = output_head_ptr +
      (query_block * kBlockTokens + threadIdx.y * 16 +
       threadIdx.x / output_line_lanes) *
          stride_sequence_o +
      (threadIdx.x % output_line_lanes) * 8;
  uint32_t output_offset = shared_output.get_permuted_offset(
      threadIdx.y * 16 + threadIdx.x / output_line_lanes,
      threadIdx.x % output_line_lanes);
  int output_row = query_block * kBlockTokens + threadIdx.y * 16 +
      threadIdx.x / output_line_lanes;
#pragma unroll
  for (int row_group = 0; row_group < 4; ++row_group)
  {
#pragma unroll
    for (int column_group = 0; column_group < 2; ++column_group)
    {
      if (output_row < query_length)
        shared_output.store_128b(output_offset, output_lane);
      output_lane += output_line_lanes * 8;
      output_offset = shared_output.advance_offset_by_column<8>(output_offset);
    }
    output_offset = shared_output.advance_offset_by_row<output_rows_per_warp>(
        output_offset - 2 * output_line_lanes);
    output_lane += output_rows_per_warp * stride_sequence_o -
        2 * output_line_lanes * 8;
    output_row += output_rows_per_warp;
  }
}

void check_launch(const char *name)
{
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, name, " launch failed: ", cudaGetErrorString(error));
}

template <typename T>
void launch_sparse_threshold_attention(
    at::Tensor query,
    at::Tensor key,
    at::Tensor query_int8,
    at::Tensor key_int8,
    at::Tensor value,
    at::Tensor output,
    at::Tensor query_scale,
    at::Tensor key_scale,
    at::Tensor query_summary,
    at::Tensor key_summary,
    at::Tensor key_score_summary,
    at::Tensor value_mean,
    at::Tensor key_summary_mean,
    at::Tensor key_summary_variance,
    at::Tensor thresholds,
    at::Tensor route,
    int prefix_blocks,
    int local_block_radius,
    int query_token_offset,
    int topology_start_tokens,
    int topology_tokens,
    int tokens_per_frame,
    int temporal_neighbor_frames,
    int residual_subblocks,
    float minimum_route_density,
    float maximum_route_density,
    float threshold_sigma,
    float softmax_scale)
{
  const int batch_size = query.size(0);
  const int num_query_heads = query.size(1);
  const int num_kv_heads = key.size(1);
  const int query_length = query.size(2);
  const int key_length = key.size(2);
  const int num_query_blocks = div_ceil(query_length, kBlockTokens);
  const int num_key_blocks = div_ceil(key_length, kBlockTokens);
  const int padded_query_blocks = query_summary.size(2);
  const int padded_key_blocks = key_summary.size(2);
  const int padded_residual_summaries = key_score_summary.size(2);
  const int route_words = route.size(3);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  dim3 query_summary_grid(padded_query_blocks, num_query_heads, batch_size);
  block_summary_kernel<T><<<query_summary_grid, kHeadDim, 0, stream>>>(
      reinterpret_cast<const T *>(query.data_ptr()),
      reinterpret_cast<half *>(query_summary.data_ptr()),
      batch_size,
      num_query_heads,
      query_length,
      padded_query_blocks,
      query.stride(0),
      query.stride(1),
      query.stride(2));
  check_launch("sparse query summary");

  dim3 key_summary_grid(padded_key_blocks, num_kv_heads, batch_size);
  kv_block_summary_kernel<T><<<key_summary_grid, kHeadDim, 0, stream>>>(
      reinterpret_cast<const T *>(key.data_ptr()),
      key_int8.data_ptr<int8_t>(),
      key_scale.data_ptr<float>(),
      reinterpret_cast<const T *>(value.data_ptr()),
      reinterpret_cast<half *>(key_summary.data_ptr()),
      reinterpret_cast<half *>(key_score_summary.data_ptr()),
      reinterpret_cast<half *>(value_mean.data_ptr()),
      batch_size,
      num_kv_heads,
      key_length,
      padded_key_blocks,
      residual_subblocks,
      padded_residual_summaries,
      key.stride(0),
      key.stride(1),
      key.stride(2),
      key_int8.stride(0),
      key_int8.stride(1),
      key_int8.stride(2),
      value.stride(0),
      value.stride(1),
      value.stride(2));
  check_launch("sparse K/V summary");

  key_summary_stats_kernel<<<
      batch_size * num_kv_heads,
      kHeadDim,
      0,
      stream>>>(
      reinterpret_cast<const half *>(key_summary.data_ptr()),
      key_summary_mean.data_ptr<float>(),
      key_summary_variance.data_ptr<float>(),
      num_key_blocks,
      padded_key_blocks);
  check_launch("sparse key summary statistics");

  dim3 threshold_grid(num_query_blocks, num_query_heads, batch_size);
  query_threshold_kernel<<<threshold_grid, kHeadDim, 0, stream>>>(
      reinterpret_cast<const half *>(query_summary.data_ptr()),
      key_summary_mean.data_ptr<float>(),
      key_summary_variance.data_ptr<float>(),
      thresholds.data_ptr<float>(),
      num_query_heads,
      num_kv_heads,
      num_query_blocks,
      padded_query_blocks,
      threshold_sigma);
  check_launch("sparse query thresholds");

  dim3 route_grid(
      route_words,
      div_ceil(num_query_blocks, kRouteTile),
      batch_size * num_query_heads);
  route_threshold_fused_kernel<<<route_grid, WARP_SIZE, 0, stream>>>(
      reinterpret_cast<const half *>(query_summary.data_ptr()),
      reinterpret_cast<const half *>(key_summary.data_ptr()),
      thresholds.data_ptr<float>(),
      route.data_ptr<int32_t>(),
      batch_size,
      num_query_heads,
      num_kv_heads,
      num_query_blocks,
      num_key_blocks,
      padded_query_blocks,
      padded_key_blocks,
      route_words,
      prefix_blocks,
      local_block_radius,
      query_token_offset,
      topology_start_tokens,
      topology_tokens,
      tokens_per_frame,
      temporal_neighbor_frames,
      minimum_route_density,
      maximum_route_density);
  check_launch("sparse fused threshold route");

  dim3 attention_grid(num_query_blocks, num_query_heads, batch_size);
  dim3 attention_block(WARP_SIZE, kWarps);
  sparse_attention_kernel<T><<<
      attention_grid,
      attention_block,
      kAttentionSharedBytes,
      stream>>>(
      query_int8.data_ptr<int8_t>(),
      key_int8.data_ptr<int8_t>(),
      reinterpret_cast<const T *>(value.data_ptr()),
      reinterpret_cast<T *>(output.data_ptr()),
      query_scale.data_ptr<float>(),
      key_scale.data_ptr<float>(),
      reinterpret_cast<const half *>(key_score_summary.data_ptr()),
      reinterpret_cast<const half *>(value_mean.data_ptr()),
      route.data_ptr<int32_t>(),
      query_length,
      key_length,
      num_query_heads,
      num_kv_heads,
      num_query_blocks,
      num_key_blocks,
      residual_subblocks,
      padded_residual_summaries,
      route_words,
      query_int8.stride(0),
      query_int8.stride(1),
      query_int8.stride(2),
      key_int8.stride(0),
      key_int8.stride(1),
      key_int8.stride(2),
      value.stride(0),
      value.stride(1),
      value.stride(2),
      output.stride(0),
      output.stride(1),
      output.stride(2),
      softmax_scale);
  check_launch("sparse attention");
}

} // namespace

static at::Tensor sol_sparse_threshold_int8_f16_attn_impl(
    at::Tensor query,
    at::Tensor key,
    at::Tensor query_int8,
    at::Tensor key_int8,
    at::Tensor value,
    at::Tensor output,
    at::Tensor query_scale,
    at::Tensor key_scale,
    int prefix_tokens,
    float threshold_sigma,
    int local_block_radius,
    int topology_start_tokens,
    int topology_tokens,
    int tokens_per_frame,
    int temporal_neighbor_frames,
    int residual_subblocks,
    float minimum_route_density,
    float maximum_route_density,
    int query_token_offset,
    float softmax_scale)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(query_int8);
  CHECK_CUDA(key_int8);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_LASTDIM_CONTIGUOUS(query);
  CHECK_LASTDIM_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(query_int8);
  CHECK_LASTDIM_CONTIGUOUS(key_int8);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(query_int8, 4);
  CHECK_DIMS(key_int8, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  TORCH_CHECK(
      query.scalar_type() == at::ScalarType::Half ||
          query.scalar_type() == at::ScalarType::BFloat16,
      "sparse attention Q/K/V must be float16 or bfloat16");
  TORCH_CHECK(
      key.scalar_type() == query.scalar_type() &&
          value.scalar_type() == query.scalar_type() &&
          output.scalar_type() == query.scalar_type(),
      "sparse attention Q/K/V/output dtypes must match");
  CHECK_DTYPE(query_int8, at::ScalarType::Char);
  CHECK_DTYPE(key_int8, at::ScalarType::Char);
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);
  TORCH_CHECK(
      query.device() == key.device() && query.device() == value.device() &&
          query.device() == output.device() &&
          query.device() == query_int8.device() &&
          query.device() == key_int8.device() &&
          query.device() == query_scale.device() &&
          query.device() == key_scale.device(),
      "sparse attention tensors must share one CUDA device");
  TORCH_CHECK(
      query.size(0) == key.size(0) && query.size(0) == value.size(0),
      "sparse attention batch sizes must match");
  TORCH_CHECK(
      key.size(1) == value.size(1) && key.size(2) == value.size(2),
      "sparse attention K/V shapes must match");
  TORCH_CHECK(
      query_int8.sizes() == query.sizes() && key_int8.sizes() == key.sizes(),
      "sparse attention INT8 Q/K must match the original Q/K shapes");
  TORCH_CHECK(
      query.size(3) == kHeadDim && key.size(3) == kHeadDim &&
          value.size(3) == kHeadDim,
      "experimental sparse attention requires head_dim=128");
  TORCH_CHECK(
      key.size(1) > 0 && query.size(1) % key.size(1) == 0,
      "sparse attention Q heads must be divisible by KV heads");
  TORCH_CHECK(query.size(2) > 0 && key.size(2) > 0, "empty attention is unsupported");
  TORCH_CHECK(
      output.sizes() == query.sizes(), "sparse attention output must match Q shape");
  const int batch_size = query.size(0);
  const int num_query_heads = query.size(1);
  const int num_kv_heads = key.size(1);
  const int num_query_blocks = div_ceil(query.size(2), kBlockTokens);
  const int num_key_blocks = div_ceil(key.size(2), kBlockTokens);
  TORCH_CHECK(
      query_scale.size(0) == batch_size &&
          query_scale.size(1) == num_query_heads &&
          query_scale.size(2) == num_query_blocks * kWarps,
      "sparse attention Q scale must have shape [B, Hq, ceil(Q/64) * 4]");
  TORCH_CHECK(
      key_scale.size(0) == batch_size &&
          key_scale.size(1) == num_kv_heads &&
          key_scale.size(2) == num_key_blocks,
      "sparse attention K scale must have shape [B, Hkv, ceil(K/64)]");
  TORCH_CHECK(
      prefix_tokens >= 0 && prefix_tokens <= key.size(2),
      "sparse attention prefix_tokens is outside the K sequence");
  TORCH_CHECK(
      query_token_offset >= 0 &&
          ((query_token_offset == 0 && prefix_tokens == 0) ||
           query_token_offset + query.size(2) <= key.size(2)),
      "sparse attention target Query range is outside the shared K sequence");
  TORCH_CHECK(
      std::isfinite(threshold_sigma),
      "sparse attention threshold_sigma must be finite");
  TORCH_CHECK(
      local_block_radius >= 0,
      "sparse attention local_block_radius must be non-negative");
  TORCH_CHECK(
      residual_subblocks == 1 || residual_subblocks == 2,
      "sparse attention residual_subblocks must be 1 or 2");
  TORCH_CHECK(
      std::isfinite(minimum_route_density) &&
          std::isfinite(maximum_route_density) &&
          minimum_route_density >= 0.0f && maximum_route_density <= 1.0f &&
          minimum_route_density <= maximum_route_density,
      "sparse attention route density bounds must satisfy 0 <= minimum <= maximum <= 1");
  TORCH_CHECK(
      topology_start_tokens >= 0 && topology_tokens >= 0 &&
          tokens_per_frame >= 0 && temporal_neighbor_frames >= 0,
      "sparse attention topology values must be non-negative");
  TORCH_CHECK(
      topology_tokens == 0 ||
          (tokens_per_frame > 0 &&
           topology_start_tokens + topology_tokens <= key.size(2)),
      "sparse attention topology is outside the shared Q/K sequence");

  const int padded_query_blocks = div_ceil(num_query_blocks, kRouteTile) * kRouteTile;
  const int padded_key_blocks = div_ceil(num_key_blocks, kRouteTile) * kRouteTile;
  const int residual_tokens = kBlockTokens / residual_subblocks;
  const int num_residual_summaries =
      div_ceil(static_cast<int>(key.size(2)), residual_tokens);
  const int padded_residual_summaries =
      div_ceil(num_residual_summaries, kSummaryTileTokens) * kSummaryTileTokens;
  const int route_words = div_ceil(num_key_blocks, kRouteTile);
  const int prefix_blocks = div_ceil(prefix_tokens, kBlockTokens);

  const auto half_options = query.options().dtype(at::ScalarType::Half);
  const auto float_options = query.options().dtype(at::ScalarType::Float);
  const auto int_options = query.options().dtype(at::ScalarType::Int);
  at::Tensor query_summary = at::empty(
      {batch_size, num_query_heads, padded_query_blocks, kHeadDim}, half_options);
  at::Tensor key_summary = at::empty(
      {batch_size, num_kv_heads, padded_key_blocks, kHeadDim}, half_options);
  at::Tensor key_score_summary = at::empty(
      {batch_size, num_kv_heads, padded_residual_summaries, kHeadDim},
      half_options);
  at::Tensor value_mean = at::empty_like(key_score_summary);
  at::Tensor route = at::empty(
      {batch_size, num_query_heads, num_query_blocks, route_words},
      int_options);

  {
    at::Tensor key_summary_mean = at::empty(
        {batch_size, num_kv_heads, kHeadDim}, float_options);
    at::Tensor key_summary_variance = at::empty_like(key_summary_mean);
    at::Tensor thresholds = at::empty(
        {batch_size, num_query_heads, num_query_blocks}, float_options);
    if (query.scalar_type() == at::ScalarType::Half)
    {
      launch_sparse_threshold_attention<half>(
          query,
          key,
          query_int8,
          key_int8,
          value,
          output,
          query_scale,
          key_scale,
          query_summary,
          key_summary,
          key_score_summary,
          value_mean,
          key_summary_mean,
          key_summary_variance,
          thresholds,
          route,
          prefix_blocks,
          local_block_radius,
          query_token_offset,
          topology_start_tokens,
          topology_tokens,
          tokens_per_frame,
          temporal_neighbor_frames,
          residual_subblocks,
          minimum_route_density,
          maximum_route_density,
          threshold_sigma,
          softmax_scale);
    }
    else
    {
      launch_sparse_threshold_attention<nv_bfloat16>(
          query,
          key,
          query_int8,
          key_int8,
          value,
          output,
          query_scale,
          key_scale,
          query_summary,
          key_summary,
          key_score_summary,
          value_mean,
          key_summary_mean,
          key_summary_variance,
          thresholds,
          route,
          prefix_blocks,
          local_block_radius,
          query_token_offset,
          topology_start_tokens,
          topology_tokens,
          tokens_per_frame,
          temporal_neighbor_frames,
          residual_subblocks,
          minimum_route_density,
          maximum_route_density,
          threshold_sigma,
          softmax_scale);
    }
    return route;
  }
}

at::Tensor sol_sparse_threshold_int8_f16_attn(
    at::Tensor query,
    at::Tensor key,
    at::Tensor query_int8,
    at::Tensor key_int8,
    at::Tensor value,
    at::Tensor output,
    at::Tensor query_scale,
    at::Tensor key_scale,
    int prefix_tokens,
    float threshold_sigma,
    int local_block_radius,
    int topology_start_tokens,
    int topology_tokens,
    int tokens_per_frame,
    int temporal_neighbor_frames,
    int residual_subblocks,
    float minimum_route_density,
    float maximum_route_density,
    int query_token_offset,
    float softmax_scale)
{
  return sol_sparse_threshold_int8_f16_attn_impl(
      query,
      key,
      query_int8,
      key_int8,
      value,
      output,
      query_scale,
      key_scale,
      prefix_tokens,
      threshold_sigma,
      local_block_radius,
      topology_start_tokens,
      topology_tokens,
      tokens_per_frame,
      temporal_neighbor_frames,
      residual_subblocks,
      minimum_route_density,
      maximum_route_density,
      query_token_offset,
      softmax_scale);
}

at::Tensor sol_sparse_route_selected(at::Tensor route)
{
  CHECK_CUDA(route);
  CHECK_DIMS(route, 4);
  TORCH_CHECK(route.is_contiguous(), "sparse attention route must be contiguous");
  TORCH_CHECK(
      route.scalar_type() == at::ScalarType::Int,
      "sparse attention route must use int32 words");
  at::Tensor selected = at::zeros(
      {1}, route.options().dtype(at::ScalarType::Long));
  if (route.numel() == 0)
    return selected;

  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>(
      div_ceil(route.numel(), threads), 4096));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  route_popcount_kernel<<<blocks, threads, 0, stream>>>(
      route.data_ptr<int32_t>(),
      route.numel(),
      reinterpret_cast<unsigned long long *>(selected.data_ptr<int64_t>()));
  check_launch("sparse route popcount");
  return selected;
}
