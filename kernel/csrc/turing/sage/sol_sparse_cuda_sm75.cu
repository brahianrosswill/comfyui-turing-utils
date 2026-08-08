/*
 * Experimental Sol-style sparse attention for SM75.
 *
 * The routing pass uses one 64-token centroid per block and selects an
 * input-relative top-p mass budget. Selected blocks are evaluated exactly with
 * FP16 tensor cores; skipped blocks keep a centroid contribution in the same
 * online softmax instead of being dropped.
 */

#include "../utils.cuh"
#include "../math.cuh"
#include "attn_utils.cuh"
#include "torch_compat.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <type_traits>

namespace {

constexpr int kBlockTokens = 64;
constexpr int kHeadDim = 128;
constexpr int kWarps = 4;
constexpr int kRouteTile = 16;
constexpr int kHalfPacks = kHeadDim / 8;
constexpr int kTilePacks = kBlockTokens * kHalfPacks;
constexpr int kTileBytes = kBlockTokens * kHeadDim * sizeof(half);

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
    const T *__restrict__ value,
    half *__restrict__ key_summary,
    half *__restrict__ value_mean,
    float *__restrict__ key_block_variance,
    int batch_size,
    int num_heads,
    int sequence_length,
    int padded_blocks,
    int64_t stride_batch_k,
    int64_t stride_head_k,
    int64_t stride_sequence_k,
    int64_t stride_batch_v,
    int64_t stride_head_v,
    int64_t stride_sequence_v)
{
  __shared__ float variance_parts[kHeadDim];
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
  float key_square_sum = 0.0f;
  float value_sum = 0.0f;
  const T *head_key = key + batch * stride_batch_k + head * stride_head_k;
  const T *head_value = value + batch * stride_batch_v + head * stride_head_v;
  for (int token = 0; token < token_count; ++token)
  {
    const float key_value = scalar_to_float(
        head_key[(token_start + token) * stride_sequence_k + dimension]);
    key_sum += key_value;
    key_square_sum = fmaf(key_value, key_value, key_square_sum);
    value_sum += scalar_to_float(head_value[(token_start + token) * stride_sequence_v + dimension]);
  }
  const int64_t output_index =
      ((static_cast<int64_t>(batch) * num_heads + head) * padded_blocks + block_index) *
          kHeadDim +
      dimension;
  if (token_count)
  {
    const float reciprocal = 1.0f / token_count;
    const float key_mean = key_sum * reciprocal;
    key_summary[output_index] = __float2half_rn(key_mean);
    value_mean[output_index] = __float2half_rn(value_sum * reciprocal);
    variance_parts[dimension] =
        fmaxf(key_square_sum * reciprocal - key_mean * key_mean, 0.0f);
  }
  else
  {
    key_summary[output_index] = __float2half_rn(0.0f);
    value_mean[output_index] = __float2half_rn(0.0f);
    variance_parts[dimension] = 0.0f;
  }
  __syncthreads();
  for (int stride = kHeadDim / 2; stride > 0; stride >>= 1)
  {
    if (dimension < stride)
      variance_parts[dimension] += variance_parts[dimension + stride];
    __syncthreads();
  }
  if (dimension == 0)
  {
    key_block_variance[
        (static_cast<int64_t>(batch) * num_heads + head) * padded_blocks +
        block_index] = variance_parts[0] / kHeadDim;
  }
}

__global__ void route_score_kernel(
    const half *__restrict__ query_summary,
    const half *__restrict__ key_summary,
    half *__restrict__ route_scores,
    int batch_size,
    int num_query_heads,
    int num_kv_heads,
    int num_query_blocks,
    int num_key_blocks,
    int padded_query_blocks,
    int padded_key_blocks)
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
      ((static_cast<int64_t>(batch) * num_kv_heads + kv_head) * padded_key_blocks +
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
      for (int local_key = 0; local_key < kRouteTile; ++local_key)
      {
        const int key_block = key_start + local_key;
        if (key_block >= num_key_blocks)
          continue;
        route_scores[
            ((static_cast<int64_t>(batch) * num_query_heads + query_head) *
                 num_query_blocks +
             query_block) *
                padded_key_blocks +
            key_block] = __float2half_rn(
                scores[local_query * kRouteTile + local_key]);
      }
    }
  }
}

__device__ __forceinline__ bool forced_route_block(
    int query_block,
    int key_block,
    int prefix_blocks,
    int local_block_radius)
{
  const int distance = query_block > key_block
      ? query_block - key_block
      : key_block - query_block;
  return query_block < prefix_blocks || key_block < prefix_blocks ||
      distance <= local_block_radius;
}

__device__ __forceinline__ float route_log_weight(
    const half *__restrict__ scores,
    const float *__restrict__ key_block_variance,
    int key_block,
    int key_length,
    float softmax_scale)
{
  const int remaining = key_length - key_block * kBlockTokens;
  const int block_length = remaining < kBlockTokens ? remaining : kBlockTokens;
  return __half2float(scores[key_block]) * softmax_scale +
      logf(static_cast<float>(block_length)) +
      0.5f * softmax_scale * softmax_scale * kHeadDim *
          key_block_variance[key_block];
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

__global__ void route_top_p_kernel(
    const half *__restrict__ route_scores,
    const float *__restrict__ key_block_variance,
    int32_t *__restrict__ route,
    int key_length,
    int num_query_heads,
    int num_kv_heads,
    int num_query_blocks,
    int num_key_blocks,
    int padded_key_blocks,
    int route_words,
    int prefix_blocks,
    int local_block_radius,
    float attention_mass_recall,
    float softmax_scale)
{
  extern __shared__ float log_weights[];
  __shared__ float reduction[kHeadDim];
  __shared__ float lower_threshold;
  __shared__ float upper_threshold;
  __shared__ float required_mass;

  const int query_block = blockIdx.x;
  const int query_head = blockIdx.y;
  const int batch = blockIdx.z;
  const int kv_head = query_head / (num_query_heads / num_kv_heads);
  const half *scores = route_scores +
      ((static_cast<int64_t>(batch) * num_query_heads + query_head) *
           num_query_blocks +
       query_block) *
          padded_key_blocks;
  const float *variance = key_block_variance +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) * padded_key_blocks;
  int32_t *route_row = route +
      ((static_cast<int64_t>(batch) * num_query_heads + query_head) *
           num_query_blocks +
       query_block) *
          route_words;

  float local_max = -3.402823466e+38F;
  for (int key_block = threadIdx.x; key_block < num_key_blocks;
       key_block += blockDim.x)
  {
    const float log_weight =
        route_log_weight(scores, variance, key_block, key_length, softmax_scale);
    log_weights[key_block] = log_weight;
    local_max = fmaxf(local_max, log_weight);
  }
  const float max_log_weight = block_reduce(local_max, reduction, true);

  float local_total = 0.0f;
  float local_forced = 0.0f;
  float local_nonforced_min = 3.402823466e+38F;
  for (int key_block = threadIdx.x; key_block < num_key_blocks;
       key_block += blockDim.x)
  {
    const float log_weight = log_weights[key_block];
    const float mass = __expf(log_weight - max_log_weight);
    const bool forced = forced_route_block(
        query_block, key_block, prefix_blocks, local_block_radius);
    local_total += mass;
    local_forced += forced ? mass : 0.0f;
    if (!forced)
      local_nonforced_min = fminf(local_nonforced_min, log_weight);
  }
  const float total_mass = block_reduce(local_total, reduction, false);
  const float forced_mass = block_reduce(local_forced, reduction, false);
  const float min_nonforced_log = -block_reduce(-local_nonforced_min, reduction, true);

  if (threadIdx.x == 0)
  {
    required_mass = attention_mass_recall * total_mass - forced_mass;
    lower_threshold = min_nonforced_log;
    upper_threshold = max_log_weight;
  }
  __syncthreads();

  if (required_mass > 0.0f)
  {
#pragma unroll
    for (int iteration = 0; iteration < 12; ++iteration)
    {
      const float candidate = 0.5f * (lower_threshold + upper_threshold);
      float local_selected_mass = 0.0f;
      for (int key_block = threadIdx.x; key_block < num_key_blocks;
           key_block += blockDim.x)
      {
        if (!forced_route_block(
                query_block, key_block, prefix_blocks, local_block_radius) &&
            log_weights[key_block] >= candidate)
          local_selected_mass += __expf(log_weights[key_block] - max_log_weight);
      }
      const float selected_mass = block_reduce(
          local_selected_mass, reduction, false);
      if (threadIdx.x == 0)
      {
        if (selected_mass >= required_mass)
          lower_threshold = candidate;
        else
          upper_threshold = candidate;
      }
      __syncthreads();
    }
  }
  else if (threadIdx.x == 0)
  {
    lower_threshold = 3.402823466e+38F;
  }
  __syncthreads();

  for (int key_block = threadIdx.x; key_block < num_key_blocks;
       key_block += blockDim.x)
  {
    bool selected = forced_route_block(
        query_block, key_block, prefix_blocks, local_block_radius);
    if (!selected)
      selected = log_weights[key_block] >= lower_threshold;
    if (selected)
    {
      atomicOr(
          reinterpret_cast<unsigned int *>(route_row + key_block / kRouteTile),
          1U << (key_block % kRouteTile));
    }
  }
}

template <typename T>
__device__ __forceinline__ void load_half_tile(
    const T *__restrict__ source,
    int64_t stride_sequence,
    int row_start,
    int row_limit,
    const smem_t<SwizzleMode::k128B, kHalfPacks> &destination)
{
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  for (int line = linear_thread; line < kTilePacks; line += kWarps * WARP_SIZE)
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

__device__ __forceinline__ void compute_fp16_qk(
    const smem_t<SwizzleMode::k128B, kHalfPacks> &query,
    const smem_t<SwizzleMode::k128B, kHalfPacks> &key,
    float score[1][4][8])
{
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
    for (int key_tile = 0; key_tile < 4; ++key_tile)
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
        key_offset - 64 * kHalfPacks, inner);
  }
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
    const T *__restrict__ query,
    const T *__restrict__ key,
    const T *__restrict__ value,
    T *__restrict__ output,
    const half *__restrict__ key_summary,
    const half *__restrict__ value_mean,
    const float *__restrict__ key_block_variance,
    const int32_t *__restrict__ route,
    int query_length,
    int key_length,
    int num_query_heads,
    int num_kv_heads,
    int num_query_blocks,
    int num_key_blocks,
    int padded_key_blocks,
    int route_words,
    int64_t stride_batch_q,
    int64_t stride_head_q,
    int64_t stride_sequence_q,
    int64_t stride_batch_k,
    int64_t stride_head_k,
    int64_t stride_sequence_k,
    int64_t stride_batch_v,
    int64_t stride_head_v,
    int64_t stride_sequence_v,
    int64_t stride_batch_o,
    int64_t stride_head_o,
    int64_t stride_sequence_o,
    float softmax_scale)
{
  static_assert(3 * kTileBytes <= 48 * 1024, "SM75 sparse attention must stay within 48 KiB");
  extern __shared__ int8_t shared_bytes[];
  smem_t<SwizzleMode::k128B, kHalfPacks> shared_query(shared_bytes);
  smem_t<SwizzleMode::k128B, kHalfPacks> shared_key(shared_bytes + kTileBytes);
  smem_t<SwizzleMode::k128B, kHalfPacks> shared_value(shared_bytes + 2 * kTileBytes);
  smem_t<SwizzleMode::k128B, kHalfPacks> shared_output(shared_bytes);

  const int query_block = blockIdx.x;
  const int query_head = blockIdx.y;
  const int batch = blockIdx.z;
  const int kv_head = query_head / (num_query_heads / num_kv_heads);
  const T *query_head_ptr =
      query + batch * stride_batch_q + query_head * stride_head_q;
  const T *key_head_ptr = key + batch * stride_batch_k + kv_head * stride_head_k;
  const T *value_head_ptr =
      value + batch * stride_batch_v + kv_head * stride_head_v;
  T *output_head_ptr =
      output + batch * stride_batch_o + query_head * stride_head_o;
  const half *key_summary_head = key_summary +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) * padded_key_blocks *
          kHeadDim;
  const half *value_mean_head = value_mean +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) * padded_key_blocks *
          kHeadDim;
  const float *key_block_variance_head = key_block_variance +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) * padded_key_blocks;
  const int32_t *route_row = route +
      ((static_cast<int64_t>(batch) * num_query_heads + query_head) *
           num_query_blocks +
       query_block) *
          route_words;

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

  load_half_tile(
      query_head_ptr,
      stride_sequence_q,
      query_block * kBlockTokens,
      query_length,
      shared_query);
  __syncthreads();

  const float scale_log2 = softmax_scale * math::log2e;
  const uint32_t value_mma_offset = shared_value.get_permuted_offset(
      threadIdx.x % 16, threadIdx.x / 16);

  // Evaluate all skipped blocks in groups of 64 centroids. This keeps their
  // normalization mass and V contribution without revisiting K/V tokens.
  for (int summary_start = 0; summary_start < num_key_blocks;
       summary_start += kBlockTokens)
  {
    load_half_tile(
        key_summary_head,
        kHeadDim,
        summary_start,
        num_key_blocks,
        shared_key);
    load_half_tile(
        value_mean_head,
        kHeadDim,
        summary_start,
        num_key_blocks,
        shared_value);
    __syncthreads();

    float score[1][4][8];
    compute_fp16_qk(shared_query, shared_key, score);
#pragma unroll
    for (int key_tile = 0; key_tile < 4; ++key_tile)
    {
#pragma unroll
      for (int element = 0; element < 8; ++element)
      {
        const int local_key = 2 * (threadIdx.x % 4) + key_tile * 16 +
            8 * (element / 4) + element % 2;
        const int key_block = summary_start + local_key;
        if (key_block >= num_key_blocks || route_selected(route_row, key_block))
        {
          score[0][key_tile][element] = -5000000.0f;
        }
        else
        {
          const int remaining = key_length - key_block * kBlockTokens;
          const int block_length = remaining < kBlockTokens ? remaining : kBlockTokens;
          score[0][key_tile][element] =
              score[0][key_tile][element] * scale_log2 +
              math::ptx_log2(static_cast<float>(block_length)) +
              0.5f * softmax_scale * softmax_scale * kHeadDim *
                  key_block_variance_head[key_block] * math::log2e;
        }
      }
    }
    update_mdo<1, 4, 8, false, false, true>(
        score, output_fragment, row_max, denominator, 1.0f);
    uint32_t probability[1][4][4];
    RS_32_to_16<1, 4>(score, probability);
    accumulate_d<1, 4, ComputeUnit::kTensorCore>(probability, denominator);
    uint32_t value_offset = value_mma_offset;
    compute_fp16_sv_permuted<4, 1, 1, 4, 8,
                             SwizzleMode::k128B, kHalfPacks, 4>(
        shared_value,
        probability,
        output_fragment,
        denominator,
        value_offset);
    __syncthreads();
  }

  // Selected blocks retain exact token-level attention. FP16 tensor-core QK
  // avoids the dense path's quantization buffers and is more accurate than the
  // INT8 baseline while sparsity supplies the speedup.
  for (int key_block = 0; key_block < num_key_blocks; ++key_block)
  {
    if (!route_selected(route_row, key_block))
      continue;
    load_half_tile(
        key_head_ptr,
        stride_sequence_k,
        key_block * kBlockTokens,
        key_length,
        shared_key);
    load_half_tile(
        value_head_ptr,
        stride_sequence_v,
        key_block * kBlockTokens,
        key_length,
        shared_value);
    __syncthreads();

    float score[1][4][8];
    compute_fp16_qk(shared_query, shared_key, score);
    const uint32_t key_lane_base =
        key_block * kBlockTokens + 2 * (threadIdx.x % 4);
    apply_out_of_bound_mask<1, 4>(key_lane_base, score, key_length);
    update_mdo<1, 4, 8, false, false, false>(
        score, output_fragment, row_max, denominator, scale_log2);
    uint32_t probability[1][4][4];
    RS_32_to_16<1, 4>(score, probability);
    accumulate_d<1, 4, ComputeUnit::kTensorCore>(probability, denominator);
    uint32_t value_offset = value_mma_offset;
    compute_fp16_sv_permuted<4, 1, 1, 4, 8,
                             SwizzleMode::k128B, kHalfPacks, 4>(
        shared_value,
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
void launch_sparse_attention(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor output,
    at::Tensor query_summary,
    at::Tensor key_summary,
    at::Tensor value_mean,
    at::Tensor key_block_variance,
    at::Tensor route_scores,
    at::Tensor route,
    int prefix_blocks,
    int local_block_radius,
    float attention_mass_recall,
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
      reinterpret_cast<const T *>(value.data_ptr()),
      reinterpret_cast<half *>(key_summary.data_ptr()),
      reinterpret_cast<half *>(value_mean.data_ptr()),
      key_block_variance.data_ptr<float>(),
      batch_size,
      num_kv_heads,
      key_length,
      padded_key_blocks,
      key.stride(0),
      key.stride(1),
      key.stride(2),
      value.stride(0),
      value.stride(1),
      value.stride(2));
  check_launch("sparse K/V summary");

  dim3 route_grid(route_words, div_ceil(num_query_blocks, kRouteTile),
                  batch_size * num_query_heads);
  route_score_kernel<<<route_grid, WARP_SIZE, 0, stream>>>(
      reinterpret_cast<const half *>(query_summary.data_ptr()),
      reinterpret_cast<const half *>(key_summary.data_ptr()),
      reinterpret_cast<half *>(route_scores.data_ptr()),
      batch_size,
      num_query_heads,
      num_kv_heads,
      num_query_blocks,
      num_key_blocks,
      padded_query_blocks,
      padded_key_blocks);
  check_launch("sparse route scores");

  dim3 selection_grid(num_query_blocks, num_query_heads, batch_size);
  route_top_p_kernel<<<
      selection_grid,
      kHeadDim,
      num_key_blocks * sizeof(float),
      stream>>>(
      reinterpret_cast<const half *>(route_scores.data_ptr()),
      key_block_variance.data_ptr<float>(),
      route.data_ptr<int32_t>(),
      key_length,
      num_query_heads,
      num_kv_heads,
      num_query_blocks,
      num_key_blocks,
      padded_key_blocks,
      route_words,
      prefix_blocks,
      local_block_radius,
      attention_mass_recall,
      softmax_scale);
  check_launch("sparse top-p route");

  dim3 attention_grid(num_query_blocks, num_query_heads, batch_size);
  dim3 attention_block(WARP_SIZE, kWarps);
  sparse_attention_kernel<T><<<attention_grid, attention_block, 3 * kTileBytes, stream>>>(
      reinterpret_cast<const T *>(query.data_ptr()),
      reinterpret_cast<const T *>(key.data_ptr()),
      reinterpret_cast<const T *>(value.data_ptr()),
      reinterpret_cast<T *>(output.data_ptr()),
      reinterpret_cast<const half *>(key_summary.data_ptr()),
      reinterpret_cast<const half *>(value_mean.data_ptr()),
      key_block_variance.data_ptr<float>(),
      route.data_ptr<int32_t>(),
      query_length,
      key_length,
      num_query_heads,
      num_kv_heads,
      num_query_blocks,
      num_key_blocks,
      padded_key_blocks,
      route_words,
      query.stride(0),
      query.stride(1),
      query.stride(2),
      key.stride(0),
      key.stride(1),
      key.stride(2),
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

at::Tensor sol_sparse_f16_attn(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor output,
    int prefix_tokens,
    float attention_mass_recall,
    int local_block_radius,
    float softmax_scale)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_LASTDIM_CONTIGUOUS(query);
  CHECK_LASTDIM_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  TORCH_CHECK(
      query.scalar_type() == at::ScalarType::Half ||
          query.scalar_type() == at::ScalarType::BFloat16,
      "sparse attention Q/K/V must be float16 or bfloat16");
  TORCH_CHECK(
      key.scalar_type() == query.scalar_type() &&
          value.scalar_type() == query.scalar_type() &&
          output.scalar_type() == query.scalar_type(),
      "sparse attention Q/K/V/output dtypes must match");
  TORCH_CHECK(
      query.device() == key.device() && query.device() == value.device() &&
          query.device() == output.device(),
      "sparse attention tensors must share one CUDA device");
  TORCH_CHECK(
      query.size(0) == key.size(0) && query.size(0) == value.size(0),
      "sparse attention batch sizes must match");
  TORCH_CHECK(
      key.size(1) == value.size(1) && key.size(2) == value.size(2),
      "sparse attention K/V shapes must match");
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
  TORCH_CHECK(
      prefix_tokens >= 0 && prefix_tokens <= key.size(2),
      "sparse attention prefix_tokens is outside the K sequence");
  TORCH_CHECK(
      attention_mass_recall > 0.0f && attention_mass_recall <= 1.0f,
      "sparse attention attention_mass_recall must be in (0, 1]");
  TORCH_CHECK(
      local_block_radius >= 0,
      "sparse attention local_block_radius must be non-negative");

  const int batch_size = query.size(0);
  const int num_query_heads = query.size(1);
  const int num_kv_heads = key.size(1);
  const int num_query_blocks = div_ceil(query.size(2), kBlockTokens);
  const int num_key_blocks = div_ceil(key.size(2), kBlockTokens);
  const int padded_query_blocks = div_ceil(num_query_blocks, kRouteTile) * kRouteTile;
  const int padded_key_blocks = div_ceil(num_key_blocks, kRouteTile) * kRouteTile;
  const int route_words = div_ceil(num_key_blocks, kRouteTile);
  const int prefix_blocks = div_ceil(prefix_tokens, kBlockTokens);

  const auto half_options = query.options().dtype(at::ScalarType::Half);
  const auto float_options = query.options().dtype(at::ScalarType::Float);
  const auto int_options = query.options().dtype(at::ScalarType::Int);
  at::Tensor query_summary = at::empty(
      {batch_size, num_query_heads, padded_query_blocks, kHeadDim}, half_options);
  at::Tensor key_summary = at::empty(
      {batch_size, num_kv_heads, padded_key_blocks, kHeadDim}, half_options);
  at::Tensor value_mean = at::empty_like(key_summary);
  at::Tensor key_block_variance = at::empty(
      {batch_size, num_kv_heads, padded_key_blocks}, float_options);
  at::Tensor route_scores = at::empty(
      {batch_size, num_query_heads, num_query_blocks, padded_key_blocks},
      half_options);
  at::Tensor route = at::zeros(
      {batch_size, num_query_heads, num_query_blocks, route_words}, int_options);

  if (query.scalar_type() == at::ScalarType::Half)
  {
    launch_sparse_attention<half>(
        query,
        key,
        value,
        output,
        query_summary,
        key_summary,
        value_mean,
        key_block_variance,
        route_scores,
        route,
        prefix_blocks,
        local_block_radius,
        attention_mass_recall,
        softmax_scale);
  }
  else
  {
    launch_sparse_attention<nv_bfloat16>(
        query,
        key,
        value,
        output,
        query_summary,
        key_summary,
        value_mean,
        key_block_variance,
        route_scores,
        route,
        prefix_blocks,
        local_block_radius,
        attention_mass_recall,
        softmax_scale);
  }
  return route;
}
