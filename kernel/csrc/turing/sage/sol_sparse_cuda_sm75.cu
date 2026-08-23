/*
 * Production Sol-style sparse attention for sm75 and newer architectures.
 *
 * One 64-token centroid per block feeds an input-adaptive mean + tau * std
 * threshold. Each Query CTA performs routing directly before its FP32 online
 * softmax and keeps the route in CTA-local shared memory/registers, so no full
 * global proxy or route map is materialized. Selected blocks are
 * evaluated with the production per-warp/per-block INT8 Sage QK path. Routing
 * and skipped-block correction use centroids reconstructed from those same
 * INT8 Q/K tensors and scales. Exact proxy scores stay in Sage's randomized
 * Hadamard domain, while diagonal route statistics are inverse-transformed to
 * the pre-Hadamard basis; the orthogonal transform preserves their dot
 * products but avoids estimating diagonal variance in the mixed basis. Query
 * summary, thresholding, and routing are fused into the attention CTA;
 * original V means remain isolated to the skipped-block approximation.
 * The official local neighborhood is fixed to +/- one 64-token block. Optional
 * exact-KV and dense-Query block masks carry model-independent modality policy.
 */

#include "../utils.cuh"
#include "../math.cuh"
#include "attn_utils.cuh"
#include "dispatch_utils.h"
#include "torch_compat.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <mutex>
#include <type_traits>

namespace {

constexpr int kBlockTokens = 64;
constexpr int kWarps = 4;
constexpr int kRouteTile = 16;
constexpr int kRouteWordBits = 32;
constexpr int kMaxKeyBlocks = 4096;
constexpr int kMaxRouteWords = kMaxKeyBlocks / kRouteWordBits;
constexpr int kSummaryTileTokens = 16;
constexpr int kMaxRouteBytes = kMaxRouteWords * sizeof(uint32_t);
constexpr int kProxyScratchBytes =
    kWarps * kSummaryTileTokens * sizeof(float);
constexpr int kSlaQueryBlockTokens = 128;

__device__ __forceinline__ bool route_convrot_negative_sign(int channel)
{
  constexpr uint32_t signs[4] = {
      0x1035997bu, 0x8087f5eeu, 0xee2e4e1au, 0x71132418u};
  return ((signs[channel >> 5] >> (channel & 31)) & 1u) == 0u;
}

// The exact QK path remains in the randomized Hadamard basis.  Routing uses
// the inverse-transformed block centroids so its diagonal variance model is
// evaluated in the pre-Hadamard basis.  The first five butterfly stages stay
// within a warp; D64/D128 need only one/two cross-warp shared-memory stages.
template <int HeadDim>
__device__ __forceinline__ float inverse_route_hadamard(
    float value,
    float *__restrict__ scratch)
{
  const int linear_thread = threadIdx.y * blockDim.x + threadIdx.x;
  const int lane = linear_thread & (WARP_SIZE - 1);
#pragma unroll
  for (int bit = 1; bit < WARP_SIZE; bit <<= 1)
  {
    const float other = __shfl_xor_sync(0xffffffffu, value, bit);
    value = (lane & bit) ? other - value : value + other;
  }
#pragma unroll
  for (int bit = WARP_SIZE; bit < HeadDim; bit <<= 1)
  {
    if (linear_thread < HeadDim)
      scratch[linear_thread] = value;
    __syncthreads();
    float other = 0.0f;
    if (linear_thread < HeadDim)
      other = scratch[linear_thread ^ bit];
    __syncthreads();
    if (linear_thread < HeadDim)
      value = (linear_thread & bit) ? other - value : value + other;
  }
  constexpr float scale = HeadDim == 64
      ? 0.125f
      : 0.08838834764831845f;
  value *= scale;
  if (linear_thread < HeadDim && route_convrot_negative_sign(linear_thread))
    value = -value;
  return value;
}

template <int HeadDim>
struct AttentionGeometry
{
  static_assert(HeadDim == 64 || HeadDim == 128);
  static constexpr int kHalfPacks = HeadDim / 8;
  static constexpr int kInt8Packs = HeadDim / 16;
  static constexpr int kTilePacks = kBlockTokens * kHalfPacks;
  static constexpr int kInt8TilePacks = kBlockTokens * kInt8Packs;
  static constexpr int kTileBytes = kBlockTokens * HeadDim * sizeof(half);
  static constexpr int kInt8TileBytes = kBlockTokens * HeadDim * sizeof(int8_t);
  static constexpr int kSummaryTileBytes =
      kSummaryTileTokens * HeadDim * sizeof(half);
  static constexpr int kAttentionSharedBytes = 2 * kTileBytes;
  static constexpr int kRouteStorageOffset =
      kTileBytes + 2 * kSummaryTileBytes;
  static constexpr int kSelectedStorageOffset =
      kRouteStorageOffset + kMaxRouteBytes;
  static constexpr int kSelectedCapacity =
      (kAttentionSharedBytes - kSelectedStorageOffset - sizeof(int)) /
      sizeof(uint16_t);
  static constexpr int kValueTiles = HeadDim / 16;
  static constexpr SwizzleMode kInt8Swizzle =
      HeadDim == 64 ? SwizzleMode::k64B : SwizzleMode::k128B;
  static_assert(
      kRouteStorageOffset + kMaxRouteBytes + kProxyScratchBytes <=
          kAttentionSharedBytes,
      "fused routing metadata must fit beside the 16-block summaries");
  static_assert(kSelectedCapacity >= 1024,
                "sparse route compaction must cover production sequences");
};

static_assert(AttentionGeometry<128>::kAttentionSharedBytes <= 64 * 1024);

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

template <int HeadDim, typename T, bool NormalizeValue>
__global__ void kv_block_summary_kernel(
    const int8_t *__restrict__ key_int8,
    const float *__restrict__ key_scale,
    const T *__restrict__ value,
    const float *__restrict__ value_scale,
    half *__restrict__ key_summary,
    half *__restrict__ key_score_summary,
    half *__restrict__ value_mean,
    int batch_size,
    int num_heads,
    int sequence_length,
    int padded_blocks,
    int residual_subblocks,
    int route_original_basis,
    int padded_residual_summaries,
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
  __shared__ float inverse_scratch[HeadDim];

  const int token_start = block_index * kBlockTokens;
  const int token_count = token_start < sequence_length
      ? min(kBlockTokens, sequence_length - token_start)
      : 0;
  float value_sum[2] = {0.0f, 0.0f};
  int quantized_key_sum[2] = {0, 0};
  float route_key_mean = 0.0f;
  const int8_t *head_key_int8 = key_int8 +
      batch * stride_batch_k_int8 + head * stride_head_k_int8;
  const T *head_value = value + batch * stride_batch_v + head * stride_head_v;
  for (int token = 0; token < token_count; ++token)
  {
    const int residual_index = token / (kBlockTokens / residual_subblocks);
    quantized_key_sum[residual_index] += static_cast<int>(
        head_key_int8[(token_start + token) * stride_sequence_k_int8 + dimension]);
    value_sum[residual_index] += scalar_to_float(
        head_value[(token_start + token) * stride_sequence_v + dimension]);
  }
  const int64_t output_index =
      ((static_cast<int64_t>(batch) * num_heads + head) * padded_blocks + block_index) *
          HeadDim +
      dimension;
  if (token_count)
  {
    const int num_key_blocks = (sequence_length + kBlockTokens - 1) / kBlockTokens;
    const float dequant_scale = key_scale[
        (static_cast<int64_t>(batch) * num_heads + head) * num_key_blocks +
        block_index];
    const int total_quantized_key_sum = quantized_key_sum[0] +
        (residual_subblocks == 2 ? quantized_key_sum[1] : 0);
    route_key_mean = static_cast<float>(total_quantized_key_sum) *
        dequant_scale / static_cast<float>(token_count);
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
              HeadDim +
          dimension;
      if (residual_count > 0)
      {
        const float residual_reciprocal = 1.0f / residual_count;
        if (key_score_summary != key_summary || residual_subblocks != 1)
        {
          key_score_summary[residual_output_index] = __float2half_rn(
              static_cast<float>(quantized_key_sum[residual_index]) *
              dequant_scale * residual_reciprocal);
        }
        float mean = value_sum[residual_index] * residual_reciprocal;
        if constexpr (NormalizeValue)
        {
          const float channel_scale = value_scale[
              (static_cast<int64_t>(batch) * num_heads + head) * HeadDim +
              dimension];
          mean /= channel_scale;
        }
        value_mean[residual_output_index] = __float2half_rn(mean);
      }
      else
      {
        if (key_score_summary != key_summary || residual_subblocks != 1)
          key_score_summary[residual_output_index] = __float2half_rn(0.0f);
        value_mean[residual_output_index] = __float2half_rn(0.0f);
      }
    }
  }
  else
  {
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
                HeadDim +
            dimension;
        if (key_score_summary != key_summary || residual_subblocks != 1)
          key_score_summary[residual_output_index] = __float2half_rn(0.0f);
        value_mean[residual_output_index] = __float2half_rn(0.0f);
      }
    }
  }
  if (route_original_basis)
    route_key_mean = inverse_route_hadamard<HeadDim>(
        route_key_mean, inverse_scratch);
  key_summary[output_index] = __float2half_rn(route_key_mean);
}

template <int HeadDim>
__global__ void key_summary_stats_kernel(
    const half *__restrict__ key_summary,
    float *__restrict__ key_summary_mean,
    float *__restrict__ key_summary_variance,
    int num_key_blocks,
    int padded_key_blocks)
{
  const int packed_head = blockIdx.x;
  const int dimension = threadIdx.x;
  if (dimension >= HeadDim)
    return;

  const half *head_summary =
      key_summary + static_cast<int64_t>(packed_head) * padded_key_blocks * HeadDim;
  float sum = 0.0f;
  float square_sum = 0.0f;
  for (int key_block = 0; key_block < num_key_blocks; ++key_block)
  {
    const float value = __half2float(
        head_summary[static_cast<int64_t>(key_block) * HeadDim + dimension]);
    sum += value;
    square_sum = fmaf(value, value, square_sum);
  }
  const float reciprocal = 1.0f / static_cast<float>(num_key_blocks);
  const float mean = sum * reciprocal;
  const int64_t output_index =
      static_cast<int64_t>(packed_head) * HeadDim + dimension;
  key_summary_mean[output_index] = mean;
  key_summary_variance[output_index] =
      fmaxf(square_sum * reciprocal - mean * mean, 0.0f);
}

template <int HeadDim>
__global__ void sla_query_summary_kernel(
    const int8_t *__restrict__ query_int8,
    const float *__restrict__ query_scale,
    half *__restrict__ query_summary,
    int query_length,
    int num_heads,
    int num_query_blocks_64,
    int64_t stride_batch,
    int64_t stride_head,
    int64_t stride_sequence)
{
  const int query_block = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int dimension = threadIdx.x;
  const int token_start = query_block * kSlaQueryBlockTokens;
  const int token_count = min(kSlaQueryBlockTokens, query_length - token_start);
  const int8_t *head_query = query_int8 +
      batch * stride_batch + head * stride_head;
  const float *head_scale = query_scale +
      (static_cast<int64_t>(batch) * num_heads + head) *
          num_query_blocks_64 * kWarps;
  float sum = 0.0f;
  for (int token = 0; token < token_count; ++token)
  {
    const int global_token = token_start + token;
    const float scale = head_scale[
        (global_token / kBlockTokens) * kWarps +
        (global_token % kBlockTokens) / (kBlockTokens / kWarps)];
    sum = fmaf(
        static_cast<float>(
            head_query[static_cast<int64_t>(global_token) * stride_sequence +
                       dimension]),
        scale,
        sum);
  }
  const int num_query_blocks_128 =
      div_ceil(query_length, kSlaQueryBlockTokens);
  const int64_t output_index =
      ((static_cast<int64_t>(batch) * num_heads + head) *
           num_query_blocks_128 +
       query_block) *
          HeadDim +
      dimension;
  query_summary[output_index] = __float2half_rn(
      sum / static_cast<float>(token_count));
}

template <int HeadDim>
__global__ void sla_key_summary_kernel(
    const int8_t *__restrict__ key_int8,
    const float *__restrict__ key_scale,
    half *__restrict__ key_summary,
    int key_length,
    int num_heads,
    int num_key_blocks,
    int64_t stride_batch,
    int64_t stride_head,
    int64_t stride_sequence)
{
  const int key_block = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int dimension = threadIdx.x;
  const int token_start = key_block * kBlockTokens;
  const int token_count = min(kBlockTokens, key_length - token_start);
  const int8_t *head_key = key_int8 + batch * stride_batch + head * stride_head;
  int quantized_sum = 0;
  for (int token = 0; token < token_count; ++token)
  {
    quantized_sum += static_cast<int>(
        head_key[static_cast<int64_t>(token_start + token) * stride_sequence +
                 dimension]);
  }
  const float scale = key_scale[
      (static_cast<int64_t>(batch) * num_heads + head) * num_key_blocks +
      key_block];
  const int64_t output_index =
      ((static_cast<int64_t>(batch) * num_heads + head) * num_key_blocks +
       key_block) *
          HeadDim +
      dimension;
  key_summary[output_index] = __float2half_rn(
      static_cast<float>(quantized_sum) * scale /
      static_cast<float>(token_count));
}

__global__ void sla_topk_route_kernel(
    const int32_t *__restrict__ topk_indices,
    uint32_t *__restrict__ route_words,
    int64_t index_count,
    int topk,
    int route_word_count,
    int num_key_blocks)
{
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= index_count)
    return;
  const int key_block = topk_indices[index];
  if (key_block < 0 || key_block >= num_key_blocks)
    return;
  const int64_t route_row = index / topk;
  atomicOr(
      route_words + route_row * route_word_count +
          key_block / kRouteWordBits,
      1U << (key_block % kRouteWordBits));
}

__global__ void sla_exact_route_kernel(
    const uint8_t *__restrict__ exact_kv_blocks,
    uint32_t *__restrict__ route_words,
    int64_t route_rows,
    int route_word_count,
    int num_key_blocks)
{
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_words = route_rows * route_word_count;
  if (index >= total_words)
    return;
  const int word_index = index % route_word_count;
  uint32_t exact_word = 0;
#pragma unroll
  for (int bit = 0; bit < kRouteWordBits; ++bit)
  {
    const int key_block = word_index * kRouteWordBits + bit;
    if (key_block < num_key_blocks && exact_kv_blocks[key_block])
      exact_word |= 1U << bit;
  }
  route_words[index] |= exact_word;
}

__device__ __forceinline__ void block_reduce_pair(
    float &first,
    float &second,
    float *__restrict__ scratch)
{
  const int linear_thread = threadIdx.y * blockDim.x + threadIdx.x;
  const int lane = linear_thread % WARP_SIZE;
  const int warp = linear_thread / WARP_SIZE;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
  {
    first += __shfl_down_sync(0xffffffff, first, offset);
    second += __shfl_down_sync(0xffffffff, second, offset);
  }
  if (lane == 0)
  {
    scratch[warp] = first;
    scratch[kWarps + warp] = second;
  }
  __syncthreads();
  if (warp == 0)
  {
    first = lane < kWarps ? scratch[lane] : 0.0f;
    second = lane < kWarps ? scratch[kWarps + lane] : 0.0f;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
      first += __shfl_down_sync(0xffffffff, first, offset);
      second += __shfl_down_sync(0xffffffff, second, offset);
    }
    if (lane == 0)
    {
      scratch[0] = first;
      scratch[1] = second;
    }
  }
  __syncthreads();
  first = scratch[0];
  second = scratch[1];
}

template <int HeadDim, int Rows, typename T>
__device__ __forceinline__ void load_half_tile(
    const T *__restrict__ source,
    int64_t stride_sequence,
    int row_start,
    int row_limit,
    const smem_t<SwizzleMode::k128B, AttentionGeometry<HeadDim>::kHalfPacks> &destination)
{
  using G = AttentionGeometry<HeadDim>;
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  static_assert(Rows > 0 && Rows <= kBlockTokens && Rows % 16 == 0);
  constexpr int tile_packs = Rows * G::kHalfPacks;
  for (int line = linear_thread; line < tile_packs; line += kWarps * WARP_SIZE)
  {
    const int row = line / G::kHalfPacks;
    const int column = line % G::kHalfPacks;
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

template <int HeadDim>
__device__ __forceinline__ void load_int8_tile(
    const int8_t *__restrict__ source,
    int64_t stride_sequence,
    int row_start,
    int row_limit,
    const smem_t<AttentionGeometry<HeadDim>::kInt8Swizzle,
                 AttentionGeometry<HeadDim>::kInt8Packs> &destination)
{
  using G = AttentionGeometry<HeadDim>;
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  for (int line = linear_thread; line < G::kInt8TilePacks;
       line += kWarps * WARP_SIZE)
  {
    const int row = line / G::kInt8Packs;
    const int column = line % G::kInt8Packs;
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

template <int HeadDim>
__device__ __forceinline__ void load_int8_tile_async(
    const int8_t *__restrict__ source,
    int64_t stride_sequence,
    int row_start,
    int row_limit,
    const smem_t<AttentionGeometry<HeadDim>::kInt8Swizzle,
                 AttentionGeometry<HeadDim>::kInt8Packs> &destination)
{
  using G = AttentionGeometry<HeadDim>;
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  for (int line = linear_thread; line < G::kInt8TilePacks;
       line += kWarps * WARP_SIZE)
  {
    const int row = line / G::kInt8Packs;
    const int column = line % G::kInt8Packs;
    const uint32_t offset = destination.get_permuted_offset(row, column);
    const int8_t *source_line = source +
        static_cast<int64_t>(row_start + row) * stride_sequence +
        column * 16;
    destination.template load_128b_async<
        cp_async::SharedMemFillMode::kFillZero>(
        offset, source_line, row_start + row < row_limit);
  }
}

template <int HeadDim>
__device__ __forceinline__ void dequantize_int8_tile(
    const smem_t<AttentionGeometry<HeadDim>::kInt8Swizzle,
                 AttentionGeometry<HeadDim>::kInt8Packs> &source,
    const float *__restrict__ scale,
    int row_start,
    int row_limit,
    const smem_t<SwizzleMode::k128B, AttentionGeometry<HeadDim>::kHalfPacks> &destination)
{
  using G = AttentionGeometry<HeadDim>;
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  for (int line = linear_thread; line < G::kTilePacks;
       line += kWarps * WARP_SIZE)
  {
    const int row = line / G::kHalfPacks;
    const int column = line % G::kHalfPacks;
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
        const int dimension = column * 8 + element;
        const uint32_t source_offset = source.get_permuted_offset(
            row, dimension / 16);
        const int8_t quantized = reinterpret_cast<const int8_t *>(
            source.base + source_offset)[dimension % 16];
        packed_half[element] = __float2half_rn(
            static_cast<float>(quantized) * dequant_scale);
      }
    }
    destination.base[offset] = packed;
  }
}

template <int HeadDim, int KeyTiles>
__device__ __forceinline__ void compute_fp16_qk(
    const smem_t<SwizzleMode::k128B, AttentionGeometry<HeadDim>::kHalfPacks> &query,
    const smem_t<SwizzleMode::k128B, AttentionGeometry<HeadDim>::kHalfPacks> &key,
    float score[1][KeyTiles][8])
{
  using G = AttentionGeometry<HeadDim>;
  static_assert(KeyTiles == 1 || KeyTiles == 2 || KeyTiles == 4);
  uint32_t query_offset = query.get_permuted_offset(
      threadIdx.y * 16 + threadIdx.x % 16, threadIdx.x / 16);
  uint32_t key_offset = key.get_permuted_offset(
      threadIdx.x % 8 + (threadIdx.x / 16) * 8,
      (threadIdx.x / 8) % 2);

#pragma unroll
  for (int inner = 0; inner < HeadDim / 16; ++inner)
  {
    uint32_t query_fragment[4];
    query.ldmatrix_m8n8x4(query_offset, query_fragment);
    query_offset = query.advance_offset_by_row<16>(query_offset);
    query_offset = query.advance_offset_by_column<2>(
        query_offset - 16 * G::kHalfPacks, inner);

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
        key_offset - KeyTiles * 16 * G::kHalfPacks, inner);
  }
}

template <int HeadDim>
__device__ __forceinline__ void compute_int8_qk(
    const smem_t<AttentionGeometry<HeadDim>::kInt8Swizzle,
                 AttentionGeometry<HeadDim>::kInt8Packs> &query,
    const smem_t<AttentionGeometry<HeadDim>::kInt8Swizzle,
                 AttentionGeometry<HeadDim>::kInt8Packs> &key,
    int32_t score[1][4][8])
{
  using G = AttentionGeometry<HeadDim>;
  uint32_t query_offset = query.get_permuted_offset(
      threadIdx.y * 16 + threadIdx.x % 16, threadIdx.x / 16);
  uint32_t key_offset = key.get_permuted_offset(
      threadIdx.x % 8 + (threadIdx.x / 16) * 8,
      (threadIdx.x / 8) % 2);
  compute_int_qk<4, 1, 1, 4, HeadDim / 32,
                 G::kInt8Swizzle, G::kInt8Packs, DataType::kInt8>(
      query, key, score, query_offset, key_offset);
}

template <int SelectedCapacity>
__device__ __forceinline__ int compact_route_words(
    const uint32_t *__restrict__ route_words,
    int *__restrict__ selected_count,
    uint16_t *__restrict__ selected_blocks,
    int active_key_blocks)
{
  // One lane performs an ascending bit scan.  Route construction is tiny
  // compared with exact attention and this deterministic compaction removes
  // the four per-thread route registers that spill on long sm86 kernels.  The
  // ascending order is intentional: changing it changes online-softmax
  // rounding even when the selected set is identical.
  int selected = 0;
  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    const int route_word_count =
        (active_key_blocks + kRouteWordBits - 1) / kRouteWordBits;
    for (int word_index = 0; word_index < route_word_count; ++word_index)
    {
      uint32_t word = route_words[word_index];
      while (word != 0)
      {
        const int bit = __ffs(static_cast<int>(word)) - 1;
        const int key_block = word_index * kRouteWordBits + bit;
        if (key_block < active_key_blocks)
        {
          if (selected < SelectedCapacity)
            selected_blocks[selected] = static_cast<uint16_t>(key_block);
          ++selected;
        }
        word &= word - 1;
      }
    }
    *selected_count = selected <= SelectedCapacity ? selected : -selected;
  }
  __syncthreads();
  return *selected_count;
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
      // global K/V prefetch.  The bitmap remains authoritative and provides a
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

template <int HeadDim>
__device__ __forceinline__ void load_quantized_value_tile(
    const int8_t *__restrict__ value,
    int padded_sequence_length,
    int key_block,
    smem_t<SwizzleMode::k64B, 4> shared_value)
{
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  constexpr int lines = HeadDim * kBlockTokens / 16;
#pragma unroll
  for (int line = linear_thread; line < lines; line += kWarps * WARP_SIZE)
  {
    const int channel = line / 4;
    const int sequence_pack = line % 4;
    const uint32_t destination = shared_value.get_permuted_offset(
        channel, sequence_pack);
    const int8_t *source = value +
        static_cast<int64_t>(channel) * padded_sequence_length +
        key_block * kBlockTokens + sequence_pack * 16;
    shared_value.base[destination] = *reinterpret_cast<const b128_t *>(source);
  }
}

template <int HeadDim>
__device__ __forceinline__ void load_quantized_value_tile_async(
    const int8_t *__restrict__ value,
    int padded_sequence_length,
    int key_block,
    smem_t<SwizzleMode::k64B, 4> shared_value)
{
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  constexpr int lines = HeadDim * kBlockTokens / 16;
#pragma unroll
  for (int line = linear_thread; line < lines; line += kWarps * WARP_SIZE)
  {
    const int channel = line / 4;
    const int sequence_pack = line % 4;
    const uint32_t destination = shared_value.get_permuted_offset(
        channel, sequence_pack);
    const int8_t *source = value +
        static_cast<int64_t>(channel) * padded_sequence_length +
        key_block * kBlockTokens + sequence_pack * 16;
    shared_value.load_128b_async(destination, source);
  }
}

template <int HeadDim, typename T, bool UseW8A8, bool ForceDense,
          bool IsCausal, bool Varlen, int ResidualSubblocks, int KeyStages,
          bool ExternalRoute = false>
__global__ void sparse_attention_kernel(
    const int8_t *__restrict__ query_int8,
    const int8_t *__restrict__ key_int8,
    const T *__restrict__ value,
    const int8_t *__restrict__ value_int8,
    const float *__restrict__ value_scale,
    T *__restrict__ output,
    const float *__restrict__ query_scale,
    const float *__restrict__ key_scale,
    const half *__restrict__ key_score_summary,
    const half *__restrict__ value_mean,
    const float *__restrict__ key_summary_mean,
    const float *__restrict__ key_summary_variance,
    const uint8_t *__restrict__ sparse_query_blocks,
    const uint8_t *__restrict__ exact_kv_blocks,
    const uint32_t *__restrict__ external_route_words,
    unsigned long long *__restrict__ selected_count,
    const int32_t *__restrict__ cu_seqlens_q,
    const int32_t *__restrict__ cu_seqlens_k,
    const int32_t *__restrict__ value_offsets,
    int query_length,
    int key_length,
    int num_query_heads,
    int num_kv_heads,
    int num_query_blocks,
    int num_key_blocks,
    int padded_residual_summaries,
    int64_t stride_batch_q_int8,
    int64_t stride_head_q_int8,
    int64_t stride_sequence_q_int8,
    int64_t stride_batch_k_int8,
    int64_t stride_head_k_int8,
    int64_t stride_sequence_k_int8,
    int64_t stride_batch_v,
    int64_t stride_head_v,
    int64_t stride_sequence_v,
    int padded_value_length,
    int total_value_length,
    int64_t stride_batch_o,
    int64_t stride_head_o,
    int64_t stride_sequence_o,
    float threshold_sigma,
    float softmax_scale,
    int route_original_basis)
{
  using G = AttentionGeometry<HeadDim>;
  static_assert(
      ResidualSubblocks == 1 || ResidualSubblocks == 2,
      "Sol residual geometry must be 1x64 or 2x32");
  static_assert(
      KeyStages == 1 || KeyStages == 2,
      "exact attention stages must cover 64 or 128 K tokens");
  static_assert(!ForceDense || KeyStages == 1);
  static_assert(!ExternalRoute || (!ForceDense && !Varlen && !IsCausal));
  static_assert(!IsCausal || ForceDense,
                "causal masking is supported only by dense W8A8");
  static_assert(
      G::kAttentionSharedBytes <= 64 * 1024,
      "sparse attention exceeds the configured shared-memory limit");
  extern __shared__ int8_t shared_bytes[];
  smem_t<SwizzleMode::k128B, G::kHalfPacks> shared_correction_query(shared_bytes);
  smem_t<SwizzleMode::k128B, G::kHalfPacks> shared_summary_key(
      shared_bytes + G::kTileBytes);
  smem_t<SwizzleMode::k128B, G::kHalfPacks> shared_summary_value(
      shared_bytes + G::kTileBytes + G::kSummaryTileBytes);
  smem_t<SwizzleMode::k128B, G::kHalfPacks> shared_output(shared_bytes);
  smem_t<G::kInt8Swizzle, G::kInt8Packs> shared_query_int8(shared_bytes);
  smem_t<G::kInt8Swizzle, G::kInt8Packs> shared_initial_query_int8(
      shared_bytes + G::kTileBytes);
  smem_t<G::kInt8Swizzle, G::kInt8Packs> shared_key_int8(
      shared_bytes + G::kInt8TileBytes);
  smem_t<SwizzleMode::k128B, G::kHalfPacks> shared_selected_value(
      shared_bytes + 2 * G::kInt8TileBytes);
  smem_t<SwizzleMode::k64B, 4> shared_selected_value_int8(
      shared_bytes + 2 * G::kInt8TileBytes);
  // Dense exact attention no longer needs the routing/selection storage once
  // it enters the K/V loop.  On sm80+ reuse that final INT8-tile region as a
  // second V stage so cp.async can overlap the next V load with the current
  // probability x V MMA.  Sparse queries retain the compact selected-block
  // list in this region; keeping their 32 KiB footprint preserves the third
  // resident CTA on GA10x instead of trading occupancy for a 40 KiB buffer.
  smem_t<SwizzleMode::k64B, 4> shared_selected_value_int8_next(
      shared_bytes + 3 * G::kInt8TileBytes);
  uint32_t *shared_route = reinterpret_cast<uint32_t *>(
      shared_bytes + G::kRouteStorageOffset);
  int *shared_selected_count = reinterpret_cast<int *>(
      shared_bytes + G::kSelectedStorageOffset);
  uint16_t *shared_selected_blocks = reinterpret_cast<uint16_t *>(
      shared_bytes + G::kSelectedStorageOffset + sizeof(int));

  const int query_block = blockIdx.x;
  const int query_head = blockIdx.y;
  const int batch = blockIdx.z;
  int query_start = 0;
  int key_start = 0;
  if constexpr (Varlen)
  {
    query_start = cu_seqlens_q[batch];
    key_start = cu_seqlens_k[batch];
    query_length = cu_seqlens_q[batch + 1] - query_start;
    key_length = cu_seqlens_k[batch + 1] - key_start;
    if (query_block * kBlockTokens >= query_length)
      return;
  }
  const int kv_head = query_head / (num_query_heads / num_kv_heads);
  const int full_key_blocks = Varlen
      ? (key_length + kBlockTokens - 1) / kBlockTokens
      : num_key_blocks;
  const int active_key_blocks = IsCausal
      ? min(full_key_blocks, query_block + 1)
      : full_key_blocks;
  const bool sparse_query = ForceDense
      ? false
      : sparse_query_blocks[query_block] != 0;
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int8_t *query_int8_head_ptr = query_int8 +
      (Varlen ? static_cast<int64_t>(query_start) * stride_sequence_q_int8
              : batch * stride_batch_q_int8) +
      query_head * stride_head_q_int8;
  const int8_t *key_int8_head_ptr = key_int8 +
      (Varlen ? static_cast<int64_t>(key_start) * stride_sequence_k_int8
              : batch * stride_batch_k_int8) +
      kv_head * stride_head_k_int8;
  const T *value_head_ptr =
      value + batch * stride_batch_v + kv_head * stride_head_v;
  const int8_t *value_int8_head_ptr = UseW8A8
      ? value_int8 + (Varlen
          ? static_cast<int64_t>(kv_head) * HeadDim * total_value_length +
              value_offsets[batch]
          : static_cast<int64_t>(batch * num_kv_heads + kv_head) *
              HeadDim * padded_value_length)
      : nullptr;
  const float *value_scale_head = UseW8A8
      ? value_scale +
          static_cast<int64_t>(batch * num_kv_heads + kv_head) * HeadDim
      : nullptr;
  T *output_head_ptr = output +
      (Varlen ? static_cast<int64_t>(query_start) * stride_sequence_o
              : batch * stride_batch_o) +
      query_head * stride_head_o;
  const float *query_scale_head = query_scale +
      (static_cast<int64_t>(batch) * num_query_heads + query_head) *
          num_query_blocks * kWarps;
  const float *key_scale_head = key_scale +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) * num_key_blocks;

  const float q_dequant_scale =
      query_scale_head[query_block * kWarps + threadIdx.y];
  const float scale_log2 = softmax_scale * math::log2e;
  const uint32_t value_mma_offset = shared_summary_value.get_permuted_offset(
      threadIdx.x % 16, threadIdx.x / 16);
  float output_fragment[1][G::kValueTiles][8];
  float row_max[1][2];
  float denominator[1][2];
#pragma unroll
  for (int value_tile = 0; value_tile < G::kValueTiles; ++value_tile)
  {
#pragma unroll
    for (int element = 0; element < 8; ++element)
      output_fragment[0][value_tile][element] = 0.0f;
  }
  row_max[0][0] = -5000000.0f;
  row_max[0][1] = -5000000.0f;
  denominator[0][0] = 1.0f;
  denominator[0][1] = 1.0f;
  if constexpr (!ForceDense)
  {
  if constexpr (ExternalRoute)
  {
    const int route_word_count =
        (active_key_blocks + kRouteWordBits - 1) / kRouteWordBits;
    const int sla_query_blocks =
        (num_query_blocks + 1) / 2;
    const uint32_t *route_head = external_route_words +
        ((static_cast<int64_t>(batch) * num_query_heads + query_head) *
             sla_query_blocks +
         query_block / 2) *
            route_word_count;
    for (int word = linear_thread; word < route_word_count;
         word += kWarps * WARP_SIZE)
      shared_route[word] = route_head[word];
    __syncthreads();
  }
  else
  {
  // Route from the same INT8 Q and per-16-token scales consumed by exact Sage.
  // Keeping this tile in shared memory also avoids another global Q read when
  // constructing the correction operand below.
  load_int8_tile_async<HeadDim>(
      query_int8_head_ptr,
      stride_sequence_q_int8,
      query_block * kBlockTokens,
      query_length,
      shared_initial_query_int8);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  const int query_token_start = query_block * kBlockTokens;
  const int query_token_count = min(kBlockTokens, query_length - query_token_start);
  const int dimension = linear_thread;
  float query_sum = 0.0f;
  if (dimension < HeadDim)
  {
#pragma unroll
    for (int warp_group = 0; warp_group < kWarps; ++warp_group)
    {
      int quantized_sum = 0;
#pragma unroll
      for (int row = 0; row < kBlockTokens / kWarps; ++row)
      {
        const int token = warp_group * (kBlockTokens / kWarps) + row;
        if (token < query_token_count)
        {
          const uint32_t source_offset = shared_initial_query_int8.get_permuted_offset(
              token, dimension / 16);
          quantized_sum += static_cast<int>(reinterpret_cast<const int8_t *>(
              shared_initial_query_int8.base + source_offset)[dimension % 16]);
        }
      }
      const float dequant_scale = query_scale_head[
          query_block * kWarps + warp_group];
      query_sum = fmaf(static_cast<float>(quantized_sum), dequant_scale, query_sum);
    }
  }
  float query_mean = query_sum / static_cast<float>(query_token_count);

  float *reduction_scratch = reinterpret_cast<float *>(
      shared_bytes + G::kRouteStorageOffset);
  if (route_original_basis)
    query_mean = inverse_route_hadamard<HeadDim>(
        query_mean, reduction_scratch);
  const float *key_mean = key_summary_mean +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) * HeadDim;
  const float *key_variance = key_summary_variance +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) * HeadDim;
  float projected_mean = 0.0f;
  float projected_variance = 0.0f;
  if (dimension < HeadDim)
  {
    projected_mean = query_mean * key_mean[dimension];
    projected_variance =
        query_mean * query_mean * key_variance[dimension];
  }
  block_reduce_pair(projected_mean, projected_variance, reduction_scratch);
  const float threshold = projected_mean + threshold_sigma *
      sqrtf(fmaxf(projected_variance, 0.0f) + 1.0e-6f);

  const int route_word_count = (active_key_blocks + kRouteWordBits - 1) / kRouteWordBits;
  if (linear_thread < route_word_count)
    shared_route[linear_thread] = 0;
  __syncthreads();

  const half *key_score_summary_head = key_score_summary +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) *
          padded_residual_summaries * HeadDim;
  const half *value_mean_head = value_mean +
      (static_cast<int64_t>(batch) * num_kv_heads + kv_head) *
          padded_residual_summaries * HeadDim;
  // The initial INT8 Q tile lives in the second half of shared memory. Expand
  // it once into the first 16 KiB, which remains resident while routing and
  // skipped-block correction share the same Tensor Core scores.
  dequantize_int8_tile<HeadDim>(
      shared_initial_query_int8,
      query_scale_head,
      query_block * kBlockTokens,
      query_length,
      shared_correction_query);
  __syncthreads();

  // Route and approximate correction in one pass over 16 summaries. The
  // Tensor Core Q*K-centroid score supplies both per-token correction and the
  // centroid route score, eliminating a separate scalar scan of every K block.
  constexpr int residual_tokens = kBlockTokens / ResidualSubblocks;
  const int num_residual_summaries =
      (key_length + residual_tokens - 1) / residual_tokens;
  float *shared_proxy_partials = reinterpret_cast<float *>(
      shared_bytes + G::kRouteStorageOffset + kMaxRouteBytes);
  for (int summary_start = 0; summary_start < num_residual_summaries;
       summary_start += kSummaryTileTokens)
  {
    load_half_tile<HeadDim, kSummaryTileTokens>(
        key_score_summary_head,
        HeadDim,
        summary_start,
        num_residual_summaries,
        shared_summary_key);
    load_half_tile<HeadDim, kSummaryTileTokens>(
        value_mean_head,
        HeadDim,
        summary_start,
        num_residual_summaries,
        shared_summary_value);
    __syncthreads();

    float score[1][1][8];
    compute_fp16_qk<HeadDim, 1>(shared_correction_query, shared_summary_key, score);

    float proxy0 = score[0][0][0] + score[0][0][2];
    float proxy1 = score[0][0][1] + score[0][0][3];
    float proxy2 = score[0][0][4] + score[0][0][6];
    float proxy3 = score[0][0][5] + score[0][0][7];
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset >= 4; offset >>= 1)
    {
      proxy0 += __shfl_down_sync(0xffffffff, proxy0, offset);
      proxy1 += __shfl_down_sync(0xffffffff, proxy1, offset);
      proxy2 += __shfl_down_sync(0xffffffff, proxy2, offset);
      proxy3 += __shfl_down_sync(0xffffffff, proxy3, offset);
    }
    if (threadIdx.x < 4)
    {
      const int column_base = 2 * threadIdx.x;
      float *warp_proxy =
          shared_proxy_partials + threadIdx.y * kSummaryTileTokens;
      warp_proxy[column_base] = proxy0;
      warp_proxy[column_base + 1] = proxy1;
      warp_proxy[column_base + 8] = proxy2;
      warp_proxy[column_base + 9] = proxy3;
    }
    __syncthreads();

    constexpr int routed_blocks = kSummaryTileTokens / ResidualSubblocks;
    if (linear_thread < routed_blocks)
    {
      const int key_block = summary_start / ResidualSubblocks + linear_thread;
      if (key_block < active_key_blocks)
      {
        float proxy_sum = 0.0f;
        int key_token_count = 0;
#pragma unroll
        for (int residual_index = 0; residual_index < ResidualSubblocks;
             ++residual_index)
        {
          const int residual_summary =
              linear_thread * ResidualSubblocks + residual_index;
          const int residual_start =
              key_block * kBlockTokens + residual_index * residual_tokens;
          const int residual_count =
              max(0, min(residual_tokens, key_length - residual_start));
          float residual_proxy = 0.0f;
#pragma unroll
          for (int warp = 0; warp < kWarps; ++warp)
          {
            residual_proxy += shared_proxy_partials[
                warp * kSummaryTileTokens + residual_summary];
          }
          proxy_sum = fmaf(
              residual_proxy,
              static_cast<float>(residual_count),
              proxy_sum);
          key_token_count += residual_count;
        }
        const float proxy_score = proxy_sum /
            (static_cast<float>(query_token_count) * key_token_count);
        const int distance = query_block > key_block
            ? query_block - key_block
            : key_block - query_block;
        if (!sparse_query || exact_kv_blocks[key_block] ||
            distance <= 1 || proxy_score > threshold)
        {
          atomicOr(
              shared_route + key_block / kRouteWordBits,
              1U << (key_block % kRouteWordBits));
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int element = 0; element < 8; ++element)
    {
      const int local_summary = 2 * (threadIdx.x % 4) +
          8 * (element / 4) + element % 2;
      const int residual_summary = summary_start + local_summary;
      const int key_block = residual_summary / ResidualSubblocks;
      const bool selected = key_block < active_key_blocks &&
          ((shared_route[key_block / kRouteWordBits] >>
            (key_block % kRouteWordBits)) & 1U);
      if (residual_summary >= num_residual_summaries || selected)
      {
        score[0][0][element] = -5000000.0f;
      }
      else
      {
        const int residual_index = residual_summary % ResidualSubblocks;
        const int residual_start =
            key_block * kBlockTokens + residual_index * residual_tokens;
        const int remaining = key_length - residual_start;
        const int block_length = remaining < residual_tokens
            ? remaining
            : residual_tokens;
        score[0][0][element] =
            score[0][0][element] * scale_log2 +
            math::ptx_log2(static_cast<float>(block_length));
      }
    }
    // W8A8 exact PV represents probabilities as U8 with an exp2 offset.
    // Keep skipped-block correction in that same online-softmax domain;
    // otherwise a later exact block compares a shifted maximum against an
    // unshifted one and rescales the correction by roughly 2^8.
    if constexpr (UseW8A8)
    {
      update_mdo<1, 1, G::kValueTiles, false, true, true>(
          score,
          output_fragment,
          row_max,
          denominator,
          1.0f,
          S_U8_OFFSET);
    }
    else
    {
      update_mdo<1, 1, G::kValueTiles, false, false, true>(
          score, output_fragment, row_max, denominator, 1.0f);
    }
    uint32_t probability[1][1][4];
    RS_32_to_16<1, 1>(score, probability);
    if constexpr (UseW8A8)
      accumulate_d<1, 1, ComputeUnit::kCudaCore>(score, denominator);
    else
      accumulate_d<1, 1, ComputeUnit::kTensorCore>(probability, denominator);
    uint32_t value_offset = value_mma_offset;
    compute_fp16_sv_permuted<4, 1, 1, 1, G::kValueTiles,
                             SwizzleMode::k128B, G::kHalfPacks, 4>(
        shared_summary_value,
        probability,
        output_fragment,
        denominator,
        value_offset);
    __syncthreads();
  }

  __syncthreads();
  }
  }

  int compact_selected_count = 0;
  int register_selected_count = 0;
  RouteWords fp16_route{};
  if constexpr (!ForceDense)
  {
    if constexpr (UseW8A8)
    {
      compact_selected_count = compact_route_words<G::kSelectedCapacity>(
          shared_route,
          shared_selected_count,
          shared_selected_blocks,
          active_key_blocks);
      if (selected_count != nullptr && sparse_query &&
          threadIdx.x == 0 && threadIdx.y == 0)
      {
        const unsigned long long count = static_cast<unsigned long long>(
          compact_selected_count >= 0
              ? compact_selected_count
              : -compact_selected_count);
        atomicAdd(selected_count, count);
      }
    }
    else
    {
      const int route_word_count =
          (active_key_blocks + kRouteWordBits - 1) / kRouteWordBits;
      const int route_lane = threadIdx.x;
      fp16_route.word0 = route_lane < route_word_count
          ? shared_route[route_lane] : 0;
      fp16_route.word1 = route_lane + WARP_SIZE < route_word_count
          ? shared_route[route_lane + WARP_SIZE] : 0;
      fp16_route.word2 = route_lane + 2 * WARP_SIZE < route_word_count
          ? shared_route[route_lane + 2 * WARP_SIZE] : 0;
      fp16_route.word3 = route_lane + 3 * WARP_SIZE < route_word_count
          ? shared_route[route_lane + 3 * WARP_SIZE] : 0;
      unsigned int count = __popc(fp16_route.word0) +
          __popc(fp16_route.word1) + __popc(fp16_route.word2) +
          __popc(fp16_route.word3);
#pragma unroll
      for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        count += __shfl_down_sync(0xffffffff, count, offset);
      register_selected_count = static_cast<int>(
          __shfl_sync(0xffffffff, count, 0));
      if (selected_count != nullptr && sparse_query &&
          threadIdx.x == 0 && threadIdx.y == 0)
        atomicAdd(
            selected_count,
            static_cast<unsigned long long>(register_selected_count));
    }
  }

  // Selected blocks retain exact token-level attention. Q/K are quantized once
  // with the production Sage per-16-row Q and per-64-row K scales, then use the
  // same SM75 INT8 Tensor Core MMA as stable Sage. V and output stay FP16/BF16
  // with FP32 online-softmax accumulation.
  load_int8_tile_async<HeadDim>(
      query_int8_head_ptr,
      stride_sequence_q_int8,
      query_block * kBlockTokens,
      query_length,
      shared_query_int8);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();
  int selected_position = 0;
  int key_block = !ForceDense && sparse_query
      ? (UseW8A8
          ? next_compact_route_block<G::kSelectedCapacity>(
              shared_route,
              shared_selected_blocks,
              compact_selected_count,
              selected_position,
              0,
              active_key_blocks)
          : next_register_route_block(fp16_route, 0, active_key_blocks))
      : 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const bool value_ping_pong = UseW8A8 && (ForceDense || !sparse_query);
#else
  constexpr bool value_ping_pong = false;
#endif
  int value_stage = 0;
  if (key_block < active_key_blocks)
  {
    load_int8_tile_async<HeadDim>(
        key_int8_head_ptr,
        stride_sequence_k_int8,
        key_block * kBlockTokens,
        key_length,
        shared_key_int8);
    if constexpr (UseW8A8)
    {
      load_quantized_value_tile_async<HeadDim>(
          value_int8_head_ptr,
          Varlen ? total_value_length : padded_value_length,
          key_block,
          shared_selected_value_int8);
      cp_async::commit_group();
    }
    else
    {
      cp_async::commit_group();
      load_half_tile<HeadDim, kBlockTokens>(
          value_head_ptr,
          stride_sequence_v,
          key_block * kBlockTokens,
          key_length,
          shared_selected_value);
    }
    cp_async::wait_group<0>();
    __syncthreads();
  }
  while (key_block < active_key_blocks)
  {
    // Integer QK accumulators are dead before online softmax consumes the
    // converted values.  Make that lifetime overlap explicit so nvcc does not
    // reserve two independent 32-register score fragments on long D128
    // instantiations.
    union ScoreStorage
    {
      int32_t integer[1][4][8];
      float floating[1][4][8];
    } score_storage;
    compute_int8_qk<HeadDim>(
        shared_query_int8, shared_key_int8, score_storage.integer);
#pragma unroll
    for (int key_tile = 0; key_tile < 4; ++key_tile)
    {
#pragma unroll
      for (int element = 0; element < 8; ++element)
        score_storage.floating[0][key_tile][element] = __int2float_rz(
            score_storage.integer[0][key_tile][element]);
    }
    float (&score)[1][4][8] = score_storage.floating;
    const int next_key_block = !ForceDense && sparse_query
        ? (UseW8A8
            ? next_compact_route_block<G::kSelectedCapacity>(
                shared_route,
                shared_selected_blocks,
                compact_selected_count,
                ++selected_position,
                key_block + 1,
                active_key_blocks)
            : next_register_route_block(
                fp16_route, key_block + 1, active_key_blocks))
        : key_block + 1;
    const bool has_next = next_key_block < active_key_blocks;
    __syncthreads();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (has_next)
    {
      load_int8_tile_async<HeadDim>(
          key_int8_head_ptr,
          stride_sequence_k_int8,
          next_key_block * kBlockTokens,
          key_length,
          shared_key_int8);
      if constexpr (UseW8A8)
      {
        if (value_ping_pong)
        {
          smem_t<SwizzleMode::k64B, 4> next_value(
              value_stage == 0
                  ? shared_selected_value_int8_next.base
                  : shared_selected_value_int8.base);
          load_quantized_value_tile_async<HeadDim>(
              value_int8_head_ptr,
              Varlen ? total_value_length : padded_value_length,
              next_key_block,
              next_value);
        }
      }
      cp_async::commit_group();
    }
#endif
    const uint32_t key_lane_base =
        key_block * kBlockTokens + 2 * (threadIdx.x % 4);
    apply_out_of_bound_mask<1, 4>(key_lane_base, score, key_length);
    if constexpr (IsCausal)
    {
      const uint32_t query_lane_base = query_block * kBlockTokens +
          threadIdx.y * 16 + threadIdx.x / 4;
      apply_causal_mask<1, 4>(query_lane_base, key_lane_base, score);
    }
    if constexpr (UseW8A8)
    {
      update_mdo<1, 4, G::kValueTiles, false, true, false>(
          score,
          output_fragment,
          row_max,
          denominator,
          scale_log2 * q_dequant_scale * key_scale_head[key_block],
          S_U8_OFFSET);
      uint32_t probability_u8[1][2][4];
      RS_to_u8<1, 4>(score, probability_u8);
      accumulate_d<1, 4, ComputeUnit::kCudaCore>(score, denominator);
      float probability_scale[1][2] = {{1.0f, 1.0f}};
      smem_t<SwizzleMode::k64B, 4> current_value(
          value_stage == 0
              ? shared_selected_value_int8.base
              : shared_selected_value_int8_next.base);
      compute_int8_sv_permuted<1, 4, G::kValueTiles, SwizzleMode::k64B, 4>(
          current_value,
          probability_scale,
          probability_u8,
          output_fragment);
    }
    else
    {
      update_mdo<1, 4, G::kValueTiles, false, false, false>(
          score,
          output_fragment,
          row_max,
          denominator,
          scale_log2 * q_dequant_scale * key_scale_head[key_block]);
      uint32_t probability[1][4][4];
      RS_32_to_16<1, 4>(score, probability);
      accumulate_d<1, 4, ComputeUnit::kTensorCore>(probability, denominator);
      uint32_t value_offset = value_mma_offset;
      compute_fp16_sv_permuted<4, 1, 1, 4, G::kValueTiles,
                               SwizzleMode::k128B, G::kHalfPacks, 4>(
          shared_selected_value,
          probability,
          output_fragment,
          denominator,
                               value_offset);
    }
    __syncthreads();
    if (has_next)
    {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
      load_int8_tile_async<HeadDim>(
          key_int8_head_ptr,
          stride_sequence_k_int8,
          next_key_block * kBlockTokens,
          key_length,
          shared_key_int8);
      cp_async::commit_group();
#endif
      if constexpr (UseW8A8)
      {
        if (!value_ping_pong)
        {
          load_quantized_value_tile_async<HeadDim>(
              value_int8_head_ptr,
              Varlen ? total_value_length : padded_value_length,
              next_key_block,
              shared_selected_value_int8);
          cp_async::commit_group();
        }
      }
      else
      {
        load_half_tile<HeadDim, kBlockTokens>(
            value_head_ptr,
            stride_sequence_v,
            next_key_block * kBlockTokens,
            key_length,
            shared_selected_value);
      }
      cp_async::wait_group<0>();
      __syncthreads();
      if (value_ping_pong)
        value_stage ^= 1;
    }
    key_block = next_key_block;
  }

  if constexpr (UseW8A8)
  {
    normalize_d<1, G::kValueTiles, ComputeUnit::kCudaCore>(
        output_fragment, row_max, denominator);
    float channel_scale[4];
    const float *scale_base = value_scale_head + (threadIdx.x % 4) * 2;
#pragma unroll
    for (int value_tile = 0; value_tile < G::kValueTiles; ++value_tile)
    {
      reinterpret_cast<float2 *>(channel_scale)[0] =
          *reinterpret_cast<const float2 *>(scale_base + value_tile * 16);
      reinterpret_cast<float2 *>(channel_scale)[1] =
          *reinterpret_cast<const float2 *>(scale_base + value_tile * 16 + 8);
#pragma unroll
      for (int element = 0; element < 8; ++element)
      {
        output_fragment[0][value_tile][element] *=
            channel_scale[(element / 4) * 2 + (element % 2)];
      }
    }
  }
  else
  {
    normalize_d<1, G::kValueTiles, ComputeUnit::kTensorCore>(
        output_fragment, row_max, denominator);
  }

  const uint32_t output_row_base = threadIdx.y * 16 + threadIdx.x / 4;
#pragma unroll
  for (int value_tile = 0; value_tile < G::kValueTiles; ++value_tile)
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
        shared_output.base + output_offset + 8 * G::kHalfPacks)[threadIdx.x % 4] =
        converted[1];
    reinterpret_cast<uint32_t *>(shared_output.base + (output_offset ^ 0x1))[threadIdx.x % 4] =
        converted[2];
    reinterpret_cast<uint32_t *>(
        shared_output.base + (output_offset ^ 0x1) + 8 * G::kHalfPacks)[threadIdx.x % 4] =
        converted[3];
  }
  __syncthreads();

  constexpr int output_line_lanes = 8;
  constexpr int output_rows_per_warp = 4;
  constexpr int output_column_groups = HeadDim / 64;
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
    for (int column_group = 0; column_group < output_column_groups; ++column_group)
    {
      if (output_row < query_length)
        shared_output.store_128b(output_offset, output_lane);
      output_lane += output_line_lanes * 8;
      output_offset = shared_output.advance_offset_by_column<8>(output_offset);
    }
    output_offset = shared_output.advance_offset_by_row<output_rows_per_warp>(
        output_offset - output_column_groups * output_line_lanes);
    output_lane += output_rows_per_warp * stride_sequence_o -
        output_column_groups * output_line_lanes * 8;
    output_row += output_rows_per_warp;
  }
}

void check_launch(const char *name)
{
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, name, " launch failed: ", cudaGetErrorString(error));
}

template <int HeadDim, typename T, bool UseW8A8, bool ForceDense,
          bool IsCausal, bool Varlen,
          bool SummariesReady = false, int ResidualSubblocks = 1,
          int KeyStages = 1>
void launch_sparse_threshold_attention(
    at::Tensor query_int8,
    at::Tensor key_int8,
    at::Tensor value,
    at::Tensor value_int8,
    at::Tensor value_scale,
    at::Tensor output,
    at::Tensor query_scale,
    at::Tensor key_scale,
    at::Tensor key_summary,
    at::Tensor key_score_summary,
    at::Tensor value_mean,
    at::Tensor key_summary_mean,
    at::Tensor key_summary_variance,
    at::Tensor sparse_query_blocks,
    at::Tensor exact_kv_blocks,
    at::Tensor selected_count,
    at::Tensor cu_seqlens_q,
    at::Tensor cu_seqlens_k,
    at::Tensor value_offsets,
    int max_seqlen_q,
    int max_seqlen_k,
    int residual_subblocks,
    float threshold_sigma,
    float softmax_scale,
    int key_tile_tokens,
    int route_original_basis)
{
  using G = AttentionGeometry<HeadDim>;
  static_assert(
      ResidualSubblocks == 1 || ResidualSubblocks == 2,
      "Sol residual geometry must be 1x64 or 2x32");
  static_assert(KeyStages == 1 || KeyStages == 2);
  static_assert(!ForceDense || KeyStages == 1);
  TORCH_INTERNAL_ASSERT(
      ForceDense || residual_subblocks == ResidualSubblocks,
      "Sol residual dispatch specialization mismatch");
  TORCH_INTERNAL_ASSERT(
      ForceDense || key_tile_tokens / kBlockTokens == KeyStages,
      "Sol K staging dispatch specialization mismatch");
  const int batch_size = Varlen
      ? cu_seqlens_q.size(0) - 1
      : query_int8.size(0);
  const int num_query_heads = query_int8.size(1);
  const int num_kv_heads = key_int8.size(1);
  const int query_length = Varlen ? max_seqlen_q : query_int8.size(2);
  const int key_length = Varlen ? max_seqlen_k : key_int8.size(2);
  const int num_query_blocks = div_ceil(query_length, kBlockTokens);
  const int num_key_blocks = div_ceil(key_length, kBlockTokens);
  const int padded_key_blocks = key_summary.size(2);
  const int padded_residual_summaries = key_score_summary.size(2);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  if constexpr (!ForceDense && !SummariesReady)
  {
    dim3 key_summary_grid(num_key_blocks, num_kv_heads, batch_size);
    kv_block_summary_kernel<HeadDim, T, UseW8A8><<<
        key_summary_grid, HeadDim, 0, stream>>>(
        key_int8.data_ptr<int8_t>(),
        key_scale.data_ptr<float>(),
        reinterpret_cast<const T *>(value.data_ptr()),
        UseW8A8 ? value_scale.data_ptr<float>() : nullptr,
        reinterpret_cast<half *>(key_summary.data_ptr()),
        reinterpret_cast<half *>(key_score_summary.data_ptr()),
        reinterpret_cast<half *>(value_mean.data_ptr()),
        batch_size,
        num_kv_heads,
        key_length,
        padded_key_blocks,
        ResidualSubblocks,
        route_original_basis,
        padded_residual_summaries,
        key_int8.stride(0),
        key_int8.stride(1),
        key_int8.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2));
    check_launch("sparse K/V summary");

    key_summary_stats_kernel<HeadDim><<<
        batch_size * num_kv_heads,
        HeadDim,
        0,
        stream>>>(
        reinterpret_cast<const half *>(key_summary.data_ptr()),
        key_summary_mean.data_ptr<float>(),
        key_summary_variance.data_ptr<float>(),
        num_key_blocks,
        padded_key_blocks);
    check_launch("sparse key summary statistics");
  }

  dim3 attention_grid(num_query_blocks, num_query_heads, batch_size);
  dim3 attention_block(WARP_SIZE, kWarps);
  auto attention_kernel =
      sparse_attention_kernel<HeadDim, T, UseW8A8, ForceDense, IsCausal,
                              Varlen, ResidualSubblocks, KeyStages>;
  configure_dynamic_shared_memory(
      attention_kernel, G::kAttentionSharedBytes, "Sol sparse attention");
  if (attention_kernel_profile_enabled())
  {
    static std::once_flag profile_once;
    std::call_once(profile_once, [=]() {
      std::ostringstream schedule;
      schedule << "head_dim=" << HeadDim
               << ",dtype="
               << (std::is_same<T, half>::value ? "fp16" : "bf16")
               << ",w8a8=" << (UseW8A8 ? 1 : 0)
               << ",dense=" << (ForceDense ? 1 : 0)
               << ",causal=" << (IsCausal ? 1 : 0)
               << ",varlen=" << (Varlen ? 1 : 0)
               << ",residual_subblocks=" << ResidualSubblocks
               << ",key_stages=" << KeyStages
               << ",query_tokens=" << query_length
               << ",key_tokens=" << key_length
               << ",query_blocks=" << num_query_blocks
               << ",heads=" << num_query_heads
               << ",kv_heads=" << num_kv_heads;
      report_cuda_kernel_profile(
          attention_kernel,
          "sol_w8a8_attention",
          schedule.str(),
          WARP_SIZE * kWarps,
          G::kAttentionSharedBytes,
          attention_grid.x,
          attention_grid.y,
          attention_grid.z);
    });
  }
  attention_kernel<<<
      attention_grid,
      attention_block,
      G::kAttentionSharedBytes,
      stream>>>(
      query_int8.data_ptr<int8_t>(),
      key_int8.data_ptr<int8_t>(),
      reinterpret_cast<const T *>(value.data_ptr()),
      UseW8A8 ? value_int8.data_ptr<int8_t>() : nullptr,
      UseW8A8 ? value_scale.data_ptr<float>() : nullptr,
      reinterpret_cast<T *>(output.data_ptr()),
      query_scale.data_ptr<float>(),
      key_scale.data_ptr<float>(),
      reinterpret_cast<const half *>(key_score_summary.data_ptr()),
      reinterpret_cast<const half *>(value_mean.data_ptr()),
      key_summary_mean.data_ptr<float>(),
      key_summary_variance.data_ptr<float>(),
      sparse_query_blocks.data_ptr<uint8_t>(),
      exact_kv_blocks.data_ptr<uint8_t>(),
      nullptr,
      selected_count.numel()
          ? reinterpret_cast<unsigned long long *>(selected_count.data_ptr<int64_t>())
          : nullptr,
      Varlen ? cu_seqlens_q.data_ptr<int32_t>() : nullptr,
      Varlen ? cu_seqlens_k.data_ptr<int32_t>() : nullptr,
      Varlen ? value_offsets.data_ptr<int32_t>() : nullptr,
      query_length,
      key_length,
      num_query_heads,
      num_kv_heads,
      num_query_blocks,
      num_key_blocks,
      padded_residual_summaries,
      query_int8.stride(0),
      query_int8.stride(1),
      Varlen ? query_int8.stride(0) : query_int8.stride(2),
      key_int8.stride(0),
      key_int8.stride(1),
      Varlen ? key_int8.stride(0) : key_int8.stride(2),
      value.stride(0),
      value.stride(1),
      value.stride(2),
      Varlen ? 0 : (UseW8A8 ? value_int8.size(3) : 0),
      Varlen ? value_int8.size(2) : 0,
      output.stride(0),
      output.stride(1),
      Varlen ? output.stride(0) : output.stride(2),
      threshold_sigma,
      softmax_scale,
      route_original_basis);
  check_launch("sparse attention");
}

void dispatch_sparse_threshold_attention(
    at::Tensor query_int8,
    at::Tensor key_int8,
    at::Tensor value,
    at::Tensor value_int8,
    at::Tensor value_scale,
    at::Tensor output,
    at::Tensor query_scale,
    at::Tensor key_scale,
    at::Tensor key_summary,
    at::Tensor key_score_summary,
    at::Tensor value_mean,
    at::Tensor key_summary_mean,
    at::Tensor key_summary_variance,
    at::Tensor sparse_query_blocks,
    at::Tensor exact_kv_blocks,
    at::Tensor selected_count,
    at::Tensor cu_seqlens_q,
    at::Tensor cu_seqlens_k,
    at::Tensor value_offsets,
    int max_seqlen_q,
    int max_seqlen_k,
    int residual_subblocks,
    float threshold_sigma,
    float softmax_scale,
    int key_tile_tokens,
    bool use_w8a8,
    bool force_dense,
    bool summaries_ready,
    bool is_causal,
    bool varlen,
    int route_original_basis)
{
#define LAUNCH_VARIANT(HEAD_DIM, SCALAR, USE_W8A8, FORCE_DENSE, CAUSAL, VARLEN, READY, RESIDUALS, STAGES) \
  launch_sparse_threshold_attention<HEAD_DIM, SCALAR, USE_W8A8,             \
                                    FORCE_DENSE, CAUSAL, VARLEN, READY,      \
                                    RESIDUALS, STAGES>(                      \
      query_int8, key_int8, value, value_int8, value_scale, output,          \
      query_scale, key_scale, key_summary, key_score_summary, value_mean,    \
      key_summary_mean, key_summary_variance, sparse_query_blocks,           \
      exact_kv_blocks, selected_count, cu_seqlens_q, cu_seqlens_k,          \
      value_offsets,                                                        \
      max_seqlen_q, max_seqlen_k, residual_subblocks, threshold_sigma,       \
      softmax_scale, key_tile_tokens, route_original_basis)
#define DISPATCH_FORMAT(HEAD_DIM, SCALAR)                                    \
  do                                                                         \
  {                                                                          \
    if (varlen && is_causal)                                                 \
      LAUNCH_VARIANT(HEAD_DIM, SCALAR, true, true, true, true, true, 1, 1);  \
    else if (varlen)                                                         \
      LAUNCH_VARIANT(HEAD_DIM, SCALAR, true, true, false, true, true, 1, 1); \
    else if (force_dense && is_causal)                                       \
      LAUNCH_VARIANT(HEAD_DIM, SCALAR, true, true, true, false, true, 1, 1); \
    else if (force_dense)                                                    \
      LAUNCH_VARIANT(HEAD_DIM, SCALAR, true, true, false, false, true, 1, 1);\
    else if (residual_subblocks == 2)                                        \
    {                                                                        \
      if (use_w8a8 && summaries_ready)                                       \
      {                                                                      \
        if (key_tile_tokens == 128)                                          \
          LAUNCH_VARIANT(HEAD_DIM, SCALAR, true, false, false, false, true, 2, 2); \
        else                                                                 \
          LAUNCH_VARIANT(HEAD_DIM, SCALAR, true, false, false, false, true, 2, 1); \
      }                                                                      \
      else if (use_w8a8)                                                     \
      {                                                                      \
        if (key_tile_tokens == 128)                                          \
          LAUNCH_VARIANT(HEAD_DIM, SCALAR, true, false, false, false, false, 2, 2); \
        else                                                                 \
          LAUNCH_VARIANT(HEAD_DIM, SCALAR, true, false, false, false, false, 2, 1); \
      }                                                                      \
      else if (key_tile_tokens == 128)                                       \
        LAUNCH_VARIANT(HEAD_DIM, SCALAR, false, false, false, false, false, 2, 2); \
      else                                                                   \
        LAUNCH_VARIANT(HEAD_DIM, SCALAR, false, false, false, false, false, 2, 1); \
    }                                                                        \
    else if (use_w8a8 && summaries_ready)                                    \
    {                                                                        \
      if (key_tile_tokens == 128)                                            \
        LAUNCH_VARIANT(HEAD_DIM, SCALAR, true, false, false, false, true, 1, 2); \
      else                                                                   \
        LAUNCH_VARIANT(HEAD_DIM, SCALAR, true, false, false, false, true, 1, 1); \
    }                                                                        \
    else if (use_w8a8)                                                       \
    {                                                                        \
      if (key_tile_tokens == 128)                                            \
        LAUNCH_VARIANT(HEAD_DIM, SCALAR, true, false, false, false, false, 1, 2); \
      else                                                                   \
        LAUNCH_VARIANT(HEAD_DIM, SCALAR, true, false, false, false, false, 1, 1); \
    }                                                                        \
    else if (key_tile_tokens == 128)                                         \
      LAUNCH_VARIANT(HEAD_DIM, SCALAR, false, false, false, false, false, 1, 2); \
    else                                                                     \
      LAUNCH_VARIANT(HEAD_DIM, SCALAR, false, false, false, false, false, 1, 1); \
  } while (false)

  const int head_dim = query_int8.size(-1);
  if (output.scalar_type() == at::ScalarType::Half)
  {
    if (head_dim == 64)
      DISPATCH_FORMAT(64, half);
    else
      DISPATCH_FORMAT(128, half);
  }
  else
  {
    if (head_dim == 64)
      DISPATCH_FORMAT(64, nv_bfloat16);
    else
      DISPATCH_FORMAT(128, nv_bfloat16);
  }
#undef DISPATCH_FORMAT
#undef LAUNCH_VARIANT
}

template <int HeadDim, typename T, bool UseW8A8, int KeyStages>
void launch_sla_attention(
    at::Tensor query_int8,
    at::Tensor key_int8,
    at::Tensor value,
    at::Tensor value_int8,
    at::Tensor value_scale,
    at::Tensor output,
    at::Tensor query_scale,
    at::Tensor key_scale,
    at::Tensor route_words,
    at::Tensor sparse_query_blocks,
    at::Tensor selected_count,
    float softmax_scale)
{
  using G = AttentionGeometry<HeadDim>;
  const int batch_size = query_int8.size(0);
  const int num_query_heads = query_int8.size(1);
  const int num_kv_heads = key_int8.size(1);
  const int query_length = query_int8.size(2);
  const int key_length = key_int8.size(2);
  const int num_query_blocks = div_ceil(query_length, kBlockTokens);
  const int num_key_blocks = div_ceil(key_length, kBlockTokens);
  dim3 attention_grid(num_query_blocks, num_query_heads, batch_size);
  dim3 attention_block(WARP_SIZE, kWarps);
  auto attention_kernel =
      sparse_attention_kernel<HeadDim, T, UseW8A8, false, false, false,
                              1, KeyStages, true>;
  configure_dynamic_shared_memory(
      attention_kernel, G::kAttentionSharedBytes, "SLA sparse attention");
  attention_kernel<<<
      attention_grid,
      attention_block,
      G::kAttentionSharedBytes,
      c10::cuda::getCurrentCUDAStream()>>>(
      query_int8.data_ptr<int8_t>(),
      key_int8.data_ptr<int8_t>(),
      reinterpret_cast<const T *>(value.data_ptr()),
      UseW8A8 ? value_int8.data_ptr<int8_t>() : nullptr,
      UseW8A8 ? value_scale.data_ptr<float>() : nullptr,
      reinterpret_cast<T *>(output.data_ptr()),
      query_scale.data_ptr<float>(),
      key_scale.data_ptr<float>(),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      sparse_query_blocks.data_ptr<uint8_t>(),
      nullptr,
      reinterpret_cast<const uint32_t *>(route_words.data_ptr<int32_t>()),
      selected_count.numel()
          ? reinterpret_cast<unsigned long long *>(selected_count.data_ptr<int64_t>())
          : nullptr,
      nullptr,
      nullptr,
      nullptr,
      query_length,
      key_length,
      num_query_heads,
      num_kv_heads,
      num_query_blocks,
      num_key_blocks,
      0,
      query_int8.stride(0),
      query_int8.stride(1),
      query_int8.stride(2),
      key_int8.stride(0),
      key_int8.stride(1),
      key_int8.stride(2),
      value.stride(0),
      value.stride(1),
      value.stride(2),
      UseW8A8 ? value_int8.size(3) : 0,
      0,
      output.stride(0),
      output.stride(1),
      output.stride(2),
      0.0f,
      softmax_scale,
      0);
  check_launch("SLA sparse attention");
}

void dispatch_sla_attention(
    at::Tensor query_int8,
    at::Tensor key_int8,
    at::Tensor value,
    at::Tensor value_int8,
    at::Tensor value_scale,
    at::Tensor output,
    at::Tensor query_scale,
    at::Tensor key_scale,
    at::Tensor route_words,
    at::Tensor sparse_query_blocks,
    at::Tensor selected_count,
    float softmax_scale,
    bool use_w8a8,
    int key_tile_tokens)
{
#define LAUNCH_SLA(HEAD_DIM, SCALAR, W8A8, STAGES)                           \
  launch_sla_attention<HEAD_DIM, SCALAR, W8A8, STAGES>(                     \
      query_int8, key_int8, value, value_int8, value_scale, output,          \
      query_scale, key_scale, route_words, sparse_query_blocks,              \
      selected_count, softmax_scale)
#define DISPATCH_SLA(HEAD_DIM, SCALAR)                                      \
  do                                                                         \
  {                                                                          \
    if (use_w8a8 && key_tile_tokens == 128)                                  \
      LAUNCH_SLA(HEAD_DIM, SCALAR, true, 2);                                 \
    else if (use_w8a8)                                                       \
      LAUNCH_SLA(HEAD_DIM, SCALAR, true, 1);                                 \
    else if (key_tile_tokens == 128)                                         \
      LAUNCH_SLA(HEAD_DIM, SCALAR, false, 2);                                \
    else                                                                     \
      LAUNCH_SLA(HEAD_DIM, SCALAR, false, 1);                                \
  } while (false)
  const int head_dim = query_int8.size(-1);
  if (output.scalar_type() == at::ScalarType::Half)
  {
    if (head_dim == 64)
      DISPATCH_SLA(64, half);
    else
      DISPATCH_SLA(128, half);
  }
  else
  {
    if (head_dim == 64)
      DISPATCH_SLA(64, nv_bfloat16);
    else
      DISPATCH_SLA(128, nv_bfloat16);
  }
#undef DISPATCH_SLA
#undef LAUNCH_SLA
}

constexpr int kVarlenValueChannelTile = 8;

__device__ __forceinline__ int inverse_permute_value_16(int value)
{
  return (value & 1) | (((value >> 3) & 1) << 1) |
      (((value >> 1) & 1) << 2) | (((value >> 2) & 1) << 3);
}

template <typename T>
__device__ __forceinline__ float varlen_value_to_float(T value)
{
  if constexpr (std::is_same<T, half>::value)
    return __half2float(value);
  else
    return __bfloat162float(value);
}

__device__ __forceinline__ float varlen_warp_max(float value)
{
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  return value;
}

__device__ __forceinline__ int8_t quantize_varlen_s8(float value)
{
  int converted;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(converted) : "f"(value));
  return static_cast<int8_t>(converted);
}

// Prefix offsets describe an internal channel-major V layout.  Each sequence
// is independently padded to a 64-token boundary so every attention tile is
// 16-byte aligned, while storage overhead stays below 63 tokens per sequence.
__global__ void build_varlen_value_offsets_kernel(
    const int32_t *__restrict__ cu_seqlens,
    int32_t *__restrict__ value_offsets,
    int batch_size)
{
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  int offset = 0;
  value_offsets[0] = 0;
  for (int batch = 0; batch < batch_size; ++batch)
  {
    const int length = cu_seqlens[batch + 1] - cu_seqlens[batch];
    offset += div_ceil(length, kBlockTokens) * kBlockTokens;
    value_offsets[batch + 1] = offset;
  }
}

template <typename T, int Threads>
__global__ void quantize_varlen_value_kernel(
    const T *__restrict__ value,
    const int32_t *__restrict__ cu_seqlens,
    const int32_t *__restrict__ value_offsets,
    int8_t *__restrict__ quantized,
    float *__restrict__ scale,
    int storage_tokens,
    int heads,
    int head_dim,
    int64_t stride_token,
    int64_t stride_head)
{
  constexpr int Warps = Threads / WARP_SIZE;
  const int channel_tiles = head_dim / kVarlenValueChannelTile;
  const int channel_tile = blockIdx.x % channel_tiles;
  const int batch_head = blockIdx.x / channel_tiles;
  const int head = batch_head % heads;
  const int batch = batch_head / heads;
  const int channel_start = channel_tile * kVarlenValueChannelTile;
  const int sequence_start = cu_seqlens[batch];
  const int sequence_length = cu_seqlens[batch + 1] - sequence_start;
  const int padded_start = value_offsets[batch];
  const int padded_length = value_offsets[batch + 1] - padded_start;
  const T *base = value +
      static_cast<int64_t>(sequence_start) * stride_token +
      static_cast<int64_t>(head) * stride_head + channel_start;

  float maximum[kVarlenValueChannelTile];
#pragma unroll
  for (int channel = 0; channel < kVarlenValueChannelTile; ++channel)
    maximum[channel] = 0.0f;

  int token = threadIdx.x;
  const int body = sequence_length - Threads;
  for (; token < body; token += 2 * Threads)
  {
#pragma unroll
    for (int channel = 0; channel < kVarlenValueChannelTile; ++channel)
    {
      const float first = fabsf(varlen_value_to_float(
          base[static_cast<int64_t>(token) * stride_token + channel]));
      const float second = fabsf(varlen_value_to_float(
          base[static_cast<int64_t>(token + Threads) * stride_token + channel]));
      maximum[channel] = fmaxf(maximum[channel], fmaxf(first, second));
    }
  }
  for (; token < sequence_length; token += Threads)
  {
#pragma unroll
    for (int channel = 0; channel < kVarlenValueChannelTile; ++channel)
      maximum[channel] = fmaxf(
          maximum[channel],
          fabsf(varlen_value_to_float(
              base[static_cast<int64_t>(token) * stride_token + channel])));
  }

#pragma unroll
  for (int channel = 0; channel < kVarlenValueChannelTile; ++channel)
    maximum[channel] = varlen_warp_max(maximum[channel]);

  __shared__ float warp_maximum[kVarlenValueChannelTile][Warps];
  __shared__ float inverse_scale[kVarlenValueChannelTile];
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;
  if (lane == 0)
  {
#pragma unroll
    for (int channel = 0; channel < kVarlenValueChannelTile; ++channel)
      warp_maximum[channel][warp] = maximum[channel];
  }
  __syncthreads();

  if (threadIdx.x < kVarlenValueChannelTile)
  {
    float channel_maximum = 0.0f;
#pragma unroll
    for (int source_warp = 0; source_warp < Warps; ++source_warp)
      channel_maximum = fmaxf(
          channel_maximum, warp_maximum[threadIdx.x][source_warp]);
    const float channel_scale = fmaxf(channel_maximum / 127.0f, 1.0e-12f);
    scale[static_cast<int64_t>(batch_head) * head_dim +
          channel_start + threadIdx.x] = channel_scale;
    inverse_scale[threadIdx.x] = 1.0f / channel_scale;
  }
  __syncthreads();

  const int64_t output_base =
      static_cast<int64_t>(head * head_dim + channel_start) * storage_tokens +
      padded_start;
  for (int source = sequence_length - 1 - threadIdx.x;
       source >= 0;
       source -= Threads)
  {
    const int destination =
        (source & ~15) | inverse_permute_value_16(source & 15);
#pragma unroll
    for (int channel = 0; channel < kVarlenValueChannelTile; ++channel)
      quantized[output_base +
                static_cast<int64_t>(channel) * storage_tokens + destination] =
          quantize_varlen_s8(varlen_value_to_float(
              base[static_cast<int64_t>(source) * stride_token + channel]) *
              inverse_scale[channel]);
  }
  for (int source = sequence_length + threadIdx.x;
       source < padded_length;
       source += Threads)
  {
    const int destination =
        (source & ~15) | inverse_permute_value_16(source & 15);
#pragma unroll
    for (int channel = 0; channel < kVarlenValueChannelTile; ++channel)
      quantized[output_base +
                static_cast<int64_t>(channel) * storage_tokens + destination] = 0;
  }
}

} // namespace

std::vector<at::Tensor> sla_qk_block_summaries(
    at::Tensor query_int8,
    at::Tensor key_int8,
    at::Tensor query_scale,
    at::Tensor key_scale)
{
  CHECK_CUDA(query_int8);
  CHECK_CUDA(key_int8);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_LASTDIM_CONTIGUOUS(query_int8);
  CHECK_LASTDIM_CONTIGUOUS(key_int8);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_DIMS(query_int8, 4);
  CHECK_DIMS(key_int8, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  CHECK_DTYPE(query_int8, at::ScalarType::Char);
  CHECK_DTYPE(key_int8, at::ScalarType::Char);
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);
  TORCH_CHECK(
      query_int8.device() == key_int8.device() &&
          query_int8.device() == query_scale.device() &&
          query_int8.device() == key_scale.device(),
      "SLA Q/K summary tensors must share one CUDA device");
  TORCH_CHECK(
      query_int8.size(0) == key_int8.size(0) &&
          query_int8.size(3) == key_int8.size(3),
      "SLA Q/K summary shapes are incompatible");
  const int batch_size = query_int8.size(0);
  const int query_heads = query_int8.size(1);
  const int key_heads = key_int8.size(1);
  const int query_length = query_int8.size(2);
  const int key_length = key_int8.size(2);
  const int head_dim = query_int8.size(3);
  TORCH_CHECK(
      head_dim == 64 || head_dim == 128,
      "SLA summaries require head_dim 64 or 128");
  TORCH_CHECK(
      key_heads > 0 && query_heads % key_heads == 0,
      "SLA Query heads must be divisible by KV heads");
  const int query_blocks_64 = div_ceil(query_length, kBlockTokens);
  const int query_blocks_128 = div_ceil(query_length, kSlaQueryBlockTokens);
  const int key_blocks = div_ceil(key_length, kBlockTokens);
  TORCH_CHECK(
      query_scale.sizes() == at::IntArrayRef(
          {batch_size, query_heads, query_blocks_64 * kWarps}),
      "SLA Query scale shape is incompatible");
  TORCH_CHECK(
      key_scale.sizes() ==
          at::IntArrayRef({batch_size, key_heads, key_blocks}),
      "SLA Key scale shape is incompatible");
  const auto half_options = query_int8.options().dtype(at::ScalarType::Half);
  at::Tensor query_summary = at::empty(
      {batch_size, query_heads, query_blocks_128, head_dim}, half_options);
  at::Tensor key_summary = at::empty(
      {batch_size, key_heads, key_blocks, head_dim}, half_options);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
#define LAUNCH_SLA_SUMMARIES(HEAD_DIM)                                       \
  sla_query_summary_kernel<HEAD_DIM><<<                                     \
      dim3(query_blocks_128, query_heads, batch_size),                       \
      HEAD_DIM, 0, stream>>>(                                                \
      query_int8.data_ptr<int8_t>(), query_scale.data_ptr<float>(),          \
      reinterpret_cast<half *>(query_summary.data_ptr()),                    \
      query_length, query_heads, query_blocks_64,                            \
      query_int8.stride(0), query_int8.stride(1), query_int8.stride(2));     \
  sla_key_summary_kernel<HEAD_DIM><<<                                       \
      dim3(key_blocks, key_heads, batch_size),                               \
      HEAD_DIM, 0, stream>>>(                                                \
      key_int8.data_ptr<int8_t>(), key_scale.data_ptr<float>(),              \
      reinterpret_cast<half *>(key_summary.data_ptr()),                      \
      key_length, key_heads, key_blocks,                                     \
      key_int8.stride(0), key_int8.stride(1), key_int8.stride(2))
  if (head_dim == 64)
  {
    LAUNCH_SLA_SUMMARIES(64);
  }
  else
  {
    LAUNCH_SLA_SUMMARIES(128);
  }
#undef LAUNCH_SLA_SUMMARIES
  check_launch("SLA Q/K block summaries");
  return {query_summary, key_summary};
}

at::Tensor sla_build_route_words(
    at::Tensor topk_indices,
    at::Tensor exact_kv_blocks,
    int num_key_blocks)
{
  CHECK_CUDA(topk_indices);
  CHECK_CUDA(exact_kv_blocks);
  CHECK_CONTIGUOUS(topk_indices);
  CHECK_CONTIGUOUS(exact_kv_blocks);
  CHECK_DIMS(topk_indices, 4);
  CHECK_DIMS(exact_kv_blocks, 1);
  CHECK_DTYPE(topk_indices, at::ScalarType::Int);
  CHECK_DTYPE(exact_kv_blocks, at::ScalarType::Byte);
  TORCH_CHECK(
      topk_indices.device() == exact_kv_blocks.device(),
      "SLA route tensors must share one CUDA device");
  TORCH_CHECK(num_key_blocks > 0 && num_key_blocks <= kMaxKeyBlocks,
              "SLA route K block count is outside the supported range");
  TORCH_CHECK(exact_kv_blocks.numel() == num_key_blocks,
              "SLA exact-KV policy size is incompatible");
  const int topk = topk_indices.size(3);
  TORCH_CHECK(topk > 0 && topk <= num_key_blocks,
              "SLA Top-K must be in [1, num_key_blocks]");
  const int route_word_count = div_ceil(num_key_blocks, kRouteWordBits);
  at::Tensor route_words = at::zeros(
      {topk_indices.size(0), topk_indices.size(1), topk_indices.size(2),
       route_word_count},
      topk_indices.options());
  const int Threads = 256;
  const int64_t index_count = topk_indices.numel();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  sla_topk_route_kernel<<<div_ceil(index_count, static_cast<int64_t>(Threads)),
                          Threads, 0, stream>>>(
      topk_indices.data_ptr<int32_t>(),
      reinterpret_cast<uint32_t *>(route_words.data_ptr<int32_t>()),
      index_count,
      topk,
      route_word_count,
      num_key_blocks);
  check_launch("SLA Top-K route packing");
  const int64_t route_rows = route_words.numel() / route_word_count;
  const int64_t total_words = route_words.numel();
  sla_exact_route_kernel<<<div_ceil(total_words, static_cast<int64_t>(Threads)),
                           Threads, 0, stream>>>(
      exact_kv_blocks.data_ptr<uint8_t>(),
      reinterpret_cast<uint32_t *>(route_words.data_ptr<int32_t>()),
      route_rows,
      route_word_count,
      num_key_blocks);
  check_launch("SLA exact-KV route union");
  return route_words;
}

at::Tensor sla_sparse_online_attn(
    at::Tensor query_int8,
    at::Tensor key_int8,
    at::Tensor value,
    at::Tensor value_int8,
    at::Tensor value_scale,
    at::Tensor output,
    at::Tensor query_scale,
    at::Tensor key_scale,
    at::Tensor route_words,
    at::Tensor sparse_query_blocks,
    float softmax_scale,
    int return_stats,
    int use_w8a8,
    int key_tile_tokens)
{
  CHECK_CUDA(query_int8);
  CHECK_CUDA(key_int8);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(route_words);
  CHECK_CUDA(sparse_query_blocks);
  CHECK_LASTDIM_CONTIGUOUS(query_int8);
  CHECK_LASTDIM_CONTIGUOUS(key_int8);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(route_words);
  CHECK_CONTIGUOUS(sparse_query_blocks);
  CHECK_DTYPE(query_int8, at::ScalarType::Char);
  CHECK_DTYPE(key_int8, at::ScalarType::Char);
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);
  CHECK_DTYPE(route_words, at::ScalarType::Int);
  CHECK_DTYPE(sparse_query_blocks, at::ScalarType::Byte);
  TORCH_CHECK(
      output.scalar_type() == at::ScalarType::Half ||
          output.scalar_type() == at::ScalarType::BFloat16,
      "SLA output must be float16 or bfloat16");
  TORCH_CHECK(value.scalar_type() == output.scalar_type(),
              "SLA logical V and output dtypes must match");
  TORCH_CHECK(use_w8a8 == 0 || use_w8a8 == 1,
              "SLA use_w8a8 must be 0 or 1");
  TORCH_CHECK(key_tile_tokens == 64 || key_tile_tokens == 128,
              "SLA key_tile_tokens must be 64 or 128");
  const int batch_size = query_int8.size(0);
  const int query_heads = query_int8.size(1);
  const int key_heads = key_int8.size(1);
  const int query_length = query_int8.size(2);
  const int key_length = key_int8.size(2);
  const int head_dim = query_int8.size(3);
  const int query_blocks_64 = div_ceil(query_length, kBlockTokens);
  const int query_blocks_128 = div_ceil(query_length, kSlaQueryBlockTokens);
  const int key_blocks = div_ceil(key_length, kBlockTokens);
  const int route_word_count = div_ceil(key_blocks, kRouteWordBits);
  TORCH_CHECK(
      query_int8.dim() == 4 && key_int8.dim() == 4 && value.dim() == 4 &&
          output.dim() == 4,
      "SLA Q/K/V/O must be four-dimensional");
  TORCH_CHECK(
      (head_dim == 64 || head_dim == 128) && key_int8.size(3) == head_dim &&
          value.size(3) == head_dim,
      "SLA requires matching head_dim 64 or 128");
  TORCH_CHECK(
      key_heads > 0 && query_heads % key_heads == 0,
      "SLA Query heads must be divisible by KV heads");
  TORCH_CHECK(output.sizes() == query_int8.sizes(),
              "SLA output shape is incompatible");
  TORCH_CHECK(
      use_w8a8 ||
          (value.size(0) == batch_size && value.size(1) == key_heads &&
           value.size(2) == key_length),
      "SLA FP16/BF16 V shape is incompatible");
  TORCH_CHECK(
      query_scale.sizes() == at::IntArrayRef(
          {batch_size, query_heads, query_blocks_64 * kWarps}) &&
          key_scale.sizes() ==
              at::IntArrayRef({batch_size, key_heads, key_blocks}),
      "SLA Q/K scale shapes are incompatible");
  TORCH_CHECK(
      route_words.sizes() == at::IntArrayRef(
          {batch_size, query_heads, query_blocks_128, route_word_count}) &&
          sparse_query_blocks.numel() == query_blocks_64,
      "SLA route policy shapes are incompatible");
  if (use_w8a8)
  {
    CHECK_CUDA(value_int8);
    CHECK_CUDA(value_scale);
    CHECK_CONTIGUOUS(value_int8);
    CHECK_CONTIGUOUS(value_scale);
    CHECK_DTYPE(value_int8, at::ScalarType::Char);
    CHECK_DTYPE(value_scale, at::ScalarType::Float);
    TORCH_CHECK(
        value_int8.sizes() == at::IntArrayRef(
            {batch_size, key_heads, head_dim,
             div_ceil(key_length, kBlockTokens) * kBlockTokens}) &&
            value_scale.sizes() ==
                at::IntArrayRef({batch_size, key_heads, head_dim}),
        "SLA W8A8 V shapes are incompatible");
  }
  at::Tensor selected_count = return_stats
      ? at::zeros({1}, output.options().dtype(at::ScalarType::Long))
      : at::empty({0}, output.options().dtype(at::ScalarType::Long));
  dispatch_sla_attention(
      query_int8, key_int8, value, value_int8, value_scale, output,
      query_scale, key_scale, route_words, sparse_query_blocks,
      selected_count, softmax_scale, use_w8a8 != 0, key_tile_tokens);
  return selected_count;
}

void quantize_v_int8_varlen_sm75(
    at::Tensor value,
    at::Tensor cu_seqlens_k,
    at::Tensor value_offsets,
    at::Tensor quantized,
    at::Tensor scale)
{
  CHECK_CUDA(value);
  CHECK_CUDA(cu_seqlens_k);
  CHECK_CUDA(value_offsets);
  CHECK_CUDA(quantized);
  CHECK_CUDA(scale);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_CONTIGUOUS(cu_seqlens_k);
  CHECK_CONTIGUOUS(value_offsets);
  CHECK_CONTIGUOUS(quantized);
  CHECK_CONTIGUOUS(scale);
  CHECK_DIMS(value, 3);
  CHECK_DIMS(cu_seqlens_k, 1);
  CHECK_DIMS(value_offsets, 1);
  CHECK_DIMS(quantized, 3);
  CHECK_DIMS(scale, 3);
  CHECK_DTYPE(cu_seqlens_k, at::ScalarType::Int);
  CHECK_DTYPE(value_offsets, at::ScalarType::Int);
  CHECK_DTYPE(quantized, at::ScalarType::Char);
  CHECK_DTYPE(scale, at::ScalarType::Float);
  TORCH_CHECK(
      value.scalar_type() == at::ScalarType::Half ||
          value.scalar_type() == at::ScalarType::BFloat16,
      "varlen W8A8 V must be float16 or bfloat16");
  TORCH_CHECK(cu_seqlens_k.size(0) >= 2,
              "varlen W8A8 V requires at least one sequence");
  const int batch_size = cu_seqlens_k.size(0) - 1;
  const int total_tokens = value.size(0);
  const int heads = value.size(1);
  const int head_dim = value.size(2);
  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "varlen W8A8 V requires head_dim 64 or 128");
  TORCH_CHECK(value_offsets.size(0) == batch_size + 1,
              "varlen W8A8 V offsets must be [B+1]");
  const int storage_tokens = div_ceil(
      total_tokens + batch_size * (kBlockTokens - 1), kBlockTokens) *
      kBlockTokens;
  TORCH_CHECK(
      quantized.sizes() == at::IntArrayRef({heads, head_dim, storage_tokens}),
      "varlen W8A8 quantized V must provide total_K + 63*B token slots");
  TORCH_CHECK(
      scale.sizes() == at::IntArrayRef({batch_size, heads, head_dim}),
      "varlen W8A8 V scale must be [B,Hkv,D]");
  TORCH_CHECK(
      value.device() == cu_seqlens_k.device() &&
          value.device() == value_offsets.device() &&
          value.device() == quantized.device() &&
          value.device() == scale.device(),
      "varlen W8A8 V tensors must share one CUDA device");
  constexpr int Threads = 256;
  const int blocks = batch_size * heads *
      (head_dim / kVarlenValueChannelTile);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  build_varlen_value_offsets_kernel<<<1, 1, 0, stream>>>(
      cu_seqlens_k.data_ptr<int32_t>(),
      value_offsets.data_ptr<int32_t>(),
      batch_size);
  check_launch("varlen W8A8 V offset construction");
  if (value.scalar_type() == at::ScalarType::Half)
    quantize_varlen_value_kernel<half, Threads><<<blocks, Threads, 0, stream>>>(
        reinterpret_cast<const half *>(value.data_ptr()),
        cu_seqlens_k.data_ptr<int32_t>(),
        value_offsets.data_ptr<int32_t>(),
        quantized.data_ptr<int8_t>(), scale.data_ptr<float>(),
        storage_tokens, heads, head_dim, value.stride(0), value.stride(1));
  else
    quantize_varlen_value_kernel<nv_bfloat16, Threads><<<blocks, Threads, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16 *>(value.data_ptr()),
        cu_seqlens_k.data_ptr<int32_t>(),
        value_offsets.data_ptr<int32_t>(),
        quantized.data_ptr<int8_t>(), scale.data_ptr<float>(),
        storage_tokens, heads, head_dim, value.stride(0), value.stride(1));
  check_launch("varlen W8A8 V quantization");
}

void qk_int8_sv_int8_varlen_accum_f32_attn(
    at::Tensor query_int8,
    at::Tensor key_int8,
    at::Tensor value_int8,
    at::Tensor value_scale,
    at::Tensor output,
    at::Tensor query_scale,
    at::Tensor key_scale,
    at::Tensor cu_seqlens_q,
    at::Tensor cu_seqlens_k,
    at::Tensor value_offsets,
    int max_seqlen_q,
    int max_seqlen_k,
    int is_causal,
    float softmax_scale)
{
  CHECK_CUDA(query_int8);
  CHECK_CUDA(key_int8);
  CHECK_CUDA(value_int8);
  CHECK_CUDA(value_scale);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(cu_seqlens_q);
  CHECK_CUDA(cu_seqlens_k);
  CHECK_CUDA(value_offsets);
  CHECK_CONTIGUOUS(query_int8);
  CHECK_CONTIGUOUS(key_int8);
  CHECK_CONTIGUOUS(value_int8);
  CHECK_CONTIGUOUS(value_scale);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);
  CHECK_CONTIGUOUS(value_offsets);
  CHECK_DTYPE(query_int8, at::ScalarType::Char);
  CHECK_DTYPE(key_int8, at::ScalarType::Char);
  CHECK_DTYPE(value_int8, at::ScalarType::Char);
  CHECK_DTYPE(value_scale, at::ScalarType::Float);
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);
  CHECK_DTYPE(cu_seqlens_q, at::ScalarType::Int);
  CHECK_DTYPE(cu_seqlens_k, at::ScalarType::Int);
  CHECK_DTYPE(value_offsets, at::ScalarType::Int);
  CHECK_DIMS(query_int8, 3);
  CHECK_DIMS(key_int8, 3);
  CHECK_DIMS(output, 3);
  CHECK_DIMS(cu_seqlens_q, 1);
  CHECK_DIMS(cu_seqlens_k, 1);
  CHECK_DIMS(value_offsets, 1);
  TORCH_CHECK(cu_seqlens_q.size(0) >= 2,
              "varlen W8A8 requires at least one sequence");
  const int batch_size = cu_seqlens_q.size(0) - 1;
  const int query_heads = query_int8.size(1);
  const int key_heads = key_int8.size(1);
  const int head_dim = query_int8.size(2);
  TORCH_CHECK(cu_seqlens_k.size(0) == batch_size + 1,
              "varlen W8A8 Q/K batch counts must match");
  TORCH_CHECK(value_offsets.size(0) == batch_size + 1,
              "varlen W8A8 V offsets must be [B+1]");
  TORCH_CHECK(query_heads % key_heads == 0,
              "varlen W8A8 Q heads must be divisible by KV heads");
  TORCH_CHECK((head_dim == 64 || head_dim == 128) &&
                  key_int8.size(2) == head_dim,
              "varlen W8A8 requires matching head_dim 64 or 128");
  TORCH_CHECK(output.sizes() == query_int8.sizes(),
              "varlen W8A8 output must match Q");
  TORCH_CHECK(
      value_int8.sizes() == at::IntArrayRef(
          {key_heads, head_dim,
           div_ceil(
               static_cast<int>(key_int8.size(0)) +
                   batch_size * (kBlockTokens - 1),
               kBlockTokens) * kBlockTokens}),
      "varlen W8A8 quantized V must provide an aligned total_K + 63*B upper bound");
  TORCH_CHECK(
      value_scale.sizes() == at::IntArrayRef({batch_size, key_heads, head_dim}),
      "varlen W8A8 V scale must be [B,Hkv,D]");
  TORCH_CHECK(
      query_scale.sizes() == at::IntArrayRef(
          {batch_size, query_heads, div_ceil(max_seqlen_q, 64) * 4}),
      "varlen W8A8 Q scale shape is incompatible");
  TORCH_CHECK(
      key_scale.sizes() == at::IntArrayRef(
          {batch_size, key_heads, div_ceil(max_seqlen_k, 64)}),
      "varlen W8A8 K scale shape is incompatible");
  TORCH_CHECK(is_causal == 0 || is_causal == 1,
              "is_causal must be 0 or 1");
  TORCH_CHECK(
      query_int8.device() == key_int8.device() &&
          query_int8.device() == value_int8.device() &&
          query_int8.device() == value_scale.device() &&
          query_int8.device() == output.device() &&
          query_int8.device() == query_scale.device() &&
          query_int8.device() == key_scale.device() &&
          query_int8.device() == cu_seqlens_q.device() &&
          query_int8.device() == cu_seqlens_k.device() &&
          query_int8.device() == value_offsets.device(),
      "varlen W8A8 attention tensors must share one CUDA device");

  const auto half_options = output.options().dtype(at::ScalarType::Half);
  const auto float_options = output.options().dtype(at::ScalarType::Float);
  const auto byte_options = output.options().dtype(at::ScalarType::Byte);
  const auto long_options = output.options().dtype(at::ScalarType::Long);
  at::Tensor half_empty = at::empty({0, 0, 0, 0}, half_options);
  at::Tensor float_empty = at::empty({0, 0, 0}, float_options);
  at::Tensor policy_empty = at::empty({0}, byte_options);
  at::Tensor selected_empty = at::empty({0}, long_options);
  dispatch_sparse_threshold_attention(
      query_int8, key_int8, output, value_int8, value_scale, output,
      query_scale, key_scale, half_empty, half_empty, half_empty,
      float_empty, float_empty, policy_empty, policy_empty, selected_empty,
      cu_seqlens_q, cu_seqlens_k, value_offsets,
      max_seqlen_q, max_seqlen_k,
      1, 0.0f, softmax_scale, 64, true, true, true,
      is_causal, true, 0);
}

at::Tensor sol_sparse_online_int8_f16_attn(
    at::Tensor query_int8,
    at::Tensor key_int8,
    at::Tensor value,
    at::Tensor value_int8,
    at::Tensor value_scale,
    at::Tensor output,
    at::Tensor query_scale,
    at::Tensor key_scale,
    at::Tensor sparse_query_blocks,
    at::Tensor exact_kv_blocks,
    float threshold_sigma,
    int residual_subblocks,
    float softmax_scale,
    int return_stats,
    int use_w8a8,
    int force_dense,
    int key_tile_tokens,
    int is_causal,
    int route_original_basis)
{
  CHECK_CUDA(query_int8);
  CHECK_CUDA(key_int8);
  CHECK_CUDA(value);
  CHECK_CUDA(value_int8);
  CHECK_CUDA(value_scale);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(sparse_query_blocks);
  CHECK_CUDA(exact_kv_blocks);
  CHECK_LASTDIM_CONTIGUOUS(query_int8);
  CHECK_LASTDIM_CONTIGUOUS(key_int8);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(sparse_query_blocks);
  CHECK_CONTIGUOUS(exact_kv_blocks);
  CHECK_DIMS(query_int8, 4);
  CHECK_DIMS(key_int8, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  CHECK_DIMS(sparse_query_blocks, 1);
  CHECK_DIMS(exact_kv_blocks, 1);
  TORCH_CHECK(
      value.scalar_type() == at::ScalarType::Half ||
          value.scalar_type() == at::ScalarType::BFloat16,
      "Sol attention V/output must be float16 or bfloat16");
  TORCH_CHECK(
      output.scalar_type() == value.scalar_type(),
      "Sol attention V/output dtypes must match");
  CHECK_DTYPE(query_int8, at::ScalarType::Char);
  CHECK_DTYPE(key_int8, at::ScalarType::Char);
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);
  CHECK_DTYPE(sparse_query_blocks, at::ScalarType::Byte);
  CHECK_DTYPE(exact_kv_blocks, at::ScalarType::Byte);
  TORCH_CHECK(
      value.device() == output.device() &&
          value.device() == query_int8.device() &&
          value.device() == key_int8.device() &&
          value.device() == query_scale.device() &&
          value.device() == key_scale.device() &&
          value.device() == sparse_query_blocks.device() &&
          value.device() == exact_kv_blocks.device(),
      "Sol attention tensors must share one CUDA device");
  TORCH_CHECK(
      query_int8.size(0) == key_int8.size(0) &&
          query_int8.size(0) == value.size(0),
      "Sol attention batch sizes must match");
  TORCH_CHECK(
      key_int8.size(1) == value.size(1) &&
          key_int8.size(2) == value.size(2),
      "Sol attention K/V shapes must match");
  TORCH_CHECK(
      (query_int8.size(3) == 64 || query_int8.size(3) == 128) &&
          key_int8.size(3) == query_int8.size(3) &&
          value.size(3) == query_int8.size(3),
      "Sol attention requires matching head_dim=64 or 128");
  TORCH_CHECK(
      key_int8.size(1) > 0 &&
          query_int8.size(1) % key_int8.size(1) == 0,
      "Sol attention Q heads must be divisible by KV heads");
  TORCH_CHECK(
      query_int8.size(2) > 0 && key_int8.size(2) > 0,
      "empty attention is unsupported");
  TORCH_CHECK(
      output.sizes() == query_int8.sizes(),
      "Sol attention output must match INT8 Q shape");
  TORCH_CHECK(std::isfinite(threshold_sigma), "Sol threshold_sigma must be finite");
  TORCH_CHECK(
      residual_subblocks == 1 || residual_subblocks == 2,
      "Sol residual_subblocks must be 1 or 2");
  TORCH_CHECK(use_w8a8 == 0 || use_w8a8 == 1, "use_w8a8 must be 0 or 1");
  TORCH_CHECK(force_dense == 0 || force_dense == 1, "force_dense must be 0 or 1");
  TORCH_CHECK(is_causal == 0 || is_causal == 1, "is_causal must be 0 or 1");
  TORCH_CHECK(!is_causal || force_dense,
              "causal masking is supported only by dense W8A8");
  TORCH_CHECK(
      route_original_basis == 0 || route_original_basis == 1,
      "route_original_basis must be 0 or 1");
  TORCH_CHECK(
      key_tile_tokens == 64 || key_tile_tokens == 128,
      "key_tile_tokens must be 64 or 128");
  TORCH_CHECK(!force_dense || use_w8a8,
              "the specialized dense attention path currently requires W8A8");

  const int batch_size = query_int8.size(0);
  const int num_query_heads = query_int8.size(1);
  const int num_kv_heads = key_int8.size(1);
  const int head_dim = query_int8.size(3);
  const int num_query_blocks = div_ceil(query_int8.size(2), kBlockTokens);
  const int num_key_blocks = div_ceil(key_int8.size(2), kBlockTokens);
  if (use_w8a8)
  {
    CHECK_CONTIGUOUS(value_int8);
    CHECK_CONTIGUOUS(value_scale);
    CHECK_DIMS(value_int8, 4);
    CHECK_DIMS(value_scale, 3);
    CHECK_DTYPE(value_int8, at::ScalarType::Char);
    CHECK_DTYPE(value_scale, at::ScalarType::Float);
    TORCH_CHECK(
        value_int8.device() == value.device() && value_scale.device() == value.device(),
        "W8A8 V tensors must share the attention CUDA device");
    TORCH_CHECK(
        value_int8.size(0) == batch_size &&
            value_int8.size(1) == num_kv_heads &&
            value_int8.size(2) == head_dim &&
            value_int8.size(3) >= value.size(2) &&
            value_int8.size(3) % kBlockTokens == 0,
        "W8A8 V must be [B,Hkv,D,ceil(K/64)*64]");
    TORCH_CHECK(
        value_scale.sizes() == at::IntArrayRef({batch_size, num_kv_heads, head_dim}),
        "W8A8 V scale must be [B,Hkv,D]");
  }
  TORCH_CHECK(
      num_key_blocks <= kMaxKeyBlocks,
      "Sol attention supports at most ", kMaxKeyBlocks * kBlockTokens,
      " K/V tokens per call");
  TORCH_CHECK(
      sparse_query_blocks.size(0) == num_query_blocks,
      "sparse_query_blocks must contain one byte per 64-token Query block");
  TORCH_CHECK(
      exact_kv_blocks.size(0) == num_key_blocks,
      "exact_kv_blocks must contain one byte per 64-token K/V block");
  TORCH_CHECK(
      query_scale.size(0) == batch_size &&
          query_scale.size(1) == num_query_heads &&
          query_scale.size(2) == num_query_blocks * kWarps,
      "Sol Q scale must have shape [B, Hq, ceil(Q/64) * 4]");
  TORCH_CHECK(
      key_scale.size(0) == batch_size &&
          key_scale.size(1) == num_kv_heads &&
          key_scale.size(2) == num_key_blocks,
      "Sol K scale must have shape [B, Hkv, ceil(K/64)]");

  const int padded_key_blocks = div_ceil(num_key_blocks, kRouteTile) * kRouteTile;
  const int residual_tokens = kBlockTokens / residual_subblocks;
  const int num_residual_summaries =
      div_ceil(static_cast<int>(key_int8.size(2)), residual_tokens);
  const int padded_residual_summaries =
      div_ceil(num_residual_summaries, kSummaryTileTokens) * kSummaryTileTokens;
  const auto half_options = value.options().dtype(at::ScalarType::Half);
  const auto float_options = value.options().dtype(at::ScalarType::Float);
  at::Tensor key_score_summary = force_dense
      ? at::empty({0, 0, padded_residual_summaries, 0}, half_options)
      : at::empty(
          {batch_size, num_kv_heads, padded_residual_summaries, head_dim}, half_options);
  at::Tensor key_summary = force_dense
      ? at::empty({0, 0, padded_key_blocks, 0}, half_options)
      : residual_subblocks == 1 && !route_original_basis
          ? key_score_summary
          : at::empty(
              {batch_size, num_kv_heads, padded_key_blocks, head_dim}, half_options);
  at::Tensor value_mean = force_dense
      ? at::empty({0, 0, padded_residual_summaries, 0}, half_options)
      : at::empty_like(key_score_summary);
  at::Tensor key_summary_mean = force_dense
      ? at::empty({0, 0, 0}, float_options)
      : at::empty({batch_size, num_kv_heads, head_dim}, float_options);
  at::Tensor key_summary_variance = force_dense
      ? at::empty({0, 0, 0}, float_options)
      : at::empty_like(key_summary_mean);
  at::Tensor selected_count = return_stats
      ? at::zeros({1}, value.options().dtype(at::ScalarType::Long))
      : at::empty({0}, value.options().dtype(at::ScalarType::Long));
  at::Tensor empty_cu = at::empty(
      {0}, value.options().dtype(at::ScalarType::Int));

  dispatch_sparse_threshold_attention(
      query_int8, key_int8, value, value_int8, value_scale, output,
      query_scale, key_scale, key_summary, key_score_summary, value_mean,
      key_summary_mean, key_summary_variance, sparse_query_blocks,
      exact_kv_blocks, selected_count, empty_cu, empty_cu, empty_cu, 0, 0,
      residual_subblocks, threshold_sigma,
      softmax_scale, key_tile_tokens, use_w8a8, force_dense, false,
      is_causal, false,
      route_original_basis);
  return selected_count;
}

std::vector<at::Tensor> sol_w8a8_precompute_summaries(
    at::Tensor key_int8,
    at::Tensor key_scale,
    at::Tensor value,
    at::Tensor value_scale,
    int residual_subblocks,
    int route_original_basis)
{
  CHECK_CUDA(key_int8);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(value);
  CHECK_CUDA(value_scale);
  CHECK_LASTDIM_CONTIGUOUS(key_int8);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(value_scale);
  CHECK_DIMS(key_int8, 4);
  CHECK_DIMS(key_scale, 3);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(value_scale, 3);
  CHECK_DTYPE(key_int8, at::ScalarType::Char);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);
  CHECK_DTYPE(value_scale, at::ScalarType::Float);
  TORCH_CHECK(
      value.scalar_type() == at::ScalarType::Half ||
          value.scalar_type() == at::ScalarType::BFloat16,
      "Sol W8A8 summary V must be float16 or bfloat16");
  TORCH_CHECK(
      key_int8.device() == key_scale.device() &&
          key_int8.device() == value.device() &&
          key_int8.device() == value_scale.device(),
      "Sol W8A8 summary tensors must share one CUDA device");
  TORCH_CHECK(
      key_int8.size(0) == value.size(0) &&
          key_int8.size(1) == value.size(1) &&
          key_int8.size(2) == value.size(2) &&
          (key_int8.size(3) == 64 || key_int8.size(3) == 128) &&
          value.size(3) == key_int8.size(3),
      "Sol W8A8 summary K/V shapes are incompatible");
  TORCH_CHECK(
      residual_subblocks == 1 || residual_subblocks == 2,
      "Sol residual_subblocks must be 1 or 2");
  TORCH_CHECK(
      route_original_basis == 0 || route_original_basis == 1,
      "route_original_basis must be 0 or 1");

  const int batch_size = key_int8.size(0);
  const int num_kv_heads = key_int8.size(1);
  const int key_length = key_int8.size(2);
  const int head_dim = key_int8.size(3);
  const int num_key_blocks = div_ceil(key_length, kBlockTokens);
  const int padded_key_blocks = div_ceil(num_key_blocks, kRouteTile) * kRouteTile;
  const int residual_tokens = kBlockTokens / residual_subblocks;
  const int num_residual_summaries = div_ceil(key_length, residual_tokens);
  const int padded_residual_summaries =
      div_ceil(num_residual_summaries, kSummaryTileTokens) * kSummaryTileTokens;
  TORCH_CHECK(
      key_scale.sizes() == at::IntArrayRef({batch_size, num_kv_heads, num_key_blocks}),
      "Sol W8A8 K scale shape is incompatible");
  TORCH_CHECK(
      value_scale.sizes() == at::IntArrayRef({batch_size, num_kv_heads, head_dim}),
      "Sol W8A8 V scale shape is incompatible");

  const auto half_options = value.options().dtype(at::ScalarType::Half);
  const auto float_options = value.options().dtype(at::ScalarType::Float);
  at::Tensor key_score_summary = at::empty(
      {batch_size, num_kv_heads, padded_residual_summaries, head_dim}, half_options);
  at::Tensor key_summary = residual_subblocks == 1 && !route_original_basis
      ? key_score_summary
      : at::empty(
          {batch_size, num_kv_heads, padded_key_blocks, head_dim}, half_options);
  at::Tensor value_mean = at::empty_like(key_score_summary);
  at::Tensor key_summary_mean = at::empty(
      {batch_size, num_kv_heads, head_dim}, float_options);
  at::Tensor key_summary_variance = at::empty_like(key_summary_mean);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  dim3 summary_grid(num_key_blocks, num_kv_heads, batch_size);
#define LAUNCH_SUMMARY(HEAD_DIM, SCALAR)                                     \
    kv_block_summary_kernel<HEAD_DIM, SCALAR, true><<<                       \
        summary_grid, HEAD_DIM, 0, stream>>>(                                \
        key_int8.data_ptr<int8_t>(),                                         \
        key_scale.data_ptr<float>(),                                         \
        reinterpret_cast<const SCALAR *>(value.data_ptr()),                  \
        value_scale.data_ptr<float>(),                                       \
        reinterpret_cast<half *>(key_summary.data_ptr()),                    \
        reinterpret_cast<half *>(key_score_summary.data_ptr()),              \
        reinterpret_cast<half *>(value_mean.data_ptr()),                     \
        batch_size, num_kv_heads, key_length, padded_key_blocks,             \
        residual_subblocks, route_original_basis,                            \
        padded_residual_summaries,                                           \
        key_int8.stride(0), key_int8.stride(1), key_int8.stride(2),          \
        value.stride(0), value.stride(1),                                    \
        value.stride(2))
  if (value.scalar_type() == at::ScalarType::Half)
  {
    if (head_dim == 64)
      LAUNCH_SUMMARY(64, half);
    else
      LAUNCH_SUMMARY(128, half);
  }
  else
  {
    if (head_dim == 64)
      LAUNCH_SUMMARY(64, nv_bfloat16);
    else
      LAUNCH_SUMMARY(128, nv_bfloat16);
  }
#undef LAUNCH_SUMMARY
  check_launch("sparse K/V summary");
  if (head_dim == 64)
    key_summary_stats_kernel<64><<<batch_size * num_kv_heads, 64, 0, stream>>>(
        reinterpret_cast<const half *>(key_summary.data_ptr()),
        key_summary_mean.data_ptr<float>(), key_summary_variance.data_ptr<float>(),
        num_key_blocks, padded_key_blocks);
  else
    key_summary_stats_kernel<128><<<batch_size * num_kv_heads, 128, 0, stream>>>(
        reinterpret_cast<const half *>(key_summary.data_ptr()),
        key_summary_mean.data_ptr<float>(), key_summary_variance.data_ptr<float>(),
        num_key_blocks, padded_key_blocks);
  check_launch("sparse key summary statistics");
  return {
      key_summary,
      key_score_summary,
      value_mean,
      key_summary_mean,
      key_summary_variance};
}

at::Tensor sol_sparse_online_w8a8_prequantized_attn(
    at::Tensor query_int8,
    at::Tensor key_int8,
    at::Tensor value_int8,
    at::Tensor value_scale,
    at::Tensor output,
    at::Tensor query_scale,
    at::Tensor key_scale,
    at::Tensor key_summary,
    at::Tensor key_score_summary,
    at::Tensor value_mean,
    at::Tensor key_summary_mean,
    at::Tensor key_summary_variance,
    at::Tensor sparse_query_blocks,
    at::Tensor exact_kv_blocks,
    float threshold_sigma,
    int residual_subblocks,
    float softmax_scale,
    int return_stats,
    int force_dense,
    int key_tile_tokens,
    int is_causal,
    int route_original_basis)
{
  CHECK_CUDA(query_int8);
  CHECK_CUDA(key_int8);
  CHECK_CUDA(value_int8);
  CHECK_CUDA(value_scale);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(sparse_query_blocks);
  CHECK_CUDA(exact_kv_blocks);
  CHECK_LASTDIM_CONTIGUOUS(query_int8);
  CHECK_LASTDIM_CONTIGUOUS(key_int8);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(value_int8);
  CHECK_CONTIGUOUS(value_scale);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(sparse_query_blocks);
  CHECK_CONTIGUOUS(exact_kv_blocks);
  CHECK_DTYPE(query_int8, at::ScalarType::Char);
  CHECK_DTYPE(key_int8, at::ScalarType::Char);
  CHECK_DTYPE(value_int8, at::ScalarType::Char);
  CHECK_DTYPE(value_scale, at::ScalarType::Float);
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);
  CHECK_DTYPE(sparse_query_blocks, at::ScalarType::Byte);
  CHECK_DTYPE(exact_kv_blocks, at::ScalarType::Byte);
  TORCH_CHECK(
      output.scalar_type() == at::ScalarType::Half ||
          output.scalar_type() == at::ScalarType::BFloat16,
      "prequantized Sol W8A8 output must be float16 or bfloat16");
  TORCH_CHECK(
      residual_subblocks == 1 || residual_subblocks == 2,
      "Sol residual_subblocks must be 1 or 2");
  TORCH_CHECK(force_dense == 0 || force_dense == 1, "force_dense must be 0 or 1");
  TORCH_CHECK(is_causal == 0 || is_causal == 1, "is_causal must be 0 or 1");
  TORCH_CHECK(!is_causal || force_dense,
              "causal masking is supported only by dense W8A8");
  TORCH_CHECK(
      route_original_basis == 0 || route_original_basis == 1,
      "route_original_basis must be 0 or 1");
  TORCH_CHECK(
      key_tile_tokens == 64 || key_tile_tokens == 128,
      "key_tile_tokens must be 64 or 128");

  const int batch_size = query_int8.size(0);
  const int num_query_heads = query_int8.size(1);
  const int num_kv_heads = key_int8.size(1);
  const int query_length = query_int8.size(2);
  const int key_length = key_int8.size(2);
  const int head_dim = query_int8.size(3);
  const int num_query_blocks = div_ceil(query_length, kBlockTokens);
  const int num_key_blocks = div_ceil(key_length, kBlockTokens);
  const int padded_key_blocks = div_ceil(num_key_blocks, kRouteTile) * kRouteTile;
  const int residual_tokens = kBlockTokens / residual_subblocks;
  const int num_residual_summaries = div_ceil(key_length, residual_tokens);
  const int padded_residual_summaries =
      div_ceil(num_residual_summaries, kSummaryTileTokens) * kSummaryTileTokens;
  TORCH_CHECK(
      query_int8.dim() == 4 && key_int8.dim() == 4 && output.dim() == 4,
      "prequantized Sol W8A8 Q/K/O must be four-dimensional");
  TORCH_CHECK(
      (head_dim == 64 || head_dim == 128) && key_int8.size(3) == head_dim,
      "prequantized Sol W8A8 requires matching head_dim=64 or 128");
  TORCH_CHECK(
      output.sizes() == query_int8.sizes(),
      "prequantized Sol W8A8 output must match Q");
  TORCH_CHECK(
      num_kv_heads > 0 && num_query_heads % num_kv_heads == 0,
      "prequantized Sol W8A8 Q heads must be divisible by KV heads");
  TORCH_CHECK(
      value_int8.sizes() == at::IntArrayRef(
          {batch_size, num_kv_heads, head_dim, div_ceil(key_length, kBlockTokens) * kBlockTokens}),
      "prequantized Sol W8A8 V shape is incompatible");
  TORCH_CHECK(
      value_scale.sizes() == at::IntArrayRef({batch_size, num_kv_heads, head_dim}),
      "prequantized Sol W8A8 V scale is incompatible");
  TORCH_CHECK(
      query_scale.sizes() == at::IntArrayRef({batch_size, num_query_heads, num_query_blocks * kWarps}) &&
          key_scale.sizes() == at::IntArrayRef({batch_size, num_kv_heads, num_key_blocks}),
      "prequantized Sol W8A8 Q/K scale shapes are incompatible");
  TORCH_CHECK(
      sparse_query_blocks.numel() == num_query_blocks &&
          exact_kv_blocks.numel() == num_key_blocks,
      "prequantized Sol W8A8 policy shapes are incompatible");

  if (!force_dense)
  {
    CHECK_CUDA(key_summary);
    CHECK_CUDA(key_score_summary);
    CHECK_CUDA(value_mean);
    CHECK_CUDA(key_summary_mean);
    CHECK_CUDA(key_summary_variance);
    CHECK_CONTIGUOUS(key_summary);
    CHECK_CONTIGUOUS(key_score_summary);
    CHECK_CONTIGUOUS(value_mean);
    CHECK_CONTIGUOUS(key_summary_mean);
    CHECK_CONTIGUOUS(key_summary_variance);
    CHECK_DTYPE(key_summary, at::ScalarType::Half);
    CHECK_DTYPE(key_score_summary, at::ScalarType::Half);
    CHECK_DTYPE(value_mean, at::ScalarType::Half);
    CHECK_DTYPE(key_summary_mean, at::ScalarType::Float);
    CHECK_DTYPE(key_summary_variance, at::ScalarType::Float);
    TORCH_CHECK(
        key_summary.sizes() == at::IntArrayRef({batch_size, num_kv_heads, padded_key_blocks, head_dim}) &&
            key_score_summary.sizes() == at::IntArrayRef({batch_size, num_kv_heads, padded_residual_summaries, head_dim}) &&
            value_mean.sizes() == key_score_summary.sizes() &&
            key_summary_mean.sizes() == at::IntArrayRef({batch_size, num_kv_heads, head_dim}) &&
            key_summary_variance.sizes() == key_summary_mean.sizes(),
        "precomputed Sol W8A8 summary shapes are incompatible");
  }
  TORCH_CHECK(
      query_int8.device() == key_int8.device() &&
          query_int8.device() == value_int8.device() &&
          query_int8.device() == value_scale.device() &&
          query_int8.device() == output.device() &&
          query_int8.device() == query_scale.device() &&
          query_int8.device() == key_scale.device() &&
          query_int8.device() == sparse_query_blocks.device() &&
          query_int8.device() == exact_kv_blocks.device() &&
          (force_dense ||
              (query_int8.device() == key_summary.device() &&
               query_int8.device() == key_score_summary.device() &&
               query_int8.device() == value_mean.device() &&
               query_int8.device() == key_summary_mean.device() &&
               query_int8.device() == key_summary_variance.device())),
      "prequantized Sol W8A8 tensors must share one CUDA device");

  at::Tensor selected_count = return_stats
      ? at::zeros({1}, output.options().dtype(at::ScalarType::Long))
      : at::empty({0}, output.options().dtype(at::ScalarType::Long));
  at::Tensor empty_cu = at::empty(
      {0}, output.options().dtype(at::ScalarType::Int));
  dispatch_sparse_threshold_attention(
      query_int8, key_int8, output, value_int8, value_scale, output,
      query_scale, key_scale, key_summary, key_score_summary, value_mean,
      key_summary_mean, key_summary_variance, sparse_query_blocks,
      exact_kv_blocks, selected_count, empty_cu, empty_cu, empty_cu, 0, 0,
      residual_subblocks, threshold_sigma,
      softmax_scale, key_tile_tokens, true, force_dense, true, is_causal,
      false, route_original_basis);
  return selected_count;
}
