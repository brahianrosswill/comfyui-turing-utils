/*
 * Experimental Sol-style sparse attention for SM75.
 *
 * One 64-token centroid per block feeds an input-adaptive mean + tau * std
 * threshold. Each Query CTA performs routing directly before its FP32 online
 * softmax and keeps the route in CTA-local shared memory/registers, so no full
 * global proxy or route map is materialized. Selected blocks are
 * evaluated with the production per-warp/per-block INT8 Sage QK path. Routing
 * and skipped-block correction use centroids reconstructed from those same
 * INT8 Q/K tensors and scales, so block selection matches the score domain
 * actually evaluated by Sage. Query summary, thresholding, and routing are
 * fused into the attention CTA; original V means remain isolated to the
 * skipped-block approximation.
 * The official local neighborhood is fixed to +/- one 64-token block. Optional
 * exact-KV and dense-Query block masks carry model-independent modality policy.
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
constexpr int kWarps = 4;
constexpr int kRouteTile = 16;
constexpr int kRouteWordBits = 32;
constexpr int kMaxKeyBlocks = 4096;
constexpr int kMaxRouteWords = kMaxKeyBlocks / kRouteWordBits;
constexpr int kRouteWordsPerLane = kMaxRouteWords / WARP_SIZE;
constexpr int kSummaryTileTokens = 16;
constexpr int kMaxRouteBytes = kMaxRouteWords * sizeof(uint32_t);
constexpr int kProxyScratchBytes =
    kWarps * kSummaryTileTokens * sizeof(float);

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
  static constexpr int kValueTiles = HeadDim / 16;
  static constexpr SwizzleMode kInt8Swizzle =
      HeadDim == 64 ? SwizzleMode::k64B : SwizzleMode::k128B;
  static_assert(
      kRouteStorageOffset + kMaxRouteBytes + kProxyScratchBytes <=
          kAttentionSharedBytes,
      "SM75 fused routing metadata must fit beside the 16-block summaries");
};

static_assert(AttentionGeometry<64>::kAttentionSharedBytes == 16 * 1024);
static_assert(AttentionGeometry<128>::kAttentionSharedBytes == 32 * 1024);

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
  if (dimension >= HeadDim)
    return;

  const int token_start = block_index * kBlockTokens;
  const int token_count = token_start < sequence_length
      ? min(kBlockTokens, sequence_length - token_start)
      : 0;
  float value_sum[2] = {0.0f, 0.0f};
  int quantized_key_sum[2] = {0, 0};
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
    key_summary[output_index] = __float2half_rn(
        static_cast<float>(total_quantized_key_sum) * dequant_scale /
        static_cast<float>(token_count));
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
                HeadDim +
            dimension;
        if (key_score_summary != key_summary || residual_subblocks != 1)
          key_score_summary[residual_output_index] = __float2half_rn(0.0f);
        value_mean[residual_output_index] = __float2half_rn(0.0f);
      }
    }
  }
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

__device__ __forceinline__ bool route_selected(
    const uint32_t route_words[kRouteWordsPerLane],
    int key_block)
{
  const int word_index = key_block / kRouteWordBits;
  const int owner_lane = word_index % WARP_SIZE;
  const int lane_slot = word_index / WARP_SIZE;
  const uint32_t word = __shfl_sync(
      0xffffffff, route_words[lane_slot], owner_lane);
  return (word >> (key_block % kRouteWordBits)) & 1U;
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

template <int HeadDim, typename T, bool UseW8A8, bool ForceDense>
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
    unsigned long long *__restrict__ selected_count,
    int query_length,
    int key_length,
    int num_query_heads,
    int num_kv_heads,
    int num_query_blocks,
    int num_key_blocks,
    int residual_subblocks,
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
    int64_t stride_batch_o,
    int64_t stride_head_o,
    int64_t stride_sequence_o,
    int key_blocks_per_stage,
    float threshold_sigma,
    float softmax_scale)
{
  using G = AttentionGeometry<HeadDim>;
  static_assert(
      G::kAttentionSharedBytes <= 32 * 1024,
      "SM75 sparse attention must stay within 32 KiB");
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

  const int query_block = blockIdx.x;
  const int query_head = blockIdx.y;
  const int batch = blockIdx.z;
  const int kv_head = query_head / (num_query_heads / num_kv_heads);
  const bool sparse_query = sparse_query_blocks[query_block] != 0;
  const int linear_thread = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int8_t *query_int8_head_ptr = query_int8 +
      batch * stride_batch_q_int8 + query_head * stride_head_q_int8;
  const int8_t *key_int8_head_ptr = key_int8 +
      batch * stride_batch_k_int8 + kv_head * stride_head_k_int8;
  const T *value_head_ptr =
      value + batch * stride_batch_v + kv_head * stride_head_v;
  const int8_t *value_int8_head_ptr = UseW8A8
      ? value_int8 +
          static_cast<int64_t>(batch * num_kv_heads + kv_head) *
              HeadDim * padded_value_length
      : nullptr;
  const float *value_scale_head = UseW8A8
      ? value_scale +
          static_cast<int64_t>(batch * num_kv_heads + kv_head) * HeadDim
      : nullptr;
  T *output_head_ptr =
      output + batch * stride_batch_o + query_head * stride_head_o;
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
  uint32_t local_route[kRouteWordsPerLane] = {};

  if constexpr (!ForceDense)
  {
  // Route from the same INT8 Q and per-16-token scales consumed by exact Sage.
  // Keeping this tile in shared memory also avoids another global Q read when
  // constructing the correction operand below.
  load_int8_tile<HeadDim>(
      query_int8_head_ptr,
      stride_sequence_q_int8,
      query_block * kBlockTokens,
      query_length,
      shared_initial_query_int8);
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
  const float query_mean = query_sum / static_cast<float>(query_token_count);

  float *reduction_scratch = reinterpret_cast<float *>(
      shared_bytes + G::kRouteStorageOffset);
  uint32_t *shared_route = reinterpret_cast<uint32_t *>(
      shared_bytes + G::kRouteStorageOffset);

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

  const int route_word_count = (num_key_blocks + kRouteWordBits - 1) / kRouteWordBits;
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
  const int residual_tokens = kBlockTokens / residual_subblocks;
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

    const int routed_blocks = kSummaryTileTokens / residual_subblocks;
    if (linear_thread < routed_blocks)
    {
      const int key_block = summary_start / residual_subblocks + linear_thread;
      if (key_block < num_key_blocks)
      {
        float proxy_sum = 0.0f;
        int key_token_count = 0;
#pragma unroll
        for (int residual_index = 0; residual_index < 2; ++residual_index)
        {
          if (residual_index >= residual_subblocks)
            break;
          const int residual_summary =
              linear_thread * residual_subblocks + residual_index;
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
      const int key_block = residual_summary / residual_subblocks;
      const bool selected = key_block < num_key_blocks &&
          ((shared_route[key_block / kRouteWordBits] >>
            (key_block % kRouteWordBits)) & 1U);
      if (residual_summary >= num_residual_summaries || selected)
      {
        score[0][0][element] = -5000000.0f;
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

#pragma unroll
  for (int slot = 0; slot < kRouteWordsPerLane; ++slot)
  {
    const int word_index = threadIdx.x + slot * WARP_SIZE;
    local_route[slot] = word_index < route_word_count ? shared_route[word_index] : 0;
  }
  if (selected_count != nullptr && sparse_query && threadIdx.y == 0)
  {
    unsigned int count = 0;
#pragma unroll
    for (int slot = 0; slot < kRouteWordsPerLane; ++slot)
      count += __popc(local_route[slot]);
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
      count += __shfl_down_sync(0xffffffff, count, offset);
    if (threadIdx.x == 0)
      atomicAdd(selected_count, static_cast<unsigned long long>(count));
  }
  __syncthreads();
  }

  // Selected blocks retain exact token-level attention. Q/K are quantized once
  // with the production Sage per-16-row Q and per-64-row K scales, then use the
  // same SM75 INT8 Tensor Core MMA as stable Sage. V and output stay FP16/BF16
  // with FP32 online-softmax accumulation.
  load_int8_tile<HeadDim>(
      query_int8_head_ptr,
      stride_sequence_q_int8,
      query_block * kBlockTokens,
      query_length,
      shared_query_int8);
  __syncthreads();
  // The dense W8A8 path used to share the sparse kernel's runtime two-block
  // staging loop.  Even though dense execution never needs staged routing,
  // that runtime loop kept the second-stage state live on SM75 and raised the
  // D128 register footprint.  Make the dense specialization a compile-time
  // one-block loop; sparse variants retain the selectable 64/128-token tile.
  constexpr int kMaxKeyStages = ForceDense ? 1 : 2;
  const int key_group_stride = ForceDense ? 1 : key_blocks_per_stage;
  for (int key_group = 0; key_group < num_key_blocks;
       key_group += key_group_stride)
  {
#pragma unroll
    for (int key_stage = 0; key_stage < kMaxKeyStages; ++key_stage)
    {
    if constexpr (!ForceDense)
    {
      if (key_stage >= key_blocks_per_stage)
        continue;
    }
    const int key_block = key_group + key_stage;
    if (key_block >= num_key_blocks)
      continue;
    if constexpr (!ForceDense)
    {
      if (!route_selected(local_route, key_block))
        continue;
    }
    load_int8_tile<HeadDim>(
        key_int8_head_ptr,
        stride_sequence_k_int8,
        key_block * kBlockTokens,
        key_length,
        shared_key_int8);
    if constexpr (UseW8A8)
    {
      load_quantized_value_tile<HeadDim>(
          value_int8_head_ptr,
          padded_value_length,
          key_block,
          shared_selected_value_int8);
    }
    else
    {
      load_half_tile<HeadDim, kBlockTokens>(
          value_head_ptr,
          stride_sequence_v,
          key_block * kBlockTokens,
          key_length,
          shared_selected_value);
    }
    __syncthreads();

    int32_t integer_score[1][4][8];
    compute_int8_qk<HeadDim>(shared_query_int8, shared_key_int8, integer_score);
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
      compute_int8_sv_permuted<1, 4, G::kValueTiles, SwizzleMode::k64B, 4>(
          shared_selected_value_int8,
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
    }
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
          bool SummariesReady = false>
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
    int residual_subblocks,
    float threshold_sigma,
    float softmax_scale,
    int key_tile_tokens)
{
  using G = AttentionGeometry<HeadDim>;
  const int batch_size = query_int8.size(0);
  const int num_query_heads = query_int8.size(1);
  const int num_kv_heads = key_int8.size(1);
  const int query_length = query_int8.size(2);
  const int key_length = key_int8.size(2);
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
        residual_subblocks,
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
  sparse_attention_kernel<HeadDim, T, UseW8A8, ForceDense><<<
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
      selected_count.numel()
          ? reinterpret_cast<unsigned long long *>(selected_count.data_ptr<int64_t>())
          : nullptr,
      query_length,
      key_length,
      num_query_heads,
      num_kv_heads,
      num_query_blocks,
      num_key_blocks,
      residual_subblocks,
      padded_residual_summaries,
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
      output.stride(0),
      output.stride(1),
      output.stride(2),
      key_tile_tokens / kBlockTokens,
      threshold_sigma,
      softmax_scale);
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
    int residual_subblocks,
    float threshold_sigma,
    float softmax_scale,
    int key_tile_tokens,
    bool use_w8a8,
    bool force_dense,
    bool summaries_ready)
{
#define LAUNCH_VARIANT(HEAD_DIM, SCALAR, USE_W8A8, FORCE_DENSE, READY)       \
  launch_sparse_threshold_attention<HEAD_DIM, SCALAR, USE_W8A8,             \
                                    FORCE_DENSE, READY>(                     \
      query_int8, key_int8, value, value_int8, value_scale, output,          \
      query_scale, key_scale, key_summary, key_score_summary, value_mean,    \
      key_summary_mean, key_summary_variance, sparse_query_blocks,           \
      exact_kv_blocks, selected_count, residual_subblocks, threshold_sigma,  \
      softmax_scale, key_tile_tokens)
#define DISPATCH_FORMAT(HEAD_DIM, SCALAR)                                    \
  do                                                                         \
  {                                                                          \
    if (force_dense)                                                         \
      LAUNCH_VARIANT(HEAD_DIM, SCALAR, true, true, true);                    \
    else if (use_w8a8 && summaries_ready)                                    \
      LAUNCH_VARIANT(HEAD_DIM, SCALAR, true, false, true);                   \
    else if (use_w8a8)                                                       \
      LAUNCH_VARIANT(HEAD_DIM, SCALAR, true, false, false);                  \
    else                                                                     \
      LAUNCH_VARIANT(HEAD_DIM, SCALAR, false, false, false);                 \
  } while (false)

  const int head_dim = query_int8.size(3);
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

} // namespace

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
    int key_tile_tokens)
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
      : residual_subblocks == 1
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

  dispatch_sparse_threshold_attention(
      query_int8, key_int8, value, value_int8, value_scale, output,
      query_scale, key_scale, key_summary, key_score_summary, value_mean,
      key_summary_mean, key_summary_variance, sparse_query_blocks,
      exact_kv_blocks, selected_count, residual_subblocks, threshold_sigma,
      softmax_scale, key_tile_tokens, use_w8a8, force_dense, false);
  return selected_count;
}

std::vector<at::Tensor> sol_w8a8_precompute_summaries(
    at::Tensor key_int8,
    at::Tensor key_scale,
    at::Tensor value,
    at::Tensor value_scale,
    int residual_subblocks)
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
  at::Tensor key_summary = residual_subblocks == 1
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
        residual_subblocks, padded_residual_summaries,                       \
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
    int key_tile_tokens)
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
  dispatch_sparse_threshold_attention(
      query_int8, key_int8, output, value_int8, value_scale, output,
      query_scale, key_scale, key_summary, key_score_summary, value_mean,
      key_summary_mean, key_summary_variance, sparse_query_blocks,
      exact_kv_blocks, selected_count, residual_subblocks, threshold_sigma,
      softmax_scale, key_tile_tokens, true, force_dense, true);
  return selected_count;
}
