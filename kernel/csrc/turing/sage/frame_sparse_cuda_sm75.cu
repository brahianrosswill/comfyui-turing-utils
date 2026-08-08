/*
 * Experimental structured frame-sparse Sage attention for SM75.
 *
 * Copyright (c) 2024 by SageAttention team.
 * Licensed under the Apache License, Version 2.0.
 *
 * Q/K use the production per-warp/per-block INT8 Sage quantization and SM75
 * Tensor Core MMA. V and output remain FP16/BF16, while online softmax and the
 * output accumulator remain FP32. A model-independent CSR schedule supplies
 * exact K/V blocks for each 64-token Query block. The schedule is shared by
 * every batch item and head, so this path has no online summaries, dynamic
 * routing tensor, or per-head route construction.
 */

#include "../utils.cuh"
#include "../math.cuh"
#include "attn_utils.cuh"
#include "torch_compat.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace {

constexpr int kBlockTokens = 64;
constexpr int kHeadDim = 128;
constexpr int kWarps = 4;
constexpr int kHalfPacks = kHeadDim / 8;
constexpr int kInt8Packs = kHeadDim / 16;
constexpr int kInt8TilePacks = kBlockTokens * kInt8Packs;
constexpr int kTileBytes = kBlockTokens * kHeadDim * sizeof(half);
constexpr int kInt8TileBytes = kBlockTokens * kHeadDim * sizeof(int8_t);
constexpr int kAttentionSharedBytes = 2 * kTileBytes;

template <typename T>
__device__ __forceinline__ b128_t pack_to_half(const T *source);

template <>
__device__ __forceinline__ b128_t pack_to_half<half>(const half *source)
{
  return *reinterpret_cast<const b128_t *>(source);
}

template <>
__device__ __forceinline__ b128_t pack_to_half<nv_bfloat16>(
    const nv_bfloat16 *source)
{
  return bf16_pack_to_half(source);
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
  for (int line = linear_thread; line < tile_packs;
       line += kWarps * WARP_SIZE)
  {
    const int row = line / kHalfPacks;
    const int column = line % kHalfPacks;
    const uint32_t offset = destination.get_permuted_offset(row, column);
    if (row_start + row < row_limit)
    {
      destination.base[offset] = pack_to_half(
          source + static_cast<int64_t>(row_start + row) * stride_sequence +
          column * 8);
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

template <typename T>
__global__ void frame_sparse_attention_kernel(
    const int8_t *__restrict__ query_int8,
    const int8_t *__restrict__ key_int8,
    const T *__restrict__ value,
    T *__restrict__ output,
    const float *__restrict__ query_scale,
    const float *__restrict__ key_scale,
    const int32_t *__restrict__ row_offsets,
    const int32_t *__restrict__ key_blocks,
    int query_length,
    int key_length,
    int num_query_heads,
    int num_kv_heads,
    int num_query_blocks,
    int num_key_blocks,
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
      "SM75 frame-sparse attention must stay within 32 KiB");
  extern __shared__ int8_t shared_bytes[];
  smem_t<SwizzleMode::k128B, kInt8Packs> shared_query_int8(shared_bytes);
  smem_t<SwizzleMode::k128B, kInt8Packs> shared_key_int8(
      shared_bytes + kInt8TileBytes);
  smem_t<SwizzleMode::k128B, kHalfPacks> shared_value(
      shared_bytes + 2 * kInt8TileBytes);
  smem_t<SwizzleMode::k128B, kHalfPacks> shared_output(shared_bytes);

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

  load_int8_tile(
      query_int8_head_ptr,
      stride_sequence_q_int8,
      query_block * kBlockTokens,
      query_length,
      shared_query_int8);
  __syncthreads();

  const float scale_log2 = softmax_scale * math::log2e;
  const uint32_t value_mma_offset = shared_value.get_permuted_offset(
      threadIdx.x % 16, threadIdx.x / 16);
  const int schedule_begin = row_offsets[query_block];
  const int schedule_end = row_offsets[query_block + 1];
  for (int schedule_index = schedule_begin;
       schedule_index < schedule_end;
       ++schedule_index)
  {
    const int key_block = key_blocks[schedule_index];
    if (key_block < 0 || key_block >= num_key_blocks)
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
        shared_value);
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
            __float22half2_rn(
                reinterpret_cast<float2 *>(output_fragment[0][value_tile])[pair]);
      }
      else
      {
        reinterpret_cast<nv_bfloat162 *>(converted)[pair] =
            __float22bfloat162_rn(
                reinterpret_cast<float2 *>(output_fragment[0][value_tile])[pair]);
      }
    }
    reinterpret_cast<uint32_t *>(
        shared_output.base + output_offset)[threadIdx.x % 4] = converted[0];
    reinterpret_cast<uint32_t *>(
        shared_output.base + output_offset + 8 * kHalfPacks)[threadIdx.x % 4] =
        converted[1];
    reinterpret_cast<uint32_t *>(
        shared_output.base + (output_offset ^ 0x1))[threadIdx.x % 4] = converted[2];
    reinterpret_cast<uint32_t *>(
        shared_output.base + (output_offset ^ 0x1) + 8 * kHalfPacks)
        [threadIdx.x % 4] = converted[3];
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
  TORCH_CHECK(
      error == cudaSuccess,
      name,
      " launch failed: ",
      cudaGetErrorString(error));
}

template <typename T>
void launch_frame_sparse_attention(
    at::Tensor query_int8,
    at::Tensor key_int8,
    at::Tensor value,
    at::Tensor output,
    at::Tensor query_scale,
    at::Tensor key_scale,
    at::Tensor row_offsets,
    at::Tensor key_blocks,
    float softmax_scale)
{
  const int batch_size = query_int8.size(0);
  const int num_query_heads = query_int8.size(1);
  const int num_kv_heads = key_int8.size(1);
  const int query_length = query_int8.size(2);
  const int key_length = key_int8.size(2);
  const int num_query_blocks = div_ceil(query_length, kBlockTokens);
  const int num_key_blocks = div_ceil(key_length, kBlockTokens);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  dim3 grid(num_query_blocks, num_query_heads, batch_size);
  dim3 block(WARP_SIZE, kWarps);
  frame_sparse_attention_kernel<T><<<
      grid,
      block,
      kAttentionSharedBytes,
      stream>>>(
      query_int8.data_ptr<int8_t>(),
      key_int8.data_ptr<int8_t>(),
      reinterpret_cast<const T *>(value.data_ptr()),
      reinterpret_cast<T *>(output.data_ptr()),
      query_scale.data_ptr<float>(),
      key_scale.data_ptr<float>(),
      row_offsets.data_ptr<int32_t>(),
      key_blocks.data_ptr<int32_t>(),
      query_length,
      key_length,
      num_query_heads,
      num_kv_heads,
      num_query_blocks,
      num_key_blocks,
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
  check_launch("frame-sparse attention");
}

} // namespace

at::Tensor frame_sparse_int8_f16_attn(
    at::Tensor query_int8,
    at::Tensor key_int8,
    at::Tensor value,
    at::Tensor output,
    at::Tensor query_scale,
    at::Tensor key_scale,
    at::Tensor row_offsets,
    at::Tensor key_blocks,
    float softmax_scale)
{
  CHECK_CUDA(query_int8);
  CHECK_CUDA(key_int8);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(row_offsets);
  CHECK_CUDA(key_blocks);
  CHECK_LASTDIM_CONTIGUOUS(query_int8);
  CHECK_LASTDIM_CONTIGUOUS(key_int8);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(row_offsets);
  CHECK_CONTIGUOUS(key_blocks);
  CHECK_DIMS(query_int8, 4);
  CHECK_DIMS(key_int8, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  CHECK_DIMS(row_offsets, 1);
  CHECK_DIMS(key_blocks, 1);
  CHECK_DTYPE(query_int8, at::ScalarType::Char);
  CHECK_DTYPE(key_int8, at::ScalarType::Char);
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);
  CHECK_DTYPE(row_offsets, at::ScalarType::Int);
  CHECK_DTYPE(key_blocks, at::ScalarType::Int);
  TORCH_CHECK(
      value.scalar_type() == at::ScalarType::Half ||
          value.scalar_type() == at::ScalarType::BFloat16,
      "frame-sparse V/output must be float16 or bfloat16");
  TORCH_CHECK(
      output.scalar_type() == value.scalar_type(),
      "frame-sparse V/output dtypes must match");
  TORCH_CHECK(
      query_int8.device() == key_int8.device() &&
          query_int8.device() == value.device() &&
          query_int8.device() == output.device() &&
          query_int8.device() == query_scale.device() &&
          query_int8.device() == key_scale.device() &&
          query_int8.device() == row_offsets.device() &&
          query_int8.device() == key_blocks.device(),
      "frame-sparse tensors must share one CUDA device");
  TORCH_CHECK(
      query_int8.size(0) == key_int8.size(0) &&
          query_int8.size(0) == value.size(0),
      "frame-sparse batch sizes must match");
  TORCH_CHECK(
      key_int8.size(1) == value.size(1) &&
          key_int8.size(2) == value.size(2),
      "frame-sparse K/V shapes must match");
  TORCH_CHECK(
      query_int8.size(3) == kHeadDim &&
          key_int8.size(3) == kHeadDim &&
          value.size(3) == kHeadDim,
      "frame-sparse attention requires head_dim=128");
  TORCH_CHECK(
      output.sizes() == query_int8.sizes(),
      "frame-sparse output must match Q shape");
  TORCH_CHECK(
      key_int8.size(1) > 0 && query_int8.size(1) % key_int8.size(1) == 0,
      "frame-sparse Q heads must be divisible by KV heads");
  TORCH_CHECK(
      query_int8.size(2) > 0 && key_int8.size(2) > 0,
      "empty frame-sparse attention is unsupported");
  const int batch_size = query_int8.size(0);
  const int num_query_heads = query_int8.size(1);
  const int num_kv_heads = key_int8.size(1);
  const int num_query_blocks = div_ceil(query_int8.size(2), kBlockTokens);
  const int num_key_blocks = div_ceil(key_int8.size(2), kBlockTokens);
  TORCH_CHECK(
      query_scale.sizes() == at::IntArrayRef(
          {batch_size, num_query_heads, num_query_blocks * kWarps}),
      "frame-sparse Q scale shape is incompatible");
  TORCH_CHECK(
      key_scale.sizes() == at::IntArrayRef(
          {batch_size, num_kv_heads, num_key_blocks}),
      "frame-sparse K scale shape is incompatible");
  TORCH_CHECK(
      row_offsets.numel() == num_query_blocks + 1,
      "frame-sparse row_offsets must contain one entry per Q block plus one");
  TORCH_CHECK(
      key_blocks.numel() > 0,
      "frame-sparse schedule must select at least one K block");
  TORCH_CHECK(
      std::isfinite(softmax_scale) && softmax_scale > 0.0f,
      "frame-sparse softmax scale must be finite and positive");

  if (value.scalar_type() == at::ScalarType::Half)
  {
    launch_frame_sparse_attention<half>(
        query_int8,
        key_int8,
        value,
        output,
        query_scale,
        key_scale,
        row_offsets,
        key_blocks,
        softmax_scale);
  }
  else
  {
    launch_frame_sparse_attention<nv_bfloat16>(
        query_int8,
        key_int8,
        value,
        output,
        query_scale,
        key_scale,
        row_offsets,
        key_blocks,
        softmax_scale);
  }
  return output;
}
