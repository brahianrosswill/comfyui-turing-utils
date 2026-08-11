// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
// Turing-specialized V -> signed INT8 quantization.  The channel-major,
// 16-token-permuted output is consumed directly by SM75 u8 x s8 PV MMA.

#include "../utils.cuh"
#include "torch_compat.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace {

constexpr int kChannelTile = 8;

__device__ __forceinline__ int inverse_permute_16(int value)
{
  return (value & 1) | (((value >> 3) & 1) << 1) |
      (((value >> 1) & 1) << 2) | (((value >> 2) & 1) << 3);
}

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<half>(half value)
{
  return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 value)
{
  return __bfloat162float(value);
}

__device__ __forceinline__ float warp_max(float value)
{
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  return value;
}

__device__ __forceinline__ int8_t quantize_s8(float value)
{
  int converted;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(converted) : "f"(value));
  return static_cast<int8_t>(converted);
}

template <typename T, int Threads>
__global__ void quantize_value_kernel(
    const T *__restrict__ value,
    int8_t *__restrict__ quantized,
    float *__restrict__ scale,
    int sequence_length,
    int padded_sequence_length,
    int heads,
    int head_dim,
    int64_t stride_batch,
    int64_t stride_head,
    int64_t stride_sequence)
{
  constexpr int Warps = Threads / 32;
  const int channel_tiles = head_dim / kChannelTile;
  const int channel_tile = blockIdx.x % channel_tiles;
  const int batch_head = blockIdx.x / channel_tiles;
  const int head = batch_head % heads;
  const int batch = batch_head / heads;
  const int channel_start = channel_tile * kChannelTile;
  const T *base = value + batch * stride_batch + head * stride_head + channel_start;

  float maximum[kChannelTile];
#pragma unroll
  for (int channel = 0; channel < kChannelTile; ++channel)
    maximum[channel] = 0.0f;

  int token = threadIdx.x;
  const int body = sequence_length - Threads;
  for (; token < body; token += 2 * Threads)
  {
#pragma unroll
    for (int channel = 0; channel < kChannelTile; ++channel)
    {
      const float first = fabsf(to_float(base[token * stride_sequence + channel]));
      const float second = fabsf(to_float(
          base[(token + Threads) * stride_sequence + channel]));
      maximum[channel] = fmaxf(maximum[channel], fmaxf(first, second));
    }
  }
  for (; token < sequence_length; token += Threads)
  {
#pragma unroll
    for (int channel = 0; channel < kChannelTile; ++channel)
      maximum[channel] = fmaxf(
          maximum[channel],
          fabsf(to_float(base[token * stride_sequence + channel])));
  }

#pragma unroll
  for (int channel = 0; channel < kChannelTile; ++channel)
    maximum[channel] = warp_max(maximum[channel]);

  __shared__ float warp_maximum[kChannelTile][Warps];
  __shared__ float inverse_scale[kChannelTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0)
  {
#pragma unroll
    for (int channel = 0; channel < kChannelTile; ++channel)
      warp_maximum[channel][warp] = maximum[channel];
  }
  __syncthreads();

  if (threadIdx.x < kChannelTile)
  {
    float channel_maximum = 0.0f;
#pragma unroll
    for (int source_warp = 0; source_warp < Warps; ++source_warp)
      channel_maximum = fmaxf(
          channel_maximum, warp_maximum[threadIdx.x][source_warp]);
    const float channel_scale = fmaxf(channel_maximum / 127.0f, 1.0e-12f);
    scale[(batch_head * head_dim) + channel_start + threadIdx.x] = channel_scale;
    inverse_scale[threadIdx.x] = 1.0f / channel_scale;
  }
  __syncthreads();

  const int64_t output_base =
      static_cast<int64_t>(batch_head * head_dim + channel_start) *
      padded_sequence_length;
  for (int source = sequence_length - 1 - threadIdx.x;
       source >= 0;
       source -= Threads)
  {
    const int within_group = source & 15;
    const int destination =
        (source & ~15) | inverse_permute_16(within_group);
#pragma unroll
    for (int channel = 0; channel < kChannelTile; ++channel)
    {
      quantized[output_base + channel * padded_sequence_length + destination] =
          quantize_s8(to_float(base[source * stride_sequence + channel]) *
                      inverse_scale[channel]);
    }
  }
  for (int source = sequence_length + threadIdx.x;
       source < padded_sequence_length;
       source += Threads)
  {
    const int destination =
        (source & ~15) | inverse_permute_16(source & 15);
#pragma unroll
    for (int channel = 0; channel < kChannelTile; ++channel)
      quantized[output_base + channel * padded_sequence_length + destination] = 0;
  }
}

void check_launch()
{
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(
      error == cudaSuccess,
      "Turing W8A8 V quantization launch failed: ",
      cudaGetErrorString(error));
}

} // namespace

void quantize_v_int8_sm75(
    at::Tensor value,
    at::Tensor quantized,
    at::Tensor scale)
{
  CHECK_CUDA(value);
  CHECK_CUDA(quantized);
  CHECK_CUDA(scale);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(quantized, 4);
  CHECK_DIMS(scale, 3);
  CHECK_CONTIGUOUS(quantized);
  CHECK_CONTIGUOUS(scale);
  CHECK_DTYPE(quantized, at::ScalarType::Char);
  CHECK_DTYPE(scale, at::ScalarType::Float);
  TORCH_CHECK(
      value.scalar_type() == at::ScalarType::Half ||
          value.scalar_type() == at::ScalarType::BFloat16,
      "Turing W8A8 V must be FP16 or BF16");
  TORCH_CHECK(value.stride(3) == 1, "Turing W8A8 V head dimension must be contiguous");
  TORCH_CHECK(value.size(3) > 0 && value.size(3) % kChannelTile == 0,
              "Turing W8A8 V head dimension must be divisible by 8");
  TORCH_CHECK(
      quantized.size(0) == value.size(0) &&
          quantized.size(1) == value.size(1) &&
          quantized.size(2) == value.size(3) &&
          quantized.size(3) >= value.size(2) &&
          quantized.size(3) % 64 == 0,
      "Turing W8A8 quantized V must be [B,H,D,ceil(N/64)*64]");
  TORCH_CHECK(
      scale.sizes() == at::IntArrayRef({value.size(0), value.size(1), value.size(3)}),
      "Turing W8A8 V scale must be [B,H,D]");
  TORCH_CHECK(
      value.device() == quantized.device() && value.device() == scale.device(),
      "Turing W8A8 V tensors must share a CUDA device");

  constexpr int Threads = 256;
  const int blocks = value.size(0) * value.size(1) *
      (value.size(3) / kChannelTile);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  if (value.scalar_type() == at::ScalarType::Half)
  {
    quantize_value_kernel<half, Threads><<<blocks, Threads, 0, stream>>>(
        reinterpret_cast<const half *>(value.data_ptr()),
        quantized.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        value.size(2),
        quantized.size(3),
        value.size(1),
        value.size(3),
        value.stride(0),
        value.stride(1),
        value.stride(2));
  }
  else
  {
    quantize_value_kernel<nv_bfloat16, Threads><<<blocks, Threads, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16 *>(value.data_ptr()),
        quantized.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        value.size(2),
        quantized.size(3),
        value.size(1),
        value.size(3),
        value.stride(0),
        value.stride(1),
        value.stride(2));
  }
  check_launch();
}
