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

template <typename T, bool Aligned>
__device__ __forceinline__ void load_channels(const T *ptr, float (&values)[kChannelTile])
{
  if constexpr (Aligned)
  {
    union { uint4 packed; T channels[kChannelTile]; } data;
    data.packed = *reinterpret_cast<const uint4 *>(ptr);
#pragma unroll
    for (int c = 0; c < kChannelTile; ++c)
      values[c] = to_float(data.channels[c]);
  }
  else
  {
#pragma unroll
    for (int c = 0; c < kChannelTile; ++c)
      values[c] = to_float(ptr[c]);
  }
}

template <typename T, int Threads, bool Aligned>
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
  const int body = sequence_length - 3 * Threads;
  for (; token < body; token += 4 * Threads)
  {
    float a[kChannelTile], b[kChannelTile], c[kChannelTile], d[kChannelTile];
    load_channels<T, Aligned>(base + token * stride_sequence, a);
    load_channels<T, Aligned>(base + (token + Threads) * stride_sequence, b);
    load_channels<T, Aligned>(base + (token + 2 * Threads) * stride_sequence, c);
    load_channels<T, Aligned>(base + (token + 3 * Threads) * stride_sequence, d);
#pragma unroll
    for (int channel = 0; channel < kChannelTile; ++channel)
    {
      maximum[channel] = fmaxf(maximum[channel],
          fmaxf(fmaxf(fabsf(a[channel]), fabsf(b[channel])),
                fmaxf(fabsf(c[channel]), fabsf(d[channel]))));
    }
  }
  for (; token < sequence_length; token += Threads)
  {
    float values[kChannelTile];
    load_channels<T, Aligned>(base + token * stride_sequence, values);
#pragma unroll
    for (int channel = 0; channel < kChannelTile; ++channel)
      maximum[channel] = fmaxf(
          maximum[channel],
          fabsf(values[channel]));
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

  float channel_inverse[kChannelTile];
#pragma unroll
  for (int channel = 0; channel < kChannelTile; ++channel)
    channel_inverse[channel] = inverse_scale[channel];
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
    float values[kChannelTile];
    load_channels<T, Aligned>(base + source * stride_sequence, values);
#pragma unroll
    for (int channel = 0; channel < kChannelTile; ++channel)
    {
      quantized[output_base + channel * padded_sequence_length + destination] =
          quantize_s8(values[channel] * channel_inverse[channel]);
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

template <typename T, int Threads>
void launch_quantize(at::Tensor value, at::Tensor quantized, at::Tensor scale,
                     int blocks, cudaStream_t stream)
{
  bool aligned = reinterpret_cast<uintptr_t>(value.data_ptr()) % 16 == 0;
  for (int dim = 0; dim < 3; ++dim)
    aligned &= value.size(dim) == 1 || value.stride(dim) % kChannelTile == 0;
  auto kernel = aligned ? quantize_value_kernel<T, Threads, true>
                        : quantize_value_kernel<T, Threads, false>;
  kernel<<<blocks, Threads, 0, stream>>>(
      reinterpret_cast<const T *>(value.data_ptr()), quantized.data_ptr<int8_t>(),
      scale.data_ptr<float>(), value.size(2), quantized.size(3), value.size(1),
      value.size(3), value.stride(0), value.stride(1), value.stride(2));
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

  const int blocks = value.size(0) * value.size(1) *
      (value.size(3) / kChannelTile);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  if (value.scalar_type() == at::ScalarType::Half)
  {
    if (value.size(2) <= 256)
      launch_quantize<half, 128>(value, quantized, scale, blocks, stream);
    else
      launch_quantize<half, 512>(value, quantized, scale, blocks, stream);
  }
  else
  {
    if (value.size(2) <= 256)
      launch_quantize<nv_bfloat16, 128>(value, quantized, scale, blocks, stream);
    else
      launch_quantize<nv_bfloat16, 512>(value, quantized, scale, blocks, stream);
  }
  check_launch();
}
