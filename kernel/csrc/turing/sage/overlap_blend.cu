/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Turing Utils contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fused.h"
#include "dispatch_utils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>
#include <type_traits>

namespace {

constexpr int kThreads = 256;

template <typename T>
__device__ __forceinline__ float overlap_to_float(T value) {
  if constexpr (std::is_same<T, half>::value) {
    return __half2float(value);
  } else {
    return __bfloat162float(value);
  }
}

template <typename T>
__device__ __forceinline__ T overlap_from_float(float value) {
  if constexpr (std::is_same<T, half>::value) {
    return __float2half_rn(value);
  } else {
    return __float2bfloat16_rn(value);
  }
}

template <typename T>
__global__ void overlap_blend_kernel(
    const T *__restrict__ window_values,
    const int32_t *__restrict__ local_indices,
    const float *__restrict__ weights,
    T *__restrict__ output,
    int windows,
    int window_tokens,
    int global_tokens,
    int channels,
    int64_t value_stride_batch,
    int64_t value_stride_window,
    int64_t value_stride_token) {
  const int global_token = static_cast<int>(blockIdx.x);
  const int batch = static_cast<int>(blockIdx.y);
  extern __shared__ unsigned char shared_bytes[];
  int32_t *shared_local = reinterpret_cast<int32_t *>(shared_bytes);
  float *shared_weight = reinterpret_cast<float *>(shared_local + windows);
  for (int window = threadIdx.x; window < windows; window += blockDim.x) {
    const int map_offset = global_token * windows + window;
    shared_local[window] = local_indices[map_offset];
    shared_weight[window] = weights[map_offset];
  }
  __syncthreads();

  for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
    float accumulated = 0.0f;
#pragma unroll 1
    for (int window = 0; window < windows; ++window) {
      const int local = shared_local[window];
      if (local < 0 || local >= window_tokens) continue;
      const int64_t offset = static_cast<int64_t>(batch) * value_stride_batch +
          static_cast<int64_t>(window) * value_stride_window +
          static_cast<int64_t>(local) * value_stride_token + channel;
      // Match the existing ordered PyTorch fallback exactly: it materializes
      // the FP32 multiply before the FP32 add.  Explicit round-to-nearest
      // intrinsics prevent --use_fast_math from contracting these into an FMA
      // whose last bit could differ at a tile boundary.
      const float weighted = __fmul_rn(
          shared_weight[window], overlap_to_float(window_values[offset]));
      accumulated = __fadd_rn(accumulated, weighted);
    }
    output[(static_cast<int64_t>(batch) * global_tokens + global_token) * channels +
           channel] = overlap_from_float<T>(accumulated);
  }
}

}  // namespace

at::Tensor overlap_blend_cuda(
    at::Tensor window_values,
    at::Tensor local_indices,
    at::Tensor weights) {
  TORCH_CHECK(window_values.is_cuda(), "window_values must be a CUDA tensor");
  TORCH_CHECK(local_indices.is_cuda() && weights.is_cuda(),
              "overlap maps must be CUDA tensors");
  TORCH_CHECK(window_values.device() == local_indices.device() &&
              window_values.device() == weights.device(),
              "overlap tensors must share one CUDA device");
  TORCH_CHECK(window_values.dim() == 4,
              "window_values must be [batch, windows, tokens, channels]");
  TORCH_CHECK(local_indices.dim() == 2 && weights.dim() == 2,
              "overlap maps must be [global_tokens, windows]");
  TORCH_CHECK(local_indices.scalar_type() == at::ScalarType::Int,
              "local_indices must use int32");
  TORCH_CHECK(weights.scalar_type() == at::ScalarType::Float,
              "overlap weights must use float32");
  TORCH_CHECK(window_values.scalar_type() == at::ScalarType::Half ||
              window_values.scalar_type() == at::ScalarType::BFloat16,
              "window_values must use float16 or bfloat16");
  TORCH_CHECK(window_values.stride(3) == 1,
              "window_values channels must be contiguous");
  TORCH_CHECK(local_indices.is_contiguous() && weights.is_contiguous(),
              "overlap maps must be contiguous");
  const int batch = static_cast<int>(window_values.size(0));
  const int windows = static_cast<int>(window_values.size(1));
  const int window_tokens = static_cast<int>(window_values.size(2));
  const int global_tokens = static_cast<int>(local_indices.size(0));
  const int channels = static_cast<int>(window_values.size(3));
  TORCH_CHECK(windows > 0 && windows <= 64,
              "overlap blend supports between 1 and 64 windows");
  TORCH_CHECK(local_indices.size(1) == windows && weights.sizes() == local_indices.sizes(),
              "overlap map window dimensions must match window_values");
  TORCH_CHECK(batch > 0 && window_tokens > 0 && global_tokens > 0 && channels > 0,
              "overlap blend dimensions must be positive");

  auto output = at::empty(
      {batch, global_tokens, channels},
      window_values.options());
  const dim3 grid(global_tokens, batch);
  const size_t shared_memory = static_cast<size_t>(windows) *
      (sizeof(int32_t) + sizeof(float));
  const auto stream = c10::cuda::getCurrentCUDAStream(
      window_values.get_device()).stream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(
      window_values.scalar_type(), T, {
        overlap_blend_kernel<T><<<grid, kThreads, shared_memory, stream>>>(
            reinterpret_cast<const T *>(window_values.data_ptr()),
            local_indices.data_ptr<int32_t>(),
            weights.data_ptr<float>(),
            reinterpret_cast<T *>(output.data_ptr()),
            windows,
            window_tokens,
            global_tokens,
            channels,
            window_values.stride(0),
            window_values.stride(1),
            window_values.stride(2));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
