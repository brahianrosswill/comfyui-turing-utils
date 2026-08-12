/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "torch_compat.h"

#include "dispatch_utils.h"
#include "../utils.cuh"
#include "../reduction_utils.cuh"
#include "numeric_conversion.cuh"
#include "cp_async.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdexcept>
#include <type_traits>

enum class QuantType
{
  kInt8,
  kInt4,
};

template <typename T>
__device__ __forceinline__ float convert_to_float(T val)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  if constexpr (std::is_same<T, half>::value)
  {
    return __half2float(val);
  }
  else if constexpr (std::is_same<T, nv_bfloat16>::value)
  {
    return __bfloat162float(val);
  }
}

template <typename T>
__device__ __forceinline__ T convert_from_float(float val)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  if constexpr (std::is_same<T, half>::value)
  {
    return __float2half_rn(val);
  }
  else if constexpr (std::is_same<T, nv_bfloat16>::value)
  {
    return __float2bfloat16_rn(val);
  }
}

template <uint32_t head_dim, uint32_t BLOCK_SIZE, uint32_t num_pack_per_thread = 1, bool has_sm_scale = false, bool sub_mean = false, typename T>
__global__ void QuantInt8Kernel(T *__restrict__ input, T *__restrict__ mean, int8_t *__restrict__ output, float *__restrict__ scale, float sm_scale, const uint32_t num_tokens,
                            const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                            const uint32_t stride_bz_output, const uint32_t stride_seq_output, const uint32_t stride_h_output,
                            const uint32_t stride_bz_scale, const uint32_t stride_h_scale)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");
  static_assert(num_pack_per_thread > 0, "The number of pack per thread must be greater than 0");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  constexpr uint32_t num_threads_per_token = head_dim / pack_size;

  static_assert(num_threads_per_token <= 32, "The number of threads per token must be less than or equal to warp size");

  T x_val[num_pack_per_thread][8];
  T mean_val[8];
  float x_val_float[num_pack_per_thread][8];
  float mean_val_float[8];

  uint32_t bx = blockIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t thread_base_token = bx * BLOCK_SIZE + thread_id / num_threads_per_token;
  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
  T *mean_ptr_base = mean + batch_id * stride_bz_mean + head_id * stride_h_mean + thread_id % num_threads_per_token * pack_size;
  int8_t *output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + thread_base_token * stride_seq_output + thread_id % num_threads_per_token * pack_size;
  float *scale_ptr_base = scale + batch_id * stride_bz_scale + head_id * stride_h_scale + bx;

  if constexpr (sub_mean)
  {
    *(float4*)(&mean_val[0]) = *(float4*)(mean_ptr_base);
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      mean_val_float[j] = convert_to_float(mean_val[j]);
    }
  }

  constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread;

  // load the data
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    if (thread_base_token + i * iter_stride < num_tokens)
    {
      *(float4*)(&x_val[i][0]) = *(float4*)(input_ptr_base + i * iter_stride * stride_seq_input);
#pragma unroll
      for (uint32_t j = 0; j < 8; j++)
      {
        x_val_float[i][j] = convert_to_float(x_val[i][j]);
      }

      if constexpr (sub_mean)
      {
#pragma unroll
        for (uint32_t j = 0; j < 8; j++)
        {
          x_val_float[i][j] -= mean_val_float[j];
        }
      }

      if constexpr (has_sm_scale)
      {
#pragma unroll
        for (uint32_t j = 0; j < 8; j++)
        {
          x_val_float[i][j] *= sm_scale;
        }
      }
    }
    else
    {
#pragma unroll
      for (uint32_t j = 0; j < 8; j++)
      {
        x_val_float[i][j] = 0.0f;
      }
    }
  }

  float amax_val = 0.0000001f; // prevent from dividing by zero

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      amax_val = fmaxf(amax_val, fabsf(x_val_float[i][j]));
    }
  }

  __shared__ float s_amax;
  const float block_amax_val = vllm::blockReduceMax(amax_val);
  if (thread_id == 0)
  {
    s_amax = block_amax_val;
    scale_ptr_base[0] = s_amax / 127.0f;
  }

  __syncthreads();

  float tmp_scale = 127.0f / s_amax;

  char4 o_val[num_pack_per_thread][2];

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < 2; j += 1)
    {
      o_val[i][j] = make_char4(
        float_to_int8_rn(x_val_float[i][j * 4 + 0] * tmp_scale),
        float_to_int8_rn(x_val_float[i][j * 4 + 1] * tmp_scale),
        float_to_int8_rn(x_val_float[i][j * 4 + 2] * tmp_scale),
        float_to_int8_rn(x_val_float[i][j * 4 + 3] * tmp_scale)
      );
    }
  }

  // int8 result
#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {

    if (thread_base_token + i * iter_stride < num_tokens)
    {
      *reinterpret_cast<float2*>(output_ptr_base + i * iter_stride * stride_seq_output) = *reinterpret_cast<float2*>(&o_val[i][0]);
    }
  }
}

// Fixed randomized orthogonal transform used by the W8A8/Sol quantizer.  Q
// and K share the same diagonal and normalized Walsh-Hadamard transform, so
// exact dot products are invariant while per-tile INT8 ranges become tighter.
// The signs match Comfy-Kitchen's production INT8 attention transform.
__device__ __forceinline__ bool convrot_negative_sign(int channel)
{
  constexpr uint32_t signs[4] = {
      0x1035997bu, 0x8087f5eeu, 0xee2e4e1au, 0x71132418u};
  const uint32_t bit = (signs[channel >> 5] >> (channel & 31)) & 1u;
  return bit == 0u;
}

template <uint32_t head_dim>
__device__ __forceinline__ void attention_hadamard8(float (&values)[8], int dim_pack)
{
#pragma unroll
  for (int channel = 0; channel < 8; ++channel)
  {
    if (convrot_negative_sign(dim_pack * 8 + channel))
      values[channel] = -values[channel];
  }
#pragma unroll
  for (int span = 1; span < 8; span <<= 1)
  {
#pragma unroll
    for (int base = 0; base < 8; base += span * 2)
    {
#pragma unroll
      for (int offset = 0; offset < span; ++offset)
      {
        const float left = values[base + offset];
        const float right = values[base + offset + span];
        values[base + offset] = left + right;
        values[base + offset + span] = left - right;
      }
    }
  }

  constexpr int pack_count = head_dim / 8;
  const int pack_lane = threadIdx.x & (pack_count - 1);
#pragma unroll
  for (int bit = 1; bit < pack_count; bit <<= 1)
  {
#pragma unroll
    for (int channel = 0; channel < 8; ++channel)
    {
      const float other = __shfl_xor_sync(
          0xffffffffu, values[channel], bit, pack_count);
      values[channel] =
          (pack_lane & bit) ? other - values[channel] : values[channel] + other;
    }
  }
  constexpr float normalization = head_dim == 64
      ? 0.125f
      : 0.08838834764831845f;
#pragma unroll
  for (int channel = 0; channel < 8; ++channel)
    values[channel] *= normalization;
}

template <uint32_t head_dim, uint32_t BLOCK_SIZE, uint32_t THREADS_PER_BLOCK,
          bool ROTATE, bool ALLOW_ANCHOR, typename T>
__device__ __forceinline__ void quant_int8_tile(T *__restrict__ input,
                                                int8_t *__restrict__ output,
                                                float *__restrict__ scale,
                                                const uint32_t tile_id,
                                                const uint32_t head_id,
                                                const uint32_t batch_id,
                                                const uint32_t num_tokens,
                                                const uint32_t stride_bz_input,
                                                const uint32_t stride_seq_input,
                                                const uint32_t stride_h_input,
                                                const uint32_t stride_bz_output,
                                                const uint32_t stride_seq_output,
                                                const uint32_t stride_h_output,
                                                const uint32_t stride_bz_scale,
                                                const uint32_t stride_h_scale,
                                                const int anchor_index)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  constexpr uint32_t pack_size = 8;
  constexpr uint32_t num_threads_per_token = head_dim / pack_size;
  constexpr uint32_t packs_per_tile = BLOCK_SIZE * num_threads_per_token;
  constexpr uint32_t num_pack_per_thread = (packs_per_tile + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  const uint32_t thread_id = threadIdx.x;
  const uint32_t base_token = tile_id * BLOCK_SIZE;

  __shared__ T anchor_shared[ALLOW_ANCHOR ? head_dim : 1];
  if constexpr (ALLOW_ANCHOR)
  {
    if (anchor_index >= 0)
    {
      for (uint32_t channel = thread_id; channel < head_dim;
           channel += THREADS_PER_BLOCK)
      {
        anchor_shared[channel] = input[
            batch_id * stride_bz_input + head_id * stride_h_input +
            anchor_index * stride_seq_input + channel];
      }
    }
    __syncthreads();
  }

  float x_val_float[num_pack_per_thread][8];

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    const uint32_t pack_idx = thread_id + i * THREADS_PER_BLOCK;
    const uint32_t token_offset = pack_idx / num_threads_per_token;
    const uint32_t dim_pack = pack_idx % num_threads_per_token;

    if (pack_idx < packs_per_tile && base_token + token_offset < num_tokens)
    {
      T x_val[8];
      T *input_ptr = input + batch_id * stride_bz_input + head_id * stride_h_input +
                     (base_token + token_offset) * stride_seq_input + dim_pack * pack_size;
      *(float4*)(&x_val[0]) = *(float4*)(input_ptr);
#pragma unroll
      for (uint32_t j = 0; j < 8; j++)
      {
        x_val_float[i][j] = convert_to_float(x_val[j]);
      }
      if constexpr (ALLOW_ANCHOR)
      {
        if (anchor_index >= 0)
        {
          T anchor_val[8];
#pragma unroll
          for (uint32_t j = 0; j < 8; j++)
          {
            anchor_val[j] = anchor_shared[dim_pack * pack_size + j];
            x_val_float[i][j] -= convert_to_float(anchor_val[j]);
          }
        }
      }
    }
    else
    {
#pragma unroll
      for (uint32_t j = 0; j < 8; j++)
      {
        x_val_float[i][j] = 0.0f;
      }
    }
    if constexpr (ROTATE)
    {
      attention_hadamard8<head_dim>(x_val_float[i], dim_pack);
    }
  }

  float amax_val = 0.0000001f;
#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      amax_val = fmaxf(amax_val, fabsf(x_val_float[i][j]));
    }
  }

  __shared__ float s_amax;
  const float block_amax_val = vllm::blockReduceMax(amax_val);
  if (thread_id == 0)
  {
    s_amax = block_amax_val;
    scale[batch_id * stride_bz_scale + head_id * stride_h_scale + tile_id] = s_amax / 127.0f;
  }

  __syncthreads();

  const float tmp_scale = 127.0f / s_amax;
  char4 o_val[num_pack_per_thread][2];

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < 2; j++)
    {
      o_val[i][j] = make_char4(
          float_to_int8_rn(x_val_float[i][j * 4 + 0] * tmp_scale),
          float_to_int8_rn(x_val_float[i][j * 4 + 1] * tmp_scale),
          float_to_int8_rn(x_val_float[i][j * 4 + 2] * tmp_scale),
          float_to_int8_rn(x_val_float[i][j * 4 + 3] * tmp_scale));
    }
  }

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    const uint32_t pack_idx = thread_id + i * THREADS_PER_BLOCK;
    const uint32_t token_offset = pack_idx / num_threads_per_token;
    const uint32_t dim_pack = pack_idx % num_threads_per_token;

    if (pack_idx < packs_per_tile && base_token + token_offset < num_tokens)
    {
      int8_t *output_ptr = output + batch_id * stride_bz_output + head_id * stride_h_output +
                           (base_token + token_offset) * stride_seq_output + dim_pack * pack_size;
      *reinterpret_cast<float2*>(output_ptr) = *reinterpret_cast<float2*>(&o_val[i][0]);
    }
  }
}

// Model-independent K stabilization detector adapted from Comfy-Kitchen.
// Nine evenly-spaced keys are sampled per batch/head.  The sampled key nearest
// their mean is used only when centering lowers energy without expanding the
// observed range by more than 12.5%.  A negative index disables subtraction.
constexpr int K_ANCHOR_THREADS = 128;
constexpr int K_ANCHOR_SAMPLES = 9;
constexpr int K_ANCHOR_MAX_CHANNELS = 128;

template <typename T>
__global__ __launch_bounds__(K_ANCHOR_THREADS) void DetectKAnchorKernel(
    const T *__restrict__ key,
    int *__restrict__ anchor_indices,
    const int key_length,
    const int head_dim,
    const int num_kv_heads,
    const int64_t stride_bz,
    const int64_t stride_h,
    const int64_t stride_seq)
{
  const int head_id = blockIdx.x;
  const int batch_id = blockIdx.y;
  const int thread_id = threadIdx.x;
  const int lane_id = thread_id & 31;
  const int warp_id = thread_id >> 5;
  const int64_t bh_offset =
      (int64_t)batch_id * stride_bz + (int64_t)head_id * stride_h;

  __shared__ float samples[K_ANCHOR_SAMPLES * K_ANCHOR_MAX_CHANNELS];
  __shared__ float warp_original_energy[4];
  __shared__ float warp_original_max[4];
  __shared__ float warp_candidate_distance[K_ANCHOR_SAMPLES][4];
  __shared__ float warp_best_energy[4];
  __shared__ float warp_best_max[4];
  __shared__ int selected_candidate;

  for (int index = thread_id; index < K_ANCHOR_SAMPLES * head_dim;
       index += K_ANCHOR_THREADS)
  {
    const int sample = index / head_dim;
    const int channel = index - sample * head_dim;
    const int row = sample * (key_length - 1) / (K_ANCHOR_SAMPLES - 1);
    samples[index] = convert_to_float(
        key[bh_offset + (int64_t)row * stride_seq + channel]);
  }
  __syncthreads();

  float original_energy = 0.0f;
  float original_max = 0.0f;
  float candidate_distance[K_ANCHOR_SAMPLES];
#pragma unroll
  for (int candidate = 0; candidate < K_ANCHOR_SAMPLES; ++candidate)
    candidate_distance[candidate] = 0.0f;

  for (int channel = thread_id; channel < head_dim; channel += K_ANCHOR_THREADS)
  {
    float channel_sum = 0.0f;
#pragma unroll
    for (int sample = 0; sample < K_ANCHOR_SAMPLES; ++sample)
    {
      const float value = samples[sample * head_dim + channel];
      original_energy = fmaf(value, value, original_energy);
      original_max = fmaxf(original_max, fabsf(value));
      channel_sum += value;
    }
#pragma unroll
    for (int candidate = 0; candidate < K_ANCHOR_SAMPLES; ++candidate)
    {
      const float distance =
          K_ANCHOR_SAMPLES * samples[candidate * head_dim + channel] -
          channel_sum;
      candidate_distance[candidate] =
          fmaf(distance, distance, candidate_distance[candidate]);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
  {
    original_energy += __shfl_down_sync(0xffffffffu, original_energy, offset);
    original_max = fmaxf(
        original_max, __shfl_down_sync(0xffffffffu, original_max, offset));
#pragma unroll
    for (int candidate = 0; candidate < K_ANCHOR_SAMPLES; ++candidate)
      candidate_distance[candidate] += __shfl_down_sync(
          0xffffffffu, candidate_distance[candidate], offset);
  }
  if (lane_id == 0)
  {
    warp_original_energy[warp_id] = original_energy;
    warp_original_max[warp_id] = original_max;
#pragma unroll
    for (int candidate = 0; candidate < K_ANCHOR_SAMPLES; ++candidate)
      warp_candidate_distance[candidate][warp_id] =
          candidate_distance[candidate];
  }
  __syncthreads();

  if (thread_id == 0)
  {
    int best_candidate = 0;
    float best_distance = 3.402823466e+38F;
#pragma unroll
    for (int candidate = 0; candidate < K_ANCHOR_SAMPLES; ++candidate)
    {
      float distance = 0.0f;
#pragma unroll
      for (int warp = 0; warp < 4; ++warp)
        distance += warp_candidate_distance[candidate][warp];
      if (distance < best_distance)
      {
        best_candidate = candidate;
        best_distance = distance;
      }
    }
    selected_candidate = best_candidate;
  }
  __syncthreads();

  float best_energy = 0.0f;
  float best_max = 0.0f;
  for (int channel = thread_id; channel < head_dim; channel += K_ANCHOR_THREADS)
  {
    const float anchor = samples[selected_candidate * head_dim + channel];
#pragma unroll
    for (int sample = 0; sample < K_ANCHOR_SAMPLES; ++sample)
    {
      const float residual = samples[sample * head_dim + channel] - anchor;
      best_energy = fmaf(residual, residual, best_energy);
      best_max = fmaxf(best_max, fabsf(residual));
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
  {
    best_energy += __shfl_down_sync(0xffffffffu, best_energy, offset);
    best_max = fmaxf(
        best_max, __shfl_down_sync(0xffffffffu, best_max, offset));
  }
  if (lane_id == 0)
  {
    warp_best_energy[warp_id] = best_energy;
    warp_best_max[warp_id] = best_max;
  }
  __syncthreads();

  if (thread_id == 0)
  {
    float total_original_energy = 0.0f;
    float total_original_max = 0.0f;
    float total_best_energy = 0.0f;
    float total_best_max = 0.0f;
#pragma unroll
    for (int warp = 0; warp < 4; ++warp)
    {
      total_original_energy += warp_original_energy[warp];
      total_original_max = fmaxf(total_original_max, warp_original_max[warp]);
      total_best_energy += warp_best_energy[warp];
      total_best_max = fmaxf(total_best_max, warp_best_max[warp]);
    }
    const bool improves_range =
        total_best_energy < total_original_energy &&
        total_best_max <= total_original_max * 1.125f;
    anchor_indices[batch_id * num_kv_heads + head_id] = improves_range
        ? selected_candidate * (key_length - 1) / (K_ANCHOR_SAMPLES - 1)
        : -1;
  }
}

template <uint32_t head_dim, uint32_t Q_BLOCK_SIZE, uint32_t K_BLOCK_SIZE, uint32_t THREADS_PER_BLOCK, bool ROTATE, typename T>
__global__ void QuantQKInt8Kernel(T *__restrict__ query,
                                  T *__restrict__ key,
                                  int8_t *__restrict__ query_output,
                                  int8_t *__restrict__ key_output,
                                  float *__restrict__ query_scale,
                                  float *__restrict__ key_scale,
                                  const uint32_t qo_len,
                                  const uint32_t kv_len,
                                  const uint32_t num_qo_heads,
                                  const uint32_t num_kv_heads,
                                  const uint32_t query_scale_len,
                                  const uint32_t key_scale_len,
                                  const uint32_t stride_bz_q,
                                  const uint32_t stride_seq_q,
                                  const uint32_t stride_h_q,
                                  const uint32_t stride_bz_k,
                                  const uint32_t stride_seq_k,
                                  const uint32_t stride_h_k,
                                  const uint32_t stride_bz_qo,
                                  const uint32_t stride_seq_qo,
                                  const uint32_t stride_h_qo,
                                  const uint32_t stride_bz_ko,
                                  const uint32_t stride_seq_ko,
                                  const uint32_t stride_h_ko,
                                  const uint32_t stride_bz_q_scale,
                                  const uint32_t stride_h_q_scale,
                                  const uint32_t stride_bz_k_scale,
                                  const uint32_t stride_h_k_scale,
                                  const int *__restrict__ anchor_indices)
{
  const uint32_t task_id = blockIdx.x;
  const uint32_t batch_id = blockIdx.y;
  const uint32_t num_query_tasks = num_qo_heads * query_scale_len;

  if (task_id < num_query_tasks)
  {
    const uint32_t head_id = task_id / query_scale_len;
    const uint32_t tile_id = task_id - head_id * query_scale_len;
    quant_int8_tile<head_dim, Q_BLOCK_SIZE, THREADS_PER_BLOCK, ROTATE, false, T>(
        query, query_output, query_scale, tile_id, head_id, batch_id, qo_len,
        stride_bz_q, stride_seq_q, stride_h_q,
        stride_bz_qo, stride_seq_qo, stride_h_qo,
        stride_bz_q_scale, stride_h_q_scale, -1);
  }
  else
  {
    const uint32_t key_task_id = task_id - num_query_tasks;
    const uint32_t head_id = key_task_id / key_scale_len;
    const uint32_t tile_id = key_task_id - head_id * key_scale_len;
    if (head_id < num_kv_heads)
    {
      const int anchor_index = anchor_indices == nullptr
          ? -1
          : anchor_indices[batch_id * num_kv_heads + head_id];
      quant_int8_tile<head_dim, K_BLOCK_SIZE, THREADS_PER_BLOCK, ROTATE, true, T>(
          key, key_output, key_scale, tile_id, head_id, batch_id, kv_len,
          stride_bz_k, stride_seq_k, stride_h_k,
          stride_bz_ko, stride_seq_ko, stride_h_ko,
          stride_bz_k_scale, stride_h_k_scale, anchor_index);
    }
  }
}

template <uint32_t head_dim, uint32_t TOKEN_BLOCK_SIZE,
          uint32_t num_pack_per_thread, bool Rotate, typename IndexT, typename T>
__global__ void QuantInt8VarlenKernel(T *__restrict__ input,
                                      const IndexT *__restrict__ cu_seqlens,
                                      int8_t *__restrict__ output,
                                      float *__restrict__ scale,
                                      const uint32_t stride_seq_input,
                                      const uint32_t stride_h_input,
                                      const uint32_t stride_seq_output,
                                      const uint32_t stride_h_output,
                                      const uint32_t stride_bz_scale,
                                      const uint32_t stride_h_scale)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");
  static_assert(num_pack_per_thread > 0, "The number of pack per thread must be greater than 0");

  constexpr uint32_t pack_size = 8;
  constexpr uint32_t num_threads_per_token = head_dim / pack_size;
  static_assert(num_threads_per_token <= 32, "The number of threads per token must be less than or equal to warp size");

  const uint32_t quant_block_id = blockIdx.x;
  const uint32_t head_id = blockIdx.y;
  const uint32_t batch_id = blockIdx.z;
  const uint32_t thread_id = threadIdx.x;

  const uint32_t seq_start = static_cast<uint32_t>(cu_seqlens[batch_id]);
  const uint32_t seq_end = static_cast<uint32_t>(cu_seqlens[batch_id + 1]);
  const uint32_t seq_len = seq_end - seq_start;
  const uint32_t thread_base_token = quant_block_id * TOKEN_BLOCK_SIZE + thread_id / num_threads_per_token;

  T *input_ptr_base = input + (seq_start + thread_base_token) * stride_seq_input + head_id * stride_h_input + thread_id % num_threads_per_token * pack_size;
  int8_t *output_ptr_base = output + (seq_start + thread_base_token) * stride_seq_output + head_id * stride_h_output + thread_id % num_threads_per_token * pack_size;
  float *scale_ptr = scale + batch_id * stride_bz_scale + head_id * stride_h_scale + quant_block_id;

  float x_val_float[num_pack_per_thread][8];
  constexpr uint32_t iter_stride = TOKEN_BLOCK_SIZE / num_pack_per_thread;

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    if (thread_base_token + i * iter_stride < seq_len)
    {
      T x_val[8];
      *(float4 *)(&x_val[0]) = *(float4 *)(input_ptr_base + i * iter_stride * stride_seq_input);
#pragma unroll
      for (uint32_t j = 0; j < 8; j++)
      {
        x_val_float[i][j] = convert_to_float(x_val[j]);
      }
    }
    else
    {
#pragma unroll
      for (uint32_t j = 0; j < 8; j++)
      {
        x_val_float[i][j] = 0.0f;
      }
    }
    if constexpr (Rotate)
    {
      attention_hadamard8<head_dim>(
          x_val_float[i], thread_id % num_threads_per_token);
    }
  }

  float amax_val = 0.0000001f;
#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      amax_val = fmaxf(amax_val, fabsf(x_val_float[i][j]));
    }
  }

  __shared__ float s_amax;
  const float block_amax_val = vllm::blockReduceMax(amax_val);
  if (thread_id == 0)
  {
    s_amax = block_amax_val;
    scale_ptr[0] = s_amax / 127.0f;
  }

  __syncthreads();
  float tmp_scale = 127.0f / s_amax;

  char4 o_val[num_pack_per_thread][2];
#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < 2; j++)
    {
      o_val[i][j] = make_char4(
          float_to_int8_rn(x_val_float[i][j * 4 + 0] * tmp_scale),
          float_to_int8_rn(x_val_float[i][j * 4 + 1] * tmp_scale),
          float_to_int8_rn(x_val_float[i][j * 4 + 2] * tmp_scale),
          float_to_int8_rn(x_val_float[i][j * 4 + 3] * tmp_scale));
    }
  }

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    if (thread_base_token + i * iter_stride < seq_len)
    {
      *reinterpret_cast<float2 *>(output_ptr_base + i * iter_stride * stride_seq_output) = *reinterpret_cast<float2 *>(&o_val[i][0]);
    }
  }
}

template <uint32_t head_dim, uint32_t BLOCK_SIZE, uint32_t num_pack_per_thread = 1, typename T>
__global__ void SubMeanKernel(T *__restrict__ input, T *__restrict__ mean, half *__restrict__ output, const uint32_t num_tokens,
                            const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                            const uint32_t stride_bz_output, const uint32_t stride_seq_output, const uint32_t stride_h_output)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");
  static_assert(num_pack_per_thread > 0, "The number of pack per thread must be greater than 0");

  using T2 = typename std::conditional<std::is_same<T, half>::value, half2, nv_bfloat162>::type;

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  constexpr uint32_t num_threads_per_token = head_dim / pack_size;

  static_assert(num_threads_per_token <= 32, "The number of threads per token must be less than or equal to warp size");

  T2 x_val[num_pack_per_thread][4];
  T2 mean_val[4];

  uint32_t bx = blockIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t thread_base_token = bx * BLOCK_SIZE + thread_id / num_threads_per_token;
  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
  T *mean_ptr_base = mean + batch_id * stride_bz_mean + head_id * stride_h_mean + thread_id % num_threads_per_token * pack_size;
  half *output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + thread_base_token * stride_seq_output + thread_id % num_threads_per_token * pack_size;

  *(float4*)(&mean_val[0]) = *(float4*)(mean_ptr_base);

  constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread;

  // load the data
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    if (thread_base_token + i * iter_stride < num_tokens)
    {
      *(float4*)(&x_val[i][0]) = *(float4*)(input_ptr_base + i * iter_stride * stride_seq_input);
#pragma unroll
      for (uint32_t j = 0; j < 4; j++)
      {
        x_val[i][j] = __hsub2(x_val[i][j], mean_val[j]);

        if constexpr (std::is_same<T, nv_bfloat16>::value)
        {
          ((half2*)x_val[i])[j] = __float22half2_rn(__bfloat1622float2(x_val[i][j]));
        }
      }
    }
  }

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    if (thread_base_token + i * iter_stride < num_tokens)
    {
      *reinterpret_cast<float4*>(output_ptr_base + i * iter_stride * stride_seq_output) = *reinterpret_cast<float4*>(&x_val[i][0]);
    }
  }
}

template <uint32_t head_dim, uint32_t CTA_SIZE, bool pad_zero=false, typename T>
__global__ void TransposePadPermuteKernel(T *__restrict__ input, T *__restrict__ output, const uint32_t num_tokens,
                            const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_output, const uint32_t stride_d_output, const uint32_t stride_h_output)
{

  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  uint32_t num_threads_per_token = head_dim / pack_size;
  uint32_t num_threads_per_cta = CTA_SIZE / pack_size;

  uint32_t bx = blockIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t thread_base_token = bx * CTA_SIZE + thread_id / num_threads_per_token;

  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
  T* output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + bx * CTA_SIZE + thread_id % num_threads_per_cta * pack_size + thread_id / num_threads_per_cta * stride_d_output;

  __shared__ T shared_load[CTA_SIZE][head_dim];
  __shared__ T shared_store[head_dim][CTA_SIZE];

  // 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
  // permute on the seq dimension for fp8 mma
  uint32_t smem_load_row_base = ((thread_id / num_threads_per_token) / 16) * 16;
  uint32_t smem_load_row_mod = (thread_id / num_threads_per_token) % 16;
  uint32_t smem_load_row = smem_load_row_base + (smem_load_row_mod  / 8) * 2 + ((smem_load_row_mod / 2) % 4) * 4 + (smem_load_row_mod % 2);

  constexpr cp_async::SharedMemFillMode fill_mode = pad_zero ? cp_async::SharedMemFillMode::kFillZero : cp_async::SharedMemFillMode::kNoFill;
  cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch, fill_mode>(shared_load[smem_load_row] + thread_id % num_threads_per_token * pack_size, input_ptr_base, thread_base_token < num_tokens);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  uint32_t smem_row_base = thread_id % CTA_SIZE;
  uint32_t smem_col_base = thread_id / CTA_SIZE;
  uint32_t smem_col_stride = head_dim / 8;

  // TODO: use ldmatrix to do permutation
#pragma unroll
  for (uint32_t i = 0; i < 8; i++)
  {
    shared_store[smem_col_base + i * smem_col_stride][smem_row_base] = shared_load[smem_row_base][smem_col_base + i * smem_col_stride];
  }

  __syncthreads();

  *(float4*)(output_ptr_base) = *(float4*)(&shared_store[thread_id / num_threads_per_cta][thread_id % num_threads_per_cta * pack_size]);
}


template<uint32_t pad_size, bool sub_mean = false, typename T>
__global__ void MeanScaleKernel(T *__restrict__ input, int8_t *__restrict__ output, float *__restrict__ mean, float *__restrict__ scale, const float scale_max, const uint32_t num_tokens,
                            const uint32_t stride_bz_input, const uint32_t stride_d_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_output, const uint32_t stride_d_output, const uint32_t stride_h_output,
                            const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                            const uint32_t stride_bz_scale, const uint32_t stride_h_scale)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value, "Only half and bfloat16 are supported");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16

  uint32_t head_id = blockIdx.x;
  uint32_t batch_id = blockIdx.y;
  uint32_t d_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t num_threads = blockDim.x;
  uint32_t gmem_stride = num_threads * pack_size;
  // pad the number of tokens to 16 to deal with fp8 permute in previous kernel
  uint32_t fp8_padded_num_tokens = (num_tokens + 15) / 16 * 16;
  uint32_t num_iters = fp8_padded_num_tokens / gmem_stride + ((fp8_padded_num_tokens % gmem_stride) > thread_id * pack_size);

  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + d_id * stride_d_input + thread_id * pack_size;
  int8_t *output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + d_id * stride_d_output + thread_id * pack_size;

  T x_val[8];
  float x_val_float[8];
  uint32_t x_val_fp8[2];

  float max_val = - 1000000.0f;
  float min_val = 1000000.0f;
  float sum_val = 0.0f;

  for (int i = 0; i < num_iters; i++)
  {
    *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i * gmem_stride);
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      float x_temp = convert_to_float(x_val[j]);
      max_val = fmaxf(max_val, x_temp);
      min_val = fminf(min_val, x_temp);

      if constexpr (sub_mean)
      {
        sum_val += x_temp;
      }
    }
  }

  // reduce
  __shared__ float s_amax_val;
  __shared__ float s_mean_val;

  float block_max_val = vllm::blockReduceMax(max_val);
  float block_min_val = vllm::blockReduceMin(min_val);
  float block_sum_val;

  if constexpr (sub_mean)
  {
    block_sum_val = vllm::blockReduceSum(sum_val);
  }

  if (thread_id == 0)
  {
    if constexpr (sub_mean)
    {
      s_mean_val = block_sum_val / fp8_padded_num_tokens;
      s_amax_val = fmaxf(fabsf(block_max_val - s_mean_val), fabsf(block_min_val - s_mean_val));
      mean[batch_id * stride_bz_mean + head_id * stride_h_mean + d_id] = s_mean_val;
    }
    else
    {
      s_amax_val = fmaxf(fabsf(block_max_val), fabsf(block_min_val));
    }

    scale[batch_id * stride_bz_scale + head_id * stride_h_scale + d_id] = s_amax_val / scale_max;
  }

  __syncthreads();

  float mean_val = s_mean_val;
  float recp_scale = scale_max / s_amax_val;

  // recalculate num_iters to cover all fp8 output tokens to prevent nan in random initialization
  uint32_t padded_num_tokens = (num_tokens + pad_size - 1) / pad_size * pad_size;
  num_iters = padded_num_tokens / gmem_stride + ((padded_num_tokens % gmem_stride) > thread_id * pack_size);

  for (int i = 0; i < num_iters; i++)
  {
    *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i * gmem_stride);
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      x_val_float[j] = convert_to_float(x_val[j]);
      if constexpr (sub_mean)
      {
        x_val_float[j] = (x_val_float[j] - mean_val) * recp_scale;
      }
      else
      {
        x_val_float[j] *= recp_scale;
      }
    }

    floatx4_to_e4m3x4(x_val_fp8, x_val_float, x_val_float + 2);
    floatx4_to_e4m3x4(x_val_fp8 + 1, x_val_float + 4, x_val_float + 6);

    *(uint2*)(output_ptr_base + i * gmem_stride) = *(uint2*)(&x_val_fp8[0]);
  }
}


void quant_per_block_int8_cuda(
                at::Tensor input,
                at::Tensor output,
                at::Tensor scale,
                float sm_scale,
                int block_size,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);

  CHECK_DTYPE(output, at::ScalarType::Char);
  CHECK_DTYPE(scale, at::ScalarType::Float);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_seq_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_seq_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  auto input_dtype = input.scalar_type();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {

        CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
        CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);

        dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);

        constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

        dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

        QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, true, false, c_type><<<
            grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<c_type*>(input.data_ptr()),
          nullptr,
          output.data_ptr<int8_t>(),
          reinterpret_cast<float*>(scale.data_ptr()),
          sm_scale,
          num_tokens,
          stride_bz_input, stride_seq_input, stride_h_input,
          0, 0,
          stride_bz_output, stride_seq_output, stride_h_output,
          scale.stride(0), scale.stride(1)
        );
      });
    });
  });
}

void quant_per_block_int8_cuda(
                at::Tensor input,
                at::Tensor output,
                at::Tensor scale,
                int block_size,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);

  CHECK_DTYPE(output, at::ScalarType::Char);
  CHECK_DTYPE(scale, at::ScalarType::Float);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_seq_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_seq_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  auto input_dtype = input.scalar_type();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {

        CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
        CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);

        dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);

        constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

        dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

        QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, false, c_type><<<
            grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<c_type*>(input.data_ptr()),
          nullptr,
          output.data_ptr<int8_t>(),
          reinterpret_cast<float*>(scale.data_ptr()),
          0.0f,
          num_tokens,
          stride_bz_input, stride_seq_input, stride_h_input,
          0, 0,
          stride_bz_output, stride_seq_output, stride_h_output,
          scale.stride(0), scale.stride(1)
        );
      });
    });
  });
}


// use block size 128 and warp_block size 32
void quant_per_warp_int8_cuda(
                at::Tensor input,
                at::Tensor output,
                at::Tensor scale,
                int block_size,
                int warp_block_size,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);

  CHECK_DTYPE(output, at::ScalarType::Char);
  CHECK_DTYPE(scale, at::ScalarType::Float);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_seq_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_seq_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  auto input_dtype = input.scalar_type();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_BLOCK_SIZE, {
        DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {

          CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
          CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE));

          dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE), num_heads, batch_size);

          constexpr int num_pack_per_thread = (WARP_BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

          dim3 block(WARP_BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

          QuantInt8Kernel<HEAD_DIM, WARP_BLOCK_SIZE, num_pack_per_thread, false, false, c_type><<<
              grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<c_type*>(input.data_ptr()),
            nullptr,
            output.data_ptr<int8_t>(),
            reinterpret_cast<float*>(scale.data_ptr()),
            0.0,
            num_tokens,
            stride_bz_input, stride_seq_input, stride_h_input,
            0, 0,
            stride_bz_output, stride_seq_output, stride_h_output,
            scale.stride(0), scale.stride(1)
          );
        });
      });
    });
  });
}

static void quant_qk_per_warp_int8_cuda_impl(
                at::Tensor query,
                at::Tensor key,
                at::Tensor query_output,
                at::Tensor key_output,
                at::Tensor query_scale,
                at::Tensor key_scale,
                int query_block_size,
                int query_warp_block_size,
                int key_block_size,
                int tensor_layout,
                bool rotate,
                const at::Tensor &anchor_indices)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(query_output);
  CHECK_CUDA(key_output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);

  CHECK_DTYPE(query_output, at::ScalarType::Char);
  CHECK_DTYPE(key_output, at::ScalarType::Char);
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);

  CHECK_LASTDIM_CONTIGUOUS(query);
  CHECK_LASTDIM_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(query_output);
  CHECK_CONTIGUOUS(key_output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(query_output, 4);
  CHECK_DIMS(key_output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);

  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "Query and key must have the same data type");
  TORCH_CHECK(query.size(0) == key.size(0), "Query and key batch size must match");
  TORCH_CHECK(query.size(3) == key.size(3), "Query and key head_dim must match");

  const int batch_size = query.size(0);
  const int head_dim = query.size(3);

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_seq_k, stride_seq_qo, stride_seq_ko;
  int stride_h_q, stride_h_k, stride_h_qo, stride_h_ko;

  if (tensor_layout == 0)
  {
    qo_len = query.size(1);
    kv_len = key.size(1);
    num_qo_heads = query.size(2);
    num_kv_heads = key.size(2);
    stride_seq_q = query.stride(1);
    stride_seq_k = key.stride(1);
    stride_seq_qo = query_output.stride(1);
    stride_seq_ko = key_output.stride(1);
    stride_h_q = query.stride(2);
    stride_h_k = key.stride(2);
    stride_h_qo = query_output.stride(2);
    stride_h_ko = key_output.stride(2);
  }
  else if (tensor_layout == 1)
  {
    qo_len = query.size(2);
    kv_len = key.size(2);
    num_qo_heads = query.size(1);
    num_kv_heads = key.size(1);
    stride_seq_q = query.stride(2);
    stride_seq_k = key.stride(2);
    stride_seq_qo = query_output.stride(2);
    stride_seq_ko = key_output.stride(2);
    stride_h_q = query.stride(1);
    stride_h_k = key.stride(1);
    stride_h_qo = query_output.stride(1);
    stride_h_ko = key_output.stride(1);
  }
  else
  {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }

  const int query_scale_len = ((qo_len + query_block_size - 1) / query_block_size) * (query_block_size / query_warp_block_size);
  const int key_scale_len = (kv_len + key_block_size - 1) / key_block_size;

  int *anchor_ptr = nullptr;
  if (anchor_indices.defined())
  {
    CHECK_CUDA(anchor_indices);
    CHECK_CONTIGUOUS(anchor_indices);
    CHECK_DTYPE(anchor_indices, at::ScalarType::Int);
    CHECK_DIMS(anchor_indices, 2);
    CHECK_SHAPE(anchor_indices, batch_size, num_kv_heads);
    TORCH_CHECK(
        anchor_indices.device() == key.device(),
        "K anchor scratch must be on the same CUDA device as K");
    anchor_ptr = anchor_indices.data_ptr<int>();
  }

  CHECK_SHAPE(query_output, query.size(0), query.size(1), query.size(2), query.size(3));
  CHECK_SHAPE(key_output, key.size(0), key.size(1), key.size(2), key.size(3));
  CHECK_SHAPE(query_scale, batch_size, num_qo_heads, query_scale_len);
  CHECK_SHAPE(key_scale, batch_size, num_kv_heads, key_scale_len);

  auto input_dtype = query.scalar_type();

#define LAUNCH_QK(ROTATE)                                                      \
  QuantQKInt8Kernel<HEAD_DIM, QUERY_WARP_BLOCK_SIZE, KEY_BLOCK_SIZE,           \
                    THREADS_PER_BLOCK, ROTATE, c_type><<<                      \
      grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(                    \
    reinterpret_cast<c_type*>(query.data_ptr()),                               \
    reinterpret_cast<c_type*>(key.data_ptr()),                                 \
    query_output.data_ptr<int8_t>(),                                           \
    key_output.data_ptr<int8_t>(),                                             \
    reinterpret_cast<float*>(query_scale.data_ptr()),                          \
    reinterpret_cast<float*>(key_scale.data_ptr()),                            \
    qo_len, kv_len, num_qo_heads, num_kv_heads,                                \
    query_scale_len, key_scale_len,                                            \
    query.stride(0), stride_seq_q, stride_h_q,                                 \
    key.stride(0), stride_seq_k, stride_h_k,                                   \
    query_output.stride(0), stride_seq_qo, stride_h_qo,                        \
    key_output.stride(0), stride_seq_ko, stride_h_ko,                          \
    query_scale.stride(0), query_scale.stride(1),                              \
    key_scale.stride(0), key_scale.stride(1),                                \
    anchor_ptr);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_WARP_BLOCK_SIZE(query_warp_block_size, QUERY_WARP_BLOCK_SIZE, {
      DISPATCH_BLOCK_SIZE(key_block_size, KEY_BLOCK_SIZE, {
        DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
          constexpr int THREADS_PER_BLOCK = 256;
          dim3 grid(num_qo_heads * query_scale_len + num_kv_heads * key_scale_len, batch_size);
          dim3 block(THREADS_PER_BLOCK);

          if (anchor_ptr != nullptr)
          {
            dim3 anchor_grid(num_kv_heads, batch_size);
            DetectKAnchorKernel<c_type><<<
                anchor_grid, K_ANCHOR_THREADS, 0,
                c10::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<c_type*>(key.data_ptr()), anchor_ptr,
                kv_len, HEAD_DIM, num_kv_heads,
                key.stride(0), stride_h_k, stride_seq_k);
          }

          if (rotate)
          {
            LAUNCH_QK(true)
          }
          else
          {
            LAUNCH_QK(false)
          }
        });
      });
    });
  });
#undef LAUNCH_QK
}

void quant_qk_per_warp_int8_cuda(
                at::Tensor query,
                at::Tensor key,
                at::Tensor query_output,
                at::Tensor key_output,
                at::Tensor query_scale,
                at::Tensor key_scale,
                int query_block_size,
                int query_warp_block_size,
                int key_block_size,
                int tensor_layout)
{
  quant_qk_per_warp_int8_cuda_impl(
      query, key, query_output, key_output, query_scale, key_scale,
      query_block_size, query_warp_block_size, key_block_size, tensor_layout,
      false, at::Tensor());
}

void quant_qk_per_warp_int8_rotated_cuda(
                at::Tensor query,
                at::Tensor key,
                at::Tensor query_output,
                at::Tensor key_output,
                at::Tensor query_scale,
                at::Tensor key_scale,
                int query_block_size,
                int query_warp_block_size,
                int key_block_size,
                int tensor_layout)
{
  quant_qk_per_warp_int8_cuda_impl(
      query, key, query_output, key_output, query_scale, key_scale,
      query_block_size, query_warp_block_size, key_block_size, tensor_layout,
      true, at::Tensor());
}

void quant_qk_per_warp_int8_rotated_anchored_cuda(
                at::Tensor query,
                at::Tensor key,
                at::Tensor query_output,
                at::Tensor key_output,
                at::Tensor query_scale,
                at::Tensor key_scale,
                at::Tensor anchor_indices,
                int query_block_size,
                int query_warp_block_size,
                int key_block_size,
                int tensor_layout)
{
  quant_qk_per_warp_int8_cuda_impl(
      query, key, query_output, key_output, query_scale, key_scale,
      query_block_size, query_warp_block_size, key_block_size, tensor_layout,
      true, anchor_indices);
}

void quant_per_warp_int8_varlen_cuda(
                at::Tensor input,
                at::Tensor cu_seqlens,
                at::Tensor output,
                at::Tensor scale,
                int max_seqlen,
                int block_size,
                int warp_block_size,
                bool rotate)
{
  CHECK_CUDA(input);
  CHECK_CUDA(cu_seqlens);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);

  CHECK_DTYPE(output, at::ScalarType::Char);
  CHECK_DTYPE(scale, at::ScalarType::Float);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(cu_seqlens);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 3);
  CHECK_DIMS(output, 3);
  CHECK_DIMS(scale, 3);
  CHECK_DIMS(cu_seqlens, 1);

  const int batch_size = cu_seqlens.size(0) - 1;
  const int total_tokens = input.size(0);
  const int num_heads = input.size(1);
  const int head_dim = input.size(2);

  CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2));
  CHECK_SHAPE(scale, batch_size, num_heads, (max_seqlen + block_size - 1) / block_size * (block_size / warp_block_size));

  const int stride_seq_input = input.stride(0);
  const int stride_h_input = input.stride(1);
  const int stride_seq_output = output.stride(0);
  const int stride_h_output = output.stride(1);

  auto input_dtype = input.scalar_type();
  auto index_dtype = cu_seqlens.scalar_type();

#define LAUNCH_VARLEN_QUANT(INDEX_TYPE, ROTATE)                              \
  QuantInt8VarlenKernel<HEAD_DIM, WARP_BLOCK_SIZE,                          \
      num_pack_per_thread, ROTATE, INDEX_TYPE, c_type><<<                   \
      grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(                 \
      reinterpret_cast<c_type *>(input.data_ptr()),                         \
      reinterpret_cast<INDEX_TYPE *>(cu_seqlens.data_ptr()),                \
      output.data_ptr<int8_t>(), scale.data_ptr<float>(),                    \
      stride_seq_input, stride_h_input,                                      \
      stride_seq_output, stride_h_output,                                    \
      scale.stride(0), scale.stride(1));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_BLOCK_SIZE, {
        DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
          constexpr int num_pack_per_thread = (WARP_BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;
          dim3 grid((max_seqlen + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE), num_heads, batch_size);
          dim3 block(WARP_BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

          if (index_dtype == at::ScalarType::Int)
          {
            if (rotate)
            {
              LAUNCH_VARLEN_QUANT(int32_t, true)
            }
            else
            {
              LAUNCH_VARLEN_QUANT(int32_t, false)
            }
          }
          else if (index_dtype == at::ScalarType::Long)
          {
            if (rotate)
            {
              LAUNCH_VARLEN_QUANT(int64_t, true)
            }
            else
            {
              LAUNCH_VARLEN_QUANT(int64_t, false)
            }
          }
          else
          {
            TORCH_CHECK(false, "cu_seqlens must be int32 or int64");
          }
        });
      });
    });
  });
#undef LAUNCH_VARLEN_QUANT

  (void)total_tokens;
}


void transpose_pad_permute_cuda(
                at::Tensor input,
                at::Tensor output,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);

  constexpr int CTA_SIZE = 64;

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_tokens, padded_num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_d_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_d_output = output.stride(1);
    stride_h_output = output.stride(2);

    padded_num_tokens = (num_tokens + CTA_SIZE - 1) / CTA_SIZE * CTA_SIZE;

    CHECK_SHAPE(output, batch_size, head_dim, num_heads, padded_num_tokens);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_d_output = output.stride(2);
    stride_h_output = output.stride(1);

    padded_num_tokens = (num_tokens + CTA_SIZE - 1) / CTA_SIZE * CTA_SIZE;
    CHECK_SHAPE(output, batch_size, num_heads, head_dim, padded_num_tokens);
  }

  auto input_dtype = input.scalar_type();
  auto output_dtype = output.scalar_type();

  TORCH_CHECK(input_dtype == output_dtype, "Input and output must have the same data type");

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 grid(padded_num_tokens / CTA_SIZE, num_heads, batch_size);

      static_assert(CTA_SIZE * HEAD_DIM <= 8192);

      dim3 block(CTA_SIZE * (HEAD_DIM / 8));

      TransposePadPermuteKernel<HEAD_DIM, CTA_SIZE, true, c_type><<<
          grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<c_type*>(input.data_ptr()),
        reinterpret_cast<c_type*>(output.data_ptr()),
        num_tokens,
        stride_bz_input, stride_seq_input, stride_h_input,
        stride_bz_output, stride_d_output, stride_h_output
      );
    });
  });
}

void scale_fuse_quant_cuda(
                at::Tensor input,
                at::Tensor output,
                at::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);

  // CHECK_DTYPE(output, at::ScalarType::Char);
  CHECK_DTYPE(scale, at::ScalarType::Float);

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.size(0);
  const int num_tokens_padded = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_heads, head_dim;
  int stride_d_input, stride_h_input, stride_d_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_heads = input.size(2);
    head_dim = input.size(1);
    stride_d_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_d_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_heads = input.size(1);
    head_dim = input.size(2);
    stride_d_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_d_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
  CHECK_SHAPE(scale, batch_size, num_heads, head_dim);

  constexpr int CTA_SIZE = 256;

  dim3 grid(num_heads, batch_size, head_dim);
  dim3 block(CTA_SIZE);

  auto input_dtype = input.scalar_type();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    MeanScaleKernel<64, false, c_type><<<
        grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<c_type*>(input.data_ptr()),
      reinterpret_cast<int8_t*>(output.data_ptr()),
      nullptr,
      reinterpret_cast<float*>(scale.data_ptr()),
      scale_max,
      num_tokens,
      stride_bz_input, stride_d_input, stride_h_input,
      stride_bz_output, stride_d_output, stride_h_output,
      0, 0,
      scale.stride(0), scale.stride(1)
    );
  });
}

void mean_scale_fuse_quant_cuda(
                at::Tensor input,
                at::Tensor output,
                at::Tensor mean,
                at::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(mean);
  CHECK_CUDA(scale);

  // CHECK_DTYPE(output, at::ScalarType::Char);
  CHECK_DTYPE(mean, at::ScalarType::Float);
  CHECK_DTYPE(scale, at::ScalarType::Float);

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(mean, 3);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.size(0);
  const int num_tokens_padded = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_heads, head_dim;
  int stride_d_input, stride_h_input, stride_d_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_heads = input.size(2);
    head_dim = input.size(1);
    stride_d_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_d_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_heads = input.size(1);
    head_dim = input.size(2);
    stride_d_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_d_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
  CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
  CHECK_SHAPE(scale, batch_size, num_heads, head_dim);

  constexpr int CTA_SIZE = 256;

  dim3 grid(num_heads, batch_size, head_dim);
  dim3 block(CTA_SIZE);

  auto input_dtype = input.scalar_type();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    MeanScaleKernel<64, true, c_type><<<
        grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<c_type*>(input.data_ptr()),
      reinterpret_cast<int8_t*>(output.data_ptr()),
      reinterpret_cast<float*>(mean.data_ptr()),
      reinterpret_cast<float*>(scale.data_ptr()),
      scale_max,
      num_tokens,
      stride_bz_input, stride_d_input, stride_h_input,
      stride_bz_output, stride_d_output, stride_h_output,
      mean.stride(0), mean.stride(1),
      scale.stride(0), scale.stride(1)
    );
  });
}

template <typename T, typename IndexT, uint32_t head_dim, bool is_causal>
__global__ void VarlenAttentionFwdDirectKernel(
    const T *__restrict__ query,
    const T *__restrict__ key,
    const T *__restrict__ value,
    const IndexT *__restrict__ cu_seqlens_q,
    const IndexT *__restrict__ cu_seqlens_k,
    T *__restrict__ output,
    const int num_q_heads,
    const int num_kv_heads,
    const float sm_scale)
{
  constexpr uint32_t rows_per_block = 8;
  constexpr uint32_t d_per_lane = (head_dim + 31) / 32;

  const int lane_id = threadIdx.x;
  const int row_id = threadIdx.y;
  const int q_pos = blockIdx.x * rows_per_block + row_id;
  const int q_head = blockIdx.y;
  const int batch_id = blockIdx.z;
  const int q_start = static_cast<int>(cu_seqlens_q[batch_id]);
  const int q_end = static_cast<int>(cu_seqlens_q[batch_id + 1]);
  const int k_start = static_cast<int>(cu_seqlens_k[batch_id]);
  const int k_end = static_cast<int>(cu_seqlens_k[batch_id + 1]);
  const int q_len = q_end - q_start;
  const int k_len = k_end - k_start;

  if (q_pos >= q_len)
  {
    return;
  }

  const int kv_group_size = num_q_heads / num_kv_heads;
  const int kv_head = q_head / kv_group_size;
  const T *q_ptr = query + (q_start + q_pos) * num_q_heads * head_dim + q_head * head_dim;
  T *out_ptr = output + (q_start + q_pos) * num_q_heads * head_dim + q_head * head_dim;

  float q_frag[d_per_lane];
  float o_frag[d_per_lane];
#pragma unroll
  for (uint32_t i = 0; i < d_per_lane; ++i)
  {
    const uint32_t d = lane_id + i * 32;
    q_frag[i] = (d < head_dim) ? convert_to_float(q_ptr[d]) : 0.0f;
    o_frag[i] = 0.0f;
  }

  constexpr float NEG_INF = -1.0e20f;
  float row_max = NEG_INF;
  float row_sum = 0.0f;
  int k_limit = k_len;
  if constexpr (is_causal)
  {
    k_limit = min(k_len, q_pos + 1);
  }

  for (int k_pos = 0; k_pos < k_limit; ++k_pos)
  {
    const T *k_ptr = key + (k_start + k_pos) * num_kv_heads * head_dim + kv_head * head_dim;
    const T *v_ptr = value + (k_start + k_pos) * num_kv_heads * head_dim + kv_head * head_dim;

    float score = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < d_per_lane; ++i)
    {
      const uint32_t d = lane_id + i * 32;
      if (d < head_dim)
      {
        score += q_frag[i] * convert_to_float(k_ptr[d]);
      }
    }

    score = vllm::warpReduceSum(score) * sm_scale;

    const float new_row_max = fmaxf(row_max, score);
    const float alpha = (row_max > NEG_INF * 0.5f) ? expf(row_max - new_row_max) : 0.0f;
    const float beta = expf(score - new_row_max);
    row_sum = row_sum * alpha + beta;
#pragma unroll
    for (uint32_t i = 0; i < d_per_lane; ++i)
    {
      const uint32_t d = lane_id + i * 32;
      if (d < head_dim)
      {
        o_frag[i] = o_frag[i] * alpha + beta * convert_to_float(v_ptr[d]);
      }
    }
    row_max = new_row_max;
  }

  if (row_sum <= 0.0f || row_max <= NEG_INF * 0.5f)
  {
#pragma unroll
    for (uint32_t i = 0; i < d_per_lane; ++i)
    {
      const uint32_t d = lane_id + i * 32;
      if (d < head_dim)
      {
        out_ptr[d] = convert_from_float<T>(0.0f);
      }
    }
    return;
  }

#pragma unroll
  for (uint32_t i = 0; i < d_per_lane; ++i)
  {
    const uint32_t d = lane_id + i * 32;
    if (d < head_dim)
    {
      out_ptr[d] = convert_from_float<T>(o_frag[i] / row_sum);
    }
  }
}

template <typename T, typename IndexT, uint32_t head_dim, bool is_causal>
__global__ void VarlenAttentionFwdKernel(
    const T *__restrict__ query,
    const T *__restrict__ key,
    const T *__restrict__ value,
    const IndexT *__restrict__ cu_seqlens_q,
    const IndexT *__restrict__ cu_seqlens_k,
    T *__restrict__ output,
    const int num_q_heads,
    const int num_kv_heads,
    const float sm_scale)
{
  constexpr uint32_t rows_per_block = 8;
  constexpr uint32_t block_n = 32;
  constexpr uint32_t d_per_lane = (head_dim + 31) / 32;

  const int lane_id = threadIdx.x;
  const int row_id = threadIdx.y;
  const int tid = row_id * 32 + lane_id;
  const int num_threads = 32 * rows_per_block;
  const int q_pos = blockIdx.x * rows_per_block + row_id;
  const int q_head = blockIdx.y;
  const int batch_id = blockIdx.z;
  const int q_start = static_cast<int>(cu_seqlens_q[batch_id]);
  const int q_end = static_cast<int>(cu_seqlens_q[batch_id + 1]);
  const int k_start = static_cast<int>(cu_seqlens_k[batch_id]);
  const int k_end = static_cast<int>(cu_seqlens_k[batch_id + 1]);
  const int q_len = q_end - q_start;
  const int k_len = k_end - k_start;

  const bool active = q_pos < q_len;

  const int kv_group_size = num_q_heads / num_kv_heads;
  const int kv_head = q_head / kv_group_size;
  const T *q_ptr = query + (q_start + q_pos) * num_q_heads * head_dim + q_head * head_dim;
  T *out_ptr = output + (q_start + q_pos) * num_q_heads * head_dim + q_head * head_dim;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  T *smem_k = reinterpret_cast<T *>(smem_raw);
  T *smem_v = smem_k + block_n * head_dim;

  float q_frag[d_per_lane];
  float o_frag[d_per_lane];
#pragma unroll
  for (uint32_t i = 0; i < d_per_lane; ++i)
  {
    const uint32_t d = lane_id + i * 32;
    q_frag[i] = (active && d < head_dim) ? convert_to_float(q_ptr[d]) : 0.0f;
    o_frag[i] = 0.0f;
  }

  constexpr float NEG_INF = -1.0e20f;
  float row_max = NEG_INF;
  float row_sum = 0.0f;

  int k_loop_len = k_len;
  if constexpr (is_causal)
  {
    k_loop_len = min(k_len, static_cast<int>((blockIdx.x + 1) * rows_per_block));
  }

  for (int tile_start = 0; tile_start < k_loop_len; tile_start += block_n)
  {
    const int tile_count = min(static_cast<int>(block_n), k_loop_len - tile_start);
    for (uint32_t idx = tid; idx < block_n * head_dim; idx += num_threads)
    {
      const int n = idx / head_dim;
      const int d = idx - n * head_dim;
      if (n < tile_count)
      {
        const T *k_ptr = key + (k_start + tile_start + n) * num_kv_heads * head_dim + kv_head * head_dim;
        const T *v_ptr = value + (k_start + tile_start + n) * num_kv_heads * head_dim + kv_head * head_dim;
        smem_k[idx] = k_ptr[d];
        smem_v[idx] = v_ptr[d];
      }
      else
      {
        smem_k[idx] = convert_from_float<T>(0.0f);
        smem_v[idx] = convert_from_float<T>(0.0f);
      }
    }
    __syncthreads();

    if (active)
    {
      int tile_valid = tile_count;
      if constexpr (is_causal)
      {
        tile_valid = min(tile_valid, q_pos - tile_start + 1);
        if (tile_valid < 0)
        {
          tile_valid = 0;
        }
      }

      for (int local_n = 0; local_n < tile_valid; ++local_n)
      {
        T *k_ptr = smem_k + local_n * head_dim;
        T *v_ptr = smem_v + local_n * head_dim;

        float score = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < d_per_lane; ++i)
        {
          const uint32_t d = lane_id + i * 32;
          if (d < head_dim)
          {
            score += q_frag[i] * convert_to_float(k_ptr[d]);
          }
        }

        score = vllm::warpReduceSum(score) * sm_scale;

        const float new_row_max = fmaxf(row_max, score);
        const float alpha = (row_max > NEG_INF * 0.5f) ? expf(row_max - new_row_max) : 0.0f;
        const float beta = expf(score - new_row_max);
        row_sum = row_sum * alpha + beta;
#pragma unroll
        for (uint32_t i = 0; i < d_per_lane; ++i)
        {
          const uint32_t d = lane_id + i * 32;
          if (d < head_dim)
          {
            o_frag[i] = o_frag[i] * alpha + beta * convert_to_float(v_ptr[d]);
          }
        }
        row_max = new_row_max;
      }
    }
    __syncthreads();
  }

  if (!active)
  {
    return;
  }

  if (row_sum <= 0.0f || row_max <= NEG_INF * 0.5f)
  {
#pragma unroll
    for (uint32_t i = 0; i < d_per_lane; ++i)
    {
      const uint32_t d = lane_id + i * 32;
      if (d < head_dim)
      {
        out_ptr[d] = convert_from_float<T>(0.0f);
      }
    }
    return;
  }

#pragma unroll
  for (uint32_t i = 0; i < d_per_lane; ++i)
  {
    const uint32_t d = lane_id + i * 32;
    if (d < head_dim)
    {
      out_ptr[d] = convert_from_float<T>(o_frag[i] / row_sum);
    }
  }
}

template <typename T, typename IndexT, uint32_t head_dim>
void launch_varlen_attention_fwd(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor cu_seqlens_q,
    at::Tensor cu_seqlens_k,
    at::Tensor output,
    int max_seqlen_q,
    float sm_scale,
    int is_causal)
{
  const int batch_size = cu_seqlens_q.size(0) - 1;
  const int num_q_heads = query.size(1);
  const int num_kv_heads = key.size(1);

  constexpr uint32_t rows_per_block = 8;
  dim3 grid((max_seqlen_q + rows_per_block - 1) / rows_per_block, num_q_heads, batch_size);
  dim3 block(32, rows_per_block);
  constexpr uint32_t block_n = 32;
  constexpr size_t smem_size = 2 * block_n * head_dim * sizeof(T);
  static_assert(smem_size <= 48 * 1024, "SM75 varlen attention must stay within 48 KiB shared memory");
  if (is_causal)
  {
    VarlenAttentionFwdKernel<T, IndexT, head_dim, true><<<grid, block, smem_size, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<T *>(query.data_ptr()),
        reinterpret_cast<T *>(key.data_ptr()),
        reinterpret_cast<T *>(value.data_ptr()),
        reinterpret_cast<IndexT *>(cu_seqlens_q.data_ptr()),
        reinterpret_cast<IndexT *>(cu_seqlens_k.data_ptr()),
        reinterpret_cast<T *>(output.data_ptr()),
        num_q_heads,
        num_kv_heads,
        sm_scale);
  }
  else
  {
    VarlenAttentionFwdDirectKernel<T, IndexT, head_dim, false><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<T *>(query.data_ptr()),
        reinterpret_cast<T *>(key.data_ptr()),
        reinterpret_cast<T *>(value.data_ptr()),
        reinterpret_cast<IndexT *>(cu_seqlens_q.data_ptr()),
        reinterpret_cast<IndexT *>(cu_seqlens_k.data_ptr()),
        reinterpret_cast<T *>(output.data_ptr()),
        num_q_heads,
        num_kv_heads,
        sm_scale);
  }
}

void varlen_attention_fwd_cuda(
                at::Tensor query,
                at::Tensor key,
                at::Tensor value,
                at::Tensor cu_seqlens_q,
                at::Tensor cu_seqlens_k,
                at::Tensor output,
                int max_seqlen_q,
                float sm_scale,
                int is_causal)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(cu_seqlens_q);
  CHECK_CUDA(cu_seqlens_k);
  CHECK_CUDA(output);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value);
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);
  CHECK_CONTIGUOUS(output);

  CHECK_DIMS(query, 3);
  CHECK_DIMS(key, 3);
  CHECK_DIMS(value, 3);
  CHECK_DIMS(output, 3);
  CHECK_DIMS(cu_seqlens_q, 1);
  CHECK_DIMS(cu_seqlens_k, 1);

  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query and key must have the same dtype");
  TORCH_CHECK(query.scalar_type() == value.scalar_type(), "query and value must have the same dtype");
  TORCH_CHECK(query.scalar_type() == output.scalar_type(), "query and output must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == cu_seqlens_k.scalar_type(), "cu_seqlens_q and cu_seqlens_k must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == at::ScalarType::Int || cu_seqlens_q.scalar_type() == at::ScalarType::Long,
              "cu_seqlens tensors must have dtype int32 or int64");

  const int total_q = query.size(0);
  const int total_k = key.size(0);
  const int num_q_heads = query.size(1);
  const int num_kv_heads = key.size(1);
  const int head_dim = query.size(2);

  TORCH_CHECK(key.size(2) == head_dim, "query and key must have the same head_dim");
  TORCH_CHECK(value.size(0) == total_k, "value token count must match key token count");
  TORCH_CHECK(value.size(1) == num_kv_heads, "value head count must match key head count");
  TORCH_CHECK(value.size(2) == head_dim, "value head_dim must match key head_dim");
  CHECK_SHAPE(output, total_q, num_q_heads, head_dim);
  TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads");
  TORCH_CHECK(max_seqlen_q > 0, "max_seqlen_q must be positive");

  auto input_dtype = query.scalar_type();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      if (cu_seqlens_q.scalar_type() == at::ScalarType::Int)
      {
        launch_varlen_attention_fwd<c_type, int32_t, HEAD_DIM>(
            query, key, value, cu_seqlens_q, cu_seqlens_k, output, max_seqlen_q, sm_scale, is_causal);
      }
      else
      {
        launch_varlen_attention_fwd<c_type, int64_t, HEAD_DIM>(
            query, key, value, cu_seqlens_q, cu_seqlens_k, output, max_seqlen_q, sm_scale, is_causal);
      }
    });
  });
}
