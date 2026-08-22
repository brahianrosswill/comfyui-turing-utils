/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Turing Utils contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fused.h"
#include "dispatch_utils.h"
#include "../reduction_utils.cuh"
#include "../utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace {

constexpr int kThreads = 256;
constexpr int kAnchorThreads = 128;
constexpr int kAnchorSamples = 9;
constexpr int kMaxHeadDim = 128;

template <typename T>
__device__ __forceinline__ float to_float(T value) {
  if constexpr (std::is_same<T, half>::value) {
    return __half2float(value);
  } else {
    return __bfloat162float(value);
  }
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
  if constexpr (std::is_same<T, half>::value) {
    return __float2half_rn(value);
  } else {
    return __float2bfloat16_rn(value);
  }
}

__device__ __forceinline__ int8_t quantize_int8(float value) {
  value = nearbyintf(value);
  value = fminf(127.0f, fmaxf(-127.0f, value));
  return static_cast<int8_t>(value);
}

__device__ __forceinline__ bool convrot_negative_sign(int channel) {
  constexpr uint32_t signs[4] = {
      0x1035997bu, 0x8087f5eeu, 0xee2e4e1au, 0x71132418u};
  return ((signs[channel >> 5] >> (channel & 31)) & 1u) == 0u;
}

template <int HeadDim>
__device__ __forceinline__ void hadamard(float (&values)[8], int pack) {
#pragma unroll
  for (int channel = 0; channel < 8; ++channel) {
    if (convrot_negative_sign(pack * 8 + channel)) {
      values[channel] = -values[channel];
    }
  }
#pragma unroll
  for (int span = 1; span < 8; span <<= 1) {
#pragma unroll
    for (int base = 0; base < 8; base += span * 2) {
#pragma unroll
      for (int offset = 0; offset < span; ++offset) {
        const float left = values[base + offset];
        const float right = values[base + offset + span];
        values[base + offset] = left + right;
        values[base + offset + span] = left - right;
      }
    }
  }
  constexpr int packs = HeadDim / 8;
  const int pack_lane = threadIdx.x & (packs - 1);
#pragma unroll
  for (int bit = 1; bit < packs; bit <<= 1) {
#pragma unroll
    for (int channel = 0; channel < 8; ++channel) {
      const float other = __shfl_xor_sync(0xffffffffu, values[channel], bit, packs);
      values[channel] = (pack_lane & bit)
          ? other - values[channel]
          : values[channel] + other;
    }
  }
  constexpr float scale = HeadDim == 64 ? 0.125f : 0.08838834764831845f;
#pragma unroll
  for (int channel = 0; channel < 8; ++channel) {
    values[channel] *= scale;
  }
}

template <typename T, int HeadDim, bool SplitHalf>
__device__ __forceinline__ float processed_value(
    const T *__restrict__ input,
    const T *__restrict__ norm,
    const T *__restrict__ freqs,
    int batch,
    int head,
    int token,
    int channel,
    float rrms,
    int heads,
    int sequence,
    int rot_dim,
    int64_t stride_batch,
    int64_t stride_head,
    int64_t stride_sequence,
    int64_t freq_stride_batch,
    int64_t freq_stride_sequence,
    int64_t freq_stride_pair,
    int64_t freq_stride_row,
    int64_t freq_stride_col,
    int freq_batches,
    int freq_sequence,
    bool global_norm) {
  const int64_t row = static_cast<int64_t>(batch) * stride_batch +
                      static_cast<int64_t>(head) * stride_head +
                      static_cast<int64_t>(token) * stride_sequence;
  const int norm_index = global_norm ? head * HeadDim + channel : channel;
  const float normalized = to_float(from_float<T>(
      to_float(input[row + channel]) * rrms * to_float(norm[norm_index])));
  if (channel >= rot_dim || rot_dim == 0) {
    return normalized;
  }

  const int pairs = rot_dim / 2;
  const int pair = SplitHalf ? (channel < pairs ? channel : channel - pairs)
                             : channel / 2;
  const int first = SplitHalf ? pair : pair * 2;
  const int second = SplitHalf ? pair + pairs : first + 1;
  const int first_norm = global_norm ? head * HeadDim + first : first;
  const int second_norm = global_norm ? head * HeadDim + second : second;
  const float x0 = to_float(from_float<T>(
      to_float(input[row + first]) * rrms * to_float(norm[first_norm])));
  const float x1 = to_float(from_float<T>(
      to_float(input[row + second]) * rrms * to_float(norm[second_norm])));
  const int fb = freq_batches == 1 ? 0 : batch;
  const int fs = freq_sequence == 1 ? 0 : token;
  const int64_t freq_base = static_cast<int64_t>(fb) * freq_stride_batch +
                            static_cast<int64_t>(fs) * freq_stride_sequence +
                            static_cast<int64_t>(pair) * freq_stride_pair;
  const int component = SplitHalf ? (channel >= pairs) : (channel & 1);
  const float f0 = to_float(freqs[freq_base + component * freq_stride_row]);
  const float f1 = to_float(freqs[freq_base + component * freq_stride_row + freq_stride_col]);
  return to_float(from_float<T>(fmaf(f1, x1, f0 * x0)));
}

template <typename T, int HeadDim, bool SplitHalf>
__device__ __forceinline__ float processed_tile_value(
    const T *__restrict__ tile_values,
    const T *__restrict__ norm,
    const T *__restrict__ freqs,
    int batch,
    int head,
    int token,
    int token_offset,
    int channel,
    float rrms,
    int rot_dim,
    int64_t freq_stride_batch,
    int64_t freq_stride_sequence,
    int64_t freq_stride_pair,
    int64_t freq_stride_row,
    int64_t freq_stride_col,
    int freq_batches,
    int freq_sequence,
    bool global_norm) {
  const T *row = tile_values + token_offset * HeadDim;
  const int norm_index = global_norm ? head * HeadDim + channel : channel;
  const float normalized = to_float(from_float<T>(
      to_float(row[channel]) * rrms * to_float(norm[norm_index])));
  if (channel >= rot_dim || rot_dim == 0) {
    return normalized;
  }

  const int pairs = rot_dim / 2;
  const int pair = SplitHalf ? (channel < pairs ? channel : channel - pairs)
                             : channel / 2;
  const int first = SplitHalf ? pair : pair * 2;
  const int second = SplitHalf ? pair + pairs : first + 1;
  const int first_norm = global_norm ? head * HeadDim + first : first;
  const int second_norm = global_norm ? head * HeadDim + second : second;
  const float x0 = to_float(from_float<T>(
      to_float(row[first]) * rrms * to_float(norm[first_norm])));
  const float x1 = to_float(from_float<T>(
      to_float(row[second]) * rrms * to_float(norm[second_norm])));
  const int fb = freq_batches == 1 ? 0 : batch;
  const int fs = freq_sequence == 1 ? 0 : token;
  const int64_t freq_base = static_cast<int64_t>(fb) * freq_stride_batch +
                            static_cast<int64_t>(fs) * freq_stride_sequence +
                            static_cast<int64_t>(pair) * freq_stride_pair;
  const int component = SplitHalf ? (channel >= pairs) : (channel & 1);
  const float f0 = to_float(freqs[freq_base + component * freq_stride_row]);
  const float f1 = to_float(
      freqs[freq_base + component * freq_stride_row + freq_stride_col]);
  return to_float(from_float<T>(fmaf(f1, x1, f0 * x0)));
}

template <typename T>
__global__ void global_rrms_kernel(
    const T *__restrict__ query,
    const T *__restrict__ key,
    float *__restrict__ query_rrms,
    float *__restrict__ key_rrms,
    int batch_size,
    int query_heads,
    int key_heads,
    int query_length,
    int key_length,
    int head_dim,
    int64_t q_stride_batch,
    int64_t q_stride_head,
    int64_t q_stride_sequence,
    int64_t k_stride_batch,
    int64_t k_stride_head,
    int64_t k_stride_sequence,
    float epsilon) {
  const int task = blockIdx.x;
  const int total_query = batch_size * query_length;
  const bool is_query = task < total_query;
  const int local = is_query ? task : task - total_query;
  const int length = is_query ? query_length : key_length;
  const int heads = is_query ? query_heads : key_heads;
  const int batch = local / length;
  const int token = local - batch * length;
  const T *input = is_query ? query : key;
  const int64_t stride_batch = is_query ? q_stride_batch : k_stride_batch;
  const int64_t stride_head = is_query ? q_stride_head : k_stride_head;
  const int64_t stride_sequence = is_query ? q_stride_sequence : k_stride_sequence;
  float sum = 0.0f;
  for (int index = threadIdx.x; index < heads * head_dim; index += blockDim.x) {
    const int head = index / head_dim;
    const int channel = index - head * head_dim;
    const float value = to_float(input[
        static_cast<int64_t>(batch) * stride_batch +
        static_cast<int64_t>(head) * stride_head +
        static_cast<int64_t>(token) * stride_sequence + channel]);
    sum = fmaf(value, value, sum);
  }
  sum = vllm::blockAllReduceSum(sum);
  if (threadIdx.x == 0) {
    const float rrms = rsqrtf(sum / static_cast<float>(heads * head_dim) + epsilon);
    (is_query ? query_rrms : key_rrms)[local] = rrms;
  }
}

template <typename T, int HeadDim, bool SplitHalf>
__global__ void detect_anchor_kernel(
    const T *__restrict__ key,
    const T *__restrict__ key_norm,
    const T *__restrict__ freqs,
    const float *__restrict__ key_rrms,
    int *__restrict__ anchor_indices,
    float *__restrict__ anchor_values,
    int batch_size,
    int heads,
    int sequence,
    int rot_dim,
    int64_t stride_batch,
    int64_t stride_head,
    int64_t stride_sequence,
    int64_t freq_stride_batch,
    int64_t freq_stride_sequence,
    int64_t freq_stride_pair,
    int64_t freq_stride_row,
    int64_t freq_stride_col,
    int freq_batches,
    int freq_sequence,
    float epsilon,
    bool global_norm) {
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ float samples[kAnchorSamples * kMaxHeadDim];
  __shared__ float rrms[kAnchorSamples];
  __shared__ float warp_original_energy[4];
  __shared__ float warp_original_max[4];
  __shared__ float warp_distance[kAnchorSamples][4];
  __shared__ float warp_best_energy[4];
  __shared__ float warp_best_max[4];
  __shared__ int selected;

  if (tid < kAnchorSamples) {
    const int token = tid * (sequence - 1) / (kAnchorSamples - 1);
    if (global_norm) {
      rrms[tid] = key_rrms[batch * sequence + token];
    } else {
      float sum = 0.0f;
      const int64_t row = static_cast<int64_t>(batch) * stride_batch +
                          static_cast<int64_t>(head) * stride_head +
                          static_cast<int64_t>(token) * stride_sequence;
#pragma unroll
      for (int channel = 0; channel < HeadDim; ++channel) {
        const float value = to_float(key[row + channel]);
        sum = fmaf(value, value, sum);
      }
      rrms[tid] = rsqrtf(sum / static_cast<float>(HeadDim) + epsilon);
    }
  }
  __syncthreads();

  if (tid < HeadDim) {
#pragma unroll
    for (int sample = 0; sample < kAnchorSamples; ++sample) {
      const int token = sample * (sequence - 1) / (kAnchorSamples - 1);
      samples[sample * HeadDim + tid] = processed_value<T, HeadDim, SplitHalf>(
          key, key_norm, freqs, batch, head, token, tid, rrms[sample], heads,
          sequence, rot_dim, stride_batch, stride_head, stride_sequence,
          freq_stride_batch, freq_stride_sequence, freq_stride_pair,
          freq_stride_row, freq_stride_col, freq_batches, freq_sequence,
          global_norm);
    }
  }
  __syncthreads();

  float original_energy = 0.0f;
  float original_max = 0.0f;
  float distance[kAnchorSamples];
#pragma unroll
  for (int candidate = 0; candidate < kAnchorSamples; ++candidate) distance[candidate] = 0.0f;
  for (int channel = tid; channel < HeadDim; channel += kAnchorThreads) {
    float channel_sum = 0.0f;
#pragma unroll
    for (int sample = 0; sample < kAnchorSamples; ++sample) {
      const float value = samples[sample * HeadDim + channel];
      original_energy = fmaf(value, value, original_energy);
      original_max = fmaxf(original_max, fabsf(value));
      channel_sum += value;
    }
#pragma unroll
    for (int candidate = 0; candidate < kAnchorSamples; ++candidate) {
      const float delta = kAnchorSamples * samples[candidate * HeadDim + channel] - channel_sum;
      distance[candidate] = fmaf(delta, delta, distance[candidate]);
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    original_energy += __shfl_down_sync(0xffffffffu, original_energy, offset);
    original_max = fmaxf(original_max, __shfl_down_sync(0xffffffffu, original_max, offset));
#pragma unroll
    for (int candidate = 0; candidate < kAnchorSamples; ++candidate) {
      distance[candidate] += __shfl_down_sync(0xffffffffu, distance[candidate], offset);
    }
  }
  if (lane == 0) {
    warp_original_energy[warp] = original_energy;
    warp_original_max[warp] = original_max;
#pragma unroll
    for (int candidate = 0; candidate < kAnchorSamples; ++candidate) warp_distance[candidate][warp] = distance[candidate];
  }
  __syncthreads();
  if (tid == 0) {
    selected = 0;
    float best = FLT_MAX;
#pragma unroll
    for (int candidate = 0; candidate < kAnchorSamples; ++candidate) {
      float value = 0.0f;
#pragma unroll
      for (int w = 0; w < 4; ++w) value += warp_distance[candidate][w];
      if (value < best) {
        best = value;
        selected = candidate;
      }
    }
  }
  __syncthreads();

  float best_energy = 0.0f;
  float best_max = 0.0f;
  for (int channel = tid; channel < HeadDim; channel += kAnchorThreads) {
    const float anchor = samples[selected * HeadDim + channel];
#pragma unroll
    for (int sample = 0; sample < kAnchorSamples; ++sample) {
      const float residual = samples[sample * HeadDim + channel] - anchor;
      best_energy = fmaf(residual, residual, best_energy);
      best_max = fmaxf(best_max, fabsf(residual));
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    best_energy += __shfl_down_sync(0xffffffffu, best_energy, offset);
    best_max = fmaxf(best_max, __shfl_down_sync(0xffffffffu, best_max, offset));
  }
  if (lane == 0) {
    warp_best_energy[warp] = best_energy;
    warp_best_max[warp] = best_max;
  }
  __syncthreads();
  if (tid == 0) {
    float original_total = 0.0f, original_peak = 0.0f;
    float best_total = 0.0f, best_peak = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      original_total += warp_original_energy[w];
      original_peak = fmaxf(original_peak, warp_original_max[w]);
      best_total += warp_best_energy[w];
      best_peak = fmaxf(best_peak, warp_best_max[w]);
    }
    const bool use_anchor = best_total < original_total && best_peak <= original_peak * 1.125f;
    anchor_indices[batch * heads + head] = use_anchor
        ? selected * (sequence - 1) / (kAnchorSamples - 1) : -1;
    selected = use_anchor ? selected : -1;
  }
  __syncthreads();
  for (int channel = tid; channel < HeadDim; channel += kAnchorThreads) {
    anchor_values[(batch * heads + head) * HeadDim + channel] =
        selected >= 0 ? samples[selected * HeadDim + channel] : 0.0f;
  }
}

template <typename T, int HeadDim, int BlockSize, bool SplitHalf, bool Rotate, bool AllowAnchor>
__device__ __forceinline__ void quantize_tile(
    const T *__restrict__ input,
    const T *__restrict__ norm,
    const T *__restrict__ freqs,
    const float *__restrict__ row_rrms,
    const int *__restrict__ anchor_indices,
    const float *__restrict__ anchor_values,
    int8_t *__restrict__ output,
    float *__restrict__ scales,
    int tile,
    int head,
    int batch,
    int heads,
    int sequence,
    int rot_dim,
    int64_t stride_batch,
    int64_t stride_head,
    int64_t stride_sequence,
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_sequence,
    int64_t scale_stride_batch,
    int64_t scale_stride_head,
    int64_t freq_stride_batch,
    int64_t freq_stride_sequence,
    int64_t freq_stride_pair,
    int64_t freq_stride_row,
    int64_t freq_stride_col,
    int freq_batches,
    int freq_sequence,
    float epsilon,
    bool global_norm) {
  constexpr int packs_per_token = HeadDim / 8;
  constexpr int packs_per_tile = BlockSize * packs_per_token;
  constexpr int iterations = (packs_per_tile + kThreads - 1) / kThreads;
  const int base_token = tile * BlockSize;
  const int tid = threadIdx.x;
  __shared__ float anchor[AllowAnchor ? HeadDim : 1];
  // Keep the raw tile in shared memory so RMSNorm, RoPE and quantization read
  // Q/K from global memory only once. The largest D128 K tile occupies 16 KiB.
  __shared__ T tile_values[BlockSize * HeadDim];
  __shared__ float shared_amax;
  const int anchor_index = AllowAnchor ? anchor_indices[batch * heads + head] : -1;
  if constexpr (AllowAnchor) {
    for (int channel = tid; channel < HeadDim; channel += kThreads) {
      anchor[channel] = anchor_values[(batch * heads + head) * HeadDim + channel];
    }
  }
  for (int index = tid; index < BlockSize * HeadDim; index += kThreads) {
    const int token_offset = index / HeadDim;
    const int channel = index - token_offset * HeadDim;
    const int token = base_token + token_offset;
    if (token < sequence) {
      const int64_t row = static_cast<int64_t>(batch) * stride_batch +
                          static_cast<int64_t>(head) * stride_head +
                          static_cast<int64_t>(token) * stride_sequence;
      tile_values[index] = input[row + channel];
    } else {
      tile_values[index] = from_float<T>(0.0f);
    }
  }
  __syncthreads();

  float values[iterations][8];
#pragma unroll
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const int pack_index = tid + iteration * kThreads;
    const int token_offset = pack_index / packs_per_token;
    const int pack = pack_index - token_offset * packs_per_token;
    const int token = base_token + token_offset;
    const bool valid_pack = pack_index < packs_per_tile;
    const bool valid_token = valid_pack && token < sequence;
    float rrms = 0.0f;
    float raw[8];
#pragma unroll
    for (int channel = 0; channel < 8; ++channel) raw[channel] = 0.0f;
    if (valid_token) {
#pragma unroll
      for (int channel = 0; channel < 8; ++channel) {
        raw[channel] = to_float(
            tile_values[token_offset * HeadDim + pack * 8 + channel]);
      }
    }
    if (global_norm) {
      if (valid_token) rrms = row_rrms[batch * sequence + token];
    } else {
      // Every lane in the logical token subgroup must participate in the
      // shuffle, including lanes belonging to the partial final sequence
      // tile. A full mask inside the valid-token branch deadlocks on sm75.
      float sum = 0.0f;
#pragma unroll
      for (int channel = 0; channel < 8; ++channel) sum = fmaf(raw[channel], raw[channel], sum);
#pragma unroll
      for (int offset = packs_per_token / 2; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset, packs_per_token);
      }
      rrms = rsqrtf(sum / static_cast<float>(HeadDim) + epsilon);
    }
    if (valid_token) {
#pragma unroll
      for (int channel = 0; channel < 8; ++channel) {
        const int dimension = pack * 8 + channel;
        values[iteration][channel] = processed_tile_value<T, HeadDim, SplitHalf>(
            tile_values, norm, freqs, batch, head, token, token_offset,
            dimension, rrms, rot_dim,
            freq_stride_batch, freq_stride_sequence, freq_stride_pair,
            freq_stride_row, freq_stride_col, freq_batches, freq_sequence,
            global_norm);
        if constexpr (AllowAnchor) {
          if (anchor_index >= 0) values[iteration][channel] -= anchor[dimension];
        }
      }
    } else {
#pragma unroll
      for (int channel = 0; channel < 8; ++channel) values[iteration][channel] = 0.0f;
    }
    if constexpr (Rotate) hadamard<HeadDim>(values[iteration], pack);
  }

  float amax = 0.0000001f;
#pragma unroll
  for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
    for (int channel = 0; channel < 8; ++channel) amax = fmaxf(amax, fabsf(values[iteration][channel]));
  }
  amax = vllm::blockReduceMax(amax);
  if (tid == 0) {
    shared_amax = amax;
    scales[batch * scale_stride_batch + head * scale_stride_head + tile] = amax / 127.0f;
  }
  __syncthreads();
  const float inverse_scale = 127.0f / shared_amax;
#pragma unroll
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const int pack_index = tid + iteration * kThreads;
    const int token_offset = pack_index / packs_per_token;
    const int pack = pack_index - token_offset * packs_per_token;
    const int token = base_token + token_offset;
    if (pack_index < packs_per_tile && token < sequence) {
      int8_t *destination = output + static_cast<int64_t>(batch) * output_stride_batch +
                            static_cast<int64_t>(head) * output_stride_head +
                            static_cast<int64_t>(token) * output_stride_sequence + pack * 8;
#pragma unroll
      for (int channel = 0; channel < 8; ++channel) destination[channel] = quantize_int8(values[iteration][channel] * inverse_scale);
    }
  }
}

template <typename T, int HeadDim, bool SplitHalf, bool Rotate, bool Stabilize>
__global__ void quantize_qk_kernel(
    const T *__restrict__ query,
    const T *__restrict__ key,
    const T *__restrict__ query_norm,
    const T *__restrict__ key_norm,
    const T *__restrict__ freqs,
    const float *__restrict__ query_rrms,
    const float *__restrict__ key_rrms,
    const int *__restrict__ anchor_indices,
    const float *__restrict__ anchor_values,
    int8_t *__restrict__ query_output,
    int8_t *__restrict__ key_output,
    float *__restrict__ query_scale,
    float *__restrict__ key_scale,
    int query_heads,
    int key_heads,
    int query_length,
    int key_length,
    int query_scale_length,
    int key_scale_length,
    int rot_dim,
    int64_t q_stride_batch, int64_t q_stride_head, int64_t q_stride_sequence,
    int64_t k_stride_batch, int64_t k_stride_head, int64_t k_stride_sequence,
    int64_t qo_stride_batch, int64_t qo_stride_head, int64_t qo_stride_sequence,
    int64_t ko_stride_batch, int64_t ko_stride_head, int64_t ko_stride_sequence,
    int64_t qs_stride_batch, int64_t qs_stride_head,
    int64_t ks_stride_batch, int64_t ks_stride_head,
    int64_t freq_stride_batch, int64_t freq_stride_sequence,
    int64_t freq_stride_pair, int64_t freq_stride_row, int64_t freq_stride_col,
    int freq_batches, int freq_sequence, float epsilon, bool global_norm) {
  const int task = blockIdx.x;
  const int batch = blockIdx.y;
  const int query_tasks = query_heads * query_scale_length;
  if (task < query_tasks) {
    const int head = task / query_scale_length;
    const int tile = task - head * query_scale_length;
    quantize_tile<T, HeadDim, 16, SplitHalf, Rotate, false>(
        query, query_norm, freqs, query_rrms, anchor_indices, anchor_values,
        query_output, query_scale, tile, head, batch, query_heads, query_length,
        rot_dim, q_stride_batch, q_stride_head, q_stride_sequence,
        qo_stride_batch, qo_stride_head, qo_stride_sequence,
        qs_stride_batch, qs_stride_head, freq_stride_batch, freq_stride_sequence,
        freq_stride_pair, freq_stride_row, freq_stride_col, freq_batches,
        freq_sequence, epsilon, global_norm);
  } else {
    const int local = task - query_tasks;
    const int head = local / key_scale_length;
    const int tile = local - head * key_scale_length;
    if (head < key_heads) {
      quantize_tile<T, HeadDim, 64, SplitHalf, Rotate, Stabilize>(
          key, key_norm, freqs, key_rrms, anchor_indices, anchor_values,
          key_output, key_scale, tile, head, batch, key_heads, key_length,
          rot_dim, k_stride_batch, k_stride_head, k_stride_sequence,
          ko_stride_batch, ko_stride_head, ko_stride_sequence,
          ks_stride_batch, ks_stride_head, freq_stride_batch, freq_stride_sequence,
          freq_stride_pair, freq_stride_row, freq_stride_col, freq_batches,
          freq_sequence, epsilon, global_norm);
    }
  }
}

}  // namespace

void quant_qk_rms_rope_int8_cuda(
    at::Tensor query, at::Tensor key, at::Tensor query_output,
    at::Tensor key_output, at::Tensor query_scale, at::Tensor key_scale,
    at::Tensor query_norm, at::Tensor key_norm, at::Tensor freqs,
    at::Tensor query_rrms, at::Tensor key_rrms, at::Tensor anchor_indices,
    at::Tensor anchor_values, float epsilon, int rot_dim,
    int query_block_size, int query_warp_block_size, int key_block_size,
    int tensor_layout, int norm_scope, bool split_half, bool rotate,
    bool detect_anchor) {
  CHECK_CUDA(query); CHECK_CUDA(key); CHECK_CUDA(query_output); CHECK_CUDA(key_output);
  CHECK_CUDA(query_scale); CHECK_CUDA(key_scale); CHECK_CUDA(query_norm); CHECK_CUDA(key_norm);
  CHECK_DIMS(query, 4); CHECK_DIMS(key, 4); CHECK_LASTDIM_CONTIGUOUS(query); CHECK_LASTDIM_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(query_output); CHECK_LASTDIM_CONTIGUOUS(key_output);
  CHECK_DTYPE(query_output, at::ScalarType::Char); CHECK_DTYPE(key_output, at::ScalarType::Char);
  CHECK_DTYPE(query_scale, at::ScalarType::Float); CHECK_DTYPE(key_scale, at::ScalarType::Float);
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "Q/K dtypes must match");
  TORCH_CHECK(query_norm.scalar_type() == query.scalar_type() && key_norm.scalar_type() == query.scalar_type(), "Q/K norm weights must match the input dtype");
  TORCH_CHECK(query_block_size == 64 && query_warp_block_size == 16 && key_block_size == 64, "fused Q/K preprocessing requires the production 64/16/64 quantization contract");
  TORCH_CHECK(norm_scope == 0 || norm_scope == 1, "norm_scope must be 0 (per head) or 1 (whole row)");

  const int head_dim = query.size(3);
  TORCH_CHECK(head_dim == 64 || head_dim == 128, "fused Q/K preprocessing requires head_dim 64 or 128");
  TORCH_CHECK(key.size(3) == head_dim && query.size(0) == key.size(0), "Q/K shapes are incompatible");
  int query_heads, key_heads, query_length, key_length;
  int64_t q_stride_head, q_stride_sequence, k_stride_head, k_stride_sequence;
  int64_t qo_stride_head, qo_stride_sequence, ko_stride_head, ko_stride_sequence;
  if (tensor_layout == 0) {
    query_length = query.size(1); query_heads = query.size(2);
    key_length = key.size(1); key_heads = key.size(2);
    q_stride_sequence = query.stride(1); q_stride_head = query.stride(2);
    k_stride_sequence = key.stride(1); k_stride_head = key.stride(2);
    qo_stride_sequence = query_output.stride(1); qo_stride_head = query_output.stride(2);
    ko_stride_sequence = key_output.stride(1); ko_stride_head = key_output.stride(2);
  } else if (tensor_layout == 1) {
    query_heads = query.size(1); query_length = query.size(2);
    key_heads = key.size(1); key_length = key.size(2);
    q_stride_head = query.stride(1); q_stride_sequence = query.stride(2);
    k_stride_head = key.stride(1); k_stride_sequence = key.stride(2);
    qo_stride_head = query_output.stride(1); qo_stride_sequence = query_output.stride(2);
    ko_stride_head = key_output.stride(1); ko_stride_sequence = key_output.stride(2);
  } else {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }
  TORCH_CHECK(rot_dim >= 0 && rot_dim <= head_dim && (rot_dim % 2) == 0, "rot_dim must be even and within head_dim");
  const int expected_norm_q = norm_scope == 0 ? head_dim : query_heads * head_dim;
  const int expected_norm_k = norm_scope == 0 ? head_dim : key_heads * head_dim;
  TORCH_CHECK(query_norm.numel() == expected_norm_q && key_norm.numel() == expected_norm_k, "Q/K norm weight shapes are incompatible");
  TORCH_CHECK(query_output.sizes() == query.sizes() && key_output.sizes() == key.sizes(), "Q/K output shapes must match inputs");
  const int query_scale_length = ((query_length + 63) / 64) * 4;
  const int key_scale_length = (key_length + 63) / 64;
  TORCH_CHECK(query_scale.sizes() == at::IntArrayRef({query.size(0), query_heads, query_scale_length}), "Q scale shape is incompatible");
  TORCH_CHECK(key_scale.sizes() == at::IntArrayRef({key.size(0), key_heads, key_scale_length}), "K scale shape is incompatible");

  int64_t freq_stride_batch = 0, freq_stride_sequence = 0, freq_stride_pair = 0, freq_stride_row = 0, freq_stride_col = 0;
  int freq_batches = 1, freq_sequence = 1;
  if (rot_dim > 0) {
    CHECK_CUDA(freqs); CHECK_DIMS(freqs, 6);
    TORCH_CHECK(freqs.scalar_type() == query.scalar_type(), "RoPE frequencies must match Q/K dtype");
    TORCH_CHECK(freqs.size(2) == 1 && freqs.size(3) == rot_dim / 2 && freqs.size(4) == 2 && freqs.size(5) == 2, "RoPE frequency shape is incompatible");
    freq_batches = freqs.size(0); freq_sequence = freqs.size(1);
    TORCH_CHECK((freq_batches == 1 || freq_batches == query.size(0)) && (freq_sequence == 1 || (freq_sequence == query_length && query_length == key_length)), "RoPE batch/sequence broadcasting is incompatible");
    freq_stride_batch = freqs.stride(0); freq_stride_sequence = freqs.stride(1);
    freq_stride_pair = freqs.stride(3); freq_stride_row = freqs.stride(4); freq_stride_col = freqs.stride(5);
  }

  const bool global_norm = norm_scope == 1;
  if (global_norm) {
    CHECK_CUDA(query_rrms); CHECK_CUDA(key_rrms); CHECK_DTYPE(query_rrms, at::ScalarType::Float); CHECK_DTYPE(key_rrms, at::ScalarType::Float);
    TORCH_CHECK(query_rrms.numel() == query.size(0) * query_length && key_rrms.numel() == key.size(0) * key_length, "row RRMS scratch shapes are incompatible");
  }
  const bool stabilize = anchor_indices.numel() != 0;
  TORCH_CHECK(!detect_anchor || stabilize,
              "anchor detection requires anchor output tensors");
  if (stabilize) {
    CHECK_CUDA(anchor_indices); CHECK_CUDA(anchor_values); CHECK_DTYPE(anchor_indices, at::ScalarType::Int); CHECK_DTYPE(anchor_values, at::ScalarType::Float);
    TORCH_CHECK(anchor_indices.sizes() == at::IntArrayRef({key.size(0), key_heads}), "anchor index shape is incompatible");
    TORCH_CHECK(anchor_values.sizes() == at::IntArrayRef({key.size(0), key_heads, head_dim}), "anchor value shape is incompatible");
  }

#define LAUNCH_VARIANT(SPLIT, ROTATE, STABILIZE) \
      do { \
        if constexpr (STABILIZE) { \
          if (detect_anchor) { \
            detect_anchor_kernel<scalar_t, HEAD_DIM, SPLIT><<<dim3(key_heads, key.size(0)), kAnchorThreads, 0, c10::cuda::getCurrentCUDAStream()>>>( \
                reinterpret_cast<const scalar_t *>(key.data_ptr()), reinterpret_cast<const scalar_t *>(key_norm.data_ptr()), \
                reinterpret_cast<const scalar_t *>(freqs.data_ptr()), global_norm ? key_rrms.data_ptr<float>() : nullptr, \
                anchor_indices.data_ptr<int>(), anchor_values.data_ptr<float>(), key.size(0), key_heads, key_length, rot_dim, \
                key.stride(0), k_stride_head, k_stride_sequence, freq_stride_batch, freq_stride_sequence, freq_stride_pair, \
                freq_stride_row, freq_stride_col, freq_batches, freq_sequence, epsilon, global_norm); \
          } \
        } \
        const int tasks = query_heads * query_scale_length + key_heads * key_scale_length; \
        quantize_qk_kernel<scalar_t, HEAD_DIM, SPLIT, ROTATE, STABILIZE><<<dim3(tasks, query.size(0)), kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>( \
            reinterpret_cast<const scalar_t *>(query.data_ptr()), reinterpret_cast<const scalar_t *>(key.data_ptr()), \
            reinterpret_cast<const scalar_t *>(query_norm.data_ptr()), reinterpret_cast<const scalar_t *>(key_norm.data_ptr()), \
            reinterpret_cast<const scalar_t *>(freqs.data_ptr()), global_norm ? query_rrms.data_ptr<float>() : nullptr, \
            global_norm ? key_rrms.data_ptr<float>() : nullptr, STABILIZE ? anchor_indices.data_ptr<int>() : nullptr, \
            STABILIZE ? anchor_values.data_ptr<float>() : nullptr, query_output.data_ptr<int8_t>(), key_output.data_ptr<int8_t>(), \
            query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), query_heads, key_heads, query_length, key_length, \
            query_scale_length, key_scale_length, rot_dim, query.stride(0), q_stride_head, q_stride_sequence, \
            key.stride(0), k_stride_head, k_stride_sequence, query_output.stride(0), qo_stride_head, qo_stride_sequence, \
            key_output.stride(0), ko_stride_head, ko_stride_sequence, query_scale.stride(0), query_scale.stride(1), \
            key_scale.stride(0), key_scale.stride(1), freq_stride_batch, freq_stride_sequence, freq_stride_pair, \
            freq_stride_row, freq_stride_col, freq_batches, freq_sequence, epsilon, global_norm); \
      } while (0)

  const auto dtype = query.scalar_type();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(dtype, scalar_t, {
    if (global_norm) {
      const int tasks = query.size(0) * query_length + key.size(0) * key_length;
      global_rrms_kernel<scalar_t><<<tasks, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const scalar_t *>(query.data_ptr()), reinterpret_cast<const scalar_t *>(key.data_ptr()),
          query_rrms.data_ptr<float>(), key_rrms.data_ptr<float>(), query.size(0), query_heads, key_heads,
          query_length, key_length, head_dim, query.stride(0), q_stride_head, q_stride_sequence,
          key.stride(0), k_stride_head, k_stride_sequence, epsilon);
    }
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      if (split_half) {
        if (rotate) { if (stabilize) LAUNCH_VARIANT(true, true, true); else LAUNCH_VARIANT(true, true, false); }
        else LAUNCH_VARIANT(true, false, false);
      } else {
        if (rotate) { if (stabilize) LAUNCH_VARIANT(false, true, true); else LAUNCH_VARIANT(false, true, false); }
        else LAUNCH_VARIANT(false, false, false);
      }
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#undef LAUNCH_VARIANT
}
