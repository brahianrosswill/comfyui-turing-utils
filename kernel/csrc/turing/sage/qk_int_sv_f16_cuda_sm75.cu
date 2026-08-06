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

#include "../utils.cuh"
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include "torch_compat.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "cp_async.cuh"
#include "mma.cuh"
#include "permuted_smem.cuh"
#include "../math.cuh"
#include "dispatch_utils.h"

#include "attn_utils.cuh"

#define PACK_SIZE_QK 16 // as if it is int8
#define PACK_SIZE_V 8   // fp16
#define PACK_SIZE_O 8   // fp16

// treat as if int8 tensor core
#define MMA_QK_M 16
#define MMA_QK_N 16
#define MMA_QK_K 32

// fp16 tensor core
#define MMA_SV_M 16
#define MMA_SV_N 16
#define MMA_SV_K 16

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim, DataType DTypeQK, QuantGranularity Q_GRAN, QuantGranularity K_GRAN,
        typename DTypeSVAccum = float, bool use_inst_buffer = false, typename DTypeOut = half,
        typename DTypeV = half, ComputeUnit DenominatorAccumUnit = ComputeUnit::kTensorCore,
        MaskMode mask_mode = MaskMode::kNone, bool return_lse = false, bool fuse_v_mean=false,
        bool fuse_q_mean_correction=false, bool fuse_k_mean=false>
__global__ void qk_int_sv_f16_attn_kernel(int8_t *__restrict__ Q, int8_t *__restrict__ K, DTypeV *__restrict__ V, DTypeOut *__restrict__ O, float *__restrict__ Lse,
                      float *__restrict__ Q_scale, float *__restrict__ K_scale, DTypeOut *__restrict__ V_mean,
                      uint16_t *__restrict__ K_original, float *__restrict__ Q_mean, float *__restrict__ K_mean,
                      const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                      const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
                      const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
                      const uint32_t stride_bz_v, const uint32_t stride_seq_v, const uint32_t stride_h_v,
                      const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
                      const uint32_t stride_bz_ko, const uint32_t stride_seq_ko, const uint32_t stride_h_ko,
                      const bool key_original_is_bf16,
                      float sm_scale)
{
  // compile time check
  static_assert(DTypeQK == DataType::kInt8 || DTypeQK == DataType::kInt4, "DTypeQK must be int8 or int4");
  static_assert(Q_GRAN == QuantGranularity::kPerBlock || Q_GRAN == QuantGranularity::kPerWarp || Q_GRAN == QuantGranularity::kPerThread, "Q_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp || K_GRAN == QuantGranularity::kPerThread, "K_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(std::is_same<DTypeSVAccum, float>::value || !use_inst_buffer, "use_inst_buffer only supports DTypeSVAccum as float");
  static_assert(std::is_same<DTypeSVAccum, float>::value || std::is_same<DTypeSVAccum, half>::value, "DTypeSVAccum must be float or half");
  static_assert(std::is_same<DTypeOut, half>::value || std::is_same<DTypeOut, nv_bfloat16>::value, "DTypeOut must be half or nv_bfloat16");
  static_assert(std::is_same<DTypeV, half>::value || std::is_same<DTypeV, nv_bfloat16>::value, "DTypeV must be half or nv_bfloat16");
  static_assert(head_dim % 64 == 0, "head_dim must be a multiple of 64");
  static_assert(!fuse_v_mean || std::is_same<DTypeSVAccum, half>::value, "fuse_v_mean only supports half");
  static_assert(!fuse_q_mean_correction || DTypeQK == DataType::kInt4,
                "Q-mean correction is only used by packed INT4 attention");
  static_assert(CTA_Q / CTA_K <= 2); // for efficient causal implementation

  using DTypeOut2 = typename std::conditional<std::is_same<DTypeOut, half>::value, half2, nv_bfloat162>::type;

  constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
  constexpr uint32_t num_warps_k = CTA_K / WARP_K;
  constexpr uint32_t num_warps = num_warps_q * num_warps_k;
  constexpr uint32_t num_tiles_q = WARP_Q / MMA_QK_M;
  constexpr uint32_t num_tiles_k = WARP_K / MMA_QK_N;
  constexpr uint32_t num_tiles_qk_inner = (DTypeQK == DataType::kInt8) ? (head_dim / MMA_QK_K) : (head_dim / 2 / MMA_QK_K);
  constexpr uint32_t num_tiles_v = head_dim / MMA_SV_N;

  constexpr uint32_t QK_SMEM_STRIDE = (DTypeQK == DataType::kInt8) ? (head_dim) : (head_dim / 2);
  constexpr uint32_t O_SMEM_STRIDE = head_dim;
  constexpr uint32_t V_SMEM_STRIDE = head_dim;

  extern __shared__ int8_t smem[];

  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();

  // maximize L2 hit rate
  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t head_id = blockIdx.y;

  // transfer to base 2 instead of base e with better numerical efficiency
  sm_scale *= math::log2e;

  // RS holds the fragment of S
  int32_t RS[num_tiles_q][num_tiles_k][8];
  DTypeSVAccum RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2]; // max
  float d[num_tiles_q][2]; // denominator

  uint32_t q_scale_idx, k_scale_idx;

  if constexpr (Q_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_q = gridDim.x;
    q_scale_idx = batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx;
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * num_warp_block_q + head_id * num_warp_block_q + bx * num_warps_q + get_warp_idx_q<num_warps_q, num_warps_k>();
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * (num_warp_block_q * 8) + head_id * (num_warp_block_q * 8) + bx * (num_warps_q * 8) + get_warp_idx_q<num_warps_q, num_warps_k>() * 8 + lane_id / 4;
  }

  if constexpr (K_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_block_k + (head_id / num_kv_groups) * num_block_k;
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_warp_block_k + (head_id / num_kv_groups) * num_warp_block_k + get_warp_idx_k<num_warps_q, num_warps_k>();
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * (num_warp_block_k * 4) + (head_id / num_kv_groups) * (num_warp_block_k * 4) + get_warp_idx_k<num_warps_q, num_warps_k>() * 4 + lane_id % 4;
  }

  constexpr uint32_t k_scale_advance_offset = (K_GRAN == QuantGranularity::kPerBlock) ? 1 : (K_GRAN == QuantGranularity::kPerWarp) ? (CTA_K / WARP_K) : (CTA_K / WARP_K) * 4;

  // initialize o, m, d
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      if constexpr (std::is_same<DTypeSVAccum, float>::value)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RO[fq][fv][k] = 0.0f;
        }
      }
      else if constexpr (std::is_same<DTypeSVAccum, half>::value)
      {
#pragma unroll
        for (uint32_t k = 0; k < 4; k++)
        {
          ((int32_t*)RO[fq][fv])[k] = 0;
        }
      }
    }
  }
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++)
    {
      m[fq][k] = -5000000.0f;
      d[fq][k] = 1.0f;
    }
  }

  constexpr uint32_t K_smem_idx_offset = CTA_Q;
  constexpr uint32_t V_smem_idx_offset = CTA_Q + CTA_K;

  constexpr SwizzleMode swizzle_mode_QK = (QK_SMEM_STRIDE == 32) ? SwizzleMode::k32B : (QK_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_Q(smem);
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_K(smem + K_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_V = (V_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V> smem_V(smem + V_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_O = (O_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_O, O_SMEM_STRIDE / PACK_SIZE_O> smem_O(smem);

  constexpr uint32_t qk_smem_bytes = (CTA_Q + CTA_K) * QK_SMEM_STRIDE;
  constexpr uint32_t v_smem_bytes = CTA_K * head_dim * sizeof(half);
  float *smem_Q_mean = reinterpret_cast<float *>(smem + qk_smem_bytes + v_smem_bytes);
  float *smem_K_mean = smem_Q_mean + (fuse_q_mean_correction ? head_dim : 0);
  float *smem_score_correction = smem_K_mean +
      (fuse_q_mean_correction && fuse_k_mean ? head_dim : 0);

  if constexpr (fuse_q_mean_correction)
  {
    const uint32_t linear_thread = warp_id * WARP_SIZE + lane_id;
    constexpr uint32_t num_threads = num_warps * WARP_SIZE;
    const float *q_mean_ptr = Q_mean +
        ((batch_id * num_qo_heads + head_id) * gridDim.x + bx) * head_dim;
    for (uint32_t dim = linear_thread; dim < head_dim; dim += num_threads)
    {
      smem_Q_mean[dim] = q_mean_ptr[dim];
      if constexpr (fuse_k_mean)
      {
        const uint32_t num_kv_heads = num_qo_heads / num_kv_groups;
        smem_K_mean[dim] = K_mean[(batch_id * num_kv_heads + head_id / num_kv_groups) * head_dim + dim];
      }
    }
    __syncthreads();
  }

  constexpr uint32_t global_to_shared_line_lanes_QK = (QK_SMEM_STRIDE == 32) ? 2 : (QK_SMEM_STRIDE == 64) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_QK = (QK_SMEM_STRIDE == 32) ? 16 : (QK_SMEM_STRIDE == 64) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_V = (V_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_V = (V_SMEM_STRIDE == 32) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_O = (O_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_O = (O_SMEM_STRIDE == 32) ? 8 : 4;

  constexpr uint32_t QK_smem_iters_row = QK_SMEM_STRIDE / (global_to_shared_line_lanes_QK * PACK_SIZE_QK);
  constexpr uint32_t Q_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t K_smem_iters_col = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t V_smem_iters_row = V_SMEM_STRIDE / (global_to_shared_line_lanes_V * PACK_SIZE_V);
  constexpr uint32_t V_smem_iters_col = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_V);
  constexpr uint32_t O_smem_iters_row = O_SMEM_STRIDE / (global_to_shared_line_lanes_O * PACK_SIZE_O);
  constexpr uint32_t O_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_O);

  int8_t *Q_lane_base_ptr = Q + batch_id * stride_bz_q + head_id * stride_h_q + (bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_q + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  int8_t *K_lane_base_ptr = K + batch_id * stride_bz_k + (head_id / num_kv_groups) * stride_h_k + (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_k + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  DTypeV *V_lane_base_ptr = V + batch_id * stride_bz_v + (head_id / num_kv_groups) * stride_h_v + (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V) * stride_seq_v + (lane_id % global_to_shared_line_lanes_V) * PACK_SIZE_V;
  uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * Q_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t K_smem_offset_load = smem_K.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * K_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t V_smem_offset_load = smem_V.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_V * V_smem_iters_col + lane_id / global_to_shared_line_lanes_V, lane_id % global_to_shared_line_lanes_V);

  uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16, lane_id / 16);
  uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 16, lane_id / 16);

  // for causal masking
  uint32_t Q_idx_lane_base = bx * CTA_Q + get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
  uint32_t K_idx_lane_base = get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + 2 * (lane_id % 4);

  // for loading
  uint32_t Q_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t K_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t V_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V;

  const uint32_t num_iterations = div_ceil(
      mask_mode == MaskMode::kCausal
          ? min(kv_len, (bx + 1) * CTA_Q)
          : kv_len,
      CTA_K);

  auto compute_score_correction = [&](uint32_t tile_start) {
    if constexpr (fuse_q_mean_correction)
    {
      __syncthreads();
      // Four independent 8-lane groups evaluate four K tokens at once.  The
      // former whole-warp loop serialized 16 tokens and performed 80 shuffle
      // steps per warp/tile; this schedule performs the same dot products
      // with 12 width-8 shuffles and coalesced 4-token loads.
      constexpr uint32_t correction_group_width = 8;
      constexpr uint32_t correction_groups_per_warp = WARP_SIZE / correction_group_width;
      const uint32_t correction_group = lane_id / correction_group_width;
      const uint32_t correction_lane = lane_id % correction_group_width;
      const uint32_t local_token_base = warp_id * 16;
      for (uint32_t token_batch = 0; token_batch < 4; ++token_batch)
      {
        const uint32_t local_token = local_token_base +
            token_batch * correction_groups_per_warp + correction_group;
        const uint32_t token = tile_start + local_token;
        float partial = 0.0f;
        if (token < kv_len)
        {
          uint16_t *key_ptr = K_original + batch_id * stride_bz_ko +
              (head_id / num_kv_groups) * stride_h_ko + token * stride_seq_ko;
#pragma unroll
          for (uint32_t dim = correction_lane; dim < head_dim;
               dim += correction_group_width)
          {
            float key_value;
            if (key_original_is_bf16)
            {
              key_value = __bfloat162float(
                  *reinterpret_cast<nv_bfloat16 *>(key_ptr + dim));
            }
            else
            {
              key_value = __half2float(*reinterpret_cast<half *>(key_ptr + dim));
            }
            if constexpr (fuse_k_mean)
            {
              key_value -= smem_K_mean[dim];
            }
            partial += smem_Q_mean[dim] * key_value;
          }
        }
#pragma unroll
        for (uint32_t offset = correction_group_width / 2; offset > 0; offset >>= 1)
        {
          partial += __shfl_down_sync(0xffffffff, partial, offset,
                                      correction_group_width);
        }
        if (correction_lane == 0)
        {
          smem_score_correction[local_token] = partial;
        }
      }
      __syncthreads();
    }
  };

  // load Q with predicate
  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, Q_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_Q>(
    &Q_lane_base_ptr, Q_smem_offset_load, stride_seq_q, smem_Q, Q_load_idx_lane_base, qo_len);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  // for num_tiles_qk_inner = 1, we load all Qs in register
  uint32_t RQ[num_tiles_q][4];
  if constexpr (num_tiles_qk_inner == 1)
  {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      smem_Q.ldmatrix_m8n8x4(Q_smem_offset_mma, RQ[fq]);
      Q_smem_offset_mma = smem_Q.advance_offset_by_row<16>(Q_smem_offset_mma);
    }
  }

  // load K with predicate
  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
    &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
  cp_async::commit_group();

  float q_scale = Q_scale[q_scale_idx];

  float original_sm_scale = sm_scale;
  float dequant_scale = q_scale * K_scale[k_scale_idx + 0 * k_scale_advance_offset];

  sm_scale = original_sm_scale * dequant_scale;

  // load V with predicate
  load_v_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
    &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V, V_load_idx_lane_base, kv_len);
  cp_async::commit_group();

  K_load_idx_lane_base += CTA_K;
  V_load_idx_lane_base += CTA_K;

#pragma unroll
  for (uint32_t iter = 1; iter < num_iterations - 1; iter++)
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    compute_score_correction((iter - 1) * CTA_K);

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          if constexpr (fuse_q_mean_correction)
          {
            const uint32_t local_k = 2 * (lane_id % 4) + fk * 16 +
                                     8 * (k / 4) + k % 2;
            RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale +
                                smem_score_correction[local_k];
          }
          else
          {
            RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
          }
        }
      }
    }

    // do not apply causal mask and out of bound mask for these iterations
    K_idx_lane_base += CTA_K;

    if constexpr (std::is_same<DTypeSVAccum, float>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(
          RS_f32, RO, m, d, fuse_q_mean_correction ? original_sm_scale : sm_scale);
    }
    else if constexpr (std::is_same<DTypeSVAccum, half>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, false, false>(
          RS_f32, RO, m, d, fuse_q_mean_correction ? original_sm_scale : sm_scale);
    }

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);
    }

    __syncthreads();

    // load K
    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K);
    cp_async::commit_group();

    dequant_scale = q_scale * K_scale[k_scale_idx + iter * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;

    // ensure V is ready
    cp_async::wait_group<1>();
    __syncthreads();

    if constexpr (!use_inst_buffer)
    {
      compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }
    else
    {
      compute_fp16_sv_permuted_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }

    __syncthreads();
    // load V
    load_v_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V);
    cp_async::commit_group();
    K_load_idx_lane_base += CTA_K;
    V_load_idx_lane_base += CTA_K;
  }

  // second last iter, apply causal mask
  if (num_iterations > 1)
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    compute_score_correction((num_iterations - 2) * CTA_K);

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          const uint32_t local_k = 2 * (lane_id % 4) + fk * 16 +
                                   8 * (k / 4) + k % 2;
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
          if constexpr (fuse_q_mean_correction)
          {
            RS_f32[fq][fk][k] += smem_score_correction[local_k];
          }
        }
      }
    }

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    // apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);
    K_idx_lane_base += CTA_K;

    if constexpr (std::is_same<DTypeSVAccum, float>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, original_sm_scale);
    }
    else if constexpr (std::is_same<DTypeSVAccum, half>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, false, false>(RS_f32, RO, m, d, original_sm_scale);
    }

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);
    }

    __syncthreads();

    // load K with predicate
    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
    cp_async::commit_group();

    dequant_scale = q_scale * K_scale[k_scale_idx + (num_iterations - 1) * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;

    // ensure V is ready
    cp_async::wait_group<1>();
    __syncthreads();

    if constexpr (!use_inst_buffer)
    {
      compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }
    else
    {
      compute_fp16_sv_permuted_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }

    __syncthreads();
    // load V with predicate
    load_v_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V, V_load_idx_lane_base, kv_len);
    cp_async::commit_group();
    K_load_idx_lane_base += CTA_K;
    V_load_idx_lane_base += CTA_K;
  }

  // last iter, apply causal mask and out of bound mask
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    compute_score_correction((num_iterations - 1) * CTA_K);

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
            const uint32_t local_k = 2 * (lane_id % 4) + fk * 16 +
                                     8 * (k / 4) + k % 2;
            RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
            if constexpr (fuse_q_mean_correction)
            {
              RS_f32[fq][fk][k] += smem_score_correction[local_k];
            }
        }
      }
    }

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    // check out of bound in the last iter
    apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);
    K_idx_lane_base += CTA_K;

    if constexpr (std::is_same<DTypeSVAccum, float>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, original_sm_scale);
    }
    else if constexpr (std::is_same<DTypeSVAccum, half>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, false, false>(RS_f32, RO, m, d, original_sm_scale);
    }

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);
    }

    // ensure V is ready
    cp_async::wait_group<0>();
    __syncthreads();

    if constexpr (!use_inst_buffer)
    {
      compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }
    else
    {
      compute_fp16_sv_permuted_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }

    __syncthreads();

  }

  // TODO: thread block sync mdo state for num_warps_k > 0

  normalize_d<num_tiles_q, num_tiles_v, DenominatorAccumUnit>(RO, m, d);

  // save the result
  // if (get_warp_idx_k<num_warps_q, num_warps_k>() == 0)
  // {

  // convert half to bfloat16
  if constexpr (std::is_same<DTypeSVAccum, half>::value && std::is_same<DTypeOut, nv_bfloat16>::value)
  {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++)
      {
        ((nv_bfloat162*)RO[fq][fv])[0] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[0]));
        ((nv_bfloat162*)RO[fq][fv])[1] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[1]));
        ((nv_bfloat162*)RO[fq][fv])[2] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[2]));
        ((nv_bfloat162*)RO[fq][fv])[3] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[3]));
      }
    }
  }

  // add v_mean
  if constexpr (fuse_v_mean)
  {
    DTypeOut2 v_mean[2];
    DTypeOut *V_mean_lane_ptr = V_mean + batch_id * (num_qo_heads / num_kv_groups) * head_dim + (head_id / num_kv_groups) * head_dim + lane_id % 4 * 2;
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      v_mean[0] = *((DTypeOut2*)(V_mean_lane_ptr + fv * 16));
      v_mean[1] = *((DTypeOut2*)(V_mean_lane_ptr + 8 + fv * 16));
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        ((DTypeOut2*)RO[fq][fv])[0] = __hadd2(((DTypeOut2*)RO[fq][fv])[0], v_mean[0]);
        ((DTypeOut2*)RO[fq][fv])[1] = __hadd2(((DTypeOut2*)RO[fq][fv])[1], v_mean[0]);
        ((DTypeOut2*)RO[fq][fv])[2] = __hadd2(((DTypeOut2*)RO[fq][fv])[2], v_mean[1]);
        ((DTypeOut2*)RO[fq][fv])[3] = __hadd2(((DTypeOut2*)RO[fq][fv])[3], v_mean[1]);
      }
    }
  }

  // save the result to shared memory
  uint32_t smem_O_row_base = get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      uint32_t offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O));

      if constexpr (std::is_same<DTypeSVAccum, float>::value)
      {
        // convert RO to half
        uint32_t RO_f16[4];
#pragma unroll
        for (uint32_t k = 0; k < 4; k++)
        {
          if constexpr (std::is_same<DTypeOut, half>::value)
          {
            ((half2*)RO_f16)[k] = __float22half2_rn(((float2*)RO[fq][fv])[k]);
          }
          else if constexpr (std::is_same<DTypeOut, nv_bfloat16>::value)
          {
            ((nv_bfloat162*)RO_f16)[k] = __float22bfloat162_rn(((float2*)RO[fq][fv])[k]);
          }
        }

        ((uint32_t*)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[0];
        ((uint32_t*)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[1];

        // ! permuted, make sure you know what you are doing
        ((uint32_t*)(smem_O.base + (offset_O ^ 0x1)))[lane_id % 4] = RO_f16[2];
        ((uint32_t*)(smem_O.base + (offset_O ^ 0x1) + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[3];
      }
      else if constexpr (std::is_same<DTypeSVAccum, half>::value)
      {
        ((uint32_t*)(smem_O.base + offset_O))[lane_id % 4] = ((uint32_t*)RO[fq][fv])[0];
        ((uint32_t*)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = ((uint32_t*)RO[fq][fv])[1];

        // ! permuted, make sure you know what you are doing
        ((uint32_t*)(smem_O.base + (offset_O ^ 0x1)))[lane_id % 4] = ((uint32_t*)RO[fq][fv])[2];
        ((uint32_t*)(smem_O.base + (offset_O ^ 0x1) + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = ((uint32_t*)RO[fq][fv])[3];
      }
    }
  }

  // ! do we need to sync here?
  __syncwarp();

  // shared memory to global memory
  DTypeOut *O_lane_ptr = O + batch_id * stride_bz_o + head_id * stride_h_o + (bx * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>() + lane_id / global_to_shared_line_lanes_O) * stride_seq_o + lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O;
  uint32_t offset_O = smem_O.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / global_to_shared_line_lanes_O, lane_id % global_to_shared_line_lanes_O);
  uint32_t O_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_O;

#pragma unroll
  for (uint32_t i = 0; i < O_smem_iters_col; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < O_smem_iters_row; j++)
    {
      if (O_load_idx_lane_base < qo_len)
      {
        smem_O.store_128b(offset_O, O_lane_ptr);
      }
      O_lane_ptr += (global_to_shared_line_lanes_O * PACK_SIZE_O);
      offset_O = smem_O.advance_offset_by_column<global_to_shared_line_lanes_O>(offset_O);
    }

    offset_O = smem_O.advance_offset_by_row<global_to_shared_copy_lines_per_warp_O>(offset_O - (O_smem_iters_row * global_to_shared_line_lanes_O));
    O_lane_ptr += ((global_to_shared_copy_lines_per_warp_O * stride_seq_o) - (O_smem_iters_row * global_to_shared_line_lanes_O * PACK_SIZE_O));
    O_load_idx_lane_base += global_to_shared_copy_lines_per_warp_O;
  }

  if constexpr (return_lse)
  {
    uint32_t lse_idx = bx * CTA_Q + lane_id / 4 + 8 * (lane_id % 4) + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>();
    float *lse_lane_ptr = Lse + batch_id * (qo_len * num_qo_heads) + head_id * qo_len + lse_idx;
    uint32_t fq = (lane_id % 4) / 2;
    uint32_t k = (lane_id % 4) % 2;

    if (lse_idx < qo_len && (lane_id % 4) < 2 * num_tiles_q)
    {
      lse_lane_ptr[0] = (math::ptx_log2(d[fq][k]) + m[fq][k]);
    }
  }

  // }
}

template <uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K,
          uint32_t HEAD_DIM, int QK_QUANT_GRAN, typename DTypeSVAccum,
          bool use_inst_buffer, typename DTypeOut, typename DTypeV, MaskMode mask_mode,
          bool RETURN_LSE, bool fuse_v_mean, DataType DTypeQK = DataType::kInt8,
          bool fuse_q_mean_correction = false, bool fuse_k_mean = false>
static void launch_qk_int_sv_f16_attn(at::Tensor query,
                                      at::Tensor key,
                                      at::Tensor value,
                                      at::Tensor output,
                                      at::Tensor query_scale,
                                      at::Tensor key_scale,
                                      at::Tensor value_mean,
                                      at::Tensor lse,
                                      int batch_size,
                                      int qo_len,
                                      int kv_len,
                                      int num_qo_heads,
                                      int num_kv_heads,
                                      int num_kv_groups,
                                      int stride_bz_q,
                                      int stride_seq_q,
                                      int stride_h_q,
                                      int stride_bz_k,
                                      int stride_seq_k,
                                      int stride_h_k,
                                      int stride_bz_v,
                                      int stride_seq_v,
                                      int stride_h_v,
                                      int stride_bz_o,
                                      int stride_seq_o,
                                      int stride_h_o,
                                      float sm_scale,
                                      at::Tensor key_original = at::Tensor(),
                                      at::Tensor query_mean = at::Tensor(),
                                      at::Tensor key_mean = at::Tensor(),
                                      int stride_bz_ko = 0,
                                      int stride_seq_ko = 0,
                                      int stride_h_ko = 0)
{
  if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp))
  {
    CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q));
    CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));
  }
  else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread))
  {
    CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q) * 8);
    CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K) * 4);
  }
  else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerBlock))
  {
    CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q));
    CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K));
  }
  else
  {
    static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerBlock) ||
                  QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp) ||
                  QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread),
                  "Unsupported quantization granularity");
  }

  if constexpr (fuse_v_mean)
  {
    CHECK_SHAPE(value_mean, batch_size, num_kv_heads, HEAD_DIM);
  }

  if constexpr (fuse_q_mean_correction)
  {
    CHECK_SHAPE(query_mean, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q), HEAD_DIM);
    if constexpr (fuse_k_mean)
    {
      CHECK_SHAPE(key_mean, batch_size, num_kv_heads, 1, HEAD_DIM);
    }
  }

  constexpr size_t qk_smem_stride =
      DTypeQK == DataType::kInt4 ? HEAD_DIM / 2 : HEAD_DIM;
  constexpr size_t correction_smem = fuse_q_mean_correction
      ? (HEAD_DIM + (fuse_k_mean ? HEAD_DIM : 0) + CTA_K) * sizeof(float)
      : 0;
  constexpr size_t smem_max = std::max(
      (CTA_Q + CTA_K) * qk_smem_stride + CTA_K * HEAD_DIM * sizeof(half) +
          correction_smem,
      CTA_Q * HEAD_DIM * sizeof(half));
  static_assert(smem_max <= 48 * 1024, "SM75 attention must stay within 48 KiB shared memory");

  auto kernel_func = qk_int_sv_f16_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM,
                                               DTypeQK,
                                               static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                               static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                               DTypeSVAccum, use_inst_buffer, DTypeOut, DTypeV,
                                               ComputeUnit::kTensorCore, mask_mode,
                                               RETURN_LSE, fuse_v_mean,
                                               fuse_q_mean_correction, fuse_k_mean>;

  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

  dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

  kernel_func<<<grid, block, smem_max>>>(
      query.data_ptr<int8_t>(),
      key.data_ptr<int8_t>(),
      reinterpret_cast<DTypeV*>(value.data_ptr()),
      reinterpret_cast<DTypeOut*>(output.data_ptr()),
      (RETURN_LSE) ? reinterpret_cast<float*>(lse.data_ptr()) : nullptr,
      reinterpret_cast<float*>(query_scale.data_ptr()),
      reinterpret_cast<float*>(key_scale.data_ptr()),
      (fuse_v_mean) ? reinterpret_cast<DTypeOut*>(value_mean.data_ptr()) : nullptr,
      (fuse_q_mean_correction) ? reinterpret_cast<uint16_t*>(key_original.data_ptr()) : nullptr,
      (fuse_q_mean_correction) ? reinterpret_cast<float*>(query_mean.data_ptr()) : nullptr,
      (fuse_q_mean_correction && fuse_k_mean) ? reinterpret_cast<float*>(key_mean.data_ptr()) : nullptr,
      qo_len,
      kv_len,
      num_kv_groups,
      stride_bz_q, stride_seq_q, stride_h_q,
      stride_bz_k, stride_seq_k, stride_h_k,
      stride_bz_v, stride_seq_v, stride_h_v,
      stride_bz_o, stride_seq_o, stride_h_o,
      stride_bz_ko, stride_seq_ko, stride_h_ko,
      fuse_q_mean_correction && key_original.scalar_type() == at::ScalarType::BFloat16,
      sm_scale);
}

// tensor_layout 0 for [B, N, H, D], 1 for [B, H, N, D]
at::Tensor qk_int8_sv_f16_accum_f32_attn(at::Tensor query,
                    at::Tensor key,
                    at::Tensor value,
                    at::Tensor output,
                    at::Tensor query_scale,
                    at::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);

  CHECK_DTYPE(query, at::ScalarType::Char);
  CHECK_DTYPE(key, at::ScalarType::Char);
  TORCH_CHECK(value.scalar_type() == at::ScalarType::Half ||
                  value.scalar_type() == at::ScalarType::BFloat16,
              "value must be float16 or bfloat16");
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);

  const int head_dim = query.size(3);
  const int batch_size = query.size(0);

  int stride_bz_q = query.stride(0);
  int stride_bz_k = key.stride(0);
  int stride_bz_v = value.stride(0);
  int stride_bz_o = output.stride(0);

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_seq_k, stride_seq_v, stride_seq_o;
  int stride_h_q, stride_h_k, stride_h_v, stride_h_o;

  if (tensor_layout == 0)
  {
    qo_len = query.size(1);
    kv_len = key.size(1);
    num_qo_heads = query.size(2);
    num_kv_heads = key.size(2);
    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, head_dim);

    stride_seq_q = query.stride(1);
    stride_seq_k = key.stride(1);
    stride_seq_v = value.stride(1);
    stride_seq_o = output.stride(1);

    stride_h_q = query.stride(2);
    stride_h_k = key.stride(2);
    stride_h_v = value.stride(2);
    stride_h_o = output.stride(2);
  }
  else if (tensor_layout == 1)
  {
    qo_len = query.size(2);
    kv_len = key.size(2);
    num_qo_heads = query.size(1);
    num_kv_heads = key.size(1);
    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(value, batch_size, num_kv_heads, kv_len, head_dim);

    stride_seq_q = query.stride(2);
    stride_seq_k = key.stride(2);
    stride_seq_v = value.stride(2);
    stride_seq_o = output.stride(2);

    stride_h_q = query.stride(1);
    stride_h_k = key.stride(1);
    stride_h_v = value.stride(1);
    stride_h_o = output.stride(1);
  }
  else
  {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;

  at::Tensor lse = at::empty({0});
  if (return_lse)
  {
    lse = at::empty({batch_size, num_qo_heads, qo_len}, query.options().dtype(at::ScalarType::Float));
  }

  auto output_dtype = output.scalar_type();
  auto value_dtype = value.scalar_type();

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
           DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(value_dtype, DTypeV, {
            constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

            launch_qk_int_sv_f16_attn<64, 64, 16, 64, HEAD_DIM, QK_QUANT_GRAN,
                                       float, false, DTypeOut, DTypeV, mask_mode,
                                       RETURN_LSE, false>(
                query, key, value, output, query_scale, key_scale, at::Tensor(), lse,
                batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads, num_kv_groups,
                stride_bz_q, stride_seq_q, stride_h_q,
                stride_bz_k, stride_seq_k, stride_h_k,
                stride_bz_v, stride_seq_v, stride_h_v,
                stride_bz_o, stride_seq_o, stride_h_o,
                sm_scale);
           });
          });
        });
      });
    });
  });

  return lse;
}

at::Tensor qk_int8_sv_f16_accum_f16_attn(at::Tensor query,
                    at::Tensor key,
                    at::Tensor value,
                    at::Tensor output,
                    at::Tensor query_scale,
                    at::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);

  CHECK_DTYPE(query, at::ScalarType::Char);
  CHECK_DTYPE(key, at::ScalarType::Char);
  TORCH_CHECK(value.scalar_type() == at::ScalarType::Half ||
                  value.scalar_type() == at::ScalarType::BFloat16,
              "value must be float16 or bfloat16");
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);

  const int head_dim = query.size(3);
  const int batch_size = query.size(0);

  int stride_bz_q = query.stride(0);
  int stride_bz_k = key.stride(0);
  int stride_bz_v = value.stride(0);
  int stride_bz_o = output.stride(0);

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_seq_k, stride_seq_v, stride_seq_o;
  int stride_h_q, stride_h_k, stride_h_v, stride_h_o;

  if (tensor_layout == 0)
  {
    qo_len = query.size(1);
    kv_len = key.size(1);
    num_qo_heads = query.size(2);
    num_kv_heads = key.size(2);
    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, head_dim);

    stride_seq_q = query.stride(1);
    stride_seq_k = key.stride(1);
    stride_seq_v = value.stride(1);
    stride_seq_o = output.stride(1);

    stride_h_q = query.stride(2);
    stride_h_k = key.stride(2);
    stride_h_v = value.stride(2);
    stride_h_o = output.stride(2);
  }
  else if (tensor_layout == 1)
  {
    qo_len = query.size(2);
    kv_len = key.size(2);
    num_qo_heads = query.size(1);
    num_kv_heads = key.size(1);
    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(value, batch_size, num_kv_heads, kv_len, head_dim);

    stride_seq_q = query.stride(2);
    stride_seq_k = key.stride(2);
    stride_seq_v = value.stride(2);
    stride_seq_o = output.stride(2);

    stride_h_q = query.stride(1);
    stride_h_k = key.stride(1);
    stride_h_v = value.stride(1);
    stride_h_o = output.stride(1);
  }
  else
  {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());
  }

  at::Tensor lse = at::empty({0});
  if (return_lse)
  {
    lse = at::empty({batch_size, num_qo_heads, qo_len}, query.options().dtype(at::ScalarType::Float));
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;

  auto output_dtype = output.scalar_type();
  auto value_dtype = value.scalar_type();

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
           DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(value_dtype, DTypeV, {

            constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

            launch_qk_int_sv_f16_attn<64, 64, 16, 64, HEAD_DIM, QK_QUANT_GRAN,
                                       half, false, DTypeOut, DTypeV, mask_mode,
                                       RETURN_LSE, false>(
                query, key, value, output, query_scale, key_scale, at::Tensor(), lse,
                batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads, num_kv_groups,
                stride_bz_q, stride_seq_q, stride_h_q,
                stride_bz_k, stride_seq_k, stride_h_k,
                stride_bz_v, stride_seq_v, stride_h_v,
                stride_bz_o, stride_seq_o, stride_h_o,
                sm_scale);
           });
          });
        });
      });
    });
  });

  return lse;
}

at::Tensor qk_int8_sv_f16_accum_f16_attn_inst_buf(at::Tensor query,
                    at::Tensor key,
                    at::Tensor value,
                    at::Tensor output,
                    at::Tensor query_scale,
                    at::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);

  CHECK_DTYPE(query, at::ScalarType::Char);
  CHECK_DTYPE(key, at::ScalarType::Char);
  TORCH_CHECK(value.scalar_type() == at::ScalarType::Half ||
                  value.scalar_type() == at::ScalarType::BFloat16,
              "value must be float16 or bfloat16");
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);

  const int head_dim = query.size(3);
  const int batch_size = query.size(0);

  int stride_bz_q = query.stride(0);
  int stride_bz_k = key.stride(0);
  int stride_bz_v = value.stride(0);
  int stride_bz_o = output.stride(0);

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_seq_k, stride_seq_v, stride_seq_o;
  int stride_h_q, stride_h_k, stride_h_v, stride_h_o;

  if (tensor_layout == 0)
  {
    qo_len = query.size(1);
    kv_len = key.size(1);
    num_qo_heads = query.size(2);
    num_kv_heads = key.size(2);
    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, head_dim);

    stride_seq_q = query.stride(1);
    stride_seq_k = key.stride(1);
    stride_seq_v = value.stride(1);
    stride_seq_o = output.stride(1);

    stride_h_q = query.stride(2);
    stride_h_k = key.stride(2);
    stride_h_v = value.stride(2);
    stride_h_o = output.stride(2);
  }
  else if (tensor_layout == 1)
  {
    qo_len = query.size(2);
    kv_len = key.size(2);
    num_qo_heads = query.size(1);
    num_kv_heads = key.size(1);
    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(value, batch_size, num_kv_heads, kv_len, head_dim);

    stride_seq_q = query.stride(2);
    stride_seq_k = key.stride(2);
    stride_seq_v = value.stride(2);
    stride_seq_o = output.stride(2);

    stride_h_q = query.stride(1);
    stride_h_k = key.stride(1);
    stride_h_v = value.stride(1);
    stride_h_o = output.stride(1);
  }
  else
  {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());
  }

  at::Tensor lse = at::empty({0});
  if (return_lse)
  {
    lse = at::empty({batch_size, num_qo_heads, qo_len}, query.options().dtype(at::ScalarType::Float));
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;

  auto output_dtype = output.scalar_type();
  auto value_dtype = value.scalar_type();

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
           DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(value_dtype, DTypeV, {

            constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

            launch_qk_int_sv_f16_attn<64, 64, 16, 64, HEAD_DIM, QK_QUANT_GRAN,
                                       float, true, DTypeOut, DTypeV, mask_mode,
                                       RETURN_LSE, false>(
                query, key, value, output, query_scale, key_scale, at::Tensor(), lse,
                batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads, num_kv_groups,
                stride_bz_q, stride_seq_q, stride_h_q,
                stride_bz_k, stride_seq_k, stride_h_k,
                stride_bz_v, stride_seq_v, stride_h_v,
                stride_bz_o, stride_seq_o, stride_h_o,
                sm_scale);
           });
          });
        });
      });
    });
  });

  return lse;
}

at::Tensor qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(at::Tensor query,
                    at::Tensor key,
                    at::Tensor value,
                    at::Tensor output,
                    at::Tensor query_scale,
                    at::Tensor key_scale,
                    at::Tensor value_mean,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(value_mean);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(value_mean);

  CHECK_DTYPE(query, at::ScalarType::Char);
  CHECK_DTYPE(key, at::ScalarType::Char);
  TORCH_CHECK(value.scalar_type() == at::ScalarType::Half ||
                  value.scalar_type() == at::ScalarType::BFloat16,
              "value must be float16 or bfloat16");
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  CHECK_DIMS(value_mean, 3);

  const int head_dim = query.size(3);
  const int batch_size = query.size(0);

  int stride_bz_q = query.stride(0);
  int stride_bz_k = key.stride(0);
  int stride_bz_v = value.stride(0);
  int stride_bz_o = output.stride(0);

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_seq_k, stride_seq_v, stride_seq_o;
  int stride_h_q, stride_h_k, stride_h_v, stride_h_o;

  if (tensor_layout == 0)
  {
    qo_len = query.size(1);
    kv_len = key.size(1);
    num_qo_heads = query.size(2);
    num_kv_heads = key.size(2);
    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, head_dim);

    stride_seq_q = query.stride(1);
    stride_seq_k = key.stride(1);
    stride_seq_v = value.stride(1);
    stride_seq_o = output.stride(1);

    stride_h_q = query.stride(2);
    stride_h_k = key.stride(2);
    stride_h_v = value.stride(2);
    stride_h_o = output.stride(2);
  }
  else if (tensor_layout == 1)
  {
    qo_len = query.size(2);
    kv_len = key.size(2);
    num_qo_heads = query.size(1);
    num_kv_heads = key.size(1);
    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(value, batch_size, num_kv_heads, kv_len, head_dim);

    stride_seq_q = query.stride(2);
    stride_seq_k = key.stride(2);
    stride_seq_v = value.stride(2);
    stride_seq_o = output.stride(2);

    stride_h_q = query.stride(1);
    stride_h_k = key.stride(1);
    stride_h_v = value.stride(1);
    stride_h_o = output.stride(1);
  }
  else
  {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());
  }

  at::Tensor lse = at::empty({0});
  if (return_lse)
  {
    lse = at::empty({batch_size, num_qo_heads, qo_len}, query.options().dtype(at::ScalarType::Float));
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;

  auto output_dtype = output.scalar_type();
  auto value_dtype = value.scalar_type();
  auto value_mean_dtype = value_mean.scalar_type();

  TORCH_CHECK(value_mean_dtype == output_dtype, "value_mean and output must have the same dtype");

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
           DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(value_dtype, DTypeV, {

            constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

            launch_qk_int_sv_f16_attn<64, 64, 16, 64, HEAD_DIM, QK_QUANT_GRAN,
                                       half, false, DTypeOut, DTypeV, mask_mode,
                                       RETURN_LSE, true>(
                query, key, value, output, query_scale, key_scale, value_mean, lse,
                batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads, num_kv_groups,
                stride_bz_q, stride_seq_q, stride_h_q,
                stride_bz_k, stride_seq_k, stride_h_k,
                stride_bz_v, stride_seq_v, stride_h_v,
                stride_bz_o, stride_seq_o, stride_h_o,
                sm_scale);
           });
          });
        });
      });
    });
  });

  return lse;
}

at::Tensor qk_int4_sv_f16_accum_f16_attn(
                    at::Tensor query,
                    at::Tensor key,
                    at::Tensor value,
                    at::Tensor output,
                    at::Tensor query_scale,
                    at::Tensor key_scale,
                    at::Tensor key_original,
                    at::Tensor query_mean,
                    at::Tensor key_mean,
                    int tensor_layout,
                    int is_causal,
                    float sm_scale,
                    int return_lse,
                    int smooth_q,
                    int smooth_k)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_DTYPE(query, at::ScalarType::Char);
  CHECK_DTYPE(key, at::ScalarType::Char);
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);
  TORCH_CHECK(value.scalar_type() == at::ScalarType::Half ||
                  value.scalar_type() == at::ScalarType::BFloat16,
              "value must be float16 or bfloat16");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Half ||
                  output.scalar_type() == at::ScalarType::BFloat16,
              "output must be float16 or bfloat16");
  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  TORCH_CHECK(smooth_q == 0 || smooth_q == 1, "smooth_q must be 0 or 1");
  TORCH_CHECK(smooth_k == 0 || smooth_k == 1, "smooth_k must be 0 or 1");

  if (smooth_q)
  {
    CHECK_CUDA(key_original);
    CHECK_CUDA(query_mean);
    CHECK_LASTDIM_CONTIGUOUS(key_original);
    CHECK_CONTIGUOUS(query_mean);
    CHECK_DTYPE(query_mean, at::ScalarType::Float);
    CHECK_DIMS(key_original, 4);
    CHECK_DIMS(query_mean, 4);
    TORCH_CHECK(key_original.scalar_type() == at::ScalarType::Half ||
                    key_original.scalar_type() == at::ScalarType::BFloat16,
                "key_original must be float16 or bfloat16");
    if (smooth_k)
    {
      CHECK_CUDA(key_mean);
      CHECK_CONTIGUOUS(key_mean);
      CHECK_DTYPE(key_mean, at::ScalarType::Float);
      CHECK_DIMS(key_mean, 4);
    }
  }

  const int head_dim = value.size(3);
  const int packed_head_dim = query.size(3);
  const int batch_size = query.size(0);
  TORCH_CHECK(packed_head_dim * 2 == head_dim,
              "packed INT4 Q head dimension must be half the V head dimension");
  TORCH_CHECK(key.size(3) == packed_head_dim, "packed Q/K head dimensions must match");

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_seq_k, stride_seq_v, stride_seq_o;
  int stride_h_q, stride_h_k, stride_h_v, stride_h_o;
  int stride_seq_ko = 0, stride_h_ko = 0;
  if (tensor_layout == 0)
  {
    qo_len = query.size(1);
    kv_len = key.size(1);
    num_qo_heads = query.size(2);
    num_kv_heads = key.size(2);
    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, packed_head_dim);
    CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(output, batch_size, qo_len, num_qo_heads, head_dim);
    stride_seq_q = query.stride(1);
    stride_seq_k = key.stride(1);
    stride_seq_v = value.stride(1);
    stride_seq_o = output.stride(1);
    stride_h_q = query.stride(2);
    stride_h_k = key.stride(2);
    stride_h_v = value.stride(2);
    stride_h_o = output.stride(2);
    if (smooth_q)
    {
      CHECK_SHAPE(key_original, batch_size, kv_len, num_kv_heads, head_dim);
      stride_seq_ko = key_original.stride(1);
      stride_h_ko = key_original.stride(2);
    }
  }
  else if (tensor_layout == 1)
  {
    qo_len = query.size(2);
    kv_len = key.size(2);
    num_qo_heads = query.size(1);
    num_kv_heads = key.size(1);
    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, packed_head_dim);
    CHECK_SHAPE(value, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(output, batch_size, num_qo_heads, qo_len, head_dim);
    stride_seq_q = query.stride(2);
    stride_seq_k = key.stride(2);
    stride_seq_v = value.stride(2);
    stride_seq_o = output.stride(2);
    stride_h_q = query.stride(1);
    stride_h_k = key.stride(1);
    stride_h_v = value.stride(1);
    stride_h_o = output.stride(1);
    if (smooth_q)
    {
      CHECK_SHAPE(key_original, batch_size, num_kv_heads, kv_len, head_dim);
      stride_seq_ko = key_original.stride(2);
      stride_h_ko = key_original.stride(1);
    }
  }
  else
  {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }

  TORCH_CHECK(num_kv_heads > 0 && num_qo_heads % num_kv_heads == 0,
              "the Q head count must be divisible by the KV head count");
  const int num_kv_groups = num_qo_heads / num_kv_heads;
  at::Tensor lse = at::empty({0});
  if (return_lse)
  {
    lse = at::empty({batch_size, num_qo_heads, qo_len},
                    query.options().dtype(at::ScalarType::Float));
  }

  auto output_dtype = output.scalar_type();
  auto value_dtype = value.scalar_type();
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
        DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
          DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(value_dtype, DTypeV, {
            constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;
            if (smooth_q && smooth_k)
            {
              launch_qk_int_sv_f16_attn<64, 64, 16, 64, HEAD_DIM, 3,
                  half, false, DTypeOut, DTypeV, mask_mode, RETURN_LSE, false,
                  DataType::kInt4, true, true>(
                  query, key, value, output, query_scale, key_scale, at::Tensor(), lse,
                  batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads, num_kv_groups,
                  query.stride(0), stride_seq_q, stride_h_q,
                  key.stride(0), stride_seq_k, stride_h_k,
                  value.stride(0), stride_seq_v, stride_h_v,
                  output.stride(0), stride_seq_o, stride_h_o, sm_scale,
                  key_original, query_mean, key_mean,
                  key_original.stride(0), stride_seq_ko, stride_h_ko);
            }
            else if (smooth_q)
            {
              launch_qk_int_sv_f16_attn<64, 64, 16, 64, HEAD_DIM, 3,
                  half, false, DTypeOut, DTypeV, mask_mode, RETURN_LSE, false,
                  DataType::kInt4, true, false>(
                  query, key, value, output, query_scale, key_scale, at::Tensor(), lse,
                  batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads, num_kv_groups,
                  query.stride(0), stride_seq_q, stride_h_q,
                  key.stride(0), stride_seq_k, stride_h_k,
                  value.stride(0), stride_seq_v, stride_h_v,
                  output.stride(0), stride_seq_o, stride_h_o, sm_scale,
                  key_original, query_mean, at::Tensor(),
                  key_original.stride(0), stride_seq_ko, stride_h_ko);
            }
            else
            {
              launch_qk_int_sv_f16_attn<64, 64, 16, 64, HEAD_DIM, 3,
                  half, false, DTypeOut, DTypeV, mask_mode, RETURN_LSE, false,
                  DataType::kInt4, false, false>(
                  query, key, value, output, query_scale, key_scale, at::Tensor(), lse,
                  batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads, num_kv_groups,
                  query.stride(0), stride_seq_q, stride_h_q,
                  key.stride(0), stride_seq_k, stride_h_k,
                  value.stride(0), stride_seq_v, stride_h_v,
                  output.stride(0), stride_seq_o, stride_h_o, sm_scale);
            }
          });
        });
      });
    });
  });
  return lse;
}
