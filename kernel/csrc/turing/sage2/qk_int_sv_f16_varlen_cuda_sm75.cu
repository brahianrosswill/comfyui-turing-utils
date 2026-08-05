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

#define PACK_SIZE_QK 16
#define PACK_SIZE_V 8
#define PACK_SIZE_O 8

#define MMA_QK_M 16
#define MMA_QK_N 16
#define MMA_QK_K 32

#define MMA_SV_N 16

static bool is_sm75_varlen_device()
{
  int dev_id = 0;
  cudaGetDevice(&dev_id);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev_id);
  return prop.major == 7 && prop.minor == 5;
}

template <uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K,
          uint32_t head_dim, typename IndexT, typename DTypeOut,
          MaskMode mask_mode = MaskMode::kNone>
__global__ void qk_int_sv_f16_varlen_attn_kernel(
    int8_t *__restrict__ Q,
    int8_t *__restrict__ K,
    half *__restrict__ V,
    DTypeOut *__restrict__ O,
    float *__restrict__ Q_scale,
    float *__restrict__ K_scale,
    const IndexT *__restrict__ cu_seqlens_q,
    const IndexT *__restrict__ cu_seqlens_k,
    const uint32_t max_seqlen_q,
    const uint32_t num_kv_groups,
    const uint32_t stride_seq_q,
    const uint32_t stride_h_q,
    const uint32_t stride_seq_k,
    const uint32_t stride_h_k,
    const uint32_t stride_seq_v,
    const uint32_t stride_h_v,
    const uint32_t stride_seq_o,
    const uint32_t stride_h_o,
    const uint32_t stride_bz_q_scale,
    const uint32_t stride_h_q_scale,
    const uint32_t stride_bz_k_scale,
    const uint32_t stride_h_k_scale,
    float sm_scale)
{
  static_assert(head_dim % 64 == 0, "head_dim must be a multiple of 64");
  static_assert(CTA_Q / CTA_K <= 2);

  constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
  constexpr uint32_t num_warps_k = CTA_K / WARP_K;
  constexpr uint32_t num_warps = num_warps_q * num_warps_k;
  constexpr uint32_t num_tiles_q = WARP_Q / MMA_QK_M;
  constexpr uint32_t num_tiles_k = WARP_K / MMA_QK_N;
  constexpr uint32_t num_tiles_qk_inner = head_dim / MMA_QK_K;
  constexpr uint32_t num_tiles_v = head_dim / MMA_SV_N;

  constexpr uint32_t QK_SMEM_STRIDE = head_dim;
  constexpr uint32_t O_SMEM_STRIDE = head_dim;
  constexpr uint32_t V_SMEM_STRIDE = head_dim;

  extern __shared__ int8_t smem[];

  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();
  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t head_id = blockIdx.y;
  const uint32_t kv_head = head_id / num_kv_groups;

  const uint32_t q_start = static_cast<uint32_t>(cu_seqlens_q[batch_id]);
  const uint32_t q_end = static_cast<uint32_t>(cu_seqlens_q[batch_id + 1]);
  const uint32_t k_start = static_cast<uint32_t>(cu_seqlens_k[batch_id]);
  const uint32_t k_end = static_cast<uint32_t>(cu_seqlens_k[batch_id + 1]);
  const uint32_t qo_len = q_end - q_start;
  const uint32_t kv_len = k_end - k_start;

  sm_scale *= math::log2e;

  int32_t RS[num_tiles_q][num_tiles_k][8];
  float RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2];
  float d[num_tiles_q][2];

#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++)
      {
        RO[fq][fv][k] = 0.0f;
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

  int8_t *Q_lane_base_ptr = Q + (q_start + bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_q + head_id * stride_h_q + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  int8_t *K_lane_base_ptr = K + (k_start + CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_k + kv_head * stride_h_k + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  half *V_lane_base_ptr = V + (k_start + CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V) * stride_seq_v + kv_head * stride_h_v + (lane_id % global_to_shared_line_lanes_V) * PACK_SIZE_V;

  uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * Q_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t K_smem_offset_load = smem_K.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * K_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t V_smem_offset_load = smem_V.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_V * V_smem_iters_col + lane_id / global_to_shared_line_lanes_V, lane_id % global_to_shared_line_lanes_V);

  uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16, lane_id / 16);
  uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 16, lane_id / 16);

  uint32_t Q_idx_lane_base = bx * CTA_Q + get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
  uint32_t K_idx_lane_base = get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + 2 * (lane_id % 4);
  uint32_t Q_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t K_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t V_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V;

  const uint32_t q_scale_idx = batch_id * stride_bz_q_scale + head_id * stride_h_q_scale + bx * num_warps_q + get_warp_idx_q<num_warps_q, num_warps_k>();
  const uint32_t k_scale_idx = batch_id * stride_bz_k_scale + kv_head * stride_h_k_scale + get_warp_idx_k<num_warps_q, num_warps_k>();
  constexpr uint32_t k_scale_advance_offset = CTA_K / WARP_K;

  const uint32_t num_iterations = div_ceil(
      mask_mode == MaskMode::kCausal
          ? min(kv_len, (bx + 1) * CTA_Q)
          : kv_len,
      CTA_K);
  if (num_iterations == 0)
  {
    return;
  }

  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, Q_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_Q>(
      &Q_lane_base_ptr, Q_smem_offset_load, stride_seq_q, smem_Q, Q_load_idx_lane_base, qo_len);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

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

  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
  cp_async::commit_group();

  const float q_scale = Q_scale[q_scale_idx];
  const float original_sm_scale = sm_scale;
  float dequant_scale = q_scale * K_scale[k_scale_idx];
  sm_scale = original_sm_scale * dequant_scale;

  load_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V, V_load_idx_lane_base, kv_len);
  cp_async::commit_group();

  K_load_idx_lane_base += CTA_K;
  V_load_idx_lane_base += CTA_K;

#pragma unroll
  for (uint32_t iter = 1; iter < num_iterations - 1; iter++)
  {
    cp_async::wait_group<1>();
    __syncthreads();

    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DataType::kInt8>(
          smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DataType::kInt8>(
          smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

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
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
        }
      }
    }

    K_idx_lane_base += CTA_K;
    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, sm_scale);

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);
    accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);

    __syncthreads();
    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
        &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K);
    cp_async::commit_group();

    dequant_scale = q_scale * K_scale[k_scale_idx + iter * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;

    cp_async::wait_group<1>();
    __syncthreads();
    compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);

    __syncthreads();
    load_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
        &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V);
    cp_async::commit_group();
    K_load_idx_lane_base += CTA_K;
    V_load_idx_lane_base += CTA_K;
  }

  if (num_iterations > 1)
  {
    cp_async::wait_group<1>();
    __syncthreads();

    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DataType::kInt8>(
          smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DataType::kInt8>(
          smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

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
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    K_idx_lane_base += CTA_K;

    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, original_sm_scale);

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);
    accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);

    __syncthreads();
    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
        &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
    cp_async::commit_group();

    dequant_scale = q_scale * K_scale[k_scale_idx + (num_iterations - 1) * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;

    cp_async::wait_group<1>();
    __syncthreads();
    compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);

    __syncthreads();
    load_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
        &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V, V_load_idx_lane_base, kv_len);
    cp_async::commit_group();
    K_load_idx_lane_base += CTA_K;
    V_load_idx_lane_base += CTA_K;
  }

  {
    cp_async::wait_group<1>();
    __syncthreads();

    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DataType::kInt8>(
          smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DataType::kInt8>(
          smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

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
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);

    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, original_sm_scale);

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);
    accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);

    cp_async::wait_group<0>();
    __syncthreads();
    compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);

    __syncthreads();
  }

  normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kTensorCore>(RO, m, d);

  if constexpr (std::is_same<DTypeOut, nv_bfloat16>::value)
  {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 4; k++)
        {
          ((nv_bfloat162 *)RO[fq][fv])[k] = __float22bfloat162_rn(((float2 *)RO[fq][fv])[k]);
        }
      }
    }
  }

  uint32_t smem_O_row_base = get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      uint32_t offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O));
      uint32_t RO_f16[4];
#pragma unroll
      for (uint32_t k = 0; k < 4; k++)
      {
        if constexpr (std::is_same<DTypeOut, half>::value)
        {
          ((half2 *)RO_f16)[k] = __float22half2_rn(((float2 *)RO[fq][fv])[k]);
        }
        else if constexpr (std::is_same<DTypeOut, nv_bfloat16>::value)
        {
          ((nv_bfloat162 *)RO_f16)[k] = ((nv_bfloat162 *)RO[fq][fv])[k];
        }
      }

      ((uint32_t *)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[0];
      ((uint32_t *)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[1];
      ((uint32_t *)(smem_O.base + (offset_O ^ 0x1)))[lane_id % 4] = RO_f16[2];
      ((uint32_t *)(smem_O.base + (offset_O ^ 0x1) + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[3];
    }
  }

  __syncwarp();

  DTypeOut *O_lane_ptr = O + (q_start + bx * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>() + lane_id / global_to_shared_line_lanes_O) * stride_seq_o + head_id * stride_h_o + lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O;
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
}

template <uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K,
          uint32_t HEAD_DIM, typename IndexT, typename DTypeOut,
          MaskMode mask_mode>
static void launch_qk_int_sv_f16_varlen_attn(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor output,
    at::Tensor query_scale,
    at::Tensor key_scale,
    at::Tensor cu_seqlens_q,
    at::Tensor cu_seqlens_k,
    int batch_size,
    int max_seqlen_q,
    int max_seqlen_k,
    int num_qo_heads,
    int num_kv_heads,
    int num_kv_groups,
    int stride_seq_q,
    int stride_h_q,
    int stride_seq_k,
    int stride_h_k,
    int stride_seq_v,
    int stride_h_v,
    int stride_seq_o,
    int stride_h_o,
    float sm_scale)
{
  CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(max_seqlen_q, CTA_Q) * (CTA_Q / WARP_Q));
  CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(max_seqlen_k, CTA_K) * (CTA_K / WARP_K));

  constexpr size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) +
                                           CTA_K * HEAD_DIM * sizeof(int8_t) +
                                           CTA_K * HEAD_DIM * sizeof(half),
                                       CTA_Q * HEAD_DIM * sizeof(half));
  static_assert(smem_max <= 48 * 1024, "SM75 attention must stay within 48 KiB shared memory");

  auto kernel_func = qk_int_sv_f16_varlen_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, IndexT, DTypeOut, mask_mode>;
  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

  dim3 grid(div_ceil(max_seqlen_q, CTA_Q), num_qo_heads, batch_size);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

  kernel_func<<<grid, block, smem_max>>>(
      query.data_ptr<int8_t>(),
      key.data_ptr<int8_t>(),
      reinterpret_cast<half *>(value.data_ptr()),
      reinterpret_cast<DTypeOut *>(output.data_ptr()),
      reinterpret_cast<float *>(query_scale.data_ptr()),
      reinterpret_cast<float *>(key_scale.data_ptr()),
      reinterpret_cast<IndexT *>(cu_seqlens_q.data_ptr()),
      reinterpret_cast<IndexT *>(cu_seqlens_k.data_ptr()),
      max_seqlen_q,
      num_kv_groups,
      stride_seq_q, stride_h_q,
      stride_seq_k, stride_h_k,
      stride_seq_v, stride_h_v,
      stride_seq_o, stride_h_o,
      query_scale.stride(0), query_scale.stride(1),
      key_scale.stride(0), key_scale.stride(1),
      sm_scale);
}

at::Tensor qk_int8_sv_f16_varlen_accum_f32_attn(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor output,
    at::Tensor query_scale,
    at::Tensor key_scale,
    at::Tensor cu_seqlens_q,
    at::Tensor cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    int is_causal,
    float sm_scale)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(cu_seqlens_q);
  CHECK_CUDA(cu_seqlens_k);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);

  CHECK_DTYPE(query, at::ScalarType::Char);
  CHECK_DTYPE(key, at::ScalarType::Char);
  CHECK_DTYPE(value, at::ScalarType::Half);
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);

  CHECK_DIMS(query, 3);
  CHECK_DIMS(key, 3);
  CHECK_DIMS(value, 3);
  CHECK_DIMS(output, 3);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  CHECK_DIMS(cu_seqlens_q, 1);
  CHECK_DIMS(cu_seqlens_k, 1);

  TORCH_CHECK(cu_seqlens_q.scalar_type() == cu_seqlens_k.scalar_type(), "cu_seqlens_q and cu_seqlens_k must have the same dtype");

  const int batch_size = cu_seqlens_q.size(0) - 1;
  const int total_q = query.size(0);
  const int total_k = key.size(0);
  const int num_qo_heads = query.size(1);
  const int num_kv_heads = key.size(1);
  const int head_dim = query.size(2);

  CHECK_SHAPE(key, total_k, num_kv_heads, head_dim);
  CHECK_SHAPE(value, total_k, num_kv_heads, head_dim);
  CHECK_SHAPE(output, total_q, num_qo_heads, head_dim);

  if (num_qo_heads % num_kv_heads != 0)
  {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;
  auto output_dtype = output.scalar_type();
  auto index_dtype = cu_seqlens_q.scalar_type();
  at::Tensor lse = at::empty({0});

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
        constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;
        if (index_dtype == at::ScalarType::Int)
        {
          if (is_sm75_varlen_device())
          {
            launch_qk_int_sv_f16_varlen_attn<64, 64, 16, 64, HEAD_DIM, int32_t, DTypeOut, mask_mode>(
                query, key, value, output, query_scale, key_scale, cu_seqlens_q, cu_seqlens_k,
                batch_size, max_seqlen_q, max_seqlen_k, num_qo_heads, num_kv_heads, num_kv_groups,
                query.stride(0), query.stride(1),
                key.stride(0), key.stride(1),
                value.stride(0), value.stride(1),
                output.stride(0), output.stride(1),
                sm_scale);
          }
          else
          {
            launch_qk_int_sv_f16_varlen_attn<128, 64, 32, 64, HEAD_DIM, int32_t, DTypeOut, mask_mode>(
                query, key, value, output, query_scale, key_scale, cu_seqlens_q, cu_seqlens_k,
                batch_size, max_seqlen_q, max_seqlen_k, num_qo_heads, num_kv_heads, num_kv_groups,
                query.stride(0), query.stride(1),
                key.stride(0), key.stride(1),
                value.stride(0), value.stride(1),
                output.stride(0), output.stride(1),
                sm_scale);
          }
        }
        else if (index_dtype == at::ScalarType::Long)
        {
          if (is_sm75_varlen_device())
          {
            launch_qk_int_sv_f16_varlen_attn<64, 64, 16, 64, HEAD_DIM, int64_t, DTypeOut, mask_mode>(
                query, key, value, output, query_scale, key_scale, cu_seqlens_q, cu_seqlens_k,
                batch_size, max_seqlen_q, max_seqlen_k, num_qo_heads, num_kv_heads, num_kv_groups,
                query.stride(0), query.stride(1),
                key.stride(0), key.stride(1),
                value.stride(0), value.stride(1),
                output.stride(0), output.stride(1),
                sm_scale);
          }
          else
          {
            launch_qk_int_sv_f16_varlen_attn<128, 64, 32, 64, HEAD_DIM, int64_t, DTypeOut, mask_mode>(
                query, key, value, output, query_scale, key_scale, cu_seqlens_q, cu_seqlens_k,
                batch_size, max_seqlen_q, max_seqlen_k, num_qo_heads, num_kv_heads, num_kv_groups,
                query.stride(0), query.stride(1),
                key.stride(0), key.stride(1),
                value.stride(0), value.stride(1),
                output.stride(0), output.stride(1),
                sm_scale);
          }
        }
        else
        {
          TORCH_CHECK(false, "cu_seqlens must be int32 or int64");
        }
      });
    });
  });

  return lse;
}
