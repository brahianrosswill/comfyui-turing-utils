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

void quant_per_block_int8_cuda(
                at::Tensor input,
                at::Tensor output,
                at::Tensor scale,
                float sm_scale,
                int block_size,
                int tensor_layout);

void quant_per_block_int8_cuda(
                at::Tensor input,
                at::Tensor output,
                at::Tensor scale,
                int block_size,
                int tensor_layout);


void quant_per_warp_int8_cuda(
                at::Tensor input,
                at::Tensor output,
                at::Tensor scale,
                int block_size,
                int warp_block_size,
                int tensor_layout);

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
                int tensor_layout);

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
                int tensor_layout);

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
                int tensor_layout);

void quant_qk_rms_rope_int8_cuda(
                at::Tensor query,
                at::Tensor key,
                at::Tensor query_output,
                at::Tensor key_output,
                at::Tensor query_scale,
                at::Tensor key_scale,
                at::Tensor query_norm,
                at::Tensor key_norm,
                at::Tensor freqs,
                at::Tensor query_rrms,
                at::Tensor key_rrms,
                at::Tensor anchor_indices,
                at::Tensor anchor_values,
                float epsilon,
                int rot_dim,
                int query_block_size,
                int query_warp_block_size,
                int key_block_size,
                int tensor_layout,
                int norm_scope,
                bool split_half,
                bool rotate);

void quant_per_warp_int8_varlen_cuda(
                at::Tensor input,
                at::Tensor cu_seqlens,
                at::Tensor output,
                at::Tensor scale,
                int max_seqlen,
                int block_size,
                int warp_block_size,
                bool rotate);


void varlen_attention_fwd_cuda(
                at::Tensor query,
                at::Tensor key,
                at::Tensor value,
                at::Tensor cu_seqlens_q,
                at::Tensor cu_seqlens_k,
                at::Tensor output,
                int max_seqlen_q,
                float sm_scale,
                int is_causal);

at::Tensor overlap_blend_cuda(
                at::Tensor window_values,
                at::Tensor local_indices,
                at::Tensor weights);
