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

#include <vector>

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
                    int return_lse);

at::Tensor sol_sparse_online_int8_f16_attn(at::Tensor query_int8,
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
                    int key_tile_tokens,
                    int is_causal,
                    int route_original_basis);

std::vector<at::Tensor> sol_w8a8_precompute_summaries(
                    at::Tensor key_int8,
                    at::Tensor key_scale,
                    at::Tensor value,
                    at::Tensor value_scale,
                    int residual_subblocks,
                    int route_original_basis);

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
                    int key_tile_tokens,
                    int is_causal,
                    int route_original_basis);

void quantize_v_int8_sm75(at::Tensor value,
                    at::Tensor quantized,
                    at::Tensor scale);

void quantize_v_int8_varlen_sm75(at::Tensor value,
                    at::Tensor cu_seqlens_k,
                    at::Tensor value_offsets,
                    at::Tensor quantized,
                    at::Tensor scale);

void qk_int8_sv_int8_varlen_accum_f32_attn(
                    at::Tensor query_int8,
                    at::Tensor key_int8,
                    at::Tensor value_int8,
                    at::Tensor value_scale,
                    at::Tensor output,
                    at::Tensor query_scale,
                    at::Tensor key_scale,
                    at::Tensor cu_seqlens_q,
                    at::Tensor cu_seqlens_k,
                    at::Tensor value_offsets,
                    int max_seqlen_q,
                    int max_seqlen_k,
                    int is_causal,
                    float softmax_scale);


at::Tensor qk_int8_sv_f16_varlen_accum_f32_attn(at::Tensor query,
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
                    float sm_scale);
