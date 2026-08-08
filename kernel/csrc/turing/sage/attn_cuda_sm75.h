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

at::Tensor sol_sparse_threshold_f16_attn(at::Tensor query,
                    at::Tensor key,
                    at::Tensor value,
                    at::Tensor output,
                    int prefix_tokens,
                    float threshold_sigma,
                    int local_block_radius,
                    int topology_start_tokens,
                    int topology_tokens,
                    int tokens_per_frame,
                    int temporal_neighbor_frames,
                    float softmax_scale);

at::Tensor sol_sparse_route_selected(at::Tensor route);

#ifdef COMFYUI_TURING_UTILS_EXPERIMENTAL_SAGE_VARIANTS
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
                    int return_lse);

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
                    int return_lse);

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
                    int return_lse);
#endif

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

#ifdef COMFYUI_TURING_UTILS_EXPERIMENTAL_SAGE_VARIANTS
at::Tensor qk_int4_sv_f16_accum_f16_f32_attn(
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
                    int smooth_k);

at::Tensor qk_int4_sv_f16_accum_f16_f32_precomputed_attn(
                    at::Tensor query,
                    at::Tensor key,
                    at::Tensor value,
                    at::Tensor output,
                    at::Tensor query_scale,
                    at::Tensor key_scale,
                    at::Tensor score_correction,
                    int tensor_layout,
                    int is_causal,
                    float sm_scale,
                    int return_lse,
                    int q_block_start,
                    int q_block_count);
#endif
