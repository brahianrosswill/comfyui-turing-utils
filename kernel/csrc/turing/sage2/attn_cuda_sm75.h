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
