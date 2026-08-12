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

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "attn_cuda_sm75.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("qk_int8_sv_f16_accum_f32_attn", &qk_int8_sv_f16_accum_f32_attn, "QK int8 sv f16 accum f32 attn per warp");
  m.def("qk_int8_sv_f16_varlen_accum_f32_attn", &qk_int8_sv_f16_varlen_accum_f32_attn, "Varlen QK int8 sv f16 accum f32 attn per warp");
  m.def("sol_sparse_online_int8_f16_attn", &sol_sparse_online_int8_f16_attn, "Sol online-routed sparse attention with INT8 QK and FP16/BF16 V for sm75");
  m.def("sol_w8a8_precompute_summaries", &sol_w8a8_precompute_summaries, "Precompute Sol W8A8 correction summaries before releasing floating-point V");
  m.def("sol_sparse_online_w8a8_prequantized_attn", &sol_sparse_online_w8a8_prequantized_attn, "Sol W8A8 attention from prequantized Q/K/V and correction summaries");
  m.def("quantize_v_int8_sm75", &quantize_v_int8_sm75, "Channel-wise signed INT8 V quantization for sm75 W8A8 attention");
  m.def("quantize_v_int8_varlen_sm75", &quantize_v_int8_varlen_sm75, "Per-sequence channel-wise INT8 V quantization for packed sm75 W8A8 attention");
  m.def("qk_int8_sv_int8_varlen_accum_f32_attn", &qk_int8_sv_int8_varlen_accum_f32_attn, "Packed variable-length W8A8 attention for sm75");
}
