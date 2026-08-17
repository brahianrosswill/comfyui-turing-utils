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

#include <torch/extension.h>
#include <cuda_fp16.h>
#include "fused.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("quant_per_block_int8_cuda", py::overload_cast<at::Tensor, at::Tensor, at::Tensor, int, int>(&quant_per_block_int8_cuda), "quant_per_block_int8_cuda");
  m.def("quant_per_warp_int8_cuda", py::overload_cast<at::Tensor, at::Tensor, at::Tensor, int, int, int>(&quant_per_warp_int8_cuda), "quant_per_warp_int8_cuda");
  m.def("quant_qk_per_warp_int8_cuda", &quant_qk_per_warp_int8_cuda, "quant_qk_per_warp_int8_cuda");
  m.def("quant_qk_per_warp_int8_rotated_cuda", &quant_qk_per_warp_int8_rotated_cuda, "Fused randomized Hadamard Q/K INT8 quantization");
  m.def("quant_qk_per_warp_int8_rotated_anchored_cuda", &quant_qk_per_warp_int8_rotated_anchored_cuda, "Fused randomized Hadamard Q/K INT8 quantization with conditional K anchoring");
  m.def("quant_qk_rms_rope_int8_cuda", &quant_qk_rms_rope_int8_cuda, "Fused RMSNorm, RoPE, optional Hadamard/anchor stabilization, and Q/K INT8 quantization");
  m.def("quant_per_warp_int8_varlen_cuda", &quant_per_warp_int8_varlen_cuda, "quant_per_warp_int8_varlen_cuda");

  m.def("varlen_attention_fwd_cuda", &varlen_attention_fwd_cuda, "varlen_attention_fwd_cuda");
  m.def("overlap_blend_cuda", &overlap_blend_cuda, "Deterministic FP32 multi-window overlap epilogue");
  m.def("overlap_accumulate_cuda", &overlap_accumulate_cuda, "Streaming deterministic FP32 overlap accumulation");
}
