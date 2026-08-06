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
  m.def("quant_per_block_int8_cuda", py::overload_cast<at::Tensor, at::Tensor, at::Tensor, float, int, int>(&quant_per_block_int8_cuda), "quant_per_block_int8_cuda");
  m.def("quant_per_block_int8_cuda", py::overload_cast<at::Tensor, at::Tensor, at::Tensor, int, int>(&quant_per_block_int8_cuda), "quant_per_block_int8_cuda");
  m.def("quant_per_block_int8_fuse_sub_mean_cuda", py::overload_cast<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, int>(&quant_per_block_int8_fuse_sub_mean_cuda), "quant_per_block_int8_fuse_sub_mean_cuda");
  m.def("quant_per_warp_int8_cuda", py::overload_cast<at::Tensor, at::Tensor, at::Tensor, int, int, int>(&quant_per_warp_int8_cuda), "quant_per_warp_int8_cuda");
  m.def("quant_qk_per_warp_int8_cuda", &quant_qk_per_warp_int8_cuda, "quant_qk_per_warp_int8_cuda");
  m.def("quant_per_warp_int8_varlen_cuda", &quant_per_warp_int8_varlen_cuda, "quant_per_warp_int8_varlen_cuda");
  m.def("token_block_mean_cuda", &token_block_mean_cuda, "token_block_mean_cuda");
  m.def("quant_query_per_thread_int4_cuda", &quant_query_per_thread_int4_cuda, "quant_query_per_thread_int4_cuda");
  m.def("quant_key_per_thread_int4_cuda", &quant_key_per_thread_int4_cuda, "quant_key_per_thread_int4_cuda");

  m.def("sub_mean_cuda", py::overload_cast<at::Tensor, at::Tensor, at::Tensor, int>(&sub_mean_cuda), "sub_mean_cuda");

  m.def("transpose_pad_permute_cuda", py::overload_cast<at::Tensor, at::Tensor, int>(&transpose_pad_permute_cuda), "transpose_pad_permute_cuda");
  m.def("scale_fuse_quant_cuda", py::overload_cast<at::Tensor, at::Tensor, at::Tensor, int, float, int>(&scale_fuse_quant_cuda), "scale_fuse_quant_cuda");
  m.def("mean_scale_fuse_quant_cuda", py::overload_cast<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, float, int>(&mean_scale_fuse_quant_cuda), "mean_scale_fuse_quant_cuda");
  m.def("varlen_attention_fwd_cuda", &varlen_attention_fwd_cuda, "varlen_attention_fwd_cuda");
}
