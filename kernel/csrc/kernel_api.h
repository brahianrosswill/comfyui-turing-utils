#pragma once

#include "tensor_bridge.h"

namespace comfyui_turing_utils::kernels {

void turing_w4a8_linear(Tensor activation,
                        Tensor weight,
                        Tensor activation_scale,
                        Tensor weight_scale,
                        Tensor bias,
                        Tensor output,
                        int tile_policy);

void turing_codebook_w4a8_linear(Tensor activation,
                                 Tensor weight,
                                 Tensor activation_scale,
                                 Tensor group_scale,
                                 Tensor channel_scale,
                                 Tensor codebook,
                                 Tensor bias,
                                 Tensor workspace,
                                 Tensor output,
                                 int group_size,
                                 bool inline_decode,
                                 int tile_policy);

void turing_int8_linear(Tensor activation,
                        Tensor weight,
                        Tensor activation_scale,
                        Tensor weight_scale,
                        Tensor bias,
                        Tensor output,
                        int tile_policy);

void turing_dequantize_int8_bf16(Tensor accumulator,
                                 Tensor activation_scale,
                                 Tensor weight_scale,
                                 Tensor output);

void turing_swiglu_int8_convrot_quantize(Tensor input,
                                          Tensor rotated,
                                          Tensor partial_absmax,
                                          Tensor output,
                                          Tensor scales);

void turing_swiglu_int8_convrot_quantize_scaled(Tensor input,
                                                 Tensor scales,
                                                 Tensor output);

void turing_swiglu_int4_convrot_quantize(Tensor input,
                                          Tensor rotated,
                                          Tensor partial_absmax,
                                          Tensor output,
                                          Tensor scales);

void turing_gelu_int8_convrot_quantize(Tensor input,
                                        Tensor rotated,
                                        Tensor partial_absmax,
                                        Tensor output,
                                        Tensor scales);

void turing_gelu_int4_convrot_quantize(Tensor input,
                                        Tensor rotated,
                                        Tensor partial_absmax,
                                        Tensor output,
                                        Tensor scales);

void turing_bf16_int8_convrot_quantize(Tensor input,
                                        Tensor output,
                                        Tensor scales,
                                        bool swiglu,
                                        int block_threads);

void turing_bf16_int4_convrot_quantize(Tensor input,
                                        Tensor output,
                                        Tensor scales,
                                        bool swiglu,
                                        int block_threads);

void turing_bf16_gelu_int8_convrot_quantize(Tensor input,
                                             Tensor output,
                                             Tensor scales,
                                             int block_threads);

void turing_bf16_gelu_int4_convrot_quantize(Tensor input,
                                             Tensor output,
                                             Tensor scales,
                                             int block_threads);

void turing_segmented_rms_adaln(Tensor input,
                                 Tensor weight,
                                 Tensor scale,
                                 Tensor shift,
                                 Tensor segments,
                                 Tensor output,
                                 float epsilon);

void turing_layer_norm_adaln(Tensor input,
                              Tensor scale,
                              Tensor shift,
                              Tensor output,
                              float epsilon);

}  // namespace comfyui_turing_utils::kernels
