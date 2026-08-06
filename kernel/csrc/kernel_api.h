#pragma once

#include "runtime.h"

namespace svdint4::kernels {

void gemm_svd(Tensor act,
              Tensor wgt,
              Tensor out,
              Tensor ascales,
              Tensor wscales,
              Tensor lora_act,
              Tensor lora_up,
              Tensor bias,
              bool act_unsigned,
              std::vector<float> lora_scales);

void quantize_act_lora(Tensor input,
                       Tensor output,
                       Tensor oscales,
                       Tensor lora_down,
                       Tensor lora_act_out,
                       Tensor smooth);

void turing_w4a8_linear(Tensor activation,
                        Tensor weight,
                        Tensor activation_scale,
                        Tensor weight_scale,
                        Tensor bias,
                        Tensor output);

void turing_dequantize_int8_bf16(Tensor accumulator,
                                 Tensor activation_scale,
                                 Tensor weight_scale,
                                 Tensor output);

void turing_swiglu_int8_convrot_quantize(Tensor input,
                                          Tensor rotated,
                                          Tensor partial_absmax,
                                          Tensor output,
                                          Tensor scales);

void turing_bf16_int8_convrot_quantize(Tensor input,
                                        Tensor output,
                                        Tensor scales,
                                        bool swiglu,
                                        int block_threads);

void turing_segmented_rms_adaln(Tensor input,
                                 Tensor weight,
                                 Tensor scale,
                                 Tensor shift,
                                 Tensor segments,
                                 Tensor output,
                                 float epsilon);

}  // namespace svdint4::kernels
