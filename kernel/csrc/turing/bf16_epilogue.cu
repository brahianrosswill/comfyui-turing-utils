/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Turing has no native FP32 -> BF16 conversion instruction.  This epilogue
 * keeps the dequantization arithmetic in FP32, performs the CUDA-compatible
 * round-to-nearest-even conversion with integer operations, and emits one
 * aligned 16-byte store for every eight BF16 results.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "kernel_api.h"

namespace comfyui_turing_utils::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kValuesPerThread = 8;

__device__ __forceinline__ uint16_t float_to_bf16_rn_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        // Match CUDA's pre-sm80 software conversion, including its canonical
        // positive NaN payload.
        return 0x7fffU;
    }
    uint16_t upper = static_cast<uint16_t>(bits >> 16U);
    const uint32_t remainder = bits << 16U;
    if (remainder > 0x80000000U ||
        (remainder == 0x80000000U && (upper & 1U) != 0U)) {
        ++upper;
    }
    return upper;
}

__device__ __forceinline__ uint32_t pack_bf16(float low, float high) {
    return static_cast<uint32_t>(float_to_bf16_rn_bits(low)) |
           (static_cast<uint32_t>(float_to_bf16_rn_bits(high)) << 16U);
}

template <bool ScalarWeightScale>
__global__ void dequantize_int8_bf16_vec8_kernel(
    const int32_t *__restrict__ accumulator,
    const float *__restrict__ activation_scale,
    const float *__restrict__ weight_scale,
    uint4 *__restrict__ output,
    int rows,
    int columns,
    int accumulator_stride) {
    const int vectors_per_row = columns / kValuesPerThread;
    const int64_t vector_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t vector_count = static_cast<int64_t>(rows) * vectors_per_row;
    if (vector_index >= vector_count) {
        return;
    }

    const int row = static_cast<int>(vector_index / vectors_per_row);
    const int column =
        static_cast<int>(vector_index - static_cast<int64_t>(row) * vectors_per_row) *
        kValuesPerThread;
    const int32_t *input_row =
        accumulator + static_cast<int64_t>(row) * accumulator_stride;
    const int4 acc0 = *reinterpret_cast<const int4 *>(input_row + column);
    const int4 acc1 = *reinterpret_cast<const int4 *>(input_row + column + 4);
    const float x_scale = activation_scale[row];

    float w0;
    float w1;
    float w2;
    float w3;
    float w4;
    float w5;
    float w6;
    float w7;
    if constexpr (ScalarWeightScale) {
        w0 = weight_scale[0];
        w1 = w0;
        w2 = w0;
        w3 = w0;
        w4 = w0;
        w5 = w0;
        w6 = w0;
        w7 = w0;
    } else {
        const float4 scales0 =
            *reinterpret_cast<const float4 *>(weight_scale + column);
        const float4 scales1 =
            *reinterpret_cast<const float4 *>(weight_scale + column + 4);
        w0 = scales0.x;
        w1 = scales0.y;
        w2 = scales0.z;
        w3 = scales0.w;
        w4 = scales1.x;
        w5 = scales1.y;
        w6 = scales1.z;
        w7 = scales1.w;
    }

    // Preserve comfy-kitchen's operation order: (acc * row_scale) * weight_scale.
    const float v0 = static_cast<float>(acc0.x) * x_scale * w0;
    const float v1 = static_cast<float>(acc0.y) * x_scale * w1;
    const float v2 = static_cast<float>(acc0.z) * x_scale * w2;
    const float v3 = static_cast<float>(acc0.w) * x_scale * w3;
    const float v4 = static_cast<float>(acc1.x) * x_scale * w4;
    const float v5 = static_cast<float>(acc1.y) * x_scale * w5;
    const float v6 = static_cast<float>(acc1.z) * x_scale * w6;
    const float v7 = static_cast<float>(acc1.w) * x_scale * w7;

    output[vector_index] = make_uint4(
        pack_bf16(v0, v1),
        pack_bf16(v2, v3),
        pack_bf16(v4, v5),
        pack_bf16(v6, v7));
}

template <bool ScalarWeightScale>
__global__ void dequantize_int8_bf16_scalar_kernel(
    const int32_t *__restrict__ accumulator,
    const float *__restrict__ activation_scale,
    const float *__restrict__ weight_scale,
    uint16_t *__restrict__ output,
    int64_t total,
    int columns,
    int accumulator_stride) {
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int row = static_cast<int>(index / columns);
    const int column = static_cast<int>(index - static_cast<int64_t>(row) * columns);
    const int32_t acc = accumulator[
        static_cast<int64_t>(row) * accumulator_stride + column];
    const float w_scale = weight_scale[ScalarWeightScale ? 0 : column];
    const float value = static_cast<float>(acc) * activation_scale[row] * w_scale;
    output[index] = float_to_bf16_rn_bits(value);
}

}  // namespace

void turing_dequantize_int8_bf16(Tensor accumulator,
                                  Tensor activation_scale,
                                  Tensor weight_scale,
                                  Tensor output) {
    const int rows = output.size(0);
    const int columns = output.size(1);
    const int accumulator_stride = static_cast<int>(accumulator.stride(0));
    const bool scalar_weight_scale = weight_scale.numel() == 1;
    const bool vectorized =
        columns % kValuesPerThread == 0 &&
        accumulator_stride % 4 == 0 &&
        (reinterpret_cast<std::uintptr_t>(accumulator.ptr) & 15U) == 0U &&
        (reinterpret_cast<std::uintptr_t>(output.ptr) & 15U) == 0U &&
        (scalar_weight_scale ||
         (reinterpret_cast<std::uintptr_t>(weight_scale.ptr) & 15U) == 0U);

    if (vectorized) {
        const int64_t vector_count =
            static_cast<int64_t>(rows) * (columns / kValuesPerThread);
        if (vector_count > 0) {
            const int blocks = static_cast<int>(ceilDiv(vector_count, kThreads));
            if (scalar_weight_scale) {
                dequantize_int8_bf16_vec8_kernel<true>
                    <<<blocks, kThreads, 0, getCurrentCUDAStream()>>>(
                        static_cast<const int32_t *>(accumulator.ptr),
                        static_cast<const float *>(activation_scale.ptr),
                        static_cast<const float *>(weight_scale.ptr),
                        static_cast<uint4 *>(output.ptr),
                        rows,
                        columns,
                        accumulator_stride);
            } else {
                dequantize_int8_bf16_vec8_kernel<false>
                    <<<blocks, kThreads, 0, getCurrentCUDAStream()>>>(
                        static_cast<const int32_t *>(accumulator.ptr),
                        static_cast<const float *>(activation_scale.ptr),
                        static_cast<const float *>(weight_scale.ptr),
                        static_cast<uint4 *>(output.ptr),
                        rows,
                        columns,
                        accumulator_stride);
            }
        }
    } else {
        const int64_t total = static_cast<int64_t>(rows) * columns;
        if (total > 0) {
            const int blocks = static_cast<int>(ceilDiv(total, kThreads));
            if (scalar_weight_scale) {
                dequantize_int8_bf16_scalar_kernel<true>
                    <<<blocks, kThreads, 0, getCurrentCUDAStream()>>>(
                        static_cast<const int32_t *>(accumulator.ptr),
                        static_cast<const float *>(activation_scale.ptr),
                        static_cast<const float *>(weight_scale.ptr),
                        static_cast<uint16_t *>(output.ptr),
                        total,
                        columns,
                        accumulator_stride);
            } else {
                dequantize_int8_bf16_scalar_kernel<false>
                    <<<blocks, kThreads, 0, getCurrentCUDAStream()>>>(
                        static_cast<const int32_t *>(accumulator.ptr),
                        static_cast<const float *>(activation_scale.ptr),
                        static_cast<const float *>(weight_scale.ptr),
                        static_cast<uint16_t *>(output.ptr),
                        total,
                        columns,
                        accumulator_stride);
            }
        }
    }
    checkCUDA(cudaGetLastError());
}

}  // namespace comfyui_turing_utils::kernels
