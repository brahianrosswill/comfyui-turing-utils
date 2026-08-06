/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modified for ComfyUI Turing Utils: add affine RMSNorm weights and an explicit segmented
 * token-to-modulation mapping while preserving the comfy-kitchen FP32
 * reduction and fused AdaLN epilogue.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "kernel_api.h"

namespace comfyui_turing_utils::kernels {
namespace {

constexpr int kWarpThreads = 32;
constexpr int kNormThreads = 256;
constexpr int kNormWarps = kNormThreads / kWarpThreads;

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float to_float<half>(half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ float from_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ half from_float<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
struct VecWidth {
    static constexpr int value = 16 / sizeof(T);
};

template <typename T>
struct alignas(16) Vec {
    static constexpr int kWidth = VecWidth<T>::value;
    T values[kWidth];
};

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = kWarpThreads / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ int find_modulation_row(
    const int *__restrict__ segments, int segment_count, int token_row) {
    int low = 0;
    int high = segment_count;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        if (token_row < segments[mid * 3 + 1]) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    if (low >= segment_count || token_row < segments[low * 3]) {
        return -1;
    }
    return segments[low * 3 + 2];
}

template <typename T>
__global__ void segmented_rms_adaln_kernel(
    const T *__restrict__ input,
    const T *__restrict__ weight,
    const T *__restrict__ scale,
    const T *__restrict__ shift,
    const int *__restrict__ segments,
    T *__restrict__ output,
    int hidden,
    int parameter_rows,
    int scale_stride,
    int shift_stride,
    int segment_count,
    float epsilon,
    bool vectorized) {
    __shared__ float warp_sums[kNormWarps];
    __shared__ int modulation_row;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int lane = tid & (kWarpThreads - 1);
    const int warp = tid / kWarpThreads;
    const int64_t row_offset = static_cast<int64_t>(row) * hidden;
    const T *input_row = input + row_offset;

    if (tid == 0) {
        const int mapped = find_modulation_row(segments, segment_count, row);
        modulation_row = mapped >= 0 && mapped < parameter_rows ? mapped : 0;
    }

    float square_sum = 0.0f;
    if (vectorized) {
        constexpr int kWidth = VecWidth<T>::value;
        const int vector_count = hidden / kWidth;
        const Vec<T> *input_vectors = reinterpret_cast<const Vec<T> *>(input_row);
        for (int index = tid; index < vector_count; index += kNormThreads) {
            const Vec<T> values = input_vectors[index];
#pragma unroll
            for (int item = 0; item < kWidth; ++item) {
                const float value = to_float(values.values[item]);
                square_sum += value * value;
            }
        }
    } else {
        for (int col = tid; col < hidden; col += kNormThreads) {
            const float value = to_float(input_row[col]);
            square_sum += value * value;
        }
    }

    square_sum = warp_reduce_sum(square_sum);
    if (lane == 0) {
        warp_sums[warp] = square_sum;
    }
    __syncthreads();

    float total = 0.0f;
#pragma unroll
    for (int item = 0; item < kNormWarps; ++item) {
        total += warp_sums[item];
    }
    const float inverse_rms = rsqrtf(total / static_cast<float>(hidden) + epsilon);
    const int mod_row = modulation_row;
    const T *scale_row = scale + static_cast<int64_t>(mod_row) * scale_stride;
    const T *shift_row = shift + static_cast<int64_t>(mod_row) * shift_stride;
    T *output_row = output + row_offset;

    if (vectorized) {
        constexpr int kWidth = VecWidth<T>::value;
        const int vector_count = hidden / kWidth;
        const Vec<T> *input_vectors = reinterpret_cast<const Vec<T> *>(input_row);
        const Vec<T> *weight_vectors = reinterpret_cast<const Vec<T> *>(weight);
        const Vec<T> *scale_vectors = reinterpret_cast<const Vec<T> *>(scale_row);
        const Vec<T> *shift_vectors = reinterpret_cast<const Vec<T> *>(shift_row);
        Vec<T> *output_vectors = reinterpret_cast<Vec<T> *>(output_row);
        for (int index = tid; index < vector_count; index += kNormThreads) {
            const Vec<T> inputs = input_vectors[index];
            const Vec<T> weights = weight_vectors[index];
            const Vec<T> scales = scale_vectors[index];
            const Vec<T> shifts = shift_vectors[index];
            Vec<T> outputs;
#pragma unroll
            for (int item = 0; item < kWidth; ++item) {
                const float normalized = to_float(inputs.values[item]) * inverse_rms;
                const float result = normalized * to_float(weights.values[item]) *
                                         (1.0f + to_float(scales.values[item])) +
                                     to_float(shifts.values[item]);
                outputs.values[item] = from_float<T>(result);
            }
            output_vectors[index] = outputs;
        }
    } else {
        for (int col = tid; col < hidden; col += kNormThreads) {
            const float normalized = to_float(input_row[col]) * inverse_rms;
            const float result = normalized * to_float(weight[col]) *
                                     (1.0f + to_float(scale_row[col])) +
                                 to_float(shift_row[col]);
            output_row[col] = from_float<T>(result);
        }
    }
}

template <typename T>
void launch_segmented_rms_adaln(Tensor input,
                                 Tensor weight,
                                 Tensor scale,
                                 Tensor shift,
                                 Tensor segments,
                                 Tensor output,
                                 float epsilon) {
    const int rows = input.size(0);
    const int hidden = input.size(1);
    const int parameter_rows = scale.size(0);
    const int scale_stride = static_cast<int>(scale.shape.stride(0));
    const int shift_stride = static_cast<int>(shift.shape.stride(0));
    const int segment_count = segments.size(0);
    constexpr int kWidth = VecWidth<T>::value;
    const auto aligned = [](const void *pointer) {
        return (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0;
    };
    const bool vectorized =
        hidden % kWidth == 0 && scale_stride % kWidth == 0 &&
        shift_stride % kWidth == 0 && aligned(input.ptr) && aligned(weight.ptr) &&
        aligned(scale.ptr) && aligned(shift.ptr) && aligned(output.ptr);

    segmented_rms_adaln_kernel<T><<<rows, kNormThreads, 0, getCurrentCUDAStream()>>>(
        static_cast<const T *>(input.ptr),
        static_cast<const T *>(weight.ptr),
        static_cast<const T *>(scale.ptr),
        static_cast<const T *>(shift.ptr),
        static_cast<const int *>(segments.ptr),
        static_cast<T *>(output.ptr),
        hidden,
        parameter_rows,
        scale_stride,
        shift_stride,
        segment_count,
        epsilon,
        vectorized);
    checkCUDA(cudaGetLastError());
}

}  // namespace

void turing_segmented_rms_adaln(Tensor input,
                                 Tensor weight,
                                 Tensor scale,
                                 Tensor shift,
                                 Tensor segments,
                                 Tensor output,
                                 float epsilon) {
    if (input.scalar_type() == Tensor::BF16) {
        launch_segmented_rms_adaln<nv_bfloat16>(
            input, weight, scale, shift, segments, output, epsilon);
    } else if (input.scalar_type() == Tensor::FP16) {
        launch_segmented_rms_adaln<half>(
            input, weight, scale, shift, segments, output, epsilon);
    } else if (input.scalar_type() == Tensor::FP32) {
        launch_segmented_rms_adaln<float>(
            input, weight, scale, shift, segments, output, epsilon);
    } else {
        throw std::runtime_error("segmented RMSNorm+AdaLN requires float16, bfloat16, or float32 input");
    }
}

}  // namespace comfyui_turing_utils::kernels
