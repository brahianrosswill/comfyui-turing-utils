/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modified for svdint4: fold SwiGLU into the first pass of the staged
 * ConvRot INT8 activation quantizer.  The FHT, scale, and rounding order
 * intentionally match comfy-kitchen's CUDA implementation.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

#include "kernel_api.h"

namespace svdint4::kernels {
namespace {

constexpr int kWarpThreads = 32;
constexpr int kConvRotGroup = 256;
constexpr int kGroupThreads = 64;
constexpr int kGroupsPerBlock = 8;
constexpr int kRotateThreads = kGroupsPerBlock * kGroupThreads;
constexpr int kQuantThreads = 512;

template <typename T>
__device__ __forceinline__ float to_float(T value);

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
__device__ __forceinline__ half from_float<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__device__ __forceinline__ float finite_max();

template <>
__device__ __forceinline__ float finite_max<half>() {
    return 65504.0f;
}

template <>
__device__ __forceinline__ float finite_max<nv_bfloat16>() {
    return 3.38953139e38f;
}

template <typename T>
__device__ __forceinline__ float quant_div_to_float(T value, float scale) {
    const float rounded_scale = to_float(from_float<T>(scale));
    return to_float(from_float<T>(to_float(value) / rounded_scale));
}

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
    for (int offset = kWarpThreads / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

template <int Warps>
__device__ __forceinline__ float block_reduce_max(
    float value, float *warp_values, float *block_value) {
    const int lane = threadIdx.x & (kWarpThreads - 1);
    const int warp = threadIdx.x / kWarpThreads;
    value = warp_reduce_max(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        float total = lane < Warps ? warp_values[lane] : 0.0f;
        total = warp_reduce_max(total);
        if (lane == 0) {
            *block_value = total;
        }
    }
    __syncthreads();
    return *block_value;
}

template <int S>
__device__ __forceinline__ void fht_stage(
    const float *__restrict__ src, float *__restrict__ dst, int lane) {
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    dst[base] = 0.5f * (x0 + x1 + x2 - x3);
    dst[base + S] = 0.5f * (x0 + x1 - x2 + x3);
    dst[base + 2 * S] = 0.5f * (x0 - x1 + x2 + x3);
    dst[base + 3 * S] = 0.5f * (-x0 + x1 + x2 + x3);
}

template <int S, typename OutputType>
__device__ __forceinline__ float fht_store_absmax(
    const float *__restrict__ src, OutputType *__restrict__ output, int lane) {
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    const float y0 = 0.5f * (x0 + x1 + x2 - x3);
    const float y1 = 0.5f * (x0 + x1 - x2 + x3);
    const float y2 = 0.5f * (x0 - x1 + x2 + x3);
    const float y3 = 0.5f * (-x0 + x1 + x2 + x3);
    output[base] = from_float<OutputType>(y0);
    output[base + S] = from_float<OutputType>(y1);
    output[base + 2 * S] = from_float<OutputType>(y2);
    output[base + 3 * S] = from_float<OutputType>(y3);
    return fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3)));
}

template <typename InputType>
__device__ __forceinline__ float load_swiglu(
    const InputType *__restrict__ input, int64_t row_offset, int col, int k) {
    const float gate = to_float(input[row_offset + col]);
    const float up = to_float(input[row_offset + k + col]);
    return (gate / (1.0f + expf(-gate))) * up;
}

template <typename InputType>
__global__ void swiglu_rotate_amax_kernel(
    const InputType *__restrict__ input,
    InputType *__restrict__ rotated,
    float *__restrict__ partial_absmax,
    int k) {
    extern __shared__ float smem[];
    const int sub = threadIdx.x / kGroupThreads;
    const int lane = threadIdx.x % kGroupThreads;
    const int group = static_cast<int>(blockIdx.y) * kGroupsPerBlock + sub;
    const int row = static_cast<int>(blockIdx.x);
    const int groups = k / kConvRotGroup;
    const bool active = group < groups;
    const int group_col = group * kConvRotGroup;
    const int base = lane * 4;
    const int col = group_col + base;
    const int64_t input_row = static_cast<int64_t>(row) * (2 * k);
    const int64_t output_row = static_cast<int64_t>(row) * k;

    float *buf0 = smem + sub * (2 * kConvRotGroup);
    float *buf1 = buf0 + kConvRotGroup;

    const float x0 = active ? load_swiglu(input, input_row, col, k) : 0.0f;
    const float x1 = active ? load_swiglu(input, input_row, col + 1, k) : 0.0f;
    const float x2 = active ? load_swiglu(input, input_row, col + 2, k) : 0.0f;
    const float x3 = active ? load_swiglu(input, input_row, col + 3, k) : 0.0f;
    buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
    buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
    buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
    buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();

    fht_stage<4>(buf1, buf0, lane);
    __syncthreads();
    fht_stage<16>(buf0, buf1, lane);
    __syncthreads();

    float local_max = 0.0f;
    if (active) {
        local_max = fht_store_absmax<64>(
            buf1, rotated + output_row + group_col, lane);
    }
    buf0[lane] = local_max;
    __syncthreads();

    if (lane < 32) {
        float value = fmaxf(buf0[lane], buf0[lane + 32]);
        value = warp_reduce_max(value);
        if (lane == 0 && active) {
            partial_absmax[static_cast<int64_t>(row) * groups + group] = value;
        }
    }
}

template <typename InputType>
__global__ void quantize_from_partials_kernel(
    const InputType *__restrict__ rotated,
    const float *__restrict__ partial_absmax,
    int8_t *__restrict__ output,
    float *__restrict__ scales,
    int k) {
    constexpr int kWarps = kQuantThreads / kWarpThreads;
    __shared__ float warp_values[kWarps];
    __shared__ float block_value;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int groups = k / kConvRotGroup;
    const int64_t row_offset = static_cast<int64_t>(row) * k;
    const float *row_partials = partial_absmax + static_cast<int64_t>(row) * groups;

    float abs_max = 0.0f;
    for (int group = tid; group < groups; group += kQuantThreads) {
        abs_max = fmaxf(abs_max, row_partials[group]);
    }
    abs_max = block_reduce_max<kWarps>(abs_max, warp_values, &block_value);
    const float scale = fmaxf(fminf(abs_max, finite_max<InputType>()) / 127.0f, 1.0e-30f);
    if (tid == 0) {
        scales[row] = scale;
    }

    for (int col = tid; col < k; col += kQuantThreads) {
        const int64_t index = row_offset + col;
        float quantized = nearbyintf(quant_div_to_float(rotated[index], scale));
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        output[index] = static_cast<int8_t>(quantized);
    }
}

template <typename InputType>
void launch_swiglu_quantize(Tensor input,
                            Tensor rotated,
                            Tensor partial_absmax,
                            Tensor output,
                            Tensor scales) {
    const int rows = input.size(0);
    const int k = output.size(1);
    const int group_blocks = ceilDiv(k / kConvRotGroup, kGroupsPerBlock);
    const dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(group_blocks));
    constexpr size_t smem_bytes =
        kGroupsPerBlock * 2 * kConvRotGroup * sizeof(float);
    swiglu_rotate_amax_kernel<InputType><<<grid, kRotateThreads, smem_bytes, getCurrentCUDAStream()>>>(
        static_cast<const InputType *>(input.ptr),
        static_cast<InputType *>(rotated.ptr),
        static_cast<float *>(partial_absmax.ptr),
        k);
    checkCUDA(cudaGetLastError());

    quantize_from_partials_kernel<InputType><<<rows, kQuantThreads, 0, getCurrentCUDAStream()>>>(
        static_cast<const InputType *>(rotated.ptr),
        static_cast<const float *>(partial_absmax.ptr),
        static_cast<int8_t *>(output.ptr),
        static_cast<float *>(scales.ptr),
        k);
    checkCUDA(cudaGetLastError());
}

}  // namespace

void turing_swiglu_int8_convrot_quantize(Tensor input,
                                          Tensor rotated,
                                          Tensor partial_absmax,
                                          Tensor output,
                                          Tensor scales) {
    if (input.scalar_type() == Tensor::BF16) {
        launch_swiglu_quantize<nv_bfloat16>(input, rotated, partial_absmax, output, scales);
    } else if (input.scalar_type() == Tensor::FP16) {
        launch_swiglu_quantize<half>(input, rotated, partial_absmax, output, scales);
    } else {
        throw std::runtime_error("SwiGLU staged ConvRot requires float16 or bfloat16 input");
    }
}

}  // namespace svdint4::kernels
