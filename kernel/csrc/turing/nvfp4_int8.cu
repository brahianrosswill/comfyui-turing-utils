// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "kernel_api.h"

namespace comfyui_turing_utils::kernels {
namespace {

__device__ __forceinline__ float decode_e2m1(uint8_t code) {
    const float value = float((0xc8643210u >> ((code & 7) * 4)) & 15u) * 0.5f;
    return code & 8 ? -value : value;
}

__device__ __forceinline__ float load_swizzled_scale(
    const uint8_t *scale,
    int row,
    int col,
    int scale_cols) {
    constexpr int rows_per_block = 128;
    constexpr int rows_per_column = 32;
    constexpr int columns_per_group = 4;
    const int row_block = row / rows_per_block;
    const int row_remainder = row % rows_per_block;
    const int row_quadrant = row_remainder / rows_per_column;
    const int row_inner = row_remainder % rows_per_column;
    const int column_group = col / columns_per_group;
    const int column_inner = col % columns_per_group;
    const int column_group_count = (scale_cols + columns_per_group - 1) / columns_per_group;
    const int64_t offset =
        ((static_cast<int64_t>(row_block) * column_group_count + column_group) *
             rows_per_column +
         row_inner) *
            16 +
        row_quadrant * columns_per_group + column_inner;
    return __half2float(__nv_cvt_fp8_to_halfraw(scale[offset], __NV_E4M3));
}

__device__ __forceinline__ void convrot256(float (&v)[8]) {
#pragma unroll
    for (int step = 1; step <= 4; step *= 4) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            // H4's only negative entry is the anti-diagonal. XOR gathers
            // avoid four divergent cases; only FP32 addition order changes.
            const float b = __shfl_xor_sync(0xffffffff, v[j], step);
            const float c = __shfl_xor_sync(0xffffffff, v[j], 2 * step);
            const float d = __shfl_xor_sync(0xffffffff, v[j], 3 * step);
            v[j] = .5f * (v[j] + b + c - d);
        }
    }
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        const float a = v[j], c = v[j + 1];
        const float b = __shfl_xor_sync(0xffffffff, a, 16);
        const float d = __shfl_xor_sync(0xffffffff, c, 16);
        v[j] = .5f * (a + b + c - d);
        v[j + 1] = .5f * (c + d + a - b);
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const float a = v[j], b = v[j + 2], c = v[j + 4], d = v[j + 6];
        v[j] = .5f * (a + b + c - d);
        v[j + 2] = .5f * (a + b - c + d);
        v[j + 4] = .5f * (a - b + c + d);
        v[j + 6] = .5f * (-a + b + c + d);
    }
}

// Each warp rotates 256 columns at a time. Keep the row in registers until
// its scale is reduced; neither global FP32 scratch nor row-sized SMEM.
template <int GroupsPerWarp>
__global__ void nvfp4_convrot_s8_kernel(
    const uint8_t *weight, const uint8_t *blocks, const float *tensor_scale,
    int8_t *output, float *scales, int k, int scale_cols, int first_row, int stored_k, int logical_k) {
    __shared__ float warp_max[8];
    const int row = first_row + blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    float values[GroupsPerWarp][8];
    float maximum = 0.0f;
    bool nan = false;
#pragma unroll
    for (int group = 0; group < GroupsPerWarp; ++group) {
        const int base = (warp + group * 8) * 256;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = base + lane + j * 32;
            float value = 0.0f;
            if (col < logical_k) {
                const uint8_t packed = weight[int64_t(row) * (stored_k / 2) + col / 2];
                value = decode_e2m1((col & 1) ? packed & 15 : packed >> 4) *
                    (load_swizzled_scale(blocks, row, col / 16, scale_cols) * tensor_scale[0]);
            }
            values[group][j] = value;
        }
        convrot256(values[group]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            maximum = fmaxf(maximum, fabsf(values[group][j]));
            nan |= isnan(values[group][j]);
        }
    }
    const bool row_nan = __syncthreads_or(nan);
#pragma unroll
    for (int offset = 16; offset; offset /= 2)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x / 32] = maximum;
    __syncthreads();
    if (threadIdx.x < 32) {
        maximum = threadIdx.x < 8 ? warp_max[threadIdx.x] : 0.0f;
#pragma unroll
        for (int offset = 16; offset; offset /= 2)
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
        if (threadIdx.x == 0) {
            warp_max[0] = row_nan ? nanf("") : fmaxf(maximum / 127.0f, 1.0e-30f);
            scales[blockIdx.x] = warp_max[0];
        }
    }
    __syncthreads();
    // A row shares one scale. Reuse its rounded FP32 reciprocal; only values
    // at an S8 rounding boundary can differ by one code from division.
    const float reciprocal = __frcp_rn(warp_max[0]);
#pragma unroll
    for (int group = 0; group < GroupsPerWarp; ++group) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = (warp + group * 8) * 256 + lane + j * 32;
            if (col < k) {
                const float q = nearbyintf(__fmul_rn(values[group][j], reciprocal));
                output[int64_t(blockIdx.x) * k + col] = isnan(q) ? 0 : int8_t(fminf(127.0f, fmaxf(-128.0f, q)));
            }
        }
    }
}

// Unbounded-K specialization: stream rotation blocks twice, avoiding a
// register allocation proportional to K. Same quantization, no new format.
__global__ void nvfp4_convrot_s8_large_kernel(
    const uint8_t *weight, const uint8_t *blocks, const float *tensor_scale,
    int8_t *output, float *scales, int k, int scale_cols, int first_row,
    int stored_k, int logical_k) {
    __shared__ float reductions[8];
    const int row = first_row + blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    float maximum = 0.0f;
    bool nan = false;
    for (int pass = 0; pass < 2; ++pass) {
        for (int base = warp * 256; base < k; base += 2048) {
            float v[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = base + lane + j * 32;
                v[j] = 0.0f;
                if (col < logical_k) {
                    const uint8_t p = weight[int64_t(row) * (stored_k / 2) + col / 2];
                    v[j] = decode_e2m1((col & 1) ? p & 15 : p >> 4) *
                        (load_swizzled_scale(blocks, row, col / 16, scale_cols) * tensor_scale[0]);
                }
            }
            convrot256(v);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (!pass) {
                    maximum = fmaxf(maximum, fabsf(v[j]));
                    nan |= isnan(v[j]);
                } else {
                    const float q = nearbyintf(__fdiv_rn(v[j], reductions[0]));
                    output[int64_t(blockIdx.x) * k + base + lane + j * 32] =
                        isnan(q) ? 0 : int8_t(fminf(127.0f, fmaxf(-128.0f, q)));
                }
            }
        }
        if (!pass) {
            const bool row_nan = __syncthreads_or(nan);
            for (int offset = 16; offset; offset /= 2)
                maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
            if (!lane) reductions[warp] = maximum;
            __syncthreads();
            if (threadIdx.x == 0) {
                for (int i = 1; i < 8; ++i) maximum = fmaxf(maximum, reductions[i]);
                scales[blockIdx.x] = reductions[0] =
                    row_nan ? nanf("") : fmaxf(maximum / 127.0f, 1.0e-30f);
            }
            __syncthreads();
        }
    }
}
}  // namespace

void turing_nvfp4_convrot_quantize(Tensor weight, Tensor blocks, Tensor tensor_scale,
                                 Tensor output, Tensor scales, int first_row, int logical_k) {
    const int k = output.size(1);
    const auto launch = [&](auto tag) {
        nvfp4_convrot_s8_kernel<decltype(tag)::value><<<output.size(0), 256, 0, getCurrentCUDAStream()>>>(
            weight.data_ptr<uint8_t>(), blocks.data_ptr<uint8_t>(), tensor_scale.data_ptr<float>(),
            output.data_ptr<int8_t>(), scales.data_ptr<float>(), k, blocks.size(1), first_row, weight.size(1) * 2, logical_k);
    };
    if (k <= 2048) launch(std::integral_constant<int, 1>{});
    else if (k <= 4096) launch(std::integral_constant<int, 2>{});
    else if (k <= 6144) launch(std::integral_constant<int, 3>{});
    else if (k <= 8192) launch(std::integral_constant<int, 4>{});
    else if (k <= 14336) launch(std::integral_constant<int, 7>{});
    else if (k <= 16384) launch(std::integral_constant<int, 8>{});
    else nvfp4_convrot_s8_large_kernel<<<output.size(0), 256, 0, getCurrentCUDAStream()>>>(
        weight.data_ptr<uint8_t>(), blocks.data_ptr<uint8_t>(), tensor_scale.data_ptr<float>(),
        output.data_ptr<int8_t>(), scales.data_ptr<float>(), k, blocks.size(1), first_row,
        weight.size(1) * 2, logical_k);
    checkCUDA(cudaGetLastError());
}

}  // namespace comfyui_turing_utils::kernels
