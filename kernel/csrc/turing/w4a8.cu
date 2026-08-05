#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "kernel_api.h"

namespace svdint4::kernels {
namespace {

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 64;
constexpr int STATIC_SHARED_BYTES = (TILE_M + TILE_N) * TILE_K * sizeof(int8_t);
static_assert(STATIC_SHARED_BYTES <= 32 * 1024);

__device__ __forceinline__ int8_t unpack_int4(uint8_t value) {
    return static_cast<int8_t>(static_cast<int>(value ^ 0x08u) - 8);
}

__global__ void turing_w4a8_kernel(const int8_t *__restrict__ activation,
                                   const int8_t *__restrict__ weight,
                                   const float *__restrict__ activation_scale,
                                   const float *__restrict__ weight_scale,
                                   const float *__restrict__ bias,
                                   __nv_bfloat16 *__restrict__ output,
                                   int m,
                                   int n,
                                   int k) {
    __shared__ alignas(16) int8_t activation_tile[TILE_M][TILE_K];
    __shared__ alignas(16) int8_t weight_tile[TILE_N][TILE_K];

    const int lane_m = threadIdx.y;
    const int lane_n = threadIdx.x;
    const int row = blockIdx.y * TILE_M + lane_m;
    const int col = blockIdx.x * TILE_N + lane_n;
    const int tid = lane_m * TILE_N + lane_n;
    int accumulator = 0;

    for (int tile_k = 0; tile_k < k; tile_k += TILE_K) {
        for (int index = tid; index < TILE_M * TILE_K; index += TILE_M * TILE_N) {
            const int local_m = index / TILE_K;
            const int local_k = index % TILE_K;
            const int global_m = blockIdx.y * TILE_M + local_m;
            const int global_k = tile_k + local_k;
            activation_tile[local_m][local_k] =
                global_m < m && global_k < k ? activation[global_m * k + global_k] : 0;
        }

        for (int index = tid; index < TILE_N * TILE_K; index += TILE_M * TILE_N) {
            const int local_n = index / TILE_K;
            const int local_k = index % TILE_K;
            const int global_n = blockIdx.x * TILE_N + local_n;
            const int global_k = tile_k + local_k;
            int8_t unpacked = 0;
            if (global_n < n && global_k < k) {
                const uint8_t packed = reinterpret_cast<const uint8_t *>(weight)[global_n * (k / 2) + global_k / 2];
                unpacked = unpack_int4((packed >> ((global_k & 1) * 4)) & 0x0fu);
            }
            weight_tile[local_n][local_k] = unpacked;
        }
        __syncthreads();

        if (row < m && col < n) {
#pragma unroll
            for (int local_k = 0; local_k < TILE_K; local_k += 4) {
                const int packed_activation = *reinterpret_cast<const int *>(&activation_tile[lane_m][local_k]);
                const int packed_weight = *reinterpret_cast<const int *>(&weight_tile[lane_n][local_k]);
                accumulator = __dp4a(packed_activation, packed_weight, accumulator);
            }
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        float value = static_cast<float>(accumulator) * activation_scale[row] * weight_scale[col];
        if (bias != nullptr) {
            value += bias[col];
        }
        output[row * n + col] = __float2bfloat16_rn(value);
    }
}

}  // namespace

void turing_w4a8_linear(Tensor activation,
                        Tensor weight,
                        Tensor activation_scale,
                        Tensor weight_scale,
                        Tensor bias,
                        Tensor output) {
    const int m = activation.size(0);
    const int k = activation.size(1);
    const int n = weight.size(0);
    const dim3 block(TILE_N, TILE_M);
    const dim3 grid(ceilDiv(n, TILE_N), ceilDiv(m, TILE_M));
    const float *bias_ptr = bias.valid() ? static_cast<const float *>(bias.ptr) : nullptr;
    turing_w4a8_kernel<<<grid, block, 0, getCurrentCUDAStream()>>>(
        static_cast<const int8_t *>(activation.ptr),
        static_cast<const int8_t *>(weight.ptr),
        static_cast<const float *>(activation_scale.ptr),
        static_cast<const float *>(weight_scale.ptr),
        bias_ptr,
        static_cast<__nv_bfloat16 *>(output.ptr),
        m,
        n,
        k);
    checkCUDA(cudaGetLastError());
}

}  // namespace svdint4::kernels
