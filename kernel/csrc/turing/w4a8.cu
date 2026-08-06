// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <stdexcept>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/kernel/gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "kernel_api.h"

namespace svdint4::kernels {
namespace {

using cute::_0;
using cute::_1;

constexpr int FALLBACK_TILE_M = 16;
constexpr int FALLBACK_TILE_N = 16;
constexpr int FALLBACK_TILE_K = 64;

__device__ __forceinline__ int8_t unpack_int4(uint8_t value) {
    return static_cast<int8_t>(static_cast<int>(value ^ 0x08u) - 8);
}

__global__ void w4a8_compatibility_kernel(const int8_t *__restrict__ activation,
                                          const int8_t *__restrict__ weight,
                                          const float *__restrict__ activation_scale,
                                          const float *__restrict__ weight_scale,
                                          const float *__restrict__ bias,
                                          __nv_bfloat16 *__restrict__ output,
                                          int m,
                                          int n,
                                          int k) {
    __shared__ alignas(16) int8_t activation_tile[FALLBACK_TILE_M][FALLBACK_TILE_K];
    __shared__ alignas(16) int8_t weight_tile[FALLBACK_TILE_N][FALLBACK_TILE_K];

    const int lane_m = threadIdx.y;
    const int lane_n = threadIdx.x;
    const int row = blockIdx.y * FALLBACK_TILE_M + lane_m;
    const int col = blockIdx.x * FALLBACK_TILE_N + lane_n;
    const int tid = lane_m * FALLBACK_TILE_N + lane_n;
    int accumulator = 0;

    for (int tile_k = 0; tile_k < k; tile_k += FALLBACK_TILE_K) {
        for (int index = tid;
             index < FALLBACK_TILE_M * FALLBACK_TILE_K;
             index += FALLBACK_TILE_M * FALLBACK_TILE_N) {
            const int local_m = index / FALLBACK_TILE_K;
            const int local_k = index % FALLBACK_TILE_K;
            const int global_m = blockIdx.y * FALLBACK_TILE_M + local_m;
            const int global_k = tile_k + local_k;
            activation_tile[local_m][local_k] =
                global_m < m && global_k < k ? activation[global_m * k + global_k] : 0;
        }

        for (int index = tid;
             index < FALLBACK_TILE_N * FALLBACK_TILE_K;
             index += FALLBACK_TILE_M * FALLBACK_TILE_N) {
            const int local_n = index / FALLBACK_TILE_K;
            const int local_k = index % FALLBACK_TILE_K;
            const int global_n = blockIdx.x * FALLBACK_TILE_N + local_n;
            const int global_k = tile_k + local_k;
            int8_t unpacked = 0;
            if (global_n < n && global_k < k) {
                const uint8_t packed = reinterpret_cast<const uint8_t *>(weight)[
                    global_n * (k / 2) + global_k / 2];
                unpacked = unpack_int4((packed >> ((global_k & 1) * 4)) & 0x0fu);
            }
            weight_tile[local_n][local_k] = unpacked;
        }
        __syncthreads();

        if (row < m && col < n) {
#pragma unroll
            for (int local_k = 0; local_k < FALLBACK_TILE_K; local_k += 4) {
                const int packed_activation =
                    *reinterpret_cast<const int *>(&activation_tile[lane_m][local_k]);
                const int packed_weight =
                    *reinterpret_cast<const int *>(&weight_tile[lane_n][local_k]);
                accumulator = __dp4a(packed_activation, packed_weight, accumulator);
            }
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        float value = static_cast<float>(accumulator) *
                      activation_scale[row] * weight_scale[col];
        if (bias != nullptr) {
            value += bias[col];
        }
        output[row * n + col] = __float2bfloat16_rn(value);
    }
}

void launch_compatibility_kernel(const int8_t *activation,
                                 const int8_t *weight,
                                 const float *activation_scale,
                                 const float *weight_scale,
                                 const float *bias,
                                 __nv_bfloat16 *output,
                                 int m,
                                 int n,
                                 int k,
                                 cudaStream_t stream) {
    const dim3 block(FALLBACK_TILE_N, FALLBACK_TILE_M);
    const dim3 grid(ceilDiv(n, FALLBACK_TILE_N), ceilDiv(m, FALLBACK_TILE_M));
    w4a8_compatibility_kernel<<<grid, block, 0, stream>>>(
        activation,
        weight,
        activation_scale,
        weight_scale,
        bias,
        output,
        m,
        n,
        k);
}

template <int N>
struct TuringW4ToS8 {
    static_assert(N % 8 == 0, "Turing W4 conversion requires groups of eight values");
    using result_type = cutlass::Array<int8_t, N>;
    using source_type = cutlass::Array<cutlass::int4b_t, N>;

    CUTLASS_HOST_DEVICE
    result_type operator()(source_type const &source) const {
#if defined(__CUDA_ARCH__)
        result_type result;
        auto const *packed = reinterpret_cast<uint32_t const *>(&source);
        auto *unpacked = reinterpret_cast<uint32_t *>(&result);

        CUTLASS_PRAGMA_UNROLL
        for (int index = 0; index < N / 8; ++index) {
            uint32_t const value = packed[index];
            uint32_t even = value & 0x0f0f0f0fu;
            uint32_t odd = (value >> 4) & 0x0f0f0f0fu;

            // Each lane is at most 0xf, so multiplying its sign bit by 30
            // cannot carry into the adjacent byte. This sign-extends four
            // nibbles in parallel with one IMAD instead of CUTLASS's PRMT,
            // mask, shift, and merge sequence.
            even |= (even & 0x08080808u) * 30u;
            odd |= (odd & 0x08080808u) * 30u;

            asm volatile(
                "prmt.b32 %0, %2, %3, 0x5140;\n"
                "prmt.b32 %1, %2, %3, 0x7362;\n"
                : "=&r"(unpacked[index * 2]),
                  "=&r"(unpacked[index * 2 + 1])
                : "r"(even), "r"(odd));
        }
        return result;
#else
        return cutlass::NumericArrayConverter<int8_t, cutlass::int4b_t, N>{}(source);
#endif
    }
};

template <typename Output, int TBM, int TBN, int WM, int WN>
struct TuringW4A8Gemm {
    using ElementA = int8_t;
    using ElementB = cutlass::int4b_t;
    using SharedElementB = int8_t;
    using ElementC = Output;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using ThreadblockShape = cutlass::gemm::GemmShape<TBM, TBN, 64>;
    using WarpShape = cutlass::gemm::GemmShape<WM, WN, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    static constexpr int AlignmentA = 16;
    static constexpr int AlignmentB = 16;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int EpilogueStages = 1;

    using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        ElementA,
        LayoutA,
        SharedElementB,
        LayoutB,
        ElementAccumulator,
        LayoutC,
        cutlass::arch::OpClassTensorOp,
        2,
        cutlass::arch::OpMultiplyAddSaturate>;

    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        LayoutA,
        1,
        typename MmaCore::IteratorThreadMapA,
        AlignmentA>;

    // Reuse the W8A8 tile map but read 16 packed W4 values with one 64-bit
    // access. The Turing transform expands each vector before writing the
    // normal SM75 crosswise S8 shared-memory tile.
    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        LayoutB,
        0,
        typename MmaCore::IteratorThreadMapB,
        AlignmentB>;

    using TransformA = cutlass::NumericArrayConverter<
        typename MmaCore::SmemIteratorA::Element,
        typename IteratorA::Element,
        IteratorA::Fragment::kElements>;
    using TransformB = TuringW4ToS8<IteratorB::Fragment::kElements>;

    using Mma = cutlass::gemm::threadblock::MmaPipelined<
        typename MmaCore::Shape,
        IteratorA,
        typename MmaCore::SmemIteratorA,
        IteratorB,
        typename MmaCore::SmemIteratorB,
        ElementAccumulator,
        LayoutC,
        typename MmaCore::MmaPolicy,
        TransformA,
        TransformB>;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        ThreadblockShape,
        WarpShape,
        ElementC,
        AlignmentC,
        EpilogueStages>;
    using Accumulator = cutlass::epilogue::threadblock::VisitorAccFetch;
    using ActivationScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
        ThreadMap,
        ElementCompute,
        cute::Stride<_1, _0, int32_t>>;
    using WeightScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap,
        ElementCompute,
        cute::Stride<_0, _1, int32_t>>;
    using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap,
        ElementCompute,
        cute::Stride<_0, _1, int32_t>>;
    using ScaleActivation = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies,
        ElementCompute,
        ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using ScaledActivation = cutlass::epilogue::threadblock::Sm80EVT<
        ScaleActivation,
        Accumulator,
        ActivationScale>;
    using ScaleWeight = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies,
        ElementCompute,
        ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using ScaledOutput = cutlass::epilogue::threadblock::Sm80EVT<
        ScaleWeight,
        ScaledActivation,
        WeightScale>;
    using AddBias = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::plus,
        ElementC,
        ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using BiasedOutput = cutlass::epilogue::threadblock::Sm80EVT<
        AddBias,
        ScaledOutput,
        Bias>;
    using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
        ThreadMap,
        ElementC,
        cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, _1, int64_t>>;
    using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<Store, BiasedOutput>;

    using EpilogueBase = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA,
        LayoutA,
        cutlass::ComplexTransform::kNone,
        AlignmentA,
        SharedElementB,
        LayoutB,
        cutlass::ComplexTransform::kNone,
        16,
        ElementC,
        LayoutC,
        AlignmentC,
        ElementAccumulator,
        ElementCompute,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm75,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        Callbacks,
        ThreadblockSwizzle,
        2,
        cutlass::arch::OpMultiplyAddSaturate,
        EpilogueStages>;
    using Epilogue = typename EpilogueBase::Epilogue;
    using GemmKernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
        Mma,
        Epilogue,
        ThreadblockSwizzle>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    static_assert(sizeof(typename GemmKernel::SharedStorage) <= 48 * 1024,
                  "Turing W4A8 must stay within the default 48 KiB shared-memory limit");

    static bool run(const int8_t *activation,
                    const int8_t *weight,
                    const float *activation_scale,
                    const float *weight_scale,
                    const float *bias,
                    Output *output,
                    int m,
                    int n,
                    int k,
                    cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(m, n, k);
        typename Callbacks::Arguments callbacks{
            {{{{},
               {const_cast<float *>(activation_scale), 0.0f, {_1{}, _0{}, m}},
               {}},
              {const_cast<float *>(weight_scale), 0.0f, {_0{}, _1{}, n}},
              {}},
             {const_cast<float *>(bias), 0.0f, {_0{}, _1{}, n}},
             {}},
            {output, {n, _1{}, static_cast<int64_t>(m) * n}}};
        typename Gemm::Arguments arguments(
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem,
            1,
            callbacks,
            const_cast<int8_t *>(activation),
            reinterpret_cast<ElementB *>(const_cast<int8_t *>(weight)),
            nullptr,
            nullptr,
            static_cast<int64_t>(m) * k,
            static_cast<int64_t>(n) * k,
            0,
            0,
            k,
            k,
            0,
            0);
        Gemm gemm;
        if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
            return false;
        }
        if (Gemm::get_workspace_size(arguments) != 0) {
            return false;
        }
        if (gemm.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess) {
            return false;
        }
        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

template <int TBM, int TBN, int WM, int WN>
bool run_tile(const int8_t *activation,
              const int8_t *weight,
              const float *activation_scale,
              const float *weight_scale,
              const float *bias,
              __nv_bfloat16 *output,
              int m,
              int n,
              int k,
              cudaStream_t stream) {
    return TuringW4A8Gemm<cutlass::bfloat16_t, TBM, TBN, WM, WN>::run(
        activation,
        weight,
        activation_scale,
        weight_scale,
        bias,
        reinterpret_cast<cutlass::bfloat16_t *>(output),
        m,
        n,
        k,
        stream);
}

bool dispatch(const int8_t *activation,
              const int8_t *weight,
              const float *activation_scale,
              const float *weight_scale,
              const float *bias,
              __nv_bfloat16 *output,
              int m,
              int n,
              int k,
              cudaStream_t stream) {
    if (m <= 32) {
        return run_tile<16, 64, 16, 32>(
            activation, weight, activation_scale, weight_scale, bias, output, m, n, k, stream);
    }
    if (m <= 128 && n < 8192) {
        return run_tile<32, 64, 32, 32>(
            activation, weight, activation_scale, weight_scale, bias, output, m, n, k, stream);
    }
    if (m <= 512) {
        return run_tile<64, 128, 32, 64>(
            activation, weight, activation_scale, weight_scale, bias, output, m, n, k, stream);
    }
    if (m <= 8192) {
        return run_tile<256, 128, 64, 64>(
            activation, weight, activation_scale, weight_scale, bias, output, m, n, k, stream);
    }
    return run_tile<128, 256, 64, 64>(
        activation, weight, activation_scale, weight_scale, bias, output, m, n, k, stream);
}

}  // namespace

void turing_w4a8_linear(Tensor activation,
                        Tensor weight,
                        Tensor activation_scale,
                        Tensor weight_scale,
                        Tensor bias,
                        Tensor output) {
    const int64_t m64 = activation.size(0);
    const int64_t k64 = activation.size(1);
    const int64_t n64 = weight.size(0);
    if (m64 == 0 || n64 == 0 || k64 == 0) {
        return;
    }
    if (m64 > INT_MAX || n64 > INT_MAX || k64 > INT_MAX) {
        throw std::runtime_error("Turing W4A8 dimensions are unsupported");
    }

    const auto *activation_ptr = static_cast<const int8_t *>(activation.ptr);
    const auto *weight_ptr = static_cast<const int8_t *>(weight.ptr);
    const auto *activation_scale_ptr = static_cast<const float *>(activation_scale.ptr);
    const auto *weight_scale_ptr = static_cast<const float *>(weight_scale.ptr);
    const auto *bias_ptr = bias.valid() ? static_cast<const float *>(bias.ptr) : nullptr;
    auto *output_ptr = static_cast<__nv_bfloat16 *>(output.ptr);
    const int m = static_cast<int>(m64);
    const int n = static_cast<int>(n64);
    const int k = static_cast<int>(k64);
    const cudaStream_t stream = getCurrentCUDAStream();

    // The Tensor Core epilogue writes 128-bit vectors and the packed iterator
    // consumes complete m8n8k16 instructions. Preserve the old API for edge
    // shapes without routing normal model dimensions through a padding copy.
    if (k % 16 != 0 || n % 8 != 0) {
        launch_compatibility_kernel(
            activation_ptr,
            weight_ptr,
            activation_scale_ptr,
            weight_scale_ptr,
            bias_ptr,
            output_ptr,
            m,
            n,
            k,
            stream);
        checkCUDA(cudaGetLastError());
        return;
    }

    const bool launched = dispatch(
        activation_ptr,
        weight_ptr,
        activation_scale_ptr,
        weight_scale_ptr,
        bias_ptr,
        output_ptr,
        m,
        n,
        k,
        stream);
    if (!launched) {
        throw std::runtime_error("CUTLASS SM75 W4A8 kernel rejected the problem shape");
    }
    checkCUDA(cudaGetLastError());
}

}  // namespace svdint4::kernels
