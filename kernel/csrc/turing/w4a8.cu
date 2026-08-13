// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <climits>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/kernel/gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "kernel_api.h"

namespace comfyui_turing_utils::kernels {
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

template <typename ScaleT>
__device__ __forceinline__ float load_group_scale(ScaleT value);

template <>
__device__ __forceinline__ float load_group_scale<uint8_t>(uint8_t value) {
    return __half2float(__nv_cvt_fp8_to_halfraw(value, __NV_E4M3));
}

// CUTLASS normally loads the packed operand fragment and applies a stateless
// numeric transform before storing it to shared memory. Grouped-codebook W4
// also needs a per-group E4M3 scale and a 16-entry codebook. This iterator
// wraps the normal predicated packed-W4 load, but returns an S8 fragment after
// decoding in registers. MmaPipelined therefore writes exactly the same S8
// crosswise shared-memory tile as the existing W8A8 kernel: no decoded global
// workspace and no additional shared memory are introduced.
template <typename Shape_, typename ThreadMap_>
class TuringCodebookW4Iterator {
public:
    using Shape = Shape_;
    using Element = cutlass::int4b_t;
    using Layout = cutlass::layout::ColumnMajor;
    static int const kAdvanceRank = 0;
    using ThreadMap = ThreadMap_;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorRef = cutlass::TensorRef<Element, Layout>;
    using TensorView = cutlass::TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;
    using Pointer = Element *;
    using NonConstPointer = Element *;
    using RawIterator = cutlass::transform::threadblock::PredicatedTileIterator<
        Shape,
        Element,
        Layout,
        kAdvanceRank,
        ThreadMap,
        16>;
    using RawFragment = typename RawIterator::Fragment;
    using AccessType = typename RawIterator::AccessType;
    using Fragment = cutlass::Array<
        int8_t,
        ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
    using Mask = typename RawIterator::Mask;

    static_assert(ThreadMap::kElementsPerAccess == 16,
                  "inline codebook W4 requires one complete 16-value group per access");

    struct Params {
        typename RawIterator::Params raw;
        uint8_t const *group_scale = nullptr;
        float const *codebook = nullptr;
        int groups_per_row = 0;

        Params() = default;

        CUTLASS_HOST_DEVICE
        Params(Layout const &layout) : raw(layout) {}
    };

private:
    RawIterator raw_;
    uint8_t const *group_scale_ = nullptr;
    float const *codebook_ = nullptr;
    int groups_per_row_ = 0;
    int extent_k_ = 0;
    int extent_n_ = 0;
    int thread_k_ = 0;
    int thread_n_ = 0;
    bool enabled_ = true;

public:
    TuringCodebookW4Iterator() = default;

    CUTLASS_HOST_DEVICE
    TuringCodebookW4Iterator(Params const &params,
                             Pointer pointer,
                             TensorCoord extent,
                             int thread_id,
                             TensorCoord const &threadblock_offset,
                             int const *indices = nullptr)
        : raw_(params.raw, pointer, extent, thread_id, threadblock_offset, indices),
          group_scale_(params.group_scale),
          codebook_(params.codebook),
          groups_per_row_(params.groups_per_row),
          extent_k_(extent.row()),
          extent_n_(extent.column()) {
        auto const thread_offset = ThreadMap::initial_offset(thread_id);
        thread_k_ = threadblock_offset.row() + thread_offset.contiguous();
        thread_n_ = threadblock_offset.column() + thread_offset.strided();
    }

    CUTLASS_HOST_DEVICE
    TuringCodebookW4Iterator(Params const &params,
                             Pointer pointer,
                             TensorCoord extent,
                             int thread_id)
        : TuringCodebookW4Iterator(params, pointer, extent, thread_id, TensorCoord()) {}

    CUTLASS_HOST_DEVICE
    TuringCodebookW4Iterator &operator++() {
        ++raw_;
        thread_k_ += Shape::kRow;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    TuringCodebookW4Iterator operator++(int) {
        TuringCodebookW4Iterator self(*this);
        operator++();
        return self;
    }

    CUTLASS_HOST_DEVICE
    void clear_mask(bool enable = true) {
        raw_.clear_mask(enable);
        if (enable) {
            enabled_ = false;
        }
    }

    CUTLASS_HOST_DEVICE
    void enable_mask() {
        raw_.enable_mask();
        enabled_ = true;
    }

    CUTLASS_HOST_DEVICE
    void set_mask(Mask const &mask) {
        raw_.set_mask(mask);
        enabled_ = true;
    }

    CUTLASS_HOST_DEVICE
    void get_mask(Mask &mask) { raw_.get_mask(mask); }

    CUTLASS_DEVICE
    void load(Fragment &fragment) {
        RawFragment packed;
        packed.clear();
        raw_.load(packed);
        auto const *packed_bytes = reinterpret_cast<uint8_t const *>(&packed);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                int const access = c + s * ThreadMap::Iterations::kContiguous;
                int const base = access * ThreadMap::kElementsPerAccess;
                int const k = thread_k_ + c * ThreadMap::Delta::kContiguous;
                int const n = thread_n_ + s * ThreadMap::Delta::kStrided;
                bool const valid = enabled_ && n < extent_n_ && k < extent_k_;
                float const scale = valid
                    ? load_group_scale<uint8_t>(
                          group_scale_[static_cast<int64_t>(n) * groups_per_row_ + k / 16])
                    : 0.0f;

                CUTLASS_PRAGMA_UNROLL
                for (int element = 0; element < 16; ++element) {
                    uint8_t const byte = packed_bytes[(base + element) / 2];
                    int const code = (byte >> ((element & 1) * 4)) & 0x0f;
                    float const value = valid ? __ldg(codebook_ + code) * scale : 0.0f;
                    int const rounded = __float2int_rn(value);
                    fragment[base + element] = static_cast<int8_t>(
                        max(-127, min(127, rounded)));
                }
            }
        }
    }
};

template <typename BaseKernel>
struct TuringCodebookGemmKernel : BaseKernel {
    using Base = BaseKernel;
    using Arguments = typename Base::Arguments;

    struct Params : Base::Params {
        Params() = default;

        Params(Arguments const &args, int device_sms, int sm_occupancy)
            : Base::Params(args, device_sms, sm_occupancy) {
            set_codebook_params(args);
        }

        void update(Arguments const &args) {
            Base::Params::update(args);
            set_codebook_params(args);
        }

    private:
        void set_codebook_params(Arguments const &args) {
            this->params_B.group_scale = reinterpret_cast<uint8_t const *>(args.ptr_C);
            this->params_B.codebook = reinterpret_cast<float const *>(args.ptr_D);
            this->params_B.groups_per_row = args.problem_size.k() / 16;
        }
    };
};

enum class WeightKind {
    kInt8,
    kSignedW4,
    kCodebookW4,
};

struct TileTuneKey {
    int device;
    int m;
    int n;
    int k;

    bool operator<(TileTuneKey const &other) const {
        return std::tie(device, m, n, k) <
            std::tie(other.device, other.m, other.n, other.k);
    }
};

struct TileTuneCache {
    std::mutex mutex;
    std::map<TileTuneKey, int> selected_policy;
};

TileTuneCache w4_tile_cache;
TileTuneCache w8_tile_cache;
TileTuneCache codebook_tile_cache;

template <typename RunPolicy>
bool run_auto_tuned_tile(TileTuneCache &cache,
                         int m,
                         int n,
                         int k,
                         int heuristic_policy,
                         cudaStream_t stream,
                         RunPolicy &&run_policy) {
    int device = 0;
    checkCUDA(cudaGetDevice(&device));
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    checkCUDA(cudaStreamIsCapturing(stream, &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
        return run_policy(heuristic_policy);
    }

    const TileTuneKey key{device, m, n, k};
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (auto found = cache.selected_policy.find(key);
        found != cache.selected_policy.end()) {
        return run_policy(found->second);
    }

    std::array<int, 4> candidates{};
    int candidate_count = 0;
    if (m <= 32) {
        candidates[candidate_count++] = 1;
    } else if (m <= 128) {
        candidates[candidate_count++] = 1;
        candidates[candidate_count++] = 2;
        candidates[candidate_count++] = 3;
        candidates[candidate_count++] = 5;
    } else if (m <= 512) {
        candidates[candidate_count++] = 3;
        candidates[candidate_count++] = 4;
        candidates[candidate_count++] = 5;
    } else {
        candidates[candidate_count++] = 4;
        candidates[candidate_count++] = 5;
    }

    std::array<float, 6> best_ms;
    best_ms.fill(std::numeric_limits<float>::infinity());
    for (int round = 0; round < 2; ++round) {
        for (int index = 0; index < candidate_count; ++index) {
            const int candidate_index = round == 0
                ? index
                : candidate_count - index - 1;
            const int policy = candidates[candidate_index];
            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;
            checkCUDA(cudaEventCreate(&start));
            checkCUDA(cudaEventCreate(&stop));
            checkCUDA(cudaEventRecord(start, stream));
            const bool launched = run_policy(policy);
            checkCUDA(cudaEventRecord(stop, stream));
            checkCUDA(cudaEventSynchronize(stop));
            float elapsed_ms = std::numeric_limits<float>::infinity();
            if (launched) {
                checkCUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
                best_ms[policy] = std::min(best_ms[policy], elapsed_ms);
            }
            checkCUDA(cudaEventDestroy(stop));
            checkCUDA(cudaEventDestroy(start));
        }
    }

    int selected = heuristic_policy;
    for (int index = 0; index < candidate_count; ++index) {
        const int policy = candidates[index];
        if (best_ms[policy] < best_ms[selected] * 0.98f) {
            selected = policy;
        }
    }
    cache.selected_policy.emplace(key, selected);
    return run_policy(selected);
}

template <typename Output, WeightKind Kind, int TBM, int TBN, int WM, int WN>
struct TuringW4A8Gemm {
    static constexpr bool PackedWeight = Kind != WeightKind::kInt8;
    static constexpr bool CodebookWeight = Kind == WeightKind::kCodebookW4;
    using ElementA = int8_t;
    using ElementB = std::conditional_t<PackedWeight, cutlass::int4b_t, int8_t>;
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
    using PredicatedIteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        LayoutB,
        0,
        typename MmaCore::IteratorThreadMapB,
        AlignmentB>;
    using IteratorB = std::conditional_t<
        CodebookWeight,
        TuringCodebookW4Iterator<
            cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
            typename MmaCore::IteratorThreadMapB>,
        PredicatedIteratorB>;

    using TransformA = cutlass::NumericArrayConverter<
        typename MmaCore::SmemIteratorA::Element,
        typename IteratorA::Element,
        IteratorA::Fragment::kElements>;
    using TransformB = std::conditional_t<
        CodebookWeight,
        cutlass::NumericArrayConverter<
            typename MmaCore::SmemIteratorB::Element,
            int8_t,
            IteratorB::Fragment::kElements>,
        std::conditional_t<
            PackedWeight,
            TuringW4ToS8<IteratorB::Fragment::kElements>,
            cutlass::NumericArrayConverter<
                typename MmaCore::SmemIteratorB::Element,
                typename IteratorB::Element,
                IteratorB::Fragment::kElements>>>;

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
    using BaseGemmKernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
        Mma,
        Epilogue,
        ThreadblockSwizzle>;
    using GemmKernel = std::conditional_t<
        CodebookWeight,
        TuringCodebookGemmKernel<BaseGemmKernel>,
        BaseGemmKernel>;
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
                    int output_stride,
                    cudaStream_t stream,
                    const uint8_t *group_scale = nullptr,
                    const float *codebook = nullptr) {
        cutlass::gemm::GemmCoord problem(m, n, k);
        typename Callbacks::Arguments callbacks{
            {{{{},
               {const_cast<float *>(activation_scale), 0.0f, {_1{}, _0{}, m}},
               {}},
              {const_cast<float *>(weight_scale), 0.0f, {_0{}, _1{}, n}},
              {}},
             {const_cast<float *>(bias), 0.0f, {_0{}, _1{}, n}},
             {}},
            {output, {output_stride, _1{}, static_cast<int64_t>(m) * output_stride}}};
        typename Gemm::Arguments arguments(
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem,
            1,
            callbacks,
            const_cast<int8_t *>(activation),
            reinterpret_cast<ElementB *>(const_cast<int8_t *>(weight)),
            CodebookWeight ? reinterpret_cast<ElementC const *>(group_scale) : nullptr,
            CodebookWeight
                ? reinterpret_cast<ElementC *>(const_cast<float *>(codebook))
                : nullptr,
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
    return TuringW4A8Gemm<
        cutlass::bfloat16_t, WeightKind::kSignedW4, TBM, TBN, WM, WN>::run(
        activation,
        weight,
        activation_scale,
        weight_scale,
        bias,
        reinterpret_cast<cutlass::bfloat16_t *>(output),
        m,
        n,
        k,
        n,
        stream);
}

template <int TBM, int TBN, int WM, int WN>
bool run_int8_tile(const int8_t *activation,
                   const int8_t *weight,
                   const float *activation_scale,
                   const float *weight_scale,
                   const float *bias,
                   __nv_bfloat16 *output,
                   int m,
                   int n,
                   int k,
                   int output_stride,
                   cudaStream_t stream) {
    return TuringW4A8Gemm<
        cutlass::bfloat16_t, WeightKind::kInt8, TBM, TBN, WM, WN>::run(
        activation,
        weight,
        activation_scale,
        weight_scale,
        bias,
        reinterpret_cast<cutlass::bfloat16_t *>(output),
        m,
        n,
        k,
        output_stride,
        stream);
}

template <int TBM, int TBN, int WM, int WN>
bool run_codebook_tile(const int8_t *activation,
                       const int8_t *weight,
                       const float *activation_scale,
                       const uint8_t *group_scale,
                       const float *channel_scale,
                       const float *codebook,
                       const float *bias,
                       __nv_bfloat16 *output,
                       int m,
                       int n,
                       int k,
                       cudaStream_t stream) {
    return TuringW4A8Gemm<
        cutlass::bfloat16_t, WeightKind::kCodebookW4, TBM, TBN, WM, WN>::run(
        activation,
        weight,
        activation_scale,
        channel_scale,
        bias,
        reinterpret_cast<cutlass::bfloat16_t *>(output),
        m,
        n,
        k,
        n,
        stream,
        group_scale,
        codebook);
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
              cudaStream_t stream,
              int tile_policy) {
    const auto run_policy = [&](int policy) {
        switch (policy) {
        case 1:
            return run_tile<16, 64, 16, 32>(
                activation, weight, activation_scale, weight_scale, bias, output, m, n, k, stream);
        case 2:
            return run_tile<32, 64, 32, 32>(
                activation, weight, activation_scale, weight_scale, bias, output, m, n, k, stream);
        case 3:
            return run_tile<64, 128, 32, 64>(
                activation, weight, activation_scale, weight_scale, bias, output, m, n, k, stream);
        case 4:
            return run_tile<256, 128, 64, 64>(
                activation, weight, activation_scale, weight_scale, bias, output, m, n, k, stream);
        case 5:
            return run_tile<128, 256, 64, 64>(
                activation, weight, activation_scale, weight_scale, bias, output, m, n, k, stream);
        default:
            throw std::runtime_error("invalid Turing W4A8 tile policy");
        }
    };
    if (tile_policy != 0) {
        return run_policy(tile_policy);
    }
    int heuristic_policy;
    if (m <= 32) {
        heuristic_policy = 1;
    } else if (m <= 128 && n < 8192) {
        heuristic_policy = 2;
    } else if (m <= 512) {
        heuristic_policy = 3;
    } else if (m <= 8192) {
        heuristic_policy = 4;
    } else {
        heuristic_policy = 5;
    }
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    if (properties->major != 7 || properties->minor != 5) {
        return run_policy(heuristic_policy);
    }
    return run_auto_tuned_tile(
        w4_tile_cache, m, n, k, heuristic_policy, stream, run_policy);
}

bool dispatch_int8(const int8_t *activation,
                   const int8_t *weight,
                   const float *activation_scale,
                   const float *weight_scale,
                   const float *bias,
                   __nv_bfloat16 *output,
                   int m,
                   int n,
                   int k,
                   int output_stride,
                   cudaStream_t stream,
                   int tile_policy = 0) {
    const auto run_policy = [&](int policy) {
        switch (policy) {
        case 1:
            return run_int8_tile<16, 64, 16, 32>(
                activation, weight, activation_scale, weight_scale, bias,
                output, m, n, k, output_stride, stream);
        case 2:
            return run_int8_tile<32, 64, 32, 32>(
                activation, weight, activation_scale, weight_scale, bias,
                output, m, n, k, output_stride, stream);
        case 3:
            return run_int8_tile<64, 128, 32, 64>(
                activation, weight, activation_scale, weight_scale, bias,
                output, m, n, k, output_stride, stream);
        case 4:
            return run_int8_tile<256, 128, 64, 64>(
                activation, weight, activation_scale, weight_scale, bias,
                output, m, n, k, output_stride, stream);
        case 5:
            return run_int8_tile<128, 256, 64, 64>(
                activation, weight, activation_scale, weight_scale, bias,
                output, m, n, k, output_stride, stream);
        default:
            throw std::runtime_error("invalid Turing INT8 tile policy");
        }
    };
    if (tile_policy != 0) {
        return run_policy(tile_policy);
    }
    int heuristic_policy;
    if (m <= 32) {
        heuristic_policy = 1;
    } else if (m <= 128 && n < 8192) {
        heuristic_policy = 2;
    } else if (m <= 512) {
        heuristic_policy = 3;
    } else if (m <= 8192) {
        heuristic_policy = 4;
    } else {
        heuristic_policy = 5;
    }
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    if (properties->major != 7 || properties->minor != 5) {
        return run_policy(heuristic_policy);
    }
    return run_auto_tuned_tile(
        w8_tile_cache, m, n, k, heuristic_policy, stream, run_policy);
}

bool dispatch_codebook(const int8_t *activation,
                       const int8_t *weight,
                       const float *activation_scale,
                       const uint8_t *group_scale,
                       const float *channel_scale,
                       const float *codebook,
                       const float *bias,
                       __nv_bfloat16 *output,
                       int m,
                       int n,
                       int k,
                       cudaStream_t stream,
                       int tile_policy) {
    // The long-sequence W8A8 policy is already register-limited to one CTA per
    // SM75 SM. Inline decoding cannot reduce its CTA residency, while using it
    // for smaller policies could cross a two-CTA register threshold. Keep the
    // production inline path deliberately scoped to the long tile.
    if (m <= 8192) {
        return false;
    }
    const auto run_policy = [&](int policy) {
        if (policy == 4) {
            return run_codebook_tile<256, 128, 64, 64>(
                activation, weight, activation_scale, group_scale,
                channel_scale, codebook, bias, output, m, n, k, stream);
        }
        if (policy == 5) {
            return run_codebook_tile<128, 256, 64, 64>(
                activation, weight, activation_scale, group_scale,
                channel_scale, codebook, bias, output, m, n, k, stream);
        }
        throw std::runtime_error("invalid Turing codebook W4A8 tile policy");
    };
    if (tile_policy != 0) {
        return run_policy(tile_policy);
    }
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    if (properties->major != 7 || properties->minor != 5) {
        return run_policy(5);
    }
    return run_auto_tuned_tile(
        codebook_tile_cache, m, n, k, 5, stream, run_policy);
}

// Decode one 16-column vector per thread. This intentionally matches Kitchen's
// reference rounding and E4M3 conversion so the staged SM75 path is bit exact
// before the INT8 contraction.
__global__ void decode_codebook_w4_to_s8(
    const int8_t *__restrict__ packed_weight,
    const uint8_t *__restrict__ group_scale,
    const float *__restrict__ codebook,
    int8_t *__restrict__ output,
    int64_t vector_count,
    int packed_k,
    int k,
    int group_size) {
    __shared__ float shared_codebook[16];
    if (threadIdx.x < 16) {
        shared_codebook[threadIdx.x] = codebook[threadIdx.x];
    }
    __syncthreads();

    const int64_t vector = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vector >= vector_count) {
        return;
    }
    const int vectors_per_row = packed_k / 8;
    const int row = static_cast<int>(vector / vectors_per_row);
    const int packed_column = static_cast<int>(vector % vectors_per_row) * 8;
    const int output_column = packed_column * 2;
    const int groups_per_row = k / group_size;
    const int64_t scale_row = static_cast<int64_t>(row) * groups_per_row;
    const uint2 packed = *reinterpret_cast<const uint2 *>(
        packed_weight + static_cast<int64_t>(row) * packed_k + packed_column);
    const unsigned words[2] = {packed.x, packed.y};

    const int base_group = output_column / group_size;
    float scales[4];
    scales[0] = load_group_scale<uint8_t>(group_scale[scale_row + base_group]);
    scales[1] = scales[0];
    scales[2] = scales[0];
    scales[3] = scales[0];
    if (group_size < 16) {
        scales[1] = load_group_scale<uint8_t>(group_scale[scale_row + base_group + 1]);
        if (group_size < 8) {
            scales[2] = load_group_scale<uint8_t>(group_scale[scale_row + base_group + 2]);
            scales[3] = load_group_scale<uint8_t>(group_scale[scale_row + base_group + 3]);
        }
    }

    char4 decoded[4];
#pragma unroll
    for (int word = 0; word < 2; ++word) {
        const unsigned bytes = words[word];
#pragma unroll
        for (int byte_index = 0; byte_index < 4; ++byte_index) {
            const int pair = word * 4 + byte_index;
            const int local_group = group_size >= 16 ? 0 : (pair * 2) / group_size;
            const float scale = scales[local_group];
            const unsigned value = (bytes >> (byte_index * 8)) & 0xffu;
            const unsigned low = value & 0x0fu;
            const unsigned high = value >> 4;
            auto *target = reinterpret_cast<int8_t *>(&decoded[pair / 2]) + (pair % 2) * 2;
            target[0] = static_cast<int8_t>(max(
                -127, min(127, __float2int_rn(shared_codebook[low] * scale))));
            target[1] = static_cast<int8_t>(max(
                -127, min(127, __float2int_rn(shared_codebook[high] * scale))));
        }
    }
    *reinterpret_cast<uint4 *>(output + static_cast<int64_t>(row) * k + output_column) =
        *reinterpret_cast<uint4 *>(decoded);
}

void launch_codebook_decode(const int8_t *packed_weight,
                            const uint8_t *group_scale,
                            const float *codebook,
                            int8_t *workspace,
                            int rows,
                            int k,
                            int group_size,
                            cudaStream_t stream) {
    constexpr int threads = 256;
    const int packed_k = k / 2;
    const int64_t vector_count = static_cast<int64_t>(rows) * packed_k / 8;
    const int blocks = static_cast<int>(ceilDiv(vector_count, static_cast<int64_t>(threads)));
    decode_codebook_w4_to_s8<<<blocks, threads, 0, stream>>>(
        packed_weight,
        group_scale,
        codebook,
        workspace,
        vector_count,
        packed_k,
        k,
        group_size);
}

}  // namespace

void turing_w4a8_linear(Tensor activation,
                        Tensor weight,
                        Tensor activation_scale,
                        Tensor weight_scale,
                        Tensor bias,
                        Tensor output,
                        int tile_policy) {
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
        stream,
        tile_policy);
    if (!launched) {
        throw std::runtime_error("CUTLASS SM75 W4A8 kernel rejected the problem shape");
    }
    checkCUDA(cudaGetLastError());
}

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
                                 int tile_policy) {
    const int m = activation.size(0);
    const int k = activation.size(1);
    const int n = weight.size(0);
    const int chunk_rows = workspace.valid() ? workspace.size(0) : 0;
    if (m == 0 || n == 0 || k == 0) {
        return;
    }
    if (k % 16 != 0 || n % 8 != 0 ||
        (!inline_decode && (chunk_rows <= 0 || chunk_rows % 8 != 0))) {
        throw std::runtime_error(
            "Turing codebook W4A8 requires K%16=0, N%8=0, and an 8-row-aligned staged workspace");
    }
    if (group_size < 4 || k % group_size != 0 ||
        (16 % group_size != 0 && group_size % 16 != 0)) {
        throw std::runtime_error("unsupported Turing codebook W4A8 group size");
    }

    const auto *activation_ptr = activation.data_ptr<int8_t>();
    const auto *weight_ptr = weight.data_ptr<int8_t>();
    const auto *activation_scale_ptr = activation_scale.data_ptr<float>();
    const auto *group_scale_ptr = group_scale.data_ptr<uint8_t>();
    const auto *channel_scale_ptr = channel_scale.data_ptr<float>();
    const auto *codebook_ptr = codebook.data_ptr<float>();
    const auto *bias_ptr = bias.valid() ? bias.data_ptr<float>() : nullptr;
    auto *output_ptr = output.data_ptr<__nv_bfloat16>();
    const int packed_k = k / 2;
    const int groups_per_row = k / group_size;
    const cudaStream_t stream = getCurrentCUDAStream();

    if (inline_decode) {
        if (group_size != 16) {
            throw std::runtime_error("inline Turing codebook W4A8 requires group_size=16");
        }
        if (!dispatch_codebook(
                activation_ptr,
                weight_ptr,
                activation_scale_ptr,
                group_scale_ptr,
                channel_scale_ptr,
                codebook_ptr,
                bias_ptr,
                output_ptr,
                m,
                n,
                k,
                stream,
                tile_policy)) {
            throw std::runtime_error(
                "inline CUTLASS SM75 codebook W4A8 rejected the problem shape");
        }
        checkCUDA(cudaGetLastError());
        return;
    }

    auto *workspace_ptr = workspace.data_ptr<int8_t>();

    for (int row = 0; row < n; row += chunk_rows) {
        const int rows = std::min(chunk_rows, n - row);
        launch_codebook_decode(
            weight_ptr + static_cast<int64_t>(row) * packed_k,
            group_scale_ptr + static_cast<int64_t>(row) * groups_per_row,
            codebook_ptr,
            workspace_ptr,
            rows,
            k,
            group_size,
            stream);
        if (!dispatch_int8(
                activation_ptr,
                workspace_ptr,
                activation_scale_ptr,
                channel_scale_ptr + row,
                bias_ptr == nullptr ? nullptr : bias_ptr + row,
                output_ptr + row,
                m,
                rows,
                k,
                n,
                stream)) {
            throw std::runtime_error("CUTLASS SM75 codebook W4A8 kernel rejected a chunk");
        }
    }
    checkCUDA(cudaGetLastError());
}

void turing_int8_linear(Tensor activation,
                        Tensor weight,
                        Tensor activation_scale,
                        Tensor weight_scale,
                        Tensor bias,
                        Tensor output,
                        int tile_policy) {
    const int m = activation.size(0);
    const int k = activation.size(1);
    const int n = weight.size(0);
    if (m == 0 || n == 0 || k == 0) {
        return;
    }
    if (k % 16 != 0 || n % 8 != 0) {
        throw std::runtime_error("Turing INT8 linear requires K%16=0 and N%8=0");
    }
    const bool launched = dispatch_int8(
        activation.data_ptr<int8_t>(),
        weight.data_ptr<int8_t>(),
        activation_scale.data_ptr<float>(),
        weight_scale.data_ptr<float>(),
        bias.valid() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<__nv_bfloat16>(),
        m,
        n,
        k,
        n,
        getCurrentCUDAStream(),
        tile_policy);

    if (!launched) {
        throw std::runtime_error("CUTLASS SM75 INT8 kernel rejected the problem shape");
    }
    checkCUDA(cudaGetLastError());
}

}  // namespace comfyui_turing_utils::kernels
