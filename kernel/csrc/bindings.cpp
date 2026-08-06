#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/utils/pybind.h>

#include <cmath>
#include <optional>
#include <tuple>

#include "kernel_api.h"

namespace {

class TorchOpContext {
public:
    TorchOpContext() {
        stackCUDAStreams.push_back(at::cuda::getCurrentCUDAStream().stream());
    }

    TorchOpContext(const TorchOpContext &) = delete;
    TorchOpContext(TorchOpContext &&) = delete;

    ~TorchOpContext() {
        assert(!stackCUDAStreams.empty());
        assert(stackCUDAStreams.back() == at::cuda::getCurrentCUDAStream().stream());
        stackCUDAStreams.pop_back();
    }
};

template <typename To, typename From>
To int_cast(From value) {
    TORCH_CHECK(value >= static_cast<From>(std::numeric_limits<To>::min()) &&
                    value <= static_cast<From>(std::numeric_limits<To>::max()),
                "integer overflow while converting tensor metadata");
    return static_cast<To>(value);
}

Tensor from_torch(at::Tensor input) {
    Tensor result;

    const int ndims = int_cast<int>(input.dim());
    for (int i = 0; i < ndims; ++i) {
        result.shape.dataExtent.push_back(int_cast<int>(input.size(i)));
        result.shape.dataStride.push_back(int_cast<int>(input.stride(i)));
    }

    switch (input.scalar_type()) {
    case at::ScalarType::Char:
    case at::ScalarType::Byte:
        result.scalarType = Tensor::INT8;
        break;
    case at::ScalarType::Short:
        result.scalarType = Tensor::INT16;
        break;
    case at::ScalarType::Int:
        result.scalarType = Tensor::INT32;
        break;
    case at::ScalarType::Long:
        result.scalarType = Tensor::INT64;
        break;
    case at::ScalarType::Half:
        result.scalarType = Tensor::FP16;
        break;
    case at::ScalarType::Float:
        result.scalarType = Tensor::FP32;
        break;
    case at::ScalarType::BFloat16:
        result.scalarType = Tensor::BF16;
        break;
    case at::ScalarType::Float8_e4m3fn:
        result.scalarType = Tensor::FP8_E4M3;
        break;
    case at::ScalarType::Float8_e5m2:
        result.scalarType = Tensor::FP8_E5M2;
        break;
    default:
        TORCH_CHECK(false, "unsupported tensor dtype for Turing Utils kernel");
    }

    result.ptr = input.data_ptr();
    result.dev = Device{input.is_cuda() ? Device::CUDA : Device::CPU, input.is_cuda() ? input.get_device() : 0};
    result.owner = std::make_shared<at::Tensor>(std::move(input));
    return result;
}

void check_cuda_2d(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_half_like(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16,
                name,
                " must be float16 or bfloat16");
}

void check_float_like(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.scalar_type() == at::kHalf ||
                    t.scalar_type() == at::kBFloat16 ||
                    t.scalar_type() == at::kFloat,
                name,
                " must be float16, bfloat16, or float32");
}

Tensor maybe_tensor(const std::optional<at::Tensor> &value) {
    if (!value.has_value()) {
        return Tensor{};
    }
    return from_torch(value.value());
}

at::Tensor turing_w4a8_linear(at::Tensor activation,
                              at::Tensor weight,
                              at::Tensor activation_scale,
                              at::Tensor weight_scale,
                              std::optional<at::Tensor> bias) {
    activation = activation.contiguous();
    weight = weight.contiguous();
    activation_scale = activation_scale.to(at::kFloat).contiguous();
    weight_scale = weight_scale.to(at::kFloat).contiguous();
    if (bias.has_value()) {
        bias = bias.value().to(at::kFloat).contiguous();
    }

    check_cuda_2d(activation, "activation");
    check_cuda_2d(weight, "weight");
    TORCH_CHECK(activation.scalar_type() == at::kChar, "activation must be int8");
    TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must be packed int8 storage");
    TORCH_CHECK(activation.device() == weight.device(), "activation and weight must be on the same CUDA device");
    TORCH_CHECK(activation_scale.is_cuda() && weight_scale.is_cuda(), "scales must be CUDA tensors");
    TORCH_CHECK(activation_scale.device() == activation.device() && weight_scale.device() == activation.device(),
                "scales must be on the activation device");
    TORCH_CHECK(activation.size(1) % 4 == 0, "activation K must be divisible by 4");
    TORCH_CHECK(weight.size(1) * 2 == activation.size(1), "packed weight K must match activation K");
    TORCH_CHECK(activation_scale.numel() == activation.size(0), "activation_scale must contain one value per row");
    TORCH_CHECK(weight_scale.numel() == weight.size(0), "weight_scale must contain one value per output channel");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda() && bias.value().device() == activation.device(),
                    "bias must be on the activation device");
        TORCH_CHECK(bias.value().numel() == weight.size(0), "bias must contain one value per output channel");
    }

    const at::cuda::CUDAGuard device_guard(activation.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 ||
                    (properties->major == 7 && properties->minor >= 5),
                "turing_w4a8_linear requires sm75 or newer");

    at::Tensor output = at::empty(
        {activation.size(0), weight.size(0)}, activation.options().dtype(at::kBFloat16));
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_w4a8_linear(from_torch(activation),
                                          from_torch(weight),
                                          from_torch(activation_scale),
                                          from_torch(weight_scale),
                                          maybe_tensor(bias),
                                          from_torch(output));
    return output;
}

at::Tensor turing_dequantize_int8_bf16(at::Tensor accumulator,
                                        at::Tensor activation_scale,
                                        at::Tensor weight_scale,
                                        int64_t output_columns) {
    accumulator = accumulator.contiguous();
    activation_scale = activation_scale.reshape({-1}).to(at::kFloat).contiguous();
    weight_scale = weight_scale.reshape({-1}).to(at::kFloat).contiguous();
    check_cuda_2d(accumulator, "accumulator");
    TORCH_CHECK(accumulator.scalar_type() == at::kInt,
                "accumulator must be int32");
    TORCH_CHECK(activation_scale.is_cuda() && weight_scale.is_cuda(),
                "scales must be CUDA tensors");
    TORCH_CHECK(accumulator.device() == activation_scale.device() &&
                    accumulator.device() == weight_scale.device(),
                "accumulator and scales must use the same CUDA device");

    const int64_t rows = accumulator.size(0);
    const int64_t accumulator_columns = accumulator.size(1);
    if (output_columns < 0) {
        output_columns = accumulator_columns;
    }
    TORCH_CHECK(rows > 0 && accumulator_columns > 0,
                "INT8 BF16 epilogue dimensions must be positive");
    TORCH_CHECK(output_columns > 0 && output_columns <= accumulator_columns,
                "output_columns must be positive and no larger than the accumulator width");
    TORCH_CHECK(activation_scale.numel() == rows,
                "activation_scale must contain one value per accumulator row");
    TORCH_CHECK(weight_scale.numel() == 1 || weight_scale.numel() == output_columns,
                "weight_scale must be scalar or contain one value per output column");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    accumulator_columns <= std::numeric_limits<int>::max() &&
                    output_columns <= std::numeric_limits<int>::max(),
                "INT8 BF16 epilogue dimensions exceed the CUDA kernel range");

    const at::cuda::CUDAGuard device_guard(accumulator.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 ||
                    (properties->major == 7 && properties->minor >= 5),
                "INT8 BF16 epilogue requires sm75 or newer");

    at::Tensor output = at::empty(
        {rows, output_columns}, accumulator.options().dtype(at::kBFloat16));
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_dequantize_int8_bf16(
        from_torch(accumulator),
        from_torch(activation_scale),
        from_torch(weight_scale),
        from_torch(output));
    return output;
}

std::tuple<at::Tensor, at::Tensor> turing_swiglu_int8_convrot_quantize(
    at::Tensor input, int64_t group_size) {
    input = input.contiguous();
    check_cuda_2d(input, "input");
    check_half_like(input, "input");
    TORCH_CHECK(group_size == 256, "SwiGLU staged ConvRot only supports group_size=256");
    TORCH_CHECK(input.size(1) % 2 == 0, "SwiGLU input width must be even");

    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1) / 2;
    TORCH_CHECK(rows > 0, "SwiGLU staged ConvRot requires at least one row");
    TORCH_CHECK(hidden > 0 && hidden % group_size == 0,
                "activated SwiGLU width must be positive and divisible by 256");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max(),
                "SwiGLU staged ConvRot dimensions exceed the CUDA kernel range");

    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 || (properties->major == 7 && properties->minor >= 5),
                "SwiGLU staged ConvRot requires sm75 or newer");

    at::Tensor rotated = at::empty({rows, hidden}, input.options());
    at::Tensor partial_absmax = at::empty(
        {rows, hidden / group_size}, input.options().dtype(at::kFloat));
    at::Tensor output = at::empty({rows, hidden}, input.options().dtype(at::kChar));
    at::Tensor scales = at::empty({rows, 1}, input.options().dtype(at::kFloat));

    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_swiglu_int8_convrot_quantize(
        from_torch(input),
        from_torch(rotated),
        from_torch(partial_absmax),
        from_torch(output),
        from_torch(scales));
    return {output, scales};
}

std::tuple<at::Tensor, at::Tensor> turing_swiglu_int4_convrot_quantize(
    at::Tensor input, int64_t group_size) {
    input = input.contiguous();
    check_cuda_2d(input, "input");
    check_half_like(input, "input");
    TORCH_CHECK(group_size == 256, "SwiGLU staged INT4 ConvRot only supports group_size=256");
    TORCH_CHECK(input.size(1) % 2 == 0, "SwiGLU input width must be even");

    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1) / 2;
    TORCH_CHECK(rows > 0, "SwiGLU staged INT4 ConvRot requires at least one row");
    TORCH_CHECK(hidden > 0 && hidden % group_size == 0,
                "activated SwiGLU width must be positive and divisible by 256");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max(),
                "SwiGLU staged INT4 ConvRot dimensions exceed the CUDA kernel range");

    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 || (properties->major == 7 && properties->minor >= 5),
                "SwiGLU staged INT4 ConvRot requires sm75 or newer");

    at::Tensor rotated = at::empty({rows, hidden}, input.options());
    at::Tensor partial_absmax = at::empty(
        {rows, hidden / group_size}, input.options().dtype(at::kFloat));
    at::Tensor output = at::empty({rows, hidden / 2}, input.options().dtype(at::kChar));
    at::Tensor scales = at::empty({rows, 1}, input.options().dtype(at::kFloat));

    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_swiglu_int4_convrot_quantize(
        from_torch(input),
        from_torch(rotated),
        from_torch(partial_absmax),
        from_torch(output),
        from_torch(scales));
    return {output, scales};
}

std::tuple<at::Tensor, at::Tensor> turing_bf16_int8_convrot_quantize(
    at::Tensor input, int64_t group_size, bool swiglu) {
    input = input.contiguous();
    check_cuda_2d(input, "input");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16,
                "BF16 row-buffer ConvRot input must be bfloat16");
    TORCH_CHECK(group_size == 256,
                "BF16 row-buffer ConvRot only supports group_size=256");

    const int64_t rows = input.size(0);
    const int64_t input_columns = input.size(1);
    TORCH_CHECK(!swiglu || input_columns % 2 == 0,
                "SwiGLU BF16 row-buffer ConvRot input width must be even");
    const int64_t hidden = swiglu ? input_columns / 2 : input_columns;
    TORCH_CHECK(rows > 0 && hidden > 0 && hidden % group_size == 0,
                "BF16 row-buffer ConvRot width must be positive and divisible by 256");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max(),
                "BF16 row-buffer ConvRot dimensions exceed the CUDA kernel range");

    constexpr int64_t shared_limit = 48 * 1024;
    const auto shared_bytes = [hidden](int threads) {
        const int64_t groups_in_flight = threads / 64;
        const int64_t dynamic_bytes =
            hidden * static_cast<int64_t>(sizeof(uint16_t)) +
            groups_in_flight * 2 * 256 * static_cast<int64_t>(sizeof(float));
        // ptxas reports three additional alignment words around the static
        // warp-reduction arrays.
        const int64_t static_bytes =
            (threads / 32 + 4) * static_cast<int64_t>(sizeof(float));
        return dynamic_bytes + static_bytes;
    };
    int block_threads = 0;
    for (const int candidate : {1024, 768, 512}) {
        if (shared_bytes(candidate) < shared_limit) {
            block_threads = candidate;
            break;
        }
    }
    TORCH_CHECK(block_threads != 0,
                "BF16 row-buffer ConvRot cannot fit under the 48 KiB shared-memory limit");

    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 ||
                    (properties->major == 7 && properties->minor >= 5),
                "BF16 row-buffer ConvRot requires sm75 or newer");

    at::Tensor output = at::empty(
        {rows, hidden}, input.options().dtype(at::kChar));
    at::Tensor scales = at::empty(
        {rows, 1}, input.options().dtype(at::kFloat));
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_bf16_int8_convrot_quantize(
        from_torch(input),
        from_torch(output),
        from_torch(scales),
        swiglu,
        block_threads);
    return {output, scales};
}

std::tuple<at::Tensor, at::Tensor> turing_bf16_int4_convrot_quantize(
    at::Tensor input, int64_t group_size, bool swiglu) {
    input = input.contiguous();
    check_cuda_2d(input, "input");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16,
                "BF16 row-buffer INT4 ConvRot input must be bfloat16");
    TORCH_CHECK(group_size == 256,
                "BF16 row-buffer INT4 ConvRot only supports group_size=256");

    const int64_t rows = input.size(0);
    const int64_t input_columns = input.size(1);
    TORCH_CHECK(!swiglu || input_columns % 2 == 0,
                "SwiGLU BF16 row-buffer INT4 ConvRot input width must be even");
    const int64_t hidden = swiglu ? input_columns / 2 : input_columns;
    TORCH_CHECK(rows > 0 && hidden > 0 && hidden % group_size == 0,
                "BF16 row-buffer INT4 ConvRot width must be positive and divisible by 256");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max(),
                "BF16 row-buffer INT4 ConvRot dimensions exceed the CUDA kernel range");

    constexpr int64_t shared_limit = 48 * 1024;
    const auto shared_bytes = [hidden](int threads) {
        const int64_t groups_in_flight = threads / 64;
        const int64_t dynamic_bytes =
            hidden * static_cast<int64_t>(sizeof(uint16_t)) +
            groups_in_flight * 2 * 256 * static_cast<int64_t>(sizeof(float));
        const int64_t static_bytes =
            (threads / 32 + 4) * static_cast<int64_t>(sizeof(float));
        return dynamic_bytes + static_bytes;
    };
    int block_threads = 0;
    for (const int candidate : {1024, 768, 512}) {
        if (shared_bytes(candidate) < shared_limit) {
            block_threads = candidate;
            break;
        }
    }
    TORCH_CHECK(block_threads != 0,
                "BF16 row-buffer INT4 ConvRot cannot fit under the 48 KiB shared-memory limit");

    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 || (properties->major == 7 && properties->minor >= 5),
                "BF16 row-buffer INT4 ConvRot requires sm75 or newer");

    at::Tensor output = at::empty({rows, hidden / 2}, input.options().dtype(at::kChar));
    at::Tensor scales = at::empty({rows, 1}, input.options().dtype(at::kFloat));
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_bf16_int4_convrot_quantize(
        from_torch(input),
        from_torch(output),
        from_torch(scales),
        swiglu,
        block_threads);
    return {output, scales};
}

at::Tensor turing_segmented_rms_adaln(at::Tensor input,
                                       at::Tensor weight,
                                       at::Tensor scale,
                                       at::Tensor shift,
                                       at::Tensor segments,
                                       double epsilon) {
    input = input.contiguous();
    weight = weight.contiguous();
    segments = segments.contiguous();
    check_cuda_2d(input, "input");
    check_float_like(input, "input");
    TORCH_CHECK(weight.is_cuda() && weight.dim() == 1 && weight.is_contiguous(),
                "weight must be a contiguous 1D CUDA tensor");
    TORCH_CHECK(scale.is_cuda() && scale.dim() == 2 && scale.stride(1) == 1,
                "scale must be a CUDA matrix with contiguous rows");
    TORCH_CHECK(shift.is_cuda() && shift.dim() == 2 && shift.stride(1) == 1,
                "shift must be a CUDA matrix with contiguous rows");
    TORCH_CHECK(segments.is_cuda() && segments.dim() == 2 && segments.is_contiguous(),
                "segments must be a contiguous 2D CUDA tensor");
    TORCH_CHECK(segments.scalar_type() == at::kInt && segments.size(1) == 3,
                "segments must be int32 [start, stop, modulation_row] triples");
    TORCH_CHECK(input.device() == weight.device() &&
                    input.device() == scale.device() &&
                    input.device() == shift.device() &&
                    input.device() == segments.device(),
                "all segmented RMSNorm+AdaLN tensors must use the same CUDA device");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type() &&
                    scale.scalar_type() == input.scalar_type() &&
                    shift.scalar_type() == input.scalar_type(),
                "weight, scale, and shift dtypes must match input");

    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1);
    TORCH_CHECK(rows > 0 && hidden > 0,
                "segmented RMSNorm+AdaLN input dimensions must be positive");
    TORCH_CHECK(weight.numel() == hidden,
                "RMSNorm weight length must match the input hidden dimension");
    TORCH_CHECK(scale.size(0) > 0 && scale.size(1) == hidden,
                "scale must be [modulation_rows, hidden]");
    TORCH_CHECK(shift.sizes() == scale.sizes(), "shift shape must match scale shape");
    TORCH_CHECK(segments.size(0) > 0, "segments must contain at least one row");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max() &&
                    scale.size(0) <= std::numeric_limits<int>::max() &&
                    scale.stride(0) <= std::numeric_limits<int>::max() &&
                    shift.stride(0) <= std::numeric_limits<int>::max() &&
                    segments.size(0) <= std::numeric_limits<int>::max(),
                "segmented RMSNorm+AdaLN dimensions exceed the CUDA kernel range");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
                "RMSNorm epsilon must be finite and positive");

    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 || (properties->major == 7 && properties->minor >= 5),
                "segmented RMSNorm+AdaLN requires sm75 or newer");

    at::Tensor output = at::empty_like(input);
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_segmented_rms_adaln(
        from_torch(input),
        from_torch(weight),
        from_torch(scale),
        from_torch(shift),
        from_torch(segments),
        from_torch(output),
        static_cast<float>(epsilon));
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("turing_w4a8_linear",
          &turing_w4a8_linear,
          pybind11::arg("activation"),
          pybind11::arg("weight"),
          pybind11::arg("activation_scale"),
          pybind11::arg("weight_scale"),
          pybind11::arg("bias") = std::nullopt);
    m.def("turing_dequantize_int8_bf16",
          &turing_dequantize_int8_bf16,
          pybind11::arg("accumulator"),
          pybind11::arg("activation_scale"),
          pybind11::arg("weight_scale"),
          pybind11::arg("output_columns") = -1);
    m.def("turing_swiglu_int8_convrot_quantize",
          &turing_swiglu_int8_convrot_quantize,
          pybind11::arg("input"),
          pybind11::arg("group_size") = 256);
    m.def("turing_swiglu_int4_convrot_quantize",
          &turing_swiglu_int4_convrot_quantize,
          pybind11::arg("input"),
          pybind11::arg("group_size") = 256);
    m.def("turing_bf16_int8_convrot_quantize",
          &turing_bf16_int8_convrot_quantize,
          pybind11::arg("input"),
          pybind11::arg("group_size") = 256,
          pybind11::arg("swiglu") = false);
    m.def("turing_bf16_int4_convrot_quantize",
          &turing_bf16_int4_convrot_quantize,
          pybind11::arg("input"),
          pybind11::arg("group_size") = 256,
          pybind11::arg("swiglu") = false);
    m.def("turing_segmented_rms_adaln",
          &turing_segmented_rms_adaln,
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("scale"),
          pybind11::arg("shift"),
          pybind11::arg("segments"),
          pybind11::arg("epsilon") = 1.0e-5);
}
