from __future__ import annotations

import logging
import sys

import torch


LOG = logging.getLogger("comfyui-turing-utils")
BACKEND_NAME = "turing_utils_sm75"
TURING_SHARED_MEMORY_LIMIT = 48 * 1024
_NVIDIA_TURING_WITHOUT_TENSOR_CORES = (
    "GTX 1630",
    "GTX 1650",
    "GTX 1660",
    "T500",
    "T550",
    "T600",
    "MX450",
    "MX550",
    "CMP 30HX",
    "T1000",
    "T1200",
    "T2000",
)
_PREFLIGHTED_DEVICES: set[int] = set()
_PREFLIGHTED_KITCHEN: set[tuple[int, bool, bool]] = set()


def is_supported_turing_device(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    if torch.cuda.get_device_capability(index) != (7, 5):
        return False
    name = torch.cuda.get_device_name(index)
    return not any(model in name for model in _NVIDIA_TURING_WITHOUT_TENSOR_CORES)


def _kernel_available() -> bool:
    try:
        from comfyui_turing_utils_kernel import _C
    except (ImportError, OSError):
        return False
    return hasattr(_C, "turing_w4a8_linear")


def backend_available() -> bool:
    try:
        import comfy_kitchen
    except ImportError:
        return False
    status = comfy_kitchen.list_backends().get(BACKEND_NAME, {})
    return bool(status.get("available") and not status.get("disabled"))


def preflight_w4a8(device: torch.device) -> None:
    if not is_supported_turing_device(device):
        raise RuntimeError(f"unsupported device {device}")
    if not _kernel_available():
        raise RuntimeError("the installed comfyui-turing-utils-kernel does not provide Turing W4A8")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_DEVICES:
        return

    from comfyui_turing_utils_kernel import turing_w4a8_linear

    activation = ((torch.arange(3 * 64, device=device) % 23) - 11).to(torch.int8).reshape(3, 64)
    weight_values = ((torch.arange(5 * 64, device=device) % 15) - 7).to(torch.int8).reshape(5, 64)
    low = weight_values[:, 0::2].to(torch.int32) & 0x0f
    high = weight_values[:, 1::2].to(torch.int32) & 0x0f
    packed_weight = (low | (high << 4)).to(torch.int8)
    activation_scale = torch.linspace(0.01, 0.03, 3, device=device)
    weight_scale = torch.linspace(0.02, 0.06, 5, device=device)
    bias = torch.linspace(-0.2, 0.2, 5, dtype=torch.bfloat16, device=device)
    output = turing_w4a8_linear(
        activation,
        packed_weight,
        activation_scale,
        weight_scale,
        bias,
    )
    reference = (
        activation.float() @ weight_values.float().t()
    ) * activation_scale[:, None] * weight_scale[None, :] + bias.float()
    if output.dtype != torch.bfloat16 or not torch.allclose(output.float(), reference, rtol=0.01, atol=0.01):
        raise RuntimeError("packed W4A8 numerical self-test failed")
    for hidden_size in (256, 8192):
        bf16_input = (
            ((torch.arange(3 * hidden_size, device=device) % 29) - 14)
            .reshape(3, hidden_size)
            .to(torch.bfloat16)
            / 16
        )
        full_weight = torch.zeros((5, hidden_size // 2), dtype=torch.int8, device=device)
        full_output = convrot_w4a4_linear(
            bf16_input,
            full_weight,
            torch.ones((5,), dtype=torch.float32, device=device),
            convrot_groupsize=256,
            quant_group_size=64,
            linear_dtype="int8",
        )
        if full_output.dtype != torch.bfloat16 or not torch.isfinite(full_output).all():
            raise RuntimeError(f"BF16 ConvRot W4A8 self-test failed for K={hidden_size}")
    swiglu_input = torch.zeros((3, 512), dtype=torch.bfloat16, device=device)
    swiglu_output = convrot_w4a4_linear(
        swiglu_input,
        torch.zeros((5, 128), dtype=torch.int8, device=device),
        torch.ones((5,), dtype=torch.float32, device=device),
        convrot_groupsize=256,
        quant_group_size=64,
        linear_dtype="int8",
        input_act="swiglu",
    )
    if swiglu_output.dtype != torch.bfloat16 or not torch.isfinite(swiglu_output).all():
        raise RuntimeError("SwiGLU W4A8 BF16 self-test failed")
    torch.cuda.synchronize(device)
    _PREFLIGHTED_DEVICES.add(index)


def preflight_kitchen(device: torch.device, w4a4: bool, w8a8: bool) -> None:
    if not is_supported_turing_device(device):
        raise RuntimeError(f"unsupported device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, w4a4, w8a8)
    if key in _PREFLIGHTED_KITCHEN:
        return

    import comfy_kitchen

    if w4a4:
        for hidden_size in (256, 16384):
            x = (
                ((torch.arange(16 * hidden_size, device=device) % 31) - 15)
                .reshape(16, hidden_size)
                .to(torch.bfloat16)
                / 16
            )
            packed_weight = torch.zeros((64, hidden_size // 2), dtype=torch.int8, device=device)
            weight_scale = torch.ones((64,), dtype=torch.float32, device=device)
            output = comfy_kitchen.convrot_w4a4_linear(
                x,
                packed_weight,
                weight_scale,
                convrot_groupsize=256,
                quant_group_size=64,
                linear_dtype="int4",
            )
            if output.dtype != torch.bfloat16 or not torch.isfinite(output).all():
                raise RuntimeError(f"Kitchen W4A4 BF16 self-test failed for K={hidden_size}")
        swiglu_input = torch.zeros((16, 512), dtype=torch.bfloat16, device=device)
        swiglu_weight = torch.zeros((64, 128), dtype=torch.int8, device=device)
        swiglu_output = convrot_w4a4_linear(
            swiglu_input,
            swiglu_weight,
            torch.ones((64,), dtype=torch.float32, device=device),
            convrot_groupsize=256,
            quant_group_size=64,
            linear_dtype="int4",
            input_act="swiglu",
        )
        if swiglu_output.dtype != torch.bfloat16 or not torch.isfinite(swiglu_output).all():
            raise RuntimeError("SwiGLU W4A4 BF16 self-test failed")
    if w8a8:
        for hidden_size in (256, 5376):
            x = (
                ((torch.arange(16 * hidden_size, device=device) % 31) - 15)
                .reshape(16, hidden_size)
                .to(torch.bfloat16)
                / 16
            )
            weight = torch.zeros((64, hidden_size), dtype=torch.int8, device=device)
            weight_scale = torch.ones((), dtype=torch.float32, device=device)
            output = comfy_kitchen.int8_linear(
                x,
                weight,
                weight_scale,
                out_dtype=torch.bfloat16,
                convrot=True,
                convrot_groupsize=256,
            )
            if output.dtype != torch.bfloat16 or not torch.isfinite(output).all():
                raise RuntimeError(f"Kitchen W8A8 BF16 self-test failed for K={hidden_size}")
        swiglu_input = torch.cat((x, x), dim=-1)
        swiglu_output = comfy_kitchen.int8_linear(
            swiglu_input,
            weight,
            weight_scale,
            out_dtype=torch.bfloat16,
            convrot=True,
            convrot_groupsize=256,
            input_act="swiglu",
        )
        if swiglu_output.dtype != torch.bfloat16 or not torch.isfinite(swiglu_output).all():
            raise RuntimeError("Kitchen SwiGLU W8A8 BF16 self-test failed")
        contraction_input = torch.zeros((129, 256), dtype=torch.bfloat16, device=device)
        contraction_output = comfy_kitchen.int8_linear(
            contraction_input,
            torch.zeros((64, 256), dtype=torch.int8, device=device),
            torch.ones((), dtype=torch.float32, device=device),
            out_dtype=torch.bfloat16,
            convrot=True,
            convrot_groupsize=256,
        )
        if contraction_output.dtype != torch.bfloat16 or not torch.isfinite(contraction_output).all():
            raise RuntimeError("Turing W8A8 BF16 contraction self-test failed")
    torch.cuda.synchronize(device)
    _PREFLIGHTED_KITCHEN.add(key)


def _convrot_int8_shared_memory_bytes(rows: int, hidden_size: int) -> int:
    if rows == 1:
        block_threads = 512
    elif hidden_size == 256:
        block_threads = 64
    elif hidden_size == 2560:
        block_threads = 640
    elif hidden_size == 6144:
        block_threads = 768
    else:
        block_threads = 1024
    groups_in_flight = block_threads // 64
    return (hidden_size + groups_in_flight * 2 * 256) * 4


def _convrot_int8_bf16_rowbuffer_fits(hidden_size: int) -> bool:
    for block_threads in (1024, 768, 512):
        groups_in_flight = block_threads // 64
        dynamic_bytes = hidden_size * 2 + groups_in_flight * 2 * 256 * 4
        # ptxas reserves three additional aligned words around the static
        # warp-reduction arrays (80/112/144 bytes for 512/768/1024 threads).
        static_bytes = (block_threads // 32 + 4) * 4
        if dynamic_bytes + static_bytes < TURING_SHARED_MEMORY_LIMIT:
            return True
    return False


def _convrot_int4_shared_memory_bytes(rows: int, hidden_size: int, element_size: int) -> int:
    if rows != 1 and hidden_size <= 4096:
        block_threads = 256
        scratch_buffers = 2
    elif rows == 1:
        block_threads = 512
        scratch_buffers = 2
    elif hidden_size == 15360:
        block_threads = 640
        scratch_buffers = 1
    else:
        block_threads = 1024
        scratch_buffers = 2
    groups_in_flight = block_threads // 64
    return (hidden_size + groups_in_flight * scratch_buffers * 256) * element_size


def _quantize_turing_int8_activation(
    x2d: torch.Tensor,
    group_size: int,
    input_act: str | None = None,
):
    from comfy_kitchen.backends import cuda as kitchen_cuda

    if input_act not in (None, "none", "swiglu"):
        raise ValueError(f"unsupported fused Turing INT8 activation: {input_act!r}")
    hidden_size = x2d.shape[1] // 2 if input_act == "swiglu" else x2d.shape[1]
    requested_shared = _convrot_int8_shared_memory_bytes(x2d.shape[0], hidden_size)
    if requested_shared < TURING_SHARED_MEMORY_LIMIT:
        if input_act == "swiglu":
            return kitchen_cuda.quantize_int8_rowwise_convrot64(
                x2d, group_size, input_act="swiglu"
            )
        return kitchen_cuda.quantize_int8_rowwise_convrot64(x2d, group_size)
    if x2d.dtype == torch.bfloat16 and _convrot_int8_bf16_rowbuffer_fits(hidden_size):
        try:
            from comfyui_turing_utils_kernel import turing_bf16_int8_convrot_quantize
        except (ImportError, AttributeError):
            pass
        else:
            return turing_bf16_int8_convrot_quantize(
                x2d,
                group_size,
                swiglu=input_act == "swiglu",
            )
    if input_act == "swiglu":
        try:
            from comfyui_turing_utils_kernel import turing_swiglu_int8_convrot_quantize
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Turing W8A8 SwiGLU requires an updated comfyui-turing-utils-kernel; "
                "reinstall the kernel package"
            ) from exc
        return turing_swiglu_int8_convrot_quantize(x2d, group_size)
    staged = getattr(kitchen_cuda, "quantize_int8_convrot_staged", None)
    if staged is None:
        raise RuntimeError(
            "Turing INT8 activation requires Kitchen staged ConvRot quantization "
            "when the 48 KiB shared-memory limit is exceeded"
        )
    return staged(x2d, group_size)


def _quantize_turing_int4_activation(
    x2d: torch.Tensor,
    group_size: int,
    input_act: str | None = None,
):
    from comfy_kitchen.backends import cuda as kitchen_cuda

    if input_act not in (None, "none", "swiglu"):
        raise ValueError(f"unsupported fused Turing INT4 activation: {input_act!r}")
    if input_act == "swiglu":
        hidden_size = x2d.shape[1] // 2
        if x2d.shape[1] % 2 != 0:
            raise ValueError("SwiGLU input width must be even")
        if x2d.dtype == torch.bfloat16 and _convrot_int8_bf16_rowbuffer_fits(hidden_size):
            try:
                from comfyui_turing_utils_kernel import turing_bf16_int4_convrot_quantize
            except (ImportError, AttributeError):
                pass
            else:
                return turing_bf16_int4_convrot_quantize(
                    x2d,
                    group_size,
                    swiglu=True,
                )
        try:
            from comfyui_turing_utils_kernel import turing_swiglu_int4_convrot_quantize
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Turing W4A4 SwiGLU requires an updated comfyui-turing-utils-kernel; reinstall the kernel package"
            ) from exc
        return turing_swiglu_int4_convrot_quantize(x2d, group_size)

    requested_shared = _convrot_int4_shared_memory_bytes(
        x2d.shape[0], x2d.shape[1], x2d.element_size()
    )
    if requested_shared < TURING_SHARED_MEMORY_LIMIT:
        return kitchen_cuda.quantize_int4_rowwise_convrot64(x2d, group_size)
    if x2d.dtype == torch.bfloat16 and _convrot_int8_bf16_rowbuffer_fits(x2d.shape[1]):
        try:
            from comfyui_turing_utils_kernel import turing_bf16_int4_convrot_quantize
        except (ImportError, AttributeError):
            pass
        else:
            return turing_bf16_int4_convrot_quantize(
                x2d,
                group_size,
                swiglu=False,
            )
    rotate = getattr(kitchen_cuda, "rotate_int8_convrot_weight", None)
    if rotate is None:
        raise RuntimeError(
            "Turing INT4 activation requires Kitchen grouped ConvRot rotation "
            "when the 48 KiB shared-memory limit is exceeded"
        )
    rotated = rotate(x2d, group_size)
    return kitchen_cuda.quantize_int4_rowwise(rotated)


def _turing_cublas_int8_bf16(
    qactivation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor | None:
    """Run Kitchen's Turing cuBLAS fallback with the bundled BF16 epilogue."""
    from comfy_kitchen.backends import cuda as kitchen_cuda

    try:
        from comfyui_turing_utils_kernel import turing_dequantize_int8_bf16
    except (ImportError, AttributeError):
        return None

    required = (
        "_C",
        "_cublas_int8_n_alignment",
        "_pad_2d_cols",
        "_pad_2d_rows",
        "_round_up",
        "_wrap_for_dlpack",
        "get_cublas_workspace",
    )
    if not all(hasattr(kitchen_cuda, name) for name in required):
        return None
    if not hasattr(kitchen_cuda._C, "cublas_gemm_int8"):
        return None

    m, k = qactivation.shape
    n = weight.shape[0]
    padded_k = kitchen_cuda._round_up(k, 16)
    padded_n = kitchen_cuda._round_up(
        n, kitchen_cuda._cublas_int8_n_alignment(qactivation)
    )
    cublas_x = kitchen_cuda._pad_2d_cols(qactivation, padded_k)
    cublas_weight = kitchen_cuda._pad_2d_rows(
        kitchen_cuda._pad_2d_cols(weight, padded_k), padded_n
    )
    accumulator = torch.empty(
        (m, padded_n), dtype=torch.int32, device=qactivation.device
    )
    stream_ptr = torch.cuda.current_stream(qactivation.device).cuda_stream
    kitchen_cuda._C.cublas_gemm_int8(
        kitchen_cuda._wrap_for_dlpack(cublas_x),
        kitchen_cuda._wrap_for_dlpack(cublas_weight),
        kitchen_cuda._wrap_for_dlpack(accumulator),
        kitchen_cuda._wrap_for_dlpack(kitchen_cuda.get_cublas_workspace()),
        stream_ptr,
    )
    return turing_dequantize_int8_bf16(
        accumulator,
        activation_scale,
        weight_scale,
        output_columns=n,
    )


def _turing_int8_gemm(
    qactivation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Keep scalar scales in fused GEMMs and use the fast no-bias BF16 epilogue."""
    from comfy_kitchen.backends import cuda as kitchen_cuda

    m, k = qactivation.shape
    n = weight.shape[0]
    activation_scale = (
        activation_scale.to(device=qactivation.device, dtype=torch.float32)
        .reshape(-1)
        .contiguous()
    )
    weight_scale = (
        weight_scale.to(device=qactivation.device, dtype=torch.float32)
        .reshape(-1)
        .contiguous()
    )
    if activation_scale.numel() != m:
        raise ValueError(
            f"Turing W8A8 activation scale must have {m} values, "
            f"got {activation_scale.numel()}"
        )
    if weight_scale.numel() not in (1, n):
        raise ValueError(
            f"Turing W8A8 weight scale must be scalar or have {n} values, "
            f"got {weight_scale.numel()}"
        )

    prefer_fused = getattr(kitchen_cuda, "_prefer_turing_fused_int8", None)
    fused_linear = getattr(kitchen_cuda, "_int8_linear_turing_quantized", None)
    if (
        callable(prefer_fused)
        and prefer_fused(m, n, k)
        and callable(fused_linear)
    ):
        output = fused_linear(
            qactivation,
            weight,
            activation_scale,
            weight_scale,
            bias,
            output_dtype,
        )
        if output is not None:
            return output

    if bias is None and output_dtype == torch.bfloat16:
        output = _turing_cublas_int8_bf16(
            qactivation,
            weight,
            activation_scale,
            weight_scale,
        )
        if output is not None:
            return output

    quantized_linear = getattr(kitchen_cuda, "_int4_linear_via_int8_values", None)
    if quantized_linear is None:
        raise RuntimeError("Turing W8A8 requires Kitchen quantized INT8 linear support")
    expanded_weight_scale = weight_scale
    if expanded_weight_scale.numel() == 1:
        expanded_weight_scale = expanded_weight_scale.expand(n).contiguous()
    return quantized_linear(
        qactivation,
        weight,
        activation_scale,
        expanded_weight_scale,
        bias,
        output_dtype,
    )


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
) -> torch.Tensor:
    from comfy_kitchen.backends import cuda as kitchen_cuda
    from comfy_kitchen.backends._activations import apply_input_act

    if (
        x.dtype != torch.bfloat16
        or not is_supported_turing_device(x.device)
        or not convrot
        or convrot_groupsize != 256
    ):
        return kitchen_cuda.int8_linear(
            x,
            weight,
            weight_scale,
            bias=bias,
            out_dtype=out_dtype,
            convrot=convrot,
            convrot_groupsize=convrot_groupsize,
            input_act=input_act,
        )

    original_shape = x.shape
    x2d = x.reshape(-1, original_shape[-1]).contiguous()
    if input_act == "swiglu":
        qactivation, activation_scale = _quantize_turing_int8_activation(
            x2d, convrot_groupsize, input_act="swiglu"
        )
    else:
        x2d = apply_input_act(x2d, input_act)
        qactivation, activation_scale = _quantize_turing_int8_activation(
            x2d, convrot_groupsize
        )

    output_dtype = out_dtype or x.dtype
    output_channels = weight.shape[0]
    output = _turing_int8_gemm(
        qactivation,
        weight.contiguous(),
        activation_scale,
        weight_scale,
        bias,
        output_dtype,
    )
    return output.reshape(*original_shape[:-1], output_channels)


def convrot_w4a4_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
    quant_group_size: int = 64,
    linear_dtype: str = "int4",
    input_act: str | None = None,
) -> torch.Tensor:
    from comfy_kitchen.backends import cuda as kitchen_cuda
    from comfy_kitchen.backends._activations import apply_input_act

    if (
        linear_dtype not in {"int4", "int8"}
        or convrot_groupsize != 256
        or quant_group_size != 64
        or x.dtype != torch.bfloat16
        or not is_supported_turing_device(x.device)
    ):
        x = apply_input_act(x, input_act)
        return kitchen_cuda.convrot_w4a4_linear(
            x,
            qweight,
            wscales,
            bias=bias,
            convrot_groupsize=convrot_groupsize,
            quant_group_size=quant_group_size,
            linear_dtype=linear_dtype,
        )

    original_shape = x.shape
    x2d = x.reshape(-1, original_shape[-1]).contiguous()
    if input_act not in (None, "none", "swiglu"):
        x2d = apply_input_act(x2d, input_act)
        input_act = None
    if linear_dtype == "int8":
        from comfyui_turing_utils_kernel import turing_w4a8_linear

        qactivation, activation_scale = _quantize_turing_int8_activation(
            x2d,
            convrot_groupsize,
            input_act=input_act,
        )
        output = turing_w4a8_linear(qactivation, qweight, activation_scale, wscales, bias)
    else:
        qactivation, activation_scale = _quantize_turing_int4_activation(
            x2d,
            convrot_groupsize,
            input_act=input_act,
        )
        output = kitchen_cuda.int4_linear(
            qactivation,
            qweight.contiguous(),
            activation_scale,
            wscales,
            bias=bias,
            out_dtype=x.dtype,
        )
    output_shape = original_shape[:-1]
    return output.reshape(*output_shape, qweight.shape[0])


def register_backend() -> bool:
    try:
        import comfy_kitchen
        from comfy_kitchen.constraints import (
            ExactDims,
            FunctionConstraints,
            MinDims,
            ParamConstraint,
            ShapeRule,
            ValidationResult,
        )
        from comfy_kitchen.registry import registry
    except ImportError:
        return False

    cuda_status = comfy_kitchen.list_backends().get("cuda", {})
    cuda_capabilities = set(cuda_status.get("capabilities", ()))
    if not {"convrot_w4a4_linear", "int8_linear"}.issubset(cuda_capabilities):
        return False
    if BACKEND_NAME in comfy_kitchen.list_backends():
        return backend_available()

    class SupportedTuringTensor(ShapeRule):
        def check(self, tensor: torch.Tensor) -> bool:
            return is_supported_turing_device(tensor.device)

        def describe(self) -> str:
            return "tensor on supported NVIDIA Turing/sm75 device"

    cuda_devices = frozenset({"cuda"})
    standard_floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    has_w4a8_kernel = _kernel_available()

    def require_convrot_256(kwargs):
        if kwargs.get("convrot") is not True:
            return ValidationResult.fail("convrot", "Turing staged INT8 requires ConvRot")
        if kwargs.get("convrot_groupsize") != 256:
            return ValidationResult.fail("convrot_groupsize", "Turing staged INT8 requires group size 256")
        return ValidationResult.ok()

    def require_w4_convrot_256(kwargs):
        if kwargs.get("convrot_groupsize") != 256:
            return ValidationResult.fail("convrot_groupsize", "Turing W4 requires ConvRot group size 256")
        if kwargs.get("quant_group_size") != 64:
            return ValidationResult.fail("quant_group_size", "Turing W4 requires quantization group size 64")
        if kwargs.get("linear_dtype") not in {"int4", "int8"}:
            return ValidationResult.fail("linear_dtype", "Turing W4 requires int4 or int8 activation")
        if kwargs.get("linear_dtype") == "int8" and not has_w4a8_kernel:
            return ValidationResult.fail("linear_dtype", "Turing W4A8 kernel is unavailable")
        return ValidationResult.ok()

    registry.register(
        BACKEND_NAME,
        sys.modules[__name__],
        {
            "int8_linear": FunctionConstraints(
                params={
                    "x": ParamConstraint(
                        dtypes=frozenset({torch.bfloat16}),
                        shape_rules=(MinDims(2), SupportedTuringTensor()),
                    ),
                    "weight": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
                    "weight_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                    "bias": ParamConstraint(dtypes=standard_floats),
                    "out_dtype": ParamConstraint(dtypes=standard_floats),
                    "convrot": ParamConstraint(dtypes=frozenset({bool})),
                    "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                    "input_act": ParamConstraint(dtypes=frozenset({str, type(None)})),
                },
                default_devices=cuda_devices,
                call_rules=(require_convrot_256,),
            ),
            "convrot_w4a4_linear": FunctionConstraints(
                params={
                    "x": ParamConstraint(
                        dtypes=frozenset({torch.bfloat16}),
                        shape_rules=(MinDims(2), SupportedTuringTensor()),
                    ),
                    "qweight": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
                    "wscales": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(1),)),
                    "bias": ParamConstraint(dtypes=standard_floats),
                    "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                    "quant_group_size": ParamConstraint(dtypes=frozenset({int})),
                    "linear_dtype": ParamConstraint(dtypes=frozenset({str})),
                    "input_act": ParamConstraint(dtypes=frozenset({str, type(None)})),
                },
                default_devices=cuda_devices,
                call_rules=(require_w4_convrot_256,),
            ),
        },
    )
    existing_priority = list(getattr(registry, "_priority", ("cuda", "triton", "eager")))
    registry.set_priority([BACKEND_NAME, *(name for name in existing_priority if name != BACKEND_NAME)])
    LOG.info("Registered Turing Utils ConvRot backend")
    return True
