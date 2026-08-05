from __future__ import annotations

import logging
import sys

import torch


LOG = logging.getLogger("comfyui-svdint4")
BACKEND_NAME = "svdint4_turing"
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
        from svdint4 import _C
    except (ImportError, OSError):
        return False
    return hasattr(_C, "turing_w4a8_linear")


def backend_available() -> bool:
    try:
        import comfy_kitchen
    except ImportError:
        return False
    status = comfy_kitchen.list_backends().get(BACKEND_NAME, {})
    return bool(status.get("available") and not status.get("disabled") and _kernel_available())


def preflight_w4a8(device: torch.device) -> None:
    if not is_supported_turing_device(device):
        raise RuntimeError(f"unsupported device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_DEVICES:
        return

    from svdint4 import turing_w4a8_linear

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
        for hidden_size in (256,):
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
    if w8a8:
        for hidden_size in (256,):
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
        fused_output = comfy_kitchen.int8_linear(
            swiglu_input,
            weight,
            weight_scale,
            out_dtype=torch.bfloat16,
            convrot=True,
            convrot_groupsize=256,
            input_act="swiglu",
        )
        if fused_output.dtype != torch.bfloat16 or not torch.isfinite(fused_output).all():
            raise RuntimeError("Kitchen fused SwiGLU W8A8 BF16 self-test failed")
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


def _quantize_turing_w4a8(x2d: torch.Tensor, group_size: int):
    from comfy_kitchen.backends import cuda as kitchen_cuda

    requested_shared = _convrot_int8_shared_memory_bytes(x2d.shape[0], x2d.shape[1])
    if requested_shared <= TURING_SHARED_MEMORY_LIMIT:
        return kitchen_cuda.quantize_int8_rowwise_convrot64(x2d, group_size)
    staged = getattr(kitchen_cuda, "quantize_int8_convrot_staged", None)
    if staged is None:
        raise RuntimeError(
            "SVDInt4 Turing W4A8 requires Kitchen staged ConvRot quantization "
            "when the 48 KiB shared-memory limit is exceeded"
        )
    return staged(x2d, group_size)


def convrot_w4a4_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
    quant_group_size: int = 64,
    linear_dtype: str = "int4",
) -> torch.Tensor:
    from comfy_kitchen.backends import cuda as kitchen_cuda

    if linear_dtype != "int8" or convrot_groupsize != 256 or quant_group_size != 64:
        return kitchen_cuda.convrot_w4a4_linear(
            x,
            qweight,
            wscales,
            bias=bias,
            convrot_groupsize=convrot_groupsize,
            quant_group_size=quant_group_size,
            linear_dtype=linear_dtype,
        )

    if x.dtype != torch.bfloat16:
        raise RuntimeError(f"SVDInt4 Turing W4A8 requires BF16 input, got {x.dtype}")
    if not is_supported_turing_device(x.device):
        raise RuntimeError(f"SVDInt4 Turing W4A8 received unsupported device {x.device}")

    from svdint4 import turing_w4a8_linear

    original_shape = x.shape
    x2d = x.reshape(-1, original_shape[-1]).contiguous()
    qactivation, activation_scale = _quantize_turing_w4a8(x2d, convrot_groupsize)
    output = turing_w4a8_linear(qactivation, qweight, activation_scale, wscales, bias)
    return output.reshape(*original_shape[:-1], qweight.shape[0])


def register_backend() -> bool:
    try:
        import comfy_kitchen
        from comfy_kitchen.constraints import ExactDims, FunctionConstraints, MinDims, ParamConstraint, ShapeRule
        from comfy_kitchen.registry import registry
    except ImportError:
        return False

    cuda_status = comfy_kitchen.list_backends().get("cuda", {})
    if "convrot_w4a4_linear" not in cuda_status.get("capabilities", ()) or not _kernel_available():
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
    registry.register(
        BACKEND_NAME,
        sys.modules[__name__],
        {
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
                },
                default_devices=cuda_devices,
            )
        },
    )
    existing_priority = list(getattr(registry, "_priority", ("cuda", "triton", "eager")))
    registry.set_priority([BACKEND_NAME, *(name for name in existing_priority if name != BACKEND_NAME)])
    LOG.info("Registered SVDInt4 Turing ConvRot backend")
    return True
