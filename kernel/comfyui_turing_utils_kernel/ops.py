from __future__ import annotations

import torch

from . import _C


def turing_w4a8_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if activation.device.type != "cuda":
        raise RuntimeError("Turing W4A8 requires CUDA tensors")
    if torch.cuda.get_device_capability(activation.device) < (7, 5):
        raise RuntimeError("Turing W4A8 requires sm75 or newer")
    return _C.turing_w4a8_linear(
        activation.contiguous(),
        weight.contiguous(),
        activation_scale.contiguous(),
        weight_scale.contiguous(),
        None if bias is None else bias.contiguous(),
    )


def turing_dequantize_int8_bf16(
    accumulator: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_columns: int = -1,
) -> torch.Tensor:
    """Dequantize an INT32 GEMM workspace directly into packed BF16 stores."""
    if accumulator.device.type != "cuda":
        raise RuntimeError("Turing BF16 epilogue requires CUDA tensors")
    if torch.cuda.get_device_capability(accumulator.device) < (7, 5):
        raise RuntimeError("Turing BF16 epilogue requires sm75 or newer")
    if accumulator.dtype != torch.int32 or accumulator.ndim != 2:
        raise TypeError("accumulator must be a 2D int32 tensor")
    return _C.turing_dequantize_int8_bf16(
        accumulator.contiguous(),
        activation_scale.contiguous(),
        weight_scale.contiguous(),
        output_columns,
    )


def turing_swiglu_int8_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse SwiGLU into the first pass of staged ConvRot INT8 quantization."""
    if x.device.type != "cuda":
        raise RuntimeError("Turing SwiGLU ConvRot requires CUDA tensors")
    if torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("Turing SwiGLU ConvRot requires sm75 or newer")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("SwiGLU ConvRot input must be float16 or bfloat16")
    if x.ndim != 2:
        raise ValueError("SwiGLU ConvRot input must be 2D [M, 2K]")
    return _C.turing_swiglu_int8_convrot_quantize(x.contiguous(), group_size)


def turing_swiglu_int4_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse SwiGLU into staged ConvRot INT4 activation quantization."""
    if x.device.type != "cuda":
        raise RuntimeError("Turing SwiGLU INT4 ConvRot requires CUDA tensors")
    if torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("Turing SwiGLU INT4 ConvRot requires sm75 or newer")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("SwiGLU INT4 ConvRot input must be float16 or bfloat16")
    if x.ndim != 2:
        raise ValueError("SwiGLU INT4 ConvRot input must be 2D [M, 2K]")
    return _C.turing_swiglu_int4_convrot_quantize(x.contiguous(), group_size)


def turing_gelu_int8_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse tanh-GELU into staged ConvRot INT8 activation quantization."""
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("Turing GELU ConvRot requires an sm75-or-newer CUDA tensor")
    if x.dtype not in (torch.float16, torch.bfloat16) or x.ndim != 2:
        raise TypeError("GELU ConvRot input must be 2D float16 or bfloat16")
    return _C.turing_gelu_int8_convrot_quantize(x.contiguous(), group_size)


def turing_gelu_int4_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse tanh-GELU into staged ConvRot INT4 activation quantization."""
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("Turing GELU INT4 ConvRot requires an sm75-or-newer CUDA tensor")
    if x.dtype not in (torch.float16, torch.bfloat16) or x.ndim != 2:
        raise TypeError("GELU INT4 ConvRot input must be 2D float16 or bfloat16")
    return _C.turing_gelu_int4_convrot_quantize(x.contiguous(), group_size)


def turing_bf16_int8_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
    *,
    swiglu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-row BF16 ConvRot quantization under the SM75 48 KiB limit."""
    if x.device.type != "cuda":
        raise RuntimeError("BF16 row-buffer ConvRot requires CUDA tensors")
    if torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("BF16 row-buffer ConvRot requires sm75 or newer")
    if x.dtype != torch.bfloat16:
        raise TypeError("BF16 row-buffer ConvRot input must be bfloat16")
    if x.ndim != 2:
        raise ValueError("BF16 row-buffer ConvRot input must be 2D")
    return _C.turing_bf16_int8_convrot_quantize(x.contiguous(), group_size, swiglu)


def turing_bf16_int4_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
    *,
    swiglu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-row BF16 ConvRot INT4 quantization under the SM75 48 KiB limit."""
    if x.device.type != "cuda":
        raise RuntimeError("BF16 row-buffer INT4 ConvRot requires CUDA tensors")
    if torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("BF16 row-buffer INT4 ConvRot requires sm75 or newer")
    if x.dtype != torch.bfloat16:
        raise TypeError("BF16 row-buffer INT4 ConvRot input must be bfloat16")
    if x.ndim != 2:
        raise ValueError("BF16 row-buffer INT4 ConvRot input must be 2D")
    return _C.turing_bf16_int4_convrot_quantize(x.contiguous(), group_size, swiglu)


def turing_bf16_gelu_int8_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-row BF16 tanh-GELU+ConvRot INT8 quantization."""
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("BF16 GELU ConvRot requires an sm75-or-newer CUDA tensor")
    if x.dtype != torch.bfloat16 or x.ndim != 2:
        raise TypeError("BF16 GELU ConvRot input must be a 2D bfloat16 tensor")
    return _C.turing_bf16_gelu_int8_convrot_quantize(x.contiguous(), group_size)


def turing_bf16_gelu_int4_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-row BF16 tanh-GELU+ConvRot INT4 quantization."""
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("BF16 GELU INT4 ConvRot requires an sm75-or-newer CUDA tensor")
    if x.dtype != torch.bfloat16 or x.ndim != 2:
        raise TypeError("BF16 GELU INT4 ConvRot input must be a 2D bfloat16 tensor")
    return _C.turing_bf16_gelu_int4_convrot_quantize(x.contiguous(), group_size)


def turing_segmented_rms_adaln(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    segments: torch.Tensor,
    epsilon: float = 1.0e-5,
) -> torch.Tensor:
    """Fused affine RMSNorm and segmented AdaLN modulation."""
    if x.device.type != "cuda":
        raise RuntimeError("Segmented RMSNorm+AdaLN requires CUDA tensors")
    if torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("Segmented RMSNorm+AdaLN requires sm75 or newer")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("segmented RMSNorm+AdaLN input must be float16, bfloat16, or float32")
    if x.ndim != 2:
        raise ValueError("segmented RMSNorm+AdaLN input must be 2D [M, K]")
    if weight.device != x.device or scale.device != x.device or shift.device != x.device:
        raise ValueError("weight, scale, and shift must be on the input device")
    weight = weight.to(dtype=x.dtype).contiguous()
    scale = scale.to(dtype=x.dtype)
    shift = shift.to(dtype=x.dtype)
    if scale.stride(-1) != 1:
        scale = scale.contiguous()
    if shift.stride(-1) != 1:
        shift = shift.contiguous()
    if segments.device != x.device or segments.dtype != torch.int32:
        raise ValueError("segments must be an int32 tensor on the input device")
    return _C.turing_segmented_rms_adaln(
        x.contiguous(),
        weight,
        scale,
        shift,
        segments.contiguous(),
        epsilon,
    )


def turing_layer_norm_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    epsilon: float = 1.0e-5,
) -> torch.Tensor:
    """Fused FP32-reduction LayerNorm and Wan AdaLN modulation."""
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("LayerNorm+AdaLN requires an sm75-or-newer CUDA tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32) or x.ndim != 3:
        raise TypeError("LayerNorm+AdaLN input must be 3D float16, bfloat16, or float32")
    scale = scale.to(device=x.device, dtype=x.dtype).contiguous()
    shift = shift.to(device=x.device, dtype=x.dtype).contiguous()
    return _C.turing_layer_norm_adaln(x.contiguous(), scale, shift, epsilon)
