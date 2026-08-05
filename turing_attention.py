"""ComfyUI adapter for the bundled Turing SageAttention2 backend."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

try:
    from .turing_ops import is_supported_turing_device
except ImportError:
    from turing_ops import is_supported_turing_device


LOG = logging.getLogger("comfyui-svdint4")


SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
SUPPORTED_INPUT_DTYPES = (*SUPPORTED_DTYPES, torch.float32)
_PREFLIGHTED_DEVICES: set[int] = set()
_LOGGED_FP32_COMPAT = False


def available() -> bool:
    try:
        from svdint4.turing_sage2 import available as kernel_available
    except (ImportError, OSError):
        return False
    return kernel_available()


def _sageattn(*args, **kwargs):
    from svdint4.turing_sage2 import sageattn

    return sageattn(*args, **kwargs)


def preflight(device: torch.device) -> None:
    if not is_supported_turing_device(device):
        raise RuntimeError(f"unsupported Turing device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_DEVICES:
        return
    from svdint4.turing_sage2 import preflight as kernel_preflight

    kernel_preflight(device)
    _PREFLIGHTED_DEVICES.add(index)


def _reshape_qkv(q, k, v, heads: int, enable_gqa: bool):
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("unreshaped Q/K/V must be three-dimensional")
    batch = q.shape[0]
    if heads <= 0 or q.shape[-1] % heads != 0:
        raise ValueError("Q inner dimension must be divisible by the head count")
    head_dim = q.shape[-1] // heads
    kv_heads = k.shape[-1] // head_dim if enable_gqa else heads
    if kv_heads <= 0 or k.shape[-1] != kv_heads * head_dim or v.shape[-1] != kv_heads * head_dim:
        raise ValueError("K/V inner dimensions do not match the Q head dimension")
    q = q.reshape(batch, -1, heads, head_dim)
    k = k.reshape(batch, -1, kv_heads, head_dim)
    v = v.reshape(batch, -1, kv_heads, head_dim)
    return q, k, v, batch, head_dim


def attention(
    original: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask=None,
    attn_precision=None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    **kwargs,
) -> torch.Tensor:
    global _LOGGED_FP32_COMPAT

    fallback_args = (q, k, v, heads)
    fallback_kwargs = {
        "mask": mask,
        "attn_precision": attn_precision,
        "skip_reshape": skip_reshape,
        "skip_output_reshape": skip_output_reshape,
        **kwargs,
    }
    if (
        not is_supported_turing_device(q.device)
        or mask is not None
        or kwargs.get("low_precision_attention", True) is False
    ):
        return original(*fallback_args, **fallback_kwargs)
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise RuntimeError(f"Turing SageAttention2 requires matching Q/K/V dtypes, got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.dtype not in SUPPORTED_INPUT_DTYPES:
        raise RuntimeError(f"Turing SageAttention2 supports FP16, BF16, or FP32 Q/K/V, got {q.dtype}")
    if q.device != k.device or q.device != v.device:
        raise RuntimeError("Turing SageAttention2 requires Q/K/V on the same CUDA device")

    input_dtype = q.dtype

    enable_gqa = bool(kwargs.get("enable_gqa", False))
    if skip_reshape:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[1] != heads:
            return original(*fallback_args, **fallback_kwargs)
        batch, _, _, head_dim = q.shape
        tensor_layout = "HND"
    else:
        try:
            q, k, v, batch, head_dim = _reshape_qkv(q, k, v, heads, enable_gqa)
        except ValueError:
            return original(*fallback_args, **fallback_kwargs)
        tensor_layout = "NHD"

    if head_dim <= 0 or head_dim > 128:
        return original(*fallback_args, **fallback_kwargs)

    if input_dtype == torch.float32:
        if not _LOGGED_FP32_COMPAT:
            LOG.info(
                "SVDInt4 Turing SageAttention2 FP32 compatibility: casting Q/K/V "
                "to BF16 for the attention kernel and restoring FP32 output"
            )
            _LOGGED_FP32_COMPAT = True
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

    output = _sageattn(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=bool(kwargs.get("is_causal", False)),
        sm_scale=kwargs.get("scale"),
        smooth_k=False,
    )
    if tensor_layout == "HND":
        if skip_output_reshape:
            result = output
        else:
            result = output.transpose(1, 2).reshape(batch, -1, heads * head_dim)
    elif skip_output_reshape:
        result = output.transpose(1, 2)
    else:
        result = output.reshape(batch, -1, heads * head_dim)
    if result.dtype != input_dtype:
        result = result.to(input_dtype)
    return result
