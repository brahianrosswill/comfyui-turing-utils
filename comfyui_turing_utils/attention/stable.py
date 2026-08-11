"""Attention backend selection, bundled Sage, and sparse patch APIs."""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Callable

import torch

from .layout import (
    ATTENTION_LAYOUT_KEY,
    ATTENTION_LAYOUT_REQUIREMENT_KEY,
    ensure_attention_layout_provider,
    has_complete_attention_layout,
)
from ..kernel_api import kernel_version, load_turing_sage
from ..quantization.dispatch import is_supported_turing_device


LOG = logging.getLogger("comfyui-turing-utils")
SUPPORTED_KERNEL_DTYPES = (torch.float16, torch.bfloat16)
SUPPORTED_INPUT_DTYPES = (*SUPPORTED_KERNEL_DTYPES, torch.float32)
SPARSE_AUTO_MIN_SEQUENCE = 4096
SPARSE_ROUTING_THRESHOLD = 1.0
SPARSE_PREFIX_POLICY = "auto"
SPARSE_SKIPPED_RESIDUAL = "1x64"
SPARSE_REFERENCE_IMAGE = False
SPARSE_REFERENCE_VIDEO = True
SPARSE_REFERENCE_AUDIO = False
SPARSE_DENSE_PREFIX_STEPS = 0
SPARSE_DENSE_SUFFIX_STEPS = 0
SPARSE_DENSE_PREFIX_LAYERS = 2
SPARSE_DENSE_SUFFIX_LAYERS = 0
FRAME_SPARSE_TEMPORAL_WINDOW_FRAMES = 2
FRAME_SPARSE_GLOBAL_ANCHOR_STRIDE = 12
FRAME_SPARSE_SINK_FRAMES = 1
FRAME_SPARSE_PATTERN = "frame_window"
FRAME_SPARSE_QUALITY_PROFILE = "custom"
FRAME_SPARSE_RADIAL_SPATIAL_RADIUS = 1
FRAME_SPARSE_RADIAL_MAX_TEMPORAL_STRIDE = 16
SPARSE_LAYOUT_KEY = ATTENTION_LAYOUT_KEY
_PREFLIGHTED_DEVICES: set[int] = set()
_PREFLIGHTED_SPARSE_DEVICES: set[int] = set()
_PREFLIGHTED_FRAME_SPARSE_DEVICES: set[int] = set()
_LOGGED_FP32_COMPAT = False
_LOGGED_TURING_KERNELS: set[tuple] = set()
_LOGGED_TURING_FALLBACKS: set[str] = set()
_LOGGED_SPARSE_KERNELS: set[tuple] = set()
_LOGGED_FRAME_SPARSE_KERNELS: set[tuple] = set()
_LOGGED_SPARSE_DENSE_REASONS: set[str] = set()


@dataclasses.dataclass(frozen=True)
class AttentionBackend:
    option: str
    attention_function: str | None
    label: str
    install_hint: str | None = None
    aliases: tuple[str, ...] = ()


_BACKENDS: dict[str, AttentionBackend] = {}
_ALIASES: dict[str, str] = {}


def _normalize_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def register_attention_backend(backend: AttentionBackend) -> None:
    if backend.option in _BACKENDS:
        raise ValueError(f"duplicate attention backend option: {backend.option}")
    normalized_aliases = {
        _normalize_key(alias) for alias in (backend.option, backend.label, *backend.aliases)
    }
    collisions = {
        alias: _ALIASES[alias] for alias in normalized_aliases if alias in _ALIASES
    }
    if collisions:
        details = ", ".join(
            f"{alias!r} is already owned by {option!r}"
            for alias, option in sorted(collisions.items())
        )
        raise ValueError(f"attention backend alias collision: {details}")
    _BACKENDS[backend.option] = backend
    for alias in normalized_aliases:
        _ALIASES[alias] = backend.option


register_attention_backend(
    AttentionBackend(
        option="auto",
        attention_function=None,
        label="auto",
        aliases=("default", "none", "comfyui_default"),
    )
)
register_attention_backend(
    AttentionBackend(
        option="sage_attn",
        attention_function="sage",
        label="sage attn",
        install_hint="Install sageattention in the ComfyUI Python environment.",
        aliases=(
            "sage",
            "sage_",
            "sageattention",
            "sage_attention",
            "sage_hybrid",
            "turing_sage",
            "turing_sage_hybrid",
        ),
    )
)
register_attention_backend(
    AttentionBackend(
        option="flash_attn",
        attention_function="flash",
        label="flash attn",
        install_hint="Install flash-attn in the ComfyUI Python environment.",
        aliases=("flash", "flash_attention", "flashattention"),
    )
)
register_attention_backend(
    AttentionBackend(
        option="sdpa",
        attention_function="pytorch",
        label="sdpa",
        aliases=("pytorch", "torch", "torch_sdpa"),
    )
)

AUTO_BACKEND_PRIORITY = ("sage_attn", "flash_attn", "sdpa")


def attention_backend_choices() -> tuple[str, ...]:
    return tuple(_BACKENDS)


def normalize_attention_backend(value: str | None) -> str:
    if value is None:
        return "auto"
    option = _ALIASES.get(_normalize_key(value))
    if option is None:
        raise ValueError(
            f"Unsupported Turing Utils attention backend {value!r}. "
            f"Supported options: {', '.join(attention_backend_choices())}"
        )
    return option


def _comfy_attention_function(name: str) -> Callable | None:
    from comfy.ldm.modules import attention as comfy_attention

    return comfy_attention.get_attention_function(name, None)


def _resolve_attention_function(backend: AttentionBackend) -> Callable | None:
    if backend.attention_function is None:
        return None
    function = _comfy_attention_function(backend.attention_function)
    if function is not None:
        return function
    message = (
        f"Turing Utils attention backend {backend.option!r} requires ComfyUI attention "
        f"function {backend.attention_function!r}, but it is not available."
    )
    if backend.install_hint:
        message = f"{message} {backend.install_hint}"
    raise RuntimeError(message)


def _select_attention_backend(option: str) -> tuple[AttentionBackend, Callable]:
    option = normalize_attention_backend(option)
    if option != "auto":
        backend = _BACKENDS[option]
        return backend, _resolve_attention_function(backend)

    for candidate in AUTO_BACKEND_PRIORITY:
        backend = _BACKENDS[candidate]
        function = _comfy_attention_function(backend.attention_function)
        if function is not None:
            return backend, function
    raise RuntimeError("No supported attention backend is available")


def bundled_available() -> bool:
    try:
        turing_sage = load_turing_sage()
    except (ImportError, OSError):
        return False
    return bool(turing_sage.available())


def bundled_sparse_available() -> bool:
    try:
        turing_sage = load_turing_sage()
    except (ImportError, OSError):
        return False
    version = kernel_version()
    try:
        version_tuple = tuple(int(part) for part in version.split(".")[:3])
    except ValueError:
        return False
    return version_tuple >= (0, 17, 0) and turing_sage.sparse_available()


def bundled_frame_sparse_available() -> bool:
    try:
        turing_sage = load_turing_sage()
    except (ImportError, OSError):
        return False
    version = kernel_version()
    try:
        version_tuple = tuple(int(part) for part in version.split(".")[:3])
    except ValueError:
        return False
    return version_tuple >= (0, 15, 0) and turing_sage.frame_sparse_available()


def _sageattn(*args, **kwargs):
    return load_turing_sage().sageattn(*args, **kwargs)


def _sol_sparse_sageattn(*args, **kwargs):
    return load_turing_sage().sol_sparse_sageattn(*args, **kwargs)


def _frame_sparse_sageattn(*args, **kwargs):
    return load_turing_sage().frame_sparse_sageattn(*args, **kwargs)


def preflight_bundled(device: torch.device) -> None:
    if not is_supported_turing_device(device):
        raise RuntimeError(f"unsupported Turing device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_DEVICES:
        return
    load_turing_sage().preflight(device)
    _PREFLIGHTED_DEVICES.add(index)


def preflight_bundled_sparse(device: torch.device) -> None:
    if not is_supported_turing_device(device):
        raise RuntimeError(f"unsupported Turing device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_SPARSE_DEVICES:
        return
    load_turing_sage().preflight_sparse(device)
    _PREFLIGHTED_SPARSE_DEVICES.add(index)


def preflight_bundled_frame_sparse(device: torch.device) -> None:
    if not is_supported_turing_device(device):
        raise RuntimeError(f"unsupported Turing device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_FRAME_SPARSE_DEVICES:
        return
    load_turing_sage().preflight_frame_sparse(device)
    _PREFLIGHTED_FRAME_SPARSE_DEVICES.add(index)


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


def _bundled_fallback(
    fallback: Callable,
    reason: str,
    fallback_args: tuple,
    fallback_kwargs: dict,
):
    if reason not in _LOGGED_TURING_FALLBACKS:
        LOG.warning(
            "Bundled Turing Sage is falling back to ComfyUI attention (%s); "
            "this message is emitted once per reason",
            reason,
        )
        _LOGGED_TURING_FALLBACKS.add(reason)
    return fallback(*fallback_args, **fallback_kwargs)


def turing_sage_attention(
    fallback: Callable,
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
    if not is_supported_turing_device(q.device):
        return _bundled_fallback(
            fallback,
            "Q/K/V are not on a supported sm75 GPU",
            fallback_args,
            fallback_kwargs,
        )
    if mask is not None:
        return _bundled_fallback(
            fallback,
            "an attention mask was supplied",
            fallback_args,
            fallback_kwargs,
        )
    if kwargs.get("low_precision_attention", True) is False:
        return _bundled_fallback(
            fallback,
            "low_precision_attention=False",
            fallback_args,
            fallback_kwargs,
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise RuntimeError(
            f"Turing Sage requires matching Q/K/V dtypes, got {q.dtype}, {k.dtype}, {v.dtype}"
        )
    if q.dtype not in SUPPORTED_INPUT_DTYPES:
        raise RuntimeError(f"Turing Sage supports FP16, BF16, or FP32 Q/K/V, got {q.dtype}")
    if q.device != k.device or q.device != v.device:
        raise RuntimeError("Turing Sage requires Q/K/V on the same CUDA device")

    input_dtype = q.dtype
    enable_gqa = bool(kwargs.get("enable_gqa", False))
    if skip_reshape:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[1] != heads:
            return _bundled_fallback(
                fallback,
                "skip_reshape Q/K/V layout is incompatible",
                fallback_args,
                fallback_kwargs,
            )
        batch, _, _, head_dim = q.shape
        tensor_layout = "HND"
    else:
        try:
            q, k, v, batch, head_dim = _reshape_qkv(q, k, v, heads, enable_gqa)
        except ValueError:
            return _bundled_fallback(
                fallback,
                "unreshaped Q/K/V layout is incompatible",
                fallback_args,
                fallback_kwargs,
            )
        tensor_layout = "NHD"

    if head_dim <= 0 or head_dim > 128:
        return _bundled_fallback(
            fallback,
            f"head_dim={head_dim} is outside the supported range",
            fallback_args,
            fallback_kwargs,
        )

    index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    sequence_axis = 2 if tensor_layout == "HND" else 1
    head_axis = 1 if tensor_layout == "HND" else 2
    kernel_key = (
        index,
        input_dtype,
        tensor_layout,
        head_dim,
        q.shape[sequence_axis],
        k.shape[sequence_axis],
        q.shape[head_axis],
        k.shape[head_axis],
    )
    if kernel_key not in _LOGGED_TURING_KERNELS:
        LOG.info(
            "Bundled Turing Sage active: device=%s dtype=%s layout=%s "
            "Q=%s K=%s V=%s heads=%d",
            q.device,
            input_dtype,
            tensor_layout,
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            heads,
        )
        _LOGGED_TURING_KERNELS.add(kernel_key)

    if input_dtype == torch.float32:
        if not _LOGGED_FP32_COMPAT:
            LOG.info(
                "Turing Sage FP32 compatibility uses BF16 Q/K/V storage and restores FP32 output"
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
        result = output if skip_output_reshape else output.transpose(1, 2).reshape(batch, -1, heads * head_dim)
    elif skip_output_reshape:
        result = output.transpose(1, 2)
    else:
        result = output.reshape(batch, -1, heads * head_dim)
    return result.to(input_dtype) if input_dtype == torch.float32 else result
