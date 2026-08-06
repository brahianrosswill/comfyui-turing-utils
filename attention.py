"""Attention backend selection and the self-contained Turing Sage backend."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable

import torch

try:
    from .turing_ops import is_supported_turing_device
except ImportError:
    from turing_ops import is_supported_turing_device


LOG = logging.getLogger("comfyui-svdint4")
SUPPORTED_KERNEL_DTYPES = (torch.float16, torch.bfloat16)
SUPPORTED_INPUT_DTYPES = (*SUPPORTED_KERNEL_DTYPES, torch.float32)
_PREFLIGHTED_DEVICES: set[tuple[int, str]] = set()
_LOGGED_FP32_COMPAT = False


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
    _BACKENDS[backend.option] = backend
    for alias in (backend.option, backend.label, *backend.aliases):
        _ALIASES[_normalize_key(alias)] = backend.option


register_attention_backend(
    AttentionBackend(
        option="sage",
        attention_function=None,
        label="sage",
        aliases=("sage_", "sage_hybrid", "turing_sage", "turing_sage_hybrid"),
    )
)
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
        aliases=("sageattention", "sage_attention"),
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
    return ("auto", "sage", "sage_attn", "flash_attn", "sdpa")


def normalize_attention_backend(value: str | None) -> str:
    if value is None:
        return "auto"
    option = _ALIASES.get(_normalize_key(value))
    if option is None:
        raise ValueError(
            f"Unsupported SVDInt4 attention backend {value!r}. "
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
        f"SVDInt4 attention backend {backend.option!r} requires ComfyUI attention "
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
        from svdint4.turing_sage import available
    except (ImportError, OSError):
        return False
    return available()


def _sageattn(*args, **kwargs):
    from svdint4 import turing_sage

    return turing_sage.sageattn(*args, **kwargs)


def preflight_bundled(device: torch.device, variant: str = "sage") -> None:
    if not is_supported_turing_device(device):
        raise RuntimeError(f"unsupported Turing device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if variant not in {"sage", "sage_"}:
        raise ValueError(f"unknown bundled Turing Sage backend: {variant}")
    key = (index, "sage")
    if key in _PREFLIGHTED_DEVICES:
        return
    from svdint4.turing_sage import preflight

    preflight(device)
    _PREFLIGHTED_DEVICES.add(key)


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
    variant: str = "sage",
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
        return fallback(*fallback_args, **fallback_kwargs)
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise RuntimeError(
            f"Turing {variant} requires matching Q/K/V dtypes, got {q.dtype}, {k.dtype}, {v.dtype}"
        )
    if q.dtype not in SUPPORTED_INPUT_DTYPES:
        raise RuntimeError(f"Turing {variant} supports FP16, BF16, or FP32 Q/K/V, got {q.dtype}")
    if q.device != k.device or q.device != v.device:
        raise RuntimeError(f"Turing {variant} requires Q/K/V on the same CUDA device")

    input_dtype = q.dtype
    enable_gqa = bool(kwargs.get("enable_gqa", False))
    if skip_reshape:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[1] != heads:
            return fallback(*fallback_args, **fallback_kwargs)
        batch, _, _, head_dim = q.shape
        tensor_layout = "HND"
    else:
        try:
            q, k, v, batch, head_dim = _reshape_qkv(q, k, v, heads, enable_gqa)
        except ValueError:
            return fallback(*fallback_args, **fallback_kwargs)
        tensor_layout = "NHD"

    if head_dim <= 0 or head_dim > 128:
        return fallback(*fallback_args, **fallback_kwargs)

    if input_dtype == torch.float32:
        if not _LOGGED_FP32_COMPAT:
            LOG.info(
                "SVDInt4 Turing Sage FP32 compatibility uses BF16 Q/K/V storage and restores FP32 output"
            )
            _LOGGED_FP32_COMPAT = True
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

    if variant not in {"sage", "sage_"}:
        raise ValueError(f"Unsupported bundled Turing Sage backend: {variant}")
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


def _bundled_turing_variant(option: str, device: torch.device | None) -> str | None:
    option = normalize_attention_backend(option)
    if device is None or not is_supported_turing_device(device):
        if option == "sage":
            raise RuntimeError(f"Attention backend {option!r} requires an NVIDIA sm75 Turing GPU")
        return None
    if option in {"auto", "sage_attn"}:
        return "sage"
    return option if option == "sage" else None


def _dtype_compatible_fallback(original: Callable, *args, **kwargs):
    qkv = args[:3]
    if (
        len(qkv) == 3
        and all(isinstance(value, torch.Tensor) for value in qkv)
        and all(value.dtype == torch.float32 for value in qkv)
    ):
        pytorch_attention = _comfy_attention_function("pytorch")
        if pytorch_attention is None:
            raise RuntimeError("ComfyUI PyTorch attention is unavailable for the FP32 fallback")
        return pytorch_attention(*args, **kwargs)
    return original(*args, **kwargs)


def make_attention_override(option: str, device: torch.device | None = None) -> Callable:
    option = normalize_attention_backend(option)
    bundled_variant = _bundled_turing_variant(option, device)
    if bundled_variant is not None:
        if not bundled_available():
            raise RuntimeError(
                "The bundled Turing Sage extensions are unavailable. "
                "Rebuild svdint4-kernel with SVDINT4_ARCH_LIST including 7.5."
            )
        preflight_bundled(device, bundled_variant)
        backend = AttentionBackend(bundled_variant, None, f"bundled Turing {bundled_variant}")
        target = turing_sage_attention
    else:
        backend, target = _select_attention_backend(option)

    def attention_override(original: Callable, *args, **kwargs):
        fallback = lambda *fallback_args, **fallback_kwargs: _dtype_compatible_fallback(
            original, *fallback_args, **fallback_kwargs
        )
        if bundled_variant is not None:
            return target(fallback, *args, variant=bundled_variant, **kwargs)
        if (
            backend.option == "sage_attn"
            and len(args) >= 3
            and all(isinstance(value, torch.Tensor) for value in args[:3])
            and any(value.dtype == torch.float32 for value in args[:3])
        ):
            return fallback(*args, **kwargs)
        return target(*args, **kwargs)

    attention_override.svdint4_attention_backend = backend.option
    return attention_override


def apply_attention_backend(model, option: str, device: torch.device | None = None):
    option = normalize_attention_backend(option)
    transformer_options = model.model_options.setdefault("transformer_options", {})
    override = make_attention_override(option, device=device)
    selected = override.svdint4_attention_backend
    transformer_options["optimized_attention_override"] = override
    transformer_options["svdint4_attention_backend"] = selected
    LOG.info("SVDInt4 attention backend override: %s (requested %s)", selected, option)
    return model
