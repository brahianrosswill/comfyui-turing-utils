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


LOG = logging.getLogger("comfyui-turing-utils")
SUPPORTED_KERNEL_DTYPES = (torch.float16, torch.bfloat16)
SUPPORTED_INPUT_DTYPES = (*SUPPORTED_KERNEL_DTYPES, torch.float32)
SPARSE_MIN_SEQUENCE = 4096
SPARSE_PREFIX_TOKENS = 512
SPARSE_ROUTE_TAU = 1.0
_PREFLIGHTED_DEVICES: set[int] = set()
_PREFLIGHTED_SPARSE_DEVICES: set[int] = set()
_LOGGED_FP32_COMPAT = False
_LOGGED_TURING_KERNELS: set[tuple] = set()
_LOGGED_TURING_FALLBACKS: set[str] = set()
_LOGGED_SPARSE_KERNELS: set[tuple] = set()
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
        from comfyui_turing_utils_kernel.turing_sage import available
    except (ImportError, OSError):
        return False
    return available()


def bundled_sparse_available() -> bool:
    try:
        from comfyui_turing_utils_kernel.turing_sage import sparse_available
    except (ImportError, OSError):
        return False
    return sparse_available()


def _sageattn(*args, **kwargs):
    from comfyui_turing_utils_kernel import turing_sage

    return turing_sage.sageattn(*args, **kwargs)


def _sol_sparse_sageattn(*args, **kwargs):
    from comfyui_turing_utils_kernel import turing_sage

    return turing_sage.sol_sparse_sageattn(*args, **kwargs)


def preflight_bundled(device: torch.device) -> None:
    if not is_supported_turing_device(device):
        raise RuntimeError(f"unsupported Turing device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_DEVICES:
        return
    from comfyui_turing_utils_kernel.turing_sage import preflight

    preflight(device)
    _PREFLIGHTED_DEVICES.add(index)


def preflight_bundled_sparse(device: torch.device) -> None:
    if not is_supported_turing_device(device):
        raise RuntimeError(f"unsupported Turing device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_SPARSE_DEVICES:
        return
    from comfyui_turing_utils_kernel.turing_sage import preflight_sparse

    preflight_sparse(device)
    _PREFLIGHTED_SPARSE_DEVICES.add(index)


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


def _sparse_dense_baseline(
    reason: str,
    fallback: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    **kwargs,
) -> torch.Tensor:
    if reason not in _LOGGED_SPARSE_DENSE_REASONS:
        LOG.info("Experimental sparse attention uses stable Sage for %s", reason)
        _LOGGED_SPARSE_DENSE_REASONS.add(reason)
    return turing_sage_attention(fallback, q, k, v, heads, **kwargs)


def turing_sol_sparse_attention(
    fallback: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask=None,
    attn_precision=None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    min_sequence_tokens: int = SPARSE_MIN_SEQUENCE,
    dense_prefix_tokens: int = SPARSE_PREFIX_TOKENS,
    route_threshold: float = SPARSE_ROUTE_TAU,
    **kwargs,
) -> torch.Tensor:
    original_q, original_k, original_v = q, k, v
    common = {
        "mask": mask,
        "attn_precision": attn_precision,
        "skip_reshape": skip_reshape,
        "skip_output_reshape": skip_output_reshape,
        **kwargs,
    }

    def dense(reason: str):
        return _sparse_dense_baseline(
            reason,
            fallback,
            original_q,
            original_k,
            original_v,
            heads,
            **common,
        )

    if not is_supported_turing_device(q.device):
        return dense("Q/K/V are not on a supported sm75 GPU")
    if mask is not None:
        return dense("an attention mask was supplied")
    if kwargs.get("low_precision_attention", True) is False:
        return dense("low_precision_attention=False")
    if bool(kwargs.get("is_causal", False)):
        return dense("causal attention")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in SUPPORTED_INPUT_DTYPES:
        return dense("Q/K/V dtypes are incompatible")
    if q.device != k.device or q.device != v.device:
        return dense("Q/K/V devices are incompatible")

    input_dtype = q.dtype
    enable_gqa = bool(kwargs.get("enable_gqa", False))
    if skip_reshape:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[1] != heads:
            return dense("skip_reshape Q/K/V layout is incompatible")
        batch, _, _, head_dim = q.shape
    else:
        try:
            q, k, v, batch, head_dim = _reshape_qkv(q, k, v, heads, enable_gqa)
        except ValueError:
            return dense("unreshaped Q/K/V layout is incompatible")
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    if head_dim != 128:
        return dense(f"head_dim={head_dim} is not 128")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        return dense("Q/K/V batch sizes are incompatible")
    if k.shape[1] != v.shape[1] or k.shape[2:] != v.shape[2:]:
        return dense("K/V shapes are incompatible")
    if k.shape[-1] != 128 or k.shape[1] <= 0 or heads % k.shape[1] != 0:
        return dense("Q/K/V head counts are incompatible")
    if q.shape[2] < min_sequence_tokens or k.shape[2] < min_sequence_tokens:
        return dense(f"sequences shorter than {min_sequence_tokens} tokens")

    prefix_tokens = min(dense_prefix_tokens, k.shape[2])
    if input_dtype == torch.float32:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    kernel_key = (
        q.device.index,
        input_dtype,
        tuple(q.shape),
        tuple(k.shape),
        min_sequence_tokens,
        prefix_tokens,
        route_threshold,
    )
    if kernel_key not in _LOGGED_SPARSE_KERNELS:
        LOG.info(
            "Experimental Turing Sol sparse attention active: dtype=%s Q=%s K=%s "
            "min_sequence=%d dense_prefix=%d route_threshold=%.2f",
            input_dtype,
            tuple(q.shape),
            tuple(k.shape),
            min_sequence_tokens,
            prefix_tokens,
            route_threshold,
        )
        _LOGGED_SPARSE_KERNELS.add(kernel_key)

    output = _sol_sparse_sageattn(
        q,
        k,
        v,
        tensor_layout="HND",
        sm_scale=kwargs.get("scale"),
        prefix_tokens=prefix_tokens,
        tau=route_threshold,
    )
    batch, _, _, head_dim = q.shape
    result = output if skip_output_reshape else output.transpose(1, 2).reshape(
        batch, -1, heads * head_dim
    )
    return result.to(input_dtype) if input_dtype == torch.float32 else result


def _uses_bundled_turing_sage(option: str, device: torch.device | None) -> bool:
    option = normalize_attention_backend(option)
    return bool(
        device is not None
        and is_supported_turing_device(device)
        and option in {"auto", "sage_attn"}
    )


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
    bundled_turing = _uses_bundled_turing_sage(option, device)
    if bundled_turing:
        if not bundled_available():
            raise RuntimeError(
                "The bundled Turing Sage extensions are unavailable. "
                "Rebuild comfyui-turing-utils-kernel with COMFYUI_TURING_UTILS_ARCH_LIST including 7.5."
            )
        preflight_bundled(device)
        backend = _BACKENDS["sage_attn"]
        target = turing_sage_attention
        implementation = "bundled_turing_sage"
    else:
        backend, target = _select_attention_backend(option)
        implementation = f"comfy:{backend.attention_function}"

    def attention_override(original: Callable, *args, **kwargs):
        fallback = lambda *fallback_args, **fallback_kwargs: _dtype_compatible_fallback(
            original, *fallback_args, **fallback_kwargs
        )
        if bundled_turing:
            return target(fallback, *args, **kwargs)
        if (
            backend.option == "sage_attn"
            and len(args) >= 3
            and all(isinstance(value, torch.Tensor) for value in args[:3])
            and any(value.dtype == torch.float32 for value in args[:3])
        ):
            return fallback(*args, **kwargs)
        return target(*args, **kwargs)

    attention_override.turing_utils_attention_backend = backend.option
    attention_override.turing_utils_attention_implementation = implementation
    return attention_override


def make_sparse_attention_override(
    device: torch.device,
    min_sequence_tokens: int = SPARSE_MIN_SEQUENCE,
    dense_prefix_tokens: int = SPARSE_PREFIX_TOKENS,
    route_threshold: float = SPARSE_ROUTE_TAU,
) -> Callable:
    min_sequence_tokens = int(min_sequence_tokens)
    dense_prefix_tokens = int(dense_prefix_tokens)
    route_threshold = float(route_threshold)
    if min_sequence_tokens < 1:
        raise ValueError("min_sequence_tokens must be positive")
    if dense_prefix_tokens < 0:
        raise ValueError("dense_prefix_tokens must be non-negative")
    if route_threshold < 0:
        raise ValueError("route_threshold must be non-negative")
    if not is_supported_turing_device(device):
        raise RuntimeError("Sol sparse attention requires an sm75 Turing GPU")
    if not bundled_sparse_available():
        raise RuntimeError(
            "The experimental Turing sparse extension is unavailable. "
            "Rebuild comfyui-turing-utils-kernel 0.9.0 or newer with sm75 enabled."
        )
    preflight_bundled(device)
    preflight_bundled_sparse(device)

    def attention_override(original: Callable, *args, **kwargs):
        fallback = lambda *fallback_args, **fallback_kwargs: _dtype_compatible_fallback(
            original, *fallback_args, **fallback_kwargs
        )
        return turing_sol_sparse_attention(
            fallback,
            *args,
            min_sequence_tokens=min_sequence_tokens,
            dense_prefix_tokens=dense_prefix_tokens,
            route_threshold=route_threshold,
            **kwargs,
        )

    attention_override.turing_utils_attention_backend = "sol_sparse_attn"
    attention_override.turing_utils_attention_implementation = "bundled_turing_sol_sparse_experimental"
    return attention_override


def apply_sparse_attention_patch(
    model,
    min_sequence_tokens: int = SPARSE_MIN_SEQUENCE,
    dense_prefix_tokens: int = SPARSE_PREFIX_TOKENS,
    route_threshold: float = SPARSE_ROUTE_TAU,
):
    patched = model.clone()
    override = make_sparse_attention_override(
        patched.load_device,
        min_sequence_tokens=min_sequence_tokens,
        dense_prefix_tokens=dense_prefix_tokens,
        route_threshold=route_threshold,
    )
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    transformer_options["optimized_attention_override"] = override
    transformer_options["turing_utils_attention_backend"] = "sol_sparse_attn"
    transformer_options["turing_utils_attention_implementation"] = (
        "bundled_turing_sol_sparse_experimental"
    )
    LOG.info(
        "Sol sparse attention patch enabled: min_sequence=%d dense_prefix=%d "
        "route_threshold=%.2f",
        min_sequence_tokens,
        dense_prefix_tokens,
        route_threshold,
    )
    return patched


def apply_attention_backend(model, option: str, device: torch.device | None = None):
    option = normalize_attention_backend(option)
    transformer_options = model.model_options.setdefault("transformer_options", {})
    override = make_attention_override(option, device=device)
    selected = override.turing_utils_attention_backend
    implementation = override.turing_utils_attention_implementation
    transformer_options["optimized_attention_override"] = override
    transformer_options["turing_utils_attention_backend"] = selected
    transformer_options["turing_utils_attention_implementation"] = implementation
    LOG.info(
        "Turing Utils attention backend override: %s via %s (requested %s)",
        selected,
        implementation,
        option,
    )
    return model
