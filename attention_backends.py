from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable

import torch


LOG = logging.getLogger("comfyui-svdint4")


@dataclasses.dataclass(frozen=True)
class AttentionBackend:
    option: str
    attention_function: str | None
    label: str
    install_hint: str | None = None
    aliases: tuple[str, ...] = ()


_BACKENDS: dict[str, AttentionBackend] = {}
_ALIASES: dict[str, str] = {}


def register_attention_backend(backend: AttentionBackend) -> None:
    if backend.option in _BACKENDS:
        raise ValueError(f"duplicate attention backend option: {backend.option}")
    _BACKENDS[backend.option] = backend
    for alias in (backend.option, backend.label, *backend.aliases):
        _ALIASES[_normalize_key(alias)] = backend.option


def _normalize_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


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
        aliases=("sage", "sageattention", "sage_attention"),
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
            f"Unsupported SVDInt4 attention backend {value!r}. "
            f"Supported options: {', '.join(attention_backend_choices())}"
        )
    return option


def _resolve_attention_function(backend: AttentionBackend) -> Callable | None:
    if backend.attention_function is None:
        return None

    from comfy.ldm.modules import attention

    func = attention.get_attention_function(backend.attention_function, None)
    if func is None:
        message = (
            f"SVDInt4 attention backend {backend.option!r} requires ComfyUI attention "
            f"function {backend.attention_function!r}, but it is not available."
        )
        if backend.install_hint:
            message = f"{message} {backend.install_hint}"
        raise RuntimeError(message)
    return func


def _select_attention_backend(option: str) -> tuple[AttentionBackend, Callable]:
    option = normalize_attention_backend(option)
    if option != "auto":
        backend = _BACKENDS[option]
        return backend, _resolve_attention_function(backend)

    from comfy.ldm.modules import attention

    for candidate in AUTO_BACKEND_PRIORITY:
        backend = _BACKENDS[candidate]
        func = attention.get_attention_function(backend.attention_function, None)
        if func is not None:
            return backend, func
    raise RuntimeError("No supported attention backend is available")


def _use_bundled_turing_sage(option: str, device: torch.device | None) -> bool:
    if device is None or normalize_attention_backend(option) not in {"auto", "sage_attn"}:
        return False
    try:
        from .turing_ops import is_supported_turing_device
    except ImportError:
        from turing_ops import is_supported_turing_device
    return is_supported_turing_device(device)


def make_attention_override(option: str, device: torch.device | None = None) -> Callable | None:
    option = normalize_attention_backend(option)
    bundled_turing = _use_bundled_turing_sage(option, device)
    if bundled_turing:
        try:
            from . import turing_attention
        except ImportError:
            import turing_attention
        if not turing_attention.available():
            raise RuntimeError(
                "The bundled Turing SageAttention2 extensions are unavailable. "
                "Rebuild svdint4-kernel with SVDINT4_ARCH_LIST including 7.5."
            )
        turing_attention.preflight(device)
        backend = AttentionBackend("turing_sage2", None, "bundled Turing SageAttention2")
        target = turing_attention.attention
    else:
        backend, target = _select_attention_backend(option)

    def attention_override(original: Callable, *args, **kwargs):
        if bundled_turing:
            return target(original, *args, **kwargs)
        if (
            backend.option == "sage_attn"
            and len(args) >= 3
            and all(isinstance(value, torch.Tensor) for value in args[:3])
            and any(value.dtype == torch.float32 for value in args[:3])
        ):
            return original(*args, **kwargs)
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
