from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable


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


def make_attention_override(option: str) -> Callable | None:
    backend, target = _select_attention_backend(option)

    def attention_override(original: Callable, *args, **kwargs):
        return target(*args, **kwargs)

    attention_override.svdint4_attention_backend = backend.option
    return attention_override


def apply_attention_backend(model, option: str):
    option = normalize_attention_backend(option)
    transformer_options = model.model_options.setdefault("transformer_options", {})
    override = make_attention_override(option)
    selected = override.svdint4_attention_backend
    transformer_options["optimized_attention_override"] = override
    transformer_options["svdint4_attention_backend"] = selected
    LOG.info("SVDInt4 attention backend override: %s (requested %s)", selected, option)
    return model
