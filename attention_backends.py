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
        option="default",
        attention_function=None,
        label="default",
        aliases=("auto", "none", "comfyui_default"),
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


def attention_backend_choices() -> tuple[str, ...]:
    return tuple(_BACKENDS)


def normalize_attention_backend(value: str | None) -> str:
    if value is None:
        return "default"
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


def make_attention_override(option: str) -> Callable | None:
    backend = _BACKENDS[normalize_attention_backend(option)]
    target = _resolve_attention_function(backend)
    if target is None:
        return None

    def attention_override(original: Callable, *args, **kwargs):
        return target(*args, **kwargs)

    attention_override.svdint4_attention_backend = backend.option
    return attention_override


def apply_attention_backend(model, option: str):
    option = normalize_attention_backend(option)
    transformer_options = model.model_options.setdefault("transformer_options", {})

    if option == "default":
        transformer_options.pop("optimized_attention_override", None)
        transformer_options.pop("svdint4_attention_backend", None)
        return model

    transformer_options["optimized_attention_override"] = make_attention_override(option)
    transformer_options["svdint4_attention_backend"] = option
    LOG.info("SVDInt4 attention backend override: %s", option)
    return model
