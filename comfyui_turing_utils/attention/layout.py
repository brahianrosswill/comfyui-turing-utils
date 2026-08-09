"""Model-independent attention-layout contract and provider registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


ATTENTION_LAYOUT_KEY = "turing_utils_attention_layout"
ATTENTION_LAYOUT_REQUIREMENT_KEY = "turing_utils_attention_layout_required"


@dataclass(frozen=True)
class LayoutProviderStatus:
    """Result returned by a model adapter asked to publish attention layout."""

    model_kind: str | None
    installed: bool
    reason: str | None = None

    @property
    def required(self) -> bool:
        return self.model_kind is not None


LayoutProviderInstaller = Callable[[object], LayoutProviderStatus]
_PROVIDERS: list[LayoutProviderInstaller] = []


def register_attention_layout_provider(installer: LayoutProviderInstaller) -> None:
    """Register a loader-independent model adapter exactly once."""
    if installer not in _PROVIDERS:
        _PROVIDERS.append(installer)


def ensure_attention_layout_provider(model) -> LayoutProviderStatus:
    """Install the first adapter that recognizes ``model``.

    Providers return ``model_kind=None`` when they do not recognize a model.
    This keeps sparse attention independent of concrete diffusion-model types.
    """
    for installer in tuple(_PROVIDERS):
        status = installer(model)
        if status.required:
            return status
    return LayoutProviderStatus(None, False, None)


def has_complete_attention_layout(
    transformer_options,
    sequence_length: int | None = None,
    *,
    provider: str | None = None,
) -> bool:
    """Validate the common packed-video topology consumed by sparse backends."""
    if not isinstance(transformer_options, dict):
        return False
    layout = transformer_options.get(ATTENTION_LAYOUT_KEY)
    if not isinstance(layout, dict):
        return False
    if provider is not None and layout.get("provider") != provider:
        return False
    names = (
        "dense_prefix_tokens",
        "topology_start_tokens",
        "topology_tokens",
        "tokens_per_frame",
        "spatial_tokens_height",
        "spatial_tokens_width",
        "layer_index",
        "layer_count",
    )
    values = {name: layout.get(name) for name in names}
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in values.values()
    ):
        return False
    prefix = values["dense_prefix_tokens"]
    start = values["topology_start_tokens"]
    tokens = values["topology_tokens"]
    frame_tokens = values["tokens_per_frame"]
    height = values["spatial_tokens_height"]
    width = values["spatial_tokens_width"]
    layer_index = values["layer_index"]
    layer_count = values["layer_count"]
    if (
        prefix < 0
        or start != prefix
        or tokens <= 0
        or frame_tokens <= 0
        or tokens % frame_tokens != 0
        or height <= 0
        or width <= 0
        or height * width != frame_tokens
        or layer_count <= 0
        or layer_index < 0
        or layer_index >= layer_count
    ):
        return False
    return sequence_length is None or start + tokens == int(sequence_length)


__all__ = [
    "ATTENTION_LAYOUT_KEY",
    "ATTENTION_LAYOUT_REQUIREMENT_KEY",
    "LayoutProviderStatus",
    "ensure_attention_layout_provider",
    "has_complete_attention_layout",
    "register_attention_layout_provider",
]
