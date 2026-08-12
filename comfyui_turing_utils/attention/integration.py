"""Model-neutral integration helpers for the prepared-attention protocol."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .protocol import (
    AttentionExecutionOutcome,
    PreparedAttention,
    QKTransformSpec,
    execute_prepared_attention,
)


@dataclass(frozen=True, slots=True)
class AttentionSiteStatus:
    model_kind: str | None
    installed: int = 0
    reason: str | None = None

    @property
    def matched(self) -> bool:
        return self.model_kind is not None


AttentionSiteInstaller = Callable[[object, torch.device], AttentionSiteStatus]
_SITE_INSTALLERS: list[AttentionSiteInstaller] = []


def register_attention_site_installer(installer: AttentionSiteInstaller) -> None:
    if installer not in _SITE_INSTALLERS:
        _SITE_INSTALLERS.append(installer)


def ensure_prepared_attention_sites(
    model,
    device: torch.device,
) -> AttentionSiteStatus:
    for installer in tuple(_SITE_INSTALLERS):
        status = installer(model, device)
        if status.matched:
            return status
    return AttentionSiteStatus(None, 0, None)


def execute_projected_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    heads: int,
    qk_transform: QKTransformSpec,
    transformer_options: dict | None,
    container_factory: Callable,
    scale: float | None = None,
    is_causal: bool = False,
    skip_output_reshape: bool = False,
    observer_requirements: frozenset[str] = frozenset(),
) -> AttentionExecutionOutcome:
    """Wrap projected HND tensors and execute them transactionally."""
    request = PreparedAttention.from_hnd(
        container_factory(query),
        container_factory(key),
        container_factory(value),
        heads=heads,
        qk_transform=qk_transform,
        transformer_options=transformer_options,
        scale=scale,
        is_causal=is_causal,
        skip_output_reshape=skip_output_reshape,
        observer_requirements=observer_requirements,
    )
    return execute_prepared_attention(request)


__all__ = [
    "AttentionSiteStatus",
    "ensure_prepared_attention_sites",
    "execute_projected_attention",
    "register_attention_site_installer",
]
