"""Shared installation mechanics for sparse attention strategies."""

from __future__ import annotations

from dataclasses import dataclass

from .integration import AttentionSiteStatus, ensure_prepared_attention_sites
from .layout import (
    ATTENTION_LAYOUT_REQUIREMENT_KEY,
    LayoutProviderStatus,
    ensure_attention_layout_provider,
)
from .protocol import ATTENTION_EXECUTOR_KEY
from .stable import LOG


@dataclass(frozen=True, slots=True)
class SparsePatchInstallation:
    model: object
    layout: LayoutProviderStatus
    attention_sites: AttentionSiteStatus | None


def install_sparse_attention_override(
    model,
    override,
    *,
    strategy: str,
    backend: str,
    implementation: str,
) -> SparsePatchInstallation:
    """Clone and install one sparse strategy without owning its algorithm."""
    patched = model.clone()
    layout_status = ensure_attention_layout_provider(patched)
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    if layout_status.required:
        transformer_options[ATTENTION_LAYOUT_REQUIREMENT_KEY] = layout_status.model_kind
        if not layout_status.installed:
            LOG.warning(
                "%s %s attention will stay dense because its runtime layout "
                "provider could not be installed: %s",
                layout_status.model_kind,
                strategy,
                layout_status.reason,
            )

    transformer_options["optimized_attention_override"] = override
    prepared_executor = getattr(override, "prepared_attention_executor", None)
    site_status = None
    if callable(prepared_executor):
        transformer_options[ATTENTION_EXECUTOR_KEY] = prepared_executor
        site_status = ensure_prepared_attention_sites(patched, patched.load_device)
        if site_status.matched and site_status.reason is not None:
            LOG.info(
                "%s prepared-attention fusion was not installed: %s",
                site_status.model_kind,
                site_status.reason,
            )
    else:
        transformer_options.pop(ATTENTION_EXECUTOR_KEY, None)
    transformer_options["turing_utils_attention_backend"] = backend
    transformer_options["turing_utils_attention_implementation"] = implementation
    return SparsePatchInstallation(patched, layout_status, site_status)


__all__ = ["SparsePatchInstallation", "install_sparse_attention_override"]
