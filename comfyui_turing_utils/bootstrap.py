"""Composition root for built-in model integrations.

Importing a package submodule must not eagerly import every model adapter.
ComfyUI's plugin entry point calls :func:`bootstrap_builtin_integrations`
before node registration, while standalone tools can opt in explicitly.
"""

from __future__ import annotations

import threading


_BOOTSTRAP_LOCK = threading.Lock()
_BOOTSTRAPPED = False


def bootstrap_builtin_integrations() -> None:
    """Register the built-in adapters exactly once for this interpreter."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAPPED:
            return

        from .adapters.minimax.acceleration import apply_minimax_adapter
        from .adapters.minimax.acceleration import install_minimax_attention_sites
        from .adapters.minimax.layout import ensure_minimax_attention_layout_provider
        from .adapters.registry import ModelAdapter, register_model_adapter
        from .adapters.wan import apply_wan_adapter, install_wan_attention_sites
        from .adapters.wan_layout import ensure_wan_attention_layout_provider
        from .attention.integration import register_attention_site_installer
        from .attention.layout import register_attention_layout_provider
        from .runtime.stage_barrier import install_stage_barrier_scheduler

        register_attention_layout_provider(ensure_minimax_attention_layout_provider)
        register_attention_layout_provider(ensure_wan_attention_layout_provider)
        register_attention_site_installer(install_minimax_attention_sites)
        register_attention_site_installer(install_wan_attention_sites)
        register_model_adapter(ModelAdapter("minimax_h3", apply_minimax_adapter))
        register_model_adapter(ModelAdapter("wan", apply_wan_adapter))
        install_stage_barrier_scheduler()
        _BOOTSTRAPPED = True


__all__ = ["bootstrap_builtin_integrations"]
