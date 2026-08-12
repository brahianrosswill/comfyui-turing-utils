"""Implementation package for ComfyUI Turing Utils.

The package root intentionally has no ComfyUI imports. Node discovery is owned
by :mod:`comfyui_turing_utils.registration`, while backend modules remain
independently importable by validation tools and tests.
"""

from .attention.layout import register_attention_layout_provider
from .attention.integration import register_attention_site_installer
from .adapters.minimax.acceleration import apply_minimax_adapter
from .adapters.minimax.acceleration import install_minimax_attention_sites
from .adapters.minimax.layout import ensure_minimax_attention_layout_provider
from .adapters.registry import ModelAdapter, register_model_adapter
from .adapters.wan import apply_wan_adapter
from .adapters.wan import install_wan_attention_sites
from .adapters.wan_layout import ensure_wan_attention_layout_provider


register_attention_layout_provider(ensure_minimax_attention_layout_provider)
register_attention_layout_provider(ensure_wan_attention_layout_provider)
register_attention_site_installer(install_minimax_attention_sites)
register_attention_site_installer(install_wan_attention_sites)
register_model_adapter(ModelAdapter("minimax_h3", apply_minimax_adapter))
register_model_adapter(ModelAdapter("wan", apply_wan_adapter))

del (
    ModelAdapter,
    apply_minimax_adapter,
    apply_wan_adapter,
    ensure_minimax_attention_layout_provider,
    ensure_wan_attention_layout_provider,
    install_minimax_attention_sites,
    install_wan_attention_sites,
    register_attention_layout_provider,
    register_attention_site_installer,
    register_model_adapter,
)
