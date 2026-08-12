"""MiniMax H3 model integration."""

from .acceleration import apply_minimax_adapter
from .layout import ensure_minimax_attention_layout_provider
from .progressive import apply_h3_progressive_resolution_patch

__all__ = [
    "apply_h3_progressive_resolution_patch",
    "apply_minimax_adapter",
    "ensure_minimax_attention_layout_provider",
]
