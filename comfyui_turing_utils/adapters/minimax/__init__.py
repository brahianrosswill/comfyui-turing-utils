"""MiniMax H3 model integration."""

from .acceleration import apply_minimax_adapter
from .layout import ensure_minimax_attention_layout_provider

__all__ = [
    "apply_minimax_adapter",
    "ensure_minimax_attention_layout_provider",
]
