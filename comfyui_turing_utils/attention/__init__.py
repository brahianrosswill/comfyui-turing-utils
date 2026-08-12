"""Public attention backend and patch API."""

from .patches import (
    apply_attention_backend,
    apply_sparse_attention_patch,
    make_attention_override,
    make_sparse_attention_override,
)
from .sparse import turing_sol_sparse_attention
from .tuning import apply_attention_kernel_tuning_patch
from .stable import (
    AttentionBackend,
    attention_backend_choices,
    bundled_available,
    bundled_sparse_available,
    normalize_attention_backend,
    preflight_bundled,
    preflight_bundled_sparse,
    register_attention_backend,
    turing_sage_attention,
)

__all__ = [
    "AttentionBackend",
    "apply_attention_backend",
    "apply_sparse_attention_patch",
    "apply_attention_kernel_tuning_patch",
    "attention_backend_choices",
    "bundled_available",
    "bundled_sparse_available",
    "make_attention_override",
    "make_sparse_attention_override",
    "normalize_attention_backend",
    "preflight_bundled",
    "preflight_bundled_sparse",
    "register_attention_backend",
    "turing_sage_attention",
    "turing_sol_sparse_attention",
]
