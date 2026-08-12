"""Shared contract for adapter-owned Q/K preprocessing."""

from __future__ import annotations

import dataclasses

import torch


QK_PREPROCESSOR_KEY = "turing_utils_qk_preprocessor"


@dataclasses.dataclass(frozen=True, slots=True)
class QKPreprocessSpec:
    """Model semantics required before Q/K enter an attention kernel."""

    query_norm: torch.Tensor
    key_norm: torch.Tensor
    freqs: torch.Tensor | None
    epsilon: float
    rot_dim: int
    norm_scope: str
    split_half: bool


def qk_preprocessor(transformer_options):
    if not isinstance(transformer_options, dict):
        return None
    processor = transformer_options.get(QK_PREPROCESSOR_KEY)
    return processor if callable(processor) else None


__all__ = ["QK_PREPROCESSOR_KEY", "QKPreprocessSpec", "qk_preprocessor"]
