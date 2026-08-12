"""Explicit numerical release gate for bundled SM75 attention backends.

This module is never imported by an attention hot path.  Callers opt into the
gate from validation tooling or a tuning workflow, so normal inference pays no
allocation, synchronization, or metric-reduction cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AttentionCorrectnessResult:
    candidate: str
    reference: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    finite: bool
    max_abs: float
    mean_abs: float
    relative_l2: float
    cosine: float
    selected_blocks: int | None = None
    possible_blocks: int | None = None


def attention_error_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    candidate_name: str = "candidate",
    reference_name: str = "reference",
    selected_blocks: int | None = None,
    possible_blocks: int | None = None,
) -> AttentionCorrectnessResult:
    """Compute synchronization-explicit metrics without hiding invalid output."""
    if candidate.shape != reference.shape:
        raise AssertionError(
            f"{candidate_name} shape {tuple(candidate.shape)} does not match "
            f"{reference_name} shape {tuple(reference.shape)}"
        )
    actual = candidate.float()
    expected = reference.float()
    finite = bool(torch.isfinite(actual).all().item())
    error = actual - expected
    reference_norm = torch.linalg.vector_norm(expected).clamp_min(1.0e-12)
    candidate_norm = torch.linalg.vector_norm(actual)
    cosine_denominator = (candidate_norm * reference_norm).clamp_min(1.0e-12)
    return AttentionCorrectnessResult(
        candidate=candidate_name,
        reference=reference_name,
        shape=tuple(candidate.shape),
        dtype=candidate.dtype,
        finite=finite,
        max_abs=float(error.abs().max().item()),
        mean_abs=float(error.abs().mean().item()),
        relative_l2=float((torch.linalg.vector_norm(error) / reference_norm).item()),
        cosine=float((torch.dot(actual.flatten(), expected.flatten()) / cosine_denominator).item()),
        selected_blocks=selected_blocks,
        possible_blocks=possible_blocks,
    )


def require_attention_correctness(
    result: AttentionCorrectnessResult,
    *,
    max_abs: float,
    relative_l2: float,
    cosine: float,
) -> None:
    failures = []
    if not result.finite:
        failures.append("non-finite output")
    if result.max_abs > float(max_abs):
        failures.append(f"max_abs={result.max_abs:.6g}>{max_abs:.6g}")
    if result.relative_l2 > float(relative_l2):
        failures.append(f"relative_l2={result.relative_l2:.6g}>{relative_l2:.6g}")
    if result.cosine < float(cosine):
        failures.append(f"cosine={result.cosine:.6g}<{cosine:.6g}")
    if (
        result.possible_blocks is not None
        and result.selected_blocks != result.possible_blocks
    ):
        failures.append(
            f"dense route selected {result.selected_blocks}/{result.possible_blocks} blocks"
        )
    if failures:
        raise AssertionError(
            f"{result.candidate} correctness gate against {result.reference} failed: "
            + "; ".join(failures)
        )


def run_attention_correctness_gate(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    use_w8a8: bool = False,
    max_abs: float | None = None,
    relative_l2: float | None = None,
    cosine: float | None = None,
) -> AttentionCorrectnessResult:
    """Compare a fully selected Sol route with its matching dense backend.

    FP16-PV Sol is gated against stable Sage.  INT8-PV Sol is gated against the
    route-free W8A8 backend, isolating routing/online-softmax correctness from
    the intentionally different V quantization error.
    """
    if not query.is_cuda:
        raise ValueError("the attention correctness gate requires CUDA tensors")
    from .core import sageattn, sol_sparse_sageattn, w8a8attn

    reference_name = "w8a8" if use_w8a8 else "sage"
    reference = (
        w8a8attn(query, key, value)
        if use_w8a8
        else sageattn(query, key, value, smooth_k=False)
    )
    candidate, selected, possible = sol_sparse_sageattn(
        query,
        key,
        value,
        threshold_sigma=-1000.0,
        return_stats=True,
        use_w8a8=use_w8a8,
    )
    result = attention_error_metrics(
        candidate,
        reference,
        candidate_name="sol_w8a8" if use_w8a8 else "sol",
        reference_name=reference_name,
        selected_blocks=int(selected.item()),
        possible_blocks=int(possible),
    )
    require_attention_correctness(
        result,
        max_abs=0.04 if max_abs is None else max_abs,
        relative_l2=0.025 if relative_l2 is None else relative_l2,
        cosine=0.999 if cosine is None else cosine,
    )
    return result


__all__ = [
    "AttentionCorrectnessResult",
    "attention_error_metrics",
    "require_attention_correctness",
    "run_attention_correctness_gate",
]
