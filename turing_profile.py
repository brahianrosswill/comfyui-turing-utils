"""Temporary always-on CUDA timing for the Turing H3 performance audit.

The profiler records CUDA events asynchronously for two 50-block windows.
The first window captures cold-start behavior and the second captures steady
state.  It then disables itself so normal sampling does not keep paying the
event-recording overhead.  This module is intentionally temporary and should
be removed after the Turing measurements are complete.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from collections import defaultdict

import torch


LOG = logging.getLogger("comfyui-svdint4")

BLOCKS_PER_WINDOW = 50
WINDOW_COUNT = 2


@dataclasses.dataclass
class _Sample:
    label: str
    start: torch.cuda.Event
    end: torch.cuda.Event


@dataclasses.dataclass
class _DeviceProfile:
    device_index: int
    active_depth: int = 0
    blocks: int = 0
    windows: int = 0
    enabled: bool = True
    dtype: torch.dtype | None = None
    previous_block_end: torch.cuda.Event | None = None
    samples: list[_Sample] = dataclasses.field(default_factory=list)


_PROFILES: dict[int, _DeviceProfile] = {}


def _device_index(value) -> int | None:
    device = getattr(value, "device", value)
    if not isinstance(device, torch.device) or device.type != "cuda":
        return None
    return device.index if device.index is not None else torch.cuda.current_device()


def _profile(value) -> _DeviceProfile | None:
    index = _device_index(value)
    if index is None:
        return None
    state = _PROFILES.get(index)
    if state is None:
        state = _DeviceProfile(index)
        _PROFILES[index] = state
    return state


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


@contextlib.contextmanager
def cuda_region(label: str, value):
    """Record a nested CUDA region only while a profiled block is active."""
    state = _profile(value)
    if state is None or not state.enabled or state.active_depth <= 0:
        yield
        return

    start = _event()
    end = _event()
    start.record()
    try:
        yield
    finally:
        end.record()
        state.samples.append(_Sample(label, start, end))


@contextlib.contextmanager
def profile_block(value):
    """Record one DiT block and report after each complete 50-block window."""
    state = _profile(value)
    if state is None or not state.enabled:
        yield
        return

    start = _event()
    start.record()
    if state.previous_block_end is not None:
        state.samples.append(
            _Sample("gap.inter_block", state.previous_block_end, start)
        )
    state.dtype = getattr(value, "dtype", state.dtype)
    state.active_depth += 1
    try:
        yield
    finally:
        end = _event()
        end.record()
        state.samples.append(_Sample("block.total", start, end))
        state.active_depth -= 1
        state.previous_block_end = end
        state.blocks += 1
        if state.blocks >= BLOCKS_PER_WINDOW:
            _report(state, end)


def _aggregate(state: _DeviceProfile) -> dict[str, tuple[int, float]]:
    calls: dict[str, int] = defaultdict(int)
    totals: dict[str, float] = defaultdict(float)
    for sample in state.samples:
        calls[sample.label] += 1
        totals[sample.label] += sample.start.elapsed_time(sample.end)
    return {label: (calls[label], totals[label]) for label in totals}


def _sum_prefix(
    aggregate: dict[str, tuple[int, float]],
    prefixes: tuple[str, ...],
) -> float:
    return sum(
        total
        for label, (_calls, total) in aggregate.items()
        if label.startswith(prefixes)
    )


def _log_group(
    title: str,
    aggregate: dict[str, tuple[int, float]],
    prefix: str,
    denominator: float,
) -> None:
    rows = [
        (label, calls, total)
        for label, (calls, total) in aggregate.items()
        if label.startswith(prefix)
    ]
    if not rows:
        return
    LOG.warning("[SVDInt4 Turing profile] %s", title)
    for label, calls, total in sorted(rows, key=lambda row: row[2], reverse=True):
        percent = 100.0 * total / denominator if denominator > 0.0 else 0.0
        LOG.warning(
            "  %-54s calls=%4d total=%9.3f ms avg=%8.3f ms %6.2f%%",
            label,
            calls,
            total,
            total / calls,
            percent,
        )


def _report(state: _DeviceProfile, final_event: torch.cuda.Event) -> None:
    final_event.synchronize()
    aggregate = _aggregate(state)
    block_total = aggregate.get("block.total", (0, 0.0))[1]
    inter_block = aggregate.get("gap.inter_block", (0, 0.0))[1]
    phase_total = _sum_prefix(aggregate, ("phase.",))
    unaccounted = max(0.0, block_total - phase_total)

    state.windows += 1
    kind = "cold" if state.windows == 1 else "steady"
    LOG.warning(
        "[SVDInt4 Turing profile] window=%d/%d (%s) device=cuda:%d "
        "dtype=%s blocks=%d block_cuda=%9.3f ms inter_block=%9.3f ms",
        state.windows,
        WINDOW_COUNT,
        kind,
        state.device_index,
        state.dtype,
        state.blocks,
        block_total,
        inter_block,
    )
    _log_group("exclusive block phases", aggregate, "phase.", block_total)
    if unaccounted > 0.01:
        LOG.warning(
            "  %-54s total=%9.3f ms %6.2f%%",
            "derived.block_unaccounted",
            unaccounted,
            100.0 * unaccounted / block_total if block_total > 0.0 else 0.0,
        )
    _log_group("nested kernel details", aggregate, "detail.", block_total)

    attention_known = _sum_prefix(
        aggregate,
        (
            "detail.linear.qkv.",
            "detail.linear.out_proj.",
            "detail.attention.",
        ),
    )
    mlp_known = _sum_prefix(
        aggregate,
        ("detail.linear.fc1.", "detail.linear.fc2."),
    )
    attention_total = aggregate.get("phase.attention", (0, 0.0))[1]
    mlp_total = aggregate.get("phase.mlp", (0, 0.0))[1]
    LOG.warning(
        "[SVDInt4 Turing profile] derived attention_other=%9.3f ms "
        "(primarily Q/K RMSNorm+RoPE and dispatch gaps), mlp_other=%9.3f ms",
        max(0.0, attention_total - attention_known),
        max(0.0, mlp_total - mlp_known),
    )

    state.samples.clear()
    state.blocks = 0
    state.previous_block_end = None
    if state.windows >= WINDOW_COUNT:
        state.enabled = False
        LOG.warning(
            "[SVDInt4 Turing profile] capture complete; profiling is now disabled "
            "for the rest of this process"
        )


def _reset_for_tests() -> None:
    _PROFILES.clear()
