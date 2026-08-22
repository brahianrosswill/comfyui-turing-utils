"""Opt-in CUDA phase timing with no events or synchronization by default."""

from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Callable

import torch

from .kernel_api import (
    attention_kernel_architectures,
    attention_runtime_profile_schema,
    kernel_version,
)


LOG = logging.getLogger("comfyui-turing-utils")


def _profile_call_limit() -> int:
    value = os.environ.get("COMFYUI_TURING_UTILS_PROFILE_CALLS", "0").strip()
    try:
        return max(int(value), 0)
    except ValueError:
        LOG.warning(
            "Ignoring invalid COMFYUI_TURING_UTILS_PROFILE_CALLS=%r; expected a non-negative integer",
            value,
        )
        return 0


def _runtime_profile_metadata() -> dict[str, str | int | bool]:
    architectures = attention_kernel_architectures()
    architecture_text = ",".join(architectures) if architectures else "unknown"
    result: dict[str, str | int | bool] = {
        "kernel": kernel_version(),
        "compiled_attention": architecture_text,
        "profile_schema": attention_runtime_profile_schema(),
        "device": "unknown",
        "device_sm": "unknown",
        "native_arch": "unknown",
    }
    try:
        index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        device_arch = f"sm{major}{minor}"
        result["device"] = str(torch.cuda.get_device_name(index))
        result["device_sm"] = device_arch
        if architectures:
            result["native_arch"] = any(
                architecture.split("+")[0] == device_arch
                for architecture in architectures
            )
    except (AttributeError, RuntimeError):
        pass
    return result


class CudaPhaseProfiler:
    """Collect asynchronous event pairs and report one bounded attention window."""

    def __init__(self, call_limit: int):
        self.call_limit = int(call_limit)
        self.calls = 0
        self.records: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.shapes = Counter()
        self.reported = False
        self.pending_report = False
        self.defer_sync_to_sampler_boundary = False

    @property
    def enabled(self) -> bool:
        return (
            self.call_limit > 0
            and not self.reported
            and not self.pending_report
        )

    def defer_to_sampler_boundary(self) -> None:
        """Use an existing outer sampler fence instead of a mid-model sync."""
        self.defer_sync_to_sampler_boundary = True

    def call(self, phase: str, function: Callable, /, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(*args, **kwargs)
        end.record()
        self.records.append((str(phase), start, end))
        return result

    def complete_attention(self, query_or_shape) -> None:
        if not self.enabled:
            return
        self.calls += 1
        shape = (
            tuple(query_or_shape.shape)
            if isinstance(query_or_shape, torch.Tensor)
            else tuple(query_or_shape)
        )
        self.shapes[str(shape)] += 1
        if self.calls < self.call_limit:
            return

        if not self.records:
            self.reported = True
            return
        self.pending_report = True
        if self.defer_sync_to_sampler_boundary:
            return
        self.records[-1][2].synchronize()
        self.report_after_synchronize()

    def report_after_synchronize(self) -> bool:
        """Emit a pending report after the caller has already drained CUDA."""
        if not self.pending_report or self.reported or not self.records:
            return False
        totals = Counter()
        counts = Counter()
        for phase, start, end in self.records:
            totals[phase] += start.elapsed_time(end)
            counts[phase] += 1
        total = sum(totals.values())
        LOG.warning(
            "[Turing profile] attention_calls=%d recorded_cuda=%.3f ms shapes=[%s]",
            self.calls,
            total,
            ",".join(f"{shape}:{count}" for shape, count in sorted(self.shapes.items())),
        )
        runtime = _runtime_profile_metadata()
        LOG.warning(
            "[Turing profile] runtime device=%s device_sm=%s kernel=%s "
            "compiled_attention=[%s] native_arch=%s profile_schema=%s",
            runtime["device"],
            runtime["device_sm"],
            runtime["kernel"],
            runtime["compiled_attention"],
            runtime["native_arch"],
            runtime["profile_schema"],
        )
        for phase, elapsed in totals.most_common():
            LOG.warning(
                "  %-32s calls=%4d total=%10.3f ms avg=%8.3f ms %6.2f%%",
                phase,
                counts[phase],
                elapsed,
                elapsed / counts[phase],
                elapsed * 100.0 / total if total else 0.0,
            )
        self.records.clear()
        self.pending_report = False
        self.reported = True
        return True


CUDA_PHASE_PROFILER = CudaPhaseProfiler(_profile_call_limit())
__all__ = ["CUDA_PHASE_PROFILER"]
