"""Runtime capability and resource-planning services."""

from .capabilities import (
    CapabilityResult,
    KernelCapabilities,
    RuntimeCapabilities,
    kernel_capabilities,
    runtime_capabilities,
)
from .diagnostics import FEATURES, runtime_diagnostics
from .stage_barrier import (
    STAGE_BARRIER_NODE_ID,
    install_stage_barrier_scheduler,
    stage_barrier_candidates,
)

__all__ = [
    "CapabilityResult",
    "KernelCapabilities",
    "RuntimeCapabilities",
    "kernel_capabilities",
    "runtime_capabilities",
    "FEATURES",
    "runtime_diagnostics",
    "STAGE_BARRIER_NODE_ID",
    "install_stage_barrier_scheduler",
    "stage_barrier_candidates",
]
