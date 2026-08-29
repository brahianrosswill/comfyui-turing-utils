"""Stage-aware ComfyUI execution ordering.

Stage barriers are ordinary data-flow nodes.  This module only changes which
*ready* node ComfyUI picks next; it never bypasses a graph dependency and never
blocks the executor thread.  Barrier dependencies therefore remain the source
of truth, including the dependency-inversion case where a low-numbered barrier
needs the output of a higher-numbered one.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping


STAGE_BARRIER_NODE_ID = "TuringUtilsStageBarrier"
_PATCH_MARKER = "_turing_utils_stage_barrier_scheduler"


def _barrier_stage(dynprompt, node_id: str) -> int | None:
    try:
        node = dynprompt.get_node(node_id)
    except (KeyError, TypeError, AttributeError):
        return None
    if node.get("class_type") != STAGE_BARRIER_NODE_ID:
        return None
    try:
        return max(0, int(node.get("inputs", {}).get("stage", 0)))
    except (TypeError, ValueError):
        # ComfyUI validates the widget before execution.  Keeping malformed
        # workflows in stage zero here avoids breaking scheduler diagnostics.
        return 0


def _graph_predecessors(
    pending: set[str], blocking: Mapping[str, Mapping[str, object]]
) -> dict[str, set[str]]:
    predecessors = {node_id: set() for node_id in pending}
    for source in pending:
        for target in blocking.get(source, {}):
            if target in pending:
                predecessors[target].add(source)
    return predecessors


def _reachable_barriers(
    source: str,
    barriers: set[str],
    pending: set[str],
    blocking: Mapping[str, Mapping[str, object]],
) -> set[str]:
    found = set()
    visited = {source}
    stack = list(blocking.get(source, {}))
    while stack:
        node_id = stack.pop()
        if node_id in visited or node_id not in pending:
            continue
        visited.add(node_id)
        if node_id in barriers:
            found.add(node_id)
        stack.extend(blocking.get(node_id, {}))
    return found


def _ancestors(
    targets: Iterable[str], predecessors: Mapping[str, set[str]]
) -> set[str]:
    required = set(targets)
    stack = list(required)
    while stack:
        node_id = stack.pop()
        for source in predecessors.get(node_id, ()):
            if source not in required:
                required.add(source)
                stack.append(source)
    return required


def stage_barrier_candidates(
    dynprompt,
    pending_nodes: Iterable[str],
    blocking: Mapping[str, Mapping[str, object]],
    available_nodes: Iterable[str],
) -> list[str]:
    """Return the ready nodes allowed to advance the current barrier phase.

    A barrier's effective priority is the smallest stage of any downstream
    barrier that depends on it.  This is priority inheritance: a stage-4
    barrier feeding a stage-1 barrier is temporarily treated as stage 1, so the
    dependency inversion cannot deadlock stage ordering.  Independent stage-1
    barriers retain the same priority and continue synchronizing normally.
    """

    available = list(available_nodes)
    pending = set(pending_nodes)
    stages = {
        node_id: stage
        for node_id in pending
        if (stage := _barrier_stage(dynprompt, node_id)) is not None
    }
    if not stages:
        return available

    barrier_ids = set(stages)
    predecessors = _graph_predecessors(pending, blocking)
    barrier_predecessors = {node_id: set() for node_id in barrier_ids}
    effective_stage = dict(stages)

    for source in barrier_ids:
        descendants = _reachable_barriers(
            source, barrier_ids, pending, blocking
        )
        if descendants:
            effective_stage[source] = min(
                stages[source], *(stages[target] for target in descendants)
            )
        for target in descendants:
            barrier_predecessors[target].add(source)

    # Only barrier roots can run without violating real data dependencies.
    # An actual graph cycle has no roots; falling back lets ComfyUI emit its
    # normal dependency-cycle diagnostic instead of hiding it here.
    roots = [
        node_id
        for node_id in barrier_ids
        if not barrier_predecessors[node_id]
    ]
    if not roots:
        return available

    priority = min(effective_stage[node_id] for node_id in roots)
    targets = [
        node_id
        for node_id in roots
        if effective_stage[node_id] == priority
    ]
    required = _ancestors(targets, predecessors)
    candidates = [node_id for node_id in available if node_id in required]
    return candidates or available


def install_stage_barrier_scheduler() -> bool:
    """Wrap ComfyUI's ready-node picker exactly once."""

    try:
        from comfy_execution.graph import ExecutionList
    except (ImportError, AttributeError):
        logging.warning(
            "Stage Barrier ordering is unavailable: this ComfyUI build has no "
            "compatible ExecutionList scheduler"
        )
        return False

    original = getattr(ExecutionList, "ux_friendly_pick_node", None)
    if original is None:
        logging.warning(
            "Stage Barrier ordering is unavailable: ComfyUI's ready-node picker "
            "was not found"
        )
        return False
    if getattr(original, _PATCH_MARKER, False):
        return True

    def stage_aware_pick_node(self, node_list):
        candidates = stage_barrier_candidates(
            self.dynprompt,
            self.pendingNodes,
            self.blocking,
            node_list,
        )
        return original(self, candidates)

    setattr(stage_aware_pick_node, _PATCH_MARKER, True)
    setattr(stage_aware_pick_node, "_turing_utils_original", original)
    ExecutionList.ux_friendly_pick_node = stage_aware_pick_node
    logging.info("Enabled stage-aware workflow barrier scheduling")
    return True


__all__ = [
    "STAGE_BARRIER_NODE_ID",
    "install_stage_barrier_scheduler",
    "stage_barrier_candidates",
]
