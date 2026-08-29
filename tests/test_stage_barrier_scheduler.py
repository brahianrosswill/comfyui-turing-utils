from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "comfyui_turing_utils" / "runtime" / "stage_barrier.py"
)
SPEC = importlib.util.spec_from_file_location(
    "turing_utils_stage_barrier_test_target", MODULE_PATH
)
stage_barrier_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stage_barrier_module)
STAGE_BARRIER_NODE_ID = stage_barrier_module.STAGE_BARRIER_NODE_ID
stage_barrier_candidates = stage_barrier_module.stage_barrier_candidates


class _Prompt:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_node(self, node_id):
        return self.nodes[node_id]


def _normal():
    return {"class_type": "Normal", "inputs": {}}


def _barrier(stage):
    return {
        "class_type": STAGE_BARRIER_NODE_ID,
        "inputs": {"stage": stage},
    }


def _blocking(nodes, edges):
    result = {node_id: {} for node_id in nodes}
    for source, target in edges:
        result[source][target] = {}
    return result


class StageBarrierSchedulerTest(unittest.TestCase):
    def test_lower_stage_upstream_is_selected_before_higher_stage(self):
        nodes = {
            "low_work": _normal(),
            "low": _barrier(0),
            "high_work": _normal(),
            "high": _barrier(1),
        }
        candidates = stage_barrier_candidates(
            _Prompt(nodes),
            nodes,
            _blocking(
                nodes,
                (("low_work", "low"), ("high_work", "high")),
            ),
            ["high_work", "low_work"],
        )
        self.assertEqual(candidates, ["low_work"])

    def test_same_stage_barriers_hold_unrelated_downstream_work(self):
        nodes = {
            "ready_barrier": _barrier(0),
            "other_work": _normal(),
            "other_barrier": _barrier(0),
            "unrelated": _normal(),
        }
        candidates = stage_barrier_candidates(
            _Prompt(nodes),
            nodes,
            _blocking(nodes, (("other_work", "other_barrier"),)),
            ["unrelated", "ready_barrier", "other_work"],
        )
        self.assertEqual(candidates, ["ready_barrier", "other_work"])

    def test_higher_dependency_inherits_lower_stage_priority(self):
        nodes = {
            "high_work": _normal(),
            "high_dependency": _barrier(4),
            "deferred_low": _barrier(0),
            "independent_mid": _barrier(1),
        }
        candidates = stage_barrier_candidates(
            _Prompt(nodes),
            nodes,
            _blocking(
                nodes,
                (
                    ("high_work", "high_dependency"),
                    ("high_dependency", "deferred_low"),
                ),
            ),
            ["independent_mid", "high_work"],
        )
        self.assertEqual(candidates, ["high_work"])

    def test_independent_low_barrier_is_not_blocked_by_inverted_dependency(self):
        nodes = {
            "high_work": _normal(),
            "high_dependency": _barrier(4),
            "deferred_low": _barrier(0),
            "independent_low": _barrier(0),
        }
        candidates = stage_barrier_candidates(
            _Prompt(nodes),
            nodes,
            _blocking(
                nodes,
                (
                    ("high_work", "high_dependency"),
                    ("high_dependency", "deferred_low"),
                ),
            ),
            ["high_work", "independent_low"],
        )
        self.assertEqual(candidates, ["high_work", "independent_low"])

    def test_real_graph_cycle_falls_back_to_comfyui_diagnostics(self):
        nodes = {
            "first": _barrier(0),
            "second": _barrier(1),
            "free": _normal(),
        }
        candidates = stage_barrier_candidates(
            _Prompt(nodes),
            nodes,
            _blocking(nodes, (("first", "second"), ("second", "first"))),
            ["free"],
        )
        self.assertEqual(candidates, ["free"])


if __name__ == "__main__":
    unittest.main()
