from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

import attention as attention_backends  # noqa: E402
import attention_nodes  # noqa: E402


class FakePatcher:
    def __init__(self):
        self.load_device = torch.device("cuda", 0)
        self.model_options = {"transformer_options": {"existing": True}}

    def clone(self):
        clone = FakePatcher()
        clone.model_options = {
            "transformer_options": self.model_options["transformer_options"].copy()
        }
        return clone


class SparseAttentionNodeTest(unittest.TestCase):
    def test_schema_exposes_tunable_sparse_parameters(self):
        inputs = attention_nodes.SolSparseAttentionPatch.INPUT_TYPES()["required"]
        self.assertEqual(
            tuple(inputs),
            (
                "model",
                "min_sequence_tokens",
                "routing_threshold",
                "prefix_policy",
                "manual_prefix_tokens",
                "local_block_radius",
                "temporal_neighbor_frames",
                "skipped_residual",
                "minimum_route_density",
                "maximum_route_density",
                "dense_warmup_ratio",
                "dense_tail_ratio",
                "dense_prefix_layers",
            ),
        )
        self.assertEqual(inputs["min_sequence_tokens"][1]["default"], 0)
        self.assertEqual(inputs["routing_threshold"][1]["default"], 1.0)
        self.assertEqual(inputs["prefix_policy"][0][0], "auto")
        self.assertEqual(inputs["manual_prefix_tokens"][1]["default"], 0)
        self.assertEqual(inputs["local_block_radius"][1]["default"], 1)
        self.assertEqual(inputs["temporal_neighbor_frames"][1]["default"], 1)
        self.assertEqual(inputs["skipped_residual"][0][0], "2x32")
        self.assertEqual(inputs["minimum_route_density"][1]["default"], 0.0)
        self.assertEqual(inputs["maximum_route_density"][1]["default"], 1.0)
        self.assertEqual(inputs["dense_warmup_ratio"][1]["default"], 0.25)
        self.assertEqual(inputs["dense_tail_ratio"][1]["default"], 0.0)
        self.assertEqual(inputs["dense_prefix_layers"][1]["default"], 2)
        optional = attention_nodes.SolSparseAttentionPatch.INPUT_TYPES()["optional"]
        self.assertEqual(tuple(optional), ("debug_route_density",))
        self.assertFalse(optional["debug_route_density"][1]["default"])

    def test_patch_clones_model_and_installs_generic_override(self):
        model = FakePatcher()
        override = object()
        with mock.patch(
            "attention.make_sparse_attention_override", return_value=override
        ) as make_override:
            patched = attention_backends.apply_sparse_attention_patch(
                model,
                min_sequence_tokens=8192,
                routing_threshold=0.85,
                prefix_policy="manual",
                manual_prefix_tokens=256,
                local_block_radius=2,
                temporal_neighbor_frames=2,
                skipped_residual="1x64",
                minimum_route_density=0.2,
                maximum_route_density=0.7,
                dense_warmup_ratio=0.25,
                dense_tail_ratio=0.1,
                dense_prefix_layers=3,
            )

        self.assertIsNot(patched, model)
        self.assertNotIn(
            "optimized_attention_override", model.model_options["transformer_options"]
        )
        options = patched.model_options["transformer_options"]
        self.assertIs(options["optimized_attention_override"], override)
        self.assertTrue(options["existing"])
        make_override.assert_called_once_with(
            torch.device("cuda", 0),
            min_sequence_tokens=8192,
            routing_threshold=0.85,
            prefix_policy="manual",
            manual_prefix_tokens=256,
            local_block_radius=2,
            temporal_neighbor_frames=2,
            skipped_residual="1x64",
            minimum_route_density=0.2,
            maximum_route_density=0.7,
            dense_warmup_ratio=0.25,
            dense_tail_ratio=0.1,
            dense_prefix_layers=3,
            debug_route_density=False,
        )

    def test_node_returns_the_patched_model(self):
        model = object()
        patched = object()
        with mock.patch(
            "attention_nodes.apply_sparse_attention_patch", return_value=patched
        ) as apply_patch:
            output = attention_nodes.SolSparseAttentionPatch().patch(
                model,
                8192,
                0.85,
                "manual",
                256,
                2,
                2,
                "1x64",
                0.2,
                0.7,
                0.25,
                0.1,
                3,
            )
        self.assertEqual(output, (patched,))
        apply_patch.assert_called_once_with(
            model,
            min_sequence_tokens=8192,
            routing_threshold=0.85,
            prefix_policy="manual",
            manual_prefix_tokens=256,
            local_block_radius=2,
            temporal_neighbor_frames=2,
            skipped_residual="1x64",
            minimum_route_density=0.2,
            maximum_route_density=0.7,
            dense_warmup_ratio=0.25,
            dense_tail_ratio=0.1,
            dense_prefix_layers=3,
            debug_route_density=False,
        )


if __name__ == "__main__":
    unittest.main()
