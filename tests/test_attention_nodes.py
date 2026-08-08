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
            ("model", "min_sequence_tokens", "dense_prefix_tokens", "route_threshold"),
        )
        self.assertEqual(inputs["min_sequence_tokens"][1]["default"], 4096)
        self.assertEqual(inputs["dense_prefix_tokens"][1]["default"], 512)
        self.assertEqual(inputs["route_threshold"][1]["default"], 1.0)

    def test_patch_clones_model_and_installs_generic_override(self):
        model = FakePatcher()
        override = object()
        with mock.patch(
            "attention.make_sparse_attention_override", return_value=override
        ) as make_override:
            patched = attention_backends.apply_sparse_attention_patch(
                model,
                min_sequence_tokens=8192,
                dense_prefix_tokens=256,
                route_threshold=1.5,
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
            dense_prefix_tokens=256,
            route_threshold=1.5,
        )

    def test_node_returns_the_patched_model(self):
        model = object()
        patched = object()
        with mock.patch(
            "attention_nodes.apply_sparse_attention_patch", return_value=patched
        ) as apply_patch:
            output = attention_nodes.SolSparseAttentionPatch().patch(
                model, 8192, 256, 1.5
            )
        self.assertEqual(output, (patched,))
        apply_patch.assert_called_once_with(
            model,
            min_sequence_tokens=8192,
            dense_prefix_tokens=256,
            route_threshold=1.5,
        )


if __name__ == "__main__":
    unittest.main()
