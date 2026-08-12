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
    def test_kernel_tuning_node_has_safe_defaults(self):
        inputs = attention_nodes.AttentionKernelTuningPatch.INPUT_TYPES()["required"]
        self.assertEqual(tuple(inputs), ("model", "key_tile", "hadamard_qk", "adaptive_k_anchor"))
        self.assertEqual(inputs["key_tile"][0][0], "auto")
        self.assertTrue(inputs["hadamard_qk"][1]["default"])
        self.assertTrue(inputs["adaptive_k_anchor"][1]["default"])

    def test_kernel_tuning_patch_is_order_independent_model_metadata(self):
        model = FakePatcher()
        patched = attention_backends.apply_attention_kernel_tuning_patch(
            model,
            key_tile="128",
            rotate_qk=False,
            stabilize_k=True,
        )
        self.assertIsNot(patched, model)
        self.assertNotIn(
            "turing_utils_attention_tuning",
            model.model_options["transformer_options"],
        )
        self.assertEqual(
            patched.model_options["transformer_options"]["turing_utils_attention_tuning"],
            {"key_tile_tokens": 128, "rotate_qk": False, "stabilize_k": False},
        )

    def test_schema_exposes_tunable_sparse_parameters(self):
        inputs = attention_nodes.SolSparseAttentionPatch.INPUT_TYPES()["required"]
        self.assertEqual(
            tuple(inputs),
            (
                "model",
                "routing_threshold",
                "prefix_policy",
                "manual_prefix_tokens",
                "skipped_residual",
                "sparse_reference_image",
                "sparse_reference_video",
                "sparse_reference_audio",
                "dense_prefix_steps",
                "dense_suffix_steps",
                "dense_prefix_layers",
                "dense_suffix_layers",
            ),
        )
        self.assertEqual(inputs["routing_threshold"][1]["default"], 1.0)
        self.assertEqual(inputs["prefix_policy"][0][0], "auto")
        self.assertEqual(inputs["manual_prefix_tokens"][1]["default"], 0)
        self.assertEqual(inputs["skipped_residual"][0][0], "1x64")
        self.assertFalse(inputs["sparse_reference_image"][1]["default"])
        self.assertTrue(inputs["sparse_reference_video"][1]["default"])
        self.assertFalse(inputs["sparse_reference_audio"][1]["default"])
        self.assertEqual(inputs["dense_prefix_steps"][0], "INT")
        self.assertEqual(inputs["dense_prefix_steps"][1]["default"], 0)
        self.assertEqual(inputs["dense_suffix_steps"][0], "INT")
        self.assertEqual(inputs["dense_suffix_steps"][1]["default"], 0)
        self.assertEqual(inputs["dense_prefix_layers"][1]["default"], 2)
        self.assertEqual(inputs["dense_suffix_layers"][1]["default"], 0)
        optional = attention_nodes.SolSparseAttentionPatch.INPUT_TYPES()["optional"]
        self.assertEqual(tuple(optional), ("use_w8a8", "debug_route_density"))
        self.assertFalse(optional["use_w8a8"][1]["default"])
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
                skipped_residual="1x64",
                sparse_reference_image=True,
                sparse_reference_video=False,
                sparse_reference_audio=True,
                dense_prefix_steps=2,
                dense_suffix_steps=1,
                dense_prefix_layers=3,
                dense_suffix_layers=4,
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
            skipped_residual="1x64",
            sparse_reference_image=True,
            sparse_reference_video=False,
            sparse_reference_audio=True,
            dense_prefix_steps=2,
            dense_suffix_steps=1,
            dense_prefix_layers=3,
            dense_suffix_layers=4,
            debug_route_density=False,
            use_w8a8=False,
        )

    def test_node_returns_the_patched_model(self):
        model = object()
        patched = object()
        with mock.patch(
            "attention_nodes.apply_sparse_attention_patch", return_value=patched
        ) as apply_patch:
            output = attention_nodes.SolSparseAttentionPatch().patch(
                model,
                0.85,
                "manual",
                256,
                "1x64",
                True,
                False,
                True,
                2,
                1,
                3,
                4,
            )
        self.assertEqual(output, (patched,))
        apply_patch.assert_called_once_with(
            model,
            routing_threshold=0.85,
            prefix_policy="manual",
            manual_prefix_tokens=256,
            skipped_residual="1x64",
            sparse_reference_image=True,
            sparse_reference_video=False,
            sparse_reference_audio=True,
            dense_prefix_steps=2,
            dense_suffix_steps=1,
            dense_prefix_layers=3,
            dense_suffix_layers=4,
            use_w8a8=False,
            debug_route_density=False,
        )

if __name__ == "__main__":
    unittest.main()
