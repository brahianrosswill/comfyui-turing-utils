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
            debug_route_density=False,
        )

    def test_frame_sparse_schema_exposes_structured_video_parameters(self):
        inputs = attention_nodes.FrameSparseAttentionPatch.INPUT_TYPES()["required"]
        self.assertEqual(
            tuple(inputs),
            (
                "model",
                "quality_profile",
                "sparse_pattern",
                "prefix_policy",
                "manual_prefix_tokens",
                "temporal_window_frames",
                "global_anchor_stride",
                "rotate_global_anchors",
                "sink_frames",
                "radial_spatial_radius",
                "radial_max_temporal_stride",
                "dense_prefix_steps",
                "dense_suffix_steps",
                "dense_prefix_layers",
                "dense_suffix_layers",
            ),
        )
        self.assertEqual(inputs["prefix_policy"][0][0], "auto")
        self.assertEqual(inputs["quality_profile"][0][0], "custom")
        self.assertEqual(inputs["sparse_pattern"][0][0], "frame_window")
        self.assertEqual(inputs["temporal_window_frames"][1]["default"], 2)
        self.assertEqual(inputs["global_anchor_stride"][1]["default"], 12)
        self.assertTrue(inputs["rotate_global_anchors"][1]["default"])
        self.assertEqual(inputs["sink_frames"][1]["default"], 1)
        self.assertEqual(inputs["radial_spatial_radius"][1]["default"], 1)
        self.assertEqual(inputs["radial_max_temporal_stride"][1]["default"], 16)
        self.assertEqual(inputs["dense_prefix_steps"][1]["default"], 0)
        self.assertEqual(inputs["dense_suffix_steps"][1]["default"], 0)
        self.assertEqual(inputs["dense_prefix_layers"][1]["default"], 1)
        self.assertEqual(inputs["dense_suffix_layers"][1]["default"], 1)

    def test_frame_sparse_node_returns_patched_model(self):
        model = object()
        patched = object()
        with mock.patch(
            "attention_nodes.apply_frame_sparse_attention_patch", return_value=patched
        ) as apply_patch:
            output = attention_nodes.FrameSparseAttentionPatch().patch(
                model,
                "custom",
                "radial",
                "manual",
                256,
                3,
                16,
                False,
                2,
                1,
                8,
                4,
                1,
                2,
                3,
                True,
            )
        self.assertEqual(output, (patched,))
        apply_patch.assert_called_once_with(
            model,
            quality_profile="custom",
            sparse_pattern="radial",
            prefix_policy="manual",
            manual_prefix_tokens=256,
            temporal_window_frames=3,
            global_anchor_stride=16,
            rotate_global_anchors=False,
            sink_frames=2,
            radial_spatial_radius=1,
            radial_max_temporal_stride=8,
            dense_prefix_steps=4,
            dense_suffix_steps=1,
            dense_prefix_layers=2,
            dense_suffix_layers=3,
            debug_route_density=True,
        )


if __name__ == "__main__":
    unittest.main()
