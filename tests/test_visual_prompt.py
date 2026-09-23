from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.nodes.visual_prompt import (  # noqa: E402
    MaskToVisualPrompts,
    visual_prompts_from_mask,
)


class VisualPromptTest(unittest.TestCase):
    def test_schema_exposes_common_prompt_contracts_and_index_last(self):
        schema = MaskToVisualPrompts.define_schema()
        self.assertEqual(schema.node_id, "TuringUtilsMaskToVisualPrompts")
        self.assertEqual(
            [output.id for output in schema.outputs],
            [
                "guidance",
                "mask",
                "positive_coords",
                "negative_coords",
                "bbox",
                "bounding_box",
                "index",
            ],
        )
        self.assertEqual(schema.outputs[4].io_type, "BBOX")
        self.assertEqual(schema.outputs[5].io_type, "BOUNDING_BOX")

    def test_selects_one_mask_and_builds_both_bbox_formats(self):
        masks = torch.zeros(3, 20, 30)
        masks[1, 5:15, 10:20] = 0.8
        output = MaskToVisualPrompts.execute(
            guidance=masks,
            index=1,
            mask_threshold=0.5,
            positive_point_count=3,
            negative_point_count=4,
            bbox_padding=0.1,
        ).result

        selected, mask, positive_json, negative_json, legacy, canonical, index = output
        self.assertEqual(tuple(selected.shape), (1, 20, 30))
        self.assertTrue(torch.equal(selected, mask))
        self.assertEqual(index, 1)
        self.assertEqual(legacy, [{"startX": 9, "startY": 4, "endX": 21, "endY": 16}])
        self.assertEqual(canonical, [[{"x": 9, "y": 4, "width": 12, "height": 12}]])

        positive = json.loads(positive_json)
        negative = json.loads(negative_json)
        self.assertEqual(len(positive), 3)
        self.assertEqual(len(negative), 4)
        for point in positive:
            self.assertGreaterEqual(point["x"], 10)
            self.assertLess(point["x"], 20)
            self.assertGreaterEqual(point["y"], 5)
            self.assertLess(point["y"], 15)
        for point in negative:
            self.assertEqual(bool(masks[1, point["y"], point["x"]] >= 0.5), False)

    def test_image_guidance_is_selected_and_converted_to_mask(self):
        images = torch.zeros(2, 8, 12, 3)
        images[1, 2:6, 4:9, 1] = 1.0
        output = MaskToVisualPrompts.execute(
            guidance=images,
            index=1,
            mask_threshold=0.5,
            positive_point_count=1,
            negative_point_count=0,
            bbox_padding=0.0,
        ).result
        self.assertEqual(tuple(output[0].shape), (1, 8, 12, 3))
        self.assertEqual(tuple(output[1].shape), (1, 8, 12))
        self.assertEqual(float(output[1].sum()), 20.0)
        self.assertEqual(json.loads(output[3]), [])

    def test_rejects_empty_guidance_and_bad_index(self):
        masks = torch.zeros(2, 8, 8)
        with self.assertRaisesRegex(ValueError, "no foreground"):
            visual_prompts_from_mask(masks[:1], 0.5, 1, 0, 0.0)
        with self.assertRaisesRegex(ValueError, "outside the guidance batch"):
            MaskToVisualPrompts.execute(
                guidance=masks,
                index=2,
                mask_threshold=0.5,
                positive_point_count=1,
                negative_point_count=0,
                bbox_padding=0.0,
            )


if __name__ == "__main__":
    unittest.main()
