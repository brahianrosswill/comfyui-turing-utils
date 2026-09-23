from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.nodes.video_roi import (  # noqa: E402
    VideoMaskGuidedCrop,
    VideoMaskGuidedStitch,
    crop_video_by_mask,
    stitch_video_crops,
)


def _crop(images, masks, **overrides):
    options = {
        "width": 32,
        "height": 32,
        "context_scale": 1.0,
        "missing_mode": "interpolate",
        "smooth_window": 1,
        "mask_threshold": 0.5,
    }
    options.update(overrides)
    return crop_video_by_mask(images, masks, **options)


class VideoMaskRoiTest(unittest.TestCase):
    def test_schemas_keep_crop_info_last_and_share_one_custom_type(self):
        crop = VideoMaskGuidedCrop.define_schema()
        stitch = VideoMaskGuidedStitch.define_schema()
        self.assertEqual(crop.node_id, "TuringUtilsVideoMaskGuidedCrop")
        self.assertEqual(stitch.node_id, "TuringUtilsVideoMaskGuidedStitch")
        self.assertEqual([item.id for item in crop.outputs], ["images", "masks", "crop_info"])
        self.assertEqual(crop.outputs[-1].io_type, stitch.inputs[3].io_type)
        self.assertEqual([item.id for item in stitch.inputs[:4]], [
            "base_images",
            "cropped_images",
            "cropped_masks",
            "crop_info",
        ])

    def test_crop_uses_requested_ratio_and_moves_inside_source_frame(self):
        images = torch.ones(1, 40, 80, 3)
        masks = torch.zeros(1, 40, 80)
        masks[:, 15:25, 70:78] = 1.0

        crops, crop_masks, info = _crop(
            images,
            masks,
            width=32,
            height=16,
            context_scale=2.0,
        )

        self.assertEqual(tuple(crops.shape), (1, 16, 32, 3))
        self.assertEqual(tuple(crop_masks.shape), (1, 16, 32))
        x, y, width, height = info["boxes"][0]
        self.assertAlmostEqual(width / height, 2.0, places=6)
        self.assertGreaterEqual(x, 0.0)
        self.assertGreaterEqual(y, 0.0)
        self.assertLessEqual(x + width, 80.0)
        self.assertLessEqual(y + height, 40.0)
        self.assertAlmostEqual(x + width, 80.0, places=6)
        self.assertTrue(torch.allclose(crops, torch.ones_like(crops)))

    def test_impossible_mask_extent_uses_largest_in_frame_ratio_without_padding(self):
        images = torch.ones(1, 40, 20, 3)
        masks = torch.ones(1, 40, 20)
        crops, _, info = _crop(images, masks, width=64, height=16, context_scale=3.0)
        x, y, width, height = info["boxes"][0]
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(width, 20.0)
        self.assertAlmostEqual(height, 5.0)
        self.assertGreaterEqual(y, 0.0)
        self.assertLessEqual(y + height, 40.0)
        self.assertTrue(torch.allclose(crops, torch.ones_like(crops)))

    def test_interpolate_fills_center_gap_and_holds_sequence_edges(self):
        images = torch.zeros(5, 40, 100, 3)
        masks = torch.zeros(5, 40, 100)
        masks[1, 10:20, 10:20] = 1.0
        masks[3, 10:20, 50:60] = 1.0

        _, cropped_masks, info = _crop(images, masks)
        centers = [x + width * 0.5 for x, _y, width, _height in info["boxes"]]
        self.assertEqual(info["mask_valid"], [False, True, False, True, False])
        self.assertAlmostEqual(centers[0], 15.0)
        self.assertAlmostEqual(centers[1], 15.0)
        self.assertAlmostEqual(centers[2], 35.0)
        self.assertAlmostEqual(centers[3], 55.0)
        self.assertAlmostEqual(centers[4], 55.0)
        self.assertEqual(float(cropped_masks[0].max()), 0.0)
        self.assertEqual(float(cropped_masks[2].max()), 0.0)
        self.assertEqual(float(cropped_masks[4].max()), 0.0)

    def test_hold_keeps_last_box_until_a_new_observation(self):
        images = torch.zeros(3, 40, 100, 3)
        masks = torch.zeros(3, 40, 100)
        masks[0, 10:20, 10:20] = 1.0
        masks[2, 10:20, 50:60] = 1.0
        _, _, info = _crop(images, masks, missing_mode="hold")
        centers = [x + width * 0.5 for x, _y, width, _height in info["boxes"]]
        self.assertEqual(centers, [15.0, 15.0, 55.0])

    def test_stitch_uses_inverse_float_transform_and_only_composites_mask(self):
        base = torch.zeros(1, 32, 64, 3)
        masks = torch.zeros(1, 32, 64)
        masks[:, 12:20, 20:28] = 1.0
        crops, crop_masks, info = _crop(base, masks, width=16, height=16)
        generated = torch.ones_like(crops)

        output = stitch_video_crops(base, generated, crop_masks, info, feather=0)

        self.assertGreater(float(output[:, 13:19, 21:27].mean()), 0.99)
        outside = output.clone()
        outside[:, 12:20, 20:28] = 0.0
        self.assertLess(float(outside.abs().max()), 1e-5)

    def test_missing_frame_crop_stays_continuous_but_zero_mask_prevents_paste(self):
        base = torch.zeros(2, 32, 64, 3)
        masks = torch.zeros(2, 32, 64)
        masks[0, 12:20, 20:28] = 1.0
        crops, crop_masks, info = _crop(base, masks, width=16, height=16)
        output = stitch_video_crops(base, torch.ones_like(crops), crop_masks, info, feather=0)
        self.assertGreater(float(output[0].max()), 0.99)
        self.assertEqual(float(output[1].max()), 0.0)

    def test_all_empty_masks_and_short_generated_batch_fail_clearly(self):
        images = torch.zeros(2, 16, 16, 3)
        masks = torch.zeros(2, 16, 16)
        with self.assertRaisesRegex(ValueError, "No frame contains"):
            _crop(images, masks)

        masks[0, 4:8, 4:8] = 1.0
        crops, crop_masks, info = _crop(images, masks)
        with self.assertRaisesRegex(ValueError, "has 1 frames but 2 are required"):
            stitch_video_crops(images, crops[:1], crop_masks, info, feather=0)

    def test_node_execution_returns_comfy_node_outputs(self):
        images = torch.zeros(1, 16, 16, 3)
        masks = torch.zeros(1, 16, 16)
        masks[:, 4:12, 4:12] = 1.0
        cropped = VideoMaskGuidedCrop.execute(
            images=images,
            masks=masks,
            width=16,
            height=16,
            context_scale=1.0,
            missing_mode="interpolate",
            smooth_window=1,
            mask_threshold=0.5,
        )
        stitched = VideoMaskGuidedStitch.execute(
            base_images=images,
            cropped_images=cropped.result[0],
            cropped_masks=cropped.result[1],
            crop_info=cropped.result[2],
            feather=0,
        )
        self.assertEqual(tuple(stitched.result[0].shape), tuple(images.shape))


if __name__ == "__main__":
    unittest.main()
