from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
CUSTOM_NODES_ROOT = PLUGIN_ROOT.parent
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(CUSTOM_NODES_ROOT))

import comfy.nested_tensor  # noqa: E402


minimax_nodes = importlib.import_module("comfyui-turing-utils.minimax_nodes")


def _h3_latent():
    video = torch.randn(1, 24, 3, 8, 12)
    audio = torch.randn(1, 32, 2, 7)
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "batch_index": [4],
    }, video, audio


class MiniMaxH3LatentResizeTest(unittest.TestCase):
    def test_schema_uses_exact_32_pixel_dimensions(self):
        inputs = minimax_nodes.MiniMaxH3LatentResize.INPUT_TYPES()["required"]
        self.assertEqual(
            tuple(inputs),
            ("samples", "width", "height", "resize_method", "resize_keyframes"),
        )
        self.assertEqual(inputs["width"][1]["step"], 32)
        self.assertEqual(inputs["height"][1]["step"], 32)
        self.assertEqual(inputs["resize_method"][1]["default"], "bilinear")
        self.assertIn("bislerp", inputs["resize_method"][0])
        self.assertTrue(inputs["resize_keyframes"][1]["default"])
        self.assertIn("conditioning", minimax_nodes.MiniMaxH3LatentResize.INPUT_TYPES()["optional"])

    def test_resizes_video_independently_and_preserves_audio(self):
        latent, original_video, original_audio = _h3_latent()
        output, conditioning = minimax_nodes.MiniMaxH3LatentResize().resize(
            latent,
            width=320,
            height=64,
            resize_method="bilinear",
        )
        resized_video, resized_audio = output["samples"].unbind()

        self.assertEqual(tuple(resized_video.shape), (1, 24, 3, 4, 20))
        self.assertIs(resized_audio, original_audio)
        self.assertEqual(tuple(original_video.shape), (1, 24, 3, 8, 12))
        self.assertEqual(output["batch_index"], [4])
        self.assertIsNot(output, latent)
        self.assertIsNone(conditioning)

    def test_same_geometry_is_a_video_noop(self):
        latent, original_video, original_audio = _h3_latent()
        output, conditioning = minimax_nodes.MiniMaxH3LatentResize().resize(
            latent,
            width=192,
            height=128,
            resize_method="area",
        )
        video, audio = output["samples"].unbind()
        self.assertIs(video, original_video)
        self.assertIs(audio, original_audio)
        self.assertIsNone(conditioning)

    def test_optionally_resizes_h3_keyframes_without_mutating_conditioning(self):
        latent, _, _ = _h3_latent()
        keyframe_latent = torch.randn(1, 24, 1, 8, 12)
        metadata = {
            "minimax_keyframes": [
                {"latent": keyframe_latent, "resolved_frame_index": 0},
            ],
            "other": "preserved",
        }
        conditioning = [[torch.randn(1, 2, 3), metadata]]

        _, resized_conditioning = minimax_nodes.MiniMaxH3LatentResize().resize(
            latent,
            width=320,
            height=64,
            resize_method="bilinear",
            resize_keyframes=True,
            conditioning=conditioning,
        )

        resized = resized_conditioning[0][1]["minimax_keyframes"][0]["latent"]
        self.assertEqual(tuple(resized.shape), (1, 24, 1, 4, 20))
        self.assertIs(conditioning[0][1], metadata)
        self.assertIs(conditioning[0][1]["minimax_keyframes"][0]["latent"], keyframe_latent)
        self.assertEqual(resized_conditioning[0][1]["other"], "preserved")

    def test_keyframe_resize_switch_can_pass_conditioning_through(self):
        latent, _, _ = _h3_latent()
        conditioning = [[torch.zeros(1, 1, 1), {"minimax_keyframes": []}]]
        _, output_conditioning = minimax_nodes.MiniMaxH3LatentResize().resize(
            latent,
            width=320,
            height=64,
            resize_keyframes=False,
            conditioning=conditioning,
        )
        self.assertIs(output_conditioning, conditioning)

    def test_rejects_unaligned_dimensions(self):
        latent, _, _ = _h3_latent()
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            minimax_nodes.MiniMaxH3LatentResize().resize(
                latent,
                width=720,
                height=480,
            )

    def test_rejects_non_h3_latent(self):
        latent = {"samples": torch.zeros(1, 4, 8, 8)}
        with self.assertRaisesRegex(ValueError, "nested video/audio"):
            minimax_nodes.MiniMaxH3LatentResize().resize(
                latent,
                width=128,
                height=128,
            )


if __name__ == "__main__":
    unittest.main()
