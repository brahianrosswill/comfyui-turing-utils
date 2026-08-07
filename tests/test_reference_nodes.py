from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import bernini_nodes  # noqa: E402
import minimax_nodes  # noqa: E402
import reference_nodes  # noqa: E402


def _spatial_kwargs(**overrides):
    values = {
        "resize_enabled": False,
        "width": 0,
        "height": 0,
        "resize_mode": "fill",
        "upscale_method": "bilinear",
        "crop_position": "center",
        "divisible_by": 1,
        "pad_color": "0, 0, 0",
    }
    values.update(overrides)
    return values


class FakeVAE:
    def __init__(self):
        self.inputs = []

    def encode(self, image):
        self.inputs.append(image)
        return torch.zeros(
            1,
            16,
            ((image.shape[0] - 1) // 4) + 1,
            image.shape[1] // 8,
            image.shape[2] // 8,
        )


class ReferenceNodesTest(unittest.TestCase):
    def test_image_hubs_chain_without_batching_different_sizes(self):
        first = reference_nodes.ReferenceImageHub.execute(
            images={"image_0": torch.zeros(1, 8, 12, 3)},
            **_spatial_kwargs(),
        )[0]
        second = reference_nodes.ReferenceImageHub.execute(
            images={"image_0": torch.ones(1, 10, 6, 3)},
            previous=first,
            **_spatial_kwargs(resize_enabled=True, width=16, height=16, resize_mode="fit"),
        )[0]

        self.assertEqual(len(second.items), 2)
        materialized = second.materialize()
        self.assertEqual([tuple(item.shape) for item in materialized], [(1, 16, 16, 3), (1, 16, 16, 3)])
        self.assertEqual(float(materialized[0].sum()), 0.0)
        self.assertGreater(float(materialized[1].sum()), 0.0)

    def test_conservative_mask_resize_does_not_drop_thin_region(self):
        image = torch.zeros(1, 16, 16, 3)
        mask = torch.zeros(1, 16, 16)
        mask[0, 7, 7] = 1.0
        options = reference_nodes.SpatialOptions(
            enabled=True,
            width=4,
            height=4,
            mode="stretch",
            method="area",
        )

        _, resized = reference_nodes._spatial_transform(image, options, mask)
        self.assertGreater(float(resized.sum()), 0.0)

    def test_video_frame_alignment_minimum_and_maximum(self):
        short = torch.ones(3, 4, 4, 3)
        long = torch.full((5, 4, 4, 3), 2.0)
        minimum = reference_nodes.VideoReferenceSet(
            (short, long),
            reference_nodes.VideoOptions(align_frames=True, frame_count_mode="minimum"),
        ).materialize()
        self.assertEqual([video.shape[0] for video in minimum], [3, 3])

        maximum = reference_nodes.VideoReferenceSet(
            (short, long),
            reference_nodes.VideoOptions(
                align_frames=True,
                frame_count_mode="maximum",
                short_video_fill="black",
            ),
        ).materialize()
        self.assertEqual([video.shape[0] for video in maximum], [5, 5])
        self.assertEqual(float(maximum[0][-1].sum()), 0.0)

    def test_spatial_resize_and_frame_alignment_are_independent(self):
        video = torch.ones(3, 8, 12, 3)
        spatial_only = reference_nodes.VideoReferenceSet(
            (video,),
            reference_nodes.VideoOptions(
                spatial=reference_nodes.SpatialOptions(enabled=True, width=4, height=4, mode="stretch", method="area"),
                align_frames=False,
                frame_count_mode="specified",
                frame_count=7,
            ),
        ).materialize()[0]
        self.assertEqual(tuple(spatial_only.shape), (3, 4, 4, 3))

        temporal_only = reference_nodes.VideoReferenceSet(
            (video,),
            reference_nodes.VideoOptions(
                spatial=reference_nodes.SpatialOptions(enabled=False, width=4, height=4),
                align_frames=True,
                frame_count_mode="specified",
                frame_count=5,
            ),
        ).materialize()[0]
        self.assertEqual(tuple(temporal_only.shape), (5, 8, 12, 3))

    def test_bernini_inpaint_uses_source_latent_and_upper_bound_mask(self):
        vae = FakeVAE()
        source = torch.zeros(5, 16, 16, 3)
        mask = torch.zeros(5, 16, 16)
        mask[2, 7, 7] = 1.0
        positive = [[torch.zeros(1), {}]]
        negative = [[torch.zeros(1), {}]]

        output = bernini_nodes.BerniniInpaintCondition.execute(
            positive,
            negative,
            vae,
            source,
            16,
            16,
            5,
            1,
            source_as_context=True,
            mask=mask,
        )
        latent = output[2]
        self.assertEqual(tuple(latent["samples"].shape), (1, 16, 2, 2, 2))
        self.assertEqual(tuple(latent["noise_mask"].shape), (1, 1, 2, 2, 2))
        self.assertGreater(float(latent["noise_mask"].sum()), 0.0)
        self.assertIs(output[0][0][1]["context_latents"][0], latent["samples"])
        self.assertEqual(
            output[0][0][1][bernini_nodes._CONTEXT_ROLES_KEY],
            ("aligned",),
        )

    def test_bernini_global_repaint_omits_noise_mask_and_source_context(self):
        output = bernini_nodes.BerniniInpaintCondition.execute(
            [[torch.zeros(1), {}]],
            [[torch.zeros(1), {}]],
            FakeVAE(),
            torch.zeros(5, 16, 16, 3),
            16,
            16,
            5,
            1,
        )
        self.assertNotIn("noise_mask", output[2])
        self.assertNotIn("context_latents", output[0][0][1])

    def test_h3_padding_uses_17n_plus_5_grid(self):
        image = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1, 1).repeat(1, 2, 2, 3)
        mask = torch.zeros(6, 2, 2)
        mask[-1] = 1.0
        padded = minimax_nodes.MiniMaxH3VideoFramesPadding().pad(image, 0, mask)
        self.assertEqual(padded[4], 22)
        self.assertEqual(padded[5], 6)
        self.assertTrue(torch.equal(padded[0][-1], image[-1]))
        self.assertTrue(torch.equal(padded[1][-1], mask[-1]))

    def test_h3_hub_delegates_native_order_and_audio_binding(self):
        images = reference_nodes.ImageReferenceSet((torch.zeros(1, 8, 8, 3),))
        videos = reference_nodes.VideoReferenceSet((torch.zeros(5, 8, 8, 3),))
        audios = reference_nodes.AudioReferenceSet(({"waveform": "paired"}, {"waveform": "standalone"}))
        sentinel = object()
        with mock.patch.object(
            minimax_nodes.MiniMaxH3ReferenceToVideo,
            "execute",
            return_value=sentinel,
        ) as execute:
            result = minimax_nodes.MiniMaxH3ReferenceConditionHub.execute(
                "clip",
                "vae",
                "audio_vae",
                "prompt",
                1344,
                768,
                124,
                audio_binding="pair_by_index",
                image_references=images,
                video_references=videos,
                audio_references=audios,
            )

        self.assertIs(result, sentinel)
        kwargs = execute.call_args.kwargs
        self.assertEqual(list(kwargs["ref_images"]), ["ref_image_0"])
        self.assertEqual(list(kwargs["ref_videos"]), ["ref_video_0"])
        self.assertEqual(list(kwargs["ref_video_audios"]), ["ref_video_audio_0"])
        self.assertEqual(list(kwargs["ref_audios"]), ["ref_audio_0"])


if __name__ == "__main__":
    unittest.main()
