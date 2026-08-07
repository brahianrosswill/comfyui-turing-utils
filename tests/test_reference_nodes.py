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
        "width": 0,
        "height": 0,
        "upscale_method": "bilinear",
        "keep_proportion": "stretch",
        "pad_color": "0, 0, 0",
        "crop_position": "center",
        "divisible_by": 1,
        "device": "cpu",
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
    def test_reference_hubs_place_previous_before_ordered_autogrow_slots(self):
        cases = (
            (reference_nodes.ReferenceImageHub, "images", "image_"),
            (reference_nodes.ReferenceAudioHub, "audios", "audio_"),
            (reference_nodes.ReferenceVideoHub, "videos", "video_"),
        )
        for hub, autogrow_id, prefix in cases:
            with self.subTest(hub=hub.__name__):
                inputs = hub.define_schema().inputs
                self.assertEqual([item.id for item in inputs[:2]], ["previous", autogrow_id])
                self.assertEqual(inputs[1].template.names[:3], [f"{prefix}{index}" for index in range(3)])

    def test_image_and_video_hubs_match_kj_resize_v2_controls_and_defaults(self):
        expected_ids = [
            "width",
            "height",
            "upscale_method",
            "keep_proportion",
            "pad_color",
            "crop_position",
            "divisible_by",
            "device",
        ]
        for hub in (reference_nodes.ReferenceImageHub, reference_nodes.ReferenceVideoHub):
            with self.subTest(hub=hub.__name__):
                spatial = hub.define_schema().inputs[2:10]
                self.assertEqual([item.id for item in spatial], expected_ids)
                self.assertEqual(spatial[0].default, 512)
                self.assertEqual(spatial[1].default, 512)
                self.assertEqual(spatial[2].default, "nearest-exact")
                self.assertEqual(
                    spatial[2].options,
                    ["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "nvidia_rtx_vsr"],
                )
                self.assertEqual(spatial[3].default, "stretch")
                self.assertEqual(
                    spatial[3].options,
                    ["stretch", "resize", "pad", "pad_edge", "pad_edge_pixel", "crop", "pillarbox_blur", "total_pixels"],
                )
                self.assertEqual(spatial[4].default, "0, 0, 0")
                self.assertEqual(spatial[5].default, "center")
                self.assertEqual(spatial[6].default, 2)
                self.assertEqual(spatial[7].default, "cpu")

    def test_video_hub_uses_end_aligned_trim_and_pad_modes(self):
        frame_inputs = reference_nodes.ReferenceVideoHub.define_schema().inputs[10:]
        self.assertEqual(
            [item.id for item in frame_inputs],
            ["frame_align_enabled", "frame_count_mode", "frame_count", "short_video_fill"],
        )
        self.assertEqual(frame_inputs[1].options, ["trim", "pad", "specified"])
        self.assertEqual(frame_inputs[1].default, "trim")

    def test_image_hubs_chain_without_batching_different_sizes(self):
        first = reference_nodes.ReferenceImageHub.execute(
            images={"image_0": torch.zeros(1, 8, 12, 3)},
            **_spatial_kwargs(),
        )[0]
        second = reference_nodes.ReferenceImageHub.execute(
            images={"image_0": torch.ones(1, 10, 6, 3)},
            previous=first,
            **_spatial_kwargs(width=16, height=16, keep_proportion="pad"),
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
            width=4,
            height=4,
            keep_proportion="stretch",
            upscale_method="area",
            divisible_by=1,
        )

        _, resized = reference_nodes._spatial_transform(image, options, mask)
        self.assertGreater(float(resized.sum()), 0.0)

    def test_kj_resize_v2_modes_produce_matching_output_geometry(self):
        image = torch.linspace(0, 1, 10 * 6 * 3).reshape(1, 10, 6, 3)
        expected_shapes = {
            "stretch": (1, 16, 16, 3),
            "resize": (1, 16, 10, 3),
            "pad": (1, 16, 16, 3),
            "pad_edge": (1, 16, 16, 3),
            "pad_edge_pixel": (1, 16, 16, 3),
            "crop": (1, 16, 16, 3),
            "pillarbox_blur": (1, 16, 16, 3),
            "total_pixels": (1, 20, 12, 3),
        }
        for mode, expected in expected_shapes.items():
            with self.subTest(mode=mode):
                output = reference_nodes._spatial_transform(
                    image,
                    reference_nodes.SpatialOptions(
                        width=16,
                        height=16,
                        upscale_method="bilinear",
                        keep_proportion=mode,
                        divisible_by=2,
                    ),
                )
                self.assertEqual(tuple(output.shape), expected)

    def test_video_frame_alignment_trim_and_pad(self):
        short = torch.ones(3, 4, 4, 3)
        long = torch.full((5, 4, 4, 3), 2.0)
        identity = reference_nodes.SpatialOptions(width=0, height=0, divisible_by=1)
        minimum = reference_nodes.VideoReferenceSet(
            (short, long),
            reference_nodes.VideoOptions(
                spatial=identity,
                align_frames=True,
                frame_count_mode="trim",
            ),
        ).materialize()
        self.assertEqual([video.shape[0] for video in minimum], [3, 3])
        self.assertTrue(torch.equal(minimum[1], long[:3]))

        maximum = reference_nodes.VideoReferenceSet(
            (short, long),
            reference_nodes.VideoOptions(
                spatial=identity,
                align_frames=True,
                frame_count_mode="pad",
                short_video_fill="black",
            ),
        ).materialize()
        self.assertEqual([video.shape[0] for video in maximum], [5, 5])
        self.assertTrue(torch.equal(maximum[0][:3], short))
        self.assertEqual(float(maximum[0][-1].sum()), 0.0)

        numbered = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1).repeat(1, 2, 2, 3)
        specified = reference_nodes.VideoOptions(
            spatial=identity,
            align_frames=True,
            frame_count_mode="specified",
            frame_count=3,
        )
        trimmed = reference_nodes.VideoReferenceSet((numbered,), specified).materialize()[0]
        self.assertTrue(torch.equal(trimmed, numbered[:3]))

    def test_spatial_resize_and_frame_alignment_are_independent(self):
        video = torch.ones(3, 8, 12, 3)
        spatial_only = reference_nodes.VideoReferenceSet(
            (video,),
            reference_nodes.VideoOptions(
                spatial=reference_nodes.SpatialOptions(
                    width=4,
                    height=4,
                    keep_proportion="stretch",
                    upscale_method="area",
                    divisible_by=1,
                ),
                align_frames=False,
                frame_count_mode="specified",
                frame_count=7,
            ),
        ).materialize()[0]
        self.assertEqual(tuple(spatial_only.shape), (3, 4, 4, 3))

        temporal_only = reference_nodes.VideoReferenceSet(
            (video,),
            reference_nodes.VideoOptions(
                spatial=reference_nodes.SpatialOptions(width=0, height=0, divisible_by=1),
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
