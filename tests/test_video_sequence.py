from __future__ import annotations

import sys
import tempfile
import unittest
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))


from comfyui_turing_utils.nodes import video_sequence as nodes  # noqa: E402


class VideoSequenceTest(unittest.TestCase):
    def test_schemas_use_stable_ids_and_no_preview_outputs(self):
        expected = {
            nodes.LoadIndexedVideoSegment: "TuringUtilsLoadIndexedVideoSegment",
            nodes.SaveIndexedVideoSegment: "TuringUtilsSaveIndexedVideoSegment",
            nodes.VideoContinuationConcat: "TuringUtilsVideoContinuationConcat",
            nodes.TrimVideoContinuationPrefix: "TuringUtilsTrimVideoContinuationPrefix",
            nodes.H3SetAudioPrefixNoiseMask: "TuringUtilsH3SetAudioPrefixNoiseMask",
        }
        for node, node_id in expected.items():
            with self.subTest(node=node.__name__):
                schema = node.define_schema()
                self.assertEqual(schema.node_id, node_id)
                self.assertNotIn("preview", [output.id for output in schema.outputs])

    def test_segment_paths_are_six_digit_and_confined_to_output(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            nodes.folder_paths, "get_output_directory", return_value=directory
        ):
            path = nodes._segment_path("series/a", 324, create=True)
            self.assertEqual(path, Path(directory) / "series" / "a" / "000324.mp4")
            self.assertTrue(path.parent.is_dir())
            with self.assertRaisesRegex(ValueError, "inside"):
                nodes._segment_path("../outside", 0)
            with self.assertRaisesRegex(ValueError, "relative"):
                nodes._segment_path("/outside", 0)

    def test_missing_segment_returns_empty_without_preview(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            nodes.folder_paths, "get_output_directory", return_value=directory
        ):
            output = nodes.LoadIndexedVideoSegment.execute("segments", 8, 21)
            self.assertEqual(output.result, (None, None, 0.0))
            self.assertIsNone(output.ui)

    def test_loader_keeps_tail_frames_and_matching_audio(self):
        images = torch.arange(10, dtype=torch.float32)[:, None, None, None].expand(10, 2, 2, 3)
        audio = {"waveform": torch.arange(20, dtype=torch.float32).reshape(1, 1, 20), "sample_rate": 48}

        class FakeVideo:
            def get_frame_rate(self):
                return Fraction(24, 1)

            def get_frame_count(self):
                return 10

            def as_trimmed(self, start_time, duration, strict_duration):
                self.trim = (start_time, duration, strict_duration)
                return self

            def get_components(self):
                return SimpleNamespace(images=images, audio=audio)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            nodes.folder_paths, "get_output_directory", return_value=directory
        ), mock.patch.object(nodes.InputImpl, "VideoFromFile", return_value=FakeVideo()):
            target = Path(directory) / "segments" / "000001.mp4"
            target.parent.mkdir()
            target.write_bytes(b"video")
            output_images, output_audio, frame_rate = nodes.LoadIndexedVideoSegment.execute(
                "segments", 1, 5
            ).result
        torch.testing.assert_close(output_images, images[-5:])
        torch.testing.assert_close(output_audio["waveform"], audio["waveform"][..., -10:])
        self.assertEqual(output_audio["sample_rate"], 48)
        self.assertEqual(frame_rate, 24.0)

    def test_saver_is_atomic_honors_overwrite_and_has_no_preview(self):
        class FakeVideo:
            def save_to(self, path, **kwargs):
                Path(path).write_bytes(b"encoded")

        images = torch.zeros(2, 4, 6, 3)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            nodes.folder_paths, "get_output_directory", return_value=directory
        ), mock.patch.object(nodes.InputImpl, "VideoFromComponents", return_value=FakeVideo()):
            result = nodes.SaveIndexedVideoSegment.execute(
                images, "segments", 12, 24.0, False
            )
            target = Path(result.result[0])
            self.assertEqual(target.name, "000012.mp4")
            self.assertEqual(target.read_bytes(), b"encoded")
            self.assertFalse(any(target.parent.glob(".*.tmp.mp4")))
            self.assertIsNone(result.ui)
            with self.assertRaises(FileExistsError):
                nodes.SaveIndexedVideoSegment.execute(images, "segments", 12, 24.0, False)
            nodes.SaveIndexedVideoSegment.execute(images, "segments", 12, 24.0, True)

    def test_concat_builds_default_masks_and_aligned_audio(self):
        prefix_images = torch.full((9, 8, 8, 3), 0.5)
        body_images = torch.full((4, 8, 8, 3), 0.25)
        prefix_audio = {
            "waveform": torch.arange(9, dtype=torch.float32).reshape(1, 1, 9),
            "sample_rate": 24,
        }
        images, mask, audio, info = nodes.VideoContinuationConcat.execute(
            body_images,
            24.0,
            False,
            0.08,
            prefix_images=prefix_images,
            prefix_audio=prefix_audio,
        ).result
        torch.testing.assert_close(images[:9], prefix_images)
        torch.testing.assert_close(images[9:], body_images)
        self.assertEqual(mask.shape, (13, 8, 8))
        self.assertEqual(mask[:9].count_nonzero().item(), 0)
        self.assertTrue(torch.all(mask[9:] == 1))
        self.assertEqual(audio["sample_rate"], 24)
        self.assertEqual(tuple(audio["waveform"].shape), (1, 1, 13))
        torch.testing.assert_close(audio["waveform"][..., :9], prefix_audio["waveform"])
        self.assertEqual(audio["waveform"][..., 9:].count_nonzero().item(), 0)
        self.assertEqual(info["prefix_audio_samples"], 9)
        self.assertEqual(info["total_audio_samples"], 13)
        self.assertTrue(info["prefix_audio_present"])

    def test_concat_accepts_lazy_audio_mappings(self):
        class LazyAudio(Mapping):
            def __init__(self, value):
                self.value = value

            def __getitem__(self, key):
                return self.value[key]

            def __iter__(self):
                return iter(self.value)

            def __len__(self):
                return len(self.value)

        body = torch.zeros(4, 8, 8, 3)
        lazy = LazyAudio({"waveform": torch.ones(1, 2, 4), "sample_rate": 24})
        audio = nodes.VideoContinuationConcat.execute(
            body, 24.0, False, 0.0, body_audio=lazy
        ).result[2]
        torch.testing.assert_close(audio["waveform"], lazy["waveform"])

    def test_concat_chroma_noise_fades_to_five_clean_frames(self):
        prefix = torch.full((9, 32, 32, 3), 0.5)
        body = torch.full((3, 32, 32, 3), 0.25)
        images, _, _, _ = nodes.VideoContinuationConcat.execute(
            body,
            24.0,
            True,
            0.1,
            noise_seed=7,
            prefix_images=prefix,
        ).result
        self.assertFalse(torch.equal(images[:4], prefix[:4]))
        torch.testing.assert_close(images[4:9], prefix[4:9], rtol=0, atol=0)
        torch.testing.assert_close(images[9:], body, rtol=0, atol=0)

    def test_concat_requires_prefix_images_for_prefix_side_data(self):
        body = torch.zeros(3, 8, 8, 3)
        with self.assertRaisesRegex(ValueError, "require prefix_images"):
            nodes.VideoContinuationConcat.execute(
                body, 24.0, False, 0.0, prefix_audio={"waveform": torch.zeros(1, 2, 4), "sample_rate": 24}
            )

    def test_trim_uses_exact_frame_rate_for_images_and_audio(self):
        info = {
            "version": 1,
            "prefix_frames": 5,
            "body_frames": 4,
            "frame_rate_numerator": 24,
            "frame_rate_denominator": 1,
            "prefix_audio_samples": 10,
            "total_audio_samples": 18,
            "audio_sample_rate": 48,
        }
        images = torch.arange(9, dtype=torch.float32)[:, None, None, None].expand(9, 2, 2, 3)
        waveform = torch.arange(18, dtype=torch.float32).reshape(1, 1, 18)
        output_images, output_audio = nodes.TrimVideoContinuationPrefix.execute(
            images, info, {"waveform": waveform, "sample_rate": 48}
        ).result
        torch.testing.assert_close(output_images, images[5:])
        torch.testing.assert_close(output_audio["waveform"], waveform[..., 10:])

    def test_h3_audio_mask_maps_waveform_boundary_to_latent_time(self):
        info = {
            "version": 1,
            "prefix_frames": 9,
            "body_frames": 4,
            "frame_rate_numerator": 24,
            "frame_rate_denominator": 1,
            "prefix_audio_samples": 9,
            "total_audio_samples": 13,
            "audio_sample_rate": 24,
        }
        samples = torch.randn(1, 32, 2, 13, dtype=torch.float16)
        metadata = object()
        latent = {"samples": samples, "metadata": metadata}
        output = nodes.H3SetAudioPrefixNoiseMask.execute(
            latent, info, "protect_prefix_generate_body"
        ).result[0]
        self.assertIs(output["samples"], samples)
        self.assertIs(output["metadata"], metadata)
        self.assertEqual(output["noise_mask"].dtype, torch.float32)
        self.assertEqual(output["noise_mask"][..., :9].count_nonzero().item(), 0)
        self.assertTrue(torch.all(output["noise_mask"][..., 9:] == 1))
        self.assertNotIn("noise_mask", latent)

        protected = nodes.H3SetAudioPrefixNoiseMask.execute(latent, info, "protect_all").result[0]
        generated = nodes.H3SetAudioPrefixNoiseMask.execute(latent, info, "generate_all").result[0]
        self.assertEqual(protected["noise_mask"].count_nonzero().item(), 0)
        self.assertTrue(torch.all(generated["noise_mask"] == 1))

    def test_h3_audio_mask_regenerates_silent_prefix_when_no_audio_was_supplied(self):
        prefix = torch.zeros(9, 8, 8, 3)
        body = torch.zeros(4, 8, 8, 3)
        _, _, _, info = nodes.VideoContinuationConcat.execute(
            body, 24.0, False, 0.0, prefix_images=prefix
        ).result
        latent = {"samples": torch.zeros(1, 32, 2, 13)}
        output = nodes.H3SetAudioPrefixNoiseMask.execute(
            latent, info, "protect_prefix_generate_body"
        ).result[0]
        self.assertFalse(info["prefix_audio_present"])
        self.assertTrue(torch.all(output["noise_mask"] == 1))


if __name__ == "__main__":
    unittest.main()
