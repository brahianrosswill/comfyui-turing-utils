from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.adapters.minimax.conditioning import (  # noqa: E402
    repair_combined_minimax_payload,
)
from comfyui_turing_utils.nodes.minimax_references import (  # noqa: E402
    H3BuildConditioning,
    H3FirstLastFrameReference,
    H3FrameReferenceData,
    H3ImageReference,
    H3ImageReferenceData,
    H3LatentInfo,
    H3ReferenceManifest,
    H3SemanticReference,
    H3SemanticReferenceData,
    H3VideoReference,
)


class _FakeVideoVAE:
    def __init__(self):
        self.inputs = []

    def encode(self, pixels):
        self.inputs.append(pixels)
        frames, height, width = pixels.shape[:3]
        latent_t = 1 if frames == 1 else ((frames - 5) // 17) * 5 + 2
        return torch.zeros(1, 24, latent_t, height // 16, width // 16)


class _FakeClip:
    def __init__(self):
        self.calls = []
        self.encoded = None

    def tokenize(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if kwargs.get("images"):
            entries = [("keyframe", index) for index, _ in enumerate(kwargs["images"])]
        elif kwargs.get("minimax_ref_items"):
            entries = [("reference", item["type"]) for item in kwargs["minimax_ref_items"]]
        else:
            entries = []
        if prompt:
            entries.append(("prompt", prompt))
        return {"qwen3vl_32b": [entries]}

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded = tokens
        return [[torch.zeros(1, 1, 1), {"semantic": True}]]


class _FakeAudioVAE:
    audio_sample_rate = 32000

    def __init__(self):
        self.inputs = []

    def encode(self, waveform):
        self.inputs.append(waveform)
        return torch.zeros(1, 32, 2, 10)


class _PayloadHolder:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return _PayloadHolder(cond)


class MiniMaxH3ReferencesTest(unittest.TestCase):
    def test_schema_uses_short_latent_name_and_separate_frame_ports(self):
        frame = H3FirstLastFrameReference.define_schema()
        semantic = H3SemanticReference.define_schema()
        build = H3BuildConditioning.define_schema()

        self.assertEqual([item.id for item in frame.inputs], [
            "vae", "latent", "first_frame", "last_frame",
        ])
        self.assertEqual([item.id for item in frame.outputs], ["first_frame", "last_frame"])
        self.assertEqual([item.id for item in semantic.inputs][2:4], ["first_frame", "last_frame"])
        self.assertEqual([item.id for item in build.inputs][1:4], ["latent", "first_frame", "last_frame"])

    def test_frame_reference_aligns_to_target_latent_canvas(self):
        vae = _FakeVideoVAE()
        latent = {"samples": torch.zeros(1, 24, 2, 6, 8)}
        image = torch.rand(1, 90, 70, 3)

        first, last = H3FirstLastFrameReference.execute(
            vae, latent=latent, first_frame=image
        ).result

        self.assertIsNone(last)
        self.assertEqual(first.role, "first_frame")
        self.assertEqual(tuple(first.image.shape), (1, 96, 128, 3))
        self.assertEqual(tuple(first.latent.shape), (1, 24, 1, 6, 8))

    def test_image_reference_without_latent_center_crops_to_32_pixel_grid(self):
        vae = _FakeVideoVAE()
        image = torch.rand(1, 67, 99, 3)

        reference = H3ImageReference.execute(
            vae, images={"image_0": image}
        ).result[0]

        self.assertEqual(tuple(reference.items[0]["image"].shape), (1, 64, 96, 3))
        self.assertEqual(tuple(reference.items[0]["latent"].shape), (1, 24, 1, 4, 6))

    def test_video_reference_normalizes_fps_and_pairs_audio_by_index(self):
        video_vae = _FakeVideoVAE()
        audio_vae = _FakeAudioVAE()
        frames = torch.rand(31, 70, 100, 3)
        audio = {
            "waveform": torch.rand(1, 2, 32000),
            "sample_rate": 32000,
        }

        reference = H3VideoReference.execute(
            video_vae,
            30.0,
            audio_vae=audio_vae,
            videos={"video_2": frames},
            video_audios={"video_audio_2": audio},
        ).result[0]
        item = reference.items[0]

        self.assertEqual(tuple(video_vae.inputs[0].shape), (22, 64, 96, 3))
        self.assertEqual(tuple(item["latent"].shape), (1, 24, 7, 4, 6))
        self.assertEqual(tuple(item["qwen_frames"].shape), (2, 64, 96, 3))
        self.assertEqual(item["timestamps"], [0.0, 0.5])
        self.assertEqual(tuple(item["audio_latent"].shape), (1, 32, 2, 10))

    def test_semantic_combines_official_keyframe_and_reference_presentations(self):
        clip = _FakeClip()
        first = H3FrameReferenceData(
            "first_frame",
            torch.rand(1, 64, 64, 3),
            torch.zeros(1, 24, 1, 4, 4),
        )
        images = H3ImageReferenceData(({
            "image": torch.rand(1, 32, 32, 3),
            "latent": torch.zeros(1, 24, 1, 2, 2),
        },))

        semantic = H3SemanticReference.execute(
            clip,
            "prompt",
            first_frame=first,
            image_reference=images,
        ).result[0]

        self.assertEqual(len(clip.calls), 2)
        self.assertEqual(clip.calls[0][0], "")
        self.assertIn("images", clip.calls[0][1])
        self.assertEqual(clip.calls[1][0], "prompt")
        self.assertIn("minimax_ref_items", clip.calls[1][1])
        self.assertEqual(
            clip.encoded["qwen3vl_32b"][0],
            [("keyframe", 0), ("reference", "image"), ("prompt", "prompt")],
        )
        self.assertEqual(
            semantic.manifest,
            H3ReferenceManifest(first_frame=True, image_count=1),
        )

    def test_build_conditioning_places_keyframe_and_generic_reference_together(self):
        target = {"samples": torch.zeros(1, 24, 7, 6, 8)}
        first = H3FrameReferenceData(
            "first_frame",
            torch.rand(1, 96, 128, 3),
            torch.zeros(1, 24, 1, 6, 8),
        )
        images = H3ImageReferenceData(({
            "image": torch.rand(1, 64, 64, 3),
            "latent": torch.zeros(1, 24, 1, 4, 4),
        },))
        base = [[torch.zeros(1, 2, 3), {"kept": True}]]
        semantic = H3SemanticReferenceData(
            base,
            H3ReferenceManifest(first_frame=True, image_count=1),
        )

        conditioning = H3BuildConditioning.execute(
            semantic,
            target,
            first_frame=first,
            image_reference=images,
        ).result[0]
        options = conditioning[0][1]

        self.assertEqual(options["minimax_frame_count"], 22)
        self.assertEqual(options["minimax_keyframes"][0]["resolved_frame_index"], 0)
        self.assertIs(options["minimax_keyframes"][0]["latent"], first.latent)
        self.assertEqual(options["minimax_refs"][0]["kind"], "image")
        self.assertIs(options["minimax_refs"][0]["latent"], images.items[0]["latent"])
        self.assertTrue(options["kept"])
        self.assertNotIn("minimax_keyframes", base[0][1])

    def test_build_rejects_different_semantic_structure(self):
        semantic = H3SemanticReferenceData(
            [[torch.zeros(1, 1, 1), {}]],
            H3ReferenceManifest(image_count=1),
        )
        with self.assertRaisesRegex(ValueError, "structures differ"):
            H3BuildConditioning.execute(
                semantic,
                {"samples": torch.zeros(1, 24, 2, 4, 4)},
            )

    def test_latent_info_uses_h3_temporal_grid(self):
        result = H3LatentInfo.execute(
            {"samples": torch.zeros(1, 24, 37, 45, 84)}
        ).result
        self.assertEqual(result, (1344, 720, 124, 24.0))

    def test_combined_payload_repair_matches_packed_layout_order(self):
        keyframe = torch.zeros(1, 24, 1, 4, 4)
        image = torch.ones(1, 24, 1, 2, 2)
        audio = torch.ones(1, 32, 2, 5)
        out = {"minimax_payload": _PayloadHolder({"cond_video_latents": [image]})}
        repaired = repair_combined_minimax_payload(out, {
            "minimax_keyframes": [{"latent": keyframe}],
            "minimax_refs": [
                {"kind": "image", "latent": image},
                {"kind": "audio", "audio_latent": audio},
            ],
        })

        payload = repaired["minimax_payload"].cond
        self.assertIs(payload["cond_video_latents"][0], keyframe)
        self.assertIs(payload["cond_video_latents"][1], image)
        self.assertIs(payload["cond_audio_latents"][0], audio)
        self.assertIsNot(repaired, out)


if __name__ == "__main__":
    unittest.main()
