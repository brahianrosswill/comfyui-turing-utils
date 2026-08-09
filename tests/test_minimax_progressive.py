from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import comfy.conds  # noqa: E402
import comfy.patcher_extension  # noqa: E402
import comfy.utils  # noqa: E402
import minimax_adapter  # noqa: E402


class FakePatcher:
    def __init__(self):
        self.wrappers = {}

    def clone(self):
        clone = FakePatcher()
        clone.wrappers = {
            wrapper_type: {
                key: values.copy() for key, values in keyed.items()
            }
            for wrapper_type, keyed in self.wrappers.items()
        }
        return clone

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)


def _h3_shapes():
    return [
        torch.Size((1, 24, 2, 8, 12)),
        torch.Size((1, 32, 2, 5)),
    ]


def _packed_h3_latent():
    shapes = _h3_shapes()
    video = torch.arange(torch.tensor(shapes[0]).prod(), dtype=torch.float32).reshape(shapes[0])
    audio = torch.arange(torch.tensor(shapes[1]).prod(), dtype=torch.float32).reshape(shapes[1])
    packed, _ = comfy.utils.pack_latents((video, audio))
    return packed, shapes, audio


def _conds(shapes, payload=None):
    model_conds = {"latent_shapes": comfy.conds.CONDConstant(shapes)}
    if payload is not None:
        model_conds["minimax_payload"] = comfy.conds.CONDConstant(payload)
    return [[{"model_conds": model_conds, "uuid": "test"}]]


class MiniMaxProgressiveResolutionTest(unittest.TestCase):
    def test_patch_clones_model_and_installs_only_model_wrappers(self):
        original = FakePatcher()
        patched = minimax_adapter.apply_h3_progressive_resolution_patch(original)

        self.assertIsNot(patched, original)
        self.assertEqual(original.wrappers, {})
        self.assertIn(
            minimax_adapter._PROGRESSIVE_OUTER_WRAPPER_KEY,
            patched.wrappers[comfy.patcher_extension.WrappersMP.OUTER_SAMPLE],
        )
        self.assertIn(
            minimax_adapter._PROGRESSIVE_COND_WRAPPER_KEY,
            patched.wrappers[comfy.patcher_extension.WrappersMP.CALC_COND_BATCH],
        )

    def test_first_model_evaluation_is_low_resolution_and_audio_is_unchanged(self):
        config = minimax_adapter._H3ProgressiveResolutionConfig(
            low_short_edge=64,
            low_resolution_steps=1,
            medium_short_edge=96,
            medium_resolution_steps=0,
            input_downscale="nearest-exact",
            output_upscale="bilinear",
            visual_condition_policy="keep_original",
        )
        outer_wrapper, cond_wrapper = minimax_adapter._make_h3_progressive_wrappers(config)
        packed, shapes, original_audio = _packed_h3_latent()
        conds = _conds(shapes)
        seen_shapes = []
        base_model = SimpleNamespace()

        def calc_executor(model, current_conds, current_x, timestep, model_options):
            current_shapes = minimax_adapter._h3_latent_shapes(current_conds)
            seen_shapes.append([tuple(shape) for shape in current_shapes])
            if len(seen_shapes) == 1:
                context = getattr(model, minimax_adapter._MEMORY_CONTEXT_ATTR)
                self.assertEqual([tuple(shape) for shape in context["latent_shapes"]], seen_shapes[0])
            return [current_x]

        def sample_executor(noise, latent, sampler, sigmas, mask, callback, disable, seed, **kwargs):
            first = cond_wrapper(calc_executor, base_model, conds, packed, torch.ones(1), {})[0]
            first_streams = comfy.utils.unpack_latents(first, shapes)
            self.assertEqual(tuple(first.shape), tuple(packed.shape))
            self.assertTrue(torch.equal(first_streams[1], original_audio))

            callback(0, first, first, 2)
            cond_wrapper(calc_executor, base_model, conds, packed, torch.ones(1), {})
            return first

        sigmas = torch.tensor([1.0, 0.6, 0.3, 0.0])
        outer_wrapper(
            sample_executor,
            packed,
            packed,
            object(),
            sigmas,
            None,
            None,
            True,
            1,
            latent_shapes=shapes,
        )

        self.assertEqual(seen_shapes[0][0], (1, 24, 2, 4, 6))
        self.assertEqual(seen_shapes[0][1], tuple(shapes[1]))
        self.assertEqual(seen_shapes[1], [tuple(shape) for shape in shapes])
        self.assertFalse(hasattr(base_model, minimax_adapter._MEMORY_CONTEXT_ATTR))

    def test_keyframe_latent_is_resized_without_reencoding_conditioning(self):
        config = minimax_adapter._H3ProgressiveResolutionConfig(
            low_short_edge=64,
            low_resolution_steps=1,
            medium_short_edge=96,
            medium_resolution_steps=0,
            input_downscale="area",
            output_upscale="bilinear",
            visual_condition_policy="resize_keyframes",
        )
        outer_wrapper, cond_wrapper = minimax_adapter._make_h3_progressive_wrappers(config)
        packed, shapes, _ = _packed_h3_latent()
        keyframe = torch.randn(1, 24, 1, 8, 12)
        payload = {
            "keyframes": [{"latent": keyframe, "resolved_frame_index": 0}],
            "cond_video_latents": [keyframe],
            "layout": object(),
        }
        conds = _conds(shapes, payload)
        seen_payloads = []

        def calc_executor(model, current_conds, current_x, timestep, model_options):
            payload_cond = current_conds[0][0]["model_conds"]["minimax_payload"]
            seen_payloads.append(payload_cond.cond)
            return [current_x]

        def sample_executor(noise, latent, sampler, sigmas, mask, callback, disable, seed, **kwargs):
            return cond_wrapper(calc_executor, None, conds, packed, torch.ones(1), {})[0]

        outer_wrapper(
            sample_executor,
            packed,
            packed,
            object(),
            torch.tensor([1.0, 0.6, 0.3, 0.0]),
            None,
            None,
            True,
            1,
            latent_shapes=shapes,
        )

        resized = seen_payloads[0]["keyframes"][0]["latent"]
        self.assertEqual(tuple(resized.shape[-2:]), (4, 6))
        self.assertIs(seen_payloads[0]["cond_video_latents"][0], resized)
        self.assertNotIn("layout", seen_payloads[0])
        self.assertEqual(tuple(keyframe.shape[-2:]), (8, 12))

    def test_stage_names_do_not_enforce_resolution_order(self):
        config = minimax_adapter._H3ProgressiveResolutionConfig(
            low_short_edge=64,
            low_resolution_steps=1,
            medium_short_edge=32,
            medium_resolution_steps=1,
            input_downscale="nearest-exact",
            output_upscale="bilinear",
            visual_condition_policy="keep_original",
        )
        outer_wrapper, cond_wrapper = minimax_adapter._make_h3_progressive_wrappers(config)
        packed, shapes, _ = _packed_h3_latent()
        conds = _conds(shapes)
        seen_video_shapes = []

        def calc_executor(model, current_conds, current_x, timestep, model_options):
            current_shapes = minimax_adapter._h3_latent_shapes(current_conds)
            seen_video_shapes.append(tuple(current_shapes[0]))
            return [current_x]

        def sample_executor(noise, latent, sampler, sigmas, mask, callback, disable, seed, **kwargs):
            for step in range(3):
                cond_wrapper(calc_executor, None, conds, packed, torch.ones(1), {})
                callback(step, packed, packed, 4)
            return packed

        outer_wrapper(
            sample_executor,
            packed,
            packed,
            object(),
            torch.tensor([1.0, 0.8, 0.6, 0.3, 0.0]),
            None,
            None,
            True,
            1,
            latent_shapes=shapes,
        )

        self.assertEqual(seen_video_shapes[0], (1, 24, 2, 4, 6))
        self.assertEqual(seen_video_shapes[1], (1, 24, 2, 2, 4))
        self.assertEqual(seen_video_shapes[2], tuple(shapes[0]))

    def test_stage_at_or_above_final_short_edge_is_a_noop(self):
        config = minimax_adapter._H3ProgressiveResolutionConfig(
            low_short_edge=256,
            low_resolution_steps=1,
            medium_short_edge=128,
            medium_resolution_steps=1,
            input_downscale="nearest-exact",
            output_upscale="bilinear",
            visual_condition_policy="keep_original",
        )
        outer_wrapper, cond_wrapper = minimax_adapter._make_h3_progressive_wrappers(config)
        packed, shapes, _ = _packed_h3_latent()
        conds = _conds(shapes)
        seen_x = []

        def calc_executor(model, current_conds, current_x, timestep, model_options):
            seen_x.append(current_x)
            return [current_x]

        def sample_executor(noise, latent, sampler, sigmas, mask, callback, disable, seed, **kwargs):
            first = cond_wrapper(calc_executor, None, conds, packed, torch.ones(1), {})[0]
            callback(0, first, first, 2)
            second = cond_wrapper(calc_executor, None, conds, packed, torch.ones(1), {})[0]
            return second

        result = outer_wrapper(
            sample_executor,
            packed,
            packed,
            object(),
            torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
            None,
            None,
            True,
            1,
            latent_shapes=shapes,
        )

        self.assertIs(seen_x[0], packed)
        self.assertIs(result, packed)

    def test_flow_transfer_preserves_high_state_when_low_velocity_is_zero(self):
        config = minimax_adapter._H3ProgressiveResolutionConfig(
            low_short_edge=64,
            low_resolution_steps=1,
            medium_short_edge=96,
            medium_resolution_steps=0,
            input_downscale="area",
            output_upscale="bilinear",
            visual_condition_policy="keep_original",
        )
        outer_wrapper, cond_wrapper = minimax_adapter._make_h3_progressive_wrappers(config)
        packed, shapes, _ = _packed_h3_latent()
        conds = _conds(shapes)

        def calc_executor(model, current_conds, current_x, timestep, model_options):
            # D_low == X_low represents a zero flow velocity.
            return [current_x]

        def sample_executor(noise, latent, sampler, sigmas, mask, callback, disable, seed, **kwargs):
            return cond_wrapper(calc_executor, None, conds, packed, torch.ones(1), {})[0]

        result = outer_wrapper(
            sample_executor,
            packed,
            packed,
            object(),
            torch.tensor([1.0, 0.6, 0.3, 0.0]),
            None,
            None,
            True,
            1,
            latent_shapes=shapes,
        )

        self.assertTrue(torch.equal(result, packed))

    def test_all_steps_may_remain_staged_for_diagnostics(self):
        config = minimax_adapter._H3ProgressiveResolutionConfig(
            low_short_edge=64,
            low_resolution_steps=6,
            medium_short_edge=96,
            medium_resolution_steps=0,
            input_downscale="area",
            output_upscale="bilinear",
            visual_condition_policy="keep_original",
        )
        outer_wrapper, cond_wrapper = minimax_adapter._make_h3_progressive_wrappers(config)
        packed, shapes, _ = _packed_h3_latent()
        conds = _conds(shapes)
        seen_video_shapes = []

        def calc_executor(model, current_conds, current_x, timestep, model_options):
            current_shapes = minimax_adapter._h3_latent_shapes(current_conds)
            seen_video_shapes.append(tuple(current_shapes[0]))
            return [current_x]

        def sample_executor(noise, latent, sampler, sigmas, mask, callback, disable, seed, **kwargs):
            for step in range(6):
                cond_wrapper(calc_executor, None, conds, packed, torch.ones(1), {})
                callback(step, packed, packed, 6)
            return packed

        outer_wrapper(
            sample_executor,
            packed,
            packed,
            object(),
            torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0]),
            None,
            None,
            True,
            1,
            latent_shapes=shapes,
        )

        self.assertEqual(seen_video_shapes, [(1, 24, 2, 4, 6)] * 6)

    def test_low_resolution_memory_plan_tracks_target_and_resized_keyframe_rows(self):
        high_shapes = _h3_shapes()
        low_shapes = [
            torch.Size((1, 24, 2, 4, 6)),
            high_shapes[1],
        ]
        old_plan = minimax_adapter._MiniMaxMemoryShape(
            200,
            full_rows=102,
            target_rows=58,
            target_visual_rows=48,
            target_audio_rows=10,
            visual_condition_rows=24,
            audio_condition_rows=0,
            hidden_size=5376,
            video_row_width=96,
            audio_row_width=32,
        )
        resized = minimax_adapter._resize_h3_memory_condition(
            minimax_adapter._MiniMaxMemoryCond(old_plan),
            high_shapes,
            low_shapes,
            resized_keyframes=1,
        ).cond

        self.assertEqual(resized.target_visual_rows, 12)
        self.assertEqual(resized.target_audio_rows, 10)
        self.assertEqual(resized.visual_condition_rows, 6)
        self.assertEqual(resized.target_rows, 22)
        self.assertEqual(resized.full_rows, 48)


if __name__ == "__main__":
    unittest.main()
