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

import comfy.context_windows  # noqa: E402
import bernini_nodes  # noqa: E402


class FakeModel:
    def __init__(self):
        self.model_options = {}
        self.added_wrappers = []
        self.removed_wrappers = []

    def clone(self):
        clone = FakeModel()
        clone.model_options = dict(self.model_options)
        return clone

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.added_wrappers.append((wrapper_type, key, wrapper))

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.removed_wrappers.append((wrapper_type, key))


class BerniniContextWindowsTest(unittest.TestCase):
    def test_wan_frame_conversion_matches_official_node(self):
        self.assertEqual(bernini_nodes._validate_context_window_frames(81, 30), (21, 7))
        self.assertEqual(bernini_nodes._validate_context_window_frames(1, 0), (1, 0))

        with self.assertRaisesRegex(ValueError, "context_overlap"):
            bernini_nodes._validate_context_window_frames(5, 8)

    def test_input_order_matches_wan_context_window_node(self):
        input_names = list(bernini_nodes.BerniniContextWindowsCore.INPUT_TYPES()["required"])
        self.assertEqual(
            input_names,
            [
                "model",
                "context_length",
                "context_overlap",
                "context_schedule",
                "context_stride",
                "closed_loop",
                "fuse_method",
                "freenoise",
                "retain_first_frame",
                "split_conds_to_windows",
            ],
        )

    def test_apply_uses_official_wan_context_options(self):
        original_install = bernini_nodes._install_bernini_absolute_rope_patch
        bernini_nodes._install_bernini_absolute_rope_patch = lambda: None
        try:
            patched = bernini_nodes.BerniniContextWindowsCore().apply(
                FakeModel(),
                context_length=81,
                context_overlap=30,
                context_schedule=comfy.context_windows.ContextSchedules.UNIFORM_LOOPED,
                context_stride=2,
                closed_loop=True,
                fuse_method=comfy.context_windows.ContextFuseMethods.PYRAMID,
                freenoise=True,
                retain_first_frame=True,
                split_conds_to_windows=True,
            )[0]
        finally:
            bernini_nodes._install_bernini_absolute_rope_patch = original_install

        handler = patched.model_options["context_handler"]
        self.assertIsInstance(handler, bernini_nodes.BerniniScheduledContextHandler)
        self.assertEqual(handler.context_schedule.name, comfy.context_windows.ContextSchedules.UNIFORM_LOOPED)
        self.assertEqual(handler.context_length, 21)
        self.assertEqual(handler.context_overlap, 7)
        self.assertEqual(handler.context_stride, 2)
        self.assertIs(handler.closed_loop, True)
        self.assertEqual(handler.dim, 2)
        self.assertIs(handler.freenoise, True)
        self.assertEqual(handler.cond_retain_index_list, [0])
        self.assertEqual(handler.latent_retain_index_list, [])
        self.assertIs(handler.split_conds_to_windows, True)
        self.assertIs(handler.causal_window_fix, True)

    def test_context_windows_are_marked_for_absolute_rope(self):
        handler = bernini_nodes.BerniniScheduledContextHandler(
            context_schedule=comfy.context_windows.get_matching_context_schedule(
                comfy.context_windows.ContextSchedules.STATIC_STANDARD
            ),
            fuse_method=comfy.context_windows.get_matching_fuse_method(
                comfy.context_windows.ContextFuseMethods.PYRAMID
            ),
            context_length=3,
            context_overlap=1,
            dim=2,
            causal_window_fix=True,
        )

        windows = handler.get_context_windows(None, torch.zeros(1, 4, 8), {})
        self.assertTrue(windows)
        self.assertTrue(all(getattr(window, "turing_utils_use_absolute_indices", False) for window in windows))

    def test_rope_wrapper_includes_causal_anchor_index(self):
        window = comfy.context_windows.IndexListContextWindow([3, 4, 5], dim=2, total_frames=8)
        window.turing_utils_use_absolute_indices = True
        window.causal_anchor_index = 2
        transformer_options = {"context_window": window}
        captured = {}

        def executor(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "ok"

        result = bernini_nodes._bernini_context_rope_wrapper(
            executor,
            None,
            None,
            None,
            None,
            None,
            transformer_options,
        )

        self.assertEqual(result, "ok")
        patched_options = captured["args"][5]
        self.assertIsNot(patched_options, transformer_options)
        self.assertEqual(patched_options[bernini_nodes._ABSOLUTE_INDEX_KEY], (2, 3, 4, 5))

    def test_prepare_sampling_budgets_context_and_anchor_without_mutating_conds(self):
        handler = type(
            "Handler",
            (),
            {"dim": 2, "context_length": 3, "causal_window_fix": True},
        )()
        context = torch.zeros(1, 16, 8, 4, 4)
        conds = {"positive": [{"context_latents": [context]}]}
        captured = {}

        def executor(model, noise_shape, estimated_conds, *args, **kwargs):
            captured["noise_shape"] = noise_shape
            captured["conds"] = estimated_conds
            return model, estimated_conds, ["loaded"]

        result = bernini_nodes._bernini_prepare_sampling_wrapper(
            executor,
            "model",
            (1, 16, 8, 4, 4),
            conds,
            model_options={"context_handler": handler},
        )

        self.assertEqual(captured["noise_shape"], [1, 16, 4, 4, 4])
        estimated = captured["conds"]["positive"][0]["context_latents"][0]
        self.assertEqual(estimated.shape, (1, 16, 4, 4, 4))
        self.assertIs(result[1], conds)
        self.assertIs(conds["positive"][0]["context_latents"][0], context)

    def test_prepare_sampling_slices_matching_video_and_keeps_independent_refs(self):
        handler = SimpleNamespace(dim=2, context_length=3, causal_window_fix=True)
        source = torch.zeros(1, 16, 8, 4, 4)
        independent_ref = torch.zeros(1, 16, 3, 6, 6)
        conds = {
            "positive": [
                {"context_latents": [source, independent_ref]}
            ]
        }
        captured = {}

        def executor(model, noise_shape, estimated_conds, *args, **kwargs):
            captured["conds"] = estimated_conds
            return model, estimated_conds, []

        bernini_nodes._bernini_prepare_sampling_wrapper(
            executor,
            object(),
            (1, 16, 8, 4, 4),
            conds,
            model_options={"context_handler": handler},
        )
        estimated = captured["conds"]["positive"][0]["context_latents"]
        self.assertEqual(estimated[0].shape, (1, 16, 4, 4, 4))
        self.assertIs(estimated[1], independent_ref)
        self.assertEqual(source.shape, (1, 16, 8, 4, 4))

    def test_prepare_sampling_keeps_packed_latent_estimate_conservative(self):
        handler = SimpleNamespace(dim=2, context_length=8, causal_window_fix=True)
        conds = {"positive": []}
        calls = []

        def executor(_model, noise_shape, passed_conds, *args, **kwargs):
            calls.append((noise_shape, passed_conds))
            return "model", passed_conds, (), 0, 0

        result = bernini_nodes._bernini_prepare_sampling_wrapper(
            executor,
            object(),
            [1, 1, 64],
            conds,
            model_options={"context_handler": handler},
        )

        self.assertEqual(calls[0][0], [1, 1, 64])
        self.assertIs(calls[0][1], conds)
        self.assertIs(result[1], conds)


if __name__ == "__main__":
    unittest.main()
