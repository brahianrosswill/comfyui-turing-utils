from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.adapters import sec  # noqa: E402
from comfyui_turing_utils.nodes.sec import (  # noqa: E402
    SeCModelLoader,
    SeCModelType,
    SeCTrackVisualConcept,
)


class _FakePredictor:
    def __init__(self):
        self.seeds = []
        self.reset_count = 0

    def init_state(self, video_path, offload_video_to_cpu, offload_state_to_cpu):
        self.init_args = (video_path, offload_video_to_cpu, offload_state_to_cpu)
        return {
            "video_height": int(video_path.shape[1]),
            "video_width": int(video_path.shape[2]),
            "num_frames": int(video_path.shape[0]),
        }

    def reset_state(self, state):
        self.reset_count += 1

    def add_new_mask(self, *, inference_state, frame_idx, obj_id, mask):
        self.seeds.append(("mask", frame_idx, obj_id, mask.copy()))
        logits = torch.from_numpy(mask.astype(np.float32))[None, None]
        return frame_idx, [obj_id], logits

    def add_new_points_or_box(
        self,
        *,
        inference_state,
        frame_idx,
        obj_id,
        points,
        labels,
        box,
    ):
        self.seeds.append(("points", frame_idx, obj_id, points, labels, box))
        height = inference_state["video_height"]
        width = inference_state["video_width"]
        logits = torch.ones(1, 1, height, width)
        return frame_idx, [obj_id], logits


class _FakeModel:
    def __init__(self, predictor, frame_count, height, width):
        self.grounding_encoder = predictor
        self.frame_count = frame_count
        self.height = height
        self.width = width
        self.calls = []

    def propagate_in_video(
        self,
        state,
        *,
        start_frame_idx,
        max_frame_num_to_track,
        reverse,
        init_mask,
        mllm_memory_size,
    ):
        self.calls.append((start_frame_idx, max_frame_num_to_track, reverse, mllm_memory_size))
        indexes = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, self.frame_count)
        for index in indexes:
            logits = torch.full((1, 1, self.height, self.width), 1.0 if index % 2 == 0 else -1.0)
            yield index, [1], logits


class _FakeHandle(sec.SeCModelHandle):
    @property
    def model(self):
        return self._fake_model


class SeCNodeTest(unittest.TestCase):
    def test_model_discovery_accepts_sec_files_and_hf_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sam2.safetensors").touch()
            (root / "SeC-4B-fp16.safetensors").touch()
            hf_model = root / "nested" / "sec-model"
            hf_model.mkdir(parents=True)
            for filename in (
                "config.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
            ):
                (hf_model / filename).touch()

            with mock.patch.object(sec, "_model_roots", return_value=(root,)):
                names = [spec.name for spec in sec.available_sec_models()]

        self.assertEqual(
            names,
            ["nested/sec-model/", "SeC-4B-fp16.safetensors"],
        )

    def test_fp8_conversion_reuses_the_state_dict(self):
        state_dict = {
            "weight": torch.zeros(4, dtype=torch.float8_e4m3fn),
            "index": torch.arange(4),
        }

        converted = sec._convert_float8_state_dict(state_dict)

        self.assertIs(converted, state_dict)
        self.assertEqual(converted["weight"].dtype, torch.float16)
        self.assertEqual(converted["index"].dtype, torch.int64)

    def test_parameter_dtype_hooks_are_scoped_and_preserve_integer_inputs(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2).half()
                self.embedding = torch.nn.Embedding(4, 2).half()

        model = TinyModel()
        count = sec._register_parameter_dtype_input_hooks(model)

        linear_output = model.linear(torch.ones(1, 2, dtype=torch.float32))
        embedding_output = model.embedding(torch.tensor([1], dtype=torch.int64))

        self.assertEqual(count, 2)
        self.assertEqual(linear_output.dtype, torch.float16)
        self.assertEqual(embedding_output.dtype, torch.float16)
        self.assertEqual(len(model._sec_dtype_input_hook_handles), 2)

    def test_schema_removes_manual_device_and_unload_controls(self):
        loader = SeCModelLoader.define_schema()
        tracker = SeCTrackVisualConcept.define_schema()

        self.assertEqual(loader.node_id, "TuringUtilsSeCModelLoader")
        self.assertEqual(tracker.node_id, "TuringUtilsSeCTrackVisualConcept")
        self.assertEqual(SeCModelType.io_type, "TURING_UTILS_SEC_MODEL")
        loader_inputs = [item.id for item in loader.inputs]
        tracker_inputs = [item.id for item in tracker.inputs]
        self.assertNotIn("device", loader_inputs)
        self.assertNotIn("offload_video_to_cpu", tracker_inputs)
        self.assertNotIn("auto_unload_model", tracker_inputs)
        self.assertEqual([item.id for item in tracker.outputs], ["masks"])

    def test_bbox_and_points_form_one_combined_prompt(self):
        frames = torch.zeros(3, 12, 16, 3)
        prompt = sec.prepare_visual_prompt(
            frames,
            annotation_frame_idx=0,
            positive_points='[{"x": 4, "y": 5}]',
            negative_points='[{"x": 12, "y": 9}]',
            bbox=[{"startX": 2, "startY": 3, "endX": 14, "endY": 11}],
            input_mask=None,
        )

        self.assertIsNone(prompt.mask)
        np.testing.assert_array_equal(prompt.points, [[4, 5], [12, 9]])
        np.testing.assert_array_equal(prompt.labels, [1, 0])
        np.testing.assert_array_equal(prompt.box, [2, 3, 14, 11])

    def test_mask_is_authoritative_and_selects_annotation_frame(self):
        frames = torch.zeros(3, 12, 16, 3)
        masks = torch.zeros(3, 12, 16)
        masks[1, 3:10, 4:13] = 1
        prompt = sec.prepare_visual_prompt(
            frames,
            annotation_frame_idx=1,
            positive_points='[{"x": 6, "y": 5}]',
            negative_points='[{"x": 1, "y": 1}]',
            bbox=[{"startX": 3, "startY": 2, "endX": 14, "endY": 11}],
            input_mask=masks,
        )

        self.assertIsNotNone(prompt.mask)
        self.assertEqual(int(prompt.mask.sum()), 7 * 9)
        self.assertIsNone(prompt.points)
        self.assertIsNone(prompt.box)

    def test_mask_rejects_conflicting_clicks(self):
        frames = torch.zeros(2, 8, 8, 3)
        mask = torch.zeros(1, 8, 8)
        mask[:, 2:6, 2:6] = 1
        with self.assertRaisesRegex(ValueError, "Positive point .* outside"):
            sec.prepare_visual_prompt(
                frames,
                annotation_frame_idx=0,
                positive_points='[{"x": 0, "y": 0}]',
                negative_points="",
                bbox=None,
                input_mask=mask,
            )
        with self.assertRaisesRegex(ValueError, "Negative point .* inside"):
            sec.prepare_visual_prompt(
                frames,
                annotation_frame_idx=0,
                positive_points="",
                negative_points='[{"x": 3, "y": 3}]',
                bbox=None,
                input_mask=mask,
            )

    @mock.patch.object(sec.comfy.model_management, "throw_exception_if_processing_interrupted")
    @mock.patch.object(sec.comfy.model_management, "intermediate_device", return_value=torch.device("cpu"))
    @mock.patch.object(sec.comfy.model_management, "load_models_gpu")
    def test_tracking_uses_comfy_lifecycle_and_cleans_state(
        self,
        load_models_gpu,
        _intermediate_device,
        _interrupt,
    ):
        frames = torch.zeros(4, 8, 10, 3)
        predictor = _FakePredictor()
        model = _FakeModel(predictor, 4, 8, 10)
        patcher = mock.Mock()
        handle = _FakeHandle(patcher=patcher, dtype=torch.float16, source="fake")
        handle._fake_model = model

        masks = sec.track_visual_concept(
            handle,
            frames,
            positive_points='[{"x": 4, "y": 4}]',
            negative_points="",
            bbox=None,
            tracking_direction="bidirectional",
            annotation_frame_idx=1,
            object_id=7,
            max_frames_to_track=-1,
            mllm_memory_size=6,
        )

        self.assertEqual(tuple(masks.shape), (4, 8, 10))
        load_models_gpu.assert_called_once()
        kwargs = load_models_gpu.call_args.kwargs
        self.assertTrue(kwargs["force_full_load"])
        self.assertGreater(kwargs["memory_required"], 0)
        self.assertEqual(predictor.init_args[1:], (True, True))
        self.assertEqual([seed[0] for seed in predictor.seeds], ["points", "points"])
        self.assertEqual([call[2] for call in model.calls], [False, True])
        self.assertGreaterEqual(predictor.reset_count, 3)


if __name__ == "__main__":
    unittest.main()
