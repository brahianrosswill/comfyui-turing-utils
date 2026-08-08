from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "kernel"))
sys.path.insert(0, str(PLUGIN_ROOT))

import attention as turing_attention  # noqa: E402


class TuringAttentionContractTest(unittest.TestCase):
    def test_fp32_uses_bf16_kernel_and_restores_fp32_output(self):
        q = torch.zeros((1, 1, 32, 128), dtype=torch.float32)
        kernel_output = torch.ones_like(q, dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sageattn", return_value=kernel_output) as sage,
        ):
            output = turing_attention.turing_sage_attention(
                mock.Mock(), q, q, q, 1,
                skip_reshape=True, skip_output_reshape=True
            )
        q_arg, k_arg, v_arg = sage.call_args.args[:3]
        self.assertEqual(q_arg.dtype, torch.bfloat16)
        self.assertEqual(k_arg.dtype, torch.bfloat16)
        self.assertEqual(v_arg.dtype, torch.bfloat16)
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, kernel_output.float())

    def test_unsupported_fp32_head_dimension_falls_back_without_casting(self):
        q = torch.zeros((1, 1, 32, 256), dtype=torch.float32)
        original = mock.Mock(return_value=q)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sageattn") as sage,
        ):
            output = turing_attention.turing_sage_attention(original, q, q, q, 1, skip_reshape=True)
        self.assertIs(output, q)
        original.assert_called_once()
        self.assertIs(original.call_args.args[0], q)
        sage.assert_not_called()

    def test_mask_uses_original_attention(self):
        q = torch.zeros((1, 1, 32, 128), dtype=torch.bfloat16)
        original = mock.Mock(return_value="fallback")
        with mock.patch("attention.is_supported_turing_device", return_value=True):
            output = turing_attention.turing_sage_attention(
                original, q, q, q, 1, mask=torch.ones(1), skip_reshape=True
            )
        self.assertEqual(output, "fallback")
        original.assert_called_once()

    def test_fp16_hnd_input_and_output_are_preserved(self):
        q = torch.zeros((1, 4, 32, 64), dtype=torch.float16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sageattn", return_value=q) as sage,
        ):
            output = turing_attention.turing_sage_attention(
                mock.Mock(), q, q[:, :2], q[:, :2], 4,
                skip_reshape=True, skip_output_reshape=True, enable_gqa=True
            )
        self.assertIs(output, q)
        self.assertEqual(sage.call_args.kwargs["tensor_layout"], "HND")

    def test_causal_and_scale_options_are_forwarded_to_stable_sage(self):
        q = torch.zeros((1, 4, 32, 64), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sageattn", return_value=q) as sage,
        ):
            turing_attention.turing_sage_attention(
                mock.Mock(),
                q,
                q,
                q,
                4,
                skip_reshape=True,
                is_causal=True,
                scale=0.125,
            )
        self.assertTrue(sage.call_args.kwargs["is_causal"])
        self.assertEqual(sage.call_args.kwargs["sm_scale"], 0.125)
        self.assertFalse(sage.call_args.kwargs["smooth_k"])
        self.assertNotIn("variant", sage.call_args.kwargs)

    def test_distinct_sequence_shapes_each_report_bundled_dispatch_once(self):
        q_short = torch.zeros((1, 4, 32, 64), dtype=torch.bfloat16)
        q_long = torch.zeros((1, 4, 96, 64), dtype=torch.bfloat16)
        turing_attention._LOGGED_TURING_KERNELS.clear()
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("attention._sageattn", side_effect=lambda q, *args, **kwargs: q),
            self.assertLogs("comfyui-turing-utils", level="INFO") as captured,
        ):
            turing_attention.turing_sage_attention(
                mock.Mock(), q_short, q_short, q_short, 4,
                skip_reshape=True, skip_output_reshape=True
            )
            turing_attention.turing_sage_attention(
                mock.Mock(), q_long, q_long, q_long, 4,
                skip_reshape=True, skip_output_reshape=True
            )
            turing_attention.turing_sage_attention(
                mock.Mock(), q_long, q_long, q_long, 4,
                skip_reshape=True, skip_output_reshape=True
            )

        dispatch_logs = [line for line in captured.output if "Bundled Turing Sage active" in line]
        self.assertEqual(len(dispatch_logs), 2)
        self.assertTrue(any("32" in line for line in dispatch_logs))
        self.assertTrue(any("96" in line for line in dispatch_logs))

    def test_bf16_unreshaped_gqa_keeps_compact_kv_heads(self):
        q = torch.zeros((1, 32, 4 * 64), dtype=torch.bfloat16)
        k = torch.zeros((1, 16, 2 * 64), dtype=torch.bfloat16)
        v = torch.zeros_like(k)
        kernel_output = torch.zeros((1, 32, 4, 64), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sageattn", return_value=kernel_output) as sage,
        ):
            output = turing_attention.turing_sage_attention(
                mock.Mock(), q, k, v, 4, enable_gqa=True
            )
        q_arg, k_arg, v_arg = sage.call_args.args[:3]
        self.assertEqual(q_arg.shape, (1, 32, 4, 64))
        self.assertEqual(k_arg.shape, (1, 16, 2, 64))
        self.assertEqual(v_arg.shape, (1, 16, 2, 64))
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(sage.call_args.kwargs["tensor_layout"], "NHD")

    def test_unsupported_head_dimension_uses_original_attention(self):
        q = torch.zeros((1, 1, 32, 256), dtype=torch.bfloat16)
        original = mock.Mock(return_value="fallback")
        with mock.patch("attention.is_supported_turing_device", return_value=True):
            output = turing_attention.turing_sage_attention(original, q, q, q, 1, skip_reshape=True)
        self.assertEqual(output, "fallback")

    def test_experimental_sparse_uses_kernel_for_generic_long_attention(self):
        q = torch.zeros((1, 4, 4096, 128), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sol_sparse_sageattn", return_value=q) as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(),
                q,
                q,
                q,
                4,
                skip_reshape=True,
                skip_output_reshape=True,
                min_sequence_tokens=4096,
            )
        self.assertIs(output, q)
        self.assertEqual(sparse.call_args.kwargs["prefix_tokens"], 0)
        self.assertEqual(sparse.call_args.kwargs["threshold_sigma"], 1.0)
        self.assertEqual(sparse.call_args.kwargs["local_block_radius"], 1)
        self.assertEqual(sparse.call_args.kwargs["temporal_neighbor_frames"], 1)
        self.assertEqual(sparse.call_args.kwargs["residual_subblocks"], 2)
        self.assertEqual(sparse.call_args.kwargs["minimum_route_density"], 0.0)
        self.assertEqual(sparse.call_args.kwargs["maximum_route_density"], 1.0)
        self.assertEqual(sparse.call_args.kwargs["topology_tokens"], 0)

    def test_experimental_sparse_keeps_short_calls_dense(self):
        q = torch.zeros((1, 4, 256, 128), dtype=torch.bfloat16)
        baseline = mock.Mock(return_value=q)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention.turing_sage_attention", baseline),
            mock.patch("attention._sol_sparse_sageattn") as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(), q, q, q, 4,
                skip_reshape=True,
            )
        self.assertIs(output, q)
        baseline.assert_called_once()
        sparse.assert_not_called()

    def test_frame_sparse_uses_video_topology_and_rotates_anchors_by_layer(self):
        q = torch.zeros((1, 4, 4096, 128), dtype=torch.bfloat16)
        turing_attention._LOGGED_FRAME_SPARSE_KERNELS.clear()
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch(
                "attention._frame_sparse_sageattn", return_value=(q, 0.125)
            ) as sparse,
        ):
            output = turing_attention.turing_frame_sparse_attention(
                mock.Mock(),
                q,
                q,
                q,
                4,
                skip_reshape=True,
                skip_output_reshape=True,
                temporal_window_frames=2,
                global_anchor_stride=12,
                rotate_global_anchors=True,
                sink_frames=1,
                transformer_options={
                    "turing_utils_attention_layout": {
                        "dense_prefix_tokens": 512,
                        "topology_start_tokens": 512,
                        "topology_tokens": 3584,
                        "tokens_per_frame": 256,
                        "spatial_tokens_height": 16,
                        "spatial_tokens_width": 16,
                        "layer_index": 15,
                        "layer_count": 50,
                    }
                },
            )

        self.assertIs(output, q)
        self.assertEqual(sparse.call_args.kwargs["prefix_tokens"], 512)
        self.assertEqual(sparse.call_args.kwargs["topology_start_tokens"], 512)
        self.assertEqual(sparse.call_args.kwargs["topology_tokens"], 3584)
        self.assertEqual(sparse.call_args.kwargs["tokens_per_frame"], 256)
        self.assertEqual(sparse.call_args.kwargs["spatial_tokens_height"], 16)
        self.assertEqual(sparse.call_args.kwargs["spatial_tokens_width"], 16)
        self.assertEqual(sparse.call_args.kwargs["temporal_window_frames"], 2)
        self.assertEqual(sparse.call_args.kwargs["global_anchor_stride"], 12)
        self.assertEqual(sparse.call_args.kwargs["global_anchor_offset"], 3)
        self.assertEqual(sparse.call_args.kwargs["sink_frames"], 1)
        self.assertTrue(sparse.call_args.kwargs["return_schedule_density"])

    def test_radial_frame_sparse_falls_back_without_spatial_topology(self):
        q = torch.zeros((1, 4, 4096, 128), dtype=torch.bfloat16)
        baseline = mock.Mock(return_value=q)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention.turing_sage_attention", baseline),
            mock.patch("attention._frame_sparse_sageattn") as sparse,
        ):
            output = turing_attention.turing_frame_sparse_attention(
                mock.Mock(),
                q,
                q,
                q,
                4,
                skip_reshape=True,
                sparse_pattern="radial",
                transformer_options={
                    "turing_utils_attention_layout": {
                        "topology_start_tokens": 512,
                        "topology_tokens": 3584,
                        "tokens_per_frame": 256,
                    }
                },
            )
        self.assertIs(output, q)
        baseline.assert_called_once()
        sparse.assert_not_called()

    def test_radial_sampling_phase_rotates_without_full_frame_anchors(self):
        q = torch.zeros((1, 4, 4096, 128), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch(
                "attention._frame_sparse_sageattn", return_value=(q, 0.1)
            ) as sparse,
        ):
            turing_attention.turing_frame_sparse_attention(
                mock.Mock(),
                q,
                q,
                q,
                4,
                skip_reshape=True,
                sparse_pattern="radial",
                global_anchor_stride=0,
                radial_max_temporal_stride=16,
                rotate_global_anchors=True,
                transformer_options={
                    "turing_utils_attention_layout": {
                        "topology_start_tokens": 512,
                        "topology_tokens": 3584,
                        "tokens_per_frame": 256,
                        "spatial_tokens_height": 16,
                        "spatial_tokens_width": 16,
                        "layer_index": 31,
                        "layer_count": 50,
                    }
                },
            )
        self.assertEqual(sparse.call_args.kwargs["global_anchor_offset"], 15)

    def test_frame_sparse_quality_profiles_override_only_sparse_policy(self):
        settings = turing_attention._resolve_frame_sparse_quality_profile(
            "fast",
            sparse_pattern="frame_window",
            temporal_window_frames=7,
            global_anchor_stride=3,
            rotate_global_anchors=False,
            sink_frames=4,
            radial_spatial_radius=3,
            radial_max_temporal_stride=4,
            dense_prefix_layers=9,
            dense_suffix_layers=9,
        )
        self.assertEqual(settings["sparse_pattern"], "radial")
        self.assertEqual(settings["temporal_window_frames"], 1)
        self.assertEqual(settings["global_anchor_stride"], 0)
        self.assertEqual(settings["radial_spatial_radius"], 0)
        self.assertEqual(settings["radial_max_temporal_stride"], 32)
        self.assertEqual(settings["dense_prefix_layers"], 1)

    def test_frame_sparse_without_video_tail_metadata_uses_stable_sage(self):
        q = torch.zeros((1, 4, 4096, 128), dtype=torch.bfloat16)
        baseline = mock.Mock(return_value=q)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention.turing_sage_attention", baseline),
            mock.patch("attention._frame_sparse_sageattn") as sparse,
        ):
            output = turing_attention.turing_frame_sparse_attention(
                mock.Mock(), q, q, q, 4, skip_reshape=True
            )
        self.assertIs(output, q)
        baseline.assert_called_once()
        sparse.assert_not_called()

    def test_experimental_sparse_keeps_cross_attention_with_prefix_dense(self):
        q = torch.zeros((1, 4, 4096, 128), dtype=torch.bfloat16)
        k = torch.zeros((1, 2, 4608, 128), dtype=torch.bfloat16)
        v = torch.zeros_like(k)
        baseline = mock.Mock(return_value=q)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention.turing_sage_attention", baseline),
            mock.patch("attention._sol_sparse_sageattn") as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(),
                q,
                k,
                v,
                4,
                skip_reshape=True,
                min_sequence_tokens=4096,
                transformer_options={
                    "turing_utils_attention_layout": {"dense_prefix_tokens": 320}
                },
            )
        self.assertIs(output, q)
        baseline.assert_called_once()
        sparse.assert_not_called()

    def test_experimental_sparse_debug_counts_route_only_once_per_shape(self):
        q = torch.zeros((1, 4, 4096, 128), dtype=torch.bfloat16)
        route = torch.zeros((1, 4, 64, 4), dtype=torch.int32)
        route_keys = set()
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch(
                "attention._sol_sparse_sageattn",
                side_effect=[(q, route), q],
            ) as sparse,
            mock.patch(
                "attention._sol_sparse_route_selected", return_value=1024
            ) as selected,
            self.assertLogs("comfyui-turing-utils", level="WARNING") as captured,
        ):
            for _ in range(2):
                output = turing_attention.turing_sol_sparse_attention(
                    mock.Mock(),
                    q,
                    q,
                    q,
                    4,
                    skip_reshape=True,
                    skip_output_reshape=True,
                    min_sequence_tokens=4096,
                    debug_route_density=True,
                    debug_route_keys=route_keys,
                )

        self.assertIs(output, q)
        self.assertEqual(sparse.call_count, 2)
        self.assertTrue(sparse.call_args_list[0].kwargs["return_route"])
        self.assertFalse(sparse.call_args_list[1].kwargs["return_route"])
        selected.assert_called_once_with(route)
        message = "\n".join(captured.output)
        self.assertIn("density=0.0625", message)
        self.assertIn("step=None/None layer=None/None", message)

    def test_experimental_sparse_debug_aggregates_once_per_step(self):
        q = torch.zeros((1, 4, 4096, 128), dtype=torch.bfloat16)
        route = torch.zeros((1, 4, 64, 4), dtype=torch.int32)
        route_state = {}
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch(
                "attention._sol_sparse_sageattn",
                side_effect=[(q, route), (q, route)],
            ) as sparse,
            mock.patch(
                "attention._sol_sparse_route_selected_device",
                side_effect=[torch.tensor([1024]), torch.tensor([2048])],
            ) as selected,
            self.assertLogs("comfyui-turing-utils", level="WARNING") as captured,
        ):
            for layer_index in (2, 3):
                output = turing_attention.turing_sol_sparse_attention(
                    mock.Mock(),
                    q,
                    q,
                    q,
                    4,
                    skip_reshape=True,
                    skip_output_reshape=True,
                    min_sequence_tokens=4096,
                    debug_route_density=True,
                    debug_route_state=route_state,
                    debug_context={
                        "step": 2,
                        "sampling_steps": 8,
                        "layer_index": layer_index,
                        "layer_count": 4,
                    },
                )

        self.assertIs(output, q)
        self.assertEqual(sparse.call_count, 2)
        self.assertTrue(all(call.kwargs["return_route"] for call in sparse.call_args_list))
        self.assertEqual(selected.call_count, 2)
        self.assertFalse(route_state)
        message = "\n".join(captured.output)
        self.assertIn("step=2/8 layers=2-3 calls=2", message)
        self.assertIn("density[min/mean/max]=0.0625/0.0938/0.1250", message)

    def test_experimental_sparse_unreshaped_gqa_restores_output_layout(self):
        q = torch.zeros((1, 4096, 4 * 128), dtype=torch.float16)
        k = torch.zeros((1, 4096, 2 * 128), dtype=torch.float16)
        v = torch.zeros_like(k)
        kernel_output = torch.zeros((1, 4, 4096, 128), dtype=torch.float16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sol_sparse_sageattn", return_value=kernel_output) as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(), q, k, v, 4, enable_gqa=True, min_sequence_tokens=4096
            )
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(sparse.call_args.args[1].shape, (1, 2, 4096, 128))

    def test_experimental_sparse_fallback_preserves_original_layout(self):
        q = torch.zeros((1, 4096, 4 * 64), dtype=torch.bfloat16)
        baseline = mock.Mock(return_value=q)
        with (
            mock.patch("attention.turing_sage_attention", baseline),
            mock.patch("attention._sol_sparse_sageattn") as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(), q, q, q, 4
            )
        self.assertIs(output, q)
        baseline.assert_called_once()
        self.assertIs(baseline.call_args.args[1], q)
        sparse.assert_not_called()

    def test_experimental_sparse_forwards_patch_parameters(self):
        q = torch.zeros((1, 4, 8192, 128), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sol_sparse_sageattn", return_value=q) as sparse,
        ):
            turing_attention.turing_sol_sparse_attention(
                mock.Mock(),
                q,
                q,
                q,
                4,
                skip_reshape=True,
                routing_threshold=0.82,
                prefix_policy="auto",
                manual_prefix_tokens=192,
                local_block_radius=3,
                skipped_residual="1x64",
                minimum_route_density=0.2,
                maximum_route_density=0.6,
                min_sequence_tokens=8000,
                transformer_options={
                    "turing_utils_attention_layout": {
                        "dense_prefix_tokens": 320,
                        "topology_start_tokens": 320,
                        "topology_tokens": 7680,
                        "tokens_per_frame": 960,
                    }
                },
            )
        self.assertEqual(sparse.call_args.kwargs["prefix_tokens"], 320)
        self.assertEqual(sparse.call_args.kwargs["threshold_sigma"], 0.82)
        self.assertEqual(sparse.call_args.kwargs["local_block_radius"], 3)
        self.assertEqual(sparse.call_args.kwargs["residual_subblocks"], 1)
        self.assertEqual(sparse.call_args.kwargs["minimum_route_density"], 0.2)
        self.assertEqual(sparse.call_args.kwargs["maximum_route_density"], 0.6)
        self.assertEqual(sparse.call_args.kwargs["topology_start_tokens"], 320)
        self.assertEqual(sparse.call_args.kwargs["topology_tokens"], 7680)
        self.assertEqual(sparse.call_args.kwargs["tokens_per_frame"], 960)

    def test_sparse_prefix_policy_never_guesses_a_generic_layout(self):
        options = {
            "turing_utils_attention_layout": {"dense_prefix_tokens": 640}
        }
        self.assertEqual(
            turing_attention._sparse_prefix_tokens("auto", 128, options, 4096),
            640,
        )
        self.assertEqual(
            turing_attention._sparse_prefix_tokens("none", 128, options, 4096),
            0,
        )
        self.assertEqual(
            turing_attention._sparse_prefix_tokens("manual", 128, options, 4096),
            128,
        )
        self.assertEqual(
            turing_attention._sparse_prefix_tokens("auto", 128, {}, 4096),
            0,
        )

    def test_sparse_dense_prefix_steps_use_sampling_progress(self):
        sample_sigmas = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
        state = {}
        self.assertTrue(
            turing_attention._sparse_dense_prefix_steps(
                {"sample_sigmas": sample_sigmas, "sigmas": sample_sigmas[0:1]},
                1,
                state,
            )
        )
        self.assertFalse(
            turing_attention._sparse_dense_prefix_steps(
                {"sample_sigmas": sample_sigmas, "sigmas": sample_sigmas[1:2]},
                1,
                state,
            )
        )

    def test_sparse_dense_prefix_steps_accept_inference_tensors(self):
        state = {}
        with torch.inference_mode():
            sample_sigmas = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
            current_sigmas = sample_sigmas[0:1]
            self.assertTrue(
                turing_attention._sparse_dense_prefix_steps(
                    {
                        "sample_sigmas": sample_sigmas,
                        "sigmas": current_sigmas,
                    },
                    1,
                    state,
                )
            )
            self.assertTrue(
                turing_attention._sparse_dense_prefix_steps(
                    {
                        "sample_sigmas": sample_sigmas,
                        "sigmas": current_sigmas,
                    },
                    1,
                    state,
                )
            )

    def test_sparse_dense_schedule_supports_a_tail_and_layer_metadata(self):
        sample_sigmas = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
        self.assertTrue(
            turing_attention._sparse_dense_schedule(
                {"sample_sigmas": sample_sigmas, "sigmas": sample_sigmas[3:4]},
                0,
                1,
                {},
            )
        )
        self.assertTrue(
            turing_attention._sparse_dense_layer(
                {
                    "turing_utils_attention_layout": {
                        "layer_index": 1,
                        "layer_count": 50,
                    }
                },
                2,
                2,
            )
        )
        self.assertFalse(
            turing_attention._sparse_dense_layer(
                {
                    "turing_utils_attention_layout": {
                        "layer_index": 2,
                        "layer_count": 50,
                    }
                },
                2,
                2,
            )
        )
        self.assertTrue(
            turing_attention._sparse_dense_layer(
                {
                    "turing_utils_attention_layout": {
                        "layer_index": 48,
                        "layer_count": 50,
                    }
                },
                2,
                2,
            )
        )

    def test_sparse_debug_tracks_step_when_dense_schedule_is_disabled(self):
        sample_sigmas = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
        state = {}
        self.assertFalse(
            turing_attention._sparse_dense_schedule(
                {"sample_sigmas": sample_sigmas, "sigmas": sample_sigmas[2:3]},
                0,
                0,
                state,
                track_step=True,
            )
        )
        self.assertEqual(state["step"], 2)
        self.assertEqual(state["sampling_steps"], 4)

    def test_experimental_sparse_fp32_uses_bf16_boundary_and_restores_output(self):
        q = torch.zeros((1, 1, 4096, 128), dtype=torch.float32)
        kernel_output = torch.ones_like(q, dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sol_sparse_sageattn", return_value=kernel_output) as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(),
                q,
                q,
                q,
                1,
                skip_reshape=True,
                skip_output_reshape=True,
                min_sequence_tokens=4096,
            )
        self.assertTrue(all(value.dtype == torch.bfloat16 for value in sparse.call_args.args[:3]))
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, kernel_output.float())


if __name__ == "__main__":
    unittest.main()
