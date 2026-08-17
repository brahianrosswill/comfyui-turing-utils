from __future__ import annotations

import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.adapters.minimax import video_vae  # noqa: E402
from comfyui_turing_utils.attention.protocol import (  # noqa: E402
    ATTENTION_EXECUTOR_KEY,
    AttentionExecutionOutcome,
)
from comfyui_turing_utils.nodes import minimax_vae as nodes  # noqa: E402


def blend(a, b, extent, dim):
    extent = min(a.shape[dim], b.shape[dim], extent)
    positions = torch.arange(extent, dtype=b.dtype, device=b.device)
    wa = (1 - positions / extent).view(
        [extent if i == dim % b.ndim else 1 for i in range(b.ndim)]
    )
    wb = (positions / extent).view(
        [extent if i == dim % b.ndim else 1 for i in range(b.ndim)]
    )
    a_slice = [slice(None)] * a.ndim
    b_slice = [slice(None)] * b.ndim
    a_slice[dim] = slice(-extent, None)
    b_slice[dim] = slice(0, extent)
    merged = a[tuple(a_slice)] * wa + b[tuple(b_slice)] * wb
    if extent == b.shape[dim]:
        return merged
    b_slice[dim] = slice(extent, None)
    return torch.cat((merged, b[tuple(b_slice)]), dim=dim)


class MiniMaxVideoVAETest(unittest.TestCase):
    def test_custom_encoder_reference_math_is_bitwise_equal(self):
        from comfy.ldm.minimax.vae import EncoderFCN3D

        torch.manual_seed(2)
        encoder = EncoderFCN3D(
            ch=32,
            ch_mult=[1],
            space_down=[1],
            time_down=[1],
            num_res_blocks=1,
            in_channels=3,
            z_channels=4,
        )
        quant_conv = torch.nn.Conv3d(8, 8, 1)
        for parameter in encoder.parameters():
            torch.nn.init.uniform_(parameter, -0.02, 0.02)
        model = SimpleNamespace(encoder=encoder, quant_conv=quant_conv)
        x = torch.randn(1, 3, 2, 8, 8)
        with torch.inference_mode():
            expected = quant_conv(encoder(x.clone()))
            actual = video_vae._encode_moments(
                model,
                x.clone(),
                SimpleNamespace(before_stage=lambda _index: None),
            )
        self.assertTrue(torch.equal(actual, expected))

    def test_split_tiles_covers_exact_extent(self):
        for length in (128, 256, 480, 720, 848, 1280):
            for tile in (256, 288, 320):
                starts, lengths, overlaps = video_vae.split_tiles(length, tile, 64, 16)
                self.assertEqual(starts[0], 0)
                self.assertEqual(starts[-1] + lengths[-1], length)
                self.assertTrue(all(value % 16 == 0 for value in starts))
                self.assertTrue(all(value % 16 == 0 for value in overlaps))
                for i, overlap in enumerate(overlaps):
                    self.assertEqual(starts[i + 1], starts[i] + lengths[i] - overlap)

    def test_tiled_encode_matches_full_pointwise_encode(self):
        model = SimpleNamespace(
            vae_ratio=2,
            blend=blend,
            _encode_moments=lambda value: value[..., ::2, ::2],
        )
        pixels = torch.arange(80, dtype=torch.float32).view(1, 1, 1, 8, 10)
        progress = mock.Mock()
        actual = video_vae._tiled_encode(
            model,
            pixels,
            4,
            2,
            tiles_per_batch=3,
            progress=progress,
        )
        expected = model._encode_moments(pixels)
        torch.testing.assert_close(actual, expected)
        expected_tiles = len(video_vae.split_tiles(8, 4, 2, 2)[0]) * len(
            video_vae.split_tiles(10, 4, 2, 2)[0]
        )
        self.assertEqual(
            sum(call.args[0] for call in progress.update.call_args_list),
            expected_tiles,
        )

    def test_shared_core_overlap_weights_cover_every_window_extent(self):
        model = SimpleNamespace(vae_ratio=2)
        with (
            mock.patch.object(video_vae, "TILE_SIZE", 12),
            mock.patch.object(video_vae, "TILE_OVERLAP", 4),
        ):
            layout = video_vae._SharedWindowLayout(
                model,
                2,
                10,
                15,
                torch.device("cpu"),
            )
            groups = layout.query_groups()
        weight_sum = torch.zeros(layout.image_tokens)
        query_count = 0
        for windows, global_indices, local_indices, weights in groups.values():
            self.assertEqual(windows.numel(), global_indices.shape[0])
            self.assertEqual(global_indices.shape, local_indices.shape)
            self.assertEqual(global_indices.shape, weights.shape)
            for row in range(global_indices.shape[0]):
                weight_sum.index_add_(0, global_indices[row], weights[row])
                query_count += global_indices.shape[1]
        torch.testing.assert_close(weight_sum, torch.ones_like(weight_sum))
        self.assertEqual(query_count, layout.window_count * layout.window_tokens)

        y_weights = layout._axis_overlap_weights(
            layout.latent_h,
            layout.y_idx,
            layout.y_len,
            torch.device("cpu"),
        )
        x_weights = layout._axis_overlap_weights(
            layout.latent_w,
            layout.x_idx,
            layout.x_len,
            torch.device("cpu"),
        )
        torch.testing.assert_close(y_weights.sum(dim=0), torch.ones(layout.latent_h))
        torch.testing.assert_close(x_weights.sum(dim=0), torch.ones(layout.latent_w))
        for index, (start, extent) in enumerate(zip(layout.y_idx, layout.y_len)):
            self.assertTrue(torch.all(y_weights[index, start : start + extent] > 0))
        for index, (start, extent) in enumerate(zip(layout.x_idx, layout.x_len)):
            self.assertTrue(torch.all(x_weights[index, start : start + extent] > 0))

    def test_pruned_overlap_preserves_partition_and_reduces_queries(self):
        model = SimpleNamespace(vae_ratio=2)
        with (
            mock.patch.object(video_vae, "TILE_SIZE", 12),
            mock.patch.object(video_vae, "TILE_OVERLAP", 4),
        ):
            layout = video_vae._SharedWindowLayout(
                model,
                2,
                10,
                15,
                torch.device("cpu"),
            )
            full = layout.query_groups(0.0)
            pruned = layout.query_groups(0.08)

        def summarize(groups):
            weight_sum = torch.zeros(layout.image_tokens)
            query_count = 0
            for _windows, global_indices, _local_indices, weights in groups.values():
                for row in range(global_indices.shape[0]):
                    weight_sum.index_add_(0, global_indices[row], weights[row])
                    query_count += global_indices.shape[1]
            return weight_sum, query_count

        full_sum, full_queries = summarize(full)
        pruned_sum, pruned_queries = summarize(pruned)
        torch.testing.assert_close(full_sum, torch.ones_like(full_sum))
        torch.testing.assert_close(pruned_sum, torch.ones_like(pruned_sum))
        self.assertLess(pruned_queries, full_queries)
        self.assertGreaterEqual(pruned_queries, layout.image_tokens)

    def test_aggressive_overlap_pruning_creates_small_suffix_batches(self):
        model = SimpleNamespace(vae_ratio=16)
        layout = video_vae._SharedWindowLayout(
            model,
            7,
            48,
            84,
            torch.device("cpu"),
        )
        groups = layout.query_groups(0.5)
        window_counts = [int(group[0].numel()) for group in groups.values()]
        self.assertEqual(layout.window_count, 28)
        self.assertIn(1, window_counts)
        self.assertTrue(
            any(
                count * 5 < video_vae._MIN_FUSED_SWIGLU_ROWS
                for count in window_counts
            )
        )

    def test_small_feed_forward_batches_bypass_fused_swiglu(self):
        module = mock.Mock()
        module.w1 = mock.Mock()
        module.w2 = object()
        module.side_effect = lambda value: value.add(1)
        small = torch.zeros(1, 5, 8)
        with (
            mock.patch.object(video_vae, "_fused_swiglu_eligible", return_value=True),
            mock.patch.object(video_vae.comfy.ops, "linear_input_act") as fused,
        ):
            actual = video_vae._feed_forward(module, small)
        torch.testing.assert_close(actual, torch.ones_like(small))
        module.assert_called_once_with(small)
        module.w1.assert_not_called()
        fused.assert_not_called()

    def test_large_feed_forward_batches_keep_fused_swiglu(self):
        module = mock.Mock()
        projected = torch.zeros(1, 64, 16)
        module.w1 = mock.Mock(return_value=projected)
        module.w2 = object()
        value = torch.zeros(1, 64, 8)
        expected = torch.ones_like(value)
        with (
            mock.patch.object(video_vae, "_fused_swiglu_eligible", return_value=True),
            mock.patch.object(
                video_vae.comfy.ops,
                "linear_input_act",
                return_value=expected,
            ) as fused,
        ):
            actual = video_vae._feed_forward(module, value)
        self.assertIs(actual, expected)
        module.assert_not_called()
        module.w1.assert_called_once_with(value)
        fused.assert_called_once_with(module.w2, projected, "swiglu")

    def test_feed_forward_rejects_incomplete_output_shape(self):
        module = mock.Mock(return_value=torch.zeros(1, 5, 7))
        module.w2 = object()
        with (
            mock.patch.object(video_vae, "_fused_swiglu_eligible", return_value=False),
            self.assertRaisesRegex(RuntimeError, "feed-forward returned"),
        ):
            video_vae._feed_forward(module, torch.zeros(1, 5, 8))

    def test_latent_fingerprint_is_stable_and_value_sensitive(self):
        latent = torch.arange(96, dtype=torch.float32).reshape(1, 3, 4, 4, 2)
        first = video_vae._latent_fingerprint(latent)
        second = video_vae._latent_fingerprint(latent.clone())
        changed = latent.clone()
        changed.flatten()[47] += 1
        self.assertEqual(first, second)
        self.assertNotEqual(first, video_vae._latent_fingerprint(changed))

    def test_spatial_plan_is_cached_for_matching_temporal_chunks(self):
        cache = {}
        model = object()
        value = torch.zeros(1, 4, 2, 3, 5)
        plan = object()
        with mock.patch.object(
            video_vae, "_SharedSpatialPlan", return_value=plan
        ) as constructor:
            first = video_vae._shared_spatial_plan(model, value, cache)
            second = video_vae._shared_spatial_plan(model, value, cache)
        self.assertIs(first, plan)
        self.assertIs(second, plan)
        constructor.assert_called_once_with(
            model, 2, 3, 5, value.device, value.dtype
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_shared_core_multiband_window_batching_is_invariant(self):
        from comfy.ldm.minimax.vae import ViT3DDecoder

        torch.manual_seed(26)
        decoder = (
            ViT3DDecoder(
                patch_size=2,
                patch_size_t=1,
                in_channels=4,
                out_channels=3,
                num_layers=4,
                heads=1,
                dim_head=64,
                rope_dim_ratio=0.75,
                operations=torch.nn,
            )
            .cuda()
            .half()
            .eval()
        )
        for parameter in decoder.parameters():
            torch.nn.init.uniform_(parameter, -0.01, 0.01)
        model = SimpleNamespace(decoder=decoder, vae_ratio=2)
        x = torch.randn(1, 4, 2, 4, 5, device="cuda", dtype=torch.float16)
        options = video_vae._attention_options("sdpa", x.device)
        session = SimpleNamespace(before_stage=lambda _index: None)
        with (
            mock.patch.object(video_vae, "TILE_SIZE", 4),
            mock.patch.object(video_vae, "TILE_OVERLAP", 2),
            torch.inference_mode(),
        ):
            layout = video_vae._SharedWindowLayout(model, 2, 4, 5, x.device)
            progress = mock.Mock()
            expected = video_vae._shared_core_multiband_decoder_forward(
                model,
                x.clone(),
                options,
                session,
                1,
                None,
            )
            actual = video_vae._shared_core_multiband_decoder_forward(
                model,
                x.clone(),
                options,
                session,
                3,
                progress,
            )
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(
                sum(call.args[0] for call in progress.update.call_args_list),
                layout.window_count,
            )

    def test_pruned_suffix_linears_are_batched_across_all_windows(self):
        from comfy.ldm.minimax.vae import ViT3DDecoder

        torch.manual_seed(31)
        decoder = ViT3DDecoder(
            patch_size=2,
            patch_size_t=1,
            in_channels=4,
            out_channels=3,
            num_layers=1,
            heads=1,
            dim_head=64,
            rope_dim_ratio=0.75,
            operations=torch.nn,
        ).eval()
        for parameter in decoder.parameters():
            torch.nn.init.uniform_(parameter, -0.01, 0.01)
        model = SimpleNamespace(decoder=decoder, vae_ratio=2)
        value = torch.randn(1, 4, 1, 6, 6)
        session = SimpleNamespace(before_stage=lambda _index: None)
        block = decoder.transformer_blocks[0]
        calls = {"qkv": [], "out": [], "w1": [], "w2": []}

        def recording_forward(name, forward):
            def run(input_value, *args, **kwargs):
                calls[name].append(tuple(input_value.shape))
                return forward(input_value, *args, **kwargs)

            return run

        with (
            mock.patch.object(video_vae, "TILE_SIZE", 4),
            mock.patch.object(video_vae, "TILE_OVERLAP", 2),
            mock.patch.object(
                block.attn.to_qkv,
                "forward",
                side_effect=recording_forward("qkv", block.attn.to_qkv.forward),
            ),
            mock.patch.object(
                block.attn.to_out,
                "forward",
                side_effect=recording_forward("out", block.attn.to_out.forward),
            ),
            mock.patch.object(
                block.ff.w1,
                "forward",
                side_effect=recording_forward("w1", block.ff.w1.forward),
            ),
            mock.patch.object(
                block.ff.w2,
                "forward",
                side_effect=recording_forward("w2", block.ff.w2.forward),
            ),
            torch.inference_mode(),
        ):
            layout = video_vae._SharedWindowLayout(model, 1, 6, 6, value.device)
            result = video_vae._shared_core_multiband_decoder_forward(
                model,
                value,
                video_vae._attention_options("sdpa", value.device),
                session,
                3,
                None,
                overlap_query_threshold=0.5,
                final_full_overlap_blocks=0,
            )

        suffix_shape = (
            layout.window_count,
            1 + decoder.num_register_tokens,
        )
        self.assertGreater(len(layout.query_groups(0.5)), 1)
        for name in calls:
            suffix_calls = [
                shape for shape in calls[name] if shape[:2] == suffix_shape
            ]
            self.assertEqual(len(suffix_calls), 1, name)
            self.assertGreater(
                suffix_calls[0][0] * suffix_calls[0][1],
                video_vae._MIN_FUSED_SWIGLU_ROWS,
            )
        self.assertEqual(result.shape[-2:], (12, 12))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_single_tensor_rope_fallback_matches_kitchen(self):
        torch.manual_seed(19)
        value = torch.randn(2, 7, 3, 64, device="cuda", dtype=torch.float16)
        ids = video_vae.h3_vae.create_token_ids((1, 1, 7), value.device, value.dtype)
        rotary = video_vae.h3_vae.RotaryEmbeddingND(48, n_dim=3).to(value.device)(ids)
        expected = video_vae._apply_split_half_rope(value, rotary)
        with mock.patch.object(
            video_vae.comfy.quant_ops.ck,
            "apply_rope_split_half1",
            None,
        ):
            actual = video_vae._apply_split_half_rope(value, rotary)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_multiband_weights_form_normalized_partition(self):
        starts_y, lengths_y, _ = video_vae.split_tiles(10, 4, 2, 2)
        starts_x, lengths_x, _ = video_vae.split_tiles(8, 4, 2, 2)
        descriptors = [
            (i, j, 0, 0, 0, 0)
            for i in range(len(starts_y))
            for j in range(len(starts_x))
        ]
        low_sum, high_sum = video_vae._multiband_denominators(
            descriptors,
            starts_y,
            lengths_y,
            starts_x,
            lengths_x,
            10,
            8,
            torch.float32,
            torch.device("cpu"),
        )
        normalized_low = torch.zeros_like(low_sum)
        normalized_high = torch.zeros_like(high_sum)
        for i, j, *_unused in descriptors:
            low, high = video_vae._multiband_window_weights(
                i,
                j,
                starts_y,
                lengths_y,
                starts_x,
                lengths_x,
                torch.float32,
                torch.device("cpu"),
            )
            y, x = starts_y[i], starts_x[j]
            normalized_low[y : y + lengths_y[i], x : x + lengths_x[j]].add_(
                low / low_sum[y : y + lengths_y[i], x : x + lengths_x[j]]
            )
            normalized_high[y : y + lengths_y[i], x : x + lengths_x[j]].add_(
                high / high_sum[y : y + lengths_y[i], x : x + lengths_x[j]]
            )
        torch.testing.assert_close(normalized_low, torch.ones_like(normalized_low))
        torch.testing.assert_close(normalized_high, torch.ones_like(normalized_high))

    def test_multiband_identical_overlaps_reconstruct_without_seam_pairs(self):
        height = width = 48
        starts_y, lengths_y, _ = video_vae.split_tiles(height, 24, 8, 2)
        starts_x, lengths_x, _ = video_vae.split_tiles(width, 24, 8, 2)
        assembler = video_vae._MultibandPixelAssembler(
            starts_y,
            lengths_y,
            starts_x,
            lengths_x,
            height,
            width,
            torch.device("cpu"),
        )
        rows, columns = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing="ij",
        )
        image = (
            torch.sin(columns * 13) + torch.cos(rows * 9) + rows * columns
        ).view(1, 1, 1, height, width)
        for i, y in enumerate(starts_y):
            for j, x in enumerate(starts_x):
                assembler.add(
                    i * len(starts_x) + j,
                    image[..., y : y + lengths_y[i], x : x + lengths_x[j]],
                )
        torch.testing.assert_close(
            assembler.finish(),
            image,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_decode_spatial_routes_fixed_boundary_policy(self):
        expected = object()
        model = SimpleNamespace(post_quant_conv=lambda value: value)
        session = mock.Mock()
        with mock.patch.object(
            video_vae, "_shared_core_multiband_decoder_forward", return_value=expected
        ) as run:
            actual = video_vae._decode_spatial(
                model,
                "latent",
                {},
                session,
                4,
                None,
            )
        self.assertIs(actual, expected)
        session.before_stage.assert_called_once_with(0)
        run.assert_called_once_with(
            model, "latent", {}, session, 4, None, None, 0.0, 36
        )

    def test_shared_state_uses_windowed_multiband_projection(self):
        decoder = SimpleNamespace(
            x_embedder=lambda value: value,
            num_register_tokens=1,
            register_tokens=torch.zeros(1, 1, 4),
            pos_embed=lambda value: value,
            transformer_blocks=[],
        )
        model = SimpleNamespace(decoder=decoder, vae_ratio=2)
        x = torch.zeros(1, 4, 1, 2, 2)
        expected = object()
        with (
            mock.patch.object(video_vae, "TILE_SIZE", 4),
            mock.patch.object(video_vae, "TILE_OVERLAP", 2),
            mock.patch.object(
                video_vae, "_project_shared_state_windows", return_value=expected
            ) as projection,
        ):
            actual = video_vae._shared_core_multiband_decoder_forward(
                model,
                x,
                {},
                mock.Mock(),
                1,
                None,
            )
        self.assertIs(actual, expected)
        self.assertEqual(projection.call_args.args[3], 1)

    def test_decoder_node_uses_optimized_runtime(self):
        vae = mock.Mock()
        vae.device = torch.device("cpu")
        latent = torch.zeros(1, 24, 2, 1, 1)
        decoded = torch.zeros(1, 2, 4, 4, 3)
        with (
            mock.patch.object(nodes, "require_h3_video_vae"),
            mock.patch.object(nodes, "decode_video", return_value=decoded) as run,
            mock.patch.object(
                nodes.comfy.model_management,
                "cuda_device_context",
                return_value=nullcontext(),
            ),
        ):
            output = nodes.MiniMaxH3VideoVAEDecode().decode(
                {"samples": latent},
                vae,
                "sdpa",
            )[0]
        run.assert_called_once_with(
            vae,
            latent,
            "sdpa",
            overlap_query_threshold=0.0,
            final_full_overlap_blocks=36,
        )
        self.assertEqual(output.shape, (2, 4, 4, 3))

    def test_encoder_node_uses_optimized_runtime(self):
        vae = mock.Mock()
        vae.device = torch.device("cpu")
        pixels = torch.zeros(5, 16, 16, 3)
        encoded = torch.zeros(1, 24, 2, 1, 1)
        with (
            mock.patch.object(nodes, "require_h3_video_vae"),
            mock.patch.object(nodes, "encode_video", return_value=encoded) as run,
            mock.patch.object(
                nodes.comfy.model_management,
                "cuda_device_context",
                return_value=nullcontext(),
            ),
        ):
            output = nodes.MiniMaxH3VideoVAEEncode().encode(pixels, vae)[0]["samples"]
        run.assert_called_once_with(vae, pixels)
        self.assertEqual(output.shape, (1, 24, 2, 1, 1))

    def test_decoder_attention_is_container_owned(self):
        module = SimpleNamespace(
            heads=1,
            dim_head=4,
            to_qkv=torch.nn.Linear(4, 12),
            to_out=torch.nn.Identity(),
            norm_q=SimpleNamespace(weight=None, eps=1e-5),
            norm_k=SimpleNamespace(weight=None, eps=1e-5),
        )
        seen = []

        def consume(q, k, v, heads, **_kwargs):
            self.assertIsInstance(q, video_vae.AttentionTensorContainer)
            self.assertIsInstance(k, video_vae.AttentionTensorContainer)
            self.assertIsInstance(v, video_vae.AttentionTensorContainer)
            seen.extend((q.take(), k.take(), v.take()))
            return torch.zeros(1, 3, 4)

        override = mock.Mock()
        override.container_function = consume
        output = video_vae._attention_forward(
            module,
            torch.randn(1, 3, 4),
            None,
            {"optimized_attention_override": override},
        )
        self.assertEqual(output.shape, (1, 3, 4))
        self.assertEqual([tuple(tensor.shape) for tensor in seen], [(1, 1, 3, 4)] * 3)
        override.assert_not_called()

    def test_decoder_attention_uses_prepared_qk_transform(self):
        module = SimpleNamespace(
            heads=1,
            dim_head=4,
            to_qkv=torch.nn.Linear(4, 12),
            to_out=torch.nn.Identity(),
            norm_q=SimpleNamespace(weight=None, eps=1e-5),
            norm_k=SimpleNamespace(weight=None, eps=1e-5),
        )
        requests = []

        def execute(request):
            requests.append(request)
            self.assertTrue(
                torch.equal(request.qk_transform.query_norm.weight, torch.ones(4))
            )
            self.assertTrue(
                torch.equal(request.qk_transform.key_norm.weight, torch.ones(4))
            )
            query, _key, _value = request.consume_qkv()
            return AttentionExecutionOutcome(
                torch.zeros(query.shape[0], query.shape[2], 4)
            )

        output = video_vae._attention_forward(
            module,
            torch.randn(1, 3, 4),
            None,
            {ATTENTION_EXECUTOR_KEY: execute},
        )
        self.assertEqual(output.shape, (1, 3, 4))
        self.assertEqual(len(requests), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_pixel_double_buffer_preserves_fp32_values(self):
        output = torch.empty(1, 1, 2, 2, 2, dtype=torch.float32)
        model = SimpleNamespace(_finalize_pixels=lambda value: value.float())
        writer = video_vae._PixelWriter(output, model, torch.device("cuda"))
        first = torch.randn(1, 1, 1, 2, 2, device="cuda", dtype=torch.float32)
        second = torch.randn(1, 1, 1, 2, 2, device="cuda", dtype=torch.float32)
        expected = torch.cat((first, second), dim=2).cpu()
        writer.write(first)
        writer.write(second)
        actual = writer.finish()
        self.assertEqual(writer.staging[0].dtype, torch.float32)
        self.assertEqual(writer.staging[1].dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

    def test_fixed_tile_geometry_and_minimal_node_controls(self):
        self.assertEqual(video_vae.TILE_SIZE, 256)
        self.assertEqual(video_vae.TILE_OVERLAP, 64)
        decoder = nodes.MiniMaxH3VideoVAEDecode.INPUT_TYPES()["required"]
        decoder_optional = nodes.MiniMaxH3VideoVAEDecode.INPUT_TYPES()["optional"]
        encoder = nodes.MiniMaxH3VideoVAEEncode.INPUT_TYPES()["required"]
        self.assertEqual(
            set(decoder),
            {
                "samples",
                "vae",
                "attention",
            },
        )
        self.assertEqual(
            set(decoder_optional),
            {
                "overlap_query_threshold",
                "final_full_overlap_blocks",
            },
        )
        self.assertEqual(set(encoder), {"pixels", "vae"})
        self.assertEqual(decoder_optional["overlap_query_threshold"][1]["default"], 0.0)
        self.assertEqual(
            decoder_optional["final_full_overlap_blocks"][1]["default"], 36
        )
        upscaler = nodes.MiniMaxH3LatentPixelUpscale.INPUT_TYPES()["required"]
        self.assertEqual(
            set(upscaler),
            {
                "samples",
                "vae",
                "width",
                "height",
                "upscale_method",
                "rtx_vsr_quality",
                "attention",
            },
        )

    def test_prepare_batched_pixels_crops_only_spatial_axes(self):
        vae = SimpleNamespace(
            crop_input=True,
            output_channels=3,
            spacial_compression_encode=lambda: 16,
        )
        pixels = torch.zeros(2, 5, 34, 50, 3)
        actual = video_vae._prepare_encode_pixels(vae, pixels)
        self.assertEqual(actual.shape, (2, 3, 5, 32, 48))

    def test_bicubic_pixel_resize_stream_stores_fp16(self):
        torch.manual_seed(4)
        pixels = torch.rand(1, 3, 5, 4, 6)
        progress = mock.Mock()
        with mock.patch.object(video_vae, "_TileProgress", return_value=progress):
            transform = video_vae._PixelResizeTransform(
                pixels.shape,
                12,
                8,
                "bicubic",
                "high",
                torch.device("cpu"),
            )
            actual = transform(pixels)
            transform.finish()
        expected = torch.nn.functional.interpolate(
            pixels.permute(0, 2, 1, 3, 4).reshape(5, 3, 4, 6),
            size=(8, 12),
            mode="bicubic",
            align_corners=False,
            antialias=False,
        ).view(1, 5, 3, 8, 12).permute(0, 2, 1, 3, 4)
        self.assertEqual(actual.dtype, torch.float16)
        torch.testing.assert_close(actual, expected.half(), rtol=0, atol=0)
        progress.update.assert_called_once_with(5)
        progress.finish.assert_called_once_with()

    def test_rtx_vsr_wrapper_uses_sdk_owned_dlpack_safely(self):
        instances = []

        class QualityLevel:
            HIGH = 3

        class VideoSuperRes:
            def __init__(self, quality):
                self.quality = quality
                self.loaded = False
                instances.append(self)

            def load(self):
                self.loaded = True

            def run(self, frame):
                return SimpleNamespace(image=frame.add(1))

        VideoSuperRes.QualityLevel = QualityLevel

        with mock.patch.dict(
            sys.modules,
            {"nvvfx": SimpleNamespace(VideoSuperRes=VideoSuperRes)},
        ):
            effect = video_vae._RTXVideoSuperResolution(12, 8, "high")
            frame = torch.zeros(3, 4, 6)
            actual = effect(frame)
        self.assertTrue(instances[0].loaded)
        self.assertEqual(instances[0].quality, QualityLevel.HIGH)
        self.assertEqual((instances[0].output_width, instances[0].output_height), (12, 8))
        torch.testing.assert_close(actual, torch.ones_like(frame))

    def test_fused_pixel_roundtrip_delegates_vae_lifecycle_to_manager(self):
        latent = torch.zeros(1, 24, 2, 2, 3)
        decoded = torch.zeros(1, 5, 32, 48, 3)
        resized = torch.zeros(1, 5, 64, 96, 3, dtype=torch.float16)
        encoded = torch.zeros(1, 24, 2, 4, 6)
        vae = SimpleNamespace(device=torch.device("cpu"))
        model = SimpleNamespace(
            decode_output_shape=lambda _shape: (1, 3, 5, 32, 48)
        )
        with (
            mock.patch.object(video_vae, "require_h3_video_vae", return_value=model),
            mock.patch.object(
                video_vae,
                "_pixel_roundtrip_stage_device",
                return_value=torch.device("cpu"),
            ),
            mock.patch.object(video_vae, "decode_video", return_value=decoded) as decode,
            mock.patch.object(video_vae, "_PixelResizeTransform") as transform_type,
            mock.patch.object(video_vae, "encode_video", return_value=encoded) as encode,
        ):
            transform = transform_type.return_value
            decode.return_value = resized
            actual = video_vae.upscale_latent_via_pixels(
                vae,
                latent,
                96,
                64,
                "bicubic",
                "high",
                "sdpa",
            )
        self.assertIs(actual, encoded)
        decode.assert_called_once_with(
            vae,
            latent,
            "sdpa",
            output_device=torch.device("cpu"),
            overlap_query_threshold=0.0,
            final_full_overlap_blocks=36,
            _pixel_transform=transform,
        )
        transform_type.assert_called_once_with(
            (1, 3, 5, 32, 48),
            96,
            64,
            "bicubic",
            "high",
            vae.device,
        )
        transform.finish.assert_called()
        encode.assert_called_once_with(vae, resized)

    def test_pixel_upscale_node_preserves_h3_audio_and_resizes_video_mask(self):
        video = torch.zeros(1, 24, 2, 2, 3)
        audio = torch.ones(1, 32, 2, 7)
        video_mask = torch.zeros(1, 1, 2, 2, 3)
        audio_mask = torch.ones_like(audio)
        nested = nodes.comfy.nested_tensor.NestedTensor((video, audio))
        nested_mask = nodes.comfy.nested_tensor.NestedTensor(
            (video_mask, audio_mask)
        )
        resized = torch.full((1, 24, 2, 4, 6), 3.0)
        vae = SimpleNamespace(device=torch.device("cpu"))
        with (
            mock.patch.object(nodes, "require_h3_video_vae"),
            mock.patch.object(
                nodes,
                "upscale_latent_via_pixels",
                return_value=resized,
            ) as run,
            mock.patch.object(
                nodes.comfy.model_management,
                "cuda_device_context",
                return_value=nullcontext(),
            ),
        ):
            output = nodes.MiniMaxH3LatentPixelUpscale().upscale(
                {"samples": nested, "noise_mask": nested_mask},
                vae,
                96,
                64,
                "bicubic",
                "high",
                "sdpa",
            )[0]
        run.assert_called_once_with(
            vae,
            video,
            96,
            64,
            "bicubic",
            "high",
            "sdpa",
            0.0,
            36,
        )
        output_video, output_audio = output["samples"].unbind()
        output_video_mask, output_audio_mask = output["noise_mask"].unbind()
        self.assertIs(output_video, resized)
        self.assertIs(output_audio, audio)
        self.assertEqual(output_video_mask.shape[-2:], (4, 6))
        self.assertIs(output_audio_mask, audio_mask)

    def test_auto_tile_batch_respects_memory_limit(self):
        vae = SimpleNamespace()
        with mock.patch.object(video_vae, "_tile_memory_budget", return_value=35):
            selected, estimate = video_vae._select_tiles_per_batch(
                vae,
                8,
                lambda count: count * 10,
                4,
            )
        self.assertEqual((selected, estimate), (3, 30))
        self.assertEqual(video_vae._AUTO_DECODE_TILE_BATCH_LIMIT, 16)
        self.assertEqual(video_vae._AUTO_ENCODE_TILE_BATCH_LIMIT, 16)

    def test_tile_progress_uses_comfy_progress_bar(self):
        bar = mock.Mock()
        terminal = mock.Mock()
        with (
            mock.patch.object(
                video_vae.comfy.utils, "ProgressBar", return_value=bar
            ) as factory,
            mock.patch.object(video_vae, "tqdm", return_value=terminal) as tqdm_factory,
        ):
            progress = video_vae._TileProgress(11, description="H3 VAE Test")
            progress.update(4)
            progress.update(3)
            progress.finish()
        factory.assert_called_once_with(11)
        tqdm_factory.assert_called_once_with(
            total=11,
            desc="H3 VAE Test",
            disable=not video_vae.comfy.utils.PROGRESS_BAR_ENABLED,
        )
        self.assertEqual([call.args[0] for call in bar.update.call_args_list], [4, 3])
        self.assertEqual(
            [call.args[0] for call in terminal.update.call_args_list], [4, 3]
        )
        terminal.close.assert_called_once_with()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_tile_progress_waits_for_cuda_events_off_thread(self):
        bar = mock.Mock()
        terminal = mock.Mock()
        with (
            mock.patch.object(video_vae.comfy.utils, "ProgressBar", return_value=bar),
            mock.patch.object(video_vae, "tqdm", return_value=terminal),
        ):
            progress = video_vae._TileProgress(5, torch.device("cuda"))
            progress.update(3)
            progress.update(2)
            progress.finish()
        self.assertEqual([call.args[0] for call in bar.update.call_args_list], [3, 2])
        self.assertEqual(
            [call.args[0] for call in terminal.update.call_args_list], [3, 2]
        )
        terminal.close.assert_called_once_with()
        self.assertIsNone(progress.worker)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_encoder_double_buffer_converts_into_fp16_destination(self):
        model = SimpleNamespace(clip_length=2, token_drop=0)
        pixels = torch.randn(1, 3, 4, 4, 4, dtype=torch.float32)
        seen = []

        def fake_encode(_model, clip, *_args):
            seen.append(clip.cpu())
            return torch.zeros(
                clip.shape[0],
                2,
                1,
                1,
                1,
                dtype=clip.dtype,
                device=clip.device,
            )

        with mock.patch.object(video_vae, "_encode_clip", side_effect=fake_encode):
            output = video_vae._encode_temporal_buffered(
                model,
                pixels,
                lambda value: value,
                torch.float16,
                torch.device("cuda"),
                256,
                64,
                None,
                1,
                None,
            )
        self.assertEqual([value.dtype for value in seen], [torch.float16] * 2)
        torch.testing.assert_close(seen[0], pixels[:, :, :2].half())
        torch.testing.assert_close(seen[1], pixels[:, :, 2:].half())
        self.assertEqual(output.dtype, torch.float16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_encoder_double_buffer_keeps_fp16_roundtrip_normalization_on_device(self):
        model = SimpleNamespace(clip_length=2, token_drop=0)
        pixels = torch.randn(1, 3, 2, 4, 4, dtype=torch.float16)
        normalized = []

        def process_input(value):
            normalized.append((value.device.type, value.dtype))
            return value.mul(2)

        def fake_encode(_model, clip, *_args):
            torch.testing.assert_close(clip.cpu(), pixels.mul(2))
            return torch.zeros(
                clip.shape[0],
                2,
                1,
                1,
                1,
                dtype=clip.dtype,
                device=clip.device,
            )

        with mock.patch.object(video_vae, "_encode_clip", side_effect=fake_encode):
            video_vae._encode_temporal_buffered(
                model,
                pixels,
                process_input,
                torch.float16,
                torch.device("cuda"),
                256,
                64,
                None,
                1,
                None,
            )
        self.assertEqual(normalized, [("cuda", torch.float16)])

    def test_retained_weights_marks_failed_first_stage_for_cleanup(self):
        module = SimpleNamespace(
            _v=object(),
            weight=torch.empty(4),
            bias=None,
        )

        def failed_prefetch(modules, *_args):
            for item in modules:
                item._prefetch = {"signature": None}
            return None

        with (
            mock.patch.object(video_vae.comfy.model_management, "NUM_STREAMS", 1),
            mock.patch.object(
                video_vae.comfy.model_management,
                "device_supports_non_blocking",
                return_value=True,
            ),
            mock.patch.object(
                video_vae.comfy.ops,
                "cast_modules_with_vbar",
                side_effect=failed_prefetch,
            ),
            mock.patch.object(
                video_vae.comfy.model_management,
                "sync_stream",
            ),
            mock.patch.object(
                video_vae.comfy.model_management,
                "synchronize",
            ),
            mock.patch.object(
                video_vae.comfy.model_management,
                "current_stream",
                return_value=SimpleNamespace(synchronize=lambda: None),
            ),
        ):
            session = video_vae._RetainedWeights([[module]], torch.device("cuda"), True)
            session.start()

        self.assertTrue(session.attempted)
        self.assertEqual(session.started, 0)
        self.assertFalse(session.enabled)
        self.assertFalse(hasattr(module, "_prefetch"))

    def test_retained_weight_cleanup_preserves_aimdo_vbar_views(self):
        signature = object()
        weight_view = object()
        bias_view = object()
        module = SimpleNamespace(
            _v=object(),
            weight=torch.empty(4),
            bias=None,
            _prefetch={"signature": signature},
            _v_signature=signature,
            _v_weight=weight_view,
            _v_bias=bias_view,
        )
        with (
            mock.patch.object(
                video_vae.comfy.model_management,
                "synchronize",
            ) as synchronize,
            mock.patch.object(
                video_vae.comfy_aimdo.model_vbar,
                "vbar_unpin",
            ) as unpin,
        ):
            session = video_vae._RetainedWeights(
                [[module]], torch.device("cuda"), True
            )
            session.attempted = True
            session.finish()

        synchronize.assert_called_once_with()
        unpin.assert_called_once_with(module._v)
        self.assertFalse(hasattr(module, "_prefetch"))
        self.assertIs(module._v_signature, signature)
        self.assertIs(module._v_weight, weight_view)
        self.assertIs(module._v_bias, bias_view)

    def test_decoder_backend_switching_preserves_cached_latent(self):
        vae = mock.Mock()
        vae.device = torch.device("cpu")
        latent = torch.arange(48, dtype=torch.float32).reshape(1, 3, 2, 4, 2)
        original = latent.clone()
        fingerprints = []

        def decode(_vae, value, attention, **_kwargs):
            fingerprints.append((attention, video_vae._latent_fingerprint(value)))
            return torch.zeros(1, 2, 4, 4, 3)

        with (
            mock.patch.object(nodes, "require_h3_video_vae"),
            mock.patch.object(nodes, "decode_video", side_effect=decode),
            mock.patch.object(
                nodes.comfy.model_management,
                "cuda_device_context",
                return_value=nullcontext(),
            ),
        ):
            node = nodes.MiniMaxH3VideoVAEDecode()
            for backend in ("sdpa", "w8a8", "sdpa"):
                node.decode({"samples": latent}, vae, backend)

        self.assertEqual([item[0] for item in fingerprints], ["sdpa", "w8a8", "sdpa"])
        self.assertEqual(len({item[1] for item in fingerprints}), 1)
        torch.testing.assert_close(latent, original)

    def test_retained_weights_release_prefix_when_cycle_does_not_fit(self):
        modules = [
            SimpleNamespace(_v=object(), weight=torch.empty(4), bias=None)
            for _ in range(2)
        ]
        calls = 0
        unpinned = []

        def prefetch(items, *_args):
            nonlocal calls
            signature = object() if calls == 0 else None
            calls += 1
            for item in items:
                item._prefetch = {"signature": signature}
            return None

        with (
            mock.patch.object(video_vae.comfy.model_management, "NUM_STREAMS", 1),
            mock.patch.object(
                video_vae.comfy.model_management,
                "device_supports_non_blocking",
                return_value=True,
            ),
            mock.patch.object(
                video_vae.comfy.ops,
                "cast_modules_with_vbar",
                side_effect=prefetch,
            ),
            mock.patch.object(video_vae.comfy.model_management, "sync_stream"),
            mock.patch.object(
                video_vae.comfy.model_management,
                "current_stream",
                return_value=SimpleNamespace(synchronize=lambda: None),
            ),
            mock.patch.object(
                video_vae.comfy_aimdo.model_vbar,
                "vbar_unpin",
                side_effect=lambda alloc: unpinned.append(alloc),
            ),
        ):
            session = video_vae._RetainedWeights(
                [[modules[0]], [modules[1]]], torch.device("cuda"), True
            )
            session.start()
            session.before_stage(0)

        self.assertFalse(session.enabled)
        self.assertEqual(unpinned, [modules[0]._v])
        self.assertTrue(all(not hasattr(item, "_prefetch") for item in modules))

    def test_retained_weights_stay_pinned_until_session_finish(self):
        modules = [
            SimpleNamespace(_v=object(), weight=torch.empty(4), bias=None)
            for _ in range(2)
        ]
        unpinned = []

        def successful_prefetch(items, *_args):
            for item in items:
                item._prefetch = {"signature": object()}
            return None

        with (
            mock.patch.object(video_vae.comfy.model_management, "NUM_STREAMS", 1),
            mock.patch.object(
                video_vae.comfy.model_management,
                "device_supports_non_blocking",
                return_value=True,
            ),
            mock.patch.object(
                video_vae.comfy.ops,
                "cast_modules_with_vbar",
                side_effect=successful_prefetch,
            ),
            mock.patch.object(
                video_vae.comfy_aimdo.model_vbar,
                "vbar_unpin",
                side_effect=lambda alloc: unpinned.append(alloc),
            ),
            mock.patch.object(
                video_vae.comfy.model_management,
                "sync_stream",
            ),
            mock.patch.object(
                video_vae.comfy.model_management,
                "synchronize",
            ),
        ):
            session = video_vae._RetainedWeights(
                [[modules[0]], [modules[1]]], torch.device("cuda"), True
            )
            session.start()
            session.before_stage(0)
            session.before_stage(1)
            self.assertEqual(unpinned, [])
            session.finish()

        self.assertEqual(unpinned, [modules[0]._v, modules[1]._v])
        self.assertTrue(all(not hasattr(item, "_prefetch") for item in modules))

    def test_retained_weights_work_without_async_streams(self):
        module = SimpleNamespace(_v=object(), weight=torch.empty(4), bias=None)

        def successful_prefetch(items, *_args):
            for item in items:
                item._prefetch = {"signature": object()}
            return None

        with (
            mock.patch.object(video_vae.comfy.model_management, "NUM_STREAMS", 0),
            mock.patch.object(
                video_vae.comfy.ops,
                "cast_modules_with_vbar",
                side_effect=successful_prefetch,
            ) as cast,
            mock.patch.object(
                video_vae.comfy_aimdo.model_vbar,
                "vbar_unpin",
            ),
            mock.patch.object(
                video_vae.comfy.model_management,
                "synchronize",
            ),
        ):
            session = video_vae._RetainedWeights([[module]], torch.device("cuda"), True)
            session.start()
            session.before_stage(0)
            session.finish()

        self.assertFalse(session.non_blocking)
        self.assertEqual(cast.call_args.args[-1], False)
        self.assertFalse(hasattr(module, "_prefetch"))

    def test_custom_decode_preserves_temporal_length_and_values(self):
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

        model = object.__new__(MiniMaxH3VideoVAE)
        torch.nn.Module.__init__(model)
        model.vae_ratio = 1
        model.vae_ratio_t = 4
        model.tokens_chunk_size = 5
        model.token_overlap = 2
        model.token_drop = 3
        model.clip_length = 17
        model.frame_pre_padding = 3
        model.frame_overlap = 5
        model.decoder = SimpleNamespace(out_channels=1)
        model._finalize_pixels = lambda value: value.float()

        z = torch.arange(12, dtype=torch.float32).view(1, 1, 12, 1, 1)

        def fake_decode(_model, value, *_args):
            return value.repeat_interleave(4, dim=2)

        model._adaptive_decode = lambda value: fake_decode(model, value)
        expected = model.decode_temporal(z.clone())
        with mock.patch.object(video_vae, "_decode_spatial", side_effect=fake_decode):
            actual = video_vae._decode_temporal(
                model,
                z.clone(),
                {},
                SimpleNamespace(before_stage=lambda _index: None),
                1,
                None,
                output_device=torch.device("cpu"),
            )
        self.assertTrue(torch.equal(actual, expected))

    def test_temporal_decode_streams_resize_without_source_pixel_store(self):
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

        model = object.__new__(MiniMaxH3VideoVAE)
        torch.nn.Module.__init__(model)
        model.vae_ratio = 1
        model.vae_ratio_t = 4
        model.tokens_chunk_size = 5
        model.token_overlap = 2
        model.token_drop = 3
        model.clip_length = 17
        model.frame_pre_padding = 3
        model.frame_overlap = 5
        model.decoder = SimpleNamespace(out_channels=1)
        model._finalize_pixels = lambda value: value.float()
        z = torch.arange(12, dtype=torch.float32).view(1, 1, 12, 1, 1)

        def fake_decode(_model, value, *_args):
            return value.repeat_interleave(4, dim=2)

        model._adaptive_decode = lambda value: fake_decode(model, value)
        source_shape = model.decode_output_shape(z.shape)
        with (
            mock.patch.object(video_vae, "_TileProgress", return_value=mock.Mock()),
            mock.patch.object(video_vae, "_decode_spatial", side_effect=fake_decode),
        ):
            transform = video_vae._PixelResizeTransform(
                source_shape,
                2,
                2,
                "nearest-exact",
                "high",
                torch.device("cpu"),
            )
            actual = video_vae._decode_temporal(
                model,
                z.clone(),
                {},
                SimpleNamespace(before_stage=lambda _index: None),
                1,
                None,
                output_device=torch.device("cpu"),
                pixel_transform=transform,
            )
        expected = model.decode_temporal(z.clone()).expand(-1, -1, -1, 2, 2).half()
        self.assertEqual(actual.dtype, torch.float16)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
