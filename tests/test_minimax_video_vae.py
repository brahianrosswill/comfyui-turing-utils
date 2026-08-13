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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_custom_decoder_reference_math_is_bitwise_equal(self):
        from comfy.ldm.minimax.vae import ViT3DDecoder

        torch.manual_seed(1)
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
        )
        for parameter in decoder.parameters():
            torch.nn.init.uniform_(parameter, -0.02, 0.02)
        decoder = decoder.cuda().half()
        x = torch.randn(1, 4, 2, 4, 4, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            expected = decoder(x.clone())
            transformer_options = video_vae._attention_options("sdpa", x.device)
            actual = video_vae._decoder_forward(
                decoder,
                x.clone(),
                transformer_options,
                SimpleNamespace(
                    before_stage=lambda _index: None,
                ),
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

    def test_tiled_decode_matches_full_pointwise_decode(self):
        model = SimpleNamespace(vae_ratio=2, blend=blend, post_quant_conv=lambda x: x)
        z = torch.arange(20, dtype=torch.float32).view(1, 1, 1, 4, 5)

        def fake_decode(_model, value, *_args):
            return value.repeat_interleave(2, -2).repeat_interleave(2, -1)

        with mock.patch.object(video_vae, "_decode_pixels", side_effect=fake_decode):
            actual = video_vae._tiled_decode(
                model,
                z,
                4,
                2,
                {},
                SimpleNamespace(before_stage=lambda _index: None),
            )
        expected = fake_decode(model, z)
        torch.testing.assert_close(actual, expected)

    def test_tiled_encode_matches_full_pointwise_encode(self):
        model = SimpleNamespace(
            vae_ratio=2,
            blend=blend,
            _encode_moments=lambda value: value[..., ::2, ::2],
        )
        pixels = torch.arange(80, dtype=torch.float32).view(1, 1, 1, 8, 10)
        actual = video_vae._tiled_encode(model, pixels, 4, 2)
        expected = model._encode_moments(pixels)
        torch.testing.assert_close(actual, expected)

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
                "288",
                "sdpa",
            )[0]
        run.assert_called_once_with(vae, latent, "288", "sdpa")
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
            output = nodes.MiniMaxH3VideoVAEEncode().encode(
                pixels, vae, "auto"
            )[0]["samples"]
        run.assert_called_once_with(vae, pixels, "auto")
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

    def test_tile_presets_are_numeric_and_auto_is_default(self):
        self.assertEqual(video_vae.TILE_PRESETS[0], "auto")
        for node in (
            nodes.MiniMaxH3VideoVAEDecode,
            nodes.MiniMaxH3VideoVAEEncode,
        ):
            required = node.INPUT_TYPES()["required"]
            self.assertEqual(required["tile_preset"][1]["default"], "auto")
            self.assertNotIn("tile_size", required)
            self.assertNotIn("tile_overlap", required)
            self.assertNotIn("activation_dtype", required)
        self.assertEqual(video_vae.resolve_tile_preset("256", 720, 1280), 256)
        self.assertEqual(video_vae.resolve_tile_preset("288", 720, 1280), 288)
        auto = video_vae.resolve_tile_preset("auto", 720, 1280)
        self.assertGreaterEqual(auto, 256)
        self.assertLessEqual(auto, 480)
        self.assertEqual(auto % 16, 0)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            video_vae.resolve_tile_preset("balanced", 720, 1280)

    def test_auto_tile_respects_memory_limit(self):
        estimator = lambda edge: edge**4
        constrained = video_vae.resolve_tile_preset(
            "auto",
            480,
            848,
            memory_limit=256**4,
            memory_estimator=estimator,
        )
        unconstrained = video_vae.resolve_tile_preset("auto", 480, 848)
        self.assertEqual(constrained, 256)
        self.assertEqual(unconstrained, 480)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            video_vae.resolve_tile_preset(
                "auto",
                480,
                848,
                memory_limit=256**4,
            )

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
            )
        self.assertEqual([value.dtype for value in seen], [torch.float16] * 2)
        torch.testing.assert_close(seen[0], pixels[:, :, :2].half())
        torch.testing.assert_close(seen[1], pixels[:, :, 2:].half())
        self.assertEqual(output.dtype, torch.float16)

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
        with mock.patch.object(video_vae, "_tiled_decode", side_effect=fake_decode):
            actual = video_vae._decode_temporal(
                model,
                z.clone(),
                256,
                64,
                {},
                SimpleNamespace(before_stage=lambda _index: None),
            )
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
