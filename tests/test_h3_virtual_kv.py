from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.adapters.minimax.virtual_kv import (  # noqa: E402
    _SOURCE_FRAMES,
    _cached_virtual_inputs,
    make_h3_virtual_kv_override,
)
from comfyui_turing_utils.attention.layout import (  # noqa: E402
    ATTENTION_LAYOUT_KEY,
    AttentionSegment,
    AttentionSemanticLayout,
    AttentionTopology,
)
from comfyui_turing_utils.attention.protocol import (  # noqa: E402
    AttentionExecutionOutcome,
    PreparedAttention,
    QKTransformSpec,
    RMSNormSpec,
    RotaryEmbeddingSpec,
)


class Owner:
    def __init__(self, value):
        self.value = value

    def peek(self):
        if self.value is None:
            raise RuntimeError("consumed")
        return self.value

    def take(self):
        value = self.peek()
        self.value = None
        return value


def rotation_table(positions: torch.Tensor) -> torch.Tensor:
    inverse = torch.exp(-torch.log(torch.tensor(10_000.0)) * torch.arange(16) / 16)
    axes = positions[:, :, None] * inverse[None, None]
    angles = axes.reshape(positions.shape[0], 48)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    return torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        1, positions.shape[0], 1, 48, 2, 2
    )


def options(frames=2, tokens_per_frame=4):
    prefix = 4
    stop = prefix + frames * tokens_per_frame
    segments = (
        AttentionSegment.for_role(0, prefix, "text"),
        AttentionSegment.for_role(
            prefix, stop, "target_video", topology_id="target_video"
        ),
    )
    layout = AttentionSemanticLayout(
        provider="minimax_h3",
        query_segments=segments,
        key_segments=segments,
        topologies=(
            AttentionTopology(
                "target_video",
                "video_grid",
                prefix,
                stop,
                tokens_per_frame,
                2,
                tokens_per_frame // 2,
            ),
        ),
        layer_index=0,
        layer_count=1,
    )
    return {ATTENTION_LAYOUT_KEY: layout.to_wire()}, prefix, stop


def request(frames=2, tokens_per_frame=4):
    transformer_options, prefix, stop = options(frames, tokens_per_frame)
    sequence = stop
    query = torch.zeros((1, 2, sequence, 128), dtype=torch.float32)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    for frame in range(frames):
        value[:, :, prefix + frame * tokens_per_frame : prefix + (frame + 1) * tokens_per_frame] = frame + 1
    positions = torch.zeros((sequence, 3), dtype=torch.float32)
    positions[:prefix, 0] = torch.arange(prefix)
    positions[prefix : prefix + tokens_per_frame, 0] = 10.0
    positions[prefix + tokens_per_frame : stop, 0] = 10.0 + 5.0 / 3.0
    transform = QKTransformSpec(
        RMSNormSpec(torch.ones(128), 1e-6, "head"),
        RMSNormSpec(torch.ones(128), 1e-6, "head"),
        RotaryEmbeddingSpec(rotation_table(positions), 96, "split_half"),
    )
    prepared = PreparedAttention.from_hnd(
        Owner(query),
        Owner(key),
        Owner(value),
        heads=2,
        qk_transform=transform,
        transformer_options=transformer_options,
    )
    return prepared, prefix, stop


class H3VirtualKVTest(unittest.TestCase):
    def _run(self, mode):
        seen = {}

        def dense_executor(prepared):
            seen["query"], seen["key"], seen["value"] = prepared.peek_qkv()
            seen["transform"] = prepared.qk_transform
            prepared.consume_qkv()
            return AttentionExecutionOutcome(
                torch.zeros((1, prepared.query_tokens, 2 * 128))
            )

        def dense_override(original, *args, **kwargs):
            return original(*args, **kwargs)

        dense_override.prepared_attention_executor = dense_executor
        prepared, prefix, stop = request()
        outcome = make_h3_virtual_kv_override(
            dense_override, mode=mode
        ).prepared_attention_executor(prepared)
        self.assertTrue(outcome.supported)
        return seen, prefix, stop

    def test_conservative_expands_only_key_and_value_to_seven_frames(self):
        seen, prefix, _ = self._run("conservative")
        self.assertEqual(seen["query"].shape[2], 12)
        self.assertEqual(seen["key"].shape[2], prefix + 7 * 4)
        self.assertEqual(seen["value"].shape[2], prefix + 7 * 4)
        frames = seen["value"][:, :, prefix:].reshape(1, 2, 7, 4, 128)
        self.assertEqual(
            [int(frames[0, 0, index, 0, 0].item()) for index in range(7)],
            [source + 1 for source in _SOURCE_FRAMES],
        )
        self.assertEqual(seen["transform"].freqs.shape[1], 12)
        self.assertEqual(seen["transform"].key_freqs.shape[1], prefix + 7 * 4)

    def test_fast_falls_back_to_exact_materialization_without_mapped_kernel(self):
        seen, _, _ = self._run("fast")
        self.assertEqual(seen["key"].shape[2], 32)
        self.assertEqual(seen["key"].shape, seen["value"].shape)
        self.assertIsNot(seen["transform"].freqs, seen["transform"].key_freqs)
        self.assertEqual(seen["transform"].key_freqs.shape[1], 32)

    def test_virtual_input_cache_accepts_inference_tensors(self):
        with torch.inference_mode():
            prepared, prefix, stop = request()
            query_freqs = prepared.qk_transform.freqs
            with self.assertRaisesRegex(RuntimeError, "version counter"):
                _ = query_freqs._version
            first = _cached_virtual_inputs(
                prepared, query_freqs, prefix, stop, tokens_per_frame=4
            )
            second = _cached_virtual_inputs(
                prepared, query_freqs, prefix, stop, tokens_per_frame=4
            )

        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

    def test_fast_uses_physical_kv_with_exact_logical_map(self):
        seen = {}

        def dense_executor(prepared):
            raise AssertionError("mapped fast path should bypass materialization")

        def mapped_executor(prepared, source_indices):
            seen["query"], seen["key"], seen["value"] = prepared.peek_qkv()
            seen["transform"] = prepared.qk_transform
            seen["source_indices"] = source_indices
            prepared.consume_qkv()
            return AttentionExecutionOutcome(
                torch.zeros((1, prepared.query_tokens, 2 * 128))
            )

        dense_executor.turing_utils_mapped_kv_executor = mapped_executor

        def dense_override(original, *args, **kwargs):
            return original(*args, **kwargs)

        dense_override.prepared_attention_executor = dense_executor
        prepared, prefix, _ = request()
        capabilities = SimpleNamespace(
            supports=lambda feature: SimpleNamespace(supported=feature == "mapped_kv")
        )
        with mock.patch(
            "comfyui_turing_utils.adapters.minimax.virtual_kv.kernel_capabilities",
            return_value=capabilities,
        ):
            outcome = make_h3_virtual_kv_override(
                dense_override, mode="fast"
            ).prepared_attention_executor(prepared)

        self.assertTrue(outcome.supported)
        self.assertEqual(seen["query"].shape, seen["key"].shape)
        self.assertEqual(seen["key"].shape, seen["value"].shape)
        self.assertEqual(seen["transform"].key_freqs.shape[1], prefix + 7 * 4)
        target = seen["source_indices"][prefix:].reshape(7, 4)
        self.assertEqual(
            [int(row[0].item() - prefix) // 4 for row in target],
            list(_SOURCE_FRAMES),
        )

    def test_residual_keeps_real_kv_physical_and_protects_two_real_slices(self):
        seen = {}

        def dense_executor(prepared):
            raise AssertionError("mapped residual path should bypass materialization")

        def residual_executor(prepared, source_indices, **policy):
            seen["query"], seen["key"], seen["value"] = prepared.peek_qkv()
            seen["transform"] = prepared.qk_transform
            seen["source_indices"] = source_indices
            seen["policy"] = policy
            prepared.consume_qkv()
            return AttentionExecutionOutcome(
                torch.zeros((1, prepared.query_tokens, 2 * 128))
            )

        dense_executor.turing_utils_mapped_residual_executor = residual_executor

        def dense_override(original, *args, **kwargs):
            return original(*args, **kwargs)

        dense_override.prepared_attention_executor = dense_executor
        prepared, prefix, stop = request()
        capabilities = SimpleNamespace(
            supports=lambda feature: SimpleNamespace(
                supported=feature == "mapped_sparse_kv"
            )
        )
        with mock.patch(
            "comfyui_turing_utils.adapters.minimax.virtual_kv.kernel_capabilities",
            return_value=capabilities,
        ):
            outcome = make_h3_virtual_kv_override(
                dense_override, mode="residual"
            ).prepared_attention_executor(prepared)

        self.assertTrue(outcome.supported)
        self.assertEqual(seen["key"].shape[2], stop)
        self.assertEqual(seen["value"].shape[2], stop)
        self.assertEqual(seen["transform"].key_freqs.shape[1], prefix + 7 * 4)
        self.assertEqual(seen["policy"]["exact_kv_ranges"], ((0, stop),))
        self.assertEqual(seen["policy"]["residual_subblocks"], 2)
        self.assertEqual(seen["policy"]["routing_threshold"], 1_000_000.0)

    def test_residual_falls_back_to_exact_materialization_without_new_kernel(self):
        seen, prefix, _ = self._run("residual")
        self.assertEqual(seen["key"].shape[2], prefix + 7 * 4)
        self.assertEqual(seen["key"].shape, seen["value"].shape)

    def test_residual_uses_mapped_fp16_sol_for_sdpa_or_sage_base(self):
        seen = {}

        def dense_executor(prepared):
            raise AssertionError("mapped FP16 residual path should bypass materialization")

        def residual_executor(prepared, source_indices, **policy):
            seen["key_tokens"] = prepared.key_tokens
            seen["source_indices"] = source_indices
            seen["policy"] = policy
            prepared.consume_qkv()
            return AttentionExecutionOutcome(
                torch.zeros((1, prepared.query_tokens, 2 * 128))
            )

        residual_executor.turing_utils_mapped_residual_capability = (
            "mapped_sparse_fp16_kv"
        )
        dense_executor.turing_utils_mapped_residual_executor = residual_executor

        def dense_override(original, *args, **kwargs):
            return original(*args, **kwargs)

        dense_override.prepared_attention_executor = dense_executor
        dense_override.turing_utils_attention_backend = "sdpa"
        prepared, prefix, stop = request()
        capabilities = SimpleNamespace(
            supports=lambda feature: SimpleNamespace(
                supported=feature == "mapped_sparse_fp16_kv"
            )
        )
        with mock.patch(
            "comfyui_turing_utils.adapters.minimax.virtual_kv.kernel_capabilities",
            return_value=capabilities,
        ):
            override = make_h3_virtual_kv_override(dense_override, mode="residual")
            outcome = override.prepared_attention_executor(prepared)

        self.assertTrue(outcome.supported)
        self.assertEqual(seen["key_tokens"], stop)
        self.assertEqual(seen["source_indices"].numel(), prefix + 7 * 4)
        self.assertEqual(seen["policy"]["exact_kv_ranges"], ((0, stop),))
        self.assertEqual(
            override.prepared_attention_executor.turing_utils_h3_virtual_kv_mapped_capability,
            "mapped_sparse_fp16_kv",
        )
        self.assertEqual(
            override.prepared_attention_executor.turing_utils_h3_virtual_kv_numeric_backend,
            "sdpa",
        )

    def test_non_five_frame_target_fails_before_dense_execution(self):
        called = False

        def dense_executor(prepared):
            nonlocal called
            called = True
            return AttentionExecutionOutcome.unsupported("unexpected")

        def dense_override(original, *args, **kwargs):
            return original(*args, **kwargs)

        dense_override.prepared_attention_executor = dense_executor
        prepared, _, _ = request(frames=3)
        executor = make_h3_virtual_kv_override(
            dense_override
        ).prepared_attention_executor
        with self.assertRaisesRegex(RuntimeError, "5-frame input"):
            executor(prepared)
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
