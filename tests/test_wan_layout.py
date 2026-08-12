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

from comfyui_turing_utils.adapters.wan_layout import (  # noqa: E402
    WAN_LAYOUT_KIND,
    build_wan_attention_layout,
    publish_wan_attention_layout,
)
from comfyui_turing_utils.attention.layout import (  # noqa: E402
    ATTENTION_LAYOUT_KEY,
    attention_semantic_layout,
    has_complete_attention_layout,
)
from comfyui_turing_utils.attention.sparse import _sparse_protected_ranges  # noqa: E402


class WanLayoutTest(unittest.TestCase):
    @staticmethod
    def model():
        return SimpleNamespace(
            patch_size=(1, 2, 2),
            ref_conv=object(),
            blocks=[object(), object(), object()],
        )

    @staticmethod
    def inputs():
        return (
            torch.zeros((1, 16, 3, 8, 10)),
            {
                "reference_latent": torch.zeros((1, 16, 8, 10)),
                "context_latents": [
                    torch.zeros((1, 16, 1, 8, 10)),
                    torch.zeros((1, 16, 4, 8, 10)),
                ],
            },
        )

    def test_sequence_matches_wan_forward_orig_order(self):
        x, kwargs = self.inputs()
        layout = build_wan_attention_layout(self.model(), x, {}, kwargs)
        self.assertEqual(layout.provider, WAN_LAYOUT_KIND)
        self.assertEqual(
            tuple((item.start, item.stop, item.role) for item in layout.query_segments),
            (
                (0, 20, "reference_image"),
                (20, 80, "target_video"),
                (80, 100, "reference_image"),
                (100, 120, "reference_video_anchor"),
                (120, 160, "reference_video"),
                (160, 180, "reference_video_anchor"),
            ),
        )
        self.assertEqual(
            tuple((item.topology_id, item.start, item.stop) for item in layout.topologies),
            (("target_video", 20, 80), ("context_video_1", 100, 180)),
        )
        self.assertIsNone(layout.validate(180, 180))

    def test_versioned_wire_drives_reference_anchor_policy(self):
        x, kwargs = self.inputs()
        options = {ATTENTION_LAYOUT_KEY: {"extension": "preserved"}}
        self.assertTrue(publish_wan_attention_layout(self.model(), x, options, kwargs))
        self.assertEqual(options[ATTENTION_LAYOUT_KEY]["extension"], "preserved")
        self.assertTrue(
            has_complete_attention_layout(options, 180, provider=WAN_LAYOUT_KIND)
        )
        self.assertEqual(attention_semantic_layout(options).protocol_version, 1)
        self.assertEqual(
            _sparse_protected_ranges(
                "auto",
                0,
                options,
                180,
                sparse_reference_image=False,
                sparse_reference_video=True,
                sparse_reference_audio=False,
            ),
            ((0, 20), (80, 120), (160, 180)),
        )

    def test_invalid_context_clears_stale_complete_layout(self):
        x, kwargs = self.inputs()
        options = {}
        self.assertTrue(publish_wan_attention_layout(self.model(), x, options, kwargs))
        kwargs["context_latents"].append(torch.zeros((1, 16, 8, 10)))
        self.assertFalse(publish_wan_attention_layout(self.model(), x, options, kwargs))
        self.assertFalse(has_complete_attention_layout(options, 180))
        self.assertEqual(
            options[ATTENTION_LAYOUT_KEY],
            {"provider": WAN_LAYOUT_KIND},
        )


if __name__ == "__main__":
    unittest.main()
