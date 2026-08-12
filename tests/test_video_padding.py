from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.nodes.minimax import MiniMaxH3VideoFramesPadding  # noqa: E402
from comfyui_turing_utils.nodes.wan import WanVideoFramesPadding  # noqa: E402


class VideoPaddingTest(unittest.TestCase):
    def test_wan_rounds_up_to_four_n_plus_one(self):
        image = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1, 1).repeat(1, 2, 2, 3)
        output = WanVideoFramesPadding().pad(image, 0)
        self.assertEqual(output[4], 9)
        self.assertEqual(output[5], 6)
        self.assertTrue(torch.equal(output[0][-1], image[-1]))

    def test_h3_rounds_up_to_seventeen_n_plus_five_with_mask(self):
        image = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1, 1).repeat(1, 2, 2, 3)
        mask = torch.zeros(6, 2, 2)
        mask[-1] = 1.0
        output = MiniMaxH3VideoFramesPadding().pad(image, 0, mask)
        self.assertEqual(output[4], 22)
        self.assertEqual(output[5], 6)
        self.assertTrue(torch.equal(output[0][-1], image[-1]))
        self.assertTrue(torch.equal(output[1][-1], mask[-1]))


if __name__ == "__main__":
    unittest.main()

