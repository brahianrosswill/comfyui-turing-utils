from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

PACKAGE_NAME = "comfyui_svdint4_testpkg"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PLUGIN_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

for module_name in ("seedvr2", "seedvr2_nodes"):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.{module_name}", PLUGIN_ROOT / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

seedvr2_nodes = sys.modules[f"{PACKAGE_NAME}.seedvr2_nodes"]


class SeedVR2NodePlanTest(unittest.TestCase):
    def test_decode_tile_is_selected_before_oom_on_22gb_video_window(self):
        tiled, tile, overlap = seedvr2_nodes._choose_vae_tiling(
            kind="decode",
            frames=5,
            height=1280,
            width=1280,
            free_gb=22.0,
        )

        self.assertTrue(tiled)
        self.assertEqual(tile, 768)
        self.assertEqual(overlap, 192)

    def test_encode_and_decode_are_planned_independently(self):
        encode = seedvr2_nodes._choose_vae_tiling(
            kind="encode",
            frames=5,
            height=1024,
            width=1024,
            free_gb=22.0,
        )
        decode = seedvr2_nodes._choose_vae_tiling(
            kind="decode",
            frames=5,
            height=1024,
            width=1024,
            free_gb=22.0,
        )

        self.assertEqual(encode, (False, 1024, 256))
        self.assertEqual(decode, (True, 768, 192))

    def test_unknown_free_memory_uses_conservative_tile(self):
        tiled, tile, overlap = seedvr2_nodes._choose_vae_tiling(
            kind="decode",
            frames=5,
            height=1280,
            width=1280,
            free_gb=0.0,
        )

        self.assertTrue(tiled)
        self.assertEqual(tile, 256)
        self.assertEqual(overlap, 64)

    def test_build_plan_uses_empirical_decode_tile(self):
        original = seedvr2_nodes._estimate_free_vram_gb
        try:
            seedvr2_nodes._estimate_free_vram_gb = lambda _device: 22.0
            image = torch.zeros((5, 1024, 1024, 3))

            plan = seedvr2_nodes._build_plan(image, resolution=1024, max_resolution=1024, batch_size=0)
        finally:
            seedvr2_nodes._estimate_free_vram_gb = original

        self.assertEqual(plan.batch_size, 5)
        self.assertFalse(plan.encode_tiled)
        self.assertTrue(plan.decode_tiled)
        self.assertEqual(plan.decode_tile_size, 768)

    def test_fallback_does_not_force_unneeded_encode_tiling(self):
        plan = seedvr2_nodes.SeedVR2Plan(
            batch_size=5,
            encode_tiled=False,
            encode_tile_size=1024,
            encode_tile_overlap=256,
            decode_tiled=True,
            decode_tile_size=768,
            decode_tile_overlap=192,
        )

        fallbacks = seedvr2_nodes._fallback_plans(plan)

        self.assertTrue(any(item.decode_tile_size == 512 for item in fallbacks))
        self.assertTrue(all(not item.encode_tiled for item in fallbacks))
        self.assertTrue(all(item.encode_tile_size == 1024 for item in fallbacks))


if __name__ == "__main__":
    unittest.main()
