from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

PACKAGE_NAME = "comfyui_svdint4_testpkg"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PLUGIN_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)
spec = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.seedvr2_vae", PLUGIN_ROOT / "seedvr2_vae.py")
seedvr2_vae = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = seedvr2_vae
assert spec.loader is not None
spec.loader.exec_module(seedvr2_vae)


class SeedVR2TilingTest(unittest.TestCase):
    def test_tile_starts_anchor_final_tile(self):
        starts = seedvr2_vae._tile_starts(length=300, tile=128, overlap=16)

        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 172)
        self.assertTrue(all(a < b for a, b in zip(starts, starts[1:])))

    def test_tile_starts_cover_without_tiny_edge_tile(self):
        length = 300
        tile = 128
        overlap = 16
        starts = seedvr2_vae._tile_starts(length=length, tile=tile, overlap=overlap)
        covered = [0] * length
        for start in starts:
            for index in range(start, min(start + tile, length)):
                covered[index] += 1

        self.assertTrue(all(value > 0 for value in covered))
        self.assertGreaterEqual(min(b - a for a, b in zip(starts, starts[1:])), 1)
        self.assertGreaterEqual(starts[-1] + tile, length)

    def test_tile_starts_single_tile_when_tile_fits(self):
        self.assertEqual(seedvr2_vae._tile_starts(length=64, tile=128, overlap=16), [0])


if __name__ == "__main__":
    unittest.main()
