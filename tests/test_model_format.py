from __future__ import annotations

import sys
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from model_format import SVDINT4_FORMAT, validate_svdint4_metadata  # noqa: E402


class ModelFormatTest(unittest.TestCase):
    def test_unique_format_and_wan_architecture(self):
        self.assertEqual(SVDINT4_FORMAT, "svdint4")
        self.assertEqual(validate_svdint4_metadata({"format": "svdint4", "architecture": "wan"}), "wan")

    def test_old_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 'svdint4'"):
            validate_svdint4_metadata({"format": "svdint4-dit-single-v2", "architecture": "wan"})

    def test_architecture_is_required(self):
        with self.assertRaisesRegex(ValueError, "architecture=None"):
            validate_svdint4_metadata({"format": "svdint4"})


if __name__ == "__main__":
    unittest.main()
