from __future__ import annotations

import sys
import unittest
import weakref
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.adapters.methods import (  # noqa: E402
    OriginalMethod,
    weak_method,
)


class AdapterMethodTest(unittest.TestCase):
    def test_original_method_preserves_bound_call_semantics(self):
        class Owner:
            def call(self, value, *, offset=0):
                return self.base + value + offset

        owner = Owner()
        owner.base = 3
        original = OriginalMethod.capture(owner.call, owner)

        self.assertEqual(original(owner, 4, offset=5), 12)

    def test_patched_method_does_not_retain_its_owner(self):
        class Owner:
            def call(self, value):
                return value + 1

        owner = Owner()
        original = OriginalMethod.capture(owner.call, owner)

        def replacement(self, value):
            return original(self, value) * 2

        owner.call = weak_method(replacement, owner)
        self.assertEqual(owner.call(4), 10)

        owner_ref = weakref.ref(owner)
        del owner
        self.assertIsNone(owner_ref())


if __name__ == "__main__":
    unittest.main()
