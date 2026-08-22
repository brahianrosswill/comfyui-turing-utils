from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.adapters.memory import (  # noqa: E402
    install_memory_hooks,
    scan_quantized_workspaces,
)


class _Root:
    def __init__(self, modules):
        self._modules = modules

    def modules(self):
        return iter(self._modules)


class CommonMemoryPlanningTest(unittest.TestCase):
    def test_workspace_scan_deduplicates_serial_shapes(self):
        weights = [
            SimpleNamespace(ndim=2, shape=(4096, 5376), kind="w8a8"),
            SimpleNamespace(ndim=2, shape=(4096, 5376), kind="w8a8"),
            SimpleNamespace(ndim=2, shape=(6144, 5376), kind="codebook_w4a8"),
            SimpleNamespace(ndim=2, shape=(1, 1), kind=None),
        ]
        root = _Root([SimpleNamespace(weight=weight) for weight in weights])
        profile = scan_quantized_workspaces(root, lambda weight: weight.kind)

        self.assertEqual(dict(profile.formats), {"codebook_w4a8": 1, "w8a8": 2})
        self.assertEqual(profile.w8_output_channels, (4096,))
        self.assertEqual(len(profile.fixed_workspaces), 1)
        self.assertEqual(
            profile.transient_bytes(1024),
            max(1024 * 4096 * 4, profile.fixed_workspaces[0]),
        )

    def test_hook_installation_is_atomic_and_idempotent(self):
        base = SimpleNamespace(
            extra_conds=lambda **kwargs: {},
            extra_conds_shapes=lambda **kwargs: {},
            memory_required=lambda shape, cond_shapes={}: 0,
            memory_usage_factor_conds=("existing",),
        )
        hooks = {
            "extra_conds": lambda **kwargs: {"new": True},
            "extra_conds_shapes": lambda **kwargs: {"new": [1]},
            "memory_required": lambda shape, cond_shapes={}: 1,
        }
        first = install_memory_hooks(
            base,
            marker="_installed",
            condition_key="packed",
            **hooks,
        )
        second = install_memory_hooks(
            base,
            marker="_installed",
            condition_key="packed",
            **hooks,
        )

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(base.memory_usage_factor_conds, ("existing", "packed"))
        self.assertEqual(base.memory_required(None), 1)


if __name__ == "__main__":
    unittest.main()
