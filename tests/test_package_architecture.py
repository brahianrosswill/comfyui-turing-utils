from __future__ import annotations

import ast
import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PackageArchitectureTest(unittest.TestCase):
    def test_kernel_package_is_accessed_only_through_facade(self):
        package = ROOT / "comfyui_turing_utils"
        offenders = []
        for path in package.rglob("*.py"):
            if path.name == "kernel_api.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                if any(name.startswith("comfyui_turing_utils_kernel") for name in modules):
                    offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_attention_depends_on_layout_contract_not_minimax_adapter(self):
        attention_root = ROOT / "comfyui_turing_utils" / "attention"
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in attention_root.glob("*.py")
        )
        self.assertNotIn("adapters.minimax", source)
        self.assertIn("ensure_attention_layout_provider", source)

    def test_builtin_model_adapters_are_registered_once(self):
        from comfyui_turing_utils.adapters.registry import registered_model_adapters

        self.assertEqual(registered_model_adapters(), ("minimax_h3", "wan"))

    def test_legacy_modules_alias_the_new_implementation(self):
        aliases = {
            "attention": "comfyui_turing_utils.attention.api",
            "minimax_adapter": "comfyui_turing_utils.adapters.minimax.acceleration",
            "minimax_layout": "comfyui_turing_utils.adapters.minimax.layout",
            "precision": "comfyui_turing_utils.precision",
            "turing_fusions": "comfyui_turing_utils.quantization.fusions",
            "turing_ops": "comfyui_turing_utils.quantization.dispatch",
            "wan_adapter": "comfyui_turing_utils.adapters.wan",
        }
        for legacy_name, implementation_name in aliases.items():
            with self.subTest(module=legacy_name):
                self.assertIs(
                    importlib.import_module(legacy_name),
                    importlib.import_module(implementation_name),
                )

    def test_node_ids_remain_stable(self):
        from comfyui_turing_utils.registration import NODE_CLASS_MAPPINGS

        self.assertEqual(
            tuple(NODE_CLASS_MAPPINGS),
            (
                "TuringUtilsConvRotDiffusionModelLoader",
                "TuringUtilsConvRotCLIPLoader",
                "TuringUtilsWanVideoFramesPadding",
                "TuringUtilsMiniMaxH3VideoFramesPadding",
                "TuringUtilsBerniniContextWindowsCore",
                "TuringUtilsBerniniInpaintCondition",
                "TuringUtilsReferenceImageHub",
                "TuringUtilsReferenceVideoHub",
                "TuringUtilsReferenceAudioHub",
                "TuringUtilsOptionalResizeImageV2",
                "TuringUtilsMiniMaxH3ReferenceConditionHub",
                "TuringUtilsH3ConcatAVLatent",
                "TuringUtilsH3SeparateAVLatent",
                "TuringUtilsMiniMaxH3LatentResize",
                "TuringUtilsMiniMaxH3ProgressiveResolutionPatch",
                "TuringUtilsSolSparseAttentionPatch",
            ),
        )


if __name__ == "__main__":
    unittest.main()
