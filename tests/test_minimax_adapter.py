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

import minimax_adapter  # noqa: E402


def _convrot_weight(kind: str):
    from comfy.quant_ops import (
        QuantizedTensor,
        TensorCoreConvRotW4A4Layout,
        TensorWiseINT8Layout,
    )

    if kind == "w8a8":
        params = TensorWiseINT8Layout.Params(
            scale=torch.ones(4, dtype=torch.float32),
            orig_dtype=torch.bfloat16,
            orig_shape=(4, 256),
            convrot=True,
            convrot_groupsize=256,
        )
        return QuantizedTensor(
            torch.zeros((4, 256), dtype=torch.int8),
            "TensorWiseINT8Layout",
            params,
        )

    linear_dtype = {"w4a4": "int4", "w4a8": "int8"}[kind]
    params = TensorCoreConvRotW4A4Layout.Params(
        scale=torch.ones((4, 4), dtype=torch.float32),
        orig_dtype=torch.bfloat16,
        orig_shape=(4, 256),
        convrot_groupsize=256,
        quant_group_size=64,
        linear_dtype=linear_dtype,
    )
    return QuantizedTensor(
        torch.zeros((4, 128), dtype=torch.uint8),
        "TensorCoreConvRotW4A4Layout",
        params,
    )


class FakePatcher:
    def __init__(self, root: torch.nn.Module):
        self.model = root
        self.object_patches = {}

    def add_object_patch(self, name, value):
        self.object_patches[name] = value


class MiniMaxAdapterTest(unittest.TestCase):
    @staticmethod
    def _types(kind: str):
        class FakeMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Identity()
                self.fc2 = SimpleNamespace(weight=_convrot_weight(kind))

            def forward(self, x):
                return x

        class FakeBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = torch.nn.Identity()
                self.norm2 = torch.nn.Identity()
                self.adaln_proj = torch.nn.Identity()
                self.attn = torch.nn.Identity()
                self.mlp = FakeMLP()

            def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
                return x

        return FakeBlock, FakeMLP

    def test_all_convrot_formats_install_object_patches_without_mutating_modules(self):
        import comfy.ldm.minimax.model as minimax_model

        for kind in ("w8a8", "w4a4", "w4a8"):
            with self.subTest(kind=kind):
                FakeBlock, _ = self._types(kind)
                root = torch.nn.Module()
                root.block = FakeBlock()
                patcher = FakePatcher(root)
                original_block_forward = root.block.forward
                original_mlp_forward = root.block.mlp.forward
                with (
                    mock.patch("minimax_adapter.is_supported_turing_device", return_value=True),
                    mock.patch.object(minimax_model, "DiTBlock", FakeBlock),
                    mock.patch.dict(
                        sys.modules,
                        {"comfyui_turing_utils_kernel": SimpleNamespace(turing_segmented_rms_adaln=mock.Mock())},
                    ),
                ):
                    count = minimax_adapter.apply_minimax_adapter(
                        patcher, torch.device("cuda", 0)
                    )

                self.assertEqual(count, 1)
                self.assertEqual(
                    set(patcher.object_patches),
                    {"block.forward", "block.mlp.forward"},
                )
                self.assertIs(root.block.forward.__func__, original_block_forward.__func__)
                self.assertIs(root.block.mlp.forward.__func__, original_mlp_forward.__func__)

    def test_changed_block_contract_disables_adapter_cleanly(self):
        import comfy.ldm.minimax.model as minimax_model

        class ChangedBlock(torch.nn.Module):
            def forward(self, x, new_required_argument):
                return x

        root = torch.nn.Module()
        root.block = ChangedBlock()
        patcher = FakePatcher(root)
        with (
            mock.patch("minimax_adapter.is_supported_turing_device", return_value=True),
            mock.patch.object(minimax_model, "DiTBlock", ChangedBlock),
            self.assertLogs("comfyui-turing-utils", level="WARNING"),
        ):
            count = minimax_adapter.apply_minimax_adapter(
                patcher, torch.device("cuda", 0)
            )

        self.assertEqual(count, 0)
        self.assertFalse(patcher.object_patches)

    def test_runtime_audit_reports_a_complete_fused_window_once(self):
        audit = minimax_adapter._RuntimeDispatchAudit(expected_blocks=2, expected_mlps=2)
        x = torch.zeros((3, 256), dtype=torch.bfloat16)
        with self.assertLogs("comfyui-turing-utils", level="INFO") as captured:
            audit.record("block", True, x)
            audit.record("mlp", True, x)
            audit.record("block", True, x)
            audit.record("mlp", True, x)
            audit.record("mlp", False, x, "should_not_be_recorded")

        messages = "\n".join(captured.output)
        self.assertIn("phase=block fused=2 fallback=0", messages)
        self.assertIn("phase=mlp fused=2 fallback=0", messages)
        self.assertNotIn("should_not_be_recorded", messages)

    def test_runtime_audit_exposes_mlp_dtype_fallback(self):
        FakeBlock, _ = self._types("w8a8")
        mlp = FakeBlock().mlp
        audit = minimax_adapter._RuntimeDispatchAudit(expected_blocks=0, expected_mlps=1)
        patched = minimax_adapter._make_mlp_forward(mlp, audit)
        x = torch.zeros((3, 256), dtype=torch.float32)

        with self.assertLogs("comfyui-turing-utils", level="WARNING") as captured:
            output = patched(x)

        self.assertIs(output, x)
        self.assertIn("phase=mlp fused=0 fallback=1", "\n".join(captured.output))
        self.assertIn("dtype=torch.float32", "\n".join(captured.output))

    def test_runtime_audit_keeps_fused_w8a8_mlp_dispatch(self):
        FakeBlock, _ = self._types("w8a8")
        mlp = FakeBlock().mlp
        audit = minimax_adapter._RuntimeDispatchAudit(expected_blocks=0, expected_mlps=1)
        patched = minimax_adapter._make_mlp_forward(mlp, audit)
        x = torch.zeros((3, 256), dtype=torch.bfloat16)
        sentinel = object()

        with (
            mock.patch(
                "minimax_adapter.turing_linear_input_act",
                return_value=sentinel,
            ) as fused,
            self.assertLogs("comfyui-turing-utils", level="INFO") as captured,
        ):
            output = patched(x)

        self.assertIs(output, sentinel)
        fused.assert_called_once_with(mlp.fc2, x, "swiglu")
        self.assertIn("phase=mlp fused=1 fallback=0", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
