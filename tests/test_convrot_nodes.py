from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from convrot_nodes import ConvRotSummary, _validate_runtime_support, configure_convrot_activation  # noqa: E402


def quant_tensor(config: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)


def quant_config(value: torch.Tensor) -> dict:
    return json.loads(value.numpy().tobytes())


class ConvRotActivationTest(unittest.TestCase):
    def test_auto_uses_w4_format_default(self):
        metadata = {
            "_quantization_metadata": json.dumps(
                {"layers": {"blocks.0.ffn.0": {"format": "convrot_w4a4"}}}
            )
        }

        updated, summary = configure_convrot_activation({}, metadata, "auto")

        config = json.loads(updated["_quantization_metadata"])["layers"]["blocks.0.ffn.0"]
        self.assertNotIn("linear_dtype", config)
        self.assertEqual(summary, ConvRotSummary(w4a4=1))

    def test_w4_override_updates_header_and_tensor_metadata(self):
        metadata = {
            "_quantization_metadata": json.dumps(
                {"layers": {"blocks.0.ffn.0": {"format": "convrot_w4a4"}}}
            )
        }
        state_dict = {
            "blocks.1.ffn.0.comfy_quant": quant_tensor(
                {"format": "convrot_w4a4", "linear_dtype": "int4"}
            )
        }

        updated, summary = configure_convrot_activation(state_dict, metadata, "int8")

        header_config = json.loads(updated["_quantization_metadata"])["layers"]["blocks.0.ffn.0"]
        self.assertEqual(header_config["linear_dtype"], "int8")
        self.assertEqual(quant_config(state_dict["blocks.1.ffn.0.comfy_quant"])["linear_dtype"], "int8")
        self.assertEqual(summary, ConvRotSummary(w4a8=2))

    def test_int4_rejects_w8_convrot(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor(
                {"format": "int8_tensorwise", "convrot": True}
            )
        }

        with self.assertRaisesRegex(ValueError, "Cannot run ConvRot W8 weights with INT4 activations"):
            configure_convrot_activation(state_dict, None, "int4")

    def test_int8_accepts_mixed_w4_and_w8_convrot(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor({"format": "convrot_w4a4"}),
            "blocks.1.ffn.0.comfy_quant": quant_tensor(
                {"format": "int8_tensorwise", "params": {"convrot": True}}
            ),
        }

        _, summary = configure_convrot_activation(state_dict, None, "int8")

        self.assertEqual(summary, ConvRotSummary(w4a8=1, w8a8=1))

    def test_auto_preserves_explicit_w4a8(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor(
                {"format": "convrot_w4a4", "params": {"linear_dtype": "int8"}}
            )
        }

        _, summary = configure_convrot_activation(state_dict, None, "auto")

        self.assertEqual(summary, ConvRotSummary(w4a8=1))

    def test_int4_overrides_explicit_w4a8(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor(
                {"format": "convrot_w4a4", "linear_dtype": "int8"}
            )
        }

        _, summary = configure_convrot_activation(state_dict, None, "int4")

        self.assertEqual(quant_config(state_dict["blocks.0.ffn.0.comfy_quant"])["linear_dtype"], "int4")
        self.assertEqual(summary, ConvRotSummary(w4a4=1))

    def test_non_convrot_model_is_rejected(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor({"format": "int8_tensorwise"})
        }

        with self.assertRaisesRegex(ValueError, "does not contain supported ConvRot"):
            configure_convrot_activation(state_dict, None, "auto")

    def test_unsupported_convrot_format_is_rejected(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor(
                {"format": "float8_e4m3fn", "convrot": True}
            )
        }

        with self.assertRaisesRegex(ValueError, "unsupported weight format"):
            configure_convrot_activation(state_dict, None, "auto")

    def test_w8_metadata_cannot_declare_int4_activations(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor(
                {"format": "int8_tensorwise", "convrot": True, "linear_dtype": "int4"}
            )
        }

        with self.assertRaisesRegex(ValueError, "W8 ConvRot supports INT8 activations only"):
            configure_convrot_activation(state_dict, None, "auto")

    def test_auto_rejects_conflicting_duplicate_metadata(self):
        metadata = {
            "_quantization_metadata": json.dumps(
                {
                    "layers": {
                        "blocks.0.ffn.0": {
                            "format": "convrot_w4a4",
                            "linear_dtype": "int8",
                        }
                    }
                }
            )
        }
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor({"format": "convrot_w4a4"})
        }

        with self.assertRaisesRegex(ValueError, "Conflicting ConvRot metadata"):
            configure_convrot_activation(state_dict, metadata, "auto")

    def test_invalid_header_json_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contains invalid JSON"):
            configure_convrot_activation({}, {"_quantization_metadata": "{"}, "auto")

    def test_invalid_tensor_json_is_rejected(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": torch.tensor(list(b"{"), dtype=torch.uint8)
        }

        with self.assertRaisesRegex(ValueError, "contains invalid quantization JSON"):
            configure_convrot_activation(state_dict, None, "auto")

    def test_invalid_activation_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported ConvRot activation_dtype"):
            configure_convrot_activation({}, None, "fp16")

    def test_w4a8_rejects_disabled_cuda_backend(self):
        backends = {
            "cuda": {
                "available": True,
                "disabled": True,
                "unavailable_reason": None,
            }
        }
        with mock.patch("comfy_kitchen.list_backends", return_value=backends):
            with self.assertRaisesRegex(RuntimeError, "requires the comfy-kitchen CUDA backend"):
                _validate_runtime_support(ConvRotSummary(w4a8=1))

    def test_w4a8_accepts_enabled_turing_cuda_backend(self):
        backends = {
            "cuda": {
                "available": True,
                "disabled": False,
                "unavailable_reason": None,
            }
        }
        with (
            mock.patch("comfy_kitchen.list_backends", return_value=backends),
            mock.patch("comfy.model_management.get_torch_device", return_value=torch.device("cuda", 0)),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        ):
            _validate_runtime_support(ConvRotSummary(w4a8=1))


if __name__ == "__main__":
    unittest.main()
