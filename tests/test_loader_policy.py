from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

import loader  # noqa: E402


class _FakeRoot:
    def __init__(self):
        self.linear = SimpleNamespace(is_svdint4=True, weight_function=None)

    def named_modules(self):
        return (("linear", self.linear),)


class _FakePatcher:
    def __init__(self):
        self.model = _FakeRoot()
        self.model_options = {}
        self.compute_dtype = None
        self.callbacks = []
        self.cached_patcher_init = None

    def model_size(self):
        return 1024

    def is_dynamic(self):
        return True

    def add_callback(self, event, callback):
        self.callbacks.append((event, callback))

    def set_model_compute_dtype(self, dtype):
        self.compute_dtype = dtype


class SVDInt4LoaderPolicyTest(unittest.TestCase):
    def test_bf16_selection_is_scoped_to_comfy_model_loading(self):
        fake_model = _FakePatcher()
        model_config = SimpleNamespace(supported_inference_dtypes=[torch.bfloat16, torch.float32])
        custom_ops = object()
        load_device = torch.device("cuda", 0)

        with (
            mock.patch.object(loader, "_HAS_COMFY_QUANTIZED_TENSOR", True),
            mock.patch.object(loader, "_install_svdint4_patch_filter"),
            mock.patch.object(loader, "_install_svdint4_lora_key_map"),
            mock.patch.object(loader, "build_loader_state_dict", return_value=({"model.key": torch.zeros(1)}, {}, {})),
            mock.patch("comfy.model_detection.unet_prefix_from_state_dict", return_value="model."),
            mock.patch("comfy.model_detection.model_config_from_unet", return_value=model_config),
            mock.patch("comfy.model_management.get_torch_device", return_value=load_device),
            mock.patch.object(loader, "select_compute_dtype", return_value=torch.bfloat16) as select_dtype,
            mock.patch.object(loader, "SVDInt4Ops", return_value=custom_ops),
            mock.patch("comfy.sd.load_diffusion_model_state_dict", return_value=fake_model) as load_state,
            mock.patch.object(loader, "apply_attention_backend") as apply_attention,
            mock.patch.object(loader, "_load_svdint4_linear") as load_kernel,
        ):
            output = loader.load_svdint4_model("model.safetensors")

        self.assertIs(output, fake_model)
        select_dtype.assert_called_once()
        self.assertEqual(
            load_state.call_args.kwargs["model_options"],
            {"custom_operations": custom_ops, "dtype": torch.bfloat16},
        )
        self.assertIs(fake_model.compute_dtype, torch.bfloat16)
        apply_attention.assert_called_once_with(fake_model, "auto", device=load_device)
        load_kernel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
