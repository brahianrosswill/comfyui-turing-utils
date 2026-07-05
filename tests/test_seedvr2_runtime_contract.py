from __future__ import annotations

import sys
import types
import unittest
import importlib.util
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import comfy.model_patcher  # noqa: E402


PACKAGE_NAME = "comfyui_svdint4_testpkg"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PLUGIN_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)
spec = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.seedvr2", PLUGIN_ROOT / "seedvr2.py")
seedvr2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = seedvr2
assert spec.loader is not None
spec.loader.exec_module(seedvr2)
SeedVR2ComfyModel = seedvr2.SeedVR2ComfyModel


class _DummyNa:
    @staticmethod
    def flatten(tensors):
        tensor = tensors[0]
        return tensor.reshape(-1, tensor.shape[-1]), torch.tensor([tensor.shape[:-1]], dtype=torch.long)

    @staticmethod
    def unflatten(tensor, shape):
        target = tuple(int(value) for value in shape[0].tolist())
        return [tensor.reshape(*target, tensor.shape[-1])]


class SeedVR2RuntimeContractTest(unittest.TestCase):
    def _make_model(self):
        return SeedVR2ComfyModel(
            dit=torch.nn.Linear(1, 1),
            na_module=_DummyNa(),
            text_pos=torch.zeros(1, 1),
            text_neg=torch.zeros(1, 1),
            dtype=torch.float32,
        )

    def test_comfy_model_protocol_fields_exist(self):
        model = self._make_model()

        self.assertTrue(hasattr(model, "device"))
        self.assertTrue(hasattr(model, "current_patcher"))
        self.assertTrue(hasattr(model, "model_options"))
        self.assertTrue(hasattr(model, "model_sampling"))
        self.assertTrue(hasattr(model, "latent_format"))
        self.assertTrue(callable(model.apply_model))
        self.assertTrue(callable(model.memory_required))
        self.assertTrue(callable(model.process_latent_in))
        self.assertTrue(callable(model.process_latent_out))

    def test_comfy_sampler_current_patcher_contract(self):
        model = self._make_model()
        patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
        )

        self.assertIsNone(model.current_patcher)
        patcher.pre_run()
        self.assertIs(model.current_patcher, patcher)
        patcher.cleanup()
        self.assertIsNone(model.current_patcher)


if __name__ == "__main__":
    unittest.main()
