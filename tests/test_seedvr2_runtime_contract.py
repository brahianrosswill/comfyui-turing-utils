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
import comfy.ops  # noqa: E402


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

    def test_flow_sampling_exposes_scheduler_sigmas(self):
        sampling = seedvr2._SeedVR2FlowSampling()

        self.assertEqual(tuple(sampling.sigmas.shape), (1000,))
        self.assertAlmostEqual(float(sampling.sigmas[0]), 0.001, places=6)
        self.assertAlmostEqual(float(sampling.sigmas[-1]), 1.0, places=6)
        self.assertAlmostEqual(float(sampling.sigma_min), float(sampling.sigmas[0]), places=6)
        self.assertAlmostEqual(float(sampling.sigma_max), float(sampling.sigmas[-1]), places=6)
        self.assertEqual(float(sampling.percent_to_sigma(0.0)), 1.0)
        self.assertEqual(float(sampling.percent_to_sigma(1.0)), 0.0)

    def test_runtime_dtype_follows_non_fp8_checkpoint_weights(self):
        module = torch.nn.Module()
        module.fp16 = torch.nn.Linear(4, 4, dtype=torch.float16)
        module.fp32 = torch.nn.LayerNorm(4, dtype=torch.float32)

        self.assertEqual(seedvr2._infer_runtime_dtype(module, torch.bfloat16), torch.float16)

    def test_patchable_vae_device_syncs_to_core(self):
        class _Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._device = torch.device("cpu")

            @property
            def device(self):
                return self._device

            @device.setter
            def device(self, value):
                self._device = torch.device(value)

        core = _Core()
        wrapper = seedvr2._PatchableVAE(core, torch.device("cpu"))

        wrapper.device = torch.device("cuda:0")

        self.assertEqual(wrapper.device, torch.device("cuda:0"))
        self.assertEqual(core.device, torch.device("cuda:0"))

    def test_vae_tile_candidates_are_conservative(self):
        candidates = seedvr2._vae_tile_candidates(1024)

        self.assertEqual(candidates[0], (1024, 256))
        self.assertIn((256, 64), candidates)
        self.assertNotIn((1536, 384), candidates)

    def test_vae_paging_guard_rejects_near_limit_headroom(self):
        vae = seedvr2.SeedVR2VAE.__new__(seedvr2.SeedVR2VAE)
        vae.model = types.SimpleNamespace(device=torch.device("cuda:0"))
        vae.patcher = types.SimpleNamespace(load_device=torch.device("cuda:0"))

        memory_required = 1024 * 1024 * 1024
        required_headroom = seedvr2._vae_required_headroom(memory_required)
        original_get_free_memory = seedvr2.model_management.get_free_memory
        try:
            seedvr2.model_management.get_free_memory = lambda _device: required_headroom - 1
            self.assertFalse(vae._has_execution_headroom(memory_required, "test"))

            seedvr2.model_management.get_free_memory = lambda _device: required_headroom
            self.assertTrue(vae._has_execution_headroom(memory_required, "test"))
        finally:
            seedvr2.model_management.get_free_memory = original_get_free_memory

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "torch build has no float8 dtype")
    def test_fp8_storage_model_reports_runtime_dtype(self):
        class _FP8StorageModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.empty((1, 1), dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                )

        model = SeedVR2ComfyModel(
            dit=_FP8StorageModule(),
            na_module=_DummyNa(),
            text_pos=torch.zeros(1, 1),
            text_neg=torch.zeros(1, 1),
            dtype=torch.float16,
        )

        self.assertEqual(model.get_dtype(), torch.float16)
        self.assertEqual(model.model_dtype(), torch.float16)

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "torch build has no float8 dtype")
    def test_runtime_dtype_uses_fallback_for_fp8_only_weights(self):
        class _FP8StorageModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.empty((4, 4), dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                )

        self.assertEqual(seedvr2._infer_runtime_dtype(_FP8StorageModule(), torch.bfloat16), torch.bfloat16)

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "torch build has no float8 dtype")
    def test_non_linear_fp8_tensors_are_cast_without_expanding_linear_storage(self):
        class _MixedFP8Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = comfy.ops.manual_cast.Linear(2, 2, bias=False, dtype=torch.float16)
                self.linear.weight = torch.nn.Parameter(
                    torch.empty((2, 2), dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                )
                self.linear.weight_comfy_model_dtype = torch.float8_e4m3fn
                self.norm = torch.nn.Module()
                self.norm.weight = torch.nn.Parameter(
                    torch.empty((2,), dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                )
                self.register_buffer("scale", torch.empty((1,), dtype=torch.float8_e4m3fn))

        model = _MixedFP8Module()
        converted, _converted_bytes = seedvr2._cast_unhandled_fp8_tensors(model, torch.float16)

        self.assertEqual(converted, 2)
        self.assertEqual(model.linear.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(model.linear.weight_comfy_model_dtype, torch.float8_e4m3fn)
        self.assertEqual(model.norm.weight.dtype, torch.float16)
        self.assertEqual(model.norm.weight_comfy_model_dtype, torch.float16)
        self.assertEqual(model.scale.dtype, torch.float16)
        self.assertEqual(model.scale_comfy_model_dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
