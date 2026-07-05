# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors_file

from .infer import VideoDiffusionInfer
from ..common.config import create_object
from ..utils.constants import get_script_directory


script_directory = get_script_directory()


def _is_svdint4_file(checkpoint_path: str) -> bool:
    try:
        from ....loader import is_svdint4_file
    except Exception:
        return False
    return is_svdint4_file(checkpoint_path)


def _load_svdint4_state(
    checkpoint_path: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
    from ....loader import build_loader_state_dict

    state, _metadata, packed_layer_tensors, w4_layer_tensors = build_loader_state_dict(checkpoint_path)
    return state, packed_layer_tensors, w4_layer_tensors


def load_seedvr2_state_dict(
    checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    if checkpoint_path.endswith((".safetensors", ".sft")):
        if _is_svdint4_file(checkpoint_path):
            state, _packed, _w4 = _load_svdint4_state(checkpoint_path)
            return state
        try:
            return load_safetensors_file(checkpoint_path, device=str(device))
        except RuntimeError as exc:
            error = str(exc).lower()
            if device.type == "mps" and any(term in error for term in ("watermark", "allocat", "memory")):
                return load_safetensors_file(checkpoint_path, device="cpu")
            raise
    raise ValueError(
        f"Unsupported SeedVR2 checkpoint format: {checkpoint_path}. "
        "Use official .safetensors weights or SVDInt4 single-file .safetensors/.sft weights."
    )


def prepare_model_structure(
    runner: VideoDiffusionInfer,
    model_type: str,
    checkpoint_path: str,
    config: OmegaConf,
    debug: "Debug",
) -> VideoDiffusionInfer:
    if debug is None:
        raise ValueError("Debug instance required for prepare_model_structure")

    is_dit = model_type == "dit"
    model_type_upper = "DiT" if is_dit else "VAE"
    model_config = config.dit.model if is_dit else config.vae.model

    debug.log(f"Creating {model_type_upper} model structure on meta device", category=model_type, force=True)
    debug.start_timer(f"{model_type}_structure")
    with torch.device("meta"):
        model = create_object(model_config)
    debug.end_timer(f"{model_type}_structure", f"{model_type_upper} structure created")

    if is_dit:
        runner.dit = model
        runner._dit_checkpoint = checkpoint_path
    else:
        runner.vae = model
        runner._vae_checkpoint = checkpoint_path
    return runner


def materialize_model(
    runner: VideoDiffusionInfer,
    model_type: str,
    device: torch.device,
    config: OmegaConf,
    debug: "Debug",
) -> None:
    if debug is None:
        raise ValueError("Debug instance required for materialize_model")

    is_dit = model_type == "dit"
    model_type_upper = "DiT" if is_dit else "VAE"
    model = runner.dit if is_dit else runner.vae
    checkpoint_path = runner._dit_checkpoint if is_dit else runner._vae_checkpoint
    override_dtype = getattr(runner, "_dit_dtype_override" if is_dit else "_vae_dtype_override", None)

    if model is None:
        debug.log(f"No {model_type_upper} model structure found", level="WARNING", category=model_type, force=True)
        return

    try:
        param_device = next(model.parameters()).device
    except StopIteration:
        param_device = torch.device("meta")
    if param_device.type != "meta":
        debug.log(f"{model_type_upper} already materialized on {param_device}", category=model_type)
        return

    offload_device = getattr(runner, f"_{model_type}_offload_device", None)
    target_device = offload_device if offload_device is not None else device
    offload_reason = " (offload device)" if offload_device is not None else ""

    debug.start_timer(f"{model_type}_materialize")
    model = _load_model_weights(
        model,
        checkpoint_path,
        target_device,
        model_type_upper,
        offload_reason,
        debug,
        override_dtype,
    )

    from .model_configuration import apply_model_specific_config

    model = apply_model_specific_config(model, runner, config, is_dit, debug)
    debug.end_timer(f"{model_type}_materialize", f"{model_type_upper} materialized")

    if is_dit:
        runner.dit = model
        runner._dit_checkpoint = None
        runner._dit_dtype_override = None
    else:
        runner.vae = model
        runner._vae_checkpoint = None
        runner._vae_dtype_override = None


def _load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    target_device: torch.device,
    model_type: str,
    offload_reason: str,
    debug: "Debug",
    override_dtype: Optional[torch.dtype] = None,
) -> torch.nn.Module:
    debug.log(
        f"Materializing {model_type} weights to {str(target_device).upper()}{offload_reason}: {checkpoint_path}",
        category=model_type.lower(),
        force=True,
    )

    is_svdint4 = _is_svdint4_file(checkpoint_path)
    debug.start_timer(f"{model_type.lower()}_weights_load")
    if is_svdint4:
        state, packed_layer_tensors, w4_layer_tensors = _load_svdint4_state(checkpoint_path)
        model = _replace_svdint4_linears(model, packed_layer_tensors, w4_layer_tensors)
    else:
        state = load_seedvr2_state_dict(checkpoint_path, target_device)
    debug.end_timer(f"{model_type.lower()}_weights_load", f"{model_type} weights loaded from file")

    if override_dtype is not None:
        state = _convert_state_dtype(state, override_dtype)

    _log_weight_stats(state, model_type, debug)
    _load_state_dict(model, state, debug)
    del state

    initialize_meta_buffers(model, target_device, debug)
    if _first_parameter_device(model) != target_device:
        model.to(target_device)
    return model


def _replace_svdint4_linears(
    model: torch.nn.Module,
    packed_layer_tensors: Dict[str, Dict[str, torch.Tensor]],
    w4_layer_tensors: Dict[str, Dict[str, torch.Tensor]],
) -> torch.nn.Module:
    from ....loader import SVDInt4LinearOp

    target_names = set(packed_layer_tensors) | set(w4_layer_tensors)
    if not target_names:
        return model

    linear_cls = type(
        f"SeedVR2SVDInt4Linear_{id(packed_layer_tensors):x}",
        (SVDInt4LinearOp,),
        {
            "packed_layer_names": frozenset(packed_layer_tensors),
            "packed_layer_tensors": packed_layer_tensors,
            "w4_layer_names": frozenset(w4_layer_tensors),
            "w4_layer_tensors": w4_layer_tensors,
        },
    )

    modules = dict(model.named_modules())
    missing = sorted(name for name in target_names if name not in modules)
    if missing:
        sample = ", ".join(missing[:8])
        raise ValueError(
            f"SVDInt4 checkpoint contains {len(missing)} Linear layer(s) that are not present "
            f"in the SeedVR2 model config. First missing: {sample}"
        )

    for name in sorted(target_names):
        module = modules[name]
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"SVDInt4 layer {name} targets {type(module).__name__}, expected torch.nn.Linear")
        parent, attr = _parent_module(model, name)
        replacement = linear_cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        replacement.train(module.training)
        setattr(parent, attr, replacement)
    return model


def _parent_module(model: torch.nn.Module, name: str) -> tuple[torch.nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _first_parameter_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _convert_state_dtype(state: Dict[str, torch.Tensor], target_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    for key, value in list(state.items()):
        if torch.is_tensor(value) and value.is_floating_point() and value.device.type != "meta":
            state[key] = value.to(dtype=target_dtype)
    return state


def _log_weight_stats(state: Dict[str, torch.Tensor], model_type: str, debug: Optional["Debug"] = None) -> None:
    if debug is None:
        return
    total_size_mb = sum(value.nelement() * value.element_size() for value in state.values()) / (1024 * 1024)
    debug.log(
        f"Materializing {model_type}: {len(state)} state entries, {total_size_mb:.2f}MB visible tensors",
        category=model_type.lower(),
    )


def _load_state_dict(model: torch.nn.Module, state: Dict[str, torch.Tensor], debug: Optional["Debug"]) -> None:
    try:
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    except TypeError:
        missing, unexpected = model.load_state_dict(state, strict=False)
    if debug:
        if missing:
            debug.log(f"Missing {len(missing)} state keys while loading model", level="WARNING", category="setup")
        if unexpected:
            debug.log(f"Unexpected {len(unexpected)} state keys while loading model", level="WARNING", category="setup")


def initialize_meta_buffers(
    model: torch.nn.Module,
    target_device: torch.device,
    debug: Optional["Debug"] = None,
) -> None:
    fixed = 0
    for module in model.modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device.type == "meta":
                module._buffers[name] = torch.empty(buffer.shape, device=target_device, dtype=buffer.dtype)
                fixed += 1
    if debug and fixed:
        debug.log(f"Initialized {fixed} meta buffer(s) on {target_device}", category="setup")
