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

import os
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import OmegaConf

from .infer import VideoDiffusionInfer
from .model_loader import prepare_model_structure, script_directory
from ..common.config import load_config
from ..optimization.compatibility import CompatibleDiT, validate_attention_mode
from ..utils.constants import find_model_file


def _config_dir_for_model(dit_model: str, debug: Optional["Debug"] = None) -> str:
    name = os.path.basename(dit_model).lower()
    if "7b" in name:
        return "configs_7b"
    if "3b" not in name and debug:
        debug.log(
            f"Could not infer SeedVR2 model size from {dit_model}; using 3B config",
            level="WARNING",
            category="runner",
            force=True,
        )
    return "configs_3b"


def _create_runner(
    dit_model: str,
    vae_model: str,
    debug: "Debug",
) -> VideoDiffusionInfer:
    debug.log(f"Creating SeedVR2 runner: DiT={dit_model}, VAE={vae_model}", category="runner", force=True)
    config_path = os.path.join(script_directory, _config_dir_for_model(dit_model, debug), "main.yaml")
    debug.start_timer("config_load")
    config = load_config(config_path)
    debug.end_timer("config_load", "Config loading")

    debug.start_timer("runner_video_infer")
    runner = VideoDiffusionInfer(config, debug)
    OmegaConf.set_readonly(runner.config, False)
    debug.end_timer("runner_video_infer", "Video diffusion inference runner initialization")
    return runner


def _empty_cache_context(
    *,
    dit_model: str,
    vae_model: str,
    dit_id: Optional[int],
    vae_id: Optional[int],
) -> Dict[str, Any]:
    return {
        "global_cache": None,
        "dit_cache": False,
        "vae_cache": False,
        "dit_id": dit_id,
        "vae_id": vae_id,
        "dit_model": dit_model,
        "vae_model": vae_model,
        "cached_dit": None,
        "cached_vae": None,
        "dit_newly_cached": False,
        "vae_newly_cached": False,
        "reusing_runner": False,
    }


def configure_runner(
    dit_model: str,
    vae_model: str,
    base_cache_dir: str,
    debug: "Debug",
    ctx: Dict[str, Any],
    dit_cache: bool = False,
    vae_cache: bool = False,
    dit_id: Optional[int] = None,
    vae_id: Optional[int] = None,
    encode_tiled: bool = False,
    encode_tile_size: Optional[Tuple[int, int]] = None,
    encode_tile_overlap: Optional[Tuple[int, int]] = None,
    decode_tiled: bool = False,
    decode_tile_size: Optional[Tuple[int, int]] = None,
    decode_tile_overlap: Optional[Tuple[int, int]] = None,
    tile_debug: str = "false",
    attention_mode: str = "sdpa",
) -> Tuple[VideoDiffusionInfer, Dict[str, Any]]:
    del dit_cache, vae_cache
    if debug is None:
        raise ValueError("Debug instance must be provided to configure_runner")

    runner = _create_runner(dit_model, vae_model, debug)
    runner.encode_tiled = encode_tiled
    runner.encode_tile_size = encode_tile_size
    runner.encode_tile_overlap = encode_tile_overlap
    runner.decode_tiled = decode_tiled
    runner.decode_tile_size = decode_tile_size
    runner.decode_tile_overlap = decode_tile_overlap
    runner.tile_debug = tile_debug
    runner._dit_device = ctx["dit_device"]
    runner._vae_device = ctx["vae_device"]
    runner._dit_offload_device = ctx["dit_offload_device"]
    runner._vae_offload_device = ctx["vae_offload_device"]
    runner._tensor_offload_device = ctx["tensor_offload_device"]
    runner._compute_dtype = ctx["compute_dtype"]
    runner._dit_attention_mode = attention_mode
    runner.debug = debug

    _prepare_dit(runner, dit_model, base_cache_dir, debug)
    _prepare_vae(runner, vae_model, base_cache_dir, debug)

    return runner, _empty_cache_context(dit_model=dit_model, vae_model=vae_model, dit_id=dit_id, vae_id=vae_id)


def _prepare_dit(
    runner: VideoDiffusionInfer,
    dit_model: str,
    base_cache_dir: str,
    debug: "Debug",
) -> None:
    checkpoint_path = find_model_file(dit_model, base_cache_dir)
    prepare_model_structure(runner, "dit", checkpoint_path, runner.config, debug)
    runner._dit_model_name = dit_model


def _prepare_vae(
    runner: VideoDiffusionInfer,
    vae_model: str,
    base_cache_dir: str,
    debug: "Debug",
) -> None:
    vae_config_path = os.path.join(
        script_directory,
        "src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml",
    )
    vae_config = load_config(vae_config_path)
    spatial_downsample_factor = vae_config.get("spatial_downsample_factor", 8)
    temporal_downsample_factor = vae_config.get("temporal_downsample_factor", 4)
    vae_config.spatial_downsample_factor = spatial_downsample_factor
    vae_config.temporal_downsample_factor = temporal_downsample_factor
    runner.config.vae.model = OmegaConf.merge(runner.config.vae.model, vae_config)

    compute_dtype = getattr(runner, "_compute_dtype", torch.bfloat16)
    runner.config.vae.dtype = str(compute_dtype).split(".")[-1]
    runner._vae_dtype_override = compute_dtype

    checkpoint_path = find_model_file(vae_model, base_cache_dir)
    prepare_model_structure(runner, "vae", checkpoint_path, runner.config, debug)
    runner._vae_model_name = vae_model
    debug.log(
        f"VAE downsample factors configured "
        f"(spatial: {spatial_downsample_factor}x, temporal: {temporal_downsample_factor}x)",
        category="vae",
    )


def apply_model_specific_config(
    model: torch.nn.Module,
    runner: VideoDiffusionInfer,
    config: OmegaConf,
    is_dit: bool,
    debug: Optional["Debug"] = None,
) -> torch.nn.Module:
    if is_dit:
        if not isinstance(model, CompatibleDiT):
            compute_dtype = getattr(runner, "_compute_dtype", torch.bfloat16)
            if debug:
                debug.log("Applying DiT compatibility wrapper", category="setup")
            model = CompatibleDiT(model, debug, compute_dtype=compute_dtype, skip_conversion=False)

        requested_attention_mode = getattr(runner, "_dit_attention_mode", "sdpa") or "sdpa"
        attention_mode = validate_attention_mode(requested_attention_mode, debug)
        compute_dtype = getattr(runner, "_compute_dtype", torch.bfloat16)
        actual_model = model.dit_model if hasattr(model, "dit_model") else model
        updated = 0
        for module in actual_model.modules():
            if type(module).__name__ == "FlashAttentionVarlen":
                module.attention_mode = attention_mode
                module.compute_dtype = compute_dtype
                updated += 1
        if debug and updated:
            debug.log(
                f"Applied {attention_mode} attention mode and compute_dtype={compute_dtype} to {updated} modules",
                category="success",
            )
        runner.dit = model
        return model

    if model.training:
        model.requires_grad_(False).eval()
    if hasattr(model, "set_causal_slicing") and hasattr(config.vae, "slicing"):
        model.set_causal_slicing(**config.vae.slicing)
    if hasattr(model, "set_memory_limit") and hasattr(config.vae, "memory_limit"):
        model.set_memory_limit(**config.vae.memory_limit)

    model.debug = debug
    model.tensor_offload_device = runner._tensor_offload_device
    for module in model.modules():
        if hasattr(module, "debug"):
            module.debug = debug
        if hasattr(module, "tensor_offload_device"):
            module.tensor_offload_device = runner._tensor_offload_device
    runner.vae = model
    return model
