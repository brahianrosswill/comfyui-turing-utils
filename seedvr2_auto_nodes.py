from __future__ import annotations

import dataclasses
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch

import comfy.model_management as model_management
import folder_paths


LOG = logging.getLogger("comfyui-svdint4")

SEEDVR2_FOLDER_NAME = "SEEDVR2"
SEEDVR2_MODEL_TYPE = "seedvr2"
MODEL_EXTENSIONS = {".safetensors", ".sft"}
DEFAULT_DIT_MODELS = [
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_ema_7b_sharp_fp16.safetensors",
]
DEFAULT_VAE_MODELS = ["ema_vae_fp16.safetensors"]
AUTO_BATCH = "auto"
_AUTO_NODE_ID = "svdint4_seedvr2_auto"


@dataclasses.dataclass(frozen=True)
class SeedVR2AutoPlan:
    batch_size: int
    temporal_overlap: int
    encode_tiled: bool
    encode_tile_size: int
    encode_tile_overlap: int
    decode_tiled: bool
    decode_tile_size: int
    decode_tile_overlap: int
    tensor_offload_device: str
    dit_offload_device: str
    vae_offload_device: str
    uniform_batch_size: bool

    def describe(self) -> str:
        encode = f"tile={self.encode_tile_size}" if self.encode_tiled else "no-tile"
        decode = f"tile={self.decode_tile_size}" if self.decode_tiled else "no-tile"
        return (
            f"batch={self.batch_size}, overlap={self.temporal_overlap}, "
            f"encode={encode}, decode={decode}, tensor_offload={self.tensor_offload_device}, "
            f"dit_offload={self.dit_offload_device}, vae_offload={self.vae_offload_device}"
        )


def _register_seedvr2_folder() -> None:
    model_dir = os.path.join(folder_paths.models_dir, SEEDVR2_FOLDER_NAME)
    folder_paths.add_model_folder_path(SEEDVR2_MODEL_TYPE, model_dir)
    try:
        os.makedirs(model_dir, exist_ok=True)
    except OSError:
        LOG.warning("Could not create SeedVR2 model directory: %s", model_dir)


def _seedvr2_model_files() -> list[str]:
    _register_seedvr2_folder()
    found: list[str] = []
    try:
        names = folder_paths.get_filename_list(SEEDVR2_MODEL_TYPE)
    except Exception:
        names = []
    for name in names:
        if Path(name).suffix.lower() in MODEL_EXTENSIONS:
            found.append(name)
    return sorted(set(found))


def _model_choices(defaults: list[str]) -> list[str]:
    return sorted(set(defaults + _seedvr2_model_files()))


def _device_choices(*, include_auto: bool = True, include_none: bool = False, include_cpu: bool = False) -> list[str]:
    choices: list[str] = []
    if include_auto:
        choices.append("auto")
    if include_none:
        choices.append("none")
    if include_cpu:
        choices.append("cpu")
    try:
        for device in model_management.get_all_torch_devices(exclude_current=False):
            dev = str(device)
            if dev not in choices:
                choices.append(dev)
    except Exception:
        dev = str(model_management.get_torch_device())
        if dev not in choices:
            choices.append(dev)
    return choices or ["cpu"]


def _resolve_auto_device(device: str) -> str:
    if device == "auto":
        return str(model_management.get_torch_device())
    return device


def _resolve_auto_offload(device: str, *, default: str = "none") -> str:
    if device == "auto":
        return default
    return device


def _round_even(value: float) -> int:
    return max(2, int(value) // 2 * 2)


def _target_dimensions(height: int, width: int, resolution: int, max_resolution: int) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image shape: {width}x{height}")
    short_edge = min(height, width)
    scale = max(float(resolution) / float(short_edge), 1.0)
    out_h = height * scale
    out_w = width * scale
    if max_resolution > 0 and max(out_h, out_w) > max_resolution:
        limit_scale = float(max_resolution) / max(out_h, out_w)
        out_h *= limit_scale
        out_w *= limit_scale
    return _round_even(out_h), _round_even(out_w)


def _valid_4n1(value: int) -> int:
    value = max(1, int(value))
    if value == 1:
        return 1
    return max(1, ((value - 1) // 4) * 4 + 1)


def _estimate_free_vram_gb(device: str) -> float:
    try:
        dev = torch.device(device)
    except Exception:
        return 0.0
    if dev.type == "cpu":
        return 0.0
    try:
        free = model_management.get_free_memory(dev)
        return float(free) / (1024**3)
    except Exception:
        return 0.0


def _choose_tile_size(height: int, width: int, memory_mode: str, free_gb: float) -> tuple[bool, int, int]:
    longest = max(height, width)
    if memory_mode == "fastest":
        return False, 0, 0
    if memory_mode == "balanced" and free_gb >= 18.0 and longest <= 1536:
        return False, 0, 0
    if memory_mode == "balanced" and free_gb >= 12.0 and longest <= 1280:
        return False, 0, 0

    candidates = [1536, 1280, 1024, 768, 640, 512]
    if memory_mode == "low_vram":
        candidates = [1024, 768, 640, 512]
    if free_gb < 8.0:
        candidates = [768, 640, 512]
    if free_gb < 6.0:
        candidates = [512]

    tile = next((item for item in candidates if item < longest), candidates[-1])
    overlap = 128 if tile >= 1024 else 64
    return True, tile, overlap


def _auto_batch_size(frame_count: int, height: int, width: int, memory_mode: str, free_gb: float) -> int:
    if frame_count <= 1:
        return 1
    megapixels = (height * width) / 1_000_000.0
    if memory_mode == "low_vram":
        target = 5 if free_gb >= 10.0 and megapixels <= 1.5 else 1
    elif memory_mode == "fastest":
        if free_gb >= 22.0 and megapixels <= 1.5:
            target = 21
        elif free_gb >= 16.0 and megapixels <= 2.0:
            target = 13
        else:
            target = 9
    else:
        if free_gb >= 18.0 and megapixels <= 1.5:
            target = 13
        elif free_gb >= 12.0 and megapixels <= 2.0:
            target = 9
        else:
            target = 5
    return min(_valid_4n1(target), _valid_4n1(frame_count))


def _build_plan(
    image: torch.Tensor,
    *,
    resolution: int,
    max_resolution: int,
    batch_size: int,
    temporal_overlap: int,
    memory_mode: str,
    dit_device: str,
    vae_device: str,
    tensor_offload_device: str,
    dit_offload_device: str,
    vae_offload_device: str,
) -> SeedVR2AutoPlan:
    frames = int(image.shape[0])
    height = int(image.shape[1])
    width = int(image.shape[2])
    target_h, target_w = _target_dimensions(height, width, resolution, max_resolution)
    main_device = _resolve_auto_device(dit_device)
    free_gb = _estimate_free_vram_gb(main_device)

    chosen_batch = _valid_4n1(batch_size) if batch_size > 0 else _auto_batch_size(frames, target_h, target_w, memory_mode, free_gb)
    chosen_batch = max(1, min(chosen_batch, _valid_4n1(frames)))
    overlap = max(0, min(int(temporal_overlap), chosen_batch - 1))

    encode_tiled, encode_tile, encode_overlap = _choose_tile_size(target_h, target_w, memory_mode, free_gb)
    decode_tiled, decode_tile, decode_overlap = _choose_tile_size(target_h, target_w, memory_mode, free_gb)
    if memory_mode == "fastest":
        encode_tiled = False
        decode_tiled = False

    tensor_offload = _resolve_auto_offload(
        tensor_offload_device,
        default="cpu" if (frames * target_h * target_w >= 64_000_000 or memory_mode == "low_vram") else "none",
    )
    dit_offload = _resolve_auto_offload(dit_offload_device, default="none")
    vae_offload = _resolve_auto_offload(vae_offload_device, default="none")
    if memory_mode == "low_vram":
        if dit_offload == "none":
            dit_offload = "cpu"
        if vae_offload == "none":
            vae_offload = "cpu"
        if tensor_offload == "none":
            tensor_offload = "cpu"

    return SeedVR2AutoPlan(
        batch_size=chosen_batch,
        temporal_overlap=overlap,
        encode_tiled=encode_tiled,
        encode_tile_size=encode_tile or 1024,
        encode_tile_overlap=encode_overlap or 128,
        decode_tiled=decode_tiled,
        decode_tile_size=decode_tile or 1024,
        decode_tile_overlap=decode_overlap or 128,
        tensor_offload_device=tensor_offload,
        dit_offload_device=dit_offload,
        vae_offload_device=vae_offload,
        uniform_batch_size=frames > chosen_batch and memory_mode != "fastest",
    )


def _fallback_plans(plan: SeedVR2AutoPlan) -> list[SeedVR2AutoPlan]:
    plans = [plan]
    batch = plan.batch_size
    while batch > 1:
        batch = _valid_4n1(max(1, batch - 4))
        plans.append(dataclasses.replace(plan, batch_size=batch, temporal_overlap=min(plan.temporal_overlap, batch - 1)))
        if batch == 1:
            break

    for tile in (1536, 1280, 1024, 768, 640, 512):
        if plans[-1].decode_tiled and tile >= plans[-1].decode_tile_size:
            continue
        overlap = 128 if tile >= 1024 else 64
        plans.append(
            dataclasses.replace(
                plans[-1],
                encode_tiled=True,
                decode_tiled=True,
                encode_tile_size=tile,
                decode_tile_size=tile,
                encode_tile_overlap=overlap,
                decode_tile_overlap=overlap,
                tensor_offload_device="cpu",
                dit_offload_device="cpu" if plans[-1].dit_offload_device == "none" else plans[-1].dit_offload_device,
                vae_offload_device="cpu" if plans[-1].vae_offload_device == "none" else plans[-1].vae_offload_device,
            )
        )

    unique: list[SeedVR2AutoPlan] = []
    seen: set[tuple[Any, ...]] = set()
    for item in plans:
        key = dataclasses.astuple(item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message or "allocation on device" in message


def _clear_after_oom() -> None:
    try:
        model_management.unload_all_models()
    except Exception:
        pass
    try:
        model_management.soft_empty_cache(force=True)
    except Exception:
        model_management.soft_empty_cache()


def _import_seedvr2_runtime():
    try:
        from .seedvr2_runtime.src.core.generation_phases import (
            decode_all_batches,
            encode_all_batches,
            postprocess_all_batches,
            upscale_all_batches,
        )
        from .seedvr2_runtime.src.core.generation_utils import (
            compute_generation_info,
            load_text_embeddings,
            log_generation_start,
            prepare_runner,
            script_directory,
            setup_generation_context,
        )
        from .seedvr2_runtime.src.optimization.memory_manager import (
            cleanup_text_embeddings,
            complete_cleanup,
        )
        from .seedvr2_runtime.src.utils.constants import get_base_cache_dir
        from .seedvr2_runtime.src.utils.debug import Debug
    except ImportError as exc:
        raise ImportError(
            "SeedVR2 auto upscaler dependencies are missing. Install the optional SeedVR2 runtime "
            "dependencies in the ComfyUI Python environment: omegaconf diffusers "
            "opencv-python psutil einops safetensors tqdm. "
            f"Original import error: {exc}"
        ) from exc
    return {
        "Debug": Debug,
        "get_base_cache_dir": get_base_cache_dir,
        "setup_generation_context": setup_generation_context,
        "prepare_runner": prepare_runner,
        "load_text_embeddings": load_text_embeddings,
        "script_directory": script_directory,
        "compute_generation_info": compute_generation_info,
        "log_generation_start": log_generation_start,
        "encode_all_batches": encode_all_batches,
        "upscale_all_batches": upscale_all_batches,
        "decode_all_batches": decode_all_batches,
        "postprocess_all_batches": postprocess_all_batches,
        "cleanup_text_embeddings": cleanup_text_embeddings,
        "complete_cleanup": complete_cleanup,
    }


class SeedVR2AutoUpscaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dit_model": (_model_choices(DEFAULT_DIT_MODELS),),
                "vae_model": (_model_choices(DEFAULT_VAE_MODELS),),
                "resolution": ("INT", {"default": 1080, "min": 16, "max": 16384, "step": 2}),
                "max_resolution": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 2}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1, "step": 1}),
                "memory_mode": (["balanced", "fastest", "low_vram"], {"default": "balanced"}),
                "batch_size": ("INT", {"default": 0, "min": 0, "max": 257, "step": 1}),
                "temporal_overlap": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "color_correction": (["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], {"default": "lab"}),
            },
            "optional": {
                "input_noise_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "latent_noise_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "dit_device": (_device_choices(), {"default": "auto"}),
                "vae_device": (_device_choices(), {"default": "auto"}),
                "tensor_offload_device": (_device_choices(include_none=True, include_cpu=True), {"default": "auto"}),
                "dit_offload_device": (_device_choices(include_none=True, include_cpu=True), {"default": "auto"}),
                "vae_offload_device": (_device_choices(include_none=True, include_cpu=True), {"default": "auto"}),
                "attention_mode": (["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"], {"default": "sdpa"}),
                "enable_debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "SVDInt4/SeedVR2"
    TITLE = "SeedVR2 Auto Upscale"

    def upscale(
        self,
        image: torch.Tensor,
        dit_model: str,
        vae_model: str,
        resolution: int,
        max_resolution: int,
        seed: int,
        memory_mode: str,
        batch_size: int,
        temporal_overlap: int,
        color_correction: str,
        input_noise_scale: float = 0.0,
        latent_noise_scale: float = 0.0,
        dit_device: str = "auto",
        vae_device: str = "auto",
        tensor_offload_device: str = "auto",
        dit_offload_device: str = "auto",
        vae_offload_device: str = "auto",
        attention_mode: str = "sdpa",
        enable_debug: bool = False,
    ):
        if image.ndim != 4 or int(image.shape[-1]) not in (3, 4):
            raise ValueError(f"SeedVR2 Auto Upscale expects IMAGE shaped [frames,h,w,3/4], got {tuple(image.shape)}")

        dit_device = _resolve_auto_device(dit_device)
        vae_device = _resolve_auto_device(vae_device)
        plan = _build_plan(
            image,
            resolution=resolution,
            max_resolution=max_resolution,
            batch_size=batch_size,
            temporal_overlap=temporal_overlap,
            memory_mode=memory_mode,
            dit_device=dit_device,
            vae_device=vae_device,
            tensor_offload_device=tensor_offload_device,
            dit_offload_device=dit_offload_device,
            vae_offload_device=vae_offload_device,
        )
        LOG.info("SeedVR2 auto planner selected: %s", plan.describe())

        last_exc: BaseException | None = None
        for attempt, candidate in enumerate(_fallback_plans(plan), start=1):
            try:
                LOG.info("SeedVR2 auto attempt %d: %s", attempt, candidate.describe())
                return (
                    self._run_once(
                        image=image,
                        dit_model=dit_model,
                        vae_model=vae_model,
                        resolution=resolution,
                        max_resolution=max_resolution,
                        seed=seed,
                        color_correction=color_correction,
                        input_noise_scale=input_noise_scale,
                        latent_noise_scale=latent_noise_scale,
                        dit_device=dit_device,
                        vae_device=vae_device,
                        attention_mode=attention_mode,
                        enable_debug=enable_debug,
                        plan=candidate,
                    ),
                )
            except BaseException as exc:
                if not _is_oom(exc):
                    raise
                last_exc = exc
                LOG.warning("SeedVR2 auto attempt %d hit OOM; retrying with a smaller plan.", attempt)
                _clear_after_oom()

        raise RuntimeError(f"SeedVR2 Auto Upscale failed after all memory fallback plans. Last error: {last_exc}") from last_exc

    def _run_once(
        self,
        *,
        image: torch.Tensor,
        dit_model: str,
        vae_model: str,
        resolution: int,
        max_resolution: int,
        seed: int,
        color_correction: str,
        input_noise_scale: float,
        latent_noise_scale: float,
        dit_device: str,
        vae_device: str,
        attention_mode: str,
        enable_debug: bool,
        plan: SeedVR2AutoPlan,
    ) -> torch.Tensor:
        self._validate_model_files_exist(dit_model, vae_model)

        runtime = _import_seedvr2_runtime()
        Debug = runtime["Debug"]
        debug = Debug(enabled=enable_debug)
        runner = None
        ctx = None

        def progress_callback(current_step: int, total_steps: int, current_frames: int, phase_name: str) -> None:
            del current_frames
            if total_steps <= 0:
                return
            LOG.info("SeedVR2 %s: %d/%d", phase_name, current_step, total_steps)

        def cleanup() -> None:
            nonlocal runner, ctx
            if runner is not None:
                runtime["complete_cleanup"](runner=runner, debug=debug, dit_cache=False, vae_cache=False)
                runner = None
            if ctx is not None:
                runtime["cleanup_text_embeddings"](ctx, debug)
                ctx = None

        try:
            ctx = runtime["setup_generation_context"](
                dit_device=torch.device(dit_device),
                vae_device=torch.device(vae_device),
                dit_offload_device=torch.device(plan.dit_offload_device) if plan.dit_offload_device != "none" else None,
                vae_offload_device=torch.device(plan.vae_offload_device) if plan.vae_offload_device != "none" else None,
                tensor_offload_device=torch.device(plan.tensor_offload_device) if plan.tensor_offload_device != "none" else None,
                debug=debug,
            )
            runner, cache_context = runtime["prepare_runner"](
                dit_model=dit_model,
                vae_model=vae_model,
                model_dir=runtime["get_base_cache_dir"](),
                debug=debug,
                ctx=ctx,
                dit_cache=False,
                vae_cache=False,
                dit_id=_AUTO_NODE_ID,
                vae_id=_AUTO_NODE_ID,
                encode_tiled=plan.encode_tiled,
                encode_tile_size=(plan.encode_tile_size, plan.encode_tile_size),
                encode_tile_overlap=(plan.encode_tile_overlap, plan.encode_tile_overlap),
                decode_tiled=plan.decode_tiled,
                decode_tile_size=(plan.decode_tile_size, plan.decode_tile_size),
                decode_tile_overlap=(plan.decode_tile_overlap, plan.decode_tile_overlap),
                tile_debug="false",
                attention_mode=attention_mode,
            )
            ctx["cache_context"] = cache_context
            ctx["text_embeds"] = runtime["load_text_embeddings"](
                runtime["script_directory"],
                ctx["dit_device"],
                ctx["compute_dtype"],
                debug,
            )

            image, gen_info = runtime["compute_generation_info"](
                ctx=ctx,
                images=image,
                resolution=resolution,
                max_resolution=max_resolution,
                batch_size=plan.batch_size,
                uniform_batch_size=plan.uniform_batch_size,
                seed=seed,
                prepend_frames=0,
                temporal_overlap=plan.temporal_overlap,
                debug=debug,
            )
            runtime["log_generation_start"](gen_info, debug)
            ctx = runtime["encode_all_batches"](
                runner,
                ctx=ctx,
                images=image,
                debug=debug,
                batch_size=plan.batch_size,
                uniform_batch_size=plan.uniform_batch_size,
                seed=seed,
                progress_callback=progress_callback,
                temporal_overlap=plan.temporal_overlap,
                resolution=resolution,
                max_resolution=max_resolution,
                input_noise_scale=input_noise_scale,
                color_correction=color_correction,
            )
            ctx = runtime["upscale_all_batches"](
                runner,
                ctx=ctx,
                debug=debug,
                progress_callback=progress_callback,
                seed=seed,
                latent_noise_scale=latent_noise_scale,
                cache_model=False,
            )
            ctx = runtime["decode_all_batches"](
                runner,
                ctx=ctx,
                debug=debug,
                progress_callback=progress_callback,
                cache_model=False,
            )
            ctx = runtime["postprocess_all_batches"](
                ctx=ctx,
                debug=debug,
                progress_callback=progress_callback,
                color_correction=color_correction,
                prepend_frames=0,
                temporal_overlap=plan.temporal_overlap,
                batch_size=plan.batch_size,
            )
            output = ctx["final_video"]
            if output.device.type != "cpu":
                output = output.cpu()
            return output.to(torch.float32).clamp_(0.0, 1.0)
        finally:
            cleanup()

    @staticmethod
    def _validate_model_files_exist(dit_model: str, vae_model: str) -> None:
        _register_seedvr2_folder()
        missing = []
        try:
            if folder_paths.get_full_path(SEEDVR2_MODEL_TYPE, dit_model) is None:
                missing.append(dit_model)
            if folder_paths.get_full_path(SEEDVR2_MODEL_TYPE, vae_model) is None:
                missing.append(vae_model)
        except Exception:
            base = Path(folder_paths.models_dir) / SEEDVR2_FOLDER_NAME
            if not (base / dit_model).is_file():
                missing.append(dit_model)
            if not (base / vae_model).is_file():
                missing.append(vae_model)
        if missing:
            base = Path(folder_paths.models_dir) / SEEDVR2_FOLDER_NAME
            raise FileNotFoundError(
                "Missing SeedVR2 model file(s): "
                + ", ".join(missing)
                + f". Put them under {base}."
            )
