from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any

import torch

import comfy.model_management as model_management
import comfy.samplers
import folder_paths

from .seedvr2 import SeedVR2Pipeline


LOG = logging.getLogger("comfyui-svdint4")

SEEDVR2_FOLDER_NAME = "SEEDVR2"
SEEDVR2_MODEL_TYPE = "seedvr2"
MODEL_EXTENSIONS = {".safetensors", ".sft"}
KNOWN_VAE_MODEL_NAMES = {"ema_vae_fp16.safetensors"}
ATTENTION_BACKENDS = ["sdpa", "sage_attn", "flash_attn"]
SEEDVR2_VAE_TILE_SIZES = (1024, 768, 512, 384, 256)
SEEDVR2_AUTO_VAE_TILE_SIZES = (768, 512, 384, 256)


@dataclasses.dataclass(frozen=True)
class SeedVR2Plan:
    batch_size: int
    encode_tiled: bool
    encode_tile_size: int
    encode_tile_overlap: int
    decode_tiled: bool
    decode_tile_size: int
    decode_tile_overlap: int

    def describe(self) -> str:
        encode = f"tile={self.encode_tile_size}" if self.encode_tiled else "no-tile"
        decode = f"tile={self.decode_tile_size}" if self.decode_tiled else "no-tile"
        return f"batch={self.batch_size}, encode={encode}, decode={decode}"


def _register_seedvr2_folder() -> None:
    model_dir = os.path.join(folder_paths.models_dir, SEEDVR2_FOLDER_NAME)
    folder_paths.add_model_folder_path(SEEDVR2_MODEL_TYPE, model_dir)
    try:
        os.makedirs(model_dir, exist_ok=True)
    except OSError:
        LOG.warning("Could not create SeedVR2 model directory: %s", model_dir)


def _seedvr2_model_dir() -> str:
    _register_seedvr2_folder()
    try:
        paths = folder_paths.get_folder_paths(SEEDVR2_MODEL_TYPE)
        if paths:
            return paths[0]
    except Exception:
        pass
    return os.path.join(folder_paths.models_dir, SEEDVR2_FOLDER_NAME)


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


def _is_vae_model_name(name: str) -> bool:
    lower = Path(name).name.lower()
    return lower in KNOWN_VAE_MODEL_NAMES or "vae" in Path(lower).stem


def _model_choices(*, vae: bool) -> list[str]:
    return [name for name in _seedvr2_model_files() if _is_vae_model_name(name) == vae]


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


def _tile_overlap(tile_size: int) -> int:
    return max(64, int(tile_size) // 4)


def _vae_work_budget_mb(free_gb: float) -> float:
    if free_gb <= 0:
        return 4 * 1024.0
    return max(1024.0, free_gb * 1024.0 * 0.70 - 1024.0)


def _estimate_vae_peak_mb(
    *,
    kind: str,
    frames: int,
    height: int,
    width: int,
    tile_size: int | None,
) -> float:
    frames = max(1, int(frames))
    megapixels = max(1.0 / 1024.0, float(height * width) / 1_000_000.0)
    if tile_size is None:
        if kind == "encode":
            return 1800.0 + 1600.0 * megapixels * frames
        return 3000.0 + 2500.0 * megapixels * frames

    if kind == "encode":
        table = {
            768: (3000.0, 950.0),
            512: (1600.0, 530.0),
            384: (1150.0, 310.0),
            256: (800.0, 150.0),
        }
    else:
        table = {
            768: (4300.0, 900.0),
            512: (2500.0, 750.0),
            384: (1700.0, 560.0),
            256: (1100.0, 280.0),
        }
    base, per_extra_frame = table.get(int(tile_size), table[256])
    spatial_overhead = 250.0 * megapixels
    return base + per_extra_frame * max(0, frames - 1) + spatial_overhead


def _choose_vae_tiling(
    *,
    kind: str,
    frames: int,
    height: int,
    width: int,
    free_gb: float,
) -> tuple[bool, int, int]:
    budget_mb = _vae_work_budget_mb(free_gb)
    regular_mb = _estimate_vae_peak_mb(
        kind=kind,
        frames=frames,
        height=height,
        width=width,
        tile_size=None,
    )
    if regular_mb <= budget_mb:
        tile = SEEDVR2_VAE_TILE_SIZES[0]
        return False, tile, _tile_overlap(tile)

    for tile in SEEDVR2_AUTO_VAE_TILE_SIZES:
        tiled_mb = _estimate_vae_peak_mb(
            kind=kind,
            frames=frames,
            height=height,
            width=width,
            tile_size=tile,
        )
        if tiled_mb <= budget_mb:
            return True, tile, _tile_overlap(tile)

    tile = SEEDVR2_AUTO_VAE_TILE_SIZES[-1]
    return True, tile, _tile_overlap(tile)


def _auto_batch_size(frame_count: int, height: int, width: int, free_gb: float) -> int:
    if frame_count <= 1:
        return 1
    megapixels = (height * width) / 1_000_000.0
    if free_gb >= 18.0 and megapixels <= 1.5:
        target = 13
    elif free_gb >= 12.0 and megapixels <= 2.0:
        target = 9
    elif free_gb >= 10.0 and megapixels <= 1.5:
        target = 5
    else:
        target = 1
    return min(_valid_4n1(target), _valid_4n1(frame_count))


def _build_plan(
    image: torch.Tensor,
    *,
    resolution: int,
    max_resolution: int,
    batch_size: int = 0,
) -> SeedVR2Plan:
    frames = int(image.shape[0])
    height = int(image.shape[1])
    width = int(image.shape[2])
    target_h, target_w = _target_dimensions(height, width, resolution, max_resolution)
    main_device = str(model_management.get_torch_device())
    free_gb = _estimate_free_vram_gb(main_device)

    chosen_batch = _valid_4n1(batch_size) if batch_size > 0 else _auto_batch_size(frames, target_h, target_w, free_gb)
    chosen_batch = max(1, min(chosen_batch, _valid_4n1(frames)))
    encode_tiled, encode_tile, encode_overlap = _choose_vae_tiling(
        kind="encode",
        frames=chosen_batch,
        height=target_h,
        width=target_w,
        free_gb=free_gb,
    )
    decode_tiled, decode_tile, decode_overlap = _choose_vae_tiling(
        kind="decode",
        frames=chosen_batch,
        height=target_h,
        width=target_w,
        free_gb=free_gb,
    )

    return SeedVR2Plan(
        batch_size=chosen_batch,
        encode_tiled=encode_tiled,
        encode_tile_size=encode_tile,
        encode_tile_overlap=encode_overlap,
        decode_tiled=decode_tiled,
        decode_tile_size=decode_tile,
        decode_tile_overlap=decode_overlap,
    )


def _fallback_plans(plan: SeedVR2Plan) -> list[SeedVR2Plan]:
    plans = [plan]
    batch = plan.batch_size
    while batch > 1:
        batch = _valid_4n1(max(1, batch - 4))
        plans.append(dataclasses.replace(plan, batch_size=batch))
        if batch == 1:
            break

    for tile in SEEDVR2_VAE_TILE_SIZES:
        if plans[-1].decode_tiled and tile >= plans[-1].decode_tile_size:
            continue
        overlap = _tile_overlap(tile)
        encode_tile = min(tile, plan.encode_tile_size) if plan.encode_tiled else plan.encode_tile_size
        plans.append(
            dataclasses.replace(
                plans[-1],
                encode_tiled=plan.encode_tiled,
                decode_tiled=True,
                encode_tile_size=encode_tile,
                decode_tile_size=tile,
                encode_tile_overlap=_tile_overlap(encode_tile),
                decode_tile_overlap=overlap,
            )
        )

    unique: list[SeedVR2Plan] = []
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


class SeedVR2Upscaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dit_model": (_model_choices(vae=False),),
                "vae_model": (_model_choices(vae=True),),
                "resolution": ("INT", {"default": 1080, "min": 16, "max": 16384, "step": 2}),
                "max_resolution": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 2}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1, "step": 1}),
                "steps": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "batch_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "retry_on_oom": ("BOOLEAN", {"default": True}),
                "attention_backend": (ATTENTION_BACKENDS, {"default": "sdpa"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "SVDInt4/SeedVR2"
    TITLE = "SeedVR2 Upscale"

    def upscale(
        self,
        image: torch.Tensor,
        dit_model: str,
        vae_model: str,
        resolution: int,
        max_resolution: int,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        batch_size: int,
        retry_on_oom: bool,
        attention_backend: str,
    ):
        if image.ndim != 4 or int(image.shape[-1]) not in (3, 4):
            raise ValueError(f"SeedVR2 Upscale expects IMAGE shaped [frames,h,w,3/4], got {tuple(image.shape)}")

        plan = _build_plan(
            image,
            resolution=resolution,
            max_resolution=max_resolution,
            batch_size=batch_size,
        )
        LOG.info("SeedVR2 planner selected: %s", plan.describe())

        last_exc: BaseException | None = None
        plans = _fallback_plans(plan) if retry_on_oom else [plan]
        for attempt, candidate in enumerate(plans, start=1):
            try:
                LOG.info("SeedVR2 attempt %d: %s", attempt, candidate.describe())
                return (
                    self._run_once(
                        image=image,
                        dit_model=dit_model,
                        vae_model=vae_model,
                        resolution=resolution,
                        max_resolution=max_resolution,
                        seed=seed,
                        steps=steps,
                        cfg=cfg,
                        sampler_name=sampler_name,
                        scheduler=scheduler,
                        denoise=denoise,
                        attention_backend=attention_backend,
                        plan=candidate,
                    ),
                )
            except BaseException as exc:
                if not _is_oom(exc):
                    raise
                last_exc = exc
                LOG.warning("SeedVR2 attempt %d hit OOM; retrying with a smaller plan.", attempt)
                _clear_after_oom()

        raise RuntimeError(f"SeedVR2 Upscale failed after all memory fallback plans. Last error: {last_exc}") from last_exc

    def _run_once(
        self,
        *,
        image: torch.Tensor,
        dit_model: str,
        vae_model: str,
        resolution: int,
        max_resolution: int,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        attention_backend: str,
        plan: SeedVR2Plan,
    ) -> torch.Tensor:
        self._validate_model_files_exist(dit_model, vae_model)
        pipeline = SeedVR2Pipeline(
            model_dir=_seedvr2_model_dir(),
            dit_model=dit_model,
            vae_model=vae_model,
            attention_backend=attention_backend,
        )
        try:
            return pipeline.upscale(
                image,
                resolution=resolution,
                max_resolution=max_resolution,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                batch_size=plan.batch_size,
                encode_tiled=plan.encode_tiled,
                encode_tile_size=plan.encode_tile_size,
                encode_tile_overlap=plan.encode_tile_overlap,
                decode_tiled=plan.decode_tiled,
                decode_tile_size=plan.decode_tile_size,
                decode_tile_overlap=plan.decode_tile_overlap,
            )
        finally:
            pipeline.close()

    @staticmethod
    def _validate_model_files_exist(dit_model: str, vae_model: str) -> None:
        if _is_vae_model_name(dit_model):
            raise ValueError(f"SeedVR2 DiT model selection points to a VAE file: {dit_model}")
        if not _is_vae_model_name(vae_model):
            raise ValueError(f"SeedVR2 VAE model selection points to a non-VAE file: {vae_model}")

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
