from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import torch
from safetensors import safe_open
from safetensors.torch import save_file


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parent.parent
KERNEL_ROOT = PLUGIN_ROOT / "kernel"
for item in (str(KERNEL_ROOT), str(PLUGIN_ROOT), str(COMFY_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

from seedvr2_runtime.src.core.generation_phases import encode_all_batches, upscale_all_batches
from seedvr2_runtime.src.core.generation_utils import (
    compute_generation_info,
    load_text_embeddings,
    prepare_runner,
    script_directory,
    setup_generation_context,
)
from seedvr2_runtime.src.optimization.memory_manager import cleanup_text_embeddings, complete_cleanup
from seedvr2_runtime.src.utils.debug import Debug
from svdint4.packing import build_svdint4_metadata, pack_bias, pack_linear_weight, pack_svd_down, pack_svd_up


LOG = logging.getLogger("seedvr2-svdint4-quant")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
FLOAT8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)


def _valid_4n1(value: int) -> int:
    value = max(1, int(value))
    if value == 1:
        return 1
    return max(1, ((value - 1) // 4) * 4 + 1)


def _discover(paths: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    images: list[Path] = []
    videos: list[Path] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    suffix = child.suffix.lower()
                    if suffix in IMAGE_EXTS:
                        images.append(child)
                    elif suffix in VIDEO_EXTS:
                        videos.append(child)
        elif path.is_file():
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTS:
                images.append(path)
            elif suffix in VIDEO_EXTS:
                videos.append(path)
    return images, videos


def _resize_max_side(frame: torch.Tensor, max_side: int) -> torch.Tensor:
    if max_side <= 0:
        return frame
    h, w = int(frame.shape[0]), int(frame.shape[1])
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = float(max_side) / float(longest)
    new_w = max(2, int(round(w * scale)) // 2 * 2)
    new_h = max(2, int(round(h * scale)) // 2 * 2)
    arr = (frame.numpy() * 255.0).round().clip(0, 255).astype("uint8")
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(resized).to(torch.float32).div_(255.0)


def _read_image(path: Path, max_side: int) -> torch.Tensor:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError(f"Could not read image: {path}")
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(arr).to(torch.float32).div_(255.0)
    frame = _resize_max_side(frame, max_side)
    return frame.unsqueeze(0).contiguous()


def _read_video(path: Path, target_frames: int, max_side: int) -> torch.Tensor:
    target_frames = _valid_4n1(target_frames)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            indices = torch.linspace(0, max(0, frame_count - 1), steps=target_frames).round().to(torch.int64).tolist()
        else:
            indices = list(range(target_frames))
        frames: list[torch.Tensor] = []
        for index in indices:
            if frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, arr = cap.read()
            if not ok:
                break
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(arr).to(torch.float32).div_(255.0)
            frames.append(_resize_max_side(frame, max_side))
        if not frames:
            raise ValueError(f"Video has no readable frames: {path}")
        while len(frames) < target_frames:
            frames.append(frames[-1].clone())
        return torch.stack(frames[:target_frames], dim=0).contiguous()
    finally:
        cap.release()


def _load_calibration_cases(
    paths: list[Path],
    *,
    image_limit: int,
    video_limit: int,
    video_frames: int,
    input_max_side: int,
) -> list[tuple[str, torch.Tensor]]:
    images, videos = _discover(paths)
    cases: list[tuple[str, torch.Tensor]] = []
    for path in images[: max(0, image_limit)]:
        cases.append((str(path), _read_image(path, input_max_side)))
    for path in videos[: max(0, video_limit)]:
        cases.append((str(path), _read_video(path, video_frames, input_max_side)))
    if not cases:
        raise ValueError("No calibration images or videos were found")
    return cases


def _dense_checkpoint_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype in FLOAT8_DTYPES:
        return tensor.to(dtype=torch.float16).contiguous()
    return tensor.contiguous()


def _collect_linear_specs(dit_model: torch.nn.Module) -> dict[str, tuple[int, int, bool]]:
    specs = {}
    for name, module in dit_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            specs[name] = (int(module.in_features), int(module.out_features), module.bias is not None)
    return specs


def _register_absmax_hooks(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], list[torch.utils.hooks.RemovableHandle]]:
    stats: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str, in_features: int):
        def hook(_module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            if int(x.shape[-1]) != in_features:
                return
            flat = x.detach().reshape(-1, in_features).abs().amax(dim=0).float().cpu()
            prev = stats.get(name)
            stats[name] = flat if prev is None else torch.maximum(prev, flat)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_pre_hook(make_hook(name, int(module.in_features))))
    return stats, handles


def collect_activation_stats(
    *,
    dit_model: str,
    vae_model: str,
    model_dir: Path,
    cases: list[tuple[str, torch.Tensor]],
    resolution: int,
    max_resolution: int,
    batch_size: int,
    temporal_overlap: int,
    device: str,
    attention_mode: str,
    encode_tiled: bool,
    encode_tile_size: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[int, int, bool]], dict[str, object]]:
    debug = Debug(enabled=False)
    ctx = setup_generation_context(
        dit_device=torch.device(device),
        vae_device=torch.device(device),
        dit_offload_device=None,
        vae_offload_device=None,
        tensor_offload_device=None,
        debug=debug,
    )
    runner = None
    handles: list[torch.utils.hooks.RemovableHandle] = []
    stats: dict[str, torch.Tensor] = {}
    specs: dict[str, tuple[int, int, bool]] = {}
    try:
        runner, cache_context = prepare_runner(
            dit_model=dit_model,
            vae_model=vae_model,
            model_dir=str(model_dir),
            debug=debug,
            ctx=ctx,
            dit_cache=False,
            vae_cache=False,
            dit_id="seedvr2_svdint4_quant",
            vae_id="seedvr2_svdint4_quant",
            encode_tiled=encode_tiled,
            encode_tile_size=(encode_tile_size, encode_tile_size),
            encode_tile_overlap=(64, 64),
            decode_tiled=False,
            decode_tile_size=(encode_tile_size, encode_tile_size),
            decode_tile_overlap=(64, 64),
            tile_debug="false",
            attention_mode=attention_mode,
        )
        ctx["cache_context"] = cache_context
        ctx["text_embeds"] = load_text_embeddings(script_directory, ctx["dit_device"], ctx["compute_dtype"], debug)
        if runner.dit is None:
            raise RuntimeError("SeedVR2 DiT model structure was not prepared")
        specs = _collect_linear_specs(runner.dit)
        stats, handles = _register_absmax_hooks(runner.dit)

        for case_index, (label, images) in enumerate(cases, start=1):
            case_batch = min(_valid_4n1(int(images.shape[0])), _valid_4n1(batch_size))
            case_ctx = dict(ctx)
            case_ctx["text_embeds"] = load_text_embeddings(script_directory, case_ctx["dit_device"], case_ctx["compute_dtype"], debug)
            LOG.info("Calibrating %d/%d: %s frames=%d", case_index, len(cases), label, int(images.shape[0]))
            processed, _info = compute_generation_info(
                ctx=case_ctx,
                images=images,
                resolution=resolution,
                max_resolution=max_resolution,
                batch_size=case_batch,
                uniform_batch_size=False,
                seed=seed + case_index,
                prepend_frames=0,
                temporal_overlap=min(temporal_overlap, max(0, case_batch - 1)),
                debug=debug,
            )
            case_ctx = encode_all_batches(
                runner,
                ctx=case_ctx,
                images=processed,
                debug=debug,
                batch_size=case_batch,
                uniform_batch_size=False,
                seed=seed + case_index,
                progress_callback=None,
                temporal_overlap=min(temporal_overlap, max(0, case_batch - 1)),
                resolution=resolution,
                max_resolution=max_resolution,
                input_noise_scale=0.0,
                color_correction="none",
            )
            case_ctx = upscale_all_batches(
                runner,
                ctx=case_ctx,
                debug=debug,
                progress_callback=None,
                seed=seed + case_index,
                latent_noise_scale=0.0,
                cache_model=True,
            )
            del case_ctx
            torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()
        if runner is not None:
            complete_cleanup(runner=runner, debug=debug, dit_cache=False, vae_cache=False)
        cleanup_text_embeddings(ctx, debug)
        torch.cuda.empty_cache()

    summary = {
        "calibration_cases": [label for label, _ in cases],
        "captured_linears": len(stats),
        "total_linears": len(specs),
    }
    return stats, specs, summary


def _smooth_from_stats(weight: torch.Tensor, x_absmax: torch.Tensor | None, alpha: float, clamp: tuple[float, float]) -> torch.Tensor:
    k = int(weight.shape[1])
    if x_absmax is None:
        return torch.ones(k, device=weight.device, dtype=weight.dtype)
    x = x_absmax.to(device=weight.device, dtype=torch.float32).clamp_min(1.0e-6)
    w = weight.detach().abs().float().amax(dim=0).clamp_min(1.0e-6)
    smooth = torch.pow(x, alpha) / torch.pow(w, 1.0 - alpha)
    smooth = smooth / smooth.mean().clamp_min(1.0e-6)
    smooth = smooth.clamp(float(clamp[0]), float(clamp[1]))
    return smooth.to(dtype=weight.dtype)


def _low_rank_residual(
    residual: torch.Tensor,
    rank: int,
    *,
    seed: int,
    oversample: int,
    niter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = int(residual.shape[0]), int(residual.shape[1])
    rank = min(rank, n, k)
    if rank <= 0:
        z_down = torch.zeros((k, 16), device=residual.device, dtype=residual.dtype)
        z_up = torch.zeros((n, 16), device=residual.device, dtype=residual.dtype)
        return z_down, z_up

    q = min(rank + max(0, oversample), n, k)
    generator = torch.Generator(device=residual.device)
    generator.manual_seed(seed)
    omega = torch.randn((k, q), device=residual.device, dtype=residual.dtype, generator=generator)
    basis = residual @ omega
    for _ in range(max(0, niter)):
        basis = residual @ (residual.transpose(0, 1) @ basis)
    basis, _r = torch.linalg.qr(basis.float(), mode="reduced")
    small = basis.transpose(0, 1) @ residual.float()
    u_hat, s, vh = torch.linalg.svd(small, full_matrices=False)
    u = basis @ u_hat[:, :rank]
    up = (u[:, :rank] * s[:rank].view(1, rank)).to(dtype=residual.dtype)
    down = vh[:rank, :].transpose(0, 1).contiguous().to(dtype=residual.dtype)
    return down, up


def quantize_checkpoint(
    *,
    checkpoint: Path,
    output: Path,
    stats: dict[str, torch.Tensor],
    specs: dict[str, tuple[int, int, bool]],
    rank: int,
    smooth_alpha: float,
    smooth_clamp: tuple[float, float],
    residual_oversample: int,
    residual_niter: int,
    device: str,
    seed: int,
) -> dict[str, object]:
    from torch import float16

    output.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    quantized = 0
    dense_kept = 0
    missing_stats: list[str] = []
    start = time.perf_counter()

    keys: list[str]
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        key_set = set(keys)
        quantized_weight_keys = {f"{name}.weight" for name in specs if f"{name}.weight" in key_set}

        for key in keys:
            if key not in quantized_weight_keys:
                if key.endswith(".bias") and key[:-5] in specs and f"{key[:-5]}.weight" in quantized_weight_keys:
                    continue
                tensors[key] = _dense_checkpoint_tensor(handle.get_tensor(key))
                dense_kept += 1
                continue

            name = key[:-7]
            weight_cpu = handle.get_tensor(key)
            weight = weight_cpu.to(device=device, dtype=float16, non_blocking=False).contiguous()
            x_absmax = stats.get(name)
            if x_absmax is None:
                missing_stats.append(name)
            smooth = _smooth_from_stats(weight, x_absmax, smooth_alpha, smooth_clamp)
            packed = pack_linear_weight(weight, smooth=smooth, return_dequant=True)
            residual = weight - packed.dequant_weight.to(device=weight.device, dtype=weight.dtype)
            down, up = _low_rank_residual(
                residual,
                rank,
                seed=seed + quantized,
                oversample=residual_oversample,
                niter=residual_niter,
            )
            tensors[f"{name}.qweight"] = packed.qweight.cpu().contiguous()
            tensors[f"{name}.wscales"] = packed.wscales.cpu().contiguous()
            tensors[f"{name}.smooth"] = packed.smooth.cpu().contiguous()
            tensors[f"{name}.svd_down"] = pack_svd_down(down, k_pad=packed.k_pad).cpu().contiguous()
            tensors[f"{name}.svd_up"] = pack_svd_up(up, n_pad=packed.n_pad, rank_pad=tensors[f"{name}.svd_down"].shape[1]).cpu().contiguous()

            bias_key = f"{name}.bias"
            if bias_key in key_set:
                bias = handle.get_tensor(bias_key).to(device=device, dtype=float16)
                tensors[f"{name}.bias_packed"] = pack_bias(bias, n_pad=packed.n_pad).cpu().contiguous()

            quantized += 1
            if quantized % 10 == 0 or quantized == len(quantized_weight_keys):
                LOG.info("Packed %d/%d Linear weights", quantized, len(quantized_weight_keys))
            del weight, residual, down, up, packed, weight_cpu
            torch.cuda.empty_cache()

    tmp = output.with_name(output.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    save_file(tensors, tmp, metadata=build_svdint4_metadata())
    os.replace(tmp, output)
    elapsed = time.perf_counter() - start
    return {
        "source": str(checkpoint),
        "output": str(output),
        "quantized_linears": quantized,
        "dense_tensors_kept": dense_kept,
        "missing_activation_stats": missing_stats,
        "rank": rank,
        "smooth_alpha": smooth_alpha,
        "smooth_clamp": list(smooth_clamp),
        "elapsed_seconds": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate and quantize a SeedVR2 DiT checkpoint to SVDInt4 single-file format.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--vae", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calib-path", required=True, type=Path, action="append")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--image-limit", type=int, default=2)
    parser.add_argument("--video-limit", type=int, default=2)
    parser.add_argument("--video-frames", type=int, default=5)
    parser.add_argument("--input-max-side", type=int, default=512)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--temporal-overlap", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-mode", default="sdpa", choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"])
    parser.add_argument("--encode-tiled", action="store_true")
    parser.add_argument("--encode-tile-size", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--smooth-alpha", type=float, default=0.5)
    parser.add_argument("--smooth-clamp-min", type=float, default=0.25)
    parser.add_argument("--smooth-clamp-max", type=float, default=4.0)
    parser.add_argument("--residual-oversample", type=int, default=8)
    parser.add_argument("--residual-niter", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sidecar", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    if args.rank <= 0 or args.rank % 16 != 0:
        raise ValueError("--rank must be a positive multiple of 16")
    model_dir = args.model_dir or args.checkpoint.parent
    cases = _load_calibration_cases(
        args.calib_path,
        image_limit=args.image_limit,
        video_limit=args.video_limit,
        video_frames=args.video_frames,
        input_max_side=args.input_max_side,
    )
    LOG.info("Loaded %d calibration case(s)", len(cases))
    stats, specs, calib_summary = collect_activation_stats(
        dit_model=args.checkpoint.name,
        vae_model=args.vae.name,
        model_dir=model_dir,
        cases=cases,
        resolution=args.resolution,
        max_resolution=args.max_resolution,
        batch_size=args.batch_size,
        temporal_overlap=args.temporal_overlap,
        device=args.device,
        attention_mode=args.attention_mode,
        encode_tiled=args.encode_tiled,
        encode_tile_size=args.encode_tile_size,
        seed=args.seed,
    )
    LOG.info("Captured activation stats for %d/%d Linear layers", len(stats), len(specs))
    quant_summary = quantize_checkpoint(
        checkpoint=args.checkpoint,
        output=args.output,
        stats=stats,
        specs=specs,
        rank=args.rank,
        smooth_alpha=args.smooth_alpha,
        smooth_clamp=(args.smooth_clamp_min, args.smooth_clamp_max),
        residual_oversample=args.residual_oversample,
        residual_niter=args.residual_niter,
        device=args.device,
        seed=args.seed,
    )
    sidecar = args.sidecar or args.output.with_name(args.output.name + ".json")
    sidecar.write_text(json.dumps({"calibration": calib_summary, "quantization": quant_summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOG.info("Wrote %s", args.output)
    LOG.info("Wrote %s", sidecar)


if __name__ == "__main__":
    main()
