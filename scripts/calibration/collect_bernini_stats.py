from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import torch

from bernini_runtime import (
    DEFAULT_GUIDANCE_MODE,
    HookCollector,
    RunningStats,
    cleanup_transformer_memory,
    configure_transformer_memory,
    disable_transformer_block_offload,
    enable_transformer_block_offload,
    is_cuda_oom,
    configure_vae_memory,
    jsonable_args,
    load_storage_style_transformer_weights,
    rank_output_prefix,
    setup_paths,
    setup_ulysses_parallel,
)


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["_manifest_index"] = len(rows)
                rows.append(row)
    return rows


def run_row(pipeline, row: dict, args: argparse.Namespace, index: int) -> None:
    case = row["case"]
    frames = int(row.get("target_frames") or row.get("frame_bucket") or args.default_video_frames)
    height = int(row.get("height") or args.default_height)
    width = int(row.get("width") or args.default_width)
    max_image_size = int(row.get("max_image_size") or max(width, height))
    pipeline(
        case.get("task_type", "v2v"),
        case["prompt"],
        video=case.get("video"),
        image=case.get("image"),
        images=case.get("images"),
        output_path=case.get("output", "unused.mp4"),
        num_frames=frames,
        max_image_size=max_image_size,
        height=height,
        width=width,
        num_inference_steps=args.num_inference_steps,
        guidance_mode=case.get("guidance_mode") or args.guidance_mode,
        planning_step=args.planning_step,
        vit_denoising_step=args.vit_denoising_step,
        vit_txt_cfg=args.vit_txt_cfg,
        vit_img_cfg=args.vit_img_cfg,
        omega_vid=args.omega_vid,
        omega_img=args.omega_img,
        omega_txt=args.omega_txt,
        omega_tgt=args.omega_tgt,
        omega_scale=args.omega_scale,
        seed=args.seed + index,
        fps=int(row.get("fps") or args.fps),
        flow_shift=args.flow_shift,
        system_prompt="",
        write_output=False,
        use_truncate=args.use_truncate,
        max_sequence_length=args.max_sequence_length,
    )


def bucket_summary(rows: list[dict]) -> dict[str, dict[str, int]]:
    keys = (
        "resolution_bucket",
        "frame_bucket",
        "conditioning_signature",
        "window_role",
        "aspect_bucket",
        "motion_bucket",
    )
    out: dict[str, dict[str, int]] = {}
    for key in keys:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(key))
            counts[value] = counts.get(value, 0) + 1
        out[key] = counts
    return out


def validate_manifest(rows: list[dict]) -> None:
    if len(rows) != 1024:
        raise ValueError(f"Bernini release calibration requires exactly 1024 windows, got {len(rows)}")
    summary = bucket_summary(rows)
    expected = {
        "resolution_bucket": {"480": 256, "540": 192, "720": 320, "1080": 256},
        "frame_bucket": {"5": 64, "9": 64, "17": 96, "33": 128, "49": 128, "65": 128, "81": 160, "97": 128, "121": 128},
        "conditioning_signature": {
            "src_video+0_images": 128,
            "src_video+1_image": 192,
            "src_video+2_images": 160,
            "src_video+3_images": 128,
            "src_video+5_images": 96,
            "src_video+ref_video+0_images": 96,
            "src_video+ref_video+1_image": 96,
            "src_video+ref_video+3_images": 80,
            "src_video+ref_video+5_images": 48,
        },
        "window_role": {"single_full": 128, "first": 192, "middle": 256, "last": 192, "tail_padded": 192, "short_video": 64},
        "aspect_bucket": {"landscape": 640, "portrait": 256, "square": 128},
        "motion_bucket": {
            "static_low": 192,
            "medium": 384,
            "high_subject": 256,
            "camera_motion": 128,
            "occlusion_complex": 64,
        },
    }
    for key, values in expected.items():
        if summary[key] != values:
            raise ValueError(f"Bernini calibration {key} quota mismatch: got {summary[key]}, expected {values}")
    required = {
        "source_video",
        "reference_images",
        "reference_image_count",
        "has_reference_video",
        "resolution_bucket",
        "size_variant",
        "frame_bucket",
        "window_frame_count",
        "window_role",
        "aspect_bucket",
        "motion_bucket",
        "case",
    }
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"calibration row {index} is missing {sorted(missing)}")
        if int(row["reference_image_count"]) > 5:
            raise ValueError(f"calibration row {index} has more than five reference images")
        if row.get("split") != "calib":
            raise ValueError(f"calibration row {index} has split={row.get('split')!r}")
        reference_images = row.get("reference_images")
        if not isinstance(reference_images, list) or len(reference_images) != int(row["reference_image_count"]):
            raise ValueError(f"calibration row {index} reference-image count is inconsistent")
        has_reference_video = row.get("reference_video") is not None
        if bool(row["has_reference_video"]) != has_reference_video:
            raise ValueError(f"calibration row {index} reference-video flag is inconsistent")
        cap = int(row.get("context_window_size", 0))
        expected_window_frames = min(int(row["frame_bucket"]), cap)
        if int(row["window_frame_count"]) != expected_window_frames:
            raise ValueError(f"calibration row {index} has an invalid context-window frame count")
        if row["window_role"] in {"single_full", "short_video"} and int(row["frame_bucket"]) > cap:
            raise ValueError(f"calibration row {index} assigns {row['window_role']} beyond the window cap")

    size_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = str(row["resolution_bucket"])
        counts = size_counts.setdefault(bucket, {"standard": 0, "nearby_aligned": 0})
        variant = str(row["size_variant"])
        if variant not in counts:
            raise ValueError(f"unsupported size_variant={variant!r}")
        counts[variant] += 1
    expected_size_counts = {
        "480": {"standard": 205, "nearby_aligned": 51},
        "540": {"standard": 154, "nearby_aligned": 38},
        "720": {"standard": 256, "nearby_aligned": 64},
        "1080": {"standard": 205, "nearby_aligned": 51},
    }
    if size_counts != expected_size_counts:
        raise ValueError(f"Bernini calibration aligned-size quota mismatch: got {size_counts}, expected {expected_size_counts}")


def release_exception_memory(exc: BaseException, device: torch.device) -> None:
    """Drop traceback frames that may retain failed CUDA forward tensors."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        tb = current.__traceback__
        if tb is not None:
            traceback.clear_frames(tb)
            current.__traceback__ = None
        next_exc = current.__cause__ or current.__context__
        current.__cause__ = None
        current.__context__ = None
        current = next_exc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def release_case_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def restore_checkpoint(
    collector: HookCollector,
    output_prefix: Path,
    rows: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> set[int]:
    summary_path = output_prefix.with_suffix(".json")
    high_path = output_prefix.with_name(output_prefix.name + "_high.pt")
    low_path = output_prefix.with_name(output_prefix.name + "_low.pt")
    paths = (summary_path, high_path, low_path)
    if not all(path.exists() for path in paths):
        return set()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = summary.get("meta", {})
    saved_args = meta.get("args", {})
    expected = {
        "manifest": str(args.manifest),
        "high_source": str(args.high_source),
        "low_source": str(args.low_source),
    }
    mismatches = {
        key: (saved_args.get(key), value)
        for key, value in expected.items()
        if str(saved_args.get(key)) != value
    }
    if mismatches:
        raise RuntimeError(f"checkpoint task mismatch for {summary_path}: {mismatches}")
    completed = int(meta.get("completed_cases", 0))
    if completed < 0 or completed > len(rows):
        raise RuntimeError(f"invalid completed_cases={completed} for {len(rows)} rows")
    if completed == 0:
        return set()

    saved_indices = meta.get("completed_case_indices")
    if saved_indices is not None:
        completed_indices = {int(index) for index in saved_indices}
    else:
        # Older checkpoints only stored a success count. Runs process rows in
        # manifest order, so reconstruct the processed prefix by excluding the
        # recorded failures. This preserves successful statistics and retries
        # only failed rows after an interrupted or OOM-completed run.
        failed_indices = {
            int(failure["index"])
            for failure in meta.get("failures", [])
            if failure.get("index") is not None
        }
        processed = min(len(rows), completed + len(failed_indices))
        completed_indices = set(range(processed)) - failed_indices
    if len(completed_indices) != completed:
        raise RuntimeError(
            f"checkpoint completed index mismatch for {summary_path}: "
            f"count={completed}, indices={len(completed_indices)}"
        )
    if any(index < 0 or index >= len(rows) for index in completed_indices):
        raise RuntimeError(f"checkpoint has out-of-range completed indices: {summary_path}")
    high = torch.load(high_path, map_location="cpu", weights_only=True)
    low = torch.load(low_path, map_location="cpu", weights_only=True)
    collector.stats = {
        "high": RunningStats.from_saved(high, summary.get("high", []), torch.device("cpu")),
        "low": RunningStats.from_saved(low, summary.get("low", []), torch.device("cpu")),
    }
    pending = len(rows) - len(completed_indices)
    print(
        f"[resume] restored {completed}/{len(rows)} completed cases from {output_prefix}; "
        f"pending={pending}",
        flush=True,
    )
    return completed_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect the required 1024-window Bernini SVDInt4 activation statistics.")
    parser.add_argument("--bernini-repo", type=Path, required=True)
    parser.add_argument("--shim-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--high-source", type=Path, required=True)
    parser.add_argument("--low-source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--default-height", type=int, default=480)
    parser.add_argument("--default-width", type=int, default=832)
    parser.add_argument("--default-video-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--planning-step", type=int, default=1)
    parser.add_argument("--vit-denoising-step", type=int, default=1)
    parser.add_argument("--vit-txt-cfg", type=float, default=1.2)
    parser.add_argument("--vit-img-cfg", type=float, default=1.0)
    parser.add_argument("--guidance-mode", default=DEFAULT_GUIDANCE_MODE)
    parser.add_argument("--omega-vid", type=float, default=1.25)
    parser.add_argument("--omega-img", type=float, default=4.5)
    parser.add_argument("--omega-txt", type=float, default=4.0)
    parser.add_argument("--omega-tgt", type=float, default=0.5)
    parser.add_argument("--omega-scale", type=float, default=0.8)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--use-truncate", action="store_true")
    parser.add_argument("--disable-vae-tiling", action="store_true")
    parser.add_argument("--disable-vae-slicing", action="store_true")
    parser.add_argument("--vae-tile-size", type=int, default=1024)
    parser.add_argument("--vae-tile-stride", type=int, default=768)
    parser.add_argument("--ulysses", type=int, default=1, help="Ulysses sequence-parallel size for calibration.")
    parser.add_argument(
        "--keep-transformers-on-gpu",
        action="store_true",
        help="Keep both high/low transformers resident on GPU. Default offloads them so Bernini swaps branches per step.",
    )
    parser.add_argument(
        "--block-offload-transformers",
        action="store_true",
        help="Keep transformer non-block modules on GPU and offload Wan blocks to CPU between block forwards.",
    )
    parser.add_argument(
        "--retry-oom-with-block-offload",
        action="store_true",
        help="On CUDA OOM, clean up transformer memory, enable block offload, and retry the current case once.",
    )
    parser.add_argument(
        "--block-offload-count",
        type=int,
        default=16,
        help="Number of trailing Wan blocks to page per branch; 0 means all blocks.",
    )
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--save-every", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    setup_paths(args.bernini_repo, args.shim_root)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    from bernini.pipeline import BerniniPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device, ps, parallel_meta = setup_ulysses_parallel(args)
    output_prefix = rank_output_prefix(args.output_prefix, parallel_meta)

    started = time.perf_counter()
    pipeline = BerniniPipeline.from_pretrained(
        str(args.model_dir),
        device=device,
        use_unipc=True,
        use_src_id_rotary_emb=True,
        interpolate_src_id=True,
        max_trained_src_id=5,
    )
    vae_memory = configure_vae_memory(pipeline.vae, args)
    source_load = {}
    if args.high_source is not None:
        source_load["high"] = load_storage_style_transformer_weights(pipeline.model.diff_dec.transformer, args.high_source, "high")
    if args.low_source is not None:
        source_load["low"] = load_storage_style_transformer_weights(getattr(pipeline.model.diff_dec, "transformer_2", None), args.low_source, "low")
    transformer_memory = configure_transformer_memory(pipeline.model.diff_dec, args, device)

    collector = HookCollector()
    collector.register_branch("high", pipeline.model.diff_dec.transformer)
    collector.register_branch("low", getattr(pipeline.model.diff_dec, "transformer_2", None))

    rows = [] if args.load_only else load_manifest(args.manifest)
    if rows:
        validate_manifest(rows)
    if ps is not None and ps.dp_size > 1:
        rows = rows[ps.dp_rank :: ps.dp_size]
    failures = []
    completed_indices = restore_checkpoint(collector, output_prefix, rows, args, device) if args.resume else set()

    def run_isolated_attempt(row: dict, index: int) -> None:
        aggregate = collector.stats
        collector.stats = {"high": RunningStats(), "low": RunningStats()}
        try:
            with torch.inference_mode():
                run_row(pipeline, row, args, index)
        except Exception:
            collector.stats = aggregate
            raise
        attempt = collector.stats
        collector.stats = aggregate
        aggregate["high"].merge(attempt["high"])
        aggregate["low"].merge(attempt["low"])
        aggregate["high"].to("cpu")
        aggregate["low"].to("cpu")
        del attempt
        release_case_memory(device)

    def current_meta() -> dict:
        return {
            "model_dir": str(args.model_dir),
            "manifest": str(args.manifest),
            "selected_cases": len(rows),
            "completed_cases": len(completed_indices),
            "completed_case_indices": sorted(completed_indices),
            "failures": failures,
            "seconds": time.perf_counter() - started,
            "args": jsonable_args(args) | {"device": str(device)},
            "parallel": parallel_meta,
            "source_load": source_load,
            "vae_memory": vae_memory,
            "transformer_memory": transformer_memory,
            "bucket_summary": bucket_summary(rows),
        }

    try:
        for i, row in enumerate(rows):
            if i in completed_indices:
                continue
            print(
                f"[case {i + 1}/{len(rows)}] id={row.get('id')} "
                f"res={row.get('resolution_bucket')} frames={row.get('frame_bucket')} "
                f"role={row.get('window_role')} cond={row.get('conditioning_signature')}",
                flush=True,
            )
            try:
                run_isolated_attempt(row, i)
                completed_indices.add(i)
            except Exception as exc:
                retried = False
                if (
                    args.retry_oom_with_block_offload
                    and is_cuda_oom(exc)
                    and not getattr(args, "block_offload_transformers", False)
                ):
                    print("[case retry] CUDA OOM; enabling transformer block offload and retrying current case", flush=True)
                    release_exception_memory(exc, device)
                    cleanup_transformer_memory(pipeline.model.diff_dec, args, device)
                    transformer_memory = enable_transformer_block_offload(
                        pipeline.model.diff_dec, args, device, args.block_offload_count
                    )
                    try:
                        run_isolated_attempt(row, i)
                        completed_indices.add(i)
                        retried = True
                    except Exception as retry_exc:
                        exc = retry_exc
                        if is_cuda_oom(retry_exc) and args.block_offload_count > 0:
                            print("[case retry] partial block offload OOM; expanding to all blocks", flush=True)
                            release_exception_memory(retry_exc, device)
                            cleanup_transformer_memory(pipeline.model.diff_dec, args, device)
                            transformer_memory = enable_transformer_block_offload(
                                pipeline.model.diff_dec, args, device, 0
                            )
                            try:
                                run_isolated_attempt(row, i)
                                completed_indices.add(i)
                                retried = True
                            except Exception as full_retry_exc:
                                exc = full_retry_exc
                if retried:
                    transformer_memory = disable_transformer_block_offload(
                        pipeline.model.diff_dec, args, device
                    )
                    if args.save_every > 0 and (i + 1) % args.save_every == 0:
                        collector.save(output_prefix, current_meta())
                    continue
                release_exception_memory(exc, device)
                cleanup_transformer_memory(pipeline.model.diff_dec, args, device)
                if getattr(args, "block_offload_transformers", False):
                    transformer_memory = enable_transformer_block_offload(
                        pipeline.model.diff_dec, args, device, args.block_offload_count
                    )
                failures.append(
                    {
                        "index": i,
                        "id": row.get("id"),
                        "resolution_bucket": row.get("resolution_bucket"),
                        "frame_bucket": row.get("frame_bucket"),
                        "window_role": row.get("window_role"),
                        "conditioning_signature": row.get("conditioning_signature"),
                        "error": repr(exc),
                    }
                )
                print(f"[case failed] {repr(exc)}", flush=True)
                if len(rows) <= 1:
                    raise
            if args.save_every > 0 and (i + 1) % args.save_every == 0:
                collector.save(output_prefix, current_meta())
    finally:
        collector.save(output_prefix, current_meta())
        collector.close()
        print(f"wrote {output_prefix}.pt/.json and branch pt files", flush=True)
        try:
            import torch.distributed as dist

            if parallel_meta.get("enabled") and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
