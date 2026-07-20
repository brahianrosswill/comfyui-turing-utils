from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable


RESOLUTION_QUOTAS = {
    "480": {"landscape": 160, "portrait": 64, "square": 32},
    "540": {"landscape": 120, "portrait": 48, "square": 24},
    "720": {"landscape": 200, "portrait": 80, "square": 40},
    "1080": {"landscape": 160, "portrait": 64, "square": 32},
}
HIGH_RISK_RESOLUTION_QUOTAS = {
    "720": {"landscape": 16, "portrait": 8, "square": 8},
}
VALIDATION_RESOLUTION_QUOTAS = {"480": 64, "540": 48, "720": 80, "1080": 64}

FRAME_QUOTAS = {
    5: 64,
    9: 64,
    17: 96,
    33: 128,
    49: 128,
    65: 128,
    81: 160,
    97: 128,
    121: 128,
}
CONDITIONING_QUOTAS = {
    "src_video+0_images": 128,
    "src_video+1_image": 192,
    "src_video+2_images": 160,
    "src_video+3_images": 128,
    "src_video+5_images": 96,
    "src_video+ref_video+0_images": 96,
    "src_video+ref_video+1_image": 96,
    "src_video+ref_video+3_images": 80,
    "src_video+ref_video+5_images": 48,
}
WINDOW_ROLE_QUOTAS = {
    "single_full": 128,
    "first": 192,
    "middle": 256,
    "last": 192,
    "tail_padded": 192,
    "short_video": 64,
}
MOTION_QUOTAS = {
    "static_low": 192,
    "medium": 384,
    "high_subject": 256,
    "camera_motion": 128,
    "occlusion_complex": 64,
}
HIGH_RISK_FRAME_QUOTAS = {81: 10, 97: 8, 121: 14}
HIGH_RISK_CONDITIONING_QUOTAS = {
    "src_video+ref_video+5_images": 10,
    "src_video+ref_video+3_images": 8,
    "src_video+5_images": 6,
    "src_video+3_images": 4,
    "src_video+ref_video+1_image": 4,
}
HIGH_RISK_WINDOW_ROLE_QUOTAS = {
    "middle": 12,
    "last": 8,
    "tail_padded": 8,
    "first": 4,
}
HIGH_RISK_MOTION_QUOTAS = {
    "high_subject": 12,
    "camera_motion": 10,
    "occlusion_complex": 10,
}
CONTEXT_WINDOW_FRAME_CAP = 81
RESOLUTION_CONTEXT_WINDOW_FRAME_CAP = {
    "1080": 33,
}

STANDARD_SIZES = {
    ("480", "landscape"): (832, 480),
    ("480", "portrait"): (480, 832),
    ("480", "square"): (640, 640),
    ("540", "landscape"): (960, 544),
    ("540", "portrait"): (544, 960),
    ("540", "square"): (768, 768),
    ("720", "landscape"): (1280, 720),
    ("720", "portrait"): (720, 1280),
    ("720", "square"): (1024, 1024),
    ("1080", "landscape"): (1920, 1088),
    ("1080", "portrait"): (1088, 1920),
    ("1080", "square"): (1536, 1536),
}

SIZE_JITTER = {
    "480": [(768, 480), (896, 512), (704, 512)],
    "540": [(896, 544), (1024, 576), (832, 608)],
    "720": [(1152, 720), (1344, 768), (1216, 832)],
    "1080": [(1792, 1024), (1920, 1088), (1664, 1152)],
}
SQUARE_SIZE_JITTER = {
    "480": [(576, 576), (704, 704)],
    "540": [(704, 704), (832, 832)],
    "720": [(960, 960), (1088, 1088)],
    "1080": [(1408, 1408), (1600, 1600)],
}


def expand_quota(quota: dict, *, rng: random.Random) -> list:
    values = []
    for key, count in quota.items():
        values.extend([key] * int(count))
    rng.shuffle(values)
    return values


def scale_quota(quota: dict, count: int) -> dict:
    if count <= 0:
        return {}
    total = sum(int(v) for v in quota.values())
    if total <= 0:
        raise ValueError("quota total must be positive")
    entries = []
    for key, value in quota.items():
        exact = int(value) * count / total
        base = int(exact)
        if value and base == 0:
            base = 1
        entries.append([key, base, exact - int(exact)])
    current = sum(item[1] for item in entries)
    if current < count:
        for item in sorted(entries, key=lambda item: item[2], reverse=True):
            if current >= count:
                break
            item[1] += 1
            current += 1
    elif current > count:
        for item in sorted(entries, key=lambda item: (item[1] <= 1, item[2])):
            if current <= count:
                break
            if item[1] > 1:
                item[1] -= 1
                current -= 1
    return {item[0]: item[1] for item in entries if item[1] > 0}


def scale_resolution_quota(quota: dict, count: int) -> dict[tuple[str, str], int]:
    flat: dict[tuple[str, str], int] = {}
    for bucket, aspect_counts in quota.items():
        for aspect, aspect_count in aspect_counts.items():
            flat[(bucket, aspect)] = int(aspect_count)
    return scale_quota(flat, count)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def all_files(root: Path, suffixes: tuple[str, ...]) -> list[str]:
    if not root.exists():
        return []
    out = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            out.append(str(path))
    return sorted(out)


def case_media_pools(rows: Iterable[dict], calib_data: Path) -> tuple[list[str], list[str], list[dict]]:
    videos: list[str] = []
    images: list[str] = []
    prompts: list[dict] = []
    for row in rows:
        case = row.get("case", {})
        prompt = case.get("prompt")
        if prompt:
            prompts.append({"prompt": prompt, "source": row.get("source"), "task_type": row.get("task_type") or case.get("task_type")})
        video = case.get("video")
        if isinstance(video, str) and Path(video).exists():
            videos.append(video)
        elif isinstance(video, list):
            videos.extend(str(v) for v in video if Path(v).exists())
        image = case.get("image")
        if isinstance(image, str) and Path(image).exists():
            images.append(image)
        case_images = case.get("images")
        if isinstance(case_images, list):
            images.extend(str(v) for v in case_images if Path(v).exists())

    videos.extend(all_files(calib_data, (".mp4", ".mov", ".webm", ".avi")))
    images.extend(all_files(calib_data, (".jpg", ".jpeg", ".png", ".webp")))
    videos = sorted(dict.fromkeys(videos))
    images = sorted(dict.fromkeys(images))
    if not videos:
        raise RuntimeError("no source videos found")
    if len(images) < 5:
        raise RuntimeError("need at least five reference images")
    if not prompts:
        prompts = [{"prompt": "Edit the source video while preserving identity and motion.", "source": "synthetic", "task_type": "v2v"}]
    return videos, images, prompts


def parse_conditioning(value: str) -> tuple[bool, int]:
    has_ref_video = "+ref_video+" in value
    if value.endswith("0_images"):
        count = 0
    elif value.endswith("1_image"):
        count = 1
    else:
        count = int(value.rsplit("+", 1)[-1].split("_", 1)[0])
    return has_ref_video, count


def resolution_sequence(
    count: int,
    *,
    rng: random.Random,
    high_risk: bool = False,
    quota: dict | None = None,
) -> list[tuple[str, str]]:
    if quota is None:
        quota = HIGH_RISK_RESOLUTION_QUOTAS if high_risk else RESOLUTION_QUOTAS
    out: list[tuple[str, str]] = []
    for key, key_count in scale_resolution_quota(quota, count).items():
        out.extend([key] * key_count)
    rng.shuffle(out)
    return out


def validation_resolution_sequence(*, rng: random.Random) -> list[tuple[str, str]]:
    aspects = ["landscape", "portrait", "square", "landscape"]
    out: list[tuple[str, str]] = []
    for bucket, count in VALIDATION_RESOLUTION_QUOTAS.items():
        for i in range(count):
            out.append((bucket, aspects[i % len(aspects)]))
    rng.shuffle(out)
    return out


def size_for(bucket: str, aspect: str, index: int, *, nearby_aligned: bool) -> tuple[int, int]:
    if not nearby_aligned:
        return STANDARD_SIZES[(bucket, aspect)]
    if aspect == "square":
        values = SQUARE_SIZE_JITTER[bucket]
        return values[index % len(values)]
    width, height = SIZE_JITTER[bucket][index % len(SIZE_JITTER[bucket])]
    return (height, width) if aspect == "portrait" else (width, height)


def context_window_cap_for(resolution_bucket: str) -> int:
    return int(RESOLUTION_CONTEXT_WINDOW_FRAME_CAP.get(str(resolution_bucket), CONTEXT_WINDOW_FRAME_CAP))


def window_frame_count_for(frame_bucket: int, context_window_cap: int) -> int:
    return min(int(frame_bucket), int(context_window_cap))


def align_window_roles(
    resolutions: list[tuple[str, str]],
    frames: list[int],
    roles: list[str],
    count: int,
) -> list[str]:
    roles = roles[:count]
    restricted = {"single_full", "short_video"}
    invalid = [
        index
        for index in range(count)
        if int(frames[index]) > context_window_cap_for(resolutions[index][0]) and roles[index] in restricted
    ]
    donors = [
        index
        for index in range(count)
        if int(frames[index]) <= context_window_cap_for(resolutions[index][0]) and roles[index] not in restricted
    ]
    if len(donors) < len(invalid):
        raise RuntimeError("window-role quotas cannot be assigned within the resolution-specific frame caps")
    for invalid_index, donor_index in zip(invalid, donors[: len(invalid)], strict=True):
        roles[invalid_index], roles[donor_index] = roles[donor_index], roles[invalid_index]
    return roles


def build_rows(
    count: int,
    *,
    split: str,
    videos: list[str],
    images: list[str],
    prompts: list[dict],
    rng: random.Random,
) -> list[dict]:
    if split == "calib":
        resolutions = resolution_sequence(count, rng=rng)
        frames = expand_quota(scale_quota(FRAME_QUOTAS, count), rng=rng)
        conditioning = expand_quota(scale_quota(CONDITIONING_QUOTAS, count), rng=rng)
        window_roles = expand_quota(scale_quota(WINDOW_ROLE_QUOTAS, count), rng=rng)
        motions = expand_quota(scale_quota(MOTION_QUOTAS, count), rng=rng)
    elif split == "highrisk":
        resolutions = resolution_sequence(count, rng=rng, high_risk=True)
        frames = expand_quota(scale_quota(HIGH_RISK_FRAME_QUOTAS, count), rng=rng)
        conditioning = expand_quota(scale_quota(HIGH_RISK_CONDITIONING_QUOTAS, count), rng=rng)
        window_roles = expand_quota(scale_quota(HIGH_RISK_WINDOW_ROLE_QUOTAS, count), rng=rng)
        motions = expand_quota(scale_quota(HIGH_RISK_MOTION_QUOTAS, count), rng=rng)
    else:
        resolutions = validation_resolution_sequence(rng=rng)
        frames = [5, 9, 17, 33, 49, 65, 81, 97, 121] * ((count // 9) + 1)
        conditioning = list(CONDITIONING_QUOTAS) * ((count // len(CONDITIONING_QUOTAS)) + 1)
        window_roles = list(WINDOW_ROLE_QUOTAS) * ((count // len(WINDOW_ROLE_QUOTAS)) + 1)
        motions = list(MOTION_QUOTAS) * ((count // len(MOTION_QUOTAS)) + 1)
        rng.shuffle(frames)
        rng.shuffle(conditioning)
        rng.shuffle(window_roles)
        rng.shuffle(motions)

    assert len(resolutions) >= count
    window_roles = align_window_roles(resolutions, frames, window_roles, count)
    resolution_totals = Counter(bucket for bucket, _aspect in resolutions[:count])
    nearby_targets = {bucket: round(total * 0.20) for bucket, total in resolution_totals.items()}
    resolution_positions: Counter[str] = Counter()
    rows = []
    for i in range(count):
        bucket, aspect = resolutions[i]
        bucket_position = resolution_positions[bucket]
        nearby_aligned = bucket_position < nearby_targets[bucket]
        resolution_positions[bucket] += 1
        width, height = size_for(bucket, aspect, bucket_position, nearby_aligned=nearby_aligned)
        frame_bucket = int(frames[i % len(frames)])
        conditioning_signature = str(conditioning[i % len(conditioning)])
        context_window_cap = context_window_cap_for(bucket)
        window_frame_count = window_frame_count_for(frame_bucket, context_window_cap)
        window_role = str(window_roles[i])
        has_ref_video, image_count = parse_conditioning(conditioning_signature)
        prompt_row = prompts[i % len(prompts)]
        source_video = videos[i % len(videos)]
        reference_video = videos[(i * 17 + 11) % len(videos)] if has_ref_video else None
        reference_images = [images[(i * 31 + j * 7) % len(images)] for j in range(image_count)]
        video_value = [source_video, reference_video] if reference_video else source_video
        row = {
            "id": f"bernini_multires_{split}_{i:04d}",
            "split": split,
            "source": prompt_row.get("source"),
            "task_type": "v2v",
            "resolution_bucket": bucket,
            "width": width,
            "height": height,
            "max_image_size": max(width, height),
            "aspect_bucket": aspect,
            "size_variant": "nearby_aligned" if nearby_aligned else "standard",
            "source_total_frames": frame_bucket,
            "target_frames": window_frame_count,
            "window_frame_count": window_frame_count,
            "frame_bucket": frame_bucket,
            "context_window_size": context_window_cap,
            "window_role": window_role,
            "source_video": source_video,
            "reference_video": reference_video,
            "reference_images": reference_images,
            "reference_image_count": image_count,
            "has_reference_video": bool(reference_video),
            "prompt": prompt_row["prompt"],
            "negative_prompt": "",
            "fps": 16,
            "sample_policy": "uniform_time",
            "conditioning_signature": conditioning_signature,
            "motion_bucket": str(motions[i % len(motions)]),
            "edit_bucket": "bernini_video_edit",
            "padding_ratio": max(0.0, (context_window_cap - window_frame_count) / context_window_cap)
            if window_role == "tail_padded"
            else 0.0,
            "case": {
                "task_type": "v2v",
                "guidance_mode": "vae_txt_vit_wapg",
                "prompt": prompt_row["prompt"],
                "video": video_value,
                "images": reference_images,
                "output": f"/tmp/bernini_multires_{split}_{i:04d}.mp4",
            },
        }
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "resolution": dict(Counter(row["resolution_bucket"] for row in rows)),
        "frames": dict(Counter(str(row["frame_bucket"]) for row in rows)),
        "conditioning": dict(Counter(row["conditioning_signature"] for row in rows)),
        "window_role": dict(Counter(row["window_role"] for row in rows)),
        "aspect": dict(Counter(row["aspect_bucket"] for row in rows)),
        "size_variant": dict(Counter(row["size_variant"] for row in rows)),
        "motion": dict(Counter(row["motion_bucket"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the required Bernini SVDInt4 1024/256 calibration manifests.")
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--calib-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--high-risk-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base_rows = load_jsonl(args.base_manifest)
    videos, images, prompts = case_media_pools(base_rows, args.calib_data)
    calib = build_rows(
        1024,
        split="calib",
        videos=videos,
        images=images,
        prompts=prompts,
        rng=rng,
    )
    validation = build_rows(256, split="validation", videos=videos, images=images, prompts=prompts, rng=rng)
    high_risk = build_rows(args.high_risk_count, split="highrisk", videos=videos, images=images, prompts=prompts, rng=rng)

    calib_name = "bernini_multires_calib1024"
    calib_path = args.output_root / f"{calib_name}.jsonl"
    validation_path = args.output_root / "bernini_multires_validation256.jsonl"
    high_risk_path = args.output_root / f"bernini_multires_highrisk{args.high_risk_count}.jsonl"
    write_jsonl(calib_path, calib)
    write_jsonl(validation_path, validation)
    write_jsonl(high_risk_path, high_risk)
    summary = {
        "artifact_type": "bernini_svdint4_calibration_manifest",
        "seed": args.seed,
        "base_manifest": str(args.base_manifest),
        "calib_manifest": str(calib_path),
        "validation_manifest": str(validation_path),
        "high_risk_manifest": str(high_risk_path),
        "video_pool": len(videos),
        "image_pool": len(images),
        "prompt_pool": len(prompts),
        "calib": summarize(calib),
        "validation": summarize(validation),
        "high_risk": summarize(high_risk),
    }
    summary_path = args.output_root / "bernini_multires_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
