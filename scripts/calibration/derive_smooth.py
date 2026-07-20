from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open


BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")
SOURCE_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "")


def parse_int_list(value: str | None) -> set[int] | None:
    if value is None or value == "":
        return None
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def keep_name(name: str, blocks: set[int] | None) -> bool:
    if blocks is None:
        return True
    match = BLOCK_RE.match(name)
    return match is not None and int(match.group(1)) in blocks


def derive_smooth(act_stat: torch.Tensor, weight: torch.Tensor, alpha: float, clamp_min: float, clamp_max: float):
    x = act_stat.float().clamp_min(1e-6)
    w = weight.float().abs().amax(dim=0).clamp_min(1e-6)
    smooth = x.pow(alpha) / w.pow(1.0 - alpha)
    smooth = smooth / smooth.median().clamp_min(1e-6)
    return smooth.clamp(clamp_min, clamp_max).to(torch.float16)


def tensor_summary(value: torch.Tensor) -> dict[str, float]:
    work = value.detach().float().flatten()
    if work.numel() == 0:
        return {
            "min": 0.0,
            "p01": 0.0,
            "median": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    qs = torch.quantile(work, torch.tensor([0.01, 0.5, 0.99], device=work.device))
    return {
        "min": float(work.min().item()),
        "p01": float(qs[0].item()),
        "median": float(qs[1].item()),
        "p99": float(qs[2].item()),
        "max": float(work.max().item()),
        "mean": float(work.mean().item()),
    }


def get_weight_tensor(handle, name: str) -> torch.Tensor | None:
    keys = set(handle.keys())
    for prefix in SOURCE_PREFIXES:
        weight_name = f"{prefix}{name}.weight"
        if weight_name in keys:
            return handle.get_tensor(weight_name)
    return None


def main():
    parser = argparse.ArgumentParser(description="Derive per-channel smooth factors from Bernini-R activation stats.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clamp-min", type=float, default=0.25)
    parser.add_argument("--clamp-max", type=float, default=4.0)
    parser.add_argument("--stat-key", default="p999", choices=("amax", "p99", "p999", "p9999"))
    parser.add_argument("--blocks", type=str, default=None, help="Comma/range list, e.g. 0,1,4-7")
    args = parser.parse_args()

    stats = torch.load(args.stats, map_location="cpu")
    smooth: dict[str, torch.Tensor] = {}
    rows = []
    clamp_min_hits = 0
    clamp_max_hits = 0
    total_channels = 0
    blocks = parse_int_list(args.blocks)
    with safe_open(args.source, framework="pt", device="cpu") as f:
        preferred_suffix = f".{args.stat_key}"
        names = sorted(k[: -len(preferred_suffix)] for k in stats if k.endswith(preferred_suffix))
        if not names and args.stat_key != "amax":
            print(f"stats key {args.stat_key!r} not found; falling back to 'amax'", flush=True)
            names = sorted(k[: -len(".amax")] for k in stats if k.endswith(".amax"))
        for name in names:
            if not keep_name(name, blocks):
                continue
            weight = get_weight_tensor(f, name)
            if weight is None:
                continue
            print(name, flush=True)
            act_key = f"{name}.{args.stat_key}"
            if act_key not in stats:
                act_key = f"{name}.amax"
            act = stats[act_key]
            value = derive_smooth(
                act,
                weight,
                alpha=args.alpha,
                clamp_min=args.clamp_min,
                clamp_max=args.clamp_max,
            )
            smooth[name] = value
            value_f = value.float()
            clamp_min_hits += int((value_f <= args.clamp_min).sum().item())
            clamp_max_hits += int((value_f >= args.clamp_max).sum().item())
            total_channels += int(value.numel())
            rows.append(
                {
                    "name": name,
                    "channels": int(value.numel()),
                    **tensor_summary(value),
                    "clamp_min_hits": int((value_f <= args.clamp_min).sum().item()),
                    "clamp_max_hits": int((value_f >= args.clamp_max).sum().item()),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(smooth, args.output)
    summary_path = args.output.with_suffix(".json")
    all_smooth = torch.cat([value.float().flatten() for value in smooth.values()]) if smooth else torch.empty(0)
    summary_path.write_text(
        json.dumps(
            {
                "source": str(args.source),
                "stats": str(args.stats),
                "alpha": args.alpha,
                "clamp_min": args.clamp_min,
                "clamp_max": args.clamp_max,
                "stat_key": args.stat_key,
                "blocks": sorted(blocks) if blocks is not None else None,
                "layer_count": len(rows),
                "channel_count": total_channels,
                "smooth_summary": tensor_summary(all_smooth),
                "clamp_min_hits": clamp_min_hits,
                "clamp_max_hits": clamp_max_hits,
                "clamp_min_hit_fraction": clamp_min_hits / max(total_channels, 1),
                "clamp_max_hit_fraction": clamp_max_hits / max(total_channels, 1),
                "layers": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
