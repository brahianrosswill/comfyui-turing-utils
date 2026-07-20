from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_branch(prefix: Path, branch: str) -> dict[str, torch.Tensor]:
    path = prefix.with_name(prefix.name + f"_{branch}.pt")
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def load_summary(prefix: Path) -> dict:
    path = prefix.with_suffix(".json")
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def merged_source_args(summaries: list[dict]) -> dict[str, str]:
    source_pairs = set()
    for summary in summaries:
        args = summary.get("meta", {}).get("args", {})
        source_pairs.add((args.get("high_source", ""), args.get("low_source", "")))
    if len(source_pairs) != 1:
        raise RuntimeError(f"cannot merge calibration shards with different source files: {sorted(source_pairs)}")
    high_source, low_source = next(iter(source_pairs))
    return {
        "high_source": high_source,
        "low_source": low_source,
    }


def row_map(summary: dict, branch: str) -> dict[str, dict]:
    return {row["name"]: row for row in summary.get(branch, [])}


def completed_cases(summaries: list[dict]) -> int:
    grouped: dict[tuple[str, str, str, str], list[int]] = {}
    for summary in summaries:
        meta = summary.get("meta", {})
        args = meta.get("args", {})
        parallel = meta.get("parallel", {})
        key = (
            str(args.get("manifest", "")),
            str(args.get("high_source", "")),
            str(args.get("low_source", "")),
            str(parallel.get("dp_rank", 0)),
        )
        grouped.setdefault(key, []).append(int(meta.get("completed_cases", 0)))

    total = 0
    for values in grouped.values():
        # Ulysses ranks process the same manifest rows with different sequence
        # shards, so their case counts should not be added together.
        total += max(values) if values else 0
    return total


def dedupe_failures(summaries: list[dict]) -> list[dict]:
    seen = set()
    failures = []
    for summary in summaries:
        meta = summary.get("meta", {})
        args = meta.get("args", {})
        manifest = str(args.get("manifest", ""))
        for failure in meta.get("failures", []):
            key = (
                manifest,
                failure.get("index"),
                failure.get("id"),
                failure.get("error"),
            )
            if key in seen:
                continue
            seen.add(key)
            failures.append(failure)
    return failures


def merge_branch(prefixes: list[Path], summaries: list[dict], branch: str) -> tuple[dict[str, torch.Tensor], list[dict]]:
    branch_tensors = [load_branch(prefix, branch) for prefix in prefixes]
    rows_by_prefix = [row_map(summary, branch) for summary in summaries]
    names = sorted({key[: -len(".amax")] for tensors in branch_tensors for key in tensors if key.endswith(".amax")})
    max_stats = ("amax", "p99", "p999", "p9999")

    out: dict[str, torch.Tensor] = {}
    summary_rows: list[dict] = []
    for name in names:
        stat_values: dict[str, list[torch.Tensor]] = {label: [] for label in max_stats}
        mean_sum = None
        token_total = 0
        call_total = 0
        shapes: set[tuple[int, ...]] = set()
        for tensors, rows in zip(branch_tensors, rows_by_prefix, strict=True):
            amax_key = f"{name}.amax"
            mean_key = f"{name}.mean_abs"
            if amax_key not in tensors or mean_key not in tensors:
                continue
            row = rows.get(name, {})
            tokens = int(row.get("tokens", 0))
            calls = int(row.get("calls", 0))
            for label in max_stats:
                stat_key = f"{name}.{label}"
                if stat_key in tensors:
                    stat_values[label].append(tensors[stat_key].float())
            weighted = tensors[mean_key].float() * max(tokens, 1)
            mean_sum = weighted if mean_sum is None else mean_sum + weighted
            token_total += max(tokens, 1)
            call_total += calls
            for shape in row.get("shapes", []):
                shapes.add(tuple(int(x) for x in shape))

        if not stat_values["amax"]:
            continue
        merged_stats = {
            label: torch.stack(values, dim=0).amax(dim=0)
            for label, values in stat_values.items()
            if values
        }
        amax = merged_stats["amax"]
        mean_abs = mean_sum / max(token_total, 1)
        for label, value in merged_stats.items():
            out[f"{name}.{label}"] = value.to(torch.float16)
        out[f"{name}.mean_abs"] = mean_abs.to(torch.float16)
        row = {
            "name": name,
            "channels": int(amax.numel()),
            "tokens": int(token_total),
            "calls": int(call_total),
            "mean_abs_mean": float(mean_abs.mean().item()),
            "mean_abs_max": float(mean_abs.max().item()),
            "shapes": [list(shape) for shape in sorted(shapes)[:16]],
        }
        for label, value in merged_stats.items():
            row[f"{label}_mean"] = float(value.mean().item())
            row[f"{label}_max"] = float(value.max().item())
        summary_rows.append(row)
    return out, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge split Bernini calibration stats.")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("prefixes", type=Path, nargs="+")
    args = parser.parse_args()

    summaries = [load_summary(prefix) for prefix in args.prefixes]
    high, high_rows = merge_branch(args.prefixes, summaries, "high")
    low, low_rows = merge_branch(args.prefixes, summaries, "low")

    completed = completed_cases(summaries)
    failures = dedupe_failures(summaries)

    payload = {
        "artifact_type": "bernini_svdint4_linear_input_stats",
        "meta": {
            "merged_from": [str(prefix) for prefix in args.prefixes],
            "calibration_sources": merged_source_args(summaries),
            "completed_cases": completed,
            "failures": failures,
        },
        "high": high,
        "low": low,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.save(high, args.output_prefix.with_name(args.output_prefix.name + "_high.pt"))
    torch.save(low, args.output_prefix.with_name(args.output_prefix.name + "_low.pt"))
    torch.save(payload, args.output_prefix.with_suffix(".pt"))

    registered = summaries[0].get("registered", {}) if summaries else {}
    summary = {
        "artifact_type": payload["artifact_type"],
        "meta": payload["meta"],
        "registered": registered,
        "high": high_rows,
        "low": low_rows,
    }
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_prefix": str(args.output_prefix),
                "completed_cases": completed,
                "failures": len(failures),
                "high_layers": len(high_rows),
                "low_layers": len(low_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
