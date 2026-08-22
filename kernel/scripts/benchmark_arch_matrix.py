#!/usr/bin/env python3
"""Run the same benchmark suite on multiple local CUDA architectures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "kernel" / "scripts" / "benchmark_backends.py"


def _device_indices(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("devices must be comma-separated CUDA indices")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=_device_indices, default=(0, 1))
    parser.add_argument("--suite", default="all", choices=("linear", "preprocess", "attention", "quality", "all"))
    parser.add_argument("--rows", default="4096,8192")
    parser.add_argument("--sequences", default="4096,8192")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmark-arch-matrix.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("architecture matrix benchmark requires CUDA")

    reports = []
    failures = []
    for index in args.devices:
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is unavailable")
        capability = tuple(torch.cuda.get_device_capability(index))
        command = [
            sys.executable,
            str(BENCHMARK),
            "--device",
            f"cuda:{index}",
            "--suite",
            args.suite,
            "--rows",
            args.rows,
            "--sequences",
            args.sequences,
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        report = {
            "device": index,
            "name": torch.cuda.get_device_name(index),
            "architecture": f"sm{capability[0]}{capability[1]}",
            "command": command,
            "returncode": completed.returncode,
            "output": completed.stdout,
        }
        reports.append(report)
        if completed.returncode:
            failures.append(report["architecture"])

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "reports": reports,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output.resolve()}")
    for report in reports:
        print(f"\n## {report['name']} ({report['architecture']})\n{report['output']}")
    if failures:
        raise RuntimeError(f"benchmark failed for: {', '.join(failures)}")


if __name__ == "__main__":
    main()
