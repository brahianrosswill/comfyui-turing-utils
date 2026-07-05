from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class LoraSpec:
    path: Path
    scale: float = 1.0


def add_lora_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lora",
        action="append",
        default=[],
        metavar="PATH[:SCALE]",
        help=(
            "LoRA adapter to fuse before quantization. Repeat for multiple "
            "adapters; SCALE defaults to 1.0."
        ),
    )


def parse_lora_spec(value: str) -> LoraSpec:
    raw = value.strip()
    if not raw:
        raise argparse.ArgumentTypeError("--lora value must not be empty")

    path_text = raw
    scale = 1.0
    if ":" in raw:
        head, tail = raw.rsplit(":", 1)
        if head and tail:
            try:
                parsed_scale = float(tail)
            except ValueError:
                parsed_scale = None
            if parsed_scale is not None:
                path_text = head
                scale = parsed_scale

    if not path_text:
        raise argparse.ArgumentTypeError("--lora path must not be empty")
    if not math.isfinite(scale):
        raise argparse.ArgumentTypeError("--lora scale must be finite")

    return LoraSpec(path=Path(path_text).expanduser(), scale=scale)


def parse_lora_specs(values: Iterable[str] | None) -> tuple[LoraSpec, ...]:
    return tuple(parse_lora_spec(value) for value in (values or ()))


def format_lora_specs(specs: Iterable[LoraSpec]) -> list[dict[str, object]]:
    return [{"path": str(spec.path), "scale": spec.scale} for spec in specs]


def reject_lora_specs(specs: Iterable[LoraSpec], *, tool_name: str) -> None:
    specs = tuple(specs)
    if not specs:
        return
    formatted = ", ".join(f"{spec.path} scale={spec.scale:g}" for spec in specs)
    raise SystemExit(
        f"{tool_name} received LoRA adapter(s), but this quantizer does not "
        f"implement LoRA fusion yet: {formatted}. Merge the adapters into the "
        "checkpoint first, or add a real fusion backend before quantization."
    )
