from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from model_format import SUPPORTED_ARCHITECTURES, SVDINT4_FORMAT  # noqa: E402


def _sidecar_path(output: Path) -> Path:
    return output.with_name(output.name + ".json")


def _count_packed_linears(keys: list[str]) -> int:
    return sum(1 for key in keys if key.endswith(".qweight"))


def repack_single_file(src: Path, dst: Path, architecture: str, sidecar: Path | None) -> None:
    tensors = {}
    with safe_open(src, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        source_metadata = handle.metadata() or {}
        for key in keys:
            tensors[key] = handle.get_tensor(key).contiguous()

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    save_file(tensors, tmp, metadata={"format": SVDINT4_FORMAT, "architecture": architecture})
    os.replace(tmp, dst)

    if sidecar is not None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "source": str(src),
            "output": str(dst),
            "format": SVDINT4_FORMAT,
            "architecture": architecture,
            "tensor_count": len(keys),
            "packed_linear_count": _count_packed_linears(keys),
            "source_metadata": source_metadata,
        }
        sidecar.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repack an SVDInt4 single-file safetensors asset with minimal metadata.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--architecture", required=True, choices=sorted(SUPPORTED_ARCHITECTURES))
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="Sidecar JSON path. Defaults to '<output>.json'. Use --no-sidecar to skip it.",
    )
    parser.add_argument("--no-sidecar", action="store_true")
    args = parser.parse_args()

    sidecar = None if args.no_sidecar else (args.sidecar or _sidecar_path(args.output))
    repack_single_file(args.input, args.output, args.architecture, sidecar)
    print(f"wrote {args.output}")
    if sidecar is not None:
        print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
