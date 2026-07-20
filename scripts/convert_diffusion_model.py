from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from model_format import SUPPORTED_ARCHITECTURES, SVDINT4_FORMAT  # noqa: E402
from svdint4.packing import pack_bias, pack_linear_weight, pack_svd_down, pack_svd_up  # noqa: E402


BLOCK_WEIGHT_RE = re.compile(r"^blocks\.(\d+)\..+\.weight$")
SOURCE_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "")


def strip_source_prefix(name: str) -> str:
    for prefix in SOURCE_PREFIXES:
        if prefix and name.startswith(prefix):
            return name[len(prefix) :]
    return name


def source_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            view = handle.get_slice(name)
            normalized = strip_source_prefix(name)
            rows.append(
                {
                    "name": name,
                    "normalized": normalized,
                    "shape": tuple(int(dim) for dim in view.get_shape()),
                }
            )
    return rows


def selected_linears(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    for row in rows:
        normalized = str(row["normalized"])
        match = BLOCK_WEIGHT_RE.match(normalized)
        if match is None or len(row["shape"]) != 2:
            continue
        selected.append(
            {
                **row,
                "block": int(match.group(1)),
                "source_base": str(row["name"])[: -len(".weight")],
                "base": normalized[: -len(".weight")],
            }
        )
    return sorted(selected, key=lambda row: (int(row["block"]), str(row["base"])))


def lowrank_svd(weight: torch.Tensor, rank: int, oversample: int, iterations: int):
    work = weight.float()
    q = min(rank + oversample, min(work.shape))
    try:
        u, s, v = torch.pca_lowrank(work, q=q, center=False, niter=iterations)
    except RuntimeError:
        u, s, vh = torch.linalg.svd(work, full_matrices=False)
        v = vh.transpose(0, 1)
    root = s[:rank].sqrt()
    down = v[:, :rank].contiguous() * root.unsqueeze(0)
    up = u[:, :rank].contiguous() * root.unsqueeze(0)
    return down, up


def quantize_linear(
    base: str,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    smooth: torch.Tensor | None,
    rank: int,
    oversample: int,
    iterations: int,
    validate: bool,
):
    if weight.dtype not in (torch.float16, torch.bfloat16):
        weight = weight.to(torch.float16)
    n, k = weight.shape
    started = time.perf_counter()
    down, up = lowrank_svd(weight, rank, oversample, iterations)
    lowrank = up @ down.transpose(0, 1)
    residual = (weight.float() - lowrank).to(weight.dtype)
    if smooth is None:
        smooth = torch.ones(k, device=weight.device, dtype=weight.dtype)
    else:
        if tuple(smooth.shape) != (k,):
            raise ValueError(f"{base}: smooth shape {tuple(smooth.shape)} does not match K={k}")
        smooth = smooth.to(device=weight.device, dtype=weight.dtype)

    packed = pack_linear_weight(residual, smooth=smooth, return_dequant=validate)
    tensors = {
        f"{base}.qweight": packed.qweight.cpu(),
        f"{base}.wscales": packed.wscales.cpu(),
        f"{base}.smooth": packed.smooth.cpu(),
        f"{base}.svd_down": pack_svd_down(down.to(weight.dtype), k_pad=packed.k_pad).cpu(),
        f"{base}.svd_up": pack_svd_up(up.to(weight.dtype), n_pad=packed.n_pad, rank_pad=rank).cpu(),
    }
    if bias is not None:
        tensors[f"{base}.bias_packed"] = pack_bias(bias.to(weight.dtype), n_pad=packed.n_pad).cpu()

    stats = {
        "name": base,
        "shape": [n, k],
        "rank": rank,
        "seconds": time.perf_counter() - started,
        "smooth_min": float(smooth.float().min().item()),
        "smooth_max": float(smooth.float().max().item()),
    }
    if validate:
        approximation = packed.dequant_weight.float() + lowrank
        error = (approximation - weight.float()).abs()
        stats.update(
            {
                "max_abs_error": float(error.max().item()),
                "mean_abs_error": float(error.mean().item()),
                "weight_abs_mean": float(weight.float().abs().mean().item()),
            }
        )
    return tensors, stats


def convert(args: argparse.Namespace) -> dict[str, object]:
    if args.rank <= 0 or args.rank % 16:
        raise ValueError("--rank must be a positive multiple of 16")
    rows = source_rows(args.input)
    selected = selected_linears(rows)
    if not selected:
        raise RuntimeError(f"no Wan block Linear weights found in {args.input}")
    if args.expected_linears and len(selected) != args.expected_linears:
        raise RuntimeError(f"found {len(selected)} block Linear weights, expected {args.expected_linears}")

    smooth_map = torch.load(args.smooth, map_location="cpu", weights_only=True) if args.smooth else {}
    selected_names = {str(row["name"]) for row in selected}
    selected_biases = {str(row["source_base"]) + ".bias" for row in selected}
    output_tensors: dict[str, torch.Tensor] = {}
    with safe_open(args.input, framework="pt", device="cpu") as handle:
        source_keys = set(handle.keys())
        for row in rows:
            name = str(row["name"])
            if name in selected_names or name in selected_biases:
                continue
            output_name = strip_source_prefix(name)
            if output_name in output_tensors:
                raise KeyError(f"duplicate output key after prefix normalization: {output_name}")
            output_tensors[output_name] = handle.get_tensor(name)

        device = torch.device(args.device)
        stats = []
        for index, row in enumerate(selected, 1):
            base = str(row["base"])
            source_base = str(row["source_base"])
            print(f"[{index}/{len(selected)}] {base}", flush=True)
            weight = handle.get_tensor(str(row["name"])).to(device, non_blocking=True)
            bias_name = source_base + ".bias"
            bias = handle.get_tensor(bias_name).to(device, non_blocking=True) if bias_name in source_keys else None
            tensors, layer_stats = quantize_linear(
                base,
                weight,
                bias,
                smooth_map.get(base),
                args.rank,
                args.oversample,
                args.iterations,
                args.validate,
            )
            output_tensors.update(tensors)
            stats.append(layer_stats)
            del weight, bias, tensors
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metadata = {
        "format": SVDINT4_FORMAT,
        "architecture": args.architecture,
        "quantization": "int4_weight_int4_activation_svd",
        "rank": str(args.rank),
        "packed_linear_count": str(len(selected)),
        "policy": "calibrated_smooth" if args.smooth else "weight_only_unit_smooth",
        "calibration_mode": "external_activation_stats" if args.smooth else "none",
    }
    if args.branch:
        metadata["branch"] = args.branch

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    save_file(output_tensors, temporary, metadata=metadata)
    os.replace(temporary, args.output)

    report = {
        "format": SVDINT4_FORMAT,
        "architecture": args.architecture,
        "input": str(args.input),
        "output": str(args.output),
        "branch": args.branch,
        "rank": args.rank,
        "smooth": str(args.smooth) if args.smooth else None,
        "calibrated": bool(args.smooth),
        "packed_linear_count": len(selected),
        "tensor_count": len(output_tensors),
        "bytes": args.output.stat().st_size,
        "layers": stats,
    }
    report_path = args.report or args.output.with_name(args.output.name + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a dense Wan DiT branch to the canonical SVDInt4 format.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--architecture", required=True, choices=sorted(SUPPORTED_ARCHITECTURES))
    parser.add_argument("--branch", choices=("high", "low"))
    parser.add_argument("--smooth", type=Path, help="Optional calibrated smooth-factor .pt file; omit for data-free conversion.")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--expected-linears", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = convert(args)
    print(json.dumps({key: value for key, value in report.items() if key != "layers"}, indent=2))


if __name__ == "__main__":
    main()
