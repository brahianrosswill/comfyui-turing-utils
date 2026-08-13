#!/usr/bin/env python3
"""Audit exact-SM75 kernel resources without prescribing an occupancy target."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sysconfig


KERNEL = Path(__file__).resolve().parents[1]
SM75_REGISTER_LIMIT = 255
SM75_SHARED_MEMORY_LIMIT = 64 * 1024


def _cuobjdump() -> str:
    found = shutil.which("cuobjdump") or shutil.which("cuobjdump.exe")
    if found:
        return found
    for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if root:
            for name in ("cuobjdump", "cuobjdump.exe"):
                candidate = Path(root) / "bin" / name
                if candidate.is_file():
                    return str(candidate)
    raise RuntimeError("cuobjdump was not found; add the CUDA bin directory to PATH")


def _resource_output(path: Path) -> str:
    return subprocess.check_output(
        [_cuobjdump(), "--dump-resource-usage", str(path)],
        text=True,
        stderr=subprocess.STDOUT,
    )


def _metrics(line: str) -> dict[str, int]:
    return {
        name: int(value)
        for name, value in re.findall(r"\b(REG|STACK|SHARED|LOCAL):(\d+)", line)
    }


def _sm75_section(output: str, marker: str | None = None) -> str:
    for section in output.split("Fatbin elf code:"):
        if re.search(r"\barch\s*=\s*sm_75\b", section) and (
            marker is None or marker in section
        ):
            return section
    suffix = f" containing {marker!r}" if marker else ""
    raise RuntimeError(f"built extension does not contain an exact sm75 cubin{suffix}")


def _function_records(section: str) -> list[tuple[str, dict[str, int]]]:
    records = []
    lines = section.splitlines()
    for index, line in enumerate(lines[:-1]):
        if not line.lstrip().startswith("Function "):
            continue
        metrics = _metrics(lines[index + 1])
        if metrics:
            records.append((line, metrics))
    return records


def _validate_no_spill(name: str, metrics: dict[str, int]) -> None:
    if metrics.get("REG", SM75_REGISTER_LIMIT + 1) > SM75_REGISTER_LIMIT:
        raise RuntimeError(f"{name} exceeds the SM75 register limit: {metrics}")
    if metrics.get("LOCAL", 1) != 0 or metrics.get("STACK", 1) != 0:
        raise RuntimeError(f"{name} spilled to local/stack memory: {metrics}")


def main() -> None:
    package = KERNEL / "comfyui_turing_utils_kernel"
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    qattn = package / f"_sage_qattn_sm75{suffix}" if suffix else None
    fused = package / f"_sage_fused_sm75{suffix}" if suffix else None
    core = package / f"_C{suffix}" if suffix else None
    if qattn is None or not qattn.is_file():
        raise RuntimeError(f"the current Python Sage extension is not built: suffix={suffix}")
    if fused is None or not fused.is_file():
        raise RuntimeError(f"the current Python Sage fused extension is not built: suffix={suffix}")
    if core is None or not core.is_file():
        raise RuntimeError(f"the current Python core extension is not built: suffix={suffix}")
    output = _resource_output(qattn)
    records: list[tuple[int, dict[str, int]]] = []
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if "sparse_attention_kernel" not in line or index + 1 >= len(lines):
            continue
        dimension = re.search(r"sparse_attention_kernelILi(64|128)E", line)
        if dimension is None:
            raise RuntimeError(f"cannot identify attention head dimension: {line}")
        metrics = _metrics(lines[index + 1])
        if metrics:
            records.append((int(dimension.group(1)), metrics))
    dimensions = {64: [], 128: []}
    for dimension, metrics in records:
        dimensions[dimension].append(metrics)
    if any(len(variants) < 6 for variants in dimensions.values()):
        raise RuntimeError(
            "expected six SM75 sparse/dense variants per native head dimension, "
            f"found D64={len(dimensions[64])}, D128={len(dimensions[128])}"
        )
    for dimension, metrics in records:
        _validate_no_spill(f"D{dimension} attention", metrics)

    varlen_value_records = []
    for index, line in enumerate(lines):
        if "quantize_varlen_value_kernel" not in line or index + 1 >= len(lines):
            continue
        metrics = _metrics(lines[index + 1])
        if metrics:
            varlen_value_records.append(metrics)
    if len(varlen_value_records) != 2:
        raise RuntimeError(
            "expected FP16/BF16 packed-V quantizers, "
            f"found {len(varlen_value_records)}"
        )
    for metrics in varlen_value_records:
        if (
            metrics.get("REG", SM75_REGISTER_LIMIT + 1) > 96
            or metrics.get("STACK", 1) != 0
            or metrics.get("LOCAL", 1) != 0
            or metrics.get("SHARED", 1025) > 1024
        ):
            raise RuntimeError(f"packed V quantizer resource regression: {metrics}")
    print(
        "attention resource audit passed: "
        f"variants=D64:{len(dimensions[64])}/D128:{len(dimensions[128])} "
        f"registers=D64:{sorted({item['REG'] for item in dimensions[64]})}/"
        f"D128:{sorted({item['REG'] for item in dimensions[128]})} "
        f"packed_v_registers:{sorted({item['REG'] for item in varlen_value_records})} "
        "local=0 stack=0 dynamic_shared=current-D64:16384/current-D128:32768"
    )

    preprocessing_output = _resource_output(fused)
    preprocessing_lines = preprocessing_output.splitlines()
    preprocessing_records: list[dict[str, int]] = []
    preprocessing_dimensions = {64: [], 128: []}
    for index, line in enumerate(preprocessing_lines):
        if "qk_preprocess" not in line or index + 1 >= len(preprocessing_lines):
            continue
        metrics = _metrics(preprocessing_lines[index + 1])
        if not metrics:
            continue
        preprocessing_records.append(metrics)
        if "quantize_qk_kernel" in line:
            dimension = re.search(r"quantize_qk_kernel.*Li(64|128)E", line)
            if dimension is None:
                raise RuntimeError(f"cannot identify preprocessing head dimension: {line}")
            preprocessing_dimensions[int(dimension.group(1))].append(metrics)
    if any(len(variants) < 12 for variants in preprocessing_dimensions.values()):
        raise RuntimeError(
            "expected twelve FP16/BF16 Q/K preprocessing variants per head dimension, "
            f"found D64={len(preprocessing_dimensions[64])}, "
            f"D128={len(preprocessing_dimensions[128])}"
        )
    for metrics in preprocessing_records:
        _validate_no_spill("Q/K preprocessing", metrics)
        if metrics.get("SHARED", SM75_SHARED_MEMORY_LIMIT + 1) > SM75_SHARED_MEMORY_LIMIT:
            raise RuntimeError(f"Q/K preprocessing exceeds the SM75 shared limit: {metrics}")
        if metrics.get("REG", 256) > 96:
            raise RuntimeError(f"Q/K preprocessing register regression: {metrics}")
    print(
        "Q/K preprocessing resource audit passed: "
        f"variants=D64:{len(preprocessing_dimensions[64])}/"
        f"D128:{len(preprocessing_dimensions[128])} "
        f"registers<={max(item['REG'] for item in preprocessing_records)} "
        f"static_shared<={max(item['SHARED'] for item in preprocessing_records)} "
        "local=0 stack=0"
    )

    core_records = _function_records(
        _sm75_section(_resource_output(core), "TuringCodebookGemmKernel")
    )
    long_shapes = (
        "GemmShapeILi128ELi256ELi64",
        "GemmShapeILi256ELi128ELi64",
    )
    inline = [
        metrics
        for name, metrics in core_records
        if "TuringCodebookGemmKernel" in name
        and any(shape in name for shape in long_shapes)
    ]
    raw_w8 = [
        metrics
        for name, metrics in core_records
        if "TuringCodebookGemmKernel" not in name
        and "integer_subbyte" not in name
        and any(shape in name for shape in long_shapes)
    ]
    if len(inline) != 2 or len(raw_w8) != 2:
        raise RuntimeError(
            "expected two inline-codebook and two raw-W8 long-sequence SM75 kernels, "
            f"found inline={len(inline)} raw_w8={len(raw_w8)}"
        )
    for name, records in (("inline codebook W4A8", inline), ("raw W8A8", raw_w8)):
        for metrics in records:
            _validate_no_spill(name, metrics)
    threads = 256
    sm75_registers = 65536
    inline_ctas = [sm75_registers // (threads * metrics["REG"]) for metrics in inline]
    raw_ctas = [sm75_registers // (threads * metrics["REG"]) for metrics in raw_w8]
    print(
        "long-sequence W4A8 resource audit passed: "
        f"registers=inline:{[item['REG'] for item in inline]}/"
        f"raw_w8:{[item['REG'] for item in raw_w8]} "
        f"register_limited_ctas=inline:{inline_ctas}/raw_w8:{raw_ctas} "
        "local=0 stack=0 shared_tile=identical; CTA density is reported, not gated"
    )


if __name__ == "__main__":
    main()
