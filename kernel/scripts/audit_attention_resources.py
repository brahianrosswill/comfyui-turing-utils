#!/usr/bin/env python3
"""Audit static SM75 attention resources from the built extension."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sysconfig


KERNEL = Path(__file__).resolve().parents[1]


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


def main() -> None:
    package = KERNEL / "comfyui_turing_utils_kernel"
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    extensions = [package / f"_sage_qattn_sm75{suffix}"] if suffix else []
    extensions = [path for path in extensions if path.is_file()]
    if len(extensions) != 1:
        raise RuntimeError(f"the current Python Sage extension is not built: suffix={suffix}")
    output = subprocess.check_output(
        [_cuobjdump(), "--dump-resource-usage", str(extensions[0])],
        text=True,
        stderr=subprocess.STDOUT,
    )
    records: list[tuple[int, dict[str, int]]] = []
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if "sparse_attention_kernel" not in line or index + 1 >= len(lines):
            continue
        dimension = re.search(r"sparse_attention_kernelILi(64|128)E", line)
        if dimension is None:
            raise RuntimeError(f"cannot identify attention head dimension: {line}")
        metrics = {
            name: int(value)
            for name, value in re.findall(
                r"\b(REG|STACK|SHARED|LOCAL):(\d+)", lines[index + 1]
            )
        }
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
    for _, metrics in records:
        if metrics.get("REG", 256) > 255:
            raise RuntimeError(f"SM75 register limit exceeded: {metrics}")
        if metrics.get("LOCAL", 1) != 0:
            raise RuntimeError(f"SM75 attention spilled to local memory: {metrics}")
    for metrics in dimensions[64]:
        if metrics.get("REG", 256) > 192 or metrics.get("STACK", 1) != 0:
            raise RuntimeError(f"native D64 resource regression: {metrics}")
    print(
        "attention resource audit passed: "
        f"variants=D64:{len(dimensions[64])}/D128:{len(dimensions[128])} "
        f"registers=D64:{sorted({item['REG'] for item in dimensions[64]})}/"
        f"D128:{sorted({item['REG'] for item in dimensions[128]})} "
        "local=0 dynamic_shared=D64:16384/D128:32768(source-gated)"
    )


if __name__ == "__main__":
    main()
