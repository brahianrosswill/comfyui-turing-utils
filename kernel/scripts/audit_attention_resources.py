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
    records = []
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if "sparse_attention_kernel" not in line or index + 1 >= len(lines):
            continue
        metrics = {
            name: int(value)
            for name, value in re.findall(
                r"\b(REG|STACK|SHARED|LOCAL):(\d+)", lines[index + 1]
            )
        }
        if metrics:
            records.append(metrics)
    if len(records) < 6:
        raise RuntimeError(f"expected six SM75 sparse/dense variants, found {len(records)}")
    for metrics in records:
        if metrics.get("REG", 256) > 255:
            raise RuntimeError(f"SM75 register limit exceeded: {metrics}")
        if metrics.get("LOCAL", 1) != 0:
            raise RuntimeError(f"SM75 attention spilled to local memory: {metrics}")
    print(
        "attention resource audit passed: "
        f"variants={len(records)} registers={sorted({item['REG'] for item in records})} "
        "local=0 dynamic_shared=32768(source-gated)"
    )


if __name__ == "__main__":
    main()
