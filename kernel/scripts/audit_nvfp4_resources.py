"""Inspect exact SM75/SM86 cubins; never infer target speed from occupancy.

Requires cuobjdump on PATH and occupancy_model.cpp built with the owning
CUDA toolkit's cuda_occupancy.h. No GPU, model weights or network required.
Dynamic GEMM shared memory comes from the current two-stage SM75 / three-
or four-stage SM80 source schedule, not cuobjdump's static SHARED field.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

from audit_attention_resources import _arch_records, _resource_output


def describe(name, arch):
    activation = re.search(r"bf16_rowbuffer_convrot_quantize_kernelILi(512|768|1024)ELb1ELb0ELb0E", name)
    if activation:
        threads = int(activation[1])
        return {"family": "bf16_swiglu_rowbuffer", "k": 14336, "threads": threads,
                "dynamic_shared": 14336 * 2 + (threads // 64) * 2 * 256 * 4}
    group = re.search(r"nvfp4_convrot_s8_kernelILi(\d+)E", name)
    if group:
        return {"family": "nvfp4_weight", "groups_per_warp": int(group[1]),
                "threads": 256, "dynamic_shared": 0}
    if "nvfp4_convrot_s8_large_kernel" in name:
        return {"family": "nvfp4_weight_large_k", "threads": 256, "dynamic_shared": 0}
    if "GemmWithEpilogueVisitor" not in name and "DefaultGemmWithVisitor" not in name:
        return None
    if "integer_subbyte" in name or "TuringCodebookGemmKernel" in name:
        return None
    # SM80 specializations are present but not executable on SM75.
    ampere = "4Sm80" in name
    if ampere and arch == 75:
        return None
    shapes = re.findall(r"GemmShapeILi(\d+)ELi(\d+)ELi(\d+)EE", name)
    if not shapes or shapes[0] not in (("128", "256", "64"), ("128", "128", "64")):
        return None
    tile = tuple(map(int, shapes[0]))
    # Audited source schedules. Itanium names encode later GemmShape uses
    # through substitutions, so a second literal GemmShape is not the warp.
    warp = (64, 64, 64) if tile[1] == 256 else (32, 64, 64)
    threads = (tile[0] // warp[0]) * (tile[1] // warp[1]) * (tile[2] // warp[2]) * 32
    stages = 2
    if ampere:
        stage = re.search(r"GemmIdentityThreadblockSwizzleILi1EEELi([34])E", name)
        if stage is None:
            raise ValueError(f"Unrecognized Ampere stage count: {name}")
        stages = int(stage[1])
    dtype = "bf16" if "bfloat16_t" in name else "fp16" if "half_t" in name else "fp32"
    return {"family": "int8_gemm", "dtype": dtype, "stages": stages,
            "tile": list(tile), "warp_tile": list(warp), "threads": threads,
            "dynamic_shared": (tile[0] + tile[1]) * tile[2] * stages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--calculator", type=Path, required=True)
    parser.add_argument("--build-log", type=Path, help="Optional ptxas -v log to report actual spill loads/stores")
    args = parser.parse_args()
    output = _resource_output(args.binary)
    spills = {}
    if args.build_log is not None:
        log = args.build_log.read_text()
        pattern = (r"Compiling entry function '([^']+)' for 'sm_(\d+)'"
                   r".*?(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
        for symbol, arch, stack, stores, loads in re.findall(pattern, log, re.S):
            spills[(int(arch), symbol)] = {"ptxas_stack_bytes": int(stack),
                                          "spill_store_bytes": int(stores), "spill_load_bytes": int(loads)}
    for arch in (75, 86):
        found = set()
        for name, metrics in _arch_records(output, f"sm_{arch}"):
            description = describe(name, arch)
            if description is None:
                continue
            found.add(description["family"])
            occupancy = json.loads(subprocess.check_output([
                str(args.calculator.resolve()), str(arch), str(description["threads"]),
                str(metrics["REG"]), str(metrics["SHARED"]),
                str(description["dynamic_shared"]),
            ], text=True))
            symbol = name.strip().removeprefix("Function ").removesuffix(":")
            print(json.dumps({"arch": arch, **description, **metrics, **occupancy,
                              **spills.get((arch, symbol), {}), "symbol": symbol,
                              "scope": "offline theoretical ceiling; not measured occupancy or latency"}))
        if not {"int8_gemm", "nvfp4_weight", "nvfp4_weight_large_k"}.issubset(found):
            raise RuntimeError(f"Missing NVFP4 kernel families in exact sm_{arch} cubin: {found}")


if __name__ == "__main__":
    main()
