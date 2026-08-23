#!/usr/bin/env python3
"""Audit exact SM75/SM86 kernel resources without prescribing occupancy."""

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


def _arch_records(output: str, architecture: str) -> list[tuple[str, dict[str, int]]]:
    sections = [
        section
        for section in output.split("Fatbin elf code:")
        if re.search(rf"\barch\s*=\s*{re.escape(architecture)}\b", section)
    ]
    if not sections:
        raise RuntimeError(
            f"built extension does not contain an exact {architecture} cubin"
        )
    return [record for section in sections for record in _function_records(section)]


def _sm75_records(output: str) -> list[tuple[str, dict[str, int]]]:
    return _arch_records(output, "sm_75")


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
    w4a8_source = (KERNEL / "csrc" / "turing" / "w4a8.cu").read_text(
        encoding="utf-8"
    )
    if "__dp4a" in w4a8_source or "w4a8_compatibility_kernel" in w4a8_source:
        raise RuntimeError(
            "legacy W4A8 edge shapes must stay on predicated/padded Tensor Core paths"
        )
    if "run_k_tail_tile" not in w4a8_source or "tail_weight" not in (
        KERNEL / "csrc" / "bindings.cpp"
    ).read_text(encoding="utf-8"):
        raise RuntimeError("W4A8 K/N Tensor Core edge handling is missing")

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
    qattn_sm75_records = _sm75_records(output)
    for line, metrics in qattn_sm75_records:
        if "sparse_attention_kernel" not in line:
            continue
        dimension = re.search(r"sparse_attention_kernelILi(64|128)E", line)
        if dimension is None:
            raise RuntimeError(f"cannot identify attention head dimension: {line}")
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
    for line, metrics in qattn_sm75_records:
        if "quantize_varlen_value_kernel" in line:
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
    preprocessing_records: list[dict[str, int]] = []
    preprocessing_dimensions = {64: [], 128: []}
    for line, metrics in _sm75_records(preprocessing_output):
        if "qk_preprocess" not in line:
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

    core_output = _resource_output(core)
    core_records = _sm75_records(core_output)
    if len(core_records) < 50:
        raise RuntimeError(
            "expected the complete SM75 core operator family, "
            f"found only {len(core_records)} kernels"
        )
    for name, metrics in core_records:
        _validate_no_spill(name, metrics)
        if metrics.get("SHARED", SM75_SHARED_MEMORY_LIMIT + 1) > SM75_SHARED_MEMORY_LIMIT:
            raise RuntimeError(f"{name} exceeds the SM75 static shared limit: {metrics}")

    expected_families = (
        ("BF16 epilogue", "dequantize_int8_bf16", 4, 64),
        ("ConvRot row-buffer", "bf16_rowbuffer_convrot_quantize_kernel", 18, 64),
        ("ConvRot staged reduction", "swiglu_rotate_amax_kernel", 4, 64),
        (
            "scaled SwiGLU shard quantizer",
            "swiglu_rotate_scaled_quantize_kernel",
            2,
            64,
        ),
        ("ConvRot staged INT8 output", "quantize_from_partials_kernel", 2, 64),
        ("ConvRot staged INT4 output", "quantize_int4_from_partials_kernel", 2, 64),
        ("segmented RMSNorm+AdaLN", "segmented_rms_adaln_kernel", 3, 96),
        ("LayerNorm+AdaLN", "layer_norm_adaln_kernel", 3, 96),
        ("codebook W4 decoder", "decode_codebook_w4_to_s8", 1, 64),
    )
    family_summary = []
    for label, marker, expected, register_limit in expected_families:
        family = [metrics for name, metrics in core_records if marker in name]
        if len(family) != expected:
            raise RuntimeError(
                f"expected {expected} {label} SM75 kernels, found {len(family)}"
            )
        maximum = max(metrics["REG"] for metrics in family)
        if maximum > register_limit:
            raise RuntimeError(
                f"{label} register regression: maximum={maximum}, limit={register_limit}"
            )
        family_summary.append(f"{label}:{len(family)}@r{maximum}")

    print(
        "core operator resource audit passed: "
        + " ".join(family_summary)
        + f" total={len(core_records)} local=0 stack=0"
    )

    w4_section_records = _function_records(
        _sm75_section(core_output, "TuringCodebookGemmKernel")
    )
    long_shapes = (
        "GemmShapeILi128ELi256ELi64",
        "GemmShapeILi256ELi128ELi64",
    )
    inline = [
        metrics
        for name, metrics in w4_section_records
        if "TuringCodebookGemmKernel" in name
        and any(shape in name for shape in long_shapes)
    ]
    raw_w8 = [
        metrics
        for name, metrics in w4_section_records
        if "TuringCodebookGemmKernel" not in name
        and "DefaultGemmWithVisitor" not in name
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

    # CUTLASS emits the SM80-specialized templates into every requested fatbin,
    # but runtime dispatch can reach them only on SM80+. Audit the exact SM86
    # cubin because that is the production Ampere schedule used by auto-tuning.
    ampere = [
        (name, metrics)
        for name, metrics in _arch_records(core_output, "sm_86")
        if "DefaultGemmWithVisitor" in name and "4Sm80" in name
    ]
    if len(ampere) != 11:
        raise RuntimeError(
            "expected eleven CUTLASS SM80 W8A8 schedules in the exact sm86 cubin, "
            f"found {len(ampere)}"
        )
    ampere_summary = []
    for name, metrics in ampere:
        _validate_no_spill("CUTLASS SM80 W8A8", metrics)
        shape = re.search(r"GemmShapeILi(\d+)ELi(\d+)ELi(\d+)", name)
        if shape is None:
            raise RuntimeError(f"cannot identify CUTLASS SM80 tile shape: {name}")
        ampere_summary.append(
            f"{shape.group(1)}x{shape.group(2)}x{shape.group(3)}@r{metrics['REG']}"
        )
    print(
        "Ampere CUTLASS W8A8 resource audit passed: "
        + " ".join(sorted(ampere_summary))
        + " local=0 stack=0"
    )


if __name__ == "__main__":
    main()
