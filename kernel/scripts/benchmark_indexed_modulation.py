#!/usr/bin/env python3
"""Bounded H3 modulation-only benchmark; no model weights or CUDA graphs."""

import argparse
from pathlib import Path
import sys
import time

import torch

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT.parents[1]))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift
from comfyui_turing_utils.quantization.fusions import indexed_modulation_rows
import comfyui_turing_utils_kernel as kernel


def measure(function, iterations):
    for _ in range(5):
        function()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / iterations
    peak = (torch.cuda.max_memory_allocated() - baseline) / (1024 ** 2)
    return elapsed, peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[4096, 20000])
    parser.add_argument("--hidden", type=int, default=5376)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if min(args.rows) <= 256 or args.hidden <= 0 or args.iterations <= 0:
        parser.error("rows must exceed 256; hidden and iterations must be positive")
    torch.manual_seed(613)
    print(f"GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} kernel={kernel.__version__}")
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for rows in args.rows:
            x = torch.randn(rows, args.hidden, device="cuda", dtype=dtype)
            other = torch.randn_like(x)
            weight = 1.0 + 0.1 * torch.randn(args.hidden, device="cuda", dtype=dtype)
            scale, shift, gate = (torch.randn(9, args.hidden * 3, device="cuda", dtype=dtype) * 0.1).chunk(3, -1)
            segments = [(0, 128, 1), (128, 256, 2), (256, rows, torch.randint(0, 3, (rows - 256,), device="cuda") * 3)]

            def native():
                stream = x.clone()
                h = _mod_scale_shift(torch.nn.functional.rms_norm(stream, (args.hidden,), weight, 1e-5), shift, scale, segments)
                _mod_gate(stream, gate, other, segments)
                h = _mod_scale_shift(torch.nn.functional.rms_norm(stream, (args.hidden,), weight, 1e-5), shift, scale, segments)
                _mod_gate(stream, gate, other, segments)
                return stream, h

            def fused():
                # Include block-local packing/allocation in every iteration.
                indices = indexed_modulation_rows(segments, rows, scale.shape[0], x.device)
                stream = x.clone()
                h = kernel.turing_segmented_rms_adaln(stream, weight, scale, shift, indices, 1e-5)
                h = kernel.turing_segmented_mod_gate_rms_adaln(stream, gate, other, weight, scale, shift, indices, 1e-5)
                kernel.turing_segmented_mod_gate(stream, gate, other, indices)
                return stream, h

            reference, candidate = native(), fused()
            torch.testing.assert_close(candidate[0], reference[0], rtol=0, atol=0)
            difference = candidate[1].float() - reference[1].float()
            max_error = difference.abs().max().item()
            relative_rms = (difference.square().mean() / reference[1].float().square().mean()).sqrt().item()
            del reference, candidate, difference
            native_ms, native_peak = measure(native, args.iterations)
            fused_ms, fused_peak = measure(fused, args.iterations)
            print(f"{dtype} M={rows} K={args.hidden}: native={native_ms:.3f}ms fused={fused_ms:.3f}ms speedup={native_ms / fused_ms:.2f}x peak_extra={native_peak:.1f}/{fused_peak:.1f}MiB index={rows * 8 / 1024:.1f}KiB max_abs={max_error:.3g} relative_rms={relative_rms:.3g}", flush=True)


if __name__ == "__main__":
    main()
