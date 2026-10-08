"""One runtime kernel versus expanded-weight controls; not model quality validation."""
import argparse
import json
import statistics
import torch
import torch.nn.functional as F
from safetensors import safe_open
from comfy_kitchen.backends import cuda as ck
from comfyui_turing_utils_kernel.nvfp4 import linear, quantize_activation
from comfyui_turing_utils_kernel.ops import turing_nvfp4_convrot_quantize, turing_int8_linear_out


def elapsed_ms(call, warmup=3, repeats=12):
    for _ in range(warmup):
        call()
    timings = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)


def paired_timings(calls, repeats=17):
    for call in calls.values():
        for _ in range(3):
            call()
    results = {name: [] for name in calls}
    names = list(calls)
    for iteration in range(repeats):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            results[name].append(elapsed_ms(calls[name], warmup=0, repeats=1))
    return {name: statistics.median(values) for name, values in results.items()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weight-key", default="blocks.0.attn.qkv_proj.weight")
    parser.add_argument("--m", type=int, default=19989)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--swiglu", action="store_true", help="Include MLP SwiGLU in all timed paths")
    args = parser.parse_args()
    torch.manual_seed(317)
    torch.backends.cuda.matmul.allow_tf32 = False
    with safe_open(args.checkpoint, framework="pt", device="cpu") as f:
        matches = [args.weight_key] if args.weight_key in f.keys() else [
            key for key in f.keys() if key.endswith(args.weight_key)]
        if len(matches) != 1:
            raise ValueError(matches)
        w = f.get_tensor(matches[0]).to(device="cuda", dtype=torch.bfloat16)
    scale = (w.float().abs().max()/(448*6)).reshape(1)
    p, b = ck.quantize_nvfp4(w, scale)
    stored = ck.dequantize_nvfp4(p, scale, b, torch.float32)
    x = torch.randn(args.m, w.shape[1] * (2 if args.swiglu else 1),
                    device="cuda", dtype=torch.bfloat16)
    qw, sw = turing_nvfp4_convrot_quantize(p, b, scale)
    output = x.new_empty((args.m, w.shape[0]))
    wh, wb = stored.half(), stored.bfloat16()

    def activated(value=x):
        if not args.swiglu:
            return value
        gate, up = value.chunk(2, dim=-1)
        return F.silu(gate) * up

    def expanded_w8():
        qx, sx = quantize_activation(x, swiglu=args.swiglu)
        turing_int8_linear_out(qx, qw, sx, sw, output)
        return output

    def serial_nvfp4():
        # Control: identical bounded conversion/GEMM, without overlap.
        qx, sx = quantize_activation(x, swiglu=args.swiglu)
        y = x.new_empty((args.m, w.shape[0]))
        rows = min(2048, max(8, (16*1024**2//qx.shape[1])//8*8))
        if rows >= 256:
            rows = rows//256*256
        for start in range(0, w.shape[0], rows):
            end = min(start+rows, w.shape[0])
            chunk, scales = turing_nvfp4_convrot_quantize(p, b, scale, start, end-start, w.shape[1])
            turing_int8_linear_out(qx, chunk, sx, scales, y[:, start:end])
            del chunk, scales
        return y

    calls = {
        "compact_nvfp4": lambda: linear(x, p, b, scale, swiglu=args.swiglu),
        "serial_nvfp4": serial_nvfp4,
        "expanded_w8a8": expanded_w8,
        "expanded_fp16": lambda: activated().half() @ wh.T,
        "expanded_bf16": lambda: activated() @ wb.T,
        "decode_bf16": lambda: activated() @ ck.dequantize_nvfp4(p, scale, b, torch.bfloat16).T,
    }
    if args.swiglu:
        calls["unfused_activation_nvfp4"] = lambda: linear(activated(), p, b, scale)
    print(json.dumps({"shape": [args.m, *w.shape], "swiglu": args.swiglu}), flush=True)
    timings = paired_timings(calls)
    for outliers in (False, True):
        if outliers:
            x[:, ::512] *= 50
        ref = activated(x[:64].float()) @ stored.T
        original = activated(x[:64].float()) @ w.float().T
        for name, call in calls.items():
            ms = None if outliers else timings[name]
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            result = call()
            torch.cuda.synchronize()
            peak_mib = (torch.cuda.max_memory_allocated()-baseline)/1024**2
            y = result[:64].float()
            del result
            print(json.dumps({"route": name, "outliers": outliers, "ms": ms,
                "peak_increment_MiB": peak_mib,
                "relL2_stored": float((y-ref).norm()/ref.norm()),
                "relL2_original": float((y-original).norm()/original.norm())}), flush=True)
    if args.profile:
        qx, sx = quantize_activation(x, swiglu=args.swiglu)
        rows = min(2048, 16*1024**2//qx.shape[1]//256*256)
        def convert():
            for start in range(0, w.shape[0], rows):
                turing_nvfp4_convrot_quantize(p, b, scale, start, min(rows, w.shape[0]-start))
        def chunk_gemm():
            for start in range(0, w.shape[0], rows):
                end = min(start+rows, w.shape[0])
                turing_int8_linear_out(qx, qw[start:end], sx, sw[start:end], output[:, start:end])
        print(json.dumps({"profile_ms": {
            "activation": elapsed_ms(lambda: quantize_activation(x, swiglu=args.swiglu)),
            "weight_conversion": elapsed_ms(convert),
            "chunk_gemm": elapsed_ms(chunk_gemm),
            "whole_gemm": elapsed_ms(lambda: turing_int8_linear_out(qx, qw, sx, sw, output)),
        }}), flush=True)


if __name__ == "__main__":
    main()
