"""Real H3 block weights, synthetic tokens; not video-quality or offload validation.

Run from the owning ComfyUI root with its installed kernel. Expanded weights
are benchmark controls only, never a proposed runtime cache. Uses core dense
attention without RoPE/SOL; therefore timings are NOT a full pipeline estimate.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from comfy_kitchen.backends import cuda as ck
from comfyui_turing_utils_kernel.nvfp4 import linear, quantize_activation
from comfyui_turing_utils_kernel.ops import (
    turing_nvfp4_convrot_quantize, turing_int8_linear_out,
)
from benchmark_nvfp4_routes import paired_timings


class Projection(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.scale = weight.float().abs().amax() / (448 * 6)
        self.packed, self.blocks = ck.quantize_nvfp4(weight, self.scale)
        self.dense = ck.dequantize_nvfp4(self.packed, self.scale, self.blocks, torch.bfloat16)
        self.qweight, self.wscale = turing_nvfp4_convrot_quantize(
            self.packed, self.blocks, self.scale)
        self.route = "compact"

    def forward(self, x, swiglu=False):
        if self.route == "compact":
            return linear(x, self.packed, self.blocks, self.scale, swiglu=swiglu)
        if self.route == "expanded_w8a8":
            qx, sx = quantize_activation(x, swiglu=swiglu)
            out = x.new_empty((x.shape[0], self.dense.shape[0]))
            turing_int8_linear_out(qx, self.qweight, sx, self.wscale, out)
            return out
        if swiglu:
            gate, up = x.chunk(2, dim=-1)
            x = F.silu(gate) * up
        if self.route == "unfused":
            return linear(x, self.packed, self.blocks, self.scale)
        return F.linear(x, self.dense)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--block", default="blocks.0")
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    if args.m <= 0 or args.repeats <= 0:
        parser.error("m and repeats must be positive")
    # ComfyUI's argument parser must not see this benchmark's arguments.
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(Path.cwd()))
    import comfy.ops
    from comfy.ldm.minimax.model import DiTBlock

    torch.manual_seed(419)
    with safe_open(args.checkpoint, framework="pt") as f:
        prefix = args.block + "."
        state = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
    if not state:
        raise ValueError(f"No weights for {args.block}")
    hidden = state["norm1.weight"].numel()
    head_dim = state["attn.q_norm.weight"].numel()
    heads = state["attn.qkv_proj.weight"].shape[0] // (3 * head_dim)
    ffn = state["mlp.fc2.weight"].shape[1]
    t_dim = state["adaln_proj.linear.weight"].shape[1]
    block = DiTBlock(hidden, heads, head_dim, ffn, t_dim, 1e-6, 1e-6,
                     dtype=torch.bfloat16, device="cpu", operations=comfy.ops.manual_cast)
    block.load_state_dict(state, strict=True)
    del state
    block = block.cuda().eval()
    projections = []
    for owner, name in ((block.attn, "qkv_proj"), (block.attn, "out_proj"),
                        (block.mlp, "fc1"), (block.mlp, "fc2")):
        projection = Projection(getattr(owner, name).weight)
        setattr(owner, name, projection)
        projections.append(projection)
    block.mlp.forward = types.MethodType(lambda self, x: self.fc2(self.fc1(x), swiglu=True), block.mlp)
    x = torch.randn(args.m, hidden, device="cuda", dtype=torch.bfloat16)
    t = torch.randn(1, t_dim, device="cuda", dtype=torch.bfloat16) * .1

    def run(route):
        for projection in projections:
            projection.route = route
        # Core block modifies the residual in place: never feed the previous
        # benchmark output back as the next benchmark input.
        return block(x.clone(), t, [(0, args.m, 0)], None)

    calls = {name: lambda name=name: run(name) for name in
             ("compact", "unfused", "expanded_w8a8", "expanded_bf16")}
    ref = run("expanded_bf16")
    if not bool(ref.isfinite().all()):
        raise RuntimeError("Non-finite reference block output")
    times = paired_timings(calls, args.repeats)
    print(json.dumps({"device": torch.cuda.get_device_name(), "block": args.block,
                      "tokens": args.m, "hidden": hidden, "ffn": ffn,
                      "scope": "real block weights, synthetic tokens, no RoPE, core dense attention",
                      "controls_resident": True}), flush=True)
    for name, call in calls.items():
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        y = call()
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - base) / 1024**2
        if not bool(y.isfinite().all()):
            raise RuntimeError(f"Non-finite output: {name}")
        error = float((y.float() - ref.float()).norm() / ref.float().norm())
        print(json.dumps({"route": name, "ms": times[name],
                          "incremental_peak_MiB": peak, "relL2_decoded_nvfp4_block": error}), flush=True)
        del y


if __name__ == "__main__":
    main()
