"""Synthetic Veda preparation benchmark, NOT a quality-matched comparison.

Run with the owning development environment and instance-scoped CUDA/temp
settings. Does not load model files or save tensors. Random predictor weights
are intentionally not a substitute for real bundle/video validation.
"""
from pathlib import Path
import argparse
import statistics
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch
from comfyui_turing_utils.adapters.minimax.veda.engine import VedaConfig, attend
from comfyui_turing_utils.adapters.minimax.veda.predictor import PredictorBundle, convert_projection
from comfyui_turing_utils.adapters.minimax.veda.plans import TilePlan, PlanTable
from comfyui_turing_utils.adapters.minimax.veda.tiling import TileShape, TiledSpan, build_tile_layout
from comfyui_turing_utils.kernel_api import load_turing_sage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--grid", type=int, nargs=3, default=(37, 30, 18))
    parser.add_argument("--prepare-chunk-tiles", type=int, default=-1,
                        help="-1 auto, 0 whole, positive experimental tile chunk")
    parser.add_argument("--available-mib", type=int, default=0,
                        help="Synthetic budget only; does NOT emulate physical VRAM pressure")
    parser.add_argument("--mixed-shapes", action="store_true",
                        help="Interleave three tile shapes across heads (synthetic plan)")
    parser.add_argument("--score-chunk-rows", type=int, default=0,
                        help="0 budget-aware auto, positive forces route-score chunk rows")
    parser.add_argument("--projected-qkv", action="store_true",
                        help="Compare full vs tile-streamed real W8A8 GEMM + RMSNorm/RoPE")
    parser.add_argument("--projection-hidden", type=int, default=7168)
    parser.add_argument("--cache-qkv-input", action="store_true")
    parser.add_argument("--projection-chunk-tiles", type=int, default=16)
    args = parser.parse_args()
    if args.available_mib:
        import comfyui_turing_utils.adapters.minimax.veda.engine as engine
        engine.runtime_memory = lambda device: (args.available_mib * 1024**2, 0, 0)
    torch.manual_seed(123)
    h, d = args.heads, 128
    t, y, x = args.grid
    s = 64 + t * y * x
    q, k, v = [torch.randn(1, h, s, d, device="cuda", dtype=torch.float16) for _ in range(3)]
    weights = [torch.randn(h, 384, d) * .01 for _ in range(2)]
    shapes = ([TileShape(4, 4, 8), TileShape(2, 8, 8), TileShape(1, 8, 16)]
              if args.mixed_shapes else [TileShape(4, 4, 8)])
    plan = TilePlan("benchmark", tuple(args.grid), shapes, [[i % len(shapes) for i in range(h)]])
    host_layout = build_tile_layout([TiledSpan(64, tuple(args.grid), plan.shapes[0])], s)
    layout = SimpleNamespace(seq_len=s, signature=(64, t, y*2, x*2, 0),
                             segments=[(0, 64, "text"), (64, s, "video")])
    sage = load_turing_sage()

    def measure(name, function):
        for _ in range(3):
            function()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = function()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
            del result
        peak = (torch.cuda.max_memory_allocated() - base) / 1024**2
        print(f"{name}: median={statistics.median(times):.3f} ms "
              f"peak_extra={peak:.1f} MiB", flush=True)

    print(torch.cuda.get_device_name(), "SM", torch.cuda.get_device_capability(), flush=True)
    print("Veda first-shape physical query blocks/head:", host_layout.n_tiles * 2,
          "nonempty:", int(((host_layout.valid_count + 63) // 64).sum()), flush=True)
    if args.projected_qkv:
        from comfyui_turing_utils.adapters.minimax import acceleration as a
        from comfyui_turing_utils.attention.protocol import QKTransformSpec, RMSNormSpec, RotaryEmbeddingSpec
        from comfy.ldm.minimax.model import rope_rotation_table
        del q, k, v
        x = torch.randn(s, args.projection_hidden, device="cuda", dtype=torch.bfloat16)
        attention = SimpleNamespace(heads=h, head_dim=d, qkv_proj=SimpleNamespace(pre_quant_scale=None),
            q_norm=torch.nn.RMSNorm(d, eps=1e-6).cuda().bfloat16().requires_grad_(False),
            k_norm=torch.nn.RMSNorm(d, eps=1e-6).cuda().bfloat16().requires_grad_(False))
        qw = torch.randint(-16, 16, (3*h*d, x.shape[1]), device="cuda", dtype=torch.int8)
        ws = torch.full((3*h*d,), .002, device="cuda")
        freqs = rope_rotation_table(torch.randn(s, d, device="cuda"), x.dtype)
        transform = QKTransformSpec(RMSNormSpec(attention.q_norm.weight, 1e-6, "head"),
            RMSNormSpec(attention.k_norm.weight, 1e-6, "head"), RotaryEmbeddingSpec(freqs,d,"split_half"))
        pq, pk = [convert_projection(w, "w8a8") for w in weights]
        bundle = PredictorBundle("w8a8", 1, h, d, .1, PlanTable([plan]), (pq,), (pk,))
        config, cache = VedaConfig(bundle), {}
        options = {"turing_utils_veda": config, "minimax_h3_layout": layout,
                   "turing_utils_attention_layout": {"layer_index": 0, "layer_count": 1},
                   "turing_utils_veda_forward_cache": cache,
                   "turing_utils_veda_projection_chunk_tiles": args.projection_chunk_tiles}
        quantized = a._cache_quantized_qkv_input(attention.qkv_proj,x,16384) if args.cache_qkv_input else None
        def full():
            q,k,v = a._project_qkv_head_group(attention,x,qw,ws,None,0,h,16384,quantized)
            q,k = a._apply_minimax_qk_transform(attention,q,k,freqs)
            return attend(*(t.transpose(0,1).unsqueeze(0) for t in (q,k,v)),
                          config=config,packed_layout=layout,layer=0,cache=cache)
        measure("W8 QKV + full Veda", full)
        measure("W8 tile-projected compact Veda", lambda: a._veda_projected_head_group(
            attention,x,transform,qw,ws,None,0,h,quantized,options))
        return
    measure("dense Sage", lambda: sage.sageattn(q, k, v))
    measure("SOL threshold=1 FP16-PV", lambda: sage.sol_sparse_sageattn(
        q, k, v, threshold_sigma=1., use_w8a8=False))
    for precision in ("w8a8", "fp16", "bf16", "fp32"):
        pq, pk = [convert_projection(w, precision) for w in weights]
        bundle = PredictorBundle(precision, 1, h, d, .1, PlanTable([plan]), (pq,), (pk,))
        config, cache = VedaConfig(bundle), {}
        measure(f"Veda {precision} keep=.1 warm layout", lambda: attend(
            q, k, v, config=config, packed_layout=layout, layer=0, cache=cache,
            prepare_chunk_tiles=args.prepare_chunk_tiles, score_chunk_rows=args.score_chunk_rows))
        cache.clear()


if __name__ == "__main__":
    main()
