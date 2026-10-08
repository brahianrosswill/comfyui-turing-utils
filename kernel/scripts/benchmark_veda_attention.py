"""Fixed-route synthetic Veda comparison; not a model quality benchmark.

Run from the owning dev instance with its environment/cache settings.
"""
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from comfyui_turing_utils.adapters.minimax.veda.engine import pack_routes, run_sparse_packed
from comfyui_turing_utils.adapters.minimax.veda.quantization import prequantize, finish_value
from comfyui_turing_utils.adapters.minimax.veda.selection import select_tiles
from comfyui_turing_utils.adapters.minimax.veda.tiling import TileShape, TiledSpan, build_tile_layout


def measure(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(20):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(samples), 3)


def main():
    torch.manual_seed(71)
    torch.set_num_threads(4)
    for rows in (4096, 19968):
        layout = build_tile_layout([TiledSpan(0, (1, 1, rows), TileShape(1, 1, 128))], rows, "cuda")
        q, k, v = [torch.randn(1, 14, rows, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        scores = torch.randn(14, layout.n_video_tiles, layout.n_video_tiles, device="cuda")
        for keep in (.1, .3):
            index, valid = select_tiles(scores, layout, keep, 1.)
            routes = pack_routes(index, valid, layout)
            for rotated in (False, True):
                def prepare():
                    return finish_value(prequantize(q, k, v, use_w8a8=rotated))
                packed = prepare()
                def attention():
                    return run_sparse_packed(packed, layout, routes, v.dtype)
                def complete():
                    return run_sparse_packed(prepare(), layout, routes, v.dtype)
                print(dict(rows=rows, heads=14, keep=keep, rotated_int8_pv=rotated,
                           preparation_ms=measure(prepare), attention_ms=measure(attention),
                           combined_ms=measure(complete)), flush=True)


if __name__ == "__main__":
    main()
