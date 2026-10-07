"""Opt-in GPU checks: VEDA_TEST_CUDA=1 ops/test-dev.sh -k veda_cuda."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from comfyui_turing_utils.kernel_api import load_kernel_extension, load_turing_sage
from comfyui_turing_utils.adapters.minimax.veda.engine import VedaConfig, attend, pack_routes, run_sparse_tiles
from comfyui_turing_utils.adapters.minimax.veda.predictor import PredictorBundle, convert_projection, project_features
from comfyui_turing_utils.adapters.minimax.veda.plans import PlanTable, TilePlan
from comfyui_turing_utils.adapters.minimax.veda.selection import select_tiles
from comfyui_turing_utils.adapters.minimax.veda.tiling import (
    TileShape, TiledSpan, build_tile_layout, gather_tiles,
)


pytestmark = pytest.mark.skipif(os.environ.get("VEDA_TEST_CUDA") != "1", reason="opt-in GPU check")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_veda_cuda_fused_gather_pool(dtype):
    from comfyui_turing_utils.adapters.minimax.veda.pooling import pool_video_tiles
    device = torch.device("cuda", 0)
    layout = build_tile_layout([TiledSpan(3, (1, 7, 19), TileShape(1, 8, 16))], 143, device)
    heads = torch.tensor([3, 1], device=device)
    # Non-contiguous head-major inputs, partial tiles, and global tokens.
    source = [torch.randn(4, 143, 128, device=device, dtype=dtype).transpose(0, 1)
              for _ in range(3)]
    actual = load_kernel_extension("_sage_qattn_sm75").veda_gather_pool(
        *source, layout.gather_index, heads, layout.valid_count, layout.n_video_tiles)
    expected = [gather_tiles(x, layout, heads) for x in source]
    for a, b in zip(actual[:3], expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert a.transpose(0, 1).is_contiguous()
    for a, b in zip(actual[3:], expected[:2]):
        torch.testing.assert_close(a, pool_video_tiles(b, layout), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_veda_cuda_native_scatter_preserves_unselected_heads(dtype):
    from comfyui_turing_utils.adapters.minimax.veda.tiling import scatter_tiles_
    device = torch.device("cuda", 0)
    layout = build_tile_layout([TiledSpan(3, (1, 7, 19), TileShape(1, 8, 16))], 143, device)
    heads = torch.tensor([3, 1], device=device)
    value = torch.randn(2, layout.num_slots, 128, device=device, dtype=dtype).transpose(0, 1)
    expected = torch.randn(144, 4, 128, device=device, dtype=dtype)
    actual = expected.clone()
    scatter_tiles_(expected, value, layout, heads)
    load_kernel_extension("_sage_qattn_sm75").veda_scatter_tiles(
        actual, value, layout.scatter_index, heads, layout.valid_count)
    torch.testing.assert_close(actual[:143], expected[:143], atol=0, rtol=0)


@pytest.mark.parametrize("ratios", [(0.1, 0.2), (0.5234, 1.0), (1.0, 0.05), (1.0, 1.0)])
def test_veda_cuda_fused_selection_matches_reference(monkeypatch, ratios):
    from comfyui_turing_utils.adapters.minimax.veda import selection
    device = torch.device("cuda", 0)
    layout = build_tile_layout([
        TiledSpan(0, (2, 9, 17), TileShape(2, 4, 16)),
        TiledSpan(306, (7, 13, 19), TileShape(4, 4, 8)),
    ], 2043, device)
    # Include a deliberately empty tile and many exact ties.
    layout.valid_count[2] = 0
    layout.kv_ok[2] = False
    scores = torch.randn(3, layout.n_video_tiles, layout.n_video_tiles, device=device).round()
    actual = [selection.select_tiles(scores[:, a:b].contiguous(), layout, *ratios, row_start=a)
              for a, b in ((0, 7), (7, layout.n_video_tiles))]
    # Explicitly select the reference evaluator, not an obsolete kernel ABI.
    monkeypatch.setattr(selection, "load_kernel_extension", lambda name: None)
    expected = [selection.select_tiles(scores[:, a:b].contiguous(), layout, *ratios, row_start=a)
                for a, b in ((0, 7), (7, layout.n_video_tiles))]
    for pair_a, pair_b in zip(actual, expected):
        for a, b in zip(pair_a, pair_b):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_veda_cuda_batched_projection_matches_individual_gemms():
    native = load_kernel_extension("_sage_qattn_sm75")
    ops = load_kernel_extension("ops")
    heads, rows = 3, 157
    x = torch.randn(heads, rows, 512, device="cuda", dtype=torch.bfloat16)
    qi, scale = ops.turing_bf16_int8_convrot_quantize(x.flatten(0, 1))
    qi, scale = qi.reshape(heads, rows, 512), scale.reshape(heads, rows, 1)
    p = convert_projection(torch.randn(heads, 384, 128), "w8a8").stage(list(range(heads)), x.device)
    actual = native.veda_projection_int8(qi, p.weight, scale, p.scale, x)
    expected = torch.stack([ops.turing_int8_linear(qi[h], p.weight[h], scale[h], p.scale[h])
                            for h in range(heads)]) + x[..., :128]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_veda_cuda_fused_routes_match_reference():
    device = torch.device("cuda", 0)
    layout = build_tile_layout([TiledSpan(3, (1, 7, 19), TileShape(1, 8, 16))], 143, device)
    scores = torch.randn(3, layout.n_video_tiles, layout.n_video_tiles, device=device)
    index, keep = select_tiles(scores, layout, 0.2, 1.0)
    actual = pack_routes(index, keep, layout)
    half = torch.arange(2, device=device)
    blocks = (index[..., None] * 2 + half).flatten(-2)
    valid = (keep[..., None] & (layout.valid_count[index][..., None] > half * 64)).flatten(-2)
    blocks = blocks.masked_fill(~valid, -1).to(torch.int32)
    exact = torch.zeros(layout.n_tiles * 2, dtype=torch.uint8, device=device)
    exact[layout.n_video_tiles * 2:] = 1
    exact &= (layout.valid_count[:, None] > half * 64).flatten().to(torch.uint8)
    expected = load_kernel_extension("_sage_qattn_sm75").sla_build_route_words(
        blocks.unsqueeze(0).contiguous(), exact, layout.n_tiles * 2)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_veda_cuda_compact_preparation_matches_whole(dtype):
    from comfyui_turing_utils.adapters.minimax.veda.prepare import prepare_compact
    from comfyui_turing_utils.adapters.minimax.veda.pooling import pool_video_tiles
    device = torch.device("cuda", 0)
    layout = build_tile_layout([TiledSpan(3, (3, 7, 19), TileShape(1, 8, 16))], 413, device)
    heads = torch.tensor([1, 0], device=device)
    source = [torch.randn(413, 2, 128, device=device, dtype=dtype) for _ in range(3)]
    packed, fq, fk = prepare_compact(*source, layout, heads, chunk_tiles=2)
    host_layout = build_tile_layout([TiledSpan(3, (3, 7, 19), TileShape(1, 8, 16))], 413)
    visited = []
    def projector(indices, head_list):
        visited.extend(indices.cpu().tolist())
        return tuple(t.index_select(0, indices)[:, head_list] for t in source)
    projected, pfq, pfk = prepare_compact(
        *source, layout, heads, chunk_tiles=2, projector=projector,
        host_layout=host_layout, head_list=[1, 0])
    assert sorted(visited) == list(range(413))  # No padded or repeated GEMM rows.
    for field in ("query_int8", "key_int8", "query_scale", "key_scale", "value"):
        torch.testing.assert_close(getattr(projected, field), getattr(packed, field), atol=0, rtol=0)
    torch.testing.assert_close(pfq, fq, atol=0, rtol=0)
    torch.testing.assert_close(pfk, fk, atol=0, rtol=0)
    gathered = [gather_tiles(x, layout, heads) for x in source]
    attention_dtype = torch.bfloat16 if dtype == torch.float32 else dtype
    expected = load_turing_sage().prequantize_sageattn(*[
        x.transpose(0, 1).unsqueeze(0).to(attention_dtype) for x in gathered])
    for field in ("query_int8", "key_int8", "query_scale", "key_scale", "value"):
        torch.testing.assert_close(getattr(packed, field), getattr(expected, field), atol=0, rtol=0)
    for actual, tiled in zip((fq, fk), gathered[:2]):
        torch.testing.assert_close(actual, pool_video_tiles(tiled, layout), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("amplitude", [1., 0.001, 0.])
def test_veda_cuda_partial_tiles_match_quantized_sdpa(dtype, amplitude):
    torch.manual_seed(71)
    device = torch.device("cuda", 0)
    layout = build_tile_layout([TiledSpan(3, (1, 7, 19), TileShape(1, 8, 16))], 143, device)
    heads = torch.arange(2, device=device)
    source = [torch.randn(143, 2, 128, device=device, dtype=dtype) for _ in range(3)]
    source[0] *= amplitude
    source[1] *= amplitude
    q, k, v = [gather_tiles(x, layout, heads) for x in source]
    scores = torch.randn(2, layout.n_video_tiles, layout.n_video_tiles, device=device)
    index, keep = select_tiles(scores, layout, 0.5, 1.0)
    routes = torch.zeros((1, 2, layout.n_tiles, (layout.n_tiles * 2 + 31) // 32),
                         device=device, dtype=torch.int32)
    routes[:, :, :layout.n_video_tiles] = pack_routes(index, keep, layout)
    actual = run_sparse_tiles(q, k, v, layout, routes)
    packed = load_turing_sage().prequantize_sageattn(*[
        x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)
    ])
    qref = packed.query_int8.float() * packed.query_scale.repeat_interleave(16, -1)[..., None]
    kref = packed.key_int8.float() * packed.key_scale.repeat_interleave(64, -1)[..., None]
    pos = torch.arange(layout.num_slots, device=device)
    word = routes[:, :, pos // 128, :][..., (pos // 64) // 32]
    mask = ((word >> ((pos // 64) % 32)) & 1).bool()
    mask[:, :, layout.n_video_tiles * 128:] = True
    mask &= layout.slot_valid[None, None, None, :].bool()
    expected = F.scaled_dot_product_attention(qref, kref, packed.value.float(), attn_mask=mask)
    expected = expected[0].transpose(0, 1)
    real = layout.slot_valid.bool()
    assert actual.dtype == dtype
    assert actual.isfinite().all()
    torch.testing.assert_close(actual[real].float(), expected[real], atol=0.008, rtol=0.02)
    pos = torch.arange(layout.num_slots, device=device)
    unused_cta = ((pos % 128) // 64) * 64 >= layout.valid_count[pos // 128]
    assert torch.count_nonzero(actual[unused_cta]) == 0


def test_veda_cuda_auto_score_rows_matches_128_row_baseline():
    torch.manual_seed(316)
    heads, dim, grid = 2, 128, (37, 30, 18)
    seq_len = 64 + 37 * 30 * 18
    projections = [convert_projection(torch.randn(heads, 384, dim) * 0.03, "w8a8")
                   for _ in range(2)]
    plan = TilePlan("test", grid, [TileShape(4, 4, 8)], [[0, 0]])
    bundle = PredictorBundle("w8a8", 1, heads, dim, .1, PlanTable([plan]),
                             (projections[0],), (projections[1],))
    config = VedaConfig(bundle)
    layout = SimpleNamespace(seq_len=seq_len, signature=(64, 37, 60, 36, 0),
                             segments=[(0, 64, "text"), (64, seq_len, "video")])
    q, k, v = [torch.randn(1, heads, seq_len, dim, device="cuda", dtype=torch.float16)
               for _ in range(3)]
    baseline = attend(q, k, v, config=config, packed_layout=layout, layer=0,
                      cache={}, score_chunk_rows=128)
    wider = attend(q, k, v, config=config, packed_layout=layout, layer=0,
                   cache={}, score_chunk_rows=256)
    automatic = attend(q, k, v, config=config, packed_layout=layout, layer=0, cache={})
    torch.testing.assert_close(wider, baseline, atol=0.002, rtol=0.002)
    torch.testing.assert_close(automatic, baseline, atol=0.002, rtol=0.002)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("scalar_scale", [False, True])
@pytest.mark.parametrize("head_start", [0, 1])
def test_veda_cuda_real_projected_qkv_matches_full_rope(cached, scalar_scale, head_start):
    from comfyui_turing_utils.adapters.minimax import acceleration as a
    from comfyui_turing_utils.attention.protocol import QKTransformSpec, RMSNormSpec, RotaryEmbeddingSpec
    from comfy.ldm.minimax.model import rope_rotation_table

    torch.manual_seed(319)
    heads, dim, seq, hidden = 4, 128, 143, 512
    x = torch.randn(seq, hidden, device="cuda", dtype=torch.bfloat16)
    attention = SimpleNamespace(heads=heads, head_dim=dim,
        qkv_proj=SimpleNamespace(pre_quant_scale=None),
        q_norm=torch.nn.RMSNorm(dim, eps=1e-6).cuda().bfloat16().requires_grad_(False),
        k_norm=torch.nn.RMSNorm(dim, eps=1e-6).cuda().bfloat16().requires_grad_(False))
    qw = torch.randint(-16, 16, (3*heads*dim, hidden), device="cuda", dtype=torch.int8)
    ws = torch.full(() if scalar_scale else (3*heads*dim,), .005, device="cuda")
    bias = torch.randn(3*heads*dim, device="cuda", dtype=x.dtype) * .01
    freqs = rope_rotation_table(torch.randn(seq, dim, device="cuda"), x.dtype)
    transform = QKTransformSpec(RMSNormSpec(attention.q_norm.weight, 1e-6, "head"),
        RMSNormSpec(attention.k_norm.weight, 1e-6, "head"), RotaryEmbeddingSpec(freqs, dim, "split_half"))
    projections = [convert_projection(torch.randn(heads, 384, dim) * .03, "w8a8") for _ in range(2)]
    plan = TilePlan("test", (1, 7, 19), [TileShape(1, 8, 16), TileShape(1, 1, 128)], [[0, 1, 0, 1]])
    bundle = PredictorBundle("w8a8", 1, heads, dim, .1, PlanTable([plan]),
                             (projections[0],), (projections[1],))
    layout = SimpleNamespace(seq_len=seq, signature=(3, 1, 14, 38, 7),
                             segments=[(0, 3, "text"), (3, 10, "audio"), (10, seq, "video")])
    options = {"turing_utils_veda": VedaConfig(bundle), "minimax_h3_layout": layout,
               "turing_utils_attention_layout": {"layer_index": 0, "layer_count": 1}}
    quantized = a._cache_quantized_qkv_input(attention.qkv_proj, x, 64) if cached else None
    q, k, v = a._project_qkv_head_group(attention, x, qw, ws, bias, head_start, heads, 64, quantized)
    q, k = a._apply_minimax_qk_transform(attention, q, k, freqs)
    expected = attend(*(t.transpose(0, 1).unsqueeze(0) for t in (q, k, v)),
                      config=options["turing_utils_veda"], packed_layout=layout, layer=0,
                      cache={}, head_start=head_start)
    actual = a._veda_projected_head_group(attention, x, transform, qw, ws, bias,
                                          head_start, heads, quantized, options)
    torch.testing.assert_close(actual, expected.transpose(1, 2).flatten(2)[0], atol=.004, rtol=.004)


def test_veda_cuda_w8a8_projection_uses_bf16_result():
    torch.manual_seed(94)
    weight = torch.randn(2, 384, 128) * 0.03
    features = torch.randn(2, 157, 384, device="cuda", dtype=torch.float32)
    projection = convert_projection(weight, "w8a8").stage([0, 1], features.device)
    actual = project_features(features, projection, "w8a8")
    expected = torch.bmm(features, weight.to(features.device)) + features[..., :128]
    assert actual.dtype == torch.bfloat16
    assert (actual.float() - expected).norm() / expected.norm() < 0.025


@pytest.mark.parametrize("precision", ["w8a8", "bf16", "fp16", "fp32"])
def test_veda_cuda_projection_transfer_preserves_noncontiguous_heads(precision):
    from comfyui_turing_utils.adapters.minimax.veda.predictor import ProjectionTransfer

    torch.manual_seed(314)
    projections = [convert_projection(torch.randn(4, 384, 128), precision) for _ in range(2)]
    heads = [3, 0, 2]
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=device)
    transfer = ProjectionTransfer(projections, heads, device, stream)
    actual = transfer.consume(device)
    assert all(t.is_pinned() for pair in transfer.sources for t in pair if t is not None)
    for expected, staged in zip(projections, actual):
        for source, target in zip(expected.select_heads(heads), (staged.weight, staged.scale)):
            if source is None:
                assert target is None
            else:
                torch.testing.assert_close(target.cpu(), source, atol=0, rtol=0)
                assert target.data_ptr() % 16 == 0
    # CPU weights are immutable/shared between ModelPatcher branches.
    actual[0].weight.zero_()
    assert torch.count_nonzero(projections[0].weight) > 0


@pytest.mark.parametrize("precision", ["w8a8", "bf16", "fp16", "fp32"])
def test_veda_cuda_head_shards_use_global_predictor_indices(precision):
    torch.manual_seed(312)
    heads, dim = 4, 128
    projections = [convert_projection(torch.randn(heads, 384, dim) * 0.03, precision)
                   for _ in range(2)]
    plan = TilePlan("test", (1, 7, 19), [TileShape(1, 8, 16), TileShape(1, 1, 128)], [[0, 1, 0, 1]])
    bundle = PredictorBundle(precision, 1, heads, dim, 0.5, PlanTable([plan]),
                             (projections[0],), (projections[1],))
    config = VedaConfig(bundle, keep_ratio=0.5)
    layout = SimpleNamespace(seq_len=143, signature=(3, 1, 14, 38, 7),
                             segments=[(0, 3, "text"), (3, 10, "audio"), (10, 143, "video")])
    q, k, v = [torch.randn(1, heads, 143, dim, device="cuda", dtype=torch.float16) for _ in range(3)]
    cache = {}
    whole = attend(q, k, v, config=config, packed_layout=layout, layer=0, cache=cache)
    def project(indices, head_list):
        return tuple(t[0].transpose(0, 1).index_select(0, indices)[:, head_list] for t in (q, k, v))
    streamed = attend(q, k, v, config=config, packed_layout=layout, layer=0,
                      cache={}, projector=project, prepare_chunk_tiles=2)
    torch.testing.assert_close(streamed, whole, atol=0.002, rtol=0.002)
    row_chunked = attend(q, k, v, config=config, packed_layout=layout, layer=0,
                        cache={}, score_chunk_rows=1)
    torch.testing.assert_close(row_chunked, whole, atol=0.002, rtol=0.002)
    assert sum(entry[1] for entry in cache["gpu_layouts"].values()) <= 4 * 1024**2
    assert len(cache["gpu_layouts"]) == 2
    assert "layer_transfer" not in cache  # Predictor GPU weights never cached.
    compact = attend(q, k, v, config=config, packed_layout=layout, layer=0,
                     cache={}, prepare_chunk_tiles=1)
    torch.testing.assert_close(compact, whole, atol=0.002, rtol=0.002)
    parts = [attend(q[:, a:b], k[:, a:b], v[:, a:b], config=config,
                    packed_layout=layout, layer=0, head_start=a, cache={}) for a, b in ((0, 2), (2, 4))]
    torch.testing.assert_close(whole, torch.cat(parts, 1), atol=0.002, rtol=0.002)
    assert whole.dtype == v.dtype
