"""Bounded heterogeneous-layout attention; no persistent device storage."""
import torch

from ....kernel_api import load_kernel_extension


def batch_workspace_bytes(items, *, ragged=None):
    """New padded buffers/output, excluding the already-live input items."""
    if ragged is None:
        ragged = hasattr(load_kernel_extension("_sage_qattn_sm75"), "veda_sparse_ragged_attn")
    if ragged:
        return sum(p.value.numel() * p.value.element_size() + l.n_tiles * 2 +
                   p.value.shape[1] * 14 * 8 for p, l, r in items)
    slots = max(layout.num_slots for packed, layout, routes in items)
    heads = sum(packed.value.shape[1] for packed, layout, routes in items)
    packed = items[0][0]
    dim, element = packed.value.shape[-1], packed.value.element_size()
    tiles = slots // 128
    return heads * (slots * dim * (2 + 2 * element) + slots // 16 * 4 +
                    slots // 64 * 4 + tiles * ((tiles * 2 + 31) // 32) * 4 + tiles * 6)


def run_heterogeneous(items, *, budget_bytes, ragged=None):
    """Each item is (Sage packed QKV, TileLayout, packed routes).

    Pads only compact attention operands to a common length. Per-head counts
    and sparse-query flags preserve every shape's own globals and partial tiles.
    Returns HND views, in item order. Caller owns scatter/head restoration.
    """
    if not items:
        raise ValueError("Veda heterogeneous batch cannot be empty")
    native = load_kernel_extension("_sage_qattn_sm75")
    if ragged is None:
        ragged = hasattr(native, "veda_sparse_ragged_attn")
    required = batch_workspace_bytes(items, ragged=ragged)
    if required > budget_bytes:
        raise RuntimeError(f"Veda heterogeneous batch needs {required} bytes, budget {budget_bytes}")
    first = items[0][0]
    device, dtype = first.value.device, first.value.dtype
    if ragged:
        outputs, policies = [], []
        for packed, layout, route in items:
            if packed.sm_scale != first.sm_scale:
                raise ValueError("Incompatible Veda heterogeneous attention scales")
            outputs.append(torch.empty_like(packed.value))
            sparse = torch.zeros(layout.n_tiles * 2, dtype=torch.uint8, device=device)
            sparse[:layout.n_video_tiles * 2] = 1
            policies.append(sparse)
        with torch.cuda.device(device):
            native.veda_sparse_ragged_attn(
                [p.query_int8 for p, l, r in items], [p.key_int8 for p, l, r in items],
                [p.value for p, l, r in items], outputs,
                [p.query_scale for p, l, r in items], [p.key_scale for p, l, r in items],
                [r for p, l, r in items], policies, [l.valid_count for p, l, r in items], first.sm_scale)
        return outputs
    slots = max(layout.num_slots for _, layout, _ in items)
    heads = sum(p.value.shape[1] for p, _, _ in items)
    dim, tiles = first.value.shape[-1], slots // 128
    q = torch.zeros((1, heads, slots, dim), device=device, dtype=torch.int8)
    k = torch.zeros_like(q)
    v = torch.zeros(q.shape, device=device, dtype=dtype)
    qs = torch.zeros((1, heads, slots // 16), device=device)
    ks = torch.zeros((1, heads, slots // 64), device=device)
    routes = torch.zeros((1, heads, tiles, (tiles * 2 + 31) // 32), device=device, dtype=torch.int32)
    counts = torch.zeros((heads, tiles), device=device, dtype=torch.int32)
    sparse = torch.zeros((heads, tiles * 2), device=device, dtype=torch.uint8)
    cursor, slices = 0, []
    for packed, layout, route in items:
        if (packed.value.device != device or packed.value.dtype != dtype or
                packed.value.shape[0] != 1 or packed.value.shape[-1] != dim or
                packed.sm_scale != first.sm_scale):
            raise ValueError("Incompatible Veda heterogeneous attention operands")
        n, length = packed.value.shape[1], layout.num_slots
        end = cursor + n
        q[:, cursor:end, :length].copy_(packed.query_int8)
        k[:, cursor:end, :length].copy_(packed.key_int8)
        v[:, cursor:end, :length].copy_(packed.value)
        qs[:, cursor:end, :length // 16].copy_(packed.query_scale)
        ks[:, cursor:end, :length // 64].copy_(packed.key_scale)
        routes[:, cursor:end, :layout.n_tiles, :route.shape[-1]].copy_(route)
        counts[cursor:end, :layout.n_tiles].copy_(layout.valid_count)
        sparse[cursor:end, :layout.n_video_tiles * 2] = 1
        slices.append((cursor, end, length))
        cursor = end
    output = torch.empty_like(v)
    native = load_kernel_extension("_sage_qattn_sm75")
    with torch.cuda.device(device):
        native.veda_sparse_heterogeneous_attn(q, k, v, output, qs, ks, routes,
                                             sparse, counts, first.sm_scale)
    return [output[:, start:end, :length] for start, end, length in slices]
