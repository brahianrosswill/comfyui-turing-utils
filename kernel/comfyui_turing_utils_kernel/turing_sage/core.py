from collections import OrderedDict
from functools import wraps
from threading import Lock
from typing import Any, Optional

import torch

from .. import _sage_fused_sm75 as _fused
from . import sm75_compile
from .quant import (
    per_warp_int8,
    per_warp_int8_varlen,
    quantize_key_per_block,
    quantize_query_per_warp,
)


_FRAME_SCHEDULE_CACHE_LIMIT = 64
_FRAME_SCHEDULE_CACHE: OrderedDict[
    tuple, tuple[torch.Tensor, torch.Tensor, float]
] = OrderedDict()
_FRAME_SCHEDULE_CACHE_LOCK = Lock()
_SOL_POLICY_CACHE_LIMIT = 64
_SOL_POLICY_CACHE: OrderedDict[
    tuple, tuple[torch.Tensor, torch.Tensor, int]
] = OrderedDict()
_SOL_POLICY_CACHE_LOCK = Lock()


def _normalize_token_ranges(ranges, sequence_length: int) -> tuple[tuple[int, int], ...]:
    normalized = []
    for item in ranges or ():
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("Sol policy ranges must contain (start, stop) pairs")
        start, stop = (int(item[0]), int(item[1]))
        if start < 0 or stop <= start or stop > sequence_length:
            raise ValueError("Sol policy range is outside the attention sequence")
        normalized.append((start, stop))
    normalized.sort()
    for previous, current in zip(normalized, normalized[1:]):
        if current[0] < previous[1]:
            raise ValueError("Sol policy ranges must not overlap")
    return tuple(normalized)


def _sol_block_policy(
    device: torch.device,
    query_length: int,
    key_length: int,
    dense_query_ranges,
    exact_kv_ranges,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    dense_ranges = _normalize_token_ranges(dense_query_ranges, query_length)
    exact_ranges = _normalize_token_ranges(exact_kv_ranges, key_length)
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    cache_key = (
        device.type,
        device_index,
        query_length,
        key_length,
        dense_ranges,
        exact_ranges,
    )
    with _SOL_POLICY_CACHE_LOCK:
        cached = _SOL_POLICY_CACHE.get(cache_key)
        if cached is not None:
            _SOL_POLICY_CACHE.move_to_end(cache_key)
            return cached

    query_blocks = (query_length + 63) // 64
    key_blocks = (key_length + 63) // 64
    sparse_query = torch.ones(query_blocks, dtype=torch.uint8)
    exact_kv = torch.zeros(key_blocks, dtype=torch.uint8)
    for start, stop in dense_ranges:
        sparse_query[start // 64 : (stop + 63) // 64] = 0
    for start, stop in exact_ranges:
        exact_kv[start // 64 : (stop + 63) // 64] = 1
    sparse_count = int(sparse_query.sum().item())
    policy = (
        sparse_query.to(device),
        exact_kv.to(device),
        sparse_count,
    )
    with _SOL_POLICY_CACHE_LOCK:
        existing = _SOL_POLICY_CACHE.get(cache_key)
        if existing is not None:
            return existing
        _SOL_POLICY_CACHE[cache_key] = policy
        while len(_SOL_POLICY_CACHE) > _SOL_POLICY_CACHE_LIMIT:
            _SOL_POLICY_CACHE.popitem(last=False)
    return policy


def _frame_sparse_schedule_cpu(
    *,
    key_length: int,
    topology_start_tokens: int,
    topology_tokens: int,
    tokens_per_frame: int,
    prefix_tokens: int,
    temporal_window_frames: int,
    global_anchor_stride: int,
    global_anchor_offset: int,
    sink_frames: int,
    sparse_pattern: str = "frame_window",
    spatial_tokens_height: int = 0,
    spatial_tokens_width: int = 0,
    radial_spatial_radius: int = 1,
    radial_max_temporal_stride: int = 16,
) -> tuple[list[int], list[int], float]:
    """Build a cached, head-independent CSR schedule for the video tail."""
    block_tokens = 64
    topology_end = topology_start_tokens + topology_tokens
    if topology_end != key_length:
        raise ValueError("frame-sparse video topology must be the contiguous K/V tail")
    if topology_tokens <= 0 or tokens_per_frame <= 0:
        raise ValueError("frame-sparse video topology must be non-empty")
    if topology_tokens % tokens_per_frame:
        raise ValueError("frame-sparse topology must contain complete token frames")
    frame_count = topology_tokens // tokens_per_frame
    if frame_count <= 0:
        raise ValueError("frame-sparse topology contains no frames")
    if sparse_pattern not in {"frame_window", "radial"}:
        raise ValueError("sparse_pattern must be frame_window or radial")
    if radial_spatial_radius < 0:
        raise ValueError("radial_spatial_radius must be non-negative")
    if radial_max_temporal_stride <= 0:
        raise ValueError("radial_max_temporal_stride must be positive")
    if sparse_pattern == "radial" and (
        spatial_tokens_height <= 0
        or spatial_tokens_width <= 0
        or spatial_tokens_height * spatial_tokens_width != tokens_per_frame
    ):
        raise ValueError(
            "radial frame sparsity requires exact spatial token height and width"
        )

    num_query_blocks = (topology_tokens + block_tokens - 1) // block_tokens
    num_key_blocks = (key_length + block_tokens - 1) // block_tokens
    forced_prefix_blocks = range(
        (prefix_tokens + block_tokens - 1) // block_tokens
    )
    sink_frame_set = set(range(min(sink_frames, frame_count)))
    anchor_frames = set(sink_frame_set)
    if global_anchor_stride > 0:
        offset = global_anchor_offset % global_anchor_stride
        anchor_frames.update(range(offset, frame_count, global_anchor_stride))

    row_offsets = [0]
    selected_key_blocks: list[int] = []

    def select_token_range(selected: set[int], token_start: int, token_end: int):
        if token_end <= token_start:
            return
        first_key_block = token_start // block_tokens
        last_key_block = (token_end - 1) // block_tokens
        selected.update(range(first_key_block, last_key_block + 1))

    complete_frame_blocks: list[tuple[int, ...]] = []
    for frame in range(frame_count):
        token_start = topology_start_tokens + frame * tokens_per_frame
        first_key_block = token_start // block_tokens
        last_key_block = (token_start + tokens_per_frame - 1) // block_tokens
        complete_frame_blocks.append(tuple(range(first_key_block, last_key_block + 1)))

    def select_complete_frame(selected: set[int], frame: int):
        selected.update(complete_frame_blocks[frame])

    spatial_tile_edge = 8
    spatial_tile_rows = (
        spatial_tokens_height + spatial_tile_edge - 1
    ) // spatial_tile_edge if spatial_tokens_height else 0
    spatial_tile_columns = (
        spatial_tokens_width + spatial_tile_edge - 1
    ) // spatial_tile_edge if spatial_tokens_width else 0
    spatial_block_cache: dict[tuple[int, tuple[tuple[int, int], ...]], tuple[int, ...]] = {}

    def query_spatial_tiles(query_start: int, query_end: int) -> set[tuple[int, int]]:
        tiles: set[tuple[int, int]] = set()
        for token in range(query_start, query_end):
            spatial_token = token % tokens_per_frame
            row, column = divmod(spatial_token, spatial_tokens_width)
            tiles.add((row // spatial_tile_edge, column // spatial_tile_edge))
        return tiles

    def select_spatial_tiles(
        selected: set[int], frame: int, query_tiles: set[tuple[int, int]]
    ):
        cache_key = (frame, tuple(sorted(query_tiles)))
        cached_blocks = spatial_block_cache.get(cache_key)
        if cached_blocks is not None:
            selected.update(cached_blocks)
            return
        frame_start = topology_start_tokens + frame * tokens_per_frame
        selected_tiles: set[tuple[int, int]] = set()
        for query_row, query_column in query_tiles:
            for tile_row in range(
                max(0, query_row - radial_spatial_radius),
                min(spatial_tile_rows, query_row + radial_spatial_radius + 1),
            ):
                for tile_column in range(
                    max(0, query_column - radial_spatial_radius),
                    min(
                        spatial_tile_columns,
                        query_column + radial_spatial_radius + 1,
                    ),
                ):
                    selected_tiles.add((tile_row, tile_column))
        blocks: set[int] = set()
        for tile_row, tile_column in selected_tiles:
            row_begin = tile_row * spatial_tile_edge
            row_end = min(row_begin + spatial_tile_edge, spatial_tokens_height)
            column_begin = tile_column * spatial_tile_edge
            column_end = min(
                column_begin + spatial_tile_edge, spatial_tokens_width
            )
            for row in range(row_begin, row_end):
                token_start = frame_start + row * spatial_tokens_width + column_begin
                select_token_range(
                    blocks,
                    token_start,
                    token_start + column_end - column_begin,
                )
        cached_blocks = tuple(sorted(blocks))
        spatial_block_cache[cache_key] = cached_blocks
        selected.update(cached_blocks)

    for query_block in range(num_query_blocks):
        query_start = query_block * block_tokens
        query_end = min(query_start + block_tokens, topology_tokens)
        first_query_frame = query_start // tokens_per_frame
        last_query_frame = (query_end - 1) // tokens_per_frame
        selected = set(forced_prefix_blocks)
        near_frames = set(
            range(
                max(0, first_query_frame - temporal_window_frames),
                min(frame_count, last_query_frame + temporal_window_frames + 1),
            )
        )
        for frame in anchor_frames | near_frames:
            select_complete_frame(selected, frame)

        if sparse_pattern == "radial":
            query_tiles = query_spatial_tiles(query_start, query_end)
            for frame in range(frame_count):
                if frame in anchor_frames or frame in near_frames:
                    continue
                distance = min(
                    abs(frame - query_frame)
                    for query_frame in range(first_query_frame, last_query_frame + 1)
                )
                temporal_stride = min(
                    1 << max(0, distance.bit_length() - 1),
                    radial_max_temporal_stride,
                )
                if (frame - global_anchor_offset) % temporal_stride:
                    continue
                select_spatial_tiles(selected, frame, query_tiles)
        row = sorted(block for block in selected if 0 <= block < num_key_blocks)
        if not row:
            raise ValueError("frame-sparse schedule produced an empty Query row")
        selected_key_blocks.extend(row)
        row_offsets.append(len(selected_key_blocks))

    density = len(selected_key_blocks) / (num_query_blocks * num_key_blocks)
    return row_offsets, selected_key_blocks, density


def _frame_sparse_schedule(
    *,
    device: torch.device,
    key_length: int,
    topology_start_tokens: int,
    topology_tokens: int,
    tokens_per_frame: int,
    prefix_tokens: int,
    temporal_window_frames: int,
    global_anchor_stride: int,
    global_anchor_offset: int,
    sink_frames: int,
    sparse_pattern: str = "frame_window",
    spatial_tokens_height: int = 0,
    spatial_tokens_width: int = 0,
    radial_spatial_radius: int = 1,
    radial_max_temporal_stride: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    cache_key = (
        device.type,
        device_index,
        key_length,
        topology_start_tokens,
        topology_tokens,
        tokens_per_frame,
        prefix_tokens,
        temporal_window_frames,
        global_anchor_stride,
        global_anchor_offset,
        sink_frames,
        sparse_pattern,
        spatial_tokens_height,
        spatial_tokens_width,
        radial_spatial_radius,
        radial_max_temporal_stride,
    )
    with _FRAME_SCHEDULE_CACHE_LOCK:
        cached = _FRAME_SCHEDULE_CACHE.get(cache_key)
        if cached is not None:
            _FRAME_SCHEDULE_CACHE.move_to_end(cache_key)
            return cached

    row_offsets, key_blocks, density = _frame_sparse_schedule_cpu(
        key_length=key_length,
        topology_start_tokens=topology_start_tokens,
        topology_tokens=topology_tokens,
        tokens_per_frame=tokens_per_frame,
        prefix_tokens=prefix_tokens,
        temporal_window_frames=temporal_window_frames,
        global_anchor_stride=global_anchor_stride,
        global_anchor_offset=global_anchor_offset,
        sink_frames=sink_frames,
        sparse_pattern=sparse_pattern,
        spatial_tokens_height=spatial_tokens_height,
        spatial_tokens_width=spatial_tokens_width,
        radial_spatial_radius=radial_spatial_radius,
        radial_max_temporal_stride=radial_max_temporal_stride,
    )
    schedule = (
        torch.tensor(row_offsets, dtype=torch.int32, device=device),
        torch.tensor(key_blocks, dtype=torch.int32, device=device),
        density,
    )
    with _FRAME_SCHEDULE_CACHE_LOCK:
        existing = _FRAME_SCHEDULE_CACHE.get(cache_key)
        if existing is not None:
            return existing
        _FRAME_SCHEDULE_CACHE[cache_key] = schedule
        while len(_FRAME_SCHEDULE_CACHE) > _FRAME_SCHEDULE_CACHE_LIMIT:
            _FRAME_SCHEDULE_CACHE.popitem(last=False)
    return schedule


def _on_input_device(function):
    @wraps(function)
    def wrapped(q: torch.Tensor, *args, **kwargs):
        with torch.cuda.device(q.device):
            return function(q, *args, **kwargs)

    return wrapped


def _validate_fixed_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str) -> None:
    if tensor_layout not in {"HND", "NHD"}:
        raise ValueError(f"Unsupported tensor_layout: {tensor_layout}")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("fixed-length Q/K/V must be four-dimensional")
    head_dim = q.size(-1)
    if k.size(-1) != head_dim or v.size(-1) != head_dim:
        raise ValueError("Q/K/V head dimensions must match")
    head_axis = 1 if tensor_layout == "HND" else 2
    seq_axis = 2 if tensor_layout == "HND" else 1
    if q.size(0) != k.size(0) or q.size(0) != v.size(0):
        raise ValueError("Q/K/V batch sizes must match")
    if k.size(head_axis) != v.size(head_axis) or k.size(seq_axis) != v.size(seq_axis):
        raise ValueError("K/V head counts and sequence lengths must match")
    if k.size(head_axis) == 0 or q.size(head_axis) % k.size(head_axis) != 0:
        raise ValueError("the Q head count must be divisible by the KV head count")
    if q.size(seq_axis) == 0 or k.size(seq_axis) == 0:
        raise ValueError("empty Q/K sequences are not supported")


def _short_sequence_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    sm_scale: Optional[float],
    return_lse: bool,
):
    """Use a bounded exact path below the SM75 kernel's 64-token CTA."""
    sequence_axis = 2 if tensor_layout == "HND" else 1
    if q.size(sequence_axis) >= 64 and k.size(sequence_axis) >= 64:
        return None

    q_hnd = q if tensor_layout == "HND" else q.transpose(1, 2)
    k_hnd = k if tensor_layout == "HND" else k.transpose(1, 2)
    v_hnd = v if tensor_layout == "HND" else v.transpose(1, 2)
    q_heads = q_hnd.size(1)
    kv_heads = k_hnd.size(1)
    scale = float(sm_scale) if sm_scale is not None else q.size(-1) ** -0.5
    output_hnd = torch.nn.functional.scaled_dot_product_attention(
        q_hnd.float(),
        k_hnd.float(),
        v_hnd.float(),
        is_causal=is_causal,
        enable_gqa=q_heads != kv_heads,
        scale=scale,
    ).to(q.dtype)
    output = output_hnd if tensor_layout == "HND" else output_hnd.transpose(1, 2)
    if not return_lse:
        return output, None

    key_for_q = torch.repeat_interleave(k_hnd.float(), q_heads // kv_heads, dim=1)
    scores = torch.matmul(q_hnd.float(), key_for_q.transpose(-2, -1)) * scale
    if is_causal:
        causal_mask = torch.ones(
            (q_hnd.size(2), k_hnd.size(2)), dtype=torch.bool, device=q.device
        ).tril()
        scores.masked_fill_(~causal_mask, float("-inf"))
    return output, torch.logsumexp(scores, dim=-1)


@_on_input_device
def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    smooth_k: bool = False,
    **kwargs: Any,
):
    """Stable SM75 Sage: per-warp INT8 Q/K and direct FP32 PV accumulation."""
    if smooth_k:
        raise ValueError("the production Turing Sage backend does not enable experimental smoothing")
    if not q.is_cuda:
        raise ValueError("Input tensors must be on CUDA")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Turing Sage Q/K/V must be float16 or bfloat16")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("Q/K/V must have matching dtypes")
    _validate_fixed_qkv(q, k, v, tensor_layout)

    short_result = _short_sequence_attention(
        q, k, v, tensor_layout, is_causal, sm_scale, return_lse
    )
    if short_result is not None:
        return short_result if return_lse else short_result[0]

    tensor_layout_id = 0 if tensor_layout == "NHD" else 1
    head_dim = q.size(-1)
    if head_dim < 64:
        padding = 64 - head_dim
    elif 64 < head_dim < 128:
        padding = 128 - head_dim
    elif head_dim > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim}")
    else:
        padding = 0
    if padding:
        q = torch.nn.functional.pad(q, (0, padding))
        k = torch.nn.functional.pad(k, (0, padding))
        v = torch.nn.functional.pad(v, (0, padding))
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("the last Q/K/V dimension must be contiguous")

    scale = float(sm_scale) if sm_scale is not None else head_dim**-0.5
    q_int8, q_scale, k_int8, k_scale = per_warp_int8(
        q,
        k,
        tensor_layout=tensor_layout,
        fuse_qk=(is_causal or (tensor_layout == "HND" and q.size(-1) == 64)),
    )
    output = torch.empty_like(q)
    lse = sm75_compile.qk_int8_sv_f16_accum_f32_attn(
        q_int8,
        k_int8,
        v.contiguous(),
        output,
        q_scale,
        k_scale,
        tensor_layout_id,
        int(is_causal),
        2,
        scale,
        int(return_lse),
    )
    output = output[..., :head_dim]
    return (output, lse / 1.44269504) if return_lse else output


@_on_input_device
def sageattn_prequantized(
    q_int8: torch.Tensor,
    q_scale: torch.Tensor,
    k_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    *,
    tensor_layout: str = "NHD",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    output: Optional[torch.Tensor] = None,
):
    """Internal bridge for adapters that can release BF16 Q/K before attention."""
    if tensor_layout not in {"HND", "NHD"}:
        raise ValueError(f"Unsupported tensor_layout: {tensor_layout}")
    if q_int8.dtype != torch.int8 or k_int8.dtype != torch.int8:
        raise TypeError("prequantized Sage Q/K must be int8")
    if v.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("prequantized Sage V must be float16 or bfloat16")
    if q_int8.ndim != 4 or k_int8.ndim != 4 or v.ndim != 4:
        raise ValueError("prequantized Sage Q/K/V must be four-dimensional")
    if q_int8.device != k_int8.device or q_int8.device != v.device:
        raise ValueError("prequantized Sage Q/K/V must share one CUDA device")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("prequantized Sage scales must be float32")
    if q_scale.device != q_int8.device or k_scale.device != q_int8.device:
        raise ValueError("prequantized Sage scales must be on the Q/K device")

    head_axis = 1 if tensor_layout == "HND" else 2
    sequence_axis = 2 if tensor_layout == "HND" else 1
    if (
        q_int8.size(0) != k_int8.size(0)
        or q_int8.size(0) != v.size(0)
        or k_int8.size(head_axis) != v.size(head_axis)
        or k_int8.size(sequence_axis) != v.size(sequence_axis)
        or q_int8.size(-1) != k_int8.size(-1)
        or q_int8.size(-1) != v.size(-1)
    ):
        raise ValueError("prequantized Sage Q/K/V shapes are incompatible")
    q_heads = q_int8.size(head_axis)
    kv_heads = k_int8.size(head_axis)
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError("prequantized Sage Q heads must be divisible by KV heads")

    head_dim = q_int8.size(-1)
    if head_dim not in (64, 128):
        raise ValueError("prequantized Sage currently requires head_dim 64 or 128")
    expected_q_tiles = ((q_int8.size(sequence_axis) + 63) // 64) * 4
    expected_k_tiles = (k_int8.size(sequence_axis) + 63) // 64
    if q_scale.shape != (q_int8.size(0), q_heads, expected_q_tiles):
        raise ValueError("prequantized Sage Q scale shape is incompatible")
    if k_scale.shape != (k_int8.size(0), kv_heads, expected_k_tiles):
        raise ValueError("prequantized Sage K scale shape is incompatible")

    tensor_layout_id = 0 if tensor_layout == "NHD" else 1
    scale = float(sm_scale) if sm_scale is not None else head_dim**-0.5
    if output is None:
        output = torch.empty(q_int8.shape, dtype=v.dtype, device=v.device)
    elif (
        output.shape != q_int8.shape
        or output.dtype != v.dtype
        or output.device != v.device
        or output.stride(-1) != 1
    ):
        raise ValueError("prequantized Sage output is incompatible")
    lse = sm75_compile.qk_int8_sv_f16_accum_f32_attn(
        q_int8.contiguous(),
        k_int8.contiguous(),
        v.contiguous(),
        output,
        q_scale.contiguous(),
        k_scale.contiguous(),
        tensor_layout_id,
        int(is_causal),
        2,
        scale,
        int(return_lse),
    )
    return (output, lse / 1.44269504) if return_lse else output


@_on_input_device
def w8a8attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tensor_layout: str = "HND",
    sm_scale: Optional[float] = None,
):
    """Experimental pure-INT8 QK/PV attention specialized for SM75.

    Q/K use the stable Sage INT8 score domain. V is quantized channel-wise to
    signed INT8 and softmax probabilities are packed to unsigned INT8 for the
    second SM75 Tensor Core MMA. A route-free specialization omits all Sol
    summaries and routing state while retaining the shared exact-token core.
    """
    if tensor_layout not in {"HND", "NHD"}:
        raise ValueError(f"Unsupported tensor_layout: {tensor_layout}")
    if tensor_layout == "NHD":
        q_hnd = q.transpose(1, 2).contiguous()
        k_hnd = k.transpose(1, 2).contiguous()
        v_hnd = v.transpose(1, 2).contiguous()
    else:
        q_hnd, k_hnd, v_hnd = q, k, v
    _validate_fixed_qkv(q_hnd, k_hnd, v_hnd, "HND")
    if q_hnd.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Turing W8A8 Q/K/V must be float16 or bfloat16")
    if q_hnd.dtype != k_hnd.dtype or q_hnd.dtype != v_hnd.dtype:
        raise TypeError("Turing W8A8 Q/K/V must have matching dtypes")
    if q_hnd.size(-1) != 128:
        raise ValueError("Turing W8A8 currently requires head_dim=128")
    output = sol_sparse_sageattn(
        q_hnd,
        k_hnd,
        v_hnd,
        tensor_layout="HND",
        sm_scale=sm_scale,
        threshold_sigma=0.0,
        residual_subblocks=1,
        use_w8a8=True,
        _force_dense=True,
    )
    return output.transpose(1, 2) if tensor_layout == "NHD" else output


@_on_input_device
def sol_sparse_sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tensor_layout: str = "HND",
    sm_scale: Optional[float] = None,
    dense_query_ranges=(),
    exact_kv_ranges=(),
    threshold_sigma: float = 1.0,
    residual_subblocks: int = 1,
    return_stats: bool = False,
    use_w8a8: bool = False,
    _force_dense: bool = False,
):
    """SM75 Sol attention with online routing and modality-aware exact ranges."""
    if not q.is_cuda:
        raise ValueError("Input tensors must be on CUDA")
    if tensor_layout != "HND":
        raise ValueError("experimental sparse attention currently requires HND layout")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Turing sparse Q/K/V must be float16 or bfloat16")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("Q/K/V must have matching dtypes")
    _validate_fixed_qkv(q, k, v, tensor_layout)
    if q.size(-1) != 128:
        raise ValueError("experimental sparse attention requires head_dim=128")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("the last Q/K/V dimension must be contiguous")

    scale = float(sm_scale) if sm_scale is not None else 128**-0.5
    residual_subblocks = int(residual_subblocks)
    if residual_subblocks not in (1, 2):
        raise ValueError("residual_subblocks must be 1 or 2")
    sparse_query_blocks, exact_kv_blocks, sparse_block_count = _sol_block_policy(
        q.device,
        q.size(2),
        k.size(2),
        dense_query_ranges,
        exact_kv_ranges,
    )
    key_block_count = (k.size(2) + 63) // 64
    possible_blocks = q.size(0) * q.size(1) * sparse_block_count * key_block_count
    if sparse_block_count == 0 and not use_w8a8:
        dense = sageattn(q, k, v, tensor_layout="HND", sm_scale=scale)
        if return_stats:
            return dense, torch.zeros(1, dtype=torch.int64, device=q.device), 0
        return dense
    if sparse_block_count == 0:
        _force_dense = True

    output = torch.empty_like(q)
    q_int8, q_scale = quantize_query_per_warp(q, tensor_layout="HND")
    k_int8, k_scale = quantize_key_per_block(k, tensor_layout="HND")
    if use_w8a8:
        padded_key_length = ((k.size(2) + 63) // 64) * 64
        value_int8 = torch.empty(
            (v.size(0), v.size(1), v.size(3), padded_key_length),
            dtype=torch.int8,
            device=v.device,
        )
        value_scale = torch.empty(
            (v.size(0), v.size(1), v.size(3)),
            dtype=torch.float32,
            device=v.device,
        )
        sm75_compile.quantize_v_int8(v, value_int8, value_scale)
    else:
        value_int8 = torch.empty(0, dtype=torch.int8, device=v.device)
        value_scale = torch.empty(0, dtype=torch.float32, device=v.device)
    selected = sm75_compile.sol_sparse_online_int8_f16_attn(
        q_int8,
        k_int8,
        v,
        value_int8,
        value_scale,
        output,
        q_scale,
        k_scale,
        sparse_query_blocks,
        exact_kv_blocks,
        float(threshold_sigma),
        residual_subblocks,
        scale,
        int(return_stats),
        int(bool(use_w8a8)),
        int(bool(_force_dense)),
    )

    # Dense Query blocks are handled by the same CUDA grid.  They bypass route
    # pruning and scan every K/V block, so no Python-side sublaunch or output
    # copy is required and Q/K/V quantization remains single-pass.
    return (output, selected, possible_blocks) if return_stats else output


@_on_input_device
def frame_sparse_sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tensor_layout: str = "HND",
    sm_scale: Optional[float] = None,
    prefix_tokens: int = 0,
    topology_start_tokens: int,
    topology_tokens: int,
    tokens_per_frame: int,
    temporal_window_frames: int = 2,
    global_anchor_stride: int = 12,
    global_anchor_offset: int = 0,
    sink_frames: int = 1,
    sparse_pattern: str = "frame_window",
    spatial_tokens_height: int = 0,
    spatial_tokens_width: int = 0,
    radial_spatial_radius: int = 1,
    radial_max_temporal_stride: int = 16,
    return_schedule_density: bool = False,
):
    """Structured frame-sparse SM75 Sage with a cached head-independent route."""
    if not q.is_cuda:
        raise ValueError("Input tensors must be on CUDA")
    if tensor_layout != "HND":
        raise ValueError("frame-sparse attention currently requires HND layout")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("frame-sparse Q/K/V must be float16 or bfloat16")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("Q/K/V must have matching dtypes")
    _validate_fixed_qkv(q, k, v, tensor_layout)
    if q.size(-1) != 128:
        raise ValueError("frame-sparse attention requires head_dim=128")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("the last Q/K/V dimension must be contiguous")
    if q.size(2) != k.size(2):
        raise ValueError("frame-sparse attention requires equal Q/K sequence lengths")

    sequence_length = q.size(2)
    prefix_tokens = int(prefix_tokens)
    topology_start_tokens = int(topology_start_tokens)
    topology_tokens = int(topology_tokens)
    tokens_per_frame = int(tokens_per_frame)
    temporal_window_frames = int(temporal_window_frames)
    global_anchor_stride = int(global_anchor_stride)
    global_anchor_offset = int(global_anchor_offset)
    sink_frames = int(sink_frames)
    sparse_pattern = str(sparse_pattern).strip().lower()
    spatial_tokens_height = int(spatial_tokens_height)
    spatial_tokens_width = int(spatial_tokens_width)
    radial_spatial_radius = int(radial_spatial_radius)
    radial_max_temporal_stride = int(radial_max_temporal_stride)
    if not 0 <= prefix_tokens <= sequence_length:
        raise ValueError("prefix_tokens is outside the shared Q/K sequence")
    if topology_start_tokens < 0 or topology_tokens <= 0 or tokens_per_frame <= 0:
        raise ValueError("frame-sparse topology values must be positive")
    if topology_start_tokens + topology_tokens != sequence_length:
        raise ValueError("frame-sparse video topology must be the contiguous sequence tail")
    if topology_tokens % tokens_per_frame:
        raise ValueError("frame-sparse topology must contain complete token frames")
    if temporal_window_frames < 0:
        raise ValueError("temporal_window_frames must be non-negative")
    if global_anchor_stride < 0:
        raise ValueError("global_anchor_stride must be non-negative")
    if sink_frames < 0:
        raise ValueError("sink_frames must be non-negative")
    if sparse_pattern not in {"frame_window", "radial"}:
        raise ValueError("sparse_pattern must be frame_window or radial")
    if radial_spatial_radius < 0:
        raise ValueError("radial_spatial_radius must be non-negative")
    if radial_max_temporal_stride <= 0:
        raise ValueError("radial_max_temporal_stride must be positive")
    if sparse_pattern == "radial" and (
        spatial_tokens_height <= 0
        or spatial_tokens_width <= 0
        or spatial_tokens_height * spatial_tokens_width != tokens_per_frame
    ):
        raise ValueError(
            "radial frame sparsity requires exact spatial token height and width"
        )

    scale = float(sm_scale) if sm_scale is not None else 128**-0.5
    output = torch.empty_like(q)
    k_int8, k_scale = quantize_key_per_block(k, tensor_layout="HND")

    # Non-video Query tokens remain exact and see every K/V token. This keeps
    # text/reference/audio outputs global even when prefix K protection is
    # explicitly reduced for video Query rows.
    if topology_start_tokens:
        q_prefix = q[:, :, :topology_start_tokens]
        prefix_output = output[:, :, :topology_start_tokens]
        padded_prefix = topology_start_tokens < 64
        if padded_prefix:
            q_prefix = torch.nn.functional.pad(
                q_prefix, (0, 0, 0, 64 - topology_start_tokens)
            )
        q_prefix_int8, q_prefix_scale = quantize_query_per_warp(
            q_prefix, tensor_layout="HND"
        )
        if padded_prefix:
            padded_output = torch.empty_like(q_prefix)
            sageattn_prequantized(
                q_prefix_int8,
                q_prefix_scale,
                k_int8,
                k_scale,
                v,
                tensor_layout="HND",
                sm_scale=scale,
                output=padded_output,
            )
            prefix_output.copy_(padded_output[:, :, :topology_start_tokens])
            del padded_output
        else:
            sageattn_prequantized(
                q_prefix_int8,
                q_prefix_scale,
                k_int8,
                k_scale,
                v,
                tensor_layout="HND",
                sm_scale=scale,
                output=prefix_output,
            )
        del q_prefix_int8, q_prefix_scale

    q_video = q[:, :, topology_start_tokens:]
    video_output = output[:, :, topology_start_tokens:]
    q_int8, q_scale = quantize_query_per_warp(q_video, tensor_layout="HND")
    row_offsets, key_blocks, density = _frame_sparse_schedule(
        device=q.device,
        key_length=sequence_length,
        topology_start_tokens=topology_start_tokens,
        topology_tokens=topology_tokens,
        tokens_per_frame=tokens_per_frame,
        prefix_tokens=prefix_tokens,
        temporal_window_frames=temporal_window_frames,
        global_anchor_stride=global_anchor_stride,
        global_anchor_offset=global_anchor_offset,
        sink_frames=sink_frames,
        sparse_pattern=sparse_pattern,
        spatial_tokens_height=spatial_tokens_height,
        spatial_tokens_width=spatial_tokens_width,
        radial_spatial_radius=radial_spatial_radius,
        radial_max_temporal_stride=radial_max_temporal_stride,
    )
    sm75_compile.frame_sparse_int8_f16_attn(
        q_int8.contiguous(),
        k_int8.contiguous(),
        v.contiguous(),
        video_output,
        q_scale.contiguous(),
        k_scale.contiguous(),
        row_offsets,
        key_blocks,
        scale,
    )
    return (output, density) if return_schedule_density else output


@_on_input_device
def sageattn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Variable-length stable Sage facade."""
    if smooth_k:
        raise ValueError("the production Turing Sage backend does not enable experimental smoothing")
    if not q.is_cuda:
        raise ValueError("Input tensors must be on CUDA")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Turing Sage Q/K/V must be float16 or bfloat16")
    if q.device != k.device or q.device != v.device or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q/K/V must share a CUDA device and dtype")

    head_dim = q.size(-1)
    if head_dim < 64:
        padding = 64 - head_dim
    elif 64 < head_dim < 128:
        padding = 128 - head_dim
    elif head_dim > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim}")
    else:
        padding = 0
    if padding:
        q = torch.nn.functional.pad(q, (0, padding))
        k = torch.nn.functional.pad(k, (0, padding))
        v = torch.nn.functional.pad(v, (0, padding))
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("the last Q/K/V dimension must be contiguous")
    if not cu_seqlens_q.is_contiguous() or not cu_seqlens_k.is_contiguous():
        raise ValueError("cu_seqlens_q/cu_seqlens_k must be contiguous")

    scale = float(sm_scale) if sm_scale is not None else head_dim**-0.5
    if max_seqlen_q >= 512:
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_varlen(
            q,
            k,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        )
        output = torch.empty_like(q)
        sm75_compile.qk_int8_sv_f16_varlen_accum_f32_attn(
            q_int8,
            k_int8,
            v.contiguous(),
            output,
            q_scale,
            k_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            int(is_causal),
            scale,
        )
        return output[..., :head_dim]

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = torch.empty_like(q)
    _fused.varlen_attention_fwd_cuda(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        output,
        max_seqlen_q,
        scale,
        int(is_causal),
    )
    return output[..., :head_dim]
