from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from typing import Any, Optional

import torch

from .. import _sage_fused_sm75 as _fused
from . import sm75_compile
from .quant import (
    per_warp_int8,
    per_warp_int8_hadamard,
    per_warp_int8_varlen,
)


_SOL_POLICY_CACHE_LIMIT = 64
_SOL_POLICY_CACHE: OrderedDict[
    tuple, tuple[torch.Tensor, torch.Tensor, int]
] = OrderedDict()
_SOL_POLICY_CACHE_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class PrequantizedSageAttention:
    query_int8: torch.Tensor
    query_scale: torch.Tensor
    key_int8: torch.Tensor
    key_scale: torch.Tensor
    value: torch.Tensor
    tensor_layout: str
    original_head_dim: int
    is_causal: bool
    sm_scale: float


@dataclass(frozen=True, slots=True)
class PrequantizedSolAttention:
    query_int8: torch.Tensor
    query_scale: torch.Tensor
    key_int8: torch.Tensor
    key_scale: torch.Tensor
    value: Optional[torch.Tensor]
    value_int8: torch.Tensor
    value_scale: torch.Tensor
    summaries: tuple[torch.Tensor, ...]
    sparse_query_blocks: torch.Tensor
    exact_kv_blocks: torch.Tensor
    output_dtype: torch.dtype
    sm_scale: float
    threshold_sigma: float
    residual_subblocks: int
    possible_blocks: int
    use_w8a8: bool
    force_dense: bool


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
def prequantize_sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = False,
    **kwargs: Any,
) -> PrequantizedSageAttention:
    """Quantize Q/K and detach V storage before allocating the output."""
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

    sequence_axis = 2 if tensor_layout == "HND" else 1
    if q.size(sequence_axis) < 64 or k.size(sequence_axis) < 64:
        raise ValueError("split Sage attention requires Q/K sequences of at least 64 tokens")
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
    return PrequantizedSageAttention(
        query_int8=q_int8,
        query_scale=q_scale,
        key_int8=k_int8,
        key_scale=k_scale,
        value=v.contiguous(),
        tensor_layout=tensor_layout,
        original_head_dim=head_dim,
        is_causal=bool(is_causal),
        sm_scale=scale,
    )


def sageattn_from_prequantized(
    quantized: PrequantizedSageAttention,
    *,
    return_lse: bool = False,
):
    with torch.cuda.device(quantized.query_int8.device):
        result = sageattn_prequantized(
            quantized.query_int8,
            quantized.query_scale,
            quantized.key_int8,
            quantized.key_scale,
            quantized.value,
            tensor_layout=quantized.tensor_layout,
            is_causal=quantized.is_causal,
            sm_scale=quantized.sm_scale,
            return_lse=return_lse,
        )
    if return_lse:
        output, lse = result
        return output[..., :quantized.original_head_dim], lse
    return result[..., :quantized.original_head_dim]


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
    short_result = _short_sequence_attention(
        q, k, v, tensor_layout, is_causal, sm_scale, return_lse
    )
    if short_result is not None:
        return short_result if return_lse else short_result[0]
    quantized = prequantize_sageattn.__wrapped__(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        sm_scale=sm_scale,
        smooth_k=smooth_k,
    )
    return sageattn_from_prequantized(quantized, return_lse=return_lse)


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
    quantized = prequantize_sol_sageattn.__wrapped__(
        q_hnd,
        k_hnd,
        v_hnd,
        tensor_layout="HND",
        sm_scale=sm_scale,
        threshold_sigma=0.0,
        residual_subblocks=1,
        use_w8a8=True,
        force_dense=True,
    )
    output = sol_sparse_sageattn_from_prequantized(quantized)
    return output.transpose(1, 2) if tensor_layout == "NHD" else output


@_on_input_device
def prequantize_sol_sageattn(
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
    use_w8a8: bool = False,
    force_dense: bool = False,
) -> PrequantizedSolAttention:
    """Prepare Sol Q/K/V and correction state before output allocation."""
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
        raise ValueError("split FP16 Sol has no sparse Query blocks; use stable Sage")
    if sparse_block_count == 0:
        force_dense = True

    q_int8, q_scale, k_int8, k_scale = per_warp_int8_hadamard(
        q, k, tensor_layout="HND"
    )
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
        if force_dense:
            half_empty = torch.empty((0, 0, 0, 0), dtype=torch.float16, device=v.device)
            float_empty = torch.empty((0, 0, 0), dtype=torch.float32, device=v.device)
            summaries = (half_empty, half_empty, half_empty, float_empty, float_empty)
        else:
            summaries = tuple(
                sm75_compile.sol_w8a8_precompute_summaries(
                    k_int8,
                    k_scale,
                    v,
                    value_scale,
                    residual_subblocks,
                )
            )
        retained_value = None
    else:
        value_int8 = torch.empty(0, dtype=torch.int8, device=v.device)
        value_scale = torch.empty(0, dtype=torch.float32, device=v.device)
        summaries = ()
        retained_value = v.contiguous()
    return PrequantizedSolAttention(
        query_int8=q_int8,
        query_scale=q_scale,
        key_int8=k_int8,
        key_scale=k_scale,
        value=retained_value,
        value_int8=value_int8,
        value_scale=value_scale,
        summaries=summaries,
        sparse_query_blocks=sparse_query_blocks,
        exact_kv_blocks=exact_kv_blocks,
        output_dtype=v.dtype,
        sm_scale=scale,
        threshold_sigma=float(threshold_sigma),
        residual_subblocks=residual_subblocks,
        possible_blocks=possible_blocks,
        use_w8a8=bool(use_w8a8),
        force_dense=bool(force_dense),
    )


def sol_sparse_sageattn_from_prequantized(
    quantized: PrequantizedSolAttention,
    *,
    return_stats: bool = False,
):
    with torch.cuda.device(quantized.query_int8.device):
        output = torch.empty(
            quantized.query_int8.shape,
            dtype=quantized.output_dtype,
            device=quantized.query_int8.device,
        )
        if quantized.use_w8a8:
            selected = sm75_compile.sol_sparse_online_w8a8_prequantized_attn(
                quantized.query_int8,
                quantized.key_int8,
                quantized.value_int8,
                quantized.value_scale,
                output,
                quantized.query_scale,
                quantized.key_scale,
                quantized.summaries,
                quantized.sparse_query_blocks,
                quantized.exact_kv_blocks,
                quantized.threshold_sigma,
                quantized.residual_subblocks,
                quantized.sm_scale,
                int(return_stats),
                int(quantized.force_dense),
            )
        else:
            selected = sm75_compile.sol_sparse_online_int8_f16_attn(
                quantized.query_int8,
                quantized.key_int8,
                quantized.value,
                quantized.value_int8,
                quantized.value_scale,
                output,
                quantized.query_scale,
                quantized.key_scale,
                quantized.sparse_query_blocks,
                quantized.exact_kv_blocks,
                quantized.threshold_sigma,
                quantized.residual_subblocks,
                quantized.sm_scale,
                int(return_stats),
                0,
                0,
            )
    return (output, selected, quantized.possible_blocks) if return_stats else output


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
    if not use_w8a8:
        sparse_query_blocks, _, sparse_block_count = _sol_block_policy(
            q.device,
            q.size(2),
            k.size(2),
            dense_query_ranges,
            exact_kv_ranges,
        )
        if sparse_block_count == 0:
            dense = sageattn(q, k, v, tensor_layout="HND", sm_scale=sm_scale)
            if return_stats:
                return dense, torch.zeros(1, dtype=torch.int64, device=q.device), 0
            return dense
        del sparse_query_blocks
    quantized = prequantize_sol_sageattn.__wrapped__(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        sm_scale=sm_scale,
        dense_query_ranges=dense_query_ranges,
        exact_kv_ranges=exact_kv_ranges,
        threshold_sigma=threshold_sigma,
        residual_subblocks=residual_subblocks,
        use_w8a8=use_w8a8,
        force_dense=_force_dense,
    )
    return sol_sparse_sageattn_from_prequantized(
        quantized,
        return_stats=return_stats,
    )

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
