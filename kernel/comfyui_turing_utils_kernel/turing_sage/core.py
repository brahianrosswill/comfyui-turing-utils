from functools import wraps
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
def sol_sparse_sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tensor_layout: str = "HND",
    sm_scale: Optional[float] = None,
    prefix_tokens: int = 0,
    threshold_sigma: float = 1.0,
    local_block_radius: int = 1,
    topology_start_tokens: int = 0,
    topology_tokens: int = 0,
    tokens_per_frame: int = 0,
    temporal_neighbor_frames: int = 0,
    return_route: bool = False,
):
    """Experimental SM75 Sol-style adaptive threshold sparse attention."""
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
    prefix_tokens = int(prefix_tokens)
    if prefix_tokens < 0 or prefix_tokens > min(q.size(2), k.size(2)):
        raise ValueError("prefix_tokens is outside the shared Q/K sequence")
    if prefix_tokens and q.size(2) != k.size(2):
        raise ValueError("prefix Query splitting requires equal Q/K sequence lengths")

    if prefix_tokens == q.size(2):
        dense = sageattn(q, k, v, tensor_layout="HND", sm_scale=scale)
        if not return_route:
            return dense
        key_blocks = (k.size(2) + 63) // 64
        route_words = (key_blocks + 15) // 16
        route = torch.empty(
            (q.size(0), q.size(1), 0, route_words),
            dtype=torch.int32,
            device=q.device,
        )
        return dense, route

    output = torch.empty_like(q)
    k_int8, k_scale = quantize_key_per_block(k, tensor_layout="HND")
    if prefix_tokens:
        q_prefix = q[:, :, :prefix_tokens]
        prefix_output = output[:, :, :prefix_tokens]
        if prefix_tokens < 64:
            q_prefix = torch.nn.functional.pad(
                q_prefix, (0, 0, 0, 64 - prefix_tokens)
            )
        q_prefix_int8, q_prefix_scale = quantize_query_per_warp(
            q_prefix, tensor_layout="HND"
        )
        if prefix_tokens < 64:
            padded_prefix_output = torch.empty_like(q_prefix)
            sageattn_prequantized(
                q_prefix_int8,
                q_prefix_scale,
                k_int8,
                k_scale,
                v,
                tensor_layout="HND",
                sm_scale=scale,
                output=padded_prefix_output,
            )
            prefix_output.copy_(padded_prefix_output[:, :, :prefix_tokens])
            del padded_prefix_output
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

    q_sparse = q[:, :, prefix_tokens:]
    sparse_output = output[:, :, prefix_tokens:]
    q_int8, q_scale = quantize_query_per_warp(q_sparse, tensor_layout="HND")
    route = sm75_compile.sol_sparse_threshold_int8_f16_attn(
        q_sparse,
        k,
        q_int8,
        k_int8,
        v,
        sparse_output,
        q_scale,
        k_scale,
        int(prefix_tokens),
        float(threshold_sigma),
        int(local_block_radius),
        int(topology_start_tokens),
        int(topology_tokens),
        int(tokens_per_frame),
        int(temporal_neighbor_frames),
        int(prefix_tokens),
        scale,
    )
    return (output, route) if return_route else output


def sol_sparse_route_selected(route: torch.Tensor) -> int:
    """Synchronize once and return the selected-block count for debug logging."""
    if not route.is_cuda or route.dtype != torch.int32 or route.ndim != 4:
        raise ValueError("route must be a four-dimensional CUDA int32 tensor")
    return int(sm75_compile.sol_sparse_route_selected(route).item())


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
