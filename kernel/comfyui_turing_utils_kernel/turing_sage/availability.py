"""Side-effect-free feature probes for the independently built extensions."""

from __future__ import annotations

import importlib

import torch


_ATTENTION_EXTENSION = "comfyui_turing_utils_kernel._sage_qattn_sm75"
_FUSION_EXTENSION = "comfyui_turing_utils_kernel._sage_fused_sm75"


def _module(name: str):
    try:
        return importlib.import_module(name)
    except (ImportError, OSError):
        return None


def _has_symbols(module_name: str, *symbols: str) -> bool:
    module = _module(module_name)
    return module is not None and all(hasattr(module, symbol) for symbol in symbols)


def integer_attention_device(device: torch.device) -> bool:
    return bool(
        device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(device) >= (7, 5)
    )


def available() -> bool:
    return _module(_ATTENTION_EXTENSION) is not None and _module(_FUSION_EXTENSION) is not None


def sparse_available() -> bool:
    return available() and _has_symbols(
        _ATTENTION_EXTENSION,
        "sol_sparse_online_int8_f16_attn",
    )


def sla_available() -> bool:
    return available() and _has_symbols(
        _ATTENTION_EXTENSION,
        "sla_qk_block_summaries",
        "sla_build_route_words",
        "sla_sparse_online_attn",
    )


def w8a8_available() -> bool:
    # Dense W8A8 owns a separate ABI even though it currently shares a module
    # with Sol/SLA.
    return available() and _has_symbols(_ATTENTION_EXTENSION, "quantize_v_int8_sm75")


def w8a8_varlen_available() -> bool:
    return w8a8_available() and _has_symbols(
        _ATTENTION_EXTENSION,
        "quantize_v_int8_varlen_sm75",
        "qk_int8_sv_int8_varlen_accum_f32_attn",
    )


def split_prequantization_available() -> bool:
    return w8a8_available() and _has_symbols(
        _ATTENTION_EXTENSION,
        "sol_w8a8_precompute_summaries",
        "sol_sparse_online_w8a8_prequantized_attn",
    )


def fused_qk_preprocessing_available() -> bool:
    return available() and _has_symbols(
        _FUSION_EXTENSION,
        "quant_qk_rms_rope_int8_cuda",
    )


def overlap_blend_available() -> bool:
    return available() and _has_symbols(_FUSION_EXTENSION, "overlap_blend_cuda")


def overlap_accumulate_available() -> bool:
    return available() and _has_symbols(
        _FUSION_EXTENSION,
        "overlap_accumulate_cuda",
    )


__all__ = [
    "available",
    "fused_qk_preprocessing_available",
    "integer_attention_device",
    "overlap_accumulate_available",
    "overlap_blend_available",
    "sla_available",
    "sparse_available",
    "split_prequantization_available",
    "w8a8_available",
    "w8a8_varlen_available",
]
