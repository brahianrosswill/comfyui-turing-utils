from __future__ import annotations

import importlib

import torch


def available() -> bool:
    try:
        importlib.import_module("comfyui_turing_utils_kernel._sage_qattn_sm75")
        importlib.import_module("comfyui_turing_utils_kernel._sage_fused_sm75")
    except (ImportError, OSError):
        return False
    return True


def sparse_available() -> bool:
    if not available():
        return False
    try:
        module = importlib.import_module("comfyui_turing_utils_kernel._sage_qattn_sm75")
    except (ImportError, OSError):
        return False
    return hasattr(module, "sol_sparse_online_int8_f16_attn")


def w8a8_available() -> bool:
    # Dense W8A8 is a production backend with its own ABI.  Do not couple its
    # availability to the Sol entry point merely because both
    # kernels currently share one extension module.
    if not available():
        return False
    try:
        module = importlib.import_module("comfyui_turing_utils_kernel._sage_qattn_sm75")
    except (ImportError, OSError):
        return False
    return hasattr(module, "quantize_v_int8_sm75")


def w8a8_varlen_available() -> bool:
    """Return whether the packed variable-length W8A8 ABI is installed."""
    if not w8a8_available():
        return False
    try:
        module = importlib.import_module("comfyui_turing_utils_kernel._sage_qattn_sm75")
    except (ImportError, OSError):
        return False
    return all(
        hasattr(module, name)
        for name in (
            "quantize_v_int8_varlen_sm75",
            "qk_int8_sv_int8_varlen_accum_f32_attn",
        )
    )


def split_prequantization_available() -> bool:
    if not w8a8_available():
        return False
    try:
        module = importlib.import_module("comfyui_turing_utils_kernel._sage_qattn_sm75")
    except (ImportError, OSError):
        return False
    return all(
        hasattr(module, name)
        for name in (
            "sol_w8a8_precompute_summaries",
            "sol_sparse_online_w8a8_prequantized_attn",
        )
    )


def fused_qk_preprocessing_available() -> bool:
    if not available():
        return False
    try:
        module = importlib.import_module(
            "comfyui_turing_utils_kernel._sage_fused_sm75"
        )
    except (ImportError, OSError):
        return False
    return hasattr(module, "quant_qk_rms_rope_int8_cuda")


def overlap_blend_available() -> bool:
    if not available():
        return False
    try:
        module = importlib.import_module(
            "comfyui_turing_utils_kernel._sage_fused_sm75"
        )
    except (ImportError, OSError):
        return False
    return hasattr(module, "overlap_blend_cuda")


def overlap_blend_compiled(window_values, local_indices, weights):
    from .custom_ops import overlap_blend_op

    return overlap_blend_op(window_values, local_indices, weights)


def sageattn(*args, **kwargs):
    """Run the stable bundled SM75 Sage attention implementation."""
    from .core import sageattn as implementation

    return implementation(*args, **kwargs)


def sageattn_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: float | None = None,
):
    from .custom_ops import sage_attention

    return sage_attention(
        q,
        k,
        v,
        tensor_layout,
        bool(is_causal),
        float(sm_scale) if sm_scale is not None else -1.0,
    )


def sageattn_varlen(*args, **kwargs):
    from .core import sageattn_varlen as implementation

    return implementation(*args, **kwargs)


def sageattn_varlen_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    is_causal: bool = False,
    sm_scale: float | None = None,
):
    from .custom_ops import sage_attention_varlen

    return sage_attention_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        bool(is_causal),
        float(sm_scale) if sm_scale is not None else -1.0,
    )


def sol_sparse_sageattn(*args, **kwargs):
    from .core import sol_sparse_sageattn as implementation

    return implementation(*args, **kwargs)


def sol_sparse_sageattn_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dense_query_ranges=(),
    exact_kv_ranges=(),
    threshold_sigma: float = 1.0,
    residual_subblocks: int = 1,
    use_w8a8: bool = False,
    sm_scale: float | None = None,
    key_tile_tokens: int = 0,
    rotate_qk: bool = True,
    stabilize_k: bool = True,
):
    from .custom_ops import sol_attention

    dense_query_ranges = tuple(dense_query_ranges)
    exact_kv_ranges = tuple(exact_kv_ranges)
    return sol_attention(
        q,
        k,
        v,
        [int(item[0]) for item in dense_query_ranges],
        [int(item[1]) for item in dense_query_ranges],
        [int(item[0]) for item in exact_kv_ranges],
        [int(item[1]) for item in exact_kv_ranges],
        float(threshold_sigma),
        int(residual_subblocks),
        bool(use_w8a8),
        float(sm_scale) if sm_scale is not None else -1.0,
        int(key_tile_tokens),
        bool(rotate_qk),
        bool(stabilize_k),
    )


def w8a8attn(*args, **kwargs):
    from .core import w8a8attn as implementation

    return implementation(*args, **kwargs)


def w8a8attn_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: float | None = None,
    key_tile_tokens: int = 0,
    rotate_qk: bool = True,
    stabilize_k: bool = True,
):
    from .custom_ops import w8a8_attention

    return w8a8_attention(
        q,
        k,
        v,
        tensor_layout,
        bool(is_causal),
        float(sm_scale) if sm_scale is not None else -1.0,
        int(key_tile_tokens),
        bool(rotate_qk),
        bool(stabilize_k),
    )


def w8a8attn_varlen(*args, **kwargs):
    from .core import w8a8attn_varlen as implementation

    return implementation(*args, **kwargs)


def w8a8attn_varlen_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    is_causal: bool = False,
    sm_scale: float | None = None,
    rotate_qk: bool = True,
):
    from .custom_ops import w8a8_attention_varlen

    return w8a8_attention_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        bool(is_causal),
        float(sm_scale) if sm_scale is not None else -1.0,
        bool(rotate_qk),
    )


def prequantize_sageattn(*args, **kwargs):
    from .core import prequantize_sageattn as implementation

    return implementation(*args, **kwargs)


def prequantize_rms_rope_qk(*args, **kwargs):
    from .core import prequantize_rms_rope_qk as implementation

    return implementation(*args, **kwargs)


def prequantize_sageattn_from_qk(*args, **kwargs):
    from .core import prequantize_sageattn_from_qk as implementation

    return implementation(*args, **kwargs)


def sageattn_from_prequantized(*args, **kwargs):
    from .core import sageattn_from_prequantized as implementation

    return implementation(*args, **kwargs)


def prequantize_sol_sageattn(*args, **kwargs):
    from .core import prequantize_sol_sageattn as implementation

    return implementation(*args, **kwargs)


def prequantize_sol_sageattn_from_qk(*args, **kwargs):
    from .core import prequantize_sol_sageattn_from_qk as implementation

    return implementation(*args, **kwargs)


def sol_sparse_sageattn_from_prequantized(*args, **kwargs):
    from .core import sol_sparse_sageattn_from_prequantized as implementation

    return implementation(*args, **kwargs)


def preflight_sparse(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Turing sparse attention requires CUDA, got {device}")
    if torch.cuda.get_device_capability(device) != (7, 5):
        raise RuntimeError(f"Turing sparse attention requires sm75, got {device}")
    if not sparse_available():
        raise RuntimeError("the Turing Sol sparse extension is not built")

    with torch.inference_mode(), torch.cuda.device(device):
        for head_dim in (64, 128):
            q_values = torch.arange(
                4 * 129 * head_dim, device=device, dtype=torch.float32
            )
            kv_values = torch.arange(
                2 * 151 * head_dim, device=device, dtype=torch.float32
            )
            q = (((q_values % 29) - 14) / 16).reshape(
                1, 4, 129, head_dim
            ).to(torch.bfloat16)
            k = ((((kv_values * 3) % 31) - 15) / 16).reshape(
                1, 2, 151, head_dim
            ).to(torch.bfloat16)
            v = ((((kv_values * 5) % 37) - 18) / 16).reshape_as(k).to(
                torch.bfloat16
            )
            output = sol_sparse_sageattn(q, k, v, threshold_sigma=-1000.0)
            reference = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), enable_gqa=True
            )
            if (
                output.dtype != torch.bfloat16
                or output.shape != q.shape
                or not torch.isfinite(output).all()
                or not torch.allclose(output.float(), reference, rtol=0.08, atol=0.06)
            ):
                raise RuntimeError(
                    f"Turing sparse attention BF16 D={head_dim} self-test failed"
                )
        torch.cuda.synchronize(device)


def preflight_w8a8(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Turing W8A8 attention requires CUDA, got {device}")
    if torch.cuda.get_device_capability(device) != (7, 5):
        raise RuntimeError(f"Turing W8A8 attention requires sm75, got {device}")
    if not w8a8_available():
        raise RuntimeError("the Turing W8A8 attention extension is not built")

    with torch.inference_mode(), torch.cuda.device(device):
        for head_dim in (64, 128):
            q_values = torch.arange(
                4 * 129 * head_dim, device=device, dtype=torch.float32
            )
            kv_values = torch.arange(
                2 * 151 * head_dim, device=device, dtype=torch.float32
            )
            q = (((q_values % 29) - 14) / 16).reshape(
                1, 4, 129, head_dim
            ).to(torch.bfloat16)
            k = ((((kv_values * 3) % 31) - 15) / 16).reshape(
                1, 2, 151, head_dim
            ).to(torch.bfloat16)
            v = ((((kv_values * 5) % 37) - 18) / 16).reshape_as(k).to(
                torch.bfloat16
            )
            output = w8a8attn(q, k, v)
            reference = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), enable_gqa=True
            )
            if (
                output.dtype != torch.bfloat16
                or output.shape != q.shape
                or not torch.isfinite(output).all()
                or not torch.allclose(output.float(), reference, rtol=0.12, atol=0.09)
            ):
                raise RuntimeError(
                    f"Turing W8A8 attention BF16 D={head_dim} self-test failed"
                )
        torch.cuda.synchronize(device)


def preflight(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Turing Sage requires CUDA, got {device}")
    if torch.cuda.get_device_capability(device) != (7, 5):
        raise RuntimeError(f"Turing Sage requires sm75, got {device}")
    if not available():
        raise RuntimeError("the bundled Turing Sage extensions are not built")

    with torch.inference_mode(), torch.cuda.device(device):
        for dtype in (torch.float16, torch.bfloat16):
            for head_dim in (64, 96, 128):
                q_values = torch.arange(4 * 65 * head_dim, device=device, dtype=torch.float32)
                kv_values = torch.arange(2 * 73 * head_dim, device=device, dtype=torch.float32)
                q = (((q_values % 29) - 14) / 32).reshape(1, 4, 65, head_dim).to(dtype)
                k = ((((kv_values * 3) % 31) - 15) / 32).reshape(1, 2, 73, head_dim).to(dtype)
                v = ((((kv_values * 5) % 37) - 18) / 32).reshape(1, 2, 73, head_dim).to(dtype)
                output = sageattn(q, k, v, tensor_layout="HND", smooth_k=False)
                reference = torch.nn.functional.scaled_dot_product_attention(
                    q.float(), k.float(), v.float(), enable_gqa=True
                )
                if (
                    output.dtype != dtype
                    or output.shape != q.shape
                    or not torch.isfinite(output).all()
                    or not torch.allclose(output.float(), reference, rtol=0.08, atol=0.05)
                ):
                    raise RuntimeError(f"Turing Sage {dtype} D={head_dim} self-test failed")

        # Exercise both the short exact facade and the large-sequence INT8 path.
        for q_len, k_len in ((65, 73), (512, 513)):
            cu_q = torch.tensor((0, q_len), dtype=torch.int32, device=device)
            cu_k = torch.tensor((0, k_len), dtype=torch.int32, device=device)
            q_values = torch.arange(q_len * 4 * 64, dtype=torch.float32, device=device)
            kv_values = torch.arange(k_len * 2 * 64, dtype=torch.float32, device=device)
            q = (((q_values % 29) - 14) / 16).reshape(q_len, 4, 64).to(torch.bfloat16)
            k = ((((kv_values * 3) % 31) - 15) / 16).reshape(k_len, 2, 64).to(torch.bfloat16)
            v = ((((kv_values * 5) % 37) - 18) / 16).reshape(k_len, 2, 64).to(torch.bfloat16)
            output = sageattn_varlen(q, k, v, cu_q, cu_k, q_len, k_len, smooth_k=False)
            reference = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0).float(),
                k.transpose(0, 1).unsqueeze(0).float(),
                v.transpose(0, 1).unsqueeze(0).float(),
                enable_gqa=True,
            ).squeeze(0).transpose(0, 1)
            if (
                output.dtype != torch.bfloat16
                or output.shape != q.shape
                or not torch.isfinite(output).all()
                or not torch.allclose(output.float(), reference, rtol=0.08, atol=0.06)
            ):
                raise RuntimeError(f"Turing Sage BF16 varlen max_q={q_len} self-test failed")
        torch.cuda.synchronize(device)


def run_attention_correctness_gate(*args, **kwargs):
    """Run the explicit Sol-vs-dense numerical release gate."""
    from .correctness import run_attention_correctness_gate as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "available",
    "fused_qk_preprocessing_available",
    "overlap_blend_available",
    "overlap_blend_compiled",
    "prequantize_sageattn",
    "prequantize_sol_sageattn",
    "preflight",
    "preflight_sparse",
    "preflight_w8a8",
    "run_attention_correctness_gate",
    "sageattn",
    "sageattn_compiled",
    "sageattn_from_prequantized",
    "sageattn_varlen",
    "sageattn_varlen_compiled",
    "sol_sparse_sageattn",
    "sol_sparse_sageattn_compiled",
    "sol_sparse_sageattn_from_prequantized",
    "split_prequantization_available",
    "sparse_available",
    "w8a8attn",
    "w8a8attn_compiled",
    "w8a8attn_varlen",
    "w8a8attn_varlen_compiled",
    "w8a8_available",
    "w8a8_varlen_available",
]
