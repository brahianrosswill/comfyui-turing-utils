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
    return hasattr(module, "sol_sparse_threshold_f16_attn")


def sageattn(*args, **kwargs):
    """Run the stable bundled SM75 Sage attention implementation."""
    from .core import sageattn as implementation

    return implementation(*args, **kwargs)


def sageattn_varlen(*args, **kwargs):
    from .core import sageattn_varlen as implementation

    return implementation(*args, **kwargs)


def sol_sparse_sageattn(*args, **kwargs):
    from .core import sol_sparse_sageattn as implementation

    return implementation(*args, **kwargs)


def sol_sparse_route_selected(*args, **kwargs):
    from .core import sol_sparse_route_selected as implementation

    return implementation(*args, **kwargs)


def preflight_sparse(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Turing sparse attention requires CUDA, got {device}")
    if torch.cuda.get_device_capability(device) != (7, 5):
        raise RuntimeError(f"Turing sparse attention requires sm75, got {device}")
    if not sparse_available():
        raise RuntimeError("the experimental Turing sparse extension is not built")

    with torch.cuda.device(device):
        q_values = torch.arange(4 * 129 * 128, device=device, dtype=torch.float32)
        kv_values = torch.arange(2 * 151 * 128, device=device, dtype=torch.float32)
        q = (((q_values % 29) - 14) / 16).reshape(1, 4, 129, 128).to(torch.bfloat16)
        k = ((((kv_values * 3) % 31) - 15) / 16).reshape(1, 2, 151, 128).to(torch.bfloat16)
        v = ((((kv_values * 5) % 37) - 18) / 16).reshape_as(k).to(torch.bfloat16)
        output = sol_sparse_sageattn(q, k, v, prefix_tokens=151)
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), enable_gqa=True
        )
        if (
            output.dtype != torch.bfloat16
            or output.shape != q.shape
            or not torch.isfinite(output).all()
            or not torch.allclose(output.float(), reference, rtol=0.08, atol=0.06)
        ):
            raise RuntimeError("Turing sparse attention BF16 self-test failed")
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


__all__ = [
    "available",
    "preflight",
    "preflight_sparse",
    "sageattn",
    "sageattn_varlen",
    "sol_sparse_sageattn",
    "sparse_available",
]
