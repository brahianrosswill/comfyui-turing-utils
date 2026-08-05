from __future__ import annotations

import importlib

import torch


def available() -> bool:
    try:
        importlib.import_module("svdint4._sage_qattn_sm75")
        importlib.import_module("svdint4._sage_fused_sm75")
    except (ImportError, OSError):
        return False
    return True


def sageattn(*args, **kwargs):
    from .core import sageattn as implementation

    return implementation(*args, **kwargs)


def sageattn_varlen(*args, **kwargs):
    from .core import sageattn_varlen as implementation

    return implementation(*args, **kwargs)


def preflight(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Turing SageAttention2 requires CUDA, got {device}")
    if torch.cuda.get_device_capability(device) != (7, 5):
        raise RuntimeError(f"Turing SageAttention2 requires sm75, got {device}")
    if not available():
        raise RuntimeError("the bundled Turing SageAttention2 extensions are not built")

    with torch.inference_mode(), torch.cuda.device(device):
        for dtype in (torch.float16, torch.bfloat16):
            for head_dim in (64, 128):
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
                    raise RuntimeError(f"Turing SageAttention2 {dtype} D={head_dim} self-test failed")
        torch.cuda.synchronize(device)


__all__ = ["available", "preflight", "sageattn", "sageattn_varlen"]
