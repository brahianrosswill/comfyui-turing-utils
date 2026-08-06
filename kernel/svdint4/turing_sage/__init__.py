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


def sageattn_sage1(*args, **kwargs):
    from .core import sageattn_sage1 as implementation

    return implementation(*args, **kwargs)


def sageattn_sage2(*args, **kwargs):
    from .core import sageattn_sage2 as implementation

    return implementation(*args, **kwargs)


def sageattn_hybrid(*args, **kwargs):
    from .core import sageattn_hybrid as implementation

    return implementation(*args, **kwargs)


sage_ = sageattn_hybrid


def preflight(device: torch.device, variant: str = "sage_") -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Turing Sage requires CUDA, got {device}")
    if torch.cuda.get_device_capability(device) != (7, 5):
        raise RuntimeError(f"Turing Sage requires sm75, got {device}")
    if not available():
        raise RuntimeError("the bundled Turing Sage extensions are not built")

    implementations = {
        "sage2": (sageattn_sage2, {"smooth_q": True, "smooth_k": True}),
        "sage1": (sageattn_sage1, {"smooth_k": True}),
        "sage_": (sageattn_hybrid, {"smooth_k": False}),
    }
    if variant not in implementations:
        raise ValueError(f"unknown bundled Turing Sage variant: {variant}")
    implementation, variant_kwargs = implementations[variant]

    with torch.inference_mode(), torch.cuda.device(device):
        for dtype in (torch.float16, torch.bfloat16):
            for head_dim in (64, 96, 128):
                q_values = torch.arange(4 * 65 * head_dim, device=device, dtype=torch.float32)
                kv_values = torch.arange(2 * 73 * head_dim, device=device, dtype=torch.float32)
                q = (((q_values % 29) - 14) / 32).reshape(1, 4, 65, head_dim).to(dtype)
                k = ((((kv_values * 3) % 31) - 15) / 32).reshape(1, 2, 73, head_dim).to(dtype)
                v = ((((kv_values * 5) % 37) - 18) / 32).reshape(1, 2, 73, head_dim).to(dtype)
                output = implementation(q, k, v, tensor_layout="HND", **variant_kwargs)
                reference = torch.nn.functional.scaled_dot_product_attention(
                    q.float(), k.float(), v.float(), enable_gqa=True
                )
                if (
                    output.dtype != dtype
                    or output.shape != q.shape
                    or not torch.isfinite(output).all()
                    or not torch.allclose(output.float(), reference, rtol=0.08, atol=0.05)
                ):
                    raise RuntimeError(
                        f"Turing {variant} {dtype} D={head_dim} self-test failed"
                    )

        # Varlen uses a separate fused path below 512 query tokens and the
        # INT8/FP16-PV path at 512 or above. Exercise both with BF16 V so a
        # stale extension cannot pass startup and fail later in a workflow.
        if variant != "sage_":
            torch.cuda.synchronize(device)
            return
        for q_len, k_len in ((65, 73), (512, 513)):
            cu_q = torch.tensor((0, q_len), dtype=torch.int32, device=device)
            cu_k = torch.tensor((0, k_len), dtype=torch.int32, device=device)
            q_values = torch.arange(q_len * 4 * 64, dtype=torch.float32, device=device)
            kv_values = torch.arange(k_len * 2 * 64, dtype=torch.float32, device=device)
            q = (((q_values % 29) - 14) / 16).reshape(q_len, 4, 64).to(torch.bfloat16)
            k = ((((kv_values * 3) % 31) - 15) / 16).reshape(k_len, 2, 64).to(torch.bfloat16)
            v = ((((kv_values * 5) % 37) - 18) / 16).reshape(k_len, 2, 64).to(torch.bfloat16)
            output = sageattn_varlen(
                q,
                k,
                v,
                cu_q,
                cu_k,
                q_len,
                k_len,
                smooth_k=False,
                variant="sage_",
            )
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
                raise RuntimeError(
                    f"Turing Sage BF16 varlen max_q={q_len} self-test failed"
                )
        torch.cuda.synchronize(device)


__all__ = [
    "available",
    "preflight",
    "sageattn",
    "sageattn_sage1",
    "sageattn_sage2",
    "sageattn_hybrid",
    "sage_",
    "sageattn_varlen",
]
