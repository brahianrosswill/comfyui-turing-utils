from __future__ import annotations

import logging

import torch

try:
    from .turing_ops import (
        backend_available,
        is_supported_turing_device,
        preflight_kitchen,
        preflight_w4a8,
        register_backend,
    )
except ImportError:
    from turing_ops import (
        backend_available,
        is_supported_turing_device,
        preflight_kitchen,
        preflight_w4a8,
        register_backend,
    )


LOG = logging.getLogger("comfyui-svdint4")


def _explicit_dtype_override() -> bool:
    try:
        import comfy.model_management as model_management
    except ImportError:
        return False
    args = getattr(model_management, "args", None)
    return bool(
        getattr(model_management, "FORCE_FP32", False)
        or getattr(args, "fp32_unet", False)
        or getattr(args, "fp64_unet", False)
        or getattr(args, "fp16_unet", False)
        or getattr(args, "fp8_e4m3fn_unet", False)
        or getattr(args, "fp8_e5m2_unet", False)
        or getattr(args, "fp8_e8m0fnu_unet", False)
    )


def _preflight_turing(summary, device: torch.device, attention_backend: str) -> None:
    if summary.w4a4 or summary.w4a8 or summary.w8a8:
        try:
            import comfy_kitchen
        except ImportError as exc:
            raise RuntimeError("Turing ConvRot execution requires comfy-kitchen") from exc

        cuda_status = comfy_kitchen.list_backends().get("cuda", {})
        if not cuda_status.get("available") or cuda_status.get("disabled"):
            reason = cuda_status.get("unavailable_reason") or "disabled"
            raise RuntimeError(f"Kitchen CUDA backend is unavailable on Turing: {reason}")
        capabilities = set(cuda_status.get("capabilities", ()))
        if (summary.w4a4 or summary.w4a8) and "convrot_w4a4_linear" not in capabilities:
            raise RuntimeError("Kitchen ConvRot W4 support is unavailable on Turing")
        if summary.w8a8 and "int8_linear" not in capabilities:
            raise RuntimeError("Kitchen W8A8 support is unavailable on Turing")

        preflight_kitchen(device, bool(summary.w4a4), bool(summary.w8a8))
        if summary.w4a8:
            if not register_backend() or not backend_available():
                raise RuntimeError("the bundled Turing W4A8 backend could not be registered")
            preflight_w4a8(device)

    if attention_backend in {"auto", "sage_attn"}:
        try:
            from .turing_attention import available, preflight
        except ImportError:
            from turing_attention import available, preflight
        if not available():
            raise RuntimeError("the bundled Turing SageAttention2 extensions are unavailable")
        preflight(device)


def select_compute_dtype(
    model_config,
    device: torch.device,
    summary,
    attention_backend: str = "auto",
) -> torch.dtype | None:
    """Select BF16 storage/compute without changing model-internal accumulation rules."""
    if model_config is None or _explicit_dtype_override():
        return None
    supported = tuple(getattr(model_config, "supported_inference_dtypes", ()) or ())
    if torch.bfloat16 not in supported:
        return None
    if device.type != "cuda" or not torch.cuda.is_available():
        return None

    index = device.index if device.index is not None else torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    if capability == (7, 5) and not is_supported_turing_device(device):
        LOG.warning(
            "BF16 forcing is disabled for %s because the bundled SM75 kernels require a Turing GPU with tensor cores",
            torch.cuda.get_device_name(index),
        )
        return None
    if capability == (7, 5):
        _preflight_turing(summary, device, attention_backend)
        LOG.info("Using BF16 activation storage with bundled Turing kernels")
    else:
        LOG.info("Using the model's declared BF16 inference mode")
    return torch.bfloat16
