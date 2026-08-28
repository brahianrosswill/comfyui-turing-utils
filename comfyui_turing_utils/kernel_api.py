"""Lazy boundary to the independently installed CUDA kernel package."""

from __future__ import annotations

import importlib
from types import ModuleType


KERNEL_PACKAGE = "comfyui_turing_utils_kernel"


def load_kernel_package() -> ModuleType:
    return importlib.import_module(KERNEL_PACKAGE)


def load_kernel_extension(name: str) -> ModuleType:
    if not name or "." in name:
        raise ValueError(f"Invalid kernel extension name: {name!r}")
    return importlib.import_module(f"{KERNEL_PACKAGE}.{name}")


def load_turing_sage() -> ModuleType:
    return importlib.import_module(f"{KERNEL_PACKAGE}.turing_sage")


def kernel_extension_has_symbol(name: str, extension: str = "_C") -> bool:
    """Check the compiled ABI instead of trusting a Python wrapper or version.

    Editable installs can temporarily pair newer Python sources with an older
    extension binary.  Feature selection must therefore inspect the extension
    that will execute the operation.
    """
    try:
        module = load_kernel_extension(extension)
    except (ImportError, OSError):
        return False
    return callable(getattr(module, name, None))


def kernel_version(default: str = "0.0.0") -> str:
    try:
        package = load_kernel_package()
    except (ImportError, OSError):
        return default
    return str(getattr(package, "__version__", default))


def qk_preprocess_protocol_schema() -> int:
    """Return the fused Q/K native ABI schema, not the Python package version."""
    try:
        extension = load_kernel_extension("_sage_fused_sm75")
    except (ImportError, OSError):
        return 0
    value = getattr(extension, "qk_preprocess_protocol_schema", 0)
    return int(value) if isinstance(value, int) else 0


def attention_kernel_architectures() -> tuple[str, ...]:
    """Return architectures embedded in the installed attention extension.

    Kernel 0.31 and newer publish this metadata from the native module.  An
    empty tuple deliberately means that an older or stale extension cannot
    prove which cubins it contains.
    """
    try:
        extension = load_kernel_extension("_sage_qattn_sm75")
    except (ImportError, OSError):
        return ()
    raw = getattr(extension, "cuda_architectures", ())
    if isinstance(raw, str):
        raw = (raw,)
    if not isinstance(raw, (tuple, list)):
        return ()
    return tuple(str(architecture) for architecture in raw if architecture)


def attention_runtime_profile_schema() -> int:
    try:
        extension = load_kernel_extension("_sage_qattn_sm75")
    except (ImportError, OSError):
        return 0
    value = getattr(extension, "runtime_profile_schema", 0)
    return int(value) if isinstance(value, int) else 0


__all__ = [
    "KERNEL_PACKAGE",
    "attention_kernel_architectures",
    "attention_runtime_profile_schema",
    "kernel_extension_has_symbol",
    "kernel_version",
    "load_kernel_extension",
    "load_kernel_package",
    "load_turing_sage",
    "qk_preprocess_protocol_schema",
]
