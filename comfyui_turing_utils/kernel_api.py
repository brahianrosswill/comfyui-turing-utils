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


__all__ = [
    "KERNEL_PACKAGE",
    "kernel_extension_has_symbol",
    "kernel_version",
    "load_kernel_extension",
    "load_kernel_package",
    "load_turing_sage",
]
