from __future__ import annotations

import functools
from typing import Any, Callable

import torch


def is_mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def retry_on_oom(
    func: Callable[..., Any],
    *args: Any,
    debug: Any = None,
    operation_name: str = "operation",
    **kwargs: Any,
) -> Any:
    del debug, operation_name
    try:
        return func(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return func(*args, **kwargs)


def oom_retry(operation_name: str = "operation") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return retry_on_oom(func, *args, operation_name=operation_name, **kwargs)

        return wrapped

    return decorator
