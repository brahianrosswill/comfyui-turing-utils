"""SeedVR2 support helpers for the ComfyUI SVDInt4 plugin.

This module keeps only the single-process inference surface needed by the
SeedVR2 node. Distributed functions are intentionally no-ops so the DiT
and VAE definitions can stay close to the official architecture without pulling
in the original runtime framework.
"""

from __future__ import annotations

import functools
import importlib.machinery
import logging
import os
import sys
import types
from math import pi
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, broadcast_tensors, nn


def ensure_triton_compat():
    if "triton.ops.matmul_perf_model" in sys.modules:
        return
    try:
        from triton.ops.matmul_perf_model import early_config_prune  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass
    if "triton.ops" not in sys.modules:
        sys.modules["triton.ops"] = types.ModuleType("triton.ops")
    matmul_perf = types.ModuleType("triton.ops.matmul_perf_model")
    matmul_perf.early_config_prune = lambda configs, *a, **kw: configs
    matmul_perf.estimate_matmul_time = lambda *a, **kw: 0.0
    sys.modules["triton.ops"].matmul_perf_model = matmul_perf
    sys.modules["triton.ops.matmul_perf_model"] = matmul_perf


def ensure_flash_attn_safe():
    if "flash_attn" in sys.modules:
        return
    try:
        import flash_attn  # noqa: F401
    except (ImportError, OSError):
        stub = types.ModuleType("flash_attn")
        stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
        stub.__file__ = None
        stub.__path__ = []
        stub.__loader__ = None
        stub.flash_attn_func = None
        stub.flash_attn_varlen_func = None
        sys.modules["flash_attn"] = stub


def ensure_xformers_flash_compat():
    if "xformers._C_flashattention" in sys.modules:
        return
    try:
        from xformers import _C_flashattention  # noqa: F401
    except (ImportError, OSError):
        class _FailingStub(types.ModuleType):
            def __getattr__(self, name):
                raise ImportError("_C_flashattention unavailable")
        stub = _FailingStub("xformers._C_flashattention")
        stub.__spec__ = importlib.machinery.ModuleSpec("xformers._C_flashattention", None)
        stub.__file__ = None
        stub.__path__ = []
        stub.__loader__ = None
        sys.modules["xformers._C_flashattention"] = stub


def ensure_bitsandbytes_safe():
    if "bitsandbytes" in sys.modules:
        return
    try:
        import bitsandbytes  # noqa: F401
    except (ImportError, OSError, RuntimeError, ValueError):
        stub = types.ModuleType("bitsandbytes")
        stub.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes", None)
        stub.__file__ = None
        stub.__path__ = []
        stub.__version__ = "0.0.0"
        sys.modules["bitsandbytes"] = stub


ensure_triton_compat()
ensure_flash_attn_safe()
ensure_xformers_flash_compat()
ensure_bitsandbytes_safe()


class Cache:
    def __init__(self, disable: bool = False, prefix: str = "", cache: Optional[dict] = None):
        self.cache = cache if cache is not None else {}
        self.disable = disable
        self.prefix = prefix

    def __call__(self, name: str, init_fn: Callable[[], Any]) -> Any:
        if self.disable:
            return init_fn()
        name = self.prefix + name
        if name not in self.cache:
            self.cache[name] = init_fn()
        return self.cache[name]

    def namespace(self, namespace: str):
        return Cache(disable=self.disable, prefix=self.prefix + namespace + ".", cache=self.cache)

    def get(self, name: str):
        return self.cache[self.prefix + name]


def get_global_rank() -> int:
    return 0


def get_local_rank() -> int:
    return 0


def get_world_size() -> int:
    return 1


def is_mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_sequence_parallel_group():
    return None


def get_sequence_parallel_rank() -> int:
    return 0


def get_sequence_parallel_world_size() -> int:
    return 1


def barrier_if_distributed():
    return None


def init_torch(*args, **kwargs):
    del args, kwargs
    return None


def convert_to_ddp(model, *args, **kwargs):
    del args, kwargs
    return model


def slice_inputs(x: Tensor, dim: int, padding: bool = True):
    del dim, padding
    return x


def gather_outputs(
    x: Tensor,
    dim: Optional[int] = None,
    *,
    gather_dim: Optional[int] = None,
    padding_dim: Optional[int] = None,
    unpad_shape: Optional[Tensor] = None,
    cache: Optional[Cache] = None,
    grad_scale: Optional[bool] = False,
    remove_padding: int = 0,
):
    del dim, gather_dim, padding_dim, unpad_shape, cache, grad_scale, remove_padding
    return x


def gather_seq_scatter_heads_qkv(qkv_tensor: Tensor, *, seq_dim: int, qkv_shape: Optional[Tensor] = None, cache: Cache = Cache(disable=True), restore_shape: bool = True):
    del seq_dim, qkv_shape, cache, restore_shape
    return qkv_tensor


def gather_heads_scatter_seq(x: Tensor, *, seq_dim: int, head_dim: int, seq_shape: Optional[Tensor] = None, cache: Cache = Cache(disable=True), restore_shape: bool = True):
    del seq_dim, head_dim, seq_shape, cache, restore_shape
    return x


def gather_seq_scatter_heads(x: Tensor, *, seq_dim: int, head_dim: int, seq_shape: Optional[Tensor] = None, cache: Cache = Cache(disable=True), restore_shape: bool = True):
    del seq_dim, head_dim, seq_shape, cache, restore_shape
    return x


def scatter_heads(x: Tensor, *, head_dim: int):
    del head_dim
    return x


def gather_heads(x: Tensor, *, head_dim: int):
    del head_dim
    return x


def remove_seqeunce_parallel_padding(x: Tensor, dim: int, unpad_dim_size: int):
    del dim, unpad_dim_size
    return x


def remove_sequence_parallel_padding(x: Tensor, dim: int, unpad_dim_size: int):
    return remove_seqeunce_parallel_padding(x, dim, unpad_dim_size)


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name or "seedvr2")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def safe_pad_operation(x: torch.Tensor, padding: Union[Tuple[int, ...], int], mode: str = "constant", value: float = 0.0) -> torch.Tensor:
    if mode in ("replicate", "reflect", "circular"):
        try:
            return F.pad(x, padding, mode=mode, value=value)
        except RuntimeError as e:
            if "not implemented for 'Half'" not in str(e):
                raise
            return F.pad(x.float(), padding, mode=mode, value=value).to(x.dtype)
    return F.pad(x, padding, mode=mode, value=value)


def safe_interpolate_operation(x: torch.Tensor, size: Optional[Union[int, Tuple[int, ...]]] = None, scale_factor: Optional[Union[float, Tuple[float, ...]]] = None, mode: str = "nearest", align_corners: Optional[bool] = None, recompute_scale_factor: Optional[bool] = None) -> torch.Tensor:
    kwargs = dict(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    if mode in ("bilinear", "bicubic", "trilinear"):
        try:
            return F.interpolate(x, **kwargs)
        except RuntimeError as e:
            if "not implemented for 'Half'" not in str(e) and "compute_indices_weights" not in str(e):
                raise
            return F.interpolate(x.float(), **kwargs).to(x.dtype)
    return F.interpolate(x, **kwargs)


def ensure_float32_precision(tensor: torch.Tensor, force_float32: bool = True) -> Tuple[torch.Tensor, torch.dtype]:
    original_dtype = tensor.dtype
    if force_float32 and original_dtype not in (torch.float32, torch.float64):
        return tensor.float(), original_dtype
    return tensor, original_dtype


def retry_on_oom(func: Callable[..., Any], *args: Any, debug: Any = None, operation_name: str = "operation", **kwargs: Any) -> Any:
    del debug, operation_name
    try:
        return func(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return func(*args, **kwargs)


def oom_retry(operation_name: str = "operation"):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return retry_on_oom(func, *args, operation_name=operation_name, **kwargs)
        return wrapped
    return decorator


try:
    from torch.amp import autocast as _torch_autocast
    def _no_autocast(fn):
        return _torch_autocast("cuda", enabled=False)(fn)
except ImportError:
    from torch.cuda.amp import autocast as _cuda_autocast
    def _no_autocast(fn):
        return _cuda_autocast(enabled=False)(fn)


def rotate_half(tensor: Tensor) -> Tensor:
    if tensor.shape[-1] % 2 != 0:
        raise ValueError(f"Rotary embedding requires an even feature dimension, got {tensor.shape[-1]}")
    pairs = tensor.view(*tensor.shape[:-1], tensor.shape[-1] // 2, 2)
    first, second = pairs.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(start_dim=-2)


@_no_autocast
def apply_rotary_emb(freqs: Tensor, tensor: Tensor, start_index: int = 0, scale: Union[float, Tensor] = 1.0, seq_dim: int = -2) -> Tensor:
    if tensor.ndim == 3:
        seq_len = tensor.shape[seq_dim]
        freqs = freqs[-seq_len:].to(tensor)
    else:
        freqs = freqs.to(device=tensor.device, dtype=tensor.dtype)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    if rot_dim > tensor.shape[-1]:
        raise ValueError(f"Feature dimension {tensor.shape[-1]} is too small for rotary dimension {rot_dim}")
    while freqs.ndim < tensor.ndim:
        freqs = freqs.unsqueeze(0)
    left = tensor[..., :start_index]
    middle = tensor[..., start_index:end_index]
    right = tensor[..., end_index:]
    middle = (middle * freqs.cos() * scale) + (rotate_half(middle) * freqs.sin() * scale)
    return torch.cat((left, middle, right), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, custom_freqs: Optional[Tensor] = None, freqs_for="lang", theta: float = 10000, max_freq: float = 10, num_freqs: int = 1, learned_freq: bool = False, use_xpos: bool = False, xpos_scale_base: int = 512, interpolate_factor: float = 1.0, theta_rescale_factor: float = 1.0, seq_before_head_dim: bool = False, cache_if_possible: bool = True):
        super().__init__()
        del xpos_scale_base
        if use_xpos:
            raise NotImplementedError("SeedVR2 does not use XPos rotary embeddings")
        if interpolate_factor < 1.0:
            raise ValueError("interpolate_factor must be >= 1.0")
        if theta_rescale_factor != 1.0 and dim > 2:
            theta *= theta_rescale_factor ** (dim / (dim - 2))
        self.freqs_for = freqs_for
        self.cache_if_possible = cache_if_possible
        self.learned_freq = learned_freq
        self.interpolate_factor = interpolate_factor
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2
        half_dim = dim // 2
        if custom_freqs is not None:
            freqs = custom_freqs.float()
        elif freqs_for == "lang":
            steps = torch.arange(half_dim, dtype=torch.float32)
            freqs = torch.pow(torch.as_tensor(theta, dtype=torch.float32), -2 * steps / dim)
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, half_dim, dtype=torch.float32) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported rotary frequency type: {freqs_for}")
        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)
        self.register_buffer("dummy", torch.tensor(0), persistent=False)
        self.register_buffer("cached_freqs", None, persistent=False)

    @property
    def device(self):
        return self.dummy.device

    def get_seq_pos(self, seq_len: int, device, dtype, offset: int = 0) -> Tensor:
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def get_axial_freqs(self, *dims: int) -> Tensor:
        if not dims:
            raise ValueError("At least one axis dimension is required")
        axis_freqs = []
        for axis, axis_size in enumerate(dims):
            if self.freqs_for == "pixel":
                positions = torch.linspace(-1, 1, steps=axis_size, device=self.device)
            else:
                positions = torch.arange(axis_size, device=self.device)
            freqs = self.forward(positions, seq_len=axis_size)
            view_shape = [1] * len(dims)
            view_shape[axis] = axis_size
            axis_freqs.append(freqs.reshape(*view_shape, freqs.shape[-1]))
        return torch.cat(broadcast_tensors(*axis_freqs), dim=-1)

    @_no_autocast
    def forward(self, positions: Tensor, seq_len: Optional[int] = None, offset: int = 0) -> Tensor:
        should_cache = self.cache_if_possible and not self.learned_freq and seq_len is not None and self.freqs_for != "pixel"
        if should_cache and self.cached_freqs is not None and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset: offset + seq_len].detach()
        phases = positions.to(dtype=self.freqs.dtype).unsqueeze(-1) * self.freqs
        freqs = torch.stack((phases, phases), dim=-1).flatten(start_dim=-2)
        if should_cache:
            self.cached_freqs = freqs.detach()
        return freqs


flash_attn_3_varlen_func = None
FLASH_ATTN_3_AVAILABLE = False
try:
    import flash_attn_interface
    flash_attn_3_varlen_func = flash_attn_interface.flash_attn_varlen_func
    FLASH_ATTN_3_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    pass

flash_attn_2_varlen_func = None
FLASH_ATTN_2_AVAILABLE = False
try:
    from flash_attn import flash_attn_varlen_func as _fa2_varlen
    import flash_attn_2_cuda  # noqa: F401
    flash_attn_2_varlen_func = _fa2_varlen
    FLASH_ATTN_2_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    pass

sageattn_func = None
SAGE_ATTN_1_AVAILABLE = False
try:
    from sageattention import sageattn as _sageattn
    sageattn_func = _sageattn
    SAGE_ATTN_1_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    pass

sageattn_varlen = None
SAGE_ATTN_2_AVAILABLE = False
try:
    from sageattention import sageattn_varlen as _sa2_varlen
    sageattn_varlen = _sa2_varlen
    SAGE_ATTN_2_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    pass

sageattn_blackwell = None
SAGE_ATTN_3_AVAILABLE = False
try:
    from sageattn3 import sageattn3_blackwell as _sa3_blackwell
    sageattn_blackwell = _sa3_blackwell
    SAGE_ATTN_3_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    try:
        from sageattention import sageattn_blackwell as _sa3_blackwell
        sageattn_blackwell = _sa3_blackwell
        SAGE_ATTN_3_AVAILABLE = True
    except (ImportError, AttributeError, OSError):
        pass

FLASH_ATTN_AVAILABLE = FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE
SAGE_ATTN_AVAILABLE = SAGE_ATTN_1_AVAILABLE or SAGE_ATTN_2_AVAILABLE or SAGE_ATTN_3_AVAILABLE


def validate_attention_mode(requested_mode: str, debug=None) -> str:
    aliases = {
        "sage_attention": "sage_attn",
        "sage1": "sageattn",
        "sage2": "sageattn_2",
        "sage3": "sageattn_3",
        "sageattn1": "sageattn",
        "sageattn2": "sageattn_2",
        "sageattn3": "sageattn_3",
        "sageattn_varlen": "sageattn_2",
        "flash_attention": "flash_attn",
    }
    requested_mode = aliases.get(requested_mode, requested_mode)

    def fallback(mode: str, resolved: str) -> str:
        if resolved != mode and debug is not None:
            log = getattr(debug, "log", None)
            if callable(log):
                log(f"Attention backend {mode!r} resolved to {resolved!r}.", level="WARNING")
        return resolved

    def best_flash() -> str:
        if FLASH_ATTN_3_AVAILABLE:
            return "flash_attn_3"
        if FLASH_ATTN_2_AVAILABLE:
            return "flash_attn_2"
        return "sdpa"

    def best_sage() -> str:
        if SAGE_ATTN_3_AVAILABLE:
            return "sageattn_3"
        if SAGE_ATTN_2_AVAILABLE:
            return "sageattn_2"
        if SAGE_ATTN_1_AVAILABLE:
            return "sageattn"
        return "sdpa"

    if requested_mode == "sdpa":
        return "sdpa"
    if requested_mode == "flash_attn":
        return fallback(requested_mode, best_flash())
    if requested_mode == "sage_attn":
        return fallback(requested_mode, best_sage())
    if requested_mode == "flash_attn_3":
        return fallback(requested_mode, "flash_attn_3" if FLASH_ATTN_3_AVAILABLE else ("flash_attn_2" if FLASH_ATTN_2_AVAILABLE else "sdpa"))
    if requested_mode == "flash_attn_2":
        return fallback(requested_mode, "flash_attn_2" if FLASH_ATTN_2_AVAILABLE else "sdpa")
    if requested_mode == "sageattn_3":
        return fallback(requested_mode, "sageattn_3" if SAGE_ATTN_3_AVAILABLE else ("sageattn_2" if SAGE_ATTN_2_AVAILABLE else ("sageattn" if SAGE_ATTN_1_AVAILABLE else "sdpa")))
    if requested_mode == "sageattn_2":
        return fallback(requested_mode, "sageattn_2" if SAGE_ATTN_2_AVAILABLE else ("sageattn" if SAGE_ATTN_1_AVAILABLE else "sdpa"))
    if requested_mode == "sageattn":
        return fallback(requested_mode, "sageattn" if SAGE_ATTN_1_AVAILABLE else ("sageattn_2" if SAGE_ATTN_2_AVAILABLE else "sdpa"))
    raise ValueError(
        f"Unsupported SeedVR2 attention backend: {requested_mode!r}. "
        "Use one of: sdpa, sage_attn, flash_attn."
    )


def _as_int(value):
    return int(value.item()) if torch.is_tensor(value) else int(value)


@torch._dynamo.disable
def call_flash_attn_2_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    if not FLASH_ATTN_2_AVAILABLE:
        raise ImportError("Flash Attention 2 is not available")
    return flash_attn_2_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=_as_int(max_seqlen_q), max_seqlen_k=_as_int(max_seqlen_k), **kwargs)


@torch._dynamo.disable
def call_flash_attn_3_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    if not FLASH_ATTN_3_AVAILABLE:
        raise ImportError("Flash Attention 3 is not available")
    fa3_kwargs = {key: val for key, val in kwargs.items() if key not in ("dropout_p", "window_size")}
    return flash_attn_3_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=_as_int(max_seqlen_q), max_seqlen_k=_as_int(max_seqlen_k), seqused_q=None, seqused_k=None, **fa3_kwargs)[0]


@torch._dynamo.disable
def call_sage_attn_1_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    del max_seqlen_q, max_seqlen_k
    if not SAGE_ATTN_1_AVAILABLE:
        raise ImportError("SageAttention 1 is not available")
    out_dtype = q.dtype
    if not (q.dtype == k.dtype == v.dtype):
        k = k.to(q.dtype)
        v = v.to(q.dtype)
    if q.dtype not in (torch.float16, torch.bfloat16):
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    is_causal = kwargs.get("causal", False)
    q_splits = list(torch.tensor_split(q, cu_seqlens_q[1:-1].long().cpu(), dim=0))
    k_splits = list(torch.tensor_split(k, cu_seqlens_k[1:-1].long().cpu(), dim=0))
    v_splits = list(torch.tensor_split(v, cu_seqlens_k[1:-1].long().cpu(), dim=0))
    out_splits = [
        sageattn_func(q_i, k_i, v_i, is_causal=is_causal, tensor_layout="NHD")
        for q_i, k_i, v_i in zip(q_splits, k_splits, v_splits)
    ]
    out = torch.cat(out_splits, dim=0)
    return out.to(out_dtype) if out.dtype != out_dtype else out


@torch._dynamo.disable
def call_sage_attn_2_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    if not SAGE_ATTN_2_AVAILABLE:
        raise ImportError("SageAttention 2 is not available")
    out_dtype = q.dtype
    if not (q.dtype == k.dtype == v.dtype):
        k = k.to(q.dtype)
        v = v.to(q.dtype)
    if q.dtype not in (torch.float16, torch.bfloat16):
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    is_causal = kwargs.get("causal", False)
    sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    out = sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, _as_int(max_seqlen_q), _as_int(max_seqlen_k), is_causal, sm_scale)
    return out.to(out_dtype) if out.dtype != out_dtype else out


@torch._dynamo.disable
def call_sage_attn_3_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    if not SAGE_ATTN_3_AVAILABLE:
        raise ImportError("SageAttention 3 is not available")
    max_seqlen_q = _as_int(max_seqlen_q)
    max_seqlen_k = _as_int(max_seqlen_k)
    seq_lens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    seq_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    if not ((seq_lens_q == seq_lens_q[0]).all() and (seq_lens_k == seq_lens_k[0]).all()):
        if SAGE_ATTN_2_AVAILABLE:
            return call_sage_attn_2_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs)
        raise RuntimeError("SageAttention 3 requires uniform sequence lengths and SageAttention 2 is unavailable")
    out_dtype = q.dtype
    if not (q.dtype == k.dtype == v.dtype):
        k = k.to(q.dtype)
        v = v.to(q.dtype)
    if q.dtype not in (torch.float16, torch.bfloat16):
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    batch_size = len(cu_seqlens_q) - 1
    heads = q.shape[1]
    dim = q.shape[2]
    q_b = q.view(batch_size, max_seqlen_q, heads, dim).transpose(1, 2)
    k_b = k.view(batch_size, max_seqlen_k, heads, dim).transpose(1, 2)
    v_b = v.view(batch_size, max_seqlen_k, heads, dim).transpose(1, 2)
    out = sageattn_blackwell(q_b, k_b, v_b, per_block_mean=False)
    out = out.transpose(1, 2).reshape(-1, heads, dim).contiguous()
    return out.to(out_dtype) if out.dtype != out_dtype else out


def _check_conv3d_memory_bug() -> bool:
    try:
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return False
        if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
            return False
        if not (hasattr(torch.backends.cudnn, "is_available") and torch.backends.cudnn.is_available()):
            return False
        version_str = torch.__version__.split("+")[0]
        torch_version = tuple(int(p) for p in version_str.split(".")[:2])
        cudnn_version = torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, "version") else None
        return torch_version >= (2, 9) and cudnn_version is not None and cudnn_version >= 91002
    except Exception:
        return False


NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND = _check_conv3d_memory_bug()


BFLOAT16_SUPPORTED = True
COMPUTE_DTYPE = torch.bfloat16
try:
    if torch.cuda.is_available():
        a = torch.randn(8, 8, dtype=torch.bfloat16, device="cuda:0")
        _ = torch.matmul(a, a)
        del a
except RuntimeError as e:
    if "CUBLAS_STATUS_NOT_SUPPORTED" in str(e):
        BFLOAT16_SUPPORTED = False
        COMPUTE_DTYPE = torch.float16
    else:
        raise


def call_rope_with_stability(method, *args, **kwargs):
    if hasattr(method, "cache_clear"):
        method.cache_clear()
    if torch.cuda.is_available():
        with torch.cuda.amp.autocast(enabled=False):
            return method(*args, **kwargs)
    return method(*args, **kwargs)


class CompatibleDiT(torch.nn.Module):
    def __init__(
        self,
        dit_model: torch.nn.Module,
        debug: Any = None,
        compute_dtype: torch.dtype = COMPUTE_DTYPE,
        skip_conversion: bool = False,
    ):
        super().__init__()
        self.dit_model = dit_model
        self.debug = debug or _NoopDebug()
        self.compute_dtype = compute_dtype
        self.model_dtype = self._detect_model_dtype()
        self.is_fp8_model = self.model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        self.is_fp16_model = self.model_dtype == torch.float16
        if not skip_conversion and self.is_fp8_model:
            self._convert_rope_freqs(target_dtype=self.compute_dtype)
        if not skip_conversion and is_mps_available() and self.model_dtype != self.compute_dtype:
            self._force_nadit_precision(target_dtype=self.compute_dtype)
        self._stabilize_rope_computations()

    def _detect_model_dtype(self) -> torch.dtype:
        try:
            return next(self.dit_model.parameters()).dtype
        except StopIteration:
            return torch.bfloat16

    def _log(self, message: str, **kwargs):
        log = getattr(self.debug, "log", None)
        if callable(log):
            log(message, **kwargs)

    def _convert_rope_freqs(self, target_dtype: torch.dtype = torch.bfloat16) -> None:
        converted = 0
        for module in self.dit_model.modules():
            if "RotaryEmbedding" in type(module).__name__ and hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                if module.rope.freqs.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    module.rope.freqs.data = module.rope.freqs.to(target_dtype)
                    converted += 1
        if converted:
            self._log(f"Converted {converted} RoPE frequency buffers from FP8 to {target_dtype}", category="precision")

    def _force_nadit_precision(self, target_dtype: torch.dtype = torch.bfloat16) -> None:
        for param in self.dit_model.parameters():
            if param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)
        for buffer in self.dit_model.buffers():
            if hasattr(buffer, "tensor_type") or hasattr(buffer, "_layout_cls"):
                continue
            if buffer.dtype != target_dtype:
                buffer.data = buffer.data.to(target_dtype)
        self.model_dtype = target_dtype
        self.is_fp8_model = target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    def _stabilize_rope_computations(self):
        if not hasattr(self.dit_model, "blocks"):
            return
        for module in self.dit_model.modules():
            if not hasattr(module, "get_axial_freqs") or hasattr(module, "_rope_wrapped"):
                continue
            original_method = module.get_axial_freqs

            def stable_rope_computation(this, *args, _original=original_method, **kwargs):
                try:
                    return _original(*args, **kwargs)
                except Exception:
                    return call_rope_with_stability(_original, *args, **kwargs)

            module._rope_wrapped = "stability"
            module._original_get_axial_freqs = original_method
            module.get_axial_freqs = types.MethodType(stable_rope_computation, module)

    def forward(self, *args, **kwargs):
        if self.is_fp8_model:
            fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
            target_dtype = self.compute_dtype
            args = tuple(arg.to(target_dtype) if isinstance(arg, torch.Tensor) and arg.dtype in fp8_dtypes else arg for arg in args)
            kwargs = {
                key: value.to(target_dtype) if isinstance(value, torch.Tensor) and value.dtype in fp8_dtypes else value
                for key, value in kwargs.items()
            }
        return self.dit_model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.dit_model, name)


class _NoopDebug:
    def log(self, *args, **kwargs):
        del args, kwargs

    def start_timer(self, *args, **kwargs):
        del args, kwargs

    def end_timer(self, *args, **kwargs):
        del args, kwargs
