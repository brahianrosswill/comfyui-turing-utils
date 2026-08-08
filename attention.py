"""Attention backend selection and the self-contained Turing Sage backend."""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Callable

import torch

try:
    from .turing_ops import is_supported_turing_device
except ImportError:
    from turing_ops import is_supported_turing_device


LOG = logging.getLogger("comfyui-turing-utils")
SUPPORTED_KERNEL_DTYPES = (torch.float16, torch.bfloat16)
SUPPORTED_INPUT_DTYPES = (*SUPPORTED_KERNEL_DTYPES, torch.float32)
SPARSE_AUTO_MIN_SEQUENCE = 4096
SPARSE_ROUTING_THRESHOLD = 1.0
SPARSE_PREFIX_POLICY = "auto"
SPARSE_LOCAL_BLOCK_RADIUS = 1
SPARSE_TEMPORAL_NEIGHBOR_FRAMES = 1
SPARSE_SKIPPED_RESIDUAL = "2x32"
SPARSE_MINIMUM_ROUTE_DENSITY = 0.0
SPARSE_MAXIMUM_ROUTE_DENSITY = 1.0
SPARSE_DENSE_PREFIX_STEPS = 0
SPARSE_DENSE_SUFFIX_STEPS = 0
SPARSE_DENSE_PREFIX_LAYERS = 1
SPARSE_DENSE_SUFFIX_LAYERS = 1
FRAME_SPARSE_TEMPORAL_WINDOW_FRAMES = 2
FRAME_SPARSE_GLOBAL_ANCHOR_STRIDE = 12
FRAME_SPARSE_SINK_FRAMES = 1
FRAME_SPARSE_PATTERN = "frame_window"
FRAME_SPARSE_QUALITY_PROFILE = "custom"
FRAME_SPARSE_RADIAL_SPATIAL_RADIUS = 1
FRAME_SPARSE_RADIAL_MAX_TEMPORAL_STRIDE = 16
SPARSE_LAYOUT_KEY = "turing_utils_attention_layout"
_PREFLIGHTED_DEVICES: set[int] = set()
_PREFLIGHTED_SPARSE_DEVICES: set[int] = set()
_PREFLIGHTED_FRAME_SPARSE_DEVICES: set[int] = set()
_LOGGED_FP32_COMPAT = False
_LOGGED_TURING_KERNELS: set[tuple] = set()
_LOGGED_TURING_FALLBACKS: set[str] = set()
_LOGGED_SPARSE_KERNELS: set[tuple] = set()
_LOGGED_FRAME_SPARSE_KERNELS: set[tuple] = set()
_LOGGED_SPARSE_DENSE_REASONS: set[str] = set()


@dataclasses.dataclass(frozen=True)
class AttentionBackend:
    option: str
    attention_function: str | None
    label: str
    install_hint: str | None = None
    aliases: tuple[str, ...] = ()


_BACKENDS: dict[str, AttentionBackend] = {}
_ALIASES: dict[str, str] = {}


def _normalize_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def register_attention_backend(backend: AttentionBackend) -> None:
    if backend.option in _BACKENDS:
        raise ValueError(f"duplicate attention backend option: {backend.option}")
    normalized_aliases = {
        _normalize_key(alias) for alias in (backend.option, backend.label, *backend.aliases)
    }
    collisions = {
        alias: _ALIASES[alias] for alias in normalized_aliases if alias in _ALIASES
    }
    if collisions:
        details = ", ".join(
            f"{alias!r} is already owned by {option!r}"
            for alias, option in sorted(collisions.items())
        )
        raise ValueError(f"attention backend alias collision: {details}")
    _BACKENDS[backend.option] = backend
    for alias in normalized_aliases:
        _ALIASES[alias] = backend.option


register_attention_backend(
    AttentionBackend(
        option="auto",
        attention_function=None,
        label="auto",
        aliases=("default", "none", "comfyui_default"),
    )
)
register_attention_backend(
    AttentionBackend(
        option="sage_attn",
        attention_function="sage",
        label="sage attn",
        install_hint="Install sageattention in the ComfyUI Python environment.",
        aliases=(
            "sage",
            "sage_",
            "sageattention",
            "sage_attention",
            "sage_hybrid",
            "turing_sage",
            "turing_sage_hybrid",
        ),
    )
)
register_attention_backend(
    AttentionBackend(
        option="flash_attn",
        attention_function="flash",
        label="flash attn",
        install_hint="Install flash-attn in the ComfyUI Python environment.",
        aliases=("flash", "flash_attention", "flashattention"),
    )
)
register_attention_backend(
    AttentionBackend(
        option="sdpa",
        attention_function="pytorch",
        label="sdpa",
        aliases=("pytorch", "torch", "torch_sdpa"),
    )
)

AUTO_BACKEND_PRIORITY = ("sage_attn", "flash_attn", "sdpa")


def attention_backend_choices() -> tuple[str, ...]:
    return tuple(_BACKENDS)


def normalize_attention_backend(value: str | None) -> str:
    if value is None:
        return "auto"
    option = _ALIASES.get(_normalize_key(value))
    if option is None:
        raise ValueError(
            f"Unsupported Turing Utils attention backend {value!r}. "
            f"Supported options: {', '.join(attention_backend_choices())}"
        )
    return option


def _comfy_attention_function(name: str) -> Callable | None:
    from comfy.ldm.modules import attention as comfy_attention

    return comfy_attention.get_attention_function(name, None)


def _resolve_attention_function(backend: AttentionBackend) -> Callable | None:
    if backend.attention_function is None:
        return None
    function = _comfy_attention_function(backend.attention_function)
    if function is not None:
        return function
    message = (
        f"Turing Utils attention backend {backend.option!r} requires ComfyUI attention "
        f"function {backend.attention_function!r}, but it is not available."
    )
    if backend.install_hint:
        message = f"{message} {backend.install_hint}"
    raise RuntimeError(message)


def _select_attention_backend(option: str) -> tuple[AttentionBackend, Callable]:
    option = normalize_attention_backend(option)
    if option != "auto":
        backend = _BACKENDS[option]
        return backend, _resolve_attention_function(backend)

    for candidate in AUTO_BACKEND_PRIORITY:
        backend = _BACKENDS[candidate]
        function = _comfy_attention_function(backend.attention_function)
        if function is not None:
            return backend, function
    raise RuntimeError("No supported attention backend is available")


def bundled_available() -> bool:
    try:
        from comfyui_turing_utils_kernel.turing_sage import available
    except (ImportError, OSError):
        return False
    return available()


def bundled_sparse_available() -> bool:
    try:
        import comfyui_turing_utils_kernel
        from comfyui_turing_utils_kernel.turing_sage import sparse_available
    except (ImportError, OSError):
        return False
    version = getattr(comfyui_turing_utils_kernel, "__version__", "0.0.0")
    try:
        version_tuple = tuple(int(part) for part in version.split(".")[:3])
    except ValueError:
        return False
    return version_tuple >= (0, 13, 0) and sparse_available()


def bundled_frame_sparse_available() -> bool:
    try:
        import comfyui_turing_utils_kernel
        from comfyui_turing_utils_kernel.turing_sage import frame_sparse_available
    except (ImportError, OSError):
        return False
    version = getattr(comfyui_turing_utils_kernel, "__version__", "0.0.0")
    try:
        version_tuple = tuple(int(part) for part in version.split(".")[:3])
    except ValueError:
        return False
    return version_tuple >= (0, 15, 0) and frame_sparse_available()


def _sageattn(*args, **kwargs):
    from comfyui_turing_utils_kernel import turing_sage

    return turing_sage.sageattn(*args, **kwargs)


def _sol_sparse_sageattn(*args, **kwargs):
    from comfyui_turing_utils_kernel import turing_sage

    return turing_sage.sol_sparse_sageattn(*args, **kwargs)


def _frame_sparse_sageattn(*args, **kwargs):
    from comfyui_turing_utils_kernel import turing_sage

    return turing_sage.frame_sparse_sageattn(*args, **kwargs)


def _sol_sparse_route_selected(route: torch.Tensor) -> int:
    from comfyui_turing_utils_kernel import turing_sage

    return turing_sage.sol_sparse_route_selected(route)


def _sol_sparse_route_selected_device(route: torch.Tensor) -> torch.Tensor:
    from comfyui_turing_utils_kernel import turing_sage

    return turing_sage._sol_sparse_route_selected_device(route)


def preflight_bundled(device: torch.device) -> None:
    if not is_supported_turing_device(device):
        raise RuntimeError(f"unsupported Turing device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_DEVICES:
        return
    from comfyui_turing_utils_kernel.turing_sage import preflight

    preflight(device)
    _PREFLIGHTED_DEVICES.add(index)


def preflight_bundled_sparse(device: torch.device) -> None:
    if not is_supported_turing_device(device):
        raise RuntimeError(f"unsupported Turing device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_SPARSE_DEVICES:
        return
    from comfyui_turing_utils_kernel.turing_sage import preflight_sparse

    preflight_sparse(device)
    _PREFLIGHTED_SPARSE_DEVICES.add(index)


def preflight_bundled_frame_sparse(device: torch.device) -> None:
    if not is_supported_turing_device(device):
        raise RuntimeError(f"unsupported Turing device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_FRAME_SPARSE_DEVICES:
        return
    from comfyui_turing_utils_kernel.turing_sage import preflight_frame_sparse

    preflight_frame_sparse(device)
    _PREFLIGHTED_FRAME_SPARSE_DEVICES.add(index)


def _reshape_qkv(q, k, v, heads: int, enable_gqa: bool):
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("unreshaped Q/K/V must be three-dimensional")
    batch = q.shape[0]
    if heads <= 0 or q.shape[-1] % heads != 0:
        raise ValueError("Q inner dimension must be divisible by the head count")
    head_dim = q.shape[-1] // heads
    kv_heads = k.shape[-1] // head_dim if enable_gqa else heads
    if kv_heads <= 0 or k.shape[-1] != kv_heads * head_dim or v.shape[-1] != kv_heads * head_dim:
        raise ValueError("K/V inner dimensions do not match the Q head dimension")
    q = q.reshape(batch, -1, heads, head_dim)
    k = k.reshape(batch, -1, kv_heads, head_dim)
    v = v.reshape(batch, -1, kv_heads, head_dim)
    return q, k, v, batch, head_dim


def _bundled_fallback(
    fallback: Callable,
    reason: str,
    fallback_args: tuple,
    fallback_kwargs: dict,
):
    if reason not in _LOGGED_TURING_FALLBACKS:
        LOG.warning(
            "Bundled Turing Sage is falling back to ComfyUI attention (%s); "
            "this message is emitted once per reason",
            reason,
        )
        _LOGGED_TURING_FALLBACKS.add(reason)
    return fallback(*fallback_args, **fallback_kwargs)


def turing_sage_attention(
    fallback: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask=None,
    attn_precision=None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    **kwargs,
) -> torch.Tensor:
    global _LOGGED_FP32_COMPAT

    fallback_args = (q, k, v, heads)
    fallback_kwargs = {
        "mask": mask,
        "attn_precision": attn_precision,
        "skip_reshape": skip_reshape,
        "skip_output_reshape": skip_output_reshape,
        **kwargs,
    }
    if not is_supported_turing_device(q.device):
        return _bundled_fallback(
            fallback,
            "Q/K/V are not on a supported sm75 GPU",
            fallback_args,
            fallback_kwargs,
        )
    if mask is not None:
        return _bundled_fallback(
            fallback,
            "an attention mask was supplied",
            fallback_args,
            fallback_kwargs,
        )
    if kwargs.get("low_precision_attention", True) is False:
        return _bundled_fallback(
            fallback,
            "low_precision_attention=False",
            fallback_args,
            fallback_kwargs,
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise RuntimeError(
            f"Turing Sage requires matching Q/K/V dtypes, got {q.dtype}, {k.dtype}, {v.dtype}"
        )
    if q.dtype not in SUPPORTED_INPUT_DTYPES:
        raise RuntimeError(f"Turing Sage supports FP16, BF16, or FP32 Q/K/V, got {q.dtype}")
    if q.device != k.device or q.device != v.device:
        raise RuntimeError("Turing Sage requires Q/K/V on the same CUDA device")

    input_dtype = q.dtype
    enable_gqa = bool(kwargs.get("enable_gqa", False))
    if skip_reshape:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[1] != heads:
            return _bundled_fallback(
                fallback,
                "skip_reshape Q/K/V layout is incompatible",
                fallback_args,
                fallback_kwargs,
            )
        batch, _, _, head_dim = q.shape
        tensor_layout = "HND"
    else:
        try:
            q, k, v, batch, head_dim = _reshape_qkv(q, k, v, heads, enable_gqa)
        except ValueError:
            return _bundled_fallback(
                fallback,
                "unreshaped Q/K/V layout is incompatible",
                fallback_args,
                fallback_kwargs,
            )
        tensor_layout = "NHD"

    if head_dim <= 0 or head_dim > 128:
        return _bundled_fallback(
            fallback,
            f"head_dim={head_dim} is outside the supported range",
            fallback_args,
            fallback_kwargs,
        )

    index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    sequence_axis = 2 if tensor_layout == "HND" else 1
    head_axis = 1 if tensor_layout == "HND" else 2
    kernel_key = (
        index,
        input_dtype,
        tensor_layout,
        head_dim,
        q.shape[sequence_axis],
        k.shape[sequence_axis],
        q.shape[head_axis],
        k.shape[head_axis],
    )
    if kernel_key not in _LOGGED_TURING_KERNELS:
        LOG.info(
            "Bundled Turing Sage active: device=%s dtype=%s layout=%s "
            "Q=%s K=%s V=%s heads=%d",
            q.device,
            input_dtype,
            tensor_layout,
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            heads,
        )
        _LOGGED_TURING_KERNELS.add(kernel_key)

    if input_dtype == torch.float32:
        if not _LOGGED_FP32_COMPAT:
            LOG.info(
                "Turing Sage FP32 compatibility uses BF16 Q/K/V storage and restores FP32 output"
            )
            _LOGGED_FP32_COMPAT = True
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

    output = _sageattn(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=bool(kwargs.get("is_causal", False)),
        sm_scale=kwargs.get("scale"),
        smooth_k=False,
    )
    if tensor_layout == "HND":
        result = output if skip_output_reshape else output.transpose(1, 2).reshape(batch, -1, heads * head_dim)
    elif skip_output_reshape:
        result = output.transpose(1, 2)
    else:
        result = output.reshape(batch, -1, heads * head_dim)
    return result.to(input_dtype) if input_dtype == torch.float32 else result


def _sparse_dense_baseline(
    reason: str,
    fallback: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    **kwargs,
) -> torch.Tensor:
    if reason not in _LOGGED_SPARSE_DENSE_REASONS:
        LOG.info("Experimental sparse attention uses stable Sage for %s", reason)
        _LOGGED_SPARSE_DENSE_REASONS.add(reason)
    return turing_sage_attention(fallback, q, k, v, heads, **kwargs)


def _sparse_prefix_tokens(
    policy: str,
    manual_tokens: int,
    transformer_options,
    sequence_limit: int,
) -> int:
    if policy == "none":
        return 0
    if policy == "manual":
        return min(manual_tokens, sequence_limit)
    layout = (
        transformer_options.get(SPARSE_LAYOUT_KEY)
        if isinstance(transformer_options, dict)
        else None
    )
    if not isinstance(layout, dict):
        return 0
    prefix_tokens = layout.get("dense_prefix_tokens", 0)
    if not isinstance(prefix_tokens, int) or isinstance(prefix_tokens, bool):
        return 0
    return min(max(prefix_tokens, 0), sequence_limit)


def _sparse_temporal_topology(transformer_options, sequence_limit: int):
    layout = (
        transformer_options.get(SPARSE_LAYOUT_KEY)
        if isinstance(transformer_options, dict)
        else None
    )
    if not isinstance(layout, dict):
        return 0, 0, 0
    values = tuple(
        layout.get(key, 0)
        for key in ("topology_start_tokens", "topology_tokens", "tokens_per_frame")
    )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        return 0, 0, 0
    start, tokens, frame_tokens = values
    if (
        start < 0
        or tokens <= 0
        or frame_tokens <= 0
        or start + tokens > sequence_limit
        or tokens % frame_tokens != 0
    ):
        return 0, 0, 0
    return start, tokens, frame_tokens


def _sparse_spatial_topology(transformer_options, tokens_per_frame: int):
    layout = (
        transformer_options.get(SPARSE_LAYOUT_KEY)
        if isinstance(transformer_options, dict)
        else None
    )
    if not isinstance(layout, dict):
        return 0, 0
    height = layout.get("spatial_tokens_height", 0)
    width = layout.get("spatial_tokens_width", 0)
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (height, width)
    ):
        return 0, 0
    if height <= 0 or width <= 0 or height * width != tokens_per_frame:
        return 0, 0
    return height, width


_FRAME_SPARSE_QUALITY_PROFILES = {
    "conservative": {
        "sparse_pattern": "frame_window",
        "temporal_window_frames": 3,
        "global_anchor_stride": 8,
        "rotate_global_anchors": True,
        "sink_frames": 2,
        "radial_spatial_radius": 1,
        "radial_max_temporal_stride": 8,
        "dense_prefix_layers": 2,
        "dense_suffix_layers": 2,
    },
    "balanced": {
        "sparse_pattern": "radial",
        "temporal_window_frames": 2,
        "global_anchor_stride": 0,
        "rotate_global_anchors": True,
        "sink_frames": 1,
        "radial_spatial_radius": 0,
        "radial_max_temporal_stride": 16,
        "dense_prefix_layers": 1,
        "dense_suffix_layers": 1,
    },
    "fast": {
        "sparse_pattern": "radial",
        "temporal_window_frames": 1,
        "global_anchor_stride": 0,
        "rotate_global_anchors": True,
        "sink_frames": 1,
        "radial_spatial_radius": 0,
        "radial_max_temporal_stride": 32,
        "dense_prefix_layers": 1,
        "dense_suffix_layers": 1,
    },
}


def _resolve_frame_sparse_quality_profile(quality_profile: str, **settings):
    quality_profile = str(quality_profile).strip().lower()
    if quality_profile == "custom":
        return settings
    try:
        return {**settings, **_FRAME_SPARSE_QUALITY_PROFILES[quality_profile]}
    except KeyError as error:
        raise ValueError(
            "quality_profile must be custom, conservative, balanced, or fast"
        ) from error


def _sparse_dense_schedule(
    transformer_options,
    prefix_steps: int,
    suffix_steps: int,
    state: dict[str, object],
    *,
    track_step: bool = False,
) -> bool:
    if (
        prefix_steps <= 0
        and suffix_steps <= 0
        and not track_step
    ) or not isinstance(transformer_options, dict):
        return False
    sample_sigmas = transformer_options.get("sample_sigmas")
    current_sigmas = transformer_options.get("sigmas")
    if not torch.is_tensor(sample_sigmas) or not torch.is_tensor(current_sigmas):
        return False
    if sample_sigmas.numel() < 2 or current_sigmas.numel() == 0:
        return False
    # Keep strong references to the tensors used for the cached decision.  ComfyUI
    # reuses the same transformer_options tensors for every block in one model
    # evaluation, then installs a new current-sigma tensor for the next sampler
    # evaluation.  Identity therefore avoids a device synchronization per block
    # and, unlike Tensor._version, is valid for tensors created in inference mode.
    if (
        state.get("sample_sigmas") is sample_sigmas
        and state.get("current_sigmas") is current_sigmas
    ):
        return bool(state["dense"])
    current = current_sigmas.flatten()[0].to(sample_sigmas)
    step = int(torch.argmin((sample_sigmas.flatten() - current).abs()).item())
    sampling_steps = sample_sigmas.numel() - 1
    effective_prefix_steps = min(prefix_steps, sampling_steps)
    effective_suffix_steps = min(suffix_steps, sampling_steps)
    dense = step < effective_prefix_steps or (
        effective_suffix_steps > 0
        and step >= sampling_steps - effective_suffix_steps
    )
    state.clear()
    state.update(
        sample_sigmas=sample_sigmas,
        current_sigmas=current_sigmas,
        dense=dense,
        step=step,
        sampling_steps=sampling_steps,
        prefix_steps=effective_prefix_steps,
        suffix_steps=effective_suffix_steps,
    )
    return dense


def _sparse_dense_prefix_steps(
    transformer_options,
    steps: int,
    state: dict[str, object],
) -> bool:
    """Compatibility wrapper retained for callers testing the prefix-step policy."""
    return _sparse_dense_schedule(transformer_options, steps, 0, state)


def _sparse_dense_layer(
    transformer_options,
    dense_prefix_layers: int,
    dense_suffix_layers: int = 0,
) -> bool:
    if (dense_prefix_layers <= 0 and dense_suffix_layers <= 0) or not isinstance(
        transformer_options, dict
    ):
        return False
    layout = transformer_options.get(SPARSE_LAYOUT_KEY)
    if not isinstance(layout, dict):
        return False
    layer_index = layout.get("layer_index")
    layer_count = layout.get("layer_count")
    if not isinstance(layer_index, int) or isinstance(layer_index, bool):
        return False
    if 0 <= layer_index < dense_prefix_layers:
        return True
    return (
        dense_suffix_layers > 0
        and isinstance(layer_count, int)
        and not isinstance(layer_count, bool)
        and layer_count > 0
        and 0 <= layer_index < layer_count
        and layer_index >= max(layer_count - dense_suffix_layers, 0)
    )


def turing_sol_sparse_attention(
    fallback: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask=None,
    attn_precision=None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    min_sequence_tokens: int = 0,
    routing_threshold: float = SPARSE_ROUTING_THRESHOLD,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    local_block_radius: int = SPARSE_LOCAL_BLOCK_RADIUS,
    temporal_neighbor_frames: int = SPARSE_TEMPORAL_NEIGHBOR_FRAMES,
    skipped_residual: str = SPARSE_SKIPPED_RESIDUAL,
    minimum_route_density: float = SPARSE_MINIMUM_ROUTE_DENSITY,
    maximum_route_density: float = SPARSE_MAXIMUM_ROUTE_DENSITY,
    debug_route_density: bool = False,
    debug_route_keys: set[tuple] | None = None,
    debug_route_state: dict[tuple, list[tuple[torch.Tensor, int, int]]] | None = None,
    debug_context: dict | None = None,
    **kwargs,
) -> torch.Tensor:
    original_q, original_k, original_v = q, k, v
    common = {
        "mask": mask,
        "attn_precision": attn_precision,
        "skip_reshape": skip_reshape,
        "skip_output_reshape": skip_output_reshape,
        **kwargs,
    }

    def dense(reason: str):
        return _sparse_dense_baseline(
            reason,
            fallback,
            original_q,
            original_k,
            original_v,
            heads,
            **common,
        )

    if not is_supported_turing_device(q.device):
        return dense("Q/K/V are not on a supported sm75 GPU")
    if mask is not None:
        return dense("an attention mask was supplied")
    if kwargs.get("low_precision_attention", True) is False:
        return dense("low_precision_attention=False")
    if bool(kwargs.get("is_causal", False)):
        return dense("causal attention")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in SUPPORTED_INPUT_DTYPES:
        return dense("Q/K/V dtypes are incompatible")
    if q.device != k.device or q.device != v.device:
        return dense("Q/K/V devices are incompatible")

    input_dtype = q.dtype
    enable_gqa = bool(kwargs.get("enable_gqa", False))
    if skip_reshape:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[1] != heads:
            return dense("skip_reshape Q/K/V layout is incompatible")
        batch, _, _, head_dim = q.shape
    else:
        try:
            q, k, v, batch, head_dim = _reshape_qkv(q, k, v, heads, enable_gqa)
        except ValueError:
            return dense("unreshaped Q/K/V layout is incompatible")
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    if head_dim != 128:
        return dense(f"head_dim={head_dim} is not 128")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        return dense("Q/K/V batch sizes are incompatible")
    if k.shape[1] != v.shape[1] or k.shape[2:] != v.shape[2:]:
        return dense("K/V shapes are incompatible")
    if k.shape[-1] != 128 or k.shape[1] <= 0 or heads % k.shape[1] != 0:
        return dense("Q/K/V head counts are incompatible")
    effective_min_sequence = min_sequence_tokens or SPARSE_AUTO_MIN_SEQUENCE
    if q.shape[2] < effective_min_sequence or k.shape[2] < effective_min_sequence:
        return dense(f"sequences shorter than {effective_min_sequence} tokens")
    skipped_residual = str(skipped_residual).strip().lower()
    residual_subblocks = {"1x64": 1, "2x32": 2}.get(skipped_residual)
    if residual_subblocks is None:
        raise ValueError("skipped_residual must be 1x64 or 2x32")
    minimum_route_density = float(minimum_route_density)
    maximum_route_density = float(maximum_route_density)
    if not 0.0 <= minimum_route_density <= maximum_route_density <= 1.0:
        raise ValueError(
            "route density bounds must satisfy 0 <= minimum <= maximum <= 1"
        )

    prefix_tokens = _sparse_prefix_tokens(
        prefix_policy,
        manual_prefix_tokens,
        kwargs.get("transformer_options"),
        min(q.shape[2], k.shape[2]),
    )
    if prefix_tokens and q.shape[2] != k.shape[2]:
        return dense("prefix Query splitting requires equal Q/K sequence lengths")
    topology_start, topology_tokens, tokens_per_frame = _sparse_temporal_topology(
        kwargs.get("transformer_options"),
        min(q.shape[2], k.shape[2]),
    )
    if input_dtype == torch.float32:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    kernel_key = (
        q.device.index,
        input_dtype,
        tuple(q.shape),
        tuple(k.shape),
        effective_min_sequence,
        prefix_tokens,
        routing_threshold,
        local_block_radius,
        topology_start,
        topology_tokens,
        tokens_per_frame,
        temporal_neighbor_frames,
        residual_subblocks,
        minimum_route_density,
        maximum_route_density,
    )
    if kernel_key not in _LOGGED_SPARSE_KERNELS:
        LOG.info(
            "Experimental Turing Sol sparse attention active: dtype=%s Q=%s K=%s "
            "min_sequence=%d prefix_policy=%s stable_prefix_q=%d sparse_target_q=%d "
            "selected_qk=int8 score_domain=int8_consistent threshold=%.2f "
            "skipped_residual=%s route_budget=[%.2f,%.2f] "
            "local_radius=%d temporal_frames=%d "
            "topology=(%d,%d,%d)",
            input_dtype,
            tuple(q.shape),
            tuple(k.shape),
            effective_min_sequence,
            prefix_policy,
            prefix_tokens,
            q.shape[2] - prefix_tokens,
            routing_threshold,
            skipped_residual,
            minimum_route_density,
            maximum_route_density,
            local_block_radius,
            temporal_neighbor_frames,
            topology_start,
            topology_tokens,
            tokens_per_frame,
        )
        _LOGGED_SPARSE_KERNELS.add(kernel_key)

    route_keys = debug_route_keys if debug_route_keys is not None else set()
    context = debug_context or {}
    step = context.get("step")
    layer_index = context.get("layer_index")
    layer_count = context.get("layer_count")
    last_sparse_layer = context.get(
        "last_sparse_layer",
        layer_count - 1 if isinstance(layer_count, int) else None,
    )
    aggregate_route_stats = (
        debug_route_density
        and debug_route_state is not None
        and isinstance(step, int)
        and not isinstance(step, bool)
        and isinstance(layer_index, int)
        and not isinstance(layer_index, bool)
        and isinstance(layer_count, int)
        and not isinstance(layer_count, bool)
        and layer_count > 0
        and 0 <= layer_index < layer_count
        and isinstance(last_sparse_layer, int)
        and not isinstance(last_sparse_layer, bool)
        and 0 <= last_sparse_layer < layer_count
    )
    collect_route_stats = debug_route_density and (
        aggregate_route_stats or kernel_key not in route_keys
    )
    sparse_result = _sol_sparse_sageattn(
        q,
        k,
        v,
        tensor_layout="HND",
        sm_scale=kwargs.get("scale"),
        prefix_tokens=prefix_tokens,
        threshold_sigma=routing_threshold,
        local_block_radius=local_block_radius,
        topology_start_tokens=topology_start,
        topology_tokens=topology_tokens,
        tokens_per_frame=tokens_per_frame,
        temporal_neighbor_frames=temporal_neighbor_frames,
        residual_subblocks=residual_subblocks,
        minimum_route_density=minimum_route_density,
        maximum_route_density=maximum_route_density,
        return_route=collect_route_stats,
    )
    if collect_route_stats:
        output, route = sparse_result
        try:
            sparse_query_tokens = q.shape[2] - prefix_tokens
            query_blocks = route.shape[2]
            key_blocks = math.ceil(k.shape[2] / 64)
            possible_blocks = route.shape[0] * route.shape[1] * query_blocks * key_blocks
            if aggregate_route_stats:
                selected_device = _sol_sparse_route_selected_device(route)
                aggregate_key = (step, context.get("sampling_steps"), kernel_key)
                entries = debug_route_state.setdefault(aggregate_key, [])
                entries.append((selected_device, possible_blocks, layer_index))
                if layer_index == last_sparse_layer:
                    selected = torch.cat([entry[0] for entry in entries]).float()
                    possible = torch.tensor(
                        [entry[1] for entry in entries],
                        device=selected.device,
                        dtype=torch.float32,
                    )
                    density = selected / possible.clamp_min(1.0)
                    summary = torch.stack(
                        (
                            selected.sum(),
                            possible.sum(),
                            density.min(),
                            density.mean(),
                            density.max(),
                        )
                    ).cpu().tolist()
                    first_layer = min(entry[2] for entry in entries)
                    last_layer = max(entry[2] for entry in entries)
                    LOG.warning(
                        "[Turing sparse debug] step=%s/%s layers=%d-%d calls=%d "
                        "selected=%d/%d density[min/mean/max]=%.4f/%.4f/%.4f "
                        "Q=%d Qsparse=%d K=%d Hq=%d Hkv=%d threshold=%.2f "
                        "prefix=%d local=%d temporal=%d residual=%s budget=[%.2f,%.2f]",
                        step,
                        context.get("sampling_steps"),
                        first_layer,
                        last_layer,
                        len(entries),
                        int(summary[0]),
                        int(summary[1]),
                        summary[2],
                        summary[3],
                        summary[4],
                        q.shape[2],
                        sparse_query_tokens,
                        k.shape[2],
                        q.shape[1],
                        k.shape[1],
                        routing_threshold,
                        prefix_tokens,
                        local_block_radius,
                        temporal_neighbor_frames,
                        skipped_residual,
                        minimum_route_density,
                        maximum_route_density,
                    )
                    del debug_route_state[aggregate_key]
            else:
                selected_blocks = _sol_sparse_route_selected(route)
                LOG.warning(
                    "[Turing sparse debug] Q=%d Qsparse=%d K=%d Hq=%d Hkv=%d selected=%d/%d "
                    "density=%.4f threshold=%.2f prefix=%d local=%d temporal=%d "
                    "residual=%s budget=[%.2f,%.2f] step=%s/%s layer=%s/%s",
                    q.shape[2],
                    sparse_query_tokens,
                    k.shape[2],
                    q.shape[1],
                    k.shape[1],
                    selected_blocks,
                    possible_blocks,
                    selected_blocks / possible_blocks if possible_blocks else 0.0,
                    routing_threshold,
                    prefix_tokens,
                    local_block_radius,
                    temporal_neighbor_frames,
                    skipped_residual,
                    minimum_route_density,
                    maximum_route_density,
                    context.get("step"),
                    context.get("sampling_steps"),
                    layer_index,
                    layer_count,
                )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            LOG.warning("[Turing sparse debug] route density unavailable: %s", error)
        if not aggregate_route_stats:
            route_keys.add(kernel_key)
    else:
        output = sparse_result
    batch, _, _, head_dim = q.shape
    result = output if skip_output_reshape else output.transpose(1, 2).reshape(
        batch, -1, heads * head_dim
    )
    return result.to(input_dtype) if input_dtype == torch.float32 else result


def turing_frame_sparse_attention(
    fallback: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask=None,
    attn_precision=None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    temporal_window_frames: int = FRAME_SPARSE_TEMPORAL_WINDOW_FRAMES,
    global_anchor_stride: int = FRAME_SPARSE_GLOBAL_ANCHOR_STRIDE,
    rotate_global_anchors: bool = True,
    sink_frames: int = FRAME_SPARSE_SINK_FRAMES,
    sparse_pattern: str = FRAME_SPARSE_PATTERN,
    radial_spatial_radius: int = FRAME_SPARSE_RADIAL_SPATIAL_RADIUS,
    radial_max_temporal_stride: int = FRAME_SPARSE_RADIAL_MAX_TEMPORAL_STRIDE,
    debug_route_density: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Structured video-tail sparsity with the stable SM75 Sage math path."""
    original_q, original_k, original_v = q, k, v
    common = {
        "mask": mask,
        "attn_precision": attn_precision,
        "skip_reshape": skip_reshape,
        "skip_output_reshape": skip_output_reshape,
        **kwargs,
    }

    def dense(reason: str):
        return _sparse_dense_baseline(
            reason,
            fallback,
            original_q,
            original_k,
            original_v,
            heads,
            **common,
        )

    if not is_supported_turing_device(q.device):
        return dense("Q/K/V are not on a supported sm75 GPU")
    if mask is not None:
        return dense("an attention mask was supplied")
    if kwargs.get("low_precision_attention", True) is False:
        return dense("low_precision_attention=False")
    if bool(kwargs.get("is_causal", False)):
        return dense("causal attention")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in SUPPORTED_INPUT_DTYPES:
        return dense("Q/K/V dtypes are incompatible")
    if q.device != k.device or q.device != v.device:
        return dense("Q/K/V devices are incompatible")

    input_dtype = q.dtype
    enable_gqa = bool(kwargs.get("enable_gqa", False))
    if skip_reshape:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[1] != heads:
            return dense("skip_reshape Q/K/V layout is incompatible")
        batch, _, _, head_dim = q.shape
    else:
        try:
            q, k, v, batch, head_dim = _reshape_qkv(q, k, v, heads, enable_gqa)
        except ValueError:
            return dense("unreshaped Q/K/V layout is incompatible")
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    if head_dim != 128:
        return dense(f"head_dim={head_dim} is not 128")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        return dense("Q/K/V batch sizes are incompatible")
    if k.shape[1] != v.shape[1] or k.shape[2:] != v.shape[2:]:
        return dense("K/V shapes are incompatible")
    if k.shape[-1] != 128 or k.shape[1] <= 0 or heads % k.shape[1] != 0:
        return dense("Q/K/V head counts are incompatible")
    if q.shape[2] != k.shape[2]:
        return dense("frame sparsity requires equal Q/K sequence lengths")
    if q.shape[2] < SPARSE_AUTO_MIN_SEQUENCE:
        return dense(f"sequences shorter than {SPARSE_AUTO_MIN_SEQUENCE} tokens")

    transformer_options = kwargs.get("transformer_options")
    topology_start, topology_tokens, tokens_per_frame = _sparse_temporal_topology(
        transformer_options,
        q.shape[2],
    )
    if topology_tokens <= 0 or topology_start + topology_tokens != q.shape[2]:
        return dense("contiguous video-tail topology metadata is unavailable")
    spatial_tokens_height, spatial_tokens_width = _sparse_spatial_topology(
        transformer_options, tokens_per_frame
    )
    if sparse_pattern == "radial" and (
        spatial_tokens_height <= 0 or spatial_tokens_width <= 0
    ):
        return dense("radial spatial topology metadata is unavailable")
    prefix_tokens = _sparse_prefix_tokens(
        prefix_policy,
        manual_prefix_tokens,
        transformer_options,
        q.shape[2],
    )
    layout = (
        transformer_options.get(SPARSE_LAYOUT_KEY)
        if isinstance(transformer_options, dict)
        else None
    )
    layer_index = layout.get("layer_index") if isinstance(layout, dict) else None
    rotation_period = (
        global_anchor_stride
        if global_anchor_stride > 0
        else radial_max_temporal_stride if sparse_pattern == "radial" else 0
    )
    anchor_offset = (
        layer_index % rotation_period
        if rotate_global_anchors
        and rotation_period > 0
        and isinstance(layer_index, int)
        and not isinstance(layer_index, bool)
        else 0
    )

    if input_dtype == torch.float32:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    kernel_key = (
        q.device.index,
        input_dtype,
        tuple(q.shape),
        tuple(k.shape),
        prefix_tokens,
        topology_start,
        topology_tokens,
        tokens_per_frame,
        temporal_window_frames,
        global_anchor_stride,
        anchor_offset,
        sink_frames,
        sparse_pattern,
        spatial_tokens_height,
        spatial_tokens_width,
        radial_spatial_radius,
        radial_max_temporal_stride,
    )
    first_kernel_use = kernel_key not in _LOGGED_FRAME_SPARSE_KERNELS
    collect_density = debug_route_density or first_kernel_use
    sparse_result = _frame_sparse_sageattn(
        q,
        k,
        v,
        tensor_layout="HND",
        sm_scale=kwargs.get("scale"),
        prefix_tokens=prefix_tokens,
        topology_start_tokens=topology_start,
        topology_tokens=topology_tokens,
        tokens_per_frame=tokens_per_frame,
        temporal_window_frames=temporal_window_frames,
        global_anchor_stride=global_anchor_stride,
        global_anchor_offset=anchor_offset,
        sink_frames=sink_frames,
        sparse_pattern=sparse_pattern,
        spatial_tokens_height=spatial_tokens_height,
        spatial_tokens_width=spatial_tokens_width,
        radial_spatial_radius=radial_spatial_radius,
        radial_max_temporal_stride=radial_max_temporal_stride,
        return_schedule_density=collect_density,
    )
    if collect_density:
        output, density = sparse_result
        if first_kernel_use:
            LOG.info(
                "Experimental Turing frame-sparse Sage active: dtype=%s Q=%s K=%s "
                "prefix_policy=%s dense_prefix_k=%d dense_prefix_q=%d "
                "video=(tokens=%d frame_tokens=%d frames=%d) window=%d "
                "pattern=%s anchor_stride=%d anchor_offset=%d sink_frames=%d "
                "radial_radius=%d radial_max_stride=%d density=%.4f",
                input_dtype,
                tuple(q.shape),
                tuple(k.shape),
                prefix_policy,
                prefix_tokens,
                topology_start,
                topology_tokens,
                tokens_per_frame,
                topology_tokens // tokens_per_frame,
                temporal_window_frames,
                sparse_pattern,
                global_anchor_stride,
                anchor_offset,
                sink_frames,
                radial_spatial_radius,
                radial_max_temporal_stride,
                density,
            )
            _LOGGED_FRAME_SPARSE_KERNELS.add(kernel_key)
        if debug_route_density and first_kernel_use:
            LOG.warning(
                "[Turing frame sparse debug] layer=%s window=%d anchor_stride=%d "
                "anchor_offset=%d sink_frames=%d density=%.4f",
                layer_index,
                temporal_window_frames,
                global_anchor_stride,
                anchor_offset,
                sink_frames,
                density,
            )
    else:
        output = sparse_result
    result = output if skip_output_reshape else output.transpose(1, 2).reshape(
        batch, -1, heads * head_dim
    )
    return result.to(input_dtype) if input_dtype == torch.float32 else result


def _uses_bundled_turing_sage(option: str, device: torch.device | None) -> bool:
    option = normalize_attention_backend(option)
    return bool(
        device is not None
        and is_supported_turing_device(device)
        and option in {"auto", "sage_attn"}
    )


def _dtype_compatible_fallback(original: Callable, *args, **kwargs):
    qkv = args[:3]
    if (
        len(qkv) == 3
        and all(isinstance(value, torch.Tensor) for value in qkv)
        and all(value.dtype == torch.float32 for value in qkv)
    ):
        pytorch_attention = _comfy_attention_function("pytorch")
        if pytorch_attention is None:
            raise RuntimeError("ComfyUI PyTorch attention is unavailable for the FP32 fallback")
        return pytorch_attention(*args, **kwargs)
    return original(*args, **kwargs)


def make_attention_override(option: str, device: torch.device | None = None) -> Callable:
    option = normalize_attention_backend(option)
    bundled_turing = _uses_bundled_turing_sage(option, device)
    if bundled_turing:
        if not bundled_available():
            raise RuntimeError(
                "The bundled Turing Sage extensions are unavailable. "
                "Rebuild comfyui-turing-utils-kernel with COMFYUI_TURING_UTILS_ARCH_LIST including 7.5."
            )
        preflight_bundled(device)
        backend = _BACKENDS["sage_attn"]
        target = turing_sage_attention
        implementation = "bundled_turing_sage"
    else:
        backend, target = _select_attention_backend(option)
        implementation = f"comfy:{backend.attention_function}"

    def attention_override(original: Callable, *args, **kwargs):
        fallback = lambda *fallback_args, **fallback_kwargs: _dtype_compatible_fallback(
            original, *fallback_args, **fallback_kwargs
        )
        if bundled_turing:
            return target(fallback, *args, **kwargs)
        if (
            backend.option == "sage_attn"
            and len(args) >= 3
            and all(isinstance(value, torch.Tensor) for value in args[:3])
            and any(value.dtype == torch.float32 for value in args[:3])
        ):
            return fallback(*args, **kwargs)
        return target(*args, **kwargs)

    attention_override.turing_utils_attention_backend = backend.option
    attention_override.turing_utils_attention_implementation = implementation
    return attention_override


def make_sparse_attention_override(
    device: torch.device,
    min_sequence_tokens: int = 0,
    routing_threshold: float = SPARSE_ROUTING_THRESHOLD,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    local_block_radius: int = SPARSE_LOCAL_BLOCK_RADIUS,
    temporal_neighbor_frames: int = SPARSE_TEMPORAL_NEIGHBOR_FRAMES,
    skipped_residual: str = SPARSE_SKIPPED_RESIDUAL,
    minimum_route_density: float = SPARSE_MINIMUM_ROUTE_DENSITY,
    maximum_route_density: float = SPARSE_MAXIMUM_ROUTE_DENSITY,
    dense_prefix_steps: int = SPARSE_DENSE_PREFIX_STEPS,
    dense_suffix_steps: int = SPARSE_DENSE_SUFFIX_STEPS,
    dense_prefix_layers: int = SPARSE_DENSE_PREFIX_LAYERS,
    dense_suffix_layers: int = SPARSE_DENSE_SUFFIX_LAYERS,
    debug_route_density: bool = False,
) -> Callable:
    min_sequence_tokens = int(min_sequence_tokens)
    routing_threshold = float(routing_threshold)
    prefix_policy = str(prefix_policy).strip().lower()
    manual_prefix_tokens = int(manual_prefix_tokens)
    local_block_radius = int(local_block_radius)
    temporal_neighbor_frames = int(temporal_neighbor_frames)
    skipped_residual = str(skipped_residual).strip().lower()
    minimum_route_density = float(minimum_route_density)
    maximum_route_density = float(maximum_route_density)
    dense_prefix_steps = int(dense_prefix_steps)
    dense_suffix_steps = int(dense_suffix_steps)
    dense_prefix_layers = int(dense_prefix_layers)
    dense_suffix_layers = int(dense_suffix_layers)
    debug_route_density = bool(debug_route_density)
    if min_sequence_tokens < 0:
        raise ValueError("min_sequence_tokens must be non-negative")
    if not math.isfinite(routing_threshold):
        raise ValueError("routing_threshold must be finite")
    if prefix_policy not in {"auto", "none", "manual"}:
        raise ValueError("prefix_policy must be auto, none, or manual")
    if manual_prefix_tokens < 0:
        raise ValueError("manual_prefix_tokens must be non-negative")
    if local_block_radius < 0:
        raise ValueError("local_block_radius must be non-negative")
    if temporal_neighbor_frames < 0:
        raise ValueError("temporal_neighbor_frames must be non-negative")
    if skipped_residual not in {"1x64", "2x32"}:
        raise ValueError("skipped_residual must be 1x64 or 2x32")
    if not 0.0 <= minimum_route_density <= maximum_route_density <= 1.0:
        raise ValueError(
            "route density bounds must satisfy 0 <= minimum <= maximum <= 1"
        )
    if dense_prefix_steps < 0:
        raise ValueError("dense_prefix_steps must be non-negative")
    if dense_suffix_steps < 0:
        raise ValueError("dense_suffix_steps must be non-negative")
    if dense_prefix_layers < 0:
        raise ValueError("dense_prefix_layers must be non-negative")
    if dense_suffix_layers < 0:
        raise ValueError("dense_suffix_layers must be non-negative")
    if not is_supported_turing_device(device):
        raise RuntimeError("Sol sparse attention requires an sm75 Turing GPU")
    if not bundled_sparse_available():
        raise RuntimeError(
            "The experimental Turing sparse extension is unavailable. "
            "Rebuild comfyui-turing-utils-kernel 0.13.0 or newer with sm75 enabled."
        )
    preflight_bundled(device)
    preflight_bundled_sparse(device)
    schedule_state: dict[str, object] = {}
    debug_route_keys: set[tuple] = set()
    debug_route_state: dict[tuple, list[tuple[torch.Tensor, int, int]]] = {}
    debug_dense_reasons: set[str] = set()

    def attention_override(original: Callable, *args, **kwargs):
        fallback = lambda *fallback_args, **fallback_kwargs: _dtype_compatible_fallback(
            original, *fallback_args, **fallback_kwargs
        )
        transformer_options = kwargs.get("transformer_options")
        dense_schedule = _sparse_dense_schedule(
            transformer_options,
            dense_prefix_steps,
            dense_suffix_steps,
            schedule_state,
        )
        dense_layer = _sparse_dense_layer(
            transformer_options,
            dense_prefix_layers,
            dense_suffix_layers,
        )
        if debug_route_density and dense_schedule:
            debug_key = f"schedule:{schedule_state.get('step')}"
            if debug_key not in debug_dense_reasons:
                LOG.warning(
                    "[Turing sparse debug] stable Sage selected by dense schedule: "
                    "step=%s/%s prefix_steps=%s suffix_steps=%s",
                    schedule_state.get("step"),
                    schedule_state.get("sampling_steps"),
                    schedule_state.get("prefix_steps"),
                    schedule_state.get("suffix_steps"),
                )
                debug_dense_reasons.add(debug_key)
        if debug_route_density and dense_layer:
            layout = transformer_options.get(SPARSE_LAYOUT_KEY, {})
            debug_key = f"layer:{layout.get('layer_index')}"
            if debug_key not in debug_dense_reasons:
                LOG.warning(
                    "[Turing sparse debug] stable Sage selected for protected layer %s/%s",
                    layout.get("layer_index"),
                    layout.get("layer_count"),
                )
                debug_dense_reasons.add(debug_key)
        if dense_schedule or dense_layer:
            return turing_sage_attention(fallback, *args, **kwargs)
        debug_context = None
        if debug_route_density:
            layout = (
                transformer_options.get(SPARSE_LAYOUT_KEY, {})
                if isinstance(transformer_options, dict)
                else {}
            )
            debug_context = {
                "step": schedule_state.get("step"),
                "sampling_steps": schedule_state.get("sampling_steps"),
                "layer_index": layout.get("layer_index"),
                "layer_count": layout.get("layer_count"),
                "last_sparse_layer": (
                    layout.get("layer_count") - dense_suffix_layers - 1
                    if isinstance(layout.get("layer_count"), int)
                    and not isinstance(layout.get("layer_count"), bool)
                    else None
                ),
            }
        return turing_sol_sparse_attention(
            fallback,
            *args,
            min_sequence_tokens=min_sequence_tokens,
            routing_threshold=routing_threshold,
            prefix_policy=prefix_policy,
            manual_prefix_tokens=manual_prefix_tokens,
            local_block_radius=local_block_radius,
            temporal_neighbor_frames=temporal_neighbor_frames,
            skipped_residual=skipped_residual,
            minimum_route_density=minimum_route_density,
            maximum_route_density=maximum_route_density,
            debug_route_density=debug_route_density,
            debug_route_keys=debug_route_keys if debug_route_density else None,
            debug_route_state=debug_route_state if debug_route_density else None,
            debug_context=debug_context,
            **kwargs,
        )

    attention_override.turing_utils_attention_backend = "sol_sparse_attn"
    attention_override.turing_utils_attention_implementation = "bundled_turing_sol_sparse_experimental"
    return attention_override


def apply_sparse_attention_patch(
    model,
    min_sequence_tokens: int = 0,
    routing_threshold: float = SPARSE_ROUTING_THRESHOLD,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    local_block_radius: int = SPARSE_LOCAL_BLOCK_RADIUS,
    temporal_neighbor_frames: int = SPARSE_TEMPORAL_NEIGHBOR_FRAMES,
    skipped_residual: str = SPARSE_SKIPPED_RESIDUAL,
    minimum_route_density: float = SPARSE_MINIMUM_ROUTE_DENSITY,
    maximum_route_density: float = SPARSE_MAXIMUM_ROUTE_DENSITY,
    dense_prefix_steps: int = SPARSE_DENSE_PREFIX_STEPS,
    dense_suffix_steps: int = SPARSE_DENSE_SUFFIX_STEPS,
    dense_prefix_layers: int = SPARSE_DENSE_PREFIX_LAYERS,
    dense_suffix_layers: int = SPARSE_DENSE_SUFFIX_LAYERS,
    debug_route_density: bool = False,
):
    patched = model.clone()
    override = make_sparse_attention_override(
        patched.load_device,
        min_sequence_tokens=min_sequence_tokens,
        routing_threshold=routing_threshold,
        prefix_policy=prefix_policy,
        manual_prefix_tokens=manual_prefix_tokens,
        local_block_radius=local_block_radius,
        temporal_neighbor_frames=temporal_neighbor_frames,
        skipped_residual=skipped_residual,
        minimum_route_density=minimum_route_density,
        maximum_route_density=maximum_route_density,
        dense_prefix_steps=dense_prefix_steps,
        dense_suffix_steps=dense_suffix_steps,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
        debug_route_density=debug_route_density,
    )
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    transformer_options["optimized_attention_override"] = override
    transformer_options["turing_utils_attention_backend"] = "sol_sparse_attn"
    transformer_options["turing_utils_attention_implementation"] = (
        "bundled_turing_sol_sparse_experimental"
    )
    LOG.info(
        "Sol sparse attention patch enabled: threshold=%.2f "
        "prefix_policy=%s manual_prefix=%d local_radius=%d temporal_frames=%d "
        "skipped_residual=%s route_budget=[%.2f,%.2f] "
        "dense_prefix_steps=%d dense_suffix_steps=%d "
        "dense_prefix_layers=%d dense_suffix_layers=%d "
        "dense_backend=bundled_turing_sage debug_route_density=%s",
        routing_threshold,
        prefix_policy,
        manual_prefix_tokens,
        local_block_radius,
        temporal_neighbor_frames,
        skipped_residual,
        minimum_route_density,
        maximum_route_density,
        dense_prefix_steps,
        dense_suffix_steps,
        dense_prefix_layers,
        dense_suffix_layers,
        debug_route_density,
    )
    return patched


def make_frame_sparse_attention_override(
    device: torch.device,
    quality_profile: str = FRAME_SPARSE_QUALITY_PROFILE,
    sparse_pattern: str = FRAME_SPARSE_PATTERN,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    temporal_window_frames: int = FRAME_SPARSE_TEMPORAL_WINDOW_FRAMES,
    global_anchor_stride: int = FRAME_SPARSE_GLOBAL_ANCHOR_STRIDE,
    rotate_global_anchors: bool = True,
    sink_frames: int = FRAME_SPARSE_SINK_FRAMES,
    radial_spatial_radius: int = FRAME_SPARSE_RADIAL_SPATIAL_RADIUS,
    radial_max_temporal_stride: int = FRAME_SPARSE_RADIAL_MAX_TEMPORAL_STRIDE,
    dense_prefix_steps: int = SPARSE_DENSE_PREFIX_STEPS,
    dense_suffix_steps: int = SPARSE_DENSE_SUFFIX_STEPS,
    dense_prefix_layers: int = SPARSE_DENSE_PREFIX_LAYERS,
    dense_suffix_layers: int = SPARSE_DENSE_SUFFIX_LAYERS,
    debug_route_density: bool = False,
) -> Callable:
    quality_profile = str(quality_profile).strip().lower()
    sparse_pattern = str(sparse_pattern).strip().lower()
    prefix_policy = str(prefix_policy).strip().lower()
    manual_prefix_tokens = int(manual_prefix_tokens)
    temporal_window_frames = int(temporal_window_frames)
    global_anchor_stride = int(global_anchor_stride)
    rotate_global_anchors = bool(rotate_global_anchors)
    sink_frames = int(sink_frames)
    radial_spatial_radius = int(radial_spatial_radius)
    radial_max_temporal_stride = int(radial_max_temporal_stride)
    dense_prefix_steps = int(dense_prefix_steps)
    dense_suffix_steps = int(dense_suffix_steps)
    dense_prefix_layers = int(dense_prefix_layers)
    dense_suffix_layers = int(dense_suffix_layers)
    debug_route_density = bool(debug_route_density)
    resolved = _resolve_frame_sparse_quality_profile(
        quality_profile,
        sparse_pattern=sparse_pattern,
        temporal_window_frames=temporal_window_frames,
        global_anchor_stride=global_anchor_stride,
        rotate_global_anchors=rotate_global_anchors,
        sink_frames=sink_frames,
        radial_spatial_radius=radial_spatial_radius,
        radial_max_temporal_stride=radial_max_temporal_stride,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
    )
    sparse_pattern = resolved["sparse_pattern"]
    temporal_window_frames = resolved["temporal_window_frames"]
    global_anchor_stride = resolved["global_anchor_stride"]
    rotate_global_anchors = resolved["rotate_global_anchors"]
    sink_frames = resolved["sink_frames"]
    radial_spatial_radius = resolved["radial_spatial_radius"]
    radial_max_temporal_stride = resolved["radial_max_temporal_stride"]
    dense_prefix_layers = resolved["dense_prefix_layers"]
    dense_suffix_layers = resolved["dense_suffix_layers"]
    if sparse_pattern not in {"frame_window", "radial"}:
        raise ValueError("sparse_pattern must be frame_window or radial")
    if prefix_policy not in {"auto", "none", "manual"}:
        raise ValueError("prefix_policy must be auto, none, or manual")
    if manual_prefix_tokens < 0:
        raise ValueError("manual_prefix_tokens must be non-negative")
    if temporal_window_frames < 0:
        raise ValueError("temporal_window_frames must be non-negative")
    if global_anchor_stride < 0:
        raise ValueError("global_anchor_stride must be non-negative")
    if sink_frames < 0:
        raise ValueError("sink_frames must be non-negative")
    if radial_spatial_radius < 0:
        raise ValueError("radial_spatial_radius must be non-negative")
    if radial_max_temporal_stride <= 0:
        raise ValueError("radial_max_temporal_stride must be positive")
    if dense_prefix_steps < 0:
        raise ValueError("dense_prefix_steps must be non-negative")
    if dense_suffix_steps < 0:
        raise ValueError("dense_suffix_steps must be non-negative")
    if dense_prefix_layers < 0:
        raise ValueError("dense_prefix_layers must be non-negative")
    if dense_suffix_layers < 0:
        raise ValueError("dense_suffix_layers must be non-negative")
    if not is_supported_turing_device(device):
        raise RuntimeError("frame-sparse attention requires an sm75 Turing GPU")
    if not bundled_frame_sparse_available():
        raise RuntimeError(
            "The experimental Turing frame-sparse extension is unavailable. "
            "Rebuild comfyui-turing-utils-kernel 0.15.0 or newer with sm75 enabled."
        )
    preflight_bundled(device)
    preflight_bundled_frame_sparse(device)
    schedule_state: dict[str, object] = {}
    debug_dense_reasons: set[str] = set()

    def attention_override(original: Callable, *args, **kwargs):
        fallback = lambda *fallback_args, **fallback_kwargs: _dtype_compatible_fallback(
            original, *fallback_args, **fallback_kwargs
        )
        transformer_options = kwargs.get("transformer_options")
        dense_schedule = _sparse_dense_schedule(
            transformer_options,
            dense_prefix_steps,
            dense_suffix_steps,
            schedule_state,
        )
        dense_layer = _sparse_dense_layer(
            transformer_options,
            dense_prefix_layers,
            dense_suffix_layers,
        )
        if debug_route_density and dense_schedule:
            debug_key = f"schedule:{schedule_state.get('step')}"
            if debug_key not in debug_dense_reasons:
                LOG.warning(
                    "[Turing frame sparse debug] stable Sage selected by dense schedule: "
                    "step=%s/%s prefix_steps=%s suffix_steps=%s",
                    schedule_state.get("step"),
                    schedule_state.get("sampling_steps"),
                    schedule_state.get("prefix_steps"),
                    schedule_state.get("suffix_steps"),
                )
                debug_dense_reasons.add(debug_key)
        if debug_route_density and dense_layer:
            layout = (
                transformer_options.get(SPARSE_LAYOUT_KEY, {})
                if isinstance(transformer_options, dict)
                else {}
            )
            debug_key = f"layer:{layout.get('layer_index')}"
            if debug_key not in debug_dense_reasons:
                LOG.warning(
                    "[Turing frame sparse debug] stable Sage selected for protected layer %s/%s",
                    layout.get("layer_index"),
                    layout.get("layer_count"),
                )
                debug_dense_reasons.add(debug_key)
        if dense_schedule or dense_layer:
            return turing_sage_attention(fallback, *args, **kwargs)
        return turing_frame_sparse_attention(
            fallback,
            *args,
            prefix_policy=prefix_policy,
            manual_prefix_tokens=manual_prefix_tokens,
            temporal_window_frames=temporal_window_frames,
            global_anchor_stride=global_anchor_stride,
            rotate_global_anchors=rotate_global_anchors,
            sink_frames=sink_frames,
            sparse_pattern=sparse_pattern,
            radial_spatial_radius=radial_spatial_radius,
            radial_max_temporal_stride=radial_max_temporal_stride,
            debug_route_density=debug_route_density,
            **kwargs,
        )

    attention_override.turing_utils_attention_backend = "frame_sparse_attn"
    attention_override.turing_utils_attention_implementation = (
        "bundled_turing_frame_sparse_experimental"
    )
    attention_override.turing_utils_frame_sparse_settings = {
        "quality_profile": quality_profile,
        "sparse_pattern": sparse_pattern,
        "temporal_window_frames": temporal_window_frames,
        "global_anchor_stride": global_anchor_stride,
        "rotate_global_anchors": rotate_global_anchors,
        "sink_frames": sink_frames,
        "radial_spatial_radius": radial_spatial_radius,
        "radial_max_temporal_stride": radial_max_temporal_stride,
        "dense_prefix_layers": dense_prefix_layers,
        "dense_suffix_layers": dense_suffix_layers,
    }
    return attention_override


def apply_frame_sparse_attention_patch(
    model,
    quality_profile: str = FRAME_SPARSE_QUALITY_PROFILE,
    sparse_pattern: str = FRAME_SPARSE_PATTERN,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    temporal_window_frames: int = FRAME_SPARSE_TEMPORAL_WINDOW_FRAMES,
    global_anchor_stride: int = FRAME_SPARSE_GLOBAL_ANCHOR_STRIDE,
    rotate_global_anchors: bool = True,
    sink_frames: int = FRAME_SPARSE_SINK_FRAMES,
    radial_spatial_radius: int = FRAME_SPARSE_RADIAL_SPATIAL_RADIUS,
    radial_max_temporal_stride: int = FRAME_SPARSE_RADIAL_MAX_TEMPORAL_STRIDE,
    dense_prefix_steps: int = SPARSE_DENSE_PREFIX_STEPS,
    dense_suffix_steps: int = SPARSE_DENSE_SUFFIX_STEPS,
    dense_prefix_layers: int = SPARSE_DENSE_PREFIX_LAYERS,
    dense_suffix_layers: int = SPARSE_DENSE_SUFFIX_LAYERS,
    debug_route_density: bool = False,
):
    patched = model.clone()
    override = make_frame_sparse_attention_override(
        patched.load_device,
        quality_profile=quality_profile,
        sparse_pattern=sparse_pattern,
        prefix_policy=prefix_policy,
        manual_prefix_tokens=manual_prefix_tokens,
        temporal_window_frames=temporal_window_frames,
        global_anchor_stride=global_anchor_stride,
        rotate_global_anchors=rotate_global_anchors,
        sink_frames=sink_frames,
        radial_spatial_radius=radial_spatial_radius,
        radial_max_temporal_stride=radial_max_temporal_stride,
        dense_prefix_steps=dense_prefix_steps,
        dense_suffix_steps=dense_suffix_steps,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
        debug_route_density=debug_route_density,
    )
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    transformer_options["optimized_attention_override"] = override
    transformer_options["turing_utils_attention_backend"] = "frame_sparse_attn"
    transformer_options["turing_utils_attention_implementation"] = (
        "bundled_turing_frame_sparse_experimental"
    )
    resolved = override.turing_utils_frame_sparse_settings
    LOG.info(
        "Frame-sparse attention patch enabled: profile=%s pattern=%s "
        "prefix_policy=%s manual_prefix=%d "
        "temporal_window=%d anchor_stride=%d rotate_anchors=%s sink_frames=%d "
        "radial_radius=%d radial_max_stride=%d "
        "dense_prefix_steps=%d dense_suffix_steps=%d "
        "dense_prefix_layers=%d dense_suffix_layers=%d "
        "dense_backend=bundled_turing_sage debug_route_density=%s",
        resolved["quality_profile"],
        resolved["sparse_pattern"],
        prefix_policy,
        manual_prefix_tokens,
        resolved["temporal_window_frames"],
        resolved["global_anchor_stride"],
        resolved["rotate_global_anchors"],
        resolved["sink_frames"],
        resolved["radial_spatial_radius"],
        resolved["radial_max_temporal_stride"],
        dense_prefix_steps,
        dense_suffix_steps,
        resolved["dense_prefix_layers"],
        resolved["dense_suffix_layers"],
        debug_route_density,
    )
    return patched


def apply_attention_backend(model, option: str, device: torch.device | None = None):
    option = normalize_attention_backend(option)
    transformer_options = model.model_options.setdefault("transformer_options", {})
    override = make_attention_override(option, device=device)
    selected = override.turing_utils_attention_backend
    implementation = override.turing_utils_attention_implementation
    transformer_options["optimized_attention_override"] = override
    transformer_options["turing_utils_attention_backend"] = selected
    transformer_options["turing_utils_attention_implementation"] = implementation
    LOG.info(
        "Turing Utils attention backend override: %s via %s (requested %s)",
        selected,
        implementation,
        option,
    )
    return model
