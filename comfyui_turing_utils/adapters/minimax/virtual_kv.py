"""MiniMax H3 static-image execution with a virtual 22-frame K/V context.

The physical network remains a five-frame H3 run (two latent-time rows).  Only
the target-video keys and values presented to attention are changed.  Queries,
attention outputs, residuals, and FFNs retain the physical sequence length.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch

from ...attention.layout import attention_semantic_layout
from ...attention.orchestration import install_attention_strategy
from ...attention.patches import attention_base_runtime
from ...attention.protocol import (
    AttentionExecutionOutcome,
    MAPPED_KV_EXECUTOR_ATTR,
    MAPPED_RESIDUAL_CAPABILITY_ATTR,
    MAPPED_RESIDUAL_EXECUTOR_ATTR,
    PreparedAttention,
)
from ...attention.stable import LOG
from ...runtime.capabilities import kernel_capabilities


H3_VIRTUAL_KV_STRATEGY = "h3_virtual_kv"
H3_VIRTUAL_KV_MODES = ("conservative", "fast", "residual")
H3_VIRTUAL_KV_RESIDUAL_SUBBLOCKS = 2
H3_VIRTUAL_KV_RESIDUAL_THRESHOLD = 1_000_000.0
_SOURCE_FRAMES = (0, 1, 1, 1, 1, 0, 1)
# H3's latent-time positions use spans [1, 4, 4, 4, 4] * 5/3.
# Expressing every position in units of the first span avoids depending on a
# model object or checkpoint-owned inverse-frequency buffer at attention time.
_VIRTUAL_TIME_UNITS = (0, 1, 5, 9, 13, 17, 18)
class _TensorOwner:
    def __init__(self, value: torch.Tensor):
        self._value = value

    def peek(self) -> torch.Tensor:
        if self._value is None:
            raise RuntimeError("attention tensor was already consumed")
        return self._value

    def take(self) -> torch.Tensor:
        value = self.peek()
        self._value = None
        return value


def _rotation_matrix(angles: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    return torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        *angles.shape, 2, 2
    ).to(dtype=dtype)


def _temporal_phase_basis(
    query_freqs: torch.Tensor,
    start: int,
    tokens_per_frame: int,
    rot_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if (
        query_freqs.ndim != 6
        or int(query_freqs.shape[1]) < start + 2 * tokens_per_frame
        or int(query_freqs.shape[-2]) != 2
        or int(query_freqs.shape[-1]) != 2
    ):
        raise RuntimeError("H3 virtual K/V requires a per-token rotation table")
    pairs = int(query_freqs.shape[-3])
    if rot_dim != pairs * 2 or pairs % 3:
        raise RuntimeError(
            "H3 virtual K/V requires the standard three-axis H3 RoPE layout"
        )
    temporal_pairs = pairs // 3
    first = query_freqs[:, start : start + tokens_per_frame]
    second = query_freqs[
        :, start + tokens_per_frame : start + 2 * tokens_per_frame
    ]
    first_temporal = first[..., :temporal_pairs, :, :].to(torch.float32)
    second_temporal = second[..., :temporal_pairs, :, :].to(torch.float32)
    phase0 = torch.atan2(first_temporal[..., 1, 0], first_temporal[..., 0, 0])
    phase1 = torch.atan2(second_temporal[..., 1, 0], second_temporal[..., 0, 0])
    delta = torch.atan2(torch.sin(phase1 - phase0), torch.cos(phase1 - phase0))
    spatial = first[..., temporal_pairs:, :, :]
    return phase0, delta, spatial, temporal_pairs


def _frequency_frame(
    phase: torch.Tensor,
    spatial: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.cat((_rotation_matrix(phase, dtype), spatial), dim=-3)


def _virtual_target_frequencies(
    query_freqs: torch.Tensor,
    start: int,
    stop: int,
    tokens_per_frame: int,
    rot_dim: int,
) -> torch.Tensor:
    phase0, delta, spatial, _ = _temporal_phase_basis(
        query_freqs, start, tokens_per_frame, rot_dim
    )
    frames = tuple(
        _frequency_frame(
            phase0 + float(units) * delta,
            spatial,
            query_freqs.dtype,
        )
        for units in _VIRTUAL_TIME_UNITS
    )
    target = torch.cat(frames, dim=1)
    return torch.cat(
        (query_freqs[:, :start], target, query_freqs[:, stop:]), dim=1
    )


def _target_video_geometry(request: PreparedAttention):
    layout = attention_semantic_layout(request.transformer_options)
    if layout is None or layout.provider != "minimax_h3":
        return None
    if layout.validate(request.query_tokens, request.key_tokens) is not None:
        # Token-refiner calls can observe a layout left by the previous H3
        # block.  They are ordinary dense attention and must not be modified.
        return None
    targets = tuple(
        segment for segment in layout.key_segments if segment.role == "target_video"
    )
    if len(targets) != 1:
        raise RuntimeError("H3 virtual K/V requires one target-video segment")
    target = targets[0]
    topologies = tuple(
        topology
        for topology in layout.topologies
        if topology.topology_id == target.topology_id
    )
    if len(topologies) != 1:
        raise RuntimeError("H3 virtual K/V target-video topology is unavailable")
    topology = topologies[0]
    tokens_per_frame = int(topology.tokens_per_frame)
    physical_frames = (target.stop - target.start) // tokens_per_frame
    if physical_frames != 2 or target.stop - target.start != 2 * tokens_per_frame:
        raise RuntimeError(
            "Configure H3 Static Virtual KV requires a 5-frame input "
            "(exactly 2 H3 latent-time slices); received "
            f"{physical_frames} latent-time slices"
        )
    return target.start, target.stop, tokens_per_frame


def _expand_target(
    tensor: torch.Tensor,
    start: int,
    stop: int,
    tokens_per_frame: int,
) -> torch.Tensor:
    batch, heads, _, head_dim = tensor.shape
    target = tensor[:, :, start:stop].reshape(
        batch, heads, 2, tokens_per_frame, head_dim
    )
    indices = torch.tensor(_SOURCE_FRAMES, dtype=torch.long, device=tensor.device)
    target = target.index_select(2, indices).reshape(
        batch, heads, 7 * tokens_per_frame, head_dim
    )
    return torch.cat((tensor[:, :, :start], target, tensor[:, :, stop:]), dim=2)


def _logical_source_indices(
    physical_tokens: int,
    start: int,
    stop: int,
    tokens_per_frame: int,
    device: torch.device,
) -> torch.Tensor:
    prefix = torch.arange(start, dtype=torch.int32, device=device)
    target = torch.cat(
        tuple(
            torch.arange(
                start + source * tokens_per_frame,
                start + (source + 1) * tokens_per_frame,
                dtype=torch.int32,
                device=device,
            )
            for source in _SOURCE_FRAMES
        )
    )
    suffix = torch.arange(stop, physical_tokens, dtype=torch.int32, device=device)
    return torch.cat((prefix, target, suffix))


def _cached_virtual_inputs(
    request: PreparedAttention,
    query_freqs: torch.Tensor,
    start: int,
    stop: int,
    tokens_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    options = request.transformer_options
    key = (
        int(query_freqs.data_ptr()),
        tuple(query_freqs.shape),
        int(getattr(query_freqs, "_version", 0)),
        start,
        stop,
        tokens_per_frame,
        request.key_tokens,
        request.qk_transform.rot_dim,
    )
    cache = None
    if isinstance(options, dict):
        cache = options.setdefault("turing_utils_h3_virtual_kv_cache", {})
        cached = cache.get(key)
        if cached is not None:
            return cached
    key_freqs = _virtual_target_frequencies(
        query_freqs,
        start,
        stop,
        tokens_per_frame,
        request.qk_transform.rot_dim,
    )
    source_indices = _logical_source_indices(
        request.key_tokens,
        start,
        stop,
        tokens_per_frame,
        query_freqs.device,
    )
    result = (key_freqs, source_indices)
    if cache is not None:
        cache.clear()
        cache[key] = result
    return result


def make_h3_virtual_kv_override(
    dense_override: Callable,
    *,
    mode: str = "conservative",
) -> Callable:
    mode = str(mode).strip().lower()
    if mode not in H3_VIRTUAL_KV_MODES:
        raise ValueError(f"unsupported H3 virtual K/V mode: {mode}")
    dense_executor = getattr(dense_override, "prepared_attention_executor", None)
    if not callable(dense_executor):
        raise RuntimeError(
            "H3 virtual K/V requires a dense backend with prepared-attention support"
        )

    def attention_override(original: Callable, *args, **kwargs):
        # Non-H3 and compatibility calls remain on the loader-selected dense
        # backend. H3 blocks use the prepared executor below.
        return dense_override(original, *args, **kwargs)

    def prepared_executor(request: PreparedAttention) -> AttentionExecutionOutcome:
        geometry = _target_video_geometry(request)
        if geometry is None:
            return dense_executor(request)
        if request.mask is not None or request.is_causal:
            return AttentionExecutionOutcome.unsupported(
                "H3 virtual K/V supports only unmasked non-causal dense attention"
            )
        query_freqs = request.qk_transform.freqs
        if not torch.is_tensor(query_freqs):
            return AttentionExecutionOutcome.unsupported(
                "H3 virtual K/V requires the H3 RoPE table"
            )
        start, stop, tokens_per_frame = geometry
        key_freqs, source_indices = _cached_virtual_inputs(
            request,
            query_freqs,
            start,
            stop,
            tokens_per_frame,
        )
        query, key, value = request.consume_qkv()
        mapped_executor = getattr(dense_executor, MAPPED_KV_EXECUTOR_ATTR, None)
        residual_executor = getattr(
            dense_executor, MAPPED_RESIDUAL_EXECUTOR_ATTR, None
        )
        residual_capability = getattr(
            residual_executor,
            MAPPED_RESIDUAL_CAPABILITY_ATTR,
            "mapped_sparse_kv",
        )
        mapped_exact = bool(
            mode == "fast"
            and callable(mapped_executor)
            and kernel_capabilities().supports("mapped_kv").supported
        )
        mapped_residual = bool(
            mode == "residual"
            and callable(residual_executor)
            and kernel_capabilities().supports(residual_capability).supported
        )
        mapped = mapped_exact or mapped_residual
        if not mapped:
            key = _expand_target(key, start, stop, tokens_per_frame)
            value = _expand_target(value, start, stop, tokens_per_frame)
        rotary = dataclasses.replace(
            request.qk_transform.rotary,
            key_freqs=key_freqs,
        )
        transform = dataclasses.replace(request.qk_transform, rotary=rotary)
        virtual = PreparedAttention.from_hnd(
            _TensorOwner(query),
            _TensorOwner(key),
            _TensorOwner(value),
            heads=request.heads,
            qk_transform=transform,
            transformer_options=request.transformer_options,
            scale=request.scale,
            mask=None,
            is_causal=False,
            low_precision_attention=request.low_precision_attention,
            skip_output_reshape=request.skip_output_reshape,
            observer_requirements=request.observer_requirements,
        )
        if mapped_exact:
            return mapped_executor(virtual, source_indices)
        if mapped_residual:
            logical_target_stop = start + 7 * tokens_per_frame
            exact_kv_ranges = [(0, start + 2 * tokens_per_frame)]
            if logical_target_stop < int(source_indices.numel()):
                exact_kv_ranges.append(
                    (logical_target_stop, int(source_indices.numel()))
                )
            outcome = residual_executor(
                virtual,
                source_indices,
                exact_kv_ranges=tuple(exact_kv_ranges),
                residual_subblocks=H3_VIRTUAL_KV_RESIDUAL_SUBBLOCKS,
                routing_threshold=H3_VIRTUAL_KV_RESIDUAL_THRESHOLD,
            )
            if outcome.supported:
                return outcome
            # Prepared executors must reject before consuming. Preserve exact
            # behavior if an otherwise capable runtime declines this shape.
            query, key, value = virtual.consume_qkv()
            key = _expand_target(key, start, stop, tokens_per_frame)
            value = _expand_target(value, start, stop, tokens_per_frame)
            virtual = PreparedAttention.from_hnd(
                _TensorOwner(query),
                _TensorOwner(key),
                _TensorOwner(value),
                heads=request.heads,
                qk_transform=transform,
                transformer_options=request.transformer_options,
                scale=request.scale,
                mask=None,
                is_causal=False,
                low_precision_attention=request.low_precision_attention,
                skip_output_reshape=request.skip_output_reshape,
                observer_requirements=request.observer_requirements,
            )
        return dense_executor(virtual)

    prepared_executor.capabilities = getattr(dense_executor, "capabilities", None)
    prepared_executor.turing_utils_h3_virtual_kv_mode = mode
    residual_executor = getattr(dense_executor, MAPPED_RESIDUAL_EXECUTOR_ATTR, None)
    prepared_executor.turing_utils_h3_virtual_kv_mapped_capability = (
        getattr(
            residual_executor,
            MAPPED_RESIDUAL_CAPABILITY_ATTR,
            "mapped_sparse_kv",
        )
        if mode == "residual"
        else "mapped_kv"
    )
    prepared_executor.turing_utils_h3_virtual_kv_numeric_backend = getattr(
        dense_override,
        "turing_utils_attention_backend",
        "sdpa",
    )
    prepared_executor.turing_utils_h3_virtual_kv_mapped_available = bool(
        (
            mode == "fast"
            and callable(getattr(dense_executor, MAPPED_KV_EXECUTOR_ATTR, None))
        )
        or (
            mode == "residual"
            and callable(
                residual_executor
            )
        )
    )
    attention_override.prepared_attention_executor = prepared_executor
    attention_override.turing_utils_attention_backend = H3_VIRTUAL_KV_STRATEGY
    attention_override.turing_utils_attention_implementation = (
        f"h3_virtual_kv:{mode}"
    )
    attention_override.turing_utils_dense_implementation = getattr(
        dense_override,
        "turing_utils_attention_implementation",
        "selected_dense_backend",
    )
    attention_override.turing_utils_h3_virtual_kv_mode = mode
    return attention_override


def apply_h3_virtual_kv(model, *, mode: str = "conservative"):
    mode = str(mode).strip().lower()
    runtime = attention_base_runtime(model, use_w8a8=None)
    if (
        runtime.dense_implementation.startswith("bundled_turing_")
        and not kernel_capabilities().supports("asymmetric_qk_rope").supported
    ):
        raise RuntimeError(
            "Configure H3 Static Virtual KV with bundled Sage/W8A8 requires "
            "comfyui-turing-utils-kernel 0.38.0 or newer; rebuild the kernel "
            "after updating the repository"
        )
    if (
        mode == "fast"
        and runtime.dense_backend == "w8a8"
        and not kernel_capabilities().supports("mapped_kv").supported
    ):
        LOG.warning(
            "H3 fast virtual K/V needs kernel 0.39.0 mapped-K/V support; "
            "using the exact materialized path until the native kernel is rebuilt"
        )
    residual_capability = (
        "mapped_sparse_kv"
        if runtime.dense_backend == "w8a8"
        else "mapped_sparse_fp16_kv"
    )
    if mode == "residual" and not kernel_capabilities().supports(
        residual_capability
    ).supported:
        LOG.warning(
            "H3 residual virtual K/V needs kernel 0.41.0 mapped-Sol support "
            "for dense backend %s; "
            "using the exact materialized path until the native kernel is rebuilt",
            runtime.dense_backend,
        )
    override = make_h3_virtual_kv_override(runtime.dense_override, mode=mode)
    installed = install_attention_strategy(
        model,
        override,
        strategy="H3 static virtual K/V",
        backend=H3_VIRTUAL_KV_STRATEGY,
        implementation=f"h3_virtual_kv:{mode}",
        runtime_config=runtime,
    )
    LOG.info(
        "H3 static virtual K/V enabled: mode=%s physical_frames=5 "
        "physical_latent_t=2 virtual_frames=22 virtual_latent_t=7 "
        "dense_backend=%s residual_virtual_groups=%s",
        mode,
        runtime.dense_backend,
        5 if mode == "residual" else 0,
    )
    return installed.model


__all__ = [
    "H3_VIRTUAL_KV_MODES",
    "H3_VIRTUAL_KV_RESIDUAL_SUBBLOCKS",
    "H3_VIRTUAL_KV_RESIDUAL_THRESHOLD",
    "H3_VIRTUAL_KV_STRATEGY",
    "apply_h3_virtual_kv",
    "make_h3_virtual_kv_override",
]
