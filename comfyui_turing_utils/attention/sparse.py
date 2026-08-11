"""Experimental Sol and frame-structured sparse attention policies."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import torch

from .layout import (
    ATTENTION_LAYOUT_REQUIREMENT_KEY,
    has_complete_attention_layout,
)
from .stable import (
    FRAME_SPARSE_GLOBAL_ANCHOR_STRIDE,
    FRAME_SPARSE_PATTERN,
    FRAME_SPARSE_QUALITY_PROFILE,
    FRAME_SPARSE_RADIAL_MAX_TEMPORAL_STRIDE,
    FRAME_SPARSE_RADIAL_SPATIAL_RADIUS,
    FRAME_SPARSE_SINK_FRAMES,
    FRAME_SPARSE_TEMPORAL_WINDOW_FRAMES,
    LOG,
    SPARSE_AUTO_MIN_SEQUENCE,
    SPARSE_LAYOUT_KEY,
    SPARSE_PREFIX_POLICY,
    SPARSE_REFERENCE_AUDIO,
    SPARSE_REFERENCE_IMAGE,
    SPARSE_REFERENCE_VIDEO,
    SPARSE_ROUTING_THRESHOLD,
    SPARSE_SKIPPED_RESIDUAL,
    SUPPORTED_INPUT_DTYPES,
    _LOGGED_FRAME_SPARSE_KERNELS,
    _LOGGED_SPARSE_DENSE_REASONS,
    _LOGGED_SPARSE_KERNELS,
    _frame_sparse_sageattn,
    _reshape_qkv,
    _sol_sparse_sageattn,
    is_supported_turing_device,
    turing_sage_attention,
)


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


def _coalesce_token_ranges(ranges, sequence_limit: int) -> tuple[tuple[int, int], ...]:
    normalized = sorted(
        (max(0, int(start)), min(sequence_limit, int(stop)))
        for start, stop in ranges
        if int(stop) > 0 and int(start) < sequence_limit and int(stop) > int(start)
    )
    merged: list[list[int]] = []
    for start, stop in normalized:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return tuple((start, stop) for start, stop in merged)


def _sparse_protected_ranges(
    policy: str,
    manual_tokens: int,
    transformer_options,
    sequence_limit: int,
    *,
    sparse_reference_image: bool,
    sparse_reference_video: bool,
    sparse_reference_audio: bool,
) -> tuple[tuple[int, int], ...]:
    """Return token spans whose Query stays dense and whose KV stays exact."""
    if policy == "none":
        return ()
    if policy == "manual":
        stop = min(max(int(manual_tokens), 0), sequence_limit)
        return ((0, stop),) if stop else ()
    layout = (
        transformer_options.get(SPARSE_LAYOUT_KEY)
        if isinstance(transformer_options, dict)
        else None
    )
    segments = layout.get("segments") if isinstance(layout, dict) else None
    if not isinstance(segments, tuple):
        prefix = _sparse_prefix_tokens(
            policy, manual_tokens, transformer_options, sequence_limit
        )
        return ((0, prefix),) if prefix else ()
    sparse_roles = {"target_video"}
    if sparse_reference_image:
        sparse_roles.add("reference_image")
    if sparse_reference_video:
        sparse_roles.add("reference_video")
    if sparse_reference_audio:
        sparse_roles.add("reference_audio")
    return _coalesce_token_ranges(
        (
            (start, stop)
            for start, stop, role in segments
            if role not in sparse_roles
        ),
        sequence_limit,
    )


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


def _required_sparse_layout_missing(transformer_options, sequence_length: int) -> bool:
    if not isinstance(transformer_options, dict):
        return False
    requirement = transformer_options.get(ATTENTION_LAYOUT_REQUIREMENT_KEY)
    if not isinstance(requirement, str) or not requirement:
        return False
    return not has_complete_attention_layout(
        transformer_options,
        sequence_length,
        provider=requirement,
    )


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
    skipped_residual: str = SPARSE_SKIPPED_RESIDUAL,
    sparse_reference_image: bool = SPARSE_REFERENCE_IMAGE,
    sparse_reference_video: bool = SPARSE_REFERENCE_VIDEO,
    sparse_reference_audio: bool = SPARSE_REFERENCE_AUDIO,
    debug_route_density: bool = False,
    debug_route_keys: set[tuple] | None = None,
    debug_route_state: dict[tuple, list[tuple[torch.Tensor, int, int]]] | None = None,
    debug_context: dict | None = None,
    use_w8a8: bool = False,
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
    transformer_options = kwargs.get("transformer_options")
    if _required_sparse_layout_missing(
        transformer_options,
        min(q.shape[2], k.shape[2]),
    ):
        return dense("required MiniMax H3 attention layout metadata is unavailable")
    skipped_residual = str(skipped_residual).strip().lower()
    residual_subblocks = {"1x64": 1, "2x32": 2}.get(skipped_residual)
    if residual_subblocks is None:
        raise ValueError("skipped_residual must be 1x64 or 2x32")
    protected_ranges = _sparse_protected_ranges(
        prefix_policy,
        manual_prefix_tokens,
        transformer_options,
        min(q.shape[2], k.shape[2]),
        sparse_reference_image=bool(sparse_reference_image),
        sparse_reference_video=bool(sparse_reference_video),
        sparse_reference_audio=bool(sparse_reference_audio),
    )
    if protected_ranges and q.shape[2] != k.shape[2]:
        return dense("protected Query/KV ranges require equal Q/K sequence lengths")
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
        protected_ranges,
        routing_threshold,
        residual_subblocks,
        bool(sparse_reference_image),
        bool(sparse_reference_video),
        bool(sparse_reference_audio),
        bool(use_w8a8),
    )
    if kernel_key not in _LOGGED_SPARSE_KERNELS:
        LOG.info(
            "Experimental Turing Sol sparse attention active: dtype=%s Q=%s K=%s "
            "min_sequence=%d prefix_policy=%s protected_ranges=%s "
            "selected_qk=int8 score_domain=int8_consistent threshold=%.2f "
            "skipped_residual=%s local_radius=1 "
            "sparse_reference=(image=%s,video=%s,audio=%s) pv=%s",
            input_dtype,
            tuple(q.shape),
            tuple(k.shape),
            effective_min_sequence,
            prefix_policy,
            protected_ranges,
            routing_threshold,
            skipped_residual,
            bool(sparse_reference_image),
            bool(sparse_reference_video),
            bool(sparse_reference_audio),
            "w8a8" if use_w8a8 else "fp16",
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
        dense_query_ranges=protected_ranges,
        exact_kv_ranges=protected_ranges,
        threshold_sigma=routing_threshold,
        residual_subblocks=residual_subblocks,
        return_stats=collect_route_stats,
        use_w8a8=bool(use_w8a8),
    )
    if collect_route_stats:
        output, selected_device, possible_blocks = sparse_result
        try:
            protected_query_tokens = sum(stop - start for start, stop in protected_ranges)
            sparse_query_tokens = q.shape[2] - protected_query_tokens
            if aggregate_route_stats:
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
                        "protected_q=%d local=1 residual=%s",
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
                        protected_query_tokens,
                        skipped_residual,
                    )
                    del debug_route_state[aggregate_key]
            else:
                selected_blocks = int(selected_device.item())
                LOG.warning(
                    "[Turing sparse debug] Q=%d Qsparse=%d K=%d Hq=%d Hkv=%d selected=%d/%d "
                    "density=%.4f threshold=%.2f protected_q=%d local=1 "
                    "residual=%s step=%s/%s layer=%s/%s",
                    q.shape[2],
                    sparse_query_tokens,
                    k.shape[2],
                    q.shape[1],
                    k.shape[1],
                    selected_blocks,
                    possible_blocks,
                    selected_blocks / possible_blocks if possible_blocks else 0.0,
                    routing_threshold,
                    protected_query_tokens,
                    skipped_residual,
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
    if _required_sparse_layout_missing(transformer_options, q.shape[2]):
        return dense("required MiniMax H3 attention layout metadata is unavailable")
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
