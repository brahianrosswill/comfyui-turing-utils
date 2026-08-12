"""Attention overrides and loader-independent ModelPatcher installation."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from .layout import ATTENTION_LAYOUT_REQUIREMENT_KEY, ensure_attention_layout_provider
from .sparse import (
    _resolve_frame_sparse_quality_profile,
    _sparse_dense_layer,
    _sparse_dense_schedule,
    inspect_frame_attention_call,
    inspect_sol_attention_call,
    prequantize_turing_frame_attention,
    prequantize_turing_sol_attention,
    turing_frame_attention_from_prequantized,
    turing_frame_sparse_attention,
    turing_sol_attention_from_prequantized,
    turing_sol_sparse_attention,
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
    SPARSE_DENSE_PREFIX_LAYERS,
    SPARSE_DENSE_PREFIX_STEPS,
    SPARSE_DENSE_SUFFIX_LAYERS,
    SPARSE_DENSE_SUFFIX_STEPS,
    SPARSE_LAYOUT_KEY,
    SPARSE_PREFIX_POLICY,
    SPARSE_REFERENCE_AUDIO,
    SPARSE_REFERENCE_IMAGE,
    SPARSE_REFERENCE_VIDEO,
    SPARSE_ROUTING_THRESHOLD,
    SPARSE_SKIPPED_RESIDUAL,
    SPARSE_USE_W8A8,
    _BACKENDS,
    _comfy_attention_function,
    _select_attention_backend,
    bundled_available,
    bundled_frame_sparse_available,
    bundled_sparse_available,
    bundled_w8a8_available,
    is_supported_turing_device,
    inspect_turing_attention_call,
    normalize_attention_backend,
    prequantize_turing_attention,
    preflight_bundled,
    preflight_bundled_frame_sparse,
    preflight_bundled_sparse,
    preflight_bundled_w8a8,
    split_prequantization_available,
    turing_attention_from_prequantized,
    turing_sage_attention,
    turing_w8a8_attention,
)


def _default_attention_fallback() -> Callable:
    from comfy.ldm.modules import attention as comfy_attention

    return comfy_attention.optimized_attention


def _container_fallback(fallback: Callable, q, k, v, heads: int, *args, **kwargs):
    return _dtype_compatible_fallback(
        fallback,
        q.take(),
        k.take(),
        v.take(),
        heads,
        *args,
        **kwargs,
    )


def _make_dense_container_function(kernel: str) -> Callable:
    fallback = _default_attention_fallback()

    def container_function(
        q,
        k,
        v,
        heads: int,
        mask=None,
        attn_precision=None,
        skip_reshape: bool = False,
        skip_output_reshape: bool = False,
        **kwargs,
    ):
        call, reason = inspect_turing_attention_call(
            q.peek(),
            k.peek(),
            v.peek(),
            heads,
            mask=mask,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            enable_gqa=bool(kwargs.get("enable_gqa", False)),
            low_precision_attention=kwargs.get("low_precision_attention", True),
            is_causal=bool(kwargs.get("is_causal", False)),
            kernel=kernel,
            require_long_sequence=True,
        )
        if reason is not None:
            return _container_fallback(
                fallback,
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
        query = q.take()
        key = k.take()
        value = v.take()
        quantized = prequantize_turing_attention(
            query,
            key,
            value,
            call,
            kernel=kernel,
            scale=kwargs.get("scale"),
            is_causal=bool(kwargs.get("is_causal", False)),
        )
        del query, key, value
        return turing_attention_from_prequantized(quantized, kernel=kernel)

    return container_function


def _uses_bundled_turing_sage(option: str, device: torch.device | None) -> bool:
    option = normalize_attention_backend(option)
    return bool(
        device is not None
        and is_supported_turing_device(device)
        and option in {"auto", "sage_attn", "w8a8"}
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
    if option == "w8a8" and not bundled_turing:
        raise RuntimeError(
            "The W8A8 attention backend is an experimental exact-sm75 kernel; "
            "select sage_attn, flash_attn, or sdpa on other GPUs."
        )
    if bundled_turing:
        if option == "w8a8" and not bundled_w8a8_available():
            raise RuntimeError(
                "The bundled Turing W8A8 extension is unavailable. "
                "Rebuild comfyui-turing-utils-kernel 0.18.0 or newer with sm75 enabled."
            )
        if not bundled_available():
            raise RuntimeError(
                "The bundled Turing Sage extensions are unavailable. "
                "Rebuild comfyui-turing-utils-kernel with COMFYUI_TURING_UTILS_ARCH_LIST including 7.5."
            )
        if option == "w8a8":
            preflight_bundled_w8a8(device)
        else:
            preflight_bundled(device)
        backend = _BACKENDS[option if option == "w8a8" else "sage_attn"]
        target = turing_w8a8_attention if option == "w8a8" else turing_sage_attention
        implementation = (
            "bundled_turing_w8a8_experimental"
            if option == "w8a8"
            else "bundled_turing_sage"
        )
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
    if bundled_turing and split_prequantization_available():
        attention_override.container_function = _make_dense_container_function(
            "w8a8" if option == "w8a8" else "sage"
        )
    return attention_override


def make_sparse_attention_override(
    device: torch.device,
    min_sequence_tokens: int = 0,
    routing_threshold: float = SPARSE_ROUTING_THRESHOLD,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    skipped_residual: str = SPARSE_SKIPPED_RESIDUAL,
    sparse_reference_image: bool = SPARSE_REFERENCE_IMAGE,
    sparse_reference_video: bool = SPARSE_REFERENCE_VIDEO,
    sparse_reference_audio: bool = SPARSE_REFERENCE_AUDIO,
    dense_prefix_steps: int = SPARSE_DENSE_PREFIX_STEPS,
    dense_suffix_steps: int = SPARSE_DENSE_SUFFIX_STEPS,
    dense_prefix_layers: int = SPARSE_DENSE_PREFIX_LAYERS,
    dense_suffix_layers: int = SPARSE_DENSE_SUFFIX_LAYERS,
    debug_route_density: bool = False,
    use_w8a8: bool = SPARSE_USE_W8A8,
) -> Callable:
    min_sequence_tokens = int(min_sequence_tokens)
    routing_threshold = float(routing_threshold)
    prefix_policy = str(prefix_policy).strip().lower()
    manual_prefix_tokens = int(manual_prefix_tokens)
    skipped_residual = str(skipped_residual).strip().lower()
    sparse_reference_image = bool(sparse_reference_image)
    sparse_reference_video = bool(sparse_reference_video)
    sparse_reference_audio = bool(sparse_reference_audio)
    dense_prefix_steps = int(dense_prefix_steps)
    dense_suffix_steps = int(dense_suffix_steps)
    dense_prefix_layers = int(dense_prefix_layers)
    dense_suffix_layers = int(dense_suffix_layers)
    debug_route_density = bool(debug_route_density)
    use_w8a8 = bool(use_w8a8)
    if min_sequence_tokens < 0:
        raise ValueError("min_sequence_tokens must be non-negative")
    if not math.isfinite(routing_threshold):
        raise ValueError("routing_threshold must be finite")
    if prefix_policy not in {"auto", "none", "manual"}:
        raise ValueError("prefix_policy must be auto, none, or manual")
    if manual_prefix_tokens < 0:
        raise ValueError("manual_prefix_tokens must be non-negative")
    if skipped_residual not in {"1x64", "2x32"}:
        raise ValueError("skipped_residual must be 1x64 or 2x32")
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
            "Rebuild comfyui-turing-utils-kernel 0.17.0 or newer with sm75 enabled."
        )
    preflight_bundled(device)
    preflight_bundled_sparse(device)
    if use_w8a8:
        if not bundled_w8a8_available():
            raise RuntimeError(
                "Sol W8A8 requires comfyui-turing-utils-kernel 0.18.0 or newer"
            )
        preflight_bundled_w8a8(device)
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
            dense_attention = turing_w8a8_attention if use_w8a8 else turing_sage_attention
            return dense_attention(fallback, *args, **kwargs)
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
            skipped_residual=skipped_residual,
            sparse_reference_image=sparse_reference_image,
            sparse_reference_video=sparse_reference_video,
            sparse_reference_audio=sparse_reference_audio,
            debug_route_density=debug_route_density,
            debug_route_keys=debug_route_keys if debug_route_density else None,
            debug_route_state=debug_route_state if debug_route_density else None,
            debug_context=debug_context,
            use_w8a8=use_w8a8,
            **kwargs,
        )

    if split_prequantization_available():
        dense_container = _make_dense_container_function(
            "w8a8" if use_w8a8 else "sage"
        )

        def container_function(
            q,
            k,
            v,
            heads: int,
            mask=None,
            attn_precision=None,
            skip_reshape: bool = False,
            skip_output_reshape: bool = False,
            **kwargs,
        ):
            transformer_options = kwargs.get("transformer_options")
            if _sparse_dense_schedule(
                transformer_options,
                dense_prefix_steps,
                dense_suffix_steps,
                schedule_state,
            ) or _sparse_dense_layer(
                transformer_options,
                dense_prefix_layers,
                dense_suffix_layers,
            ):
                return dense_container(
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            sol_call, reason = inspect_sol_attention_call(
                q.peek(),
                k.peek(),
                v.peek(),
                heads,
                mask=mask,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                min_sequence_tokens=min_sequence_tokens,
                prefix_policy=prefix_policy,
                manual_prefix_tokens=manual_prefix_tokens,
                skipped_residual=skipped_residual,
                sparse_reference_image=sparse_reference_image,
                sparse_reference_video=sparse_reference_video,
                sparse_reference_audio=sparse_reference_audio,
                transformer_options=transformer_options,
                kwargs=kwargs,
            )
            if reason is not None:
                return dense_container(
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            query = q.take()
            key = k.take()
            value = v.take()
            quantized = prequantize_turing_sol_attention(
                query,
                key,
                value,
                sol_call,
                routing_threshold=routing_threshold,
                scale=kwargs.get("scale"),
                use_w8a8=use_w8a8,
            )
            del query, key, value
            debug_key = (
                sol_call.attention.input_dtype,
                sol_call.attention.query_tokens,
                sol_call.attention.key_tokens,
                sol_call.protected_ranges,
                routing_threshold,
                sol_call.residual_subblocks,
                use_w8a8,
            )
            collect_stats = debug_route_density and debug_key not in debug_route_keys
            result = turing_sol_attention_from_prequantized(
                quantized,
                return_stats=collect_stats,
            )
            if collect_stats:
                output, selected, possible = result
                selected_blocks = int(selected.item())
                LOG.warning(
                    "[Turing sparse debug] selected=%d/%d density=%.4f Q=%d K=%d threshold=%.2f residual=%s",
                    selected_blocks,
                    possible,
                    selected_blocks / possible if possible else 0.0,
                    sol_call.attention.query_tokens,
                    sol_call.attention.key_tokens,
                    routing_threshold,
                    skipped_residual,
                )
                debug_route_keys.add(debug_key)
                return output
            return result

        attention_override.container_function = container_function

    attention_override.turing_utils_attention_backend = "sol_sparse_attn"
    attention_override.turing_utils_attention_implementation = "bundled_turing_sol_sparse_experimental"
    return attention_override


def apply_sparse_attention_patch(
    model,
    min_sequence_tokens: int = 0,
    routing_threshold: float = SPARSE_ROUTING_THRESHOLD,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    skipped_residual: str = SPARSE_SKIPPED_RESIDUAL,
    sparse_reference_image: bool = SPARSE_REFERENCE_IMAGE,
    sparse_reference_video: bool = SPARSE_REFERENCE_VIDEO,
    sparse_reference_audio: bool = SPARSE_REFERENCE_AUDIO,
    dense_prefix_steps: int = SPARSE_DENSE_PREFIX_STEPS,
    dense_suffix_steps: int = SPARSE_DENSE_SUFFIX_STEPS,
    dense_prefix_layers: int = SPARSE_DENSE_PREFIX_LAYERS,
    dense_suffix_layers: int = SPARSE_DENSE_SUFFIX_LAYERS,
    debug_route_density: bool = False,
    use_w8a8: bool = SPARSE_USE_W8A8,
):
    patched = model.clone()
    layout_status = ensure_attention_layout_provider(patched)
    override = make_sparse_attention_override(
        patched.load_device,
        min_sequence_tokens=min_sequence_tokens,
        routing_threshold=routing_threshold,
        prefix_policy=prefix_policy,
        manual_prefix_tokens=manual_prefix_tokens,
        skipped_residual=skipped_residual,
        sparse_reference_image=sparse_reference_image,
        sparse_reference_video=sparse_reference_video,
        sparse_reference_audio=sparse_reference_audio,
        dense_prefix_steps=dense_prefix_steps,
        dense_suffix_steps=dense_suffix_steps,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
        debug_route_density=debug_route_density,
        use_w8a8=use_w8a8,
    )
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    if layout_status.required:
        transformer_options[ATTENTION_LAYOUT_REQUIREMENT_KEY] = layout_status.model_kind
        if not layout_status.installed:
            LOG.warning(
                "%s sparse attention will stay dense because its runtime "
                "layout provider could not be installed: %s",
                layout_status.model_kind,
                layout_status.reason,
            )
    transformer_options["optimized_attention_override"] = override
    transformer_options["turing_utils_attention_backend"] = "sol_sparse_attn"
    transformer_options["turing_utils_attention_implementation"] = (
        "bundled_turing_sol_sparse_experimental"
    )
    LOG.info(
        "Sol sparse attention patch enabled: threshold=%.2f "
        "prefix_policy=%s manual_prefix=%d local_radius=1 "
        "skipped_residual=%s sparse_reference=(image=%s,video=%s,audio=%s) "
        "dense_prefix_steps=%d dense_suffix_steps=%d "
        "dense_prefix_layers=%d dense_suffix_layers=%d "
        "dense_backend=%s pv_backend=%s debug_route_density=%s",
        routing_threshold,
        prefix_policy,
        manual_prefix_tokens,
        skipped_residual,
        sparse_reference_image,
        sparse_reference_video,
        sparse_reference_audio,
        dense_prefix_steps,
        dense_suffix_steps,
        dense_prefix_layers,
        dense_suffix_layers,
        "bundled_turing_w8a8" if use_w8a8 else "bundled_turing_sage",
        "u8xs8_tensorcore" if use_w8a8 else "fp16_tensorcore",
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

    if split_prequantization_available():
        dense_container = _make_dense_container_function("sage")

        def container_function(
            q,
            k,
            v,
            heads: int,
            mask=None,
            attn_precision=None,
            skip_reshape: bool = False,
            skip_output_reshape: bool = False,
            **kwargs,
        ):
            transformer_options = kwargs.get("transformer_options")
            if _sparse_dense_schedule(
                transformer_options,
                dense_prefix_steps,
                dense_suffix_steps,
                schedule_state,
            ) or _sparse_dense_layer(
                transformer_options,
                dense_prefix_layers,
                dense_suffix_layers,
            ):
                return dense_container(
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            frame_call, reason = inspect_frame_attention_call(
                q.peek(),
                k.peek(),
                v.peek(),
                heads,
                mask=mask,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                prefix_policy=prefix_policy,
                manual_prefix_tokens=manual_prefix_tokens,
                global_anchor_stride=global_anchor_stride,
                rotate_global_anchors=rotate_global_anchors,
                sparse_pattern=sparse_pattern,
                radial_max_temporal_stride=radial_max_temporal_stride,
                transformer_options=transformer_options,
                kwargs=kwargs,
            )
            if reason is not None:
                return dense_container(
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            query = q.take()
            key = k.take()
            value = v.take()
            quantized = prequantize_turing_frame_attention(
                query,
                key,
                value,
                frame_call,
                scale=kwargs.get("scale"),
                temporal_window_frames=temporal_window_frames,
                global_anchor_stride=global_anchor_stride,
                sink_frames=sink_frames,
                sparse_pattern=sparse_pattern,
                radial_spatial_radius=radial_spatial_radius,
                radial_max_temporal_stride=radial_max_temporal_stride,
            )
            del query, key, value
            return turing_frame_attention_from_prequantized(
                quantized,
                return_schedule_density=False,
            )

        attention_override.container_function = container_function

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
    layout_status = ensure_attention_layout_provider(patched)
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
    if layout_status.required:
        transformer_options[ATTENTION_LAYOUT_REQUIREMENT_KEY] = layout_status.model_kind
        if not layout_status.installed:
            LOG.warning(
                "%s frame-sparse attention will stay dense because its runtime "
                "layout provider could not be installed: %s",
                layout_status.model_kind,
                layout_status.reason,
            )
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
