"""Attention overrides and loader-independent ModelPatcher installation."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from .layout import (
    ATTENTION_LAYOUT_REQUIREMENT_KEY,
    attention_semantic_layout,
    ensure_attention_layout_provider,
)
from .integration import ensure_prepared_attention_sites
from .protocol import (
    ATTENTION_EXECUTOR_KEY,
    AttentionBackendCapabilities,
    AttentionExecutionOutcome,
    PreparedAttention,
)
from .sparse import (
    _sparse_dense_layer,
    _sparse_dense_schedule,
    inspect_sol_attention_call,
    prequantize_turing_sol_attention,
    prequantize_turing_sol_attention_from_qk,
    turing_sol_attention_from_prequantized,
    turing_sol_sparse_attention,
)
from .stable import (
    LOG,
    SPARSE_DENSE_PREFIX_LAYERS,
    SPARSE_DENSE_PREFIX_STEPS,
    SPARSE_DENSE_SUFFIX_LAYERS,
    SPARSE_DENSE_SUFFIX_STEPS,
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
    bundled_sparse_available,
    bundled_w8a8_available,
    fused_qk_preprocessing_available,
    is_supported_turing_device,
    inspect_turing_attention_call,
    normalize_attention_backend,
    prequantize_turing_attention,
    prequantize_turing_attention_from_qk,
    prequantize_turing_qk,
    preflight_bundled,
    preflight_bundled_sparse,
    preflight_bundled_w8a8,
    split_prequantization_available,
    turing_attention_from_prequantized,
    turing_sage_attention,
    turing_w8a8_attention,
)
from ..profiling import CUDA_PHASE_PROFILER


def _profiled(phase: str, function: Callable, /, *args, **kwargs):
    if CUDA_PHASE_PROFILER.enabled:
        return CUDA_PHASE_PROFILER.call(phase, function, *args, **kwargs)
    return function(*args, **kwargs)


def _attention_layer_metadata(transformer_options) -> tuple[int | None, int | None]:
    layout = attention_semantic_layout(transformer_options)
    if layout is not None:
        return layout.layer_index, layout.layer_count
    raw = (
        transformer_options.get("turing_utils_attention_layout")
        if isinstance(transformer_options, dict)
        else None
    )
    if not isinstance(raw, dict):
        return None, None
    layer_index = raw.get("layer_index")
    layer_count = raw.get("layer_count")
    return (
        layer_index if isinstance(layer_index, int) and not isinstance(layer_index, bool) else None,
        layer_count if isinstance(layer_count, int) and not isinstance(layer_count, bool) else None,
    )


def _prepared_call_mismatch(request: PreparedAttention, call) -> str | None:
    expected = (
        request.heads,
        request.kv_heads,
        request.head_dim,
        request.query_tokens,
        request.key_tokens,
        request.tensor_layout,
        request.skip_output_reshape,
    )
    actual = (
        call.heads,
        call.kv_heads,
        call.head_dim,
        call.query_tokens,
        call.key_tokens,
        call.tensor_layout,
        call.skip_output_reshape,
    )
    return None if expected == actual else "prepared-attention metadata does not match Q/K/V"


def _make_dense_prepared_executor(kernel: str) -> Callable:
    capabilities = AttentionBackendCapabilities(
        supports_causal=kernel in {"sage", "w8a8"},
    )

    def executor(request: PreparedAttention) -> AttentionExecutionOutcome:
        reason = capabilities.unsupported_reason(request)
        if reason is not None:
            return AttentionExecutionOutcome.unsupported(reason)
        query_view, key_view, value_view = request.peek_qkv()
        call, reason = inspect_turing_attention_call(
            query_view,
            key_view,
            value_view,
            request.heads,
            mask=request.mask,
            skip_reshape=True,
            skip_output_reshape=request.skip_output_reshape,
            enable_gqa=request.heads != request.kv_heads,
            low_precision_attention=request.low_precision_attention,
            is_causal=request.is_causal,
            kernel=kernel,
            require_long_sequence=True,
        )
        if reason is not None:
            return AttentionExecutionOutcome.unsupported(reason)
        reason = _prepared_call_mismatch(request, call)
        if reason is not None:
            return AttentionExecutionOutcome.unsupported(reason)
        del query_view, key_view, value_view
        query, key, value = request.consume_qkv()
        qk = _profiled(
            "attention.qk_norm_rope_quant",
            prequantize_turing_qk,
            query,
            key,
            request.qk_transform,
            kernel=kernel,
            transformer_options=request.transformer_options,
        )
        del query, key
        quantized = _profiled(
            "attention.value_prepare",
            prequantize_turing_attention_from_qk,
            qk,
            value,
            call,
            kernel=kernel,
            scale=request.scale,
            is_causal=request.is_causal,
            transformer_options=request.transformer_options,
        )
        del qk, value
        return AttentionExecutionOutcome(
            _profiled(
                "attention.execute",
                turing_attention_from_prequantized,
                quantized,
                kernel=kernel,
            )
        )

    executor.capabilities = capabilities
    return executor


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
            transformer_options=kwargs.get("transformer_options"),
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
            "The W8A8 attention backend requires the bundled exact-sm75 kernel; "
            "select sage_attn, flash_attn, or sdpa on other GPUs."
        )
    if bundled_turing:
        if option == "w8a8" and not bundled_w8a8_available():
            raise RuntimeError(
                "The bundled Turing W8A8 extension is unavailable. "
                "Rebuild comfyui-turing-utils-kernel 0.23.0 or newer with sm75 enabled."
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
            "bundled_turing_w8a8"
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
    if bundled_turing and fused_qk_preprocessing_available():
        attention_override.prepared_attention_executor = _make_dense_prepared_executor(
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
            "Rebuild comfyui-turing-utils-kernel 0.23.0 or newer with sm75 enabled."
        )
    preflight_bundled(device)
    preflight_bundled_sparse(device)
    if use_w8a8:
        if not bundled_w8a8_available():
            raise RuntimeError(
                "Sol W8A8 requires comfyui-turing-utils-kernel 0.23.0 or newer"
            )
        preflight_bundled_w8a8(device)
    schedule_state: dict[str, object] = {}
    debug_route_keys: set[tuple] = set()
    debug_route_state: dict[tuple, list[tuple[torch.Tensor, int, int]]] = {}
    debug_dense_reasons: set[str] = set()
    dense_prepared_executor = _make_dense_prepared_executor(
        "w8a8" if use_w8a8 else "sage"
    )
    sparse_capabilities = AttentionBackendCapabilities(
        supports_semantic_sparse=True,
    )

    def route_debug_context(transformer_options, kernel_key: tuple):
        layer_index, layer_count = _attention_layer_metadata(transformer_options)
        step = schedule_state.get("step")
        sampling_steps = schedule_state.get("sampling_steps")
        last_sparse_layer = (
            layer_count - dense_suffix_layers - 1
            if isinstance(layer_count, int)
            and not isinstance(layer_count, bool)
            else None
        )
        aggregate = (
            debug_route_density
            and isinstance(step, int)
            and not isinstance(step, bool)
            and isinstance(layer_index, int)
            and not isinstance(layer_index, bool)
            and isinstance(layer_count, int)
            and not isinstance(layer_count, bool)
            and layer_count > 0
            and 0 <= layer_index < layer_count
            and isinstance(last_sparse_layer, int)
            and 0 <= last_sparse_layer < layer_count
        )
        return {
            "collect": debug_route_density and (
                aggregate or kernel_key not in debug_route_keys
            ),
            "aggregate": aggregate,
            "step": step,
            "sampling_steps": sampling_steps,
            "layer_index": layer_index,
            "layer_count": layer_count,
            "last_sparse_layer": last_sparse_layer,
        }

    def record_route_stats(
        selected_device: torch.Tensor,
        possible_blocks: int,
        sol_call,
        kernel_key: tuple,
        context: dict,
    ) -> None:
        protected_query_tokens = sum(
            stop - start for start, stop in sol_call.dense_query_ranges
        )
        sparse_query_tokens = (
            sol_call.attention.query_tokens - protected_query_tokens
        )
        if context["aggregate"]:
            aggregate_key = (
                context["step"],
                context["sampling_steps"],
                kernel_key,
            )
            entries = debug_route_state.setdefault(aggregate_key, [])
            entries.append(
                (selected_device, possible_blocks, context["layer_index"])
            )
            if context["layer_index"] != context["last_sparse_layer"]:
                return
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
            LOG.warning(
                "[Turing sparse debug] step=%s/%s layers=%d-%d calls=%d "
                "selected=%d/%d density[min/mean/max]=%.4f/%.4f/%.4f "
                "Q=%d Qsparse=%d K=%d Hq=%d Hkv=%d threshold=%.2f "
                "protected_q=%d local=1 residual=%s",
                context["step"],
                context["sampling_steps"],
                min(entry[2] for entry in entries),
                max(entry[2] for entry in entries),
                len(entries),
                int(summary[0]),
                int(summary[1]),
                summary[2],
                summary[3],
                summary[4],
                sol_call.attention.query_tokens,
                sparse_query_tokens,
                sol_call.attention.key_tokens,
                sol_call.attention.heads,
                sol_call.attention.kv_heads,
                routing_threshold,
                protected_query_tokens,
                skipped_residual,
            )
            del debug_route_state[aggregate_key]
            return

        selected_blocks = int(selected_device.item())
        LOG.warning(
            "[Turing sparse debug] Q=%d Qsparse=%d K=%d Hq=%d Hkv=%d "
            "selected=%d/%d density=%.4f threshold=%.2f protected_q=%d "
            "local=1 residual=%s step=%s/%s layer=%s/%s",
            sol_call.attention.query_tokens,
            sparse_query_tokens,
            sol_call.attention.key_tokens,
            sol_call.attention.heads,
            sol_call.attention.kv_heads,
            selected_blocks,
            possible_blocks,
            selected_blocks / possible_blocks if possible_blocks else 0.0,
            routing_threshold,
            protected_query_tokens,
            skipped_residual,
            context["step"],
            context["sampling_steps"],
            context["layer_index"],
            context["layer_count"],
        )
        debug_route_keys.add(kernel_key)

    def prepared_executor(request: PreparedAttention) -> AttentionExecutionOutcome:
        reason = sparse_capabilities.unsupported_reason(request)
        if reason is not None:
            return AttentionExecutionOutcome.unsupported(reason)
        transformer_options = request.transformer_options
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
            return dense_prepared_executor(request)

        query_view, key_view, value_view = request.peek_qkv()
        sol_call, reason = inspect_sol_attention_call(
            query_view,
            key_view,
            value_view,
            request.heads,
            mask=request.mask,
            skip_reshape=True,
            skip_output_reshape=request.skip_output_reshape,
            min_sequence_tokens=min_sequence_tokens,
            prefix_policy=prefix_policy,
            manual_prefix_tokens=manual_prefix_tokens,
            skipped_residual=skipped_residual,
            sparse_reference_image=sparse_reference_image,
            sparse_reference_video=sparse_reference_video,
            sparse_reference_audio=sparse_reference_audio,
            transformer_options=transformer_options,
            kwargs={
                "enable_gqa": request.heads != request.kv_heads,
                "low_precision_attention": request.low_precision_attention,
                "is_causal": request.is_causal,
            },
        )
        if reason is not None:
            return dense_prepared_executor(request)
        reason = _prepared_call_mismatch(request, sol_call.attention)
        if reason is not None:
            return AttentionExecutionOutcome.unsupported(reason)

        del query_view, key_view, value_view
        query, key, value = request.consume_qkv()
        qk = _profiled(
            "attention.qk_norm_rope_quant",
            prequantize_turing_qk,
            query,
            key,
            request.qk_transform,
            kernel="sol",
            transformer_options=transformer_options,
        )
        del query, key
        quantized = _profiled(
            "attention.value_route_prepare",
            prequantize_turing_sol_attention_from_qk,
            qk,
            value,
            sol_call,
            routing_threshold=routing_threshold,
            scale=request.scale,
            use_w8a8=use_w8a8,
            transformer_options=transformer_options,
        )
        del qk, value
        debug_key = (
            sol_call.attention.input_dtype,
            sol_call.attention.query_tokens,
            sol_call.attention.key_tokens,
            sol_call.dense_query_ranges,
            sol_call.exact_kv_ranges,
            routing_threshold,
            sol_call.residual_subblocks,
            use_w8a8,
        )
        debug_context = route_debug_context(transformer_options, debug_key)
        collect_stats = debug_context["collect"]
        result = _profiled(
            "attention.execute",
            turing_sol_attention_from_prequantized,
            quantized,
            return_stats=collect_stats,
        )
        if not collect_stats:
            return AttentionExecutionOutcome(result)
        output, selected, possible = result
        record_route_stats(selected, possible, sol_call, debug_key, debug_context)
        return AttentionExecutionOutcome(output)

    prepared_executor.capabilities = sparse_capabilities

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
            layer_index, layer_count = _attention_layer_metadata(transformer_options)
            debug_key = f"layer:{layer_index}"
            if debug_key not in debug_dense_reasons:
                LOG.warning(
                    "[Turing sparse debug] stable Sage selected for protected layer %s/%s",
                    layer_index,
                    layer_count,
                )
                debug_dense_reasons.add(debug_key)
        if dense_schedule or dense_layer:
            dense_attention = turing_w8a8_attention if use_w8a8 else turing_sage_attention
            return dense_attention(fallback, *args, **kwargs)
        debug_context = None
        if debug_route_density:
            layer_index, layer_count = _attention_layer_metadata(transformer_options)
            debug_context = {
                "step": schedule_state.get("step"),
                "sampling_steps": schedule_state.get("sampling_steps"),
                "layer_index": layer_index,
                "layer_count": layer_count,
                "last_sparse_layer": (
                    layer_count - dense_suffix_layers - 1
                    if isinstance(layer_count, int)
                    and not isinstance(layer_count, bool)
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
                transformer_options=transformer_options,
            )
            del query, key, value
            debug_key = (
                sol_call.attention.input_dtype,
                sol_call.attention.query_tokens,
                sol_call.attention.key_tokens,
                sol_call.dense_query_ranges,
                sol_call.exact_kv_ranges,
                routing_threshold,
                sol_call.residual_subblocks,
                use_w8a8,
            )
            debug_context = route_debug_context(transformer_options, debug_key)
            collect_stats = debug_context["collect"]
            result = turing_sol_attention_from_prequantized(
                quantized,
                return_stats=collect_stats,
            )
            if collect_stats:
                output, selected, possible = result
                record_route_stats(
                    selected, possible, sol_call, debug_key, debug_context
                )
                return output
            return result

        attention_override.container_function = container_function

    attention_override.turing_utils_attention_backend = "sol_sparse_attn"
    attention_override.turing_utils_attention_implementation = "bundled_turing_sol_sparse_experimental"
    if fused_qk_preprocessing_available():
        attention_override.prepared_attention_executor = prepared_executor
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
    prepared_executor = getattr(override, "prepared_attention_executor", None)
    if callable(prepared_executor):
        transformer_options[ATTENTION_EXECUTOR_KEY] = prepared_executor
        site_status = ensure_prepared_attention_sites(patched, patched.load_device)
        if site_status.matched and site_status.reason is not None:
            LOG.info(
                "%s prepared-attention fusion was not installed: %s",
                site_status.model_kind,
                site_status.reason,
            )
    else:
        transformer_options.pop(ATTENTION_EXECUTOR_KEY, None)
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


def apply_attention_backend(model, option: str, device: torch.device | None = None):
    option = normalize_attention_backend(option)
    transformer_options = model.model_options.setdefault("transformer_options", {})
    override = make_attention_override(option, device=device)
    selected = override.turing_utils_attention_backend
    implementation = override.turing_utils_attention_implementation
    transformer_options["optimized_attention_override"] = override
    prepared_executor = getattr(override, "prepared_attention_executor", None)
    if callable(prepared_executor):
        transformer_options[ATTENTION_EXECUTOR_KEY] = prepared_executor
        target_device = device if device is not None else getattr(model, "load_device", None)
        if isinstance(target_device, torch.device):
            site_status = ensure_prepared_attention_sites(model, target_device)
            if site_status.matched and site_status.reason is not None:
                LOG.info(
                    "%s prepared-attention fusion was not installed: %s",
                    site_status.model_kind,
                    site_status.reason,
                )
    else:
        transformer_options.pop(ATTENTION_EXECUTOR_KEY, None)
    transformer_options["turing_utils_attention_backend"] = selected
    transformer_options["turing_utils_attention_implementation"] = implementation
    LOG.info(
        "Turing Utils attention backend override: %s via %s (requested %s)",
        selected,
        implementation,
        option,
    )
    return model
