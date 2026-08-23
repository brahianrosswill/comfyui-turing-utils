"""Attention overrides and loader-independent ModelPatcher installation."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ..profiling import CUDA_PHASE_PROFILER
from .layout import attention_semantic_layout
from .integration import ensure_prepared_attention_sites
from .orchestration import install_sparse_attention_override
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
    inspect_sla_attention_call,
    prequantize_turing_sla_attention,
    prequantize_turing_sla_attention_from_qk,
    prequantize_turing_sol_attention,
    prequantize_turing_sol_attention_from_qk,
    turing_sol_attention_from_prequantized,
    turing_sol_sparse_attention,
    turing_sla_attention_from_prequantized,
    turing_sla_sparse_attention,
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
    SLA_DENSE_PREFIX_LAYERS,
    SLA_DENSE_PREFIX_STEPS,
    SLA_DENSE_SUFFIX_LAYERS,
    SLA_DENSE_SUFFIX_STEPS,
    SLA_SPARSITY_RATIO,
    AttentionBackend,
    _BACKENDS,
    _comfy_attention_function,
    _select_attention_backend,
    bundled_available,
    bundled_sparse_available,
    bundled_sla_available,
    bundled_w8a8_available,
    fused_qk_preprocessing_available,
    is_supported_attention_device,
    is_supported_turing_device,
    inspect_turing_attention_call,
    normalize_attention_backend,
    prequantize_turing_attention,
    prequantize_turing_attention_from_qk,
    prequantize_turing_qk,
    preflight_bundled,
    preflight_bundled_sparse,
    preflight_bundled_sla,
    preflight_bundled_w8a8,
    split_prequantization_available,
    turing_attention_from_prequantized,
    turing_sage_attention,
    turing_w8a8_attention,
)


_LOGGED_EXTERNAL_BACKEND_REJECTIONS: set[tuple[str, str]] = set()


def _profiled(phase: str, function: Callable, /, *args, **kwargs):
    if CUDA_PHASE_PROFILER.enabled:
        return CUDA_PHASE_PROFILER.call(phase, function, *args, **kwargs)
    return function(*args, **kwargs)


def _profile_route_stats(selected: torch.Tensor, possible: int) -> None:
    """Attach sparse-route density without synchronizing the sampler.

    ``selected`` stays on the device until the profiler report is emitted at
    the existing sampler-boundary fence.  Normal execution therefore pays no
    scalar readback or extra synchronization cost.
    """
    if not CUDA_PHASE_PROFILER.enabled:
        return
    selected_total = selected if selected.numel() == 1 else selected.sum()
    CUDA_PHASE_PROFILER.sample_tensor("route_selected_blocks", selected_total)
    CUDA_PHASE_PROFILER.sample("route_possible_blocks", int(possible))


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
    if kernel == "w8a8":
        def streamed_qkv_executor(
            qk,
            value: torch.Tensor,
            *,
            heads: int,
            qk_transform,
            transformer_options,
        ) -> AttentionExecutionOutcome:
            del qk_transform
            if not torch.is_tensor(value) or value.ndim != 4 or value.shape[2] < 64:
                return AttentionExecutionOutcome.unsupported(
                    "streamed QKV requires HND W8A8 with at least 64 tokens"
                )
            prototype = value[:, :, :1, :].expand(
                value.shape[0], value.shape[1], value.shape[2], value.shape[3]
            )
            call, reason = inspect_turing_attention_call(
                prototype,
                prototype,
                prototype,
                heads,
                mask=None,
                skip_reshape=True,
                skip_output_reshape=False,
                enable_gqa=False,
                low_precision_attention=True,
                is_causal=False,
                kernel="w8a8",
                require_long_sequence=True,
            )
            if reason is not None:
                return AttentionExecutionOutcome.unsupported(reason)
            quantized = _profiled(
                "attention.value_prepare",
                prequantize_turing_attention_from_qk,
                qk,
                value,
                call,
                kernel="w8a8",
                scale=None,
            )
            return AttentionExecutionOutcome(
                _profiled(
                    "attention.execute",
                    turing_attention_from_prequantized,
                    quantized,
                    kernel="w8a8",
                )
            )

        executor.turing_utils_streamed_qkv_executor = streamed_qkv_executor
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
        )
        del query, key, value
        return turing_attention_from_prequantized(quantized, kernel=kernel)

    return container_function


def _uses_bundled_turing_attention(option: str, device: torch.device | None) -> bool:
    option = normalize_attention_backend(option)
    return bool(
        device is not None
        and (
            (option == "w8a8" and is_supported_attention_device(device))
            or (option == "sage" and is_supported_turing_device(device))
        )
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


def _recoverable_external_backend_rejection(error: Exception) -> bool:
    if isinstance(error, torch.OutOfMemoryError):
        return False
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "alignment",
            "not implemented",
            "not supported",
            "unsupported",
            "no kernel image",
            "requires cuda",
        )
    )


def _external_backend_call(
    backend: AttentionBackend,
    target: Callable,
    fallback: Callable,
    *args,
    **kwargs,
):
    try:
        return target(*args, **kwargs)
    except (RuntimeError, NotImplementedError) as error:
        if (
            backend.option != "w8a8"
            or not _recoverable_external_backend_rejection(error)
        ):
            raise
        key = (backend.attention_function, str(error).splitlines()[0])
        if key not in _LOGGED_EXTERNAL_BACKEND_REJECTIONS:
            LOG.warning(
                "External %s attention rejected this call (%s); using the "
                "pre-existing ComfyUI attention backend",
                backend.attention_function,
                key[1],
            )
            _LOGGED_EXTERNAL_BACKEND_REJECTIONS.add(key)
        return fallback(*args, **kwargs)


def _make_external_container_function(
    backend: AttentionBackend,
    target: Callable,
) -> Callable:
    fallback = _default_attention_fallback()

    def container_function(q, k, v, heads: int, *args, **kwargs):
        q.peek(), k.peek(), v.peek()
        query, key, value = q.take(), k.take(), v.take()
        compatible_fallback = lambda *fallback_args, **fallback_kwargs: (
            _dtype_compatible_fallback(
                fallback,
                *fallback_args,
                **fallback_kwargs,
            )
        )
        return _external_backend_call(
            backend,
            target,
            compatible_fallback,
            query,
            key,
            value,
            heads,
            *args,
            **kwargs,
        )

    return container_function


def _uses_turing_bf16_sdpa(q, k, v) -> bool:
    return bool(
        all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v))
        and q.dtype is torch.bfloat16
        and k.dtype is torch.bfloat16
        and v.dtype is torch.bfloat16
        and q.device == k.device == v.device
        and is_supported_turing_device(q.device)
    )


def _convert_sdpa_mask_to_fp16(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Match floating additive masks to FP16 Q/K/V; boolean masks stay exact."""
    if args and torch.is_tensor(args[0]) and args[0].is_floating_point():
        positional = list(args)
        positional[0] = positional[0].to(torch.float16)
        return tuple(positional), kwargs
    mask = kwargs.get("mask")
    if torch.is_tensor(mask) and mask.is_floating_point():
        kwargs = dict(kwargs)
        kwargs["mask"] = mask.to(torch.float16)
    return args, kwargs


def _turing_sdpa_fp16(
    target: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    *args,
    **kwargs,
):
    """Avoid Turing's BF16 SDPA math fallback while preserving its API dtype."""
    use_fp16 = _uses_turing_bf16_sdpa(q, k, v)
    if not use_fp16:
        return target(q, k, v, heads, *args, **kwargs)

    q = q.to(torch.float16)
    k = k.to(torch.float16)
    v = v.to(torch.float16)
    args, kwargs = _convert_sdpa_mask_to_fp16(args, kwargs)
    return target(q, k, v, heads, *args, **kwargs).to(torch.bfloat16)


def _make_sdpa_container_function(target: Callable) -> Callable:
    """Consume Q/K/V before SDPA conversion so BF16 and FP16 overlap is bounded."""

    def container_function(q, k, v, heads: int, *args, **kwargs):
        # Validate every owner before the first take to avoid a partial transfer.
        query_view, key_view, value_view = q.peek(), k.peek(), v.peek()
        use_fp16 = _uses_turing_bf16_sdpa(query_view, key_view, value_view)
        del query_view, key_view, value_view

        query = q.take()
        if use_fp16:
            query = query.to(torch.float16)
        key = k.take()
        if use_fp16:
            key = key.to(torch.float16)
        value = v.take()
        if use_fp16:
            value = value.to(torch.float16)

        if not use_fp16:
            return target(query, key, value, heads, *args, **kwargs)
        args, kwargs = _convert_sdpa_mask_to_fp16(args, kwargs)
        return target(query, key, value, heads, *args, **kwargs).to(torch.bfloat16)

    return container_function


def make_attention_override(option: str, device: torch.device | None = None) -> Callable:
    option = normalize_attention_backend(option)
    bundled_turing = _uses_bundled_turing_attention(option, device)
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
        backend = _BACKENDS[option]
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
            backend.option == "sage"
            and len(args) >= 3
            and all(isinstance(value, torch.Tensor) for value in args[:3])
            and any(value.dtype == torch.float32 for value in args[:3])
        ):
            return fallback(*args, **kwargs)
        if backend.option == "sdpa" and len(args) >= 4:
            return _turing_sdpa_fp16(target, *args, **kwargs)
        return _external_backend_call(backend, target, fallback, *args, **kwargs)

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
    if not bundled_turing and backend.option == "sdpa":
        attention_override.container_function = _make_sdpa_container_function(target)
    elif (
        not bundled_turing
        and callable(getattr(target, "container_function", None))
    ):
        attention_override.container_function = _make_external_container_function(
            backend,
            target,
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
    if not is_supported_attention_device(device):
        raise RuntimeError(
            "Sol sparse attention requires a CUDA Tensor Core GPU (sm75 or newer)"
        )
    if not bundled_sparse_available():
        raise RuntimeError(
            "The bundled Sol sparse extension is unavailable. Rebuild "
            "comfyui-turing-utils-kernel 0.28.0 or newer with "
            "COMFYUI_TURING_UTILS_ARCH_LIST including the target GPU architecture."
        )
    preflight_bundled_sparse(device)
    if use_w8a8:
        if not bundled_w8a8_available():
            raise RuntimeError(
                "Sol W8A8 requires comfyui-turing-utils-kernel 0.28.0 or newer"
            )
    schedule_state: dict[str, object] = {}
    debug_route_keys: set[tuple] = set()
    debug_route_state: dict[tuple, list[tuple[torch.Tensor, int, int]]] = {}
    debug_dense_reasons: set[str] = set()
    dense_override = make_attention_override(
        "w8a8" if use_w8a8 else "sage", device=device
    )
    dense_prepared_executor = getattr(
        dense_override, "prepared_attention_executor", None
    )
    dense_container = getattr(dense_override, "container_function", None)
    if use_w8a8 and not callable(dense_prepared_executor):
        # The bundled force-dense W8A8 core has native sm75+ cubins.  Keep the
        # same prepared path on every supported generation instead of falling
        # back to a full floating Q/K/V materialization on newer GPUs.
        dense_prepared_executor = _make_dense_prepared_executor("w8a8")
        dense_container = _make_dense_container_function("w8a8")
    dense_streamed_qkv_executor = getattr(
        dense_prepared_executor,
        "turing_utils_streamed_qkv_executor",
        None,
    )
    sparse_capabilities = AttentionBackendCapabilities(
        supports_semantic_sparse=True,
    )

    def run_dense_prepared(request: PreparedAttention) -> AttentionExecutionOutcome:
        if callable(dense_prepared_executor):
            return dense_prepared_executor(request)
        return AttentionExecutionOutcome.unsupported(
            "the selected dense backend does not expose prepared attention"
        )

    def run_dense_container(
        q,
        k,
        v,
        heads,
        *,
        mask,
        attn_precision,
        skip_reshape,
        skip_output_reshape,
        **kwargs,
    ):
        if callable(dense_container):
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
        return dense_override(
            _default_attention_fallback(),
            q.take(),
            k.take(),
            v.take(),
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
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
            return run_dense_prepared(request)

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
            return run_dense_prepared(request)
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
        collect_debug_stats = debug_context["collect"]
        collect_stats = collect_debug_stats or CUDA_PHASE_PROFILER.enabled
        result = _profiled(
            "attention.execute",
            turing_sol_attention_from_prequantized,
            quantized,
            return_stats=collect_stats,
        )
        if not collect_stats:
            return AttentionExecutionOutcome(result)
        output, selected, possible = result
        _profile_route_stats(selected, possible)
        if collect_debug_stats:
            record_route_stats(selected, possible, sol_call, debug_key, debug_context)
        return AttentionExecutionOutcome(output)

    def streamed_qkv_executor(
        qk,
        value: torch.Tensor,
        *,
        heads: int,
        qk_transform,
        transformer_options,
    ) -> AttentionExecutionOutcome:
        """Finish attention from row-streamed Q/K INT8 and retained V BF16."""
        if (
            not use_w8a8
            or not torch.is_tensor(value)
            or value.ndim != 4
            or value.shape[2] < 64
        ):
            return AttentionExecutionOutcome.unsupported(
                "streamed QKV requires HND W8A8 with at least 64 tokens"
            )
        prototype = value[:, :, :1, :].expand(
            value.shape[0], value.shape[1], value.shape[2], value.shape[3]
        )

        def dense_result() -> AttentionExecutionOutcome:
            if not callable(dense_streamed_qkv_executor):
                return AttentionExecutionOutcome.unsupported(
                    "the selected dense backend cannot consume streamed QKV"
                )
            return dense_streamed_qkv_executor(
                qk,
                value,
                heads=heads,
                qk_transform=qk_transform,
                transformer_options=transformer_options,
            )

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
            return dense_result()

        sol_call, reason = inspect_sol_attention_call(
            prototype,
            prototype,
            prototype,
            heads,
            mask=None,
            skip_reshape=True,
            skip_output_reshape=False,
            min_sequence_tokens=min_sequence_tokens,
            prefix_policy=prefix_policy,
            manual_prefix_tokens=manual_prefix_tokens,
            skipped_residual=skipped_residual,
            sparse_reference_image=sparse_reference_image,
            sparse_reference_video=sparse_reference_video,
            sparse_reference_audio=sparse_reference_audio,
            transformer_options=transformer_options,
            kwargs={
                "enable_gqa": False,
                "low_precision_attention": True,
                "is_causal": False,
            },
        )
        if reason is not None:
            return dense_result()
        quantized = _profiled(
            "attention.value_route_prepare",
            prequantize_turing_sol_attention_from_qk,
            qk,
            value,
            sol_call,
            routing_threshold=routing_threshold,
            scale=None,
            use_w8a8=True,
        )
        debug_key = (
            sol_call.attention.input_dtype,
            sol_call.attention.query_tokens,
            sol_call.attention.key_tokens,
            sol_call.dense_query_ranges,
            sol_call.exact_kv_ranges,
            routing_threshold,
            sol_call.residual_subblocks,
            True,
        )
        debug_context = route_debug_context(transformer_options, debug_key)
        collect_debug_stats = debug_context["collect"]
        collect_stats = collect_debug_stats or CUDA_PHASE_PROFILER.enabled
        result = _profiled(
            "attention.execute",
            turing_sol_attention_from_prequantized,
            quantized,
            return_stats=collect_stats,
        )
        if not collect_stats:
            return AttentionExecutionOutcome(result)
        output, selected, possible = result
        _profile_route_stats(selected, possible)
        if collect_debug_stats:
            record_route_stats(selected, possible, sol_call, debug_key, debug_context)
        return AttentionExecutionOutcome(output)

    prepared_executor.capabilities = sparse_capabilities
    if use_w8a8:
        prepared_executor.turing_utils_streamed_qkv_executor = (
            streamed_qkv_executor
        )

    def attention_override(original: Callable, *args, **kwargs):
        fallback = lambda *fallback_args, **fallback_kwargs: dense_override(
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
                    "[Sol sparse debug] dense backend selected by schedule: "
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
                    "[Sol sparse debug] dense backend selected for protected layer %s/%s",
                    layer_index,
                    layer_count,
                )
                debug_dense_reasons.add(debug_key)
        if dense_schedule or dense_layer:
            return fallback(*args, **kwargs)
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
                return run_dense_container(
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
                return run_dense_container(
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
                sol_call.dense_query_ranges,
                sol_call.exact_kv_ranges,
                routing_threshold,
                sol_call.residual_subblocks,
                use_w8a8,
            )
            debug_context = route_debug_context(transformer_options, debug_key)
            collect_debug_stats = debug_context["collect"]
            collect_stats = collect_debug_stats or CUDA_PHASE_PROFILER.enabled
            result = turing_sol_attention_from_prequantized(
                quantized,
                return_stats=collect_stats,
            )
            if collect_stats:
                output, selected, possible = result
                _profile_route_stats(selected, possible)
                if collect_debug_stats:
                    record_route_stats(
                        selected, possible, sol_call, debug_key, debug_context
                    )
                return output
            return result

        attention_override.container_function = container_function

    attention_override.turing_utils_attention_backend = "sol_sparse_attn"
    attention_override.turing_utils_attention_implementation = "bundled_sol_sparse"
    attention_override.turing_utils_dense_implementation = (
        dense_override.turing_utils_attention_implementation
    )
    if fused_qk_preprocessing_available():
        attention_override.prepared_attention_executor = prepared_executor
    return attention_override


def make_sla_attention_override(
    device: torch.device,
    min_sequence_tokens: int = 0,
    sparsity_ratio: float = SLA_SPARSITY_RATIO,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    sparse_reference_image: bool = SPARSE_REFERENCE_IMAGE,
    sparse_reference_video: bool = SPARSE_REFERENCE_VIDEO,
    sparse_reference_audio: bool = SPARSE_REFERENCE_AUDIO,
    dense_prefix_steps: int = SLA_DENSE_PREFIX_STEPS,
    dense_suffix_steps: int = SLA_DENSE_SUFFIX_STEPS,
    dense_prefix_layers: int = SLA_DENSE_PREFIX_LAYERS,
    dense_suffix_layers: int = SLA_DENSE_SUFFIX_LAYERS,
    debug_route_density: bool = False,
    use_w8a8: bool = SPARSE_USE_W8A8,
) -> Callable:
    min_sequence_tokens = int(min_sequence_tokens)
    sparsity_ratio = float(sparsity_ratio)
    prefix_policy = str(prefix_policy).strip().lower()
    manual_prefix_tokens = int(manual_prefix_tokens)
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
    if not math.isfinite(sparsity_ratio) or not 0.0 <= sparsity_ratio < 1.0:
        raise ValueError("sparsity_ratio must be finite and in [0, 1)")
    if prefix_policy not in {"auto", "none", "manual"}:
        raise ValueError("prefix_policy must be auto, none, or manual")
    if manual_prefix_tokens < 0:
        raise ValueError("manual_prefix_tokens must be non-negative")
    if min(
        dense_prefix_steps,
        dense_suffix_steps,
        dense_prefix_layers,
        dense_suffix_layers,
    ) < 0:
        raise ValueError("dense step/layer counts must be non-negative")
    if not is_supported_attention_device(device):
        raise RuntimeError(
            "SLA sparse attention requires a CUDA Tensor Core GPU (sm75 or newer)"
        )
    if not bundled_sla_available():
        raise RuntimeError(
            "The bundled SLA extension is unavailable. Rebuild "
            "comfyui-turing-utils-kernel 0.29.1 or newer for this GPU."
        )
    preflight_bundled_sla(device)
    if use_w8a8 and not bundled_w8a8_available():
        raise RuntimeError("SLA W8A8 requires the bundled W8A8 attention ABI")

    schedule_state: dict[str, object] = {}
    debug_route_keys: set[tuple] = set()
    dense_override = make_attention_override(
        "w8a8" if use_w8a8 else "sage", device=device
    )
    dense_prepared_executor = getattr(
        dense_override, "prepared_attention_executor", None
    )
    dense_container = getattr(dense_override, "container_function", None)
    dense_streamed_qkv_executor = getattr(
        dense_prepared_executor,
        "turing_utils_streamed_qkv_executor",
        None,
    )
    sparse_capabilities = AttentionBackendCapabilities(
        supports_semantic_sparse=True,
    )

    def run_dense_prepared(request: PreparedAttention) -> AttentionExecutionOutcome:
        if callable(dense_prepared_executor):
            return dense_prepared_executor(request)
        return AttentionExecutionOutcome.unsupported(
            "the selected dense backend does not expose prepared attention"
        )

    def run_dense_container(
        q,
        k,
        v,
        heads,
        *,
        mask,
        attn_precision,
        skip_reshape,
        skip_output_reshape,
        **kwargs,
    ):
        if callable(dense_container):
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
        return dense_override(
            _default_attention_fallback(),
            q.take(),
            k.take(),
            v.take(),
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    def inspect(request_q, request_k, request_v, heads, *, mask,
                skip_reshape, skip_output_reshape, transformer_options, kwargs):
        return inspect_sla_attention_call(
            request_q,
            request_k,
            request_v,
            heads,
            mask=mask,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            min_sequence_tokens=min_sequence_tokens,
            prefix_policy=prefix_policy,
            manual_prefix_tokens=manual_prefix_tokens,
            sparse_reference_image=sparse_reference_image,
            sparse_reference_video=sparse_reference_video,
            sparse_reference_audio=sparse_reference_audio,
            transformer_options=transformer_options,
            kwargs=kwargs,
        )

    def collect_stats(result, sla_call, transformer_options):
        output, selected, possible = result
        layer_index, _ = _attention_layer_metadata(transformer_options)
        debug_key = (
            schedule_state.get("step"),
            layer_index,
            sla_call.attention.query_tokens,
            sla_call.attention.key_tokens,
            sla_call.dense_query_ranges,
            sla_call.exact_kv_ranges,
        )
        if debug_key not in debug_route_keys:
            selected_blocks = int(selected.item())
            LOG.warning(
                "[Turing SLA debug] step=%s layer=%s Q=%d K=%d "
                "selected=%d/%d density=%.4f target_sparsity=%.2f "
                "protected_q=%d",
                schedule_state.get("step"),
                layer_index,
                sla_call.attention.query_tokens,
                sla_call.attention.key_tokens,
                selected_blocks,
                possible,
                selected_blocks / possible if possible else 0.0,
                sparsity_ratio,
                sum(stop - start for start, stop in sla_call.dense_query_ranges),
            )
            debug_route_keys.add(debug_key)
        return output

    def is_dense(transformer_options) -> bool:
        return sparsity_ratio == 0.0 or _sparse_dense_schedule(
            transformer_options,
            dense_prefix_steps,
            dense_suffix_steps,
            schedule_state,
        ) or _sparse_dense_layer(
            transformer_options,
            dense_prefix_layers,
            dense_suffix_layers,
        )

    def prepared_executor(request: PreparedAttention) -> AttentionExecutionOutcome:
        reason = sparse_capabilities.unsupported_reason(request)
        if reason is not None:
            return AttentionExecutionOutcome.unsupported(reason)
        transformer_options = request.transformer_options
        if is_dense(transformer_options):
            return run_dense_prepared(request)
        query_view, key_view, value_view = request.peek_qkv()
        sla_call, reason = inspect(
            query_view,
            key_view,
            value_view,
            request.heads,
            mask=request.mask,
            skip_reshape=True,
            skip_output_reshape=request.skip_output_reshape,
            transformer_options=transformer_options,
            kwargs={
                "enable_gqa": request.heads != request.kv_heads,
                "low_precision_attention": request.low_precision_attention,
                "is_causal": request.is_causal,
            },
        )
        if reason is not None:
            return run_dense_prepared(request)
        reason = _prepared_call_mismatch(request, sla_call.attention)
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
            kernel="sla",
        )
        del query, key
        quantized = _profiled(
            "attention.value_route_prepare",
            prequantize_turing_sla_attention_from_qk,
            qk,
            value,
            sla_call,
            sparsity_ratio=sparsity_ratio,
            scale=request.scale,
            use_w8a8=use_w8a8,
        )
        del qk, value
        profile_route_density = CUDA_PHASE_PROFILER.enabled
        result = _profiled(
            "attention.execute",
            turing_sla_attention_from_prequantized,
            quantized,
            return_stats=debug_route_density or profile_route_density,
        )
        if debug_route_density or profile_route_density:
            output, selected, possible = result
            _profile_route_stats(selected, possible)
            if debug_route_density:
                output = collect_stats(
                    (output, selected, possible), sla_call, transformer_options
                )
        else:
            output = result
        return AttentionExecutionOutcome(output)

    prepared_executor.capabilities = sparse_capabilities
    if use_w8a8:
        def streamed_qkv_executor(
            qk,
            value: torch.Tensor,
            *,
            heads: int,
            qk_transform,
            transformer_options,
        ) -> AttentionExecutionOutcome:
            """Finish SLA from the shared row-streamed H3 QKV representation."""
            if (
                not torch.is_tensor(value)
                or value.ndim != 4
                or value.shape[2] < 64
            ):
                return AttentionExecutionOutcome.unsupported(
                    "streamed QKV requires HND W8A8 with at least 64 tokens"
                )
            if is_dense(transformer_options):
                if not callable(dense_streamed_qkv_executor):
                    return AttentionExecutionOutcome.unsupported(
                        "the selected dense backend cannot consume streamed QKV"
                    )
                return dense_streamed_qkv_executor(
                    qk,
                    value,
                    heads=heads,
                    qk_transform=qk_transform,
                    transformer_options=transformer_options,
                )

            prototype = value[:, :, :1, :].expand(
                value.shape[0], value.shape[1], value.shape[2], value.shape[3]
            )
            sla_call, reason = inspect(
                prototype,
                prototype,
                prototype,
                heads,
                mask=None,
                skip_reshape=True,
                skip_output_reshape=False,
                transformer_options=transformer_options,
                kwargs={
                    "enable_gqa": False,
                    "low_precision_attention": True,
                    "is_causal": False,
                },
            )
            if reason is not None:
                return AttentionExecutionOutcome.unsupported(reason)
            quantized = _profiled(
                "attention.value_route_prepare",
                prequantize_turing_sla_attention_from_qk,
                qk,
                value,
                sla_call,
                sparsity_ratio=sparsity_ratio,
                scale=None,
                use_w8a8=True,
            )
            profile_route_density = CUDA_PHASE_PROFILER.enabled
            result = _profiled(
                "attention.execute",
                turing_sla_attention_from_prequantized,
                quantized,
                return_stats=debug_route_density or profile_route_density,
            )
            if debug_route_density or profile_route_density:
                output, selected, possible = result
                _profile_route_stats(selected, possible)
                if debug_route_density:
                    output = collect_stats(
                        (output, selected, possible),
                        sla_call,
                        transformer_options,
                    )
            else:
                output = result
            return AttentionExecutionOutcome(output)

        prepared_executor.turing_utils_streamed_qkv_executor = (
            streamed_qkv_executor
        )

    def attention_override(original: Callable, *args, **kwargs):
        fallback = lambda *fallback_args, **fallback_kwargs: dense_override(
            original, *fallback_args, **fallback_kwargs
        )
        if is_dense(kwargs.get("transformer_options")):
            return fallback(*args, **kwargs)
        return turing_sla_sparse_attention(
            fallback,
            *args,
            min_sequence_tokens=min_sequence_tokens,
            sparsity_ratio=sparsity_ratio,
            prefix_policy=prefix_policy,
            manual_prefix_tokens=manual_prefix_tokens,
            sparse_reference_image=sparse_reference_image,
            sparse_reference_video=sparse_reference_video,
            sparse_reference_audio=sparse_reference_audio,
            debug_route_density=debug_route_density,
            use_w8a8=use_w8a8,
            **kwargs,
        )

    if split_prequantization_available():
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
            if is_dense(transformer_options):
                return run_dense_container(
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
            sla_call, reason = inspect(
                q.peek(),
                k.peek(),
                v.peek(),
                heads,
                mask=mask,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                transformer_options=transformer_options,
                kwargs=kwargs,
            )
            if reason is not None:
                return run_dense_container(
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
            quantized = prequantize_turing_sla_attention(
                query,
                key,
                value,
                sla_call,
                sparsity_ratio=sparsity_ratio,
                scale=kwargs.get("scale"),
                use_w8a8=use_w8a8,
            )
            del query, key, value
            profile_route_density = CUDA_PHASE_PROFILER.enabled
            result = turing_sla_attention_from_prequantized(
                quantized,
                return_stats=debug_route_density or profile_route_density,
            )
            if not (debug_route_density or profile_route_density):
                return result
            output, selected, possible = result
            _profile_route_stats(selected, possible)
            if debug_route_density:
                return collect_stats(
                    (output, selected, possible), sla_call, transformer_options
                )
            return output

        attention_override.container_function = container_function

    attention_override.turing_utils_attention_backend = "sla_sparse_attn"
    attention_override.turing_utils_attention_implementation = "bundled_sla_sparse"
    attention_override.turing_utils_dense_implementation = (
        dense_override.turing_utils_attention_implementation
    )
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
    override = make_sparse_attention_override(
        model.load_device,
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
    patched = install_sparse_attention_override(
        model,
        override,
        strategy="Sol sparse",
        backend="sol_sparse_attn",
        implementation="bundled_sol_sparse",
    )
    patched = patched.model
    dense_implementation = getattr(
        override,
        "turing_utils_dense_implementation",
        "selected_dense_backend",
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
        dense_implementation,
        "u8xs8_tensorcore" if use_w8a8 else "fp16_tensorcore",
        debug_route_density,
    )
    return patched


def apply_sla_attention_patch(
    model,
    min_sequence_tokens: int = 0,
    sparsity_ratio: float = SLA_SPARSITY_RATIO,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    sparse_reference_image: bool = SPARSE_REFERENCE_IMAGE,
    sparse_reference_video: bool = SPARSE_REFERENCE_VIDEO,
    sparse_reference_audio: bool = SPARSE_REFERENCE_AUDIO,
    dense_prefix_steps: int = SLA_DENSE_PREFIX_STEPS,
    dense_suffix_steps: int = SLA_DENSE_SUFFIX_STEPS,
    dense_prefix_layers: int = SLA_DENSE_PREFIX_LAYERS,
    dense_suffix_layers: int = SLA_DENSE_SUFFIX_LAYERS,
    debug_route_density: bool = False,
    use_w8a8: bool = SPARSE_USE_W8A8,
):
    override = make_sla_attention_override(
        model.load_device,
        min_sequence_tokens=min_sequence_tokens,
        sparsity_ratio=sparsity_ratio,
        prefix_policy=prefix_policy,
        manual_prefix_tokens=manual_prefix_tokens,
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
    patched = install_sparse_attention_override(
        model,
        override,
        strategy="SLA",
        backend="sla_sparse_attn",
        implementation="bundled_sla_sparse",
    )
    patched = patched.model
    LOG.info(
        "SLA sparse attention patch enabled: sparsity_ratio=%.2f "
        "topology=128x64 smooth_k=True prefix_policy=%s manual_prefix=%d "
        "sparse_reference=(image=%s,video=%s,audio=%s) "
        "dense_prefix_steps=%d dense_suffix_steps=%d "
        "dense_prefix_layers=%d dense_suffix_layers=%d "
        "dense_backend=%s pv_backend=%s debug_route_density=%s",
        sparsity_ratio,
        prefix_policy,
        manual_prefix_tokens,
        sparse_reference_image,
        sparse_reference_video,
        sparse_reference_audio,
        dense_prefix_steps,
        dense_suffix_steps,
        dense_prefix_layers,
        dense_suffix_layers,
        override.turing_utils_dense_implementation,
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
