"""Runtime activation-memory policy for MiniMax H3 inference.

The policy deliberately uses ComfyUI's live free-memory accounting and its
``--reserve-vram`` value.  It therefore adapts to Dynamic VRAM instead of
assuming that model weights are permanently resident.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os

import torch

from ...hardware import device_capabilities


LOG = logging.getLogger("comfyui-turing-utils")
_MIB = 1024**2
_VBAR_PAGE_BYTES = 32 * _MIB
_ATTENTION_QUERY_TILE_ROWS = 64
_ATTENTION_TARGET_WAVES = 4
_ATTENTION_CTAS_PER_SM = 2
_MAX_BALANCED_SHARDS = 4
_MIN_SATURATED_GEMM_WIDTH = 1024
_LOGGED_DECISIONS: set[tuple] = set()


@dataclasses.dataclass(slots=True)
class ActivationRuntimePlan:
    """Mutable policy state owned by one sampler invocation.

    CUDA free memory moves up and down while DynamicVRAM maps weights and while
    PyTorch retires temporary buffers.  Treating every observation as a fresh
    budget allowed a later layer to promote itself back to the full path.  The
    per-operation, per-row-count low-water mark makes automatic decisions
    monotonic for the lifetime of a sampler without leaking state between
    queued executions. Attention, QKV, and MLP have different live-buffer
    boundaries, so sharing one floor between them would unnecessarily force
    the later MLP onto its streamed path.
    """

    available_floors: dict[tuple[str, int | None, int, str], int] = (
        dataclasses.field(default_factory=dict)
    )
    reclaim_requests: dict[tuple[str, int | None, int, str], int] = (
        dataclasses.field(default_factory=dict)
    )
    logged_decisions: set[tuple] = dataclasses.field(default_factory=set)

    def observe_available(
        self,
        device: torch.device,
        rows: int,
        operation: str,
        available_bytes: int,
    ) -> int:
        key = (device.type, device.index, int(rows), str(operation))
        previous = self.available_floors.get(key)
        floor = int(available_bytes) if previous is None else min(
            previous, int(available_bytes)
        )
        self.available_floors[key] = floor
        return floor

    def should_request_reclaim(
        self,
        device: torch.device,
        rows: int,
        operation: str,
        deficit_bytes: int,
    ) -> bool:
        key = (device.type, device.index, int(rows), str(operation))
        previous = self.reclaim_requests.get(key, 0)
        deficit_bytes = int(deficit_bytes)
        if deficit_bytes <= previous + 64 * _MIB:
            return False
        self.reclaim_requests[key] = deficit_bytes
        return True


@dataclasses.dataclass(frozen=True, slots=True)
class ActivationDecision:
    operation: str
    mode: str
    rows: int
    chunk_rows: int
    available_bytes: int
    full_peak_bytes: int
    streamed_peak_bytes: int
    reserve_bytes: int

    @property
    def streamed(self) -> bool:
        return 0 < self.chunk_rows < self.rows

    @property
    def tier(self) -> int:
        return 1 if self.streamed else 0


@dataclasses.dataclass(frozen=True, slots=True)
class AttentionDecision:
    mode: str
    rows: int
    heads: int
    head_group: int
    saturation_group: int
    cache_quantized_input: bool
    available_bytes: int
    estimated_peak_bytes: int
    reserve_bytes: int

    @property
    def sharded(self) -> bool:
        return 0 < self.head_group < self.heads

    @property
    def tier(self) -> int:
        return 2 if self.sharded else 0


@dataclasses.dataclass(frozen=True, slots=True)
class FFNChannelDecision:
    mode: str
    rows: int
    expanded_size: int
    chunk_rows: int
    chunk_channels: int
    available_bytes: int
    estimated_peak_bytes: int
    reserve_bytes: int

    @property
    def sharded(self) -> bool:
        return 0 < self.chunk_channels < self.expanded_size

    @property
    def tier(self) -> int:
        if self.sharded:
            return 3
        return 1 if 0 < self.chunk_rows < self.rows else 0


def _mode() -> str:
    value = os.environ.get(
        "COMFYUI_TURING_UTILS_H3_ACTIVATION_MODE", "auto"
    ).strip().lower()
    aliases = {
        "speed": "throughput",
        "fast": "throughput",
        "safe": "balanced",
        "memory": "balanced",
        "lowvram": "balanced",
    }
    value = aliases.get(value, value)
    return value if value in {"auto", "throughput", "balanced"} else "auto"


def _override_chunk_rows(operation: str) -> int | None:
    names = (
        f"COMFYUI_TURING_UTILS_H3_{operation.upper()}_CHUNK_ROWS",
        "COMFYUI_TURING_UTILS_H3_ACTIVATION_CHUNK_ROWS",
    )
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return max(int(raw), 0)
        except ValueError:
            LOG.warning("Ignoring invalid %s=%r", name, raw)
    return None


def _override_head_group() -> int | None:
    raw = os.environ.get("COMFYUI_TURING_UTILS_H3_HEAD_GROUP")
    if raw is None:
        return None
    try:
        return max(int(raw), 0)
    except ValueError:
        LOG.warning(
            "Ignoring invalid COMFYUI_TURING_UTILS_H3_HEAD_GROUP=%r", raw
        )
        return None


def _override_ffn_channels() -> int | None:
    raw = os.environ.get("COMFYUI_TURING_UTILS_H3_FFN_CHUNK_CHANNELS")
    if raw is None:
        return None
    try:
        return max(int(raw), 0)
    except ValueError:
        LOG.warning(
            "Ignoring invalid COMFYUI_TURING_UTILS_H3_FFN_CHUNK_CHANNELS=%r",
            raw,
        )
        return None


def _dynamic_vbars(
    base_model,
    device: torch.device,
    *,
    include_current: bool = False,
) -> tuple[object, ...]:
    """Return unique inactive DynamicVRAM VBARs on ``device``.

    Evicting the current diffusion VBAR between transformer operations causes
    the same weight pages to be transferred back immediately. The current
    VBAR can still be included for read-only diagnostics, but normal headroom
    requests deliberately target inactive models only.
    """
    if base_model is None:
        return ()
    device = torch.device(device)
    current = getattr(base_model, "current_patcher", None)
    candidates = []
    try:
        import comfy.model_management as model_management

        candidates.extend(model_management.loaded_models())
    except (ImportError, AttributeError, RuntimeError, TypeError):
        pass

    result = []
    seen = set()

    def resolve(patcher):
        try:
            if not patcher.is_dynamic():
                return None
            load_device = torch.device(
                getattr(
                    patcher,
                    "load_device",
                    device if patcher is current else None,
                )
            )
            if load_device != device:
                return None
            return patcher._vbar_get()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    current_vbar = resolve(current) if current is not None else None
    current_vbar_id = id(current_vbar) if current_vbar is not None else None
    for patcher in candidates:
        vbar = resolve(patcher)
        if vbar is None or id(vbar) == current_vbar_id:
            continue
        if id(vbar) in seen:
            continue
        seen.add(id(vbar))
        result.append(vbar)
    if (
        include_current
        and current_vbar is not None
        and current_vbar_id not in seen
    ):
        result.append(current_vbar)
    return tuple(result)


def _current_model_is_dynamic(base_model, device: torch.device) -> bool:
    if base_model is None:
        return False
    try:
        patcher = base_model.current_patcher
        return bool(
            patcher.is_dynamic()
            and torch.device(patcher.load_device) == torch.device(device)
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _vbar_reclaimable_bytes(vbar) -> int:
    """Count resident, unpinned 32 MiB pages without noisy VBAR analysis."""
    try:
        residency = vbar.get_residency()
        freeable_pages = sum(
            1 for status in residency if (int(status) & 1) and not (int(status) & 2)
        )
        reclaimable = freeable_pages * _VBAR_PAGE_BYTES
        loaded_size = int(vbar.loaded_size())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0
    return max(min(reclaimable, loaded_size), 0)


def _dynamic_vram_reclaimable(
    base_model,
    device: torch.device,
    *,
    include_current: bool = False,
) -> int:
    return sum(
        _vbar_reclaimable_bytes(vbar)
        for vbar in _dynamic_vbars(
            base_model,
            device,
            include_current=include_current,
        )
    )


def _runtime_memory(
    device: torch.device,
    base_model=None,
) -> tuple[int, int, int]:
    """Return immediately usable bytes, total reserve, and usable ceiling.

    Evictable model pages are intentionally excluded. Counting them here made
    the policy select a larger attention tier and then evict hot weights to
    realize that hypothetical budget, trading exact head sharding for PCIe
    reload churn. Reclaimable inactive pages remain an emergency reserve for
    ``ensure_dynamic_vram_headroom`` only.
    """
    import comfy.model_management as model_management

    total = int(model_management.get_total_memory(device))
    reserve = int(model_management.extra_reserved_memory())
    usable = max(total - reserve, 0)
    allocated = int(torch.cuda.memory_allocated(device))
    free = int(model_management.get_free_memory(device))
    # get_free_memory includes unused PyTorch cache.  The second bound is what
    # makes --reserve-vram a hard ceiling even when the display is temporarily
    # idle and nvidia-smi reports the memory as free.
    return max(min(free, usable - allocated), 0), reserve, usable


def _planning_available(
    runtime_plan: ActivationRuntimePlan | None,
    device: torch.device,
    rows: int,
    available: int,
    mode: str,
    operation: str,
) -> int:
    if runtime_plan is None or mode == "throughput":
        return available
    return runtime_plan.observe_available(
        device,
        rows,
        operation,
        available,
    )


def _should_log(
    runtime_plan: ActivationRuntimePlan | None,
    key: tuple,
) -> bool:
    logged = (
        runtime_plan.logged_decisions
        if runtime_plan is not None
        else _LOGGED_DECISIONS
    )
    if key in logged:
        return False
    logged.add(key)
    return True


def _memory_diagnostics(device: torch.device) -> tuple[int, int, int, int]:
    """Best-effort allocator and DynamicVRAM pressure counters."""
    allocated = reserved = raw_free = aimdo_usage = 0
    try:
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        raw_free = int(torch.cuda.mem_get_info(device)[0])
    except (RuntimeError, TypeError):
        pass
    try:
        import comfy.memory_management as memory_management
        import comfy_aimdo.control as aimdo_control

        if memory_management.aimdo_enabled:
            # Unlike vbars_analyze(), this is a read-only counter. The analyze
            # API intentionally reports every currently pinned page as a
            # warning and must not be called from normal policy telemetry.
            aimdo_usage = int(aimdo_control.get_total_vram_usage())
    except (ImportError, AttributeError, RuntimeError, TypeError):
        pass
    return allocated, reserved, raw_free, aimdo_usage


def _log_memory_diagnostics(device: torch.device, base_model=None) -> str:
    allocated, reserved, raw_free, aimdo_usage = _memory_diagnostics(device)
    result = (
        f"torch_active={allocated / 1024**3:.2f} GiB "
        f"torch_reserved={reserved / 1024**3:.2f} GiB "
        f"cuda_free={raw_free / 1024**3:.2f} GiB "
        f"aimdo_usage={aimdo_usage / 1024**3:.2f} GiB"
    )
    if base_model is not None:
        reclaimable = _dynamic_vram_reclaimable(base_model, device)
        result += (
            " inactive_reclaimable="
            f"{reclaimable / 1024**3:.2f} GiB"
        )
    return result


def ensure_dynamic_vram_headroom(
    base_model,
    device: torch.device,
    *,
    rows: int,
    operation: str,
    estimated_peak_bytes: int,
    runtime_plan: ActivationRuntimePlan | None = None,
) -> int:
    """Ask an AIMDO VBAR to release mappings only when a selected tier cannot fit.

    This is deliberately not used to promote a layer to a faster tier.  It is
    a last-mile handshake that protects the already selected activation plan
    from weight mappings consuming its irreducible headroom between planning
    and allocation. Non-DynamicVRAM ComfyUI installations are a no-op.
    """
    device = torch.device(device)
    if base_model is None or device.type != "cuda":
        return 0
    try:
        available, _reserve, usable = _runtime_memory(device)
    except (ImportError, RuntimeError, TypeError):
        return 0
    safety = max(768 * _MIB, int(usable * 0.075))
    desired = int(estimated_peak_bytes) + safety + 512 * _MIB
    deficit = max(desired - available, 0)
    if deficit <= 0:
        return 0
    if runtime_plan is not None and not runtime_plan.should_request_reclaim(
        device, rows, operation, deficit
    ):
        return 0

    freed = 0
    remaining = deficit
    for vbar in _dynamic_vbars(base_model, device):
        free_memory = getattr(vbar, "free_memory", None)
        if not callable(free_memory):
            continue
        try:
            released = max(int(free_memory(remaining) or 0), 0)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            LOG.debug("DynamicVRAM headroom request was unavailable: %s", error)
            continue
        freed += released
        remaining = max(remaining - released, 0)
        if remaining <= 0:
            break
    if freed > 0:
        LOG.info(
            "MiniMax H3 DynamicVRAM headroom: op=%s rows=%d requested=%.1f MiB "
            "released=%.1f MiB %s",
            operation,
            rows,
            deficit / _MIB,
            freed / _MIB,
            _log_memory_diagnostics(device, base_model),
        )
    return freed


def _align_rows(value: int, alignment: int, minimum: int) -> int:
    value = value // alignment * alignment
    return max(value, minimum) if value >= minimum else 0


def balanced_saturation_size(
    total: int,
    *,
    alignment: int,
    minimum: int,
    max_shards: int = _MAX_BALANCED_SHARDS,
) -> int:
    """Return the smallest aligned, balanced shard near the MFU plateau."""
    total = int(total)
    alignment = max(int(alignment), 1)
    target = max(
        int(minimum),
        math.ceil(total / max(int(max_shards), 1)),
    )
    legal = [
        size
        for size in range(alignment, total + 1, alignment)
        if total % size == 0 and size >= target
    ]
    if legal:
        return min(legal)
    return min(
        total,
        math.ceil(target / alignment) * alignment,
    )


def _attention_saturation_group(
    device: torch.device,
    *,
    rows: int,
    heads: int,
    head_dim: int,
    legal_groups: list[int],
) -> int:
    """Find the smallest legal group that already supplies ample GPU work.

    Attention exposes one CTA per 64-query tile and head. Four resident-grid
    waves are sufficient to make launch width cease being the dominant MFU
    limiter. QKV projection also keeps at least a 1024-channel output tile,
    while at most four balanced groups bound repeated launch/anchor overhead.
    """
    capabilities = device_capabilities(device)
    sm_count = max(int(capabilities.multiprocessor_count), 1)
    query_blocks = max(
        math.ceil(int(rows) / _ATTENTION_QUERY_TILE_ROWS),
        1,
    )
    grid_heads = math.ceil(
        sm_count * _ATTENTION_CTAS_PER_SM * _ATTENTION_TARGET_WAVES
        / query_blocks
    )
    gemm_heads = math.ceil(_MIN_SATURATED_GEMM_WIDTH / int(head_dim))
    pass_heads = math.ceil(int(heads) / _MAX_BALANCED_SHARDS)
    minimum = max(grid_heads, gemm_heads, pass_heads, 1)
    balanced = [
        group
        for group in legal_groups
        if group >= minimum and heads % group == 0
    ]
    if balanced:
        return min(balanced)
    candidates = [group for group in legal_groups if group >= minimum]
    return min(candidates) if candidates else max(legal_groups)


def estimate_attention_lifecycle_peak(
    *,
    rows: int,
    heads: int,
    head_dim: int,
    hidden_size: int,
    element_size: int,
    head_group: int,
    compact_qk: bool,
    cache_quantized_input: bool,
    quantized_value: bool,
) -> int:
    """Estimate the peak across projection, V8 preparation and attention."""
    rows = int(rows)
    heads = int(heads)
    head_dim = int(head_dim)
    hidden_size = int(hidden_size)
    element_size = int(element_size)
    group = int(head_group)
    features = rows * group * head_dim
    output = rows * heads * head_dim * element_size
    destination = 0 if group == heads else output
    input_cache = rows * (hidden_size + 4) if cache_quantized_input else 0
    if not compact_qk:
        return destination + features * 8 + input_cache

    qk_blocks = (rows + 63) // 64
    qk_scales = qk_blocks * group * 5 * 4
    compact = features * (2 + element_size) + qk_scales
    value_int8 = features if quantized_value else 0
    padded_blocks = ((qk_blocks + 15) // 16) * 16
    summaries = (
        3 * group * padded_blocks * head_dim * 2
        + 2 * group * head_dim * 4
    ) if quantized_value else 0
    result = features * element_size
    execution_peak = compact + value_int8 + summaries + result
    tile_rows = min(rows, 16_384)
    projected_tile = (
        tile_rows * group * head_dim * element_size * 2
        + tile_rows * (hidden_size + 4)
    )
    return (
        destination
        + max(compact + projected_tile, execution_peak)
        + input_cache
    )


def decide_activation_chunks(
    x: torch.Tensor,
    *,
    operation: str,
    hidden_size: int,
    expanded_size: int,
    heads: int | None = None,
    runtime_plan: ActivationRuntimePlan | None = None,
    base_model=None,
) -> ActivationDecision:
    """Select the full or streamed H3 activation path.

    ``expanded_size`` is the SwiGLU post-split width for ``mlp`` and the QKV
    inner width (heads * head_dim) for ``qkv``.
    """
    rows = int(x.shape[0])
    mode = _mode()
    if x.device.type != "cuda" or rows <= 0:
        return ActivationDecision(operation, mode, rows, 0, 0, 0, 0, 0)

    available, reserve, usable = _runtime_memory(x.device, base_model)
    planned_available = _planning_available(
        runtime_plan,
        x.device,
        rows,
        available,
        mode,
        operation,
    )
    element = int(x.element_size())
    hidden_size = int(hidden_size)
    expanded_size = int(expanded_size)

    # Leave room for one streamed block's weights, allocator fragmentation,
    # CUDA graph/kernel scratch, and the desktop compositor.  The compositor's
    # long-lived allocation is separately represented by --reserve-vram.
    safety = max(768 * _MIB, int(usable * 0.075))
    weight_scratch = 512 * _MIB
    working = max(planned_available - safety - weight_scratch, 0)

    if operation == "mlp":
        # Persistent returned hidden output plus fc1 BF16, fused fc2 A8 input,
        # and the current hidden output tile. Long H3 contractions use the
        # fixed-workspace fused fc2 rather than a full INT32 accumulator.
        persistent = rows * hidden_size * element
        per_row = 2 * expanded_size * element + expanded_size + hidden_size * element
        # At H3's 5,376x14,336 contractions a 16K-row tile already exposes
        # far more Tensor Core work than the GPU can run concurrently. Larger
        # tiles increase the transient by about 1.26 GiB per additional 16K
        # rows without improving steady-state MFU.
        cap, alignment, minimum = 16384, 256, 2048
    elif operation == "qkv":
        # Streamed projection retains Q/K INT8 and V BF16.  One tile holds the
        # packed BF16 projection plus its quantized input row.  Q/K scales are
        # allocated for ceil(rows/64) blocks and remain live with Q/K/V.
        # ``expanded_size`` is heads*head_dim. There are five FP32 scale lanes
        # per head block; infer heads from H3's 128-wide heads when possible.
        inferred_heads = (
            max(int(heads), 1)
            if heads is not None
            else max(expanded_size // 128, 1)
        )
        scale_bytes = ((rows + 63) // 64) * inferred_heads * 5 * 4
        persistent = (
            rows * (2 * expanded_size + expanded_size * element)
            + scale_bytes
        )
        per_row = (
            3 * expanded_size * element
            + hidden_size
            + 4
        )
        cap, alignment, minimum = 16384, 64, 1024
    else:
        raise ValueError(f"unknown H3 activation operation: {operation}")

    full_peak = persistent + rows * per_row
    override = _override_chunk_rows(operation)
    if mode == "throughput" and override is None:
        chunk_rows = 0
    elif override is not None:
        chunk_rows = min(rows, _align_rows(override, alignment, minimum))
    elif mode == "auto" and full_peak <= int(working * 0.86):
        chunk_rows = 0
    else:
        tile_budget = max(working - persistent, 0)
        chunk_rows = min(
            rows,
            cap,
            _align_rows(tile_budget // max(per_row, 1), alignment, minimum),
        )
        if chunk_rows <= 0:
            # A small tile is still materially safer than falling back to the
            # full activation.  An eventual CUDA OOM will then report the real
            # irreducible floor instead of requesting a multi-GiB temporary.
            chunk_rows = min(rows, minimum)

    streamed_peak = persistent + (chunk_rows or rows) * per_row
    decision = ActivationDecision(
        operation,
        mode,
        rows,
        chunk_rows,
        available,
        full_peak,
        streamed_peak,
        reserve,
    )
    key = (
        x.device.index,
        operation,
        mode,
        rows,
        chunk_rows,
        reserve // (256 * _MIB),
    )
    if _should_log(runtime_plan, key):
        LOG.info(
            "MiniMax H3 activation policy: op=%s mode=%s tier=%d rows=%d path=%s "
            "chunk_rows=%d available=%.2f GiB reserve=%.2f GiB "
            "planned_available=%.2f GiB estimated_peak(full/selected)=%.2f/%.2f GiB %s",
            operation,
            mode,
            decision.tier,
            rows,
            "streamed" if decision.streamed else "throughput",
            chunk_rows,
            available / 1024**3,
            reserve / 1024**3,
            planned_available / 1024**3,
            full_peak / 1024**3,
            streamed_peak / 1024**3,
            _log_memory_diagnostics(x.device, base_model),
        )
    return decision


def decide_attention_heads(
    x: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    compact_qk: bool,
    quantized_input: bool,
    quantized_value: bool = False,
    runtime_plan: ActivationRuntimePlan | None = None,
    base_model=None,
) -> AttentionDecision:
    """Choose a whole-head group without changing global sequence attention."""
    rows = int(x.shape[0])
    heads = int(heads)
    head_dim = int(head_dim)
    mode = _mode()
    if x.device.type != "cuda" or rows <= 0 or heads <= 0:
        return AttentionDecision(
            mode,
            rows,
            heads,
            heads,
            heads,
            False,
            0,
            0,
            0,
        )

    available, reserve, usable = _runtime_memory(x.device, base_model)
    planned_available = _planning_available(
        runtime_plan,
        x.device,
        rows,
        available,
        mode,
        "attention",
    )
    element = int(x.element_size())
    safety = max(768 * _MIB, int(usable * 0.075))
    weight_scratch = 512 * _MIB
    working = max(planned_available - safety - weight_scratch, 0)

    def peak(group: int, cache_input: bool) -> int:
        return estimate_attention_lifecycle_peak(
            rows=rows,
            heads=heads,
            head_dim=head_dim,
            hidden_size=int(x.shape[-1]),
            element_size=element,
            head_group=group,
            compact_qk=compact_qk,
            cache_quantized_input=cache_input and group != heads,
            quantized_value=quantized_value,
        )

    # A cut is legal whenever both sides cover complete ConvRot-256 blocks.
    # This is the TP-style gcd boundary: D128 permits every two heads, for
    # example, rather than only divisors such as 56/28/14/8/4/2.
    block_heads = 256 // math.gcd(256, head_dim)
    groups = [
        group
        for group in range(heads, 0, -1)
        if group % block_heads == 0
        and (heads - group) % block_heads == 0
    ]
    if not groups:
        groups = [heads]
    override = _override_head_group()
    if override is not None:
        groups = [group for group in groups if group <= override] or [groups[-1]]

    saturation_group = (
        groups[0]
        if override is not None
        else _attention_saturation_group(
            x.device,
            rows=rows,
            heads=heads,
            head_dim=head_dim,
            legal_groups=groups,
        )
    )

    # Head sharding changes only the number of exact passes, whereas evicting
    # hot model pages forces immediate PCIe reloads. Keep a small allocator
    # margin and accept another head pass instead of manufacturing headroom by
    # weight eviction.
    fit_limit = int(working * 0.96)
    full_fits = peak(heads, False) <= fit_limit
    allow_input_cache = bool(
        quantized_input and not _current_model_is_dynamic(base_model, x.device)
    )
    if (mode == "throughput" or full_fits) and override is None:
        head_group = heads
        cache_input = False
    else:
        head_group = groups[-1]
        cache_input = False
        fitting = []
        for group in groups:
            cached_fits = (
                allow_input_cache and peak(group, True) <= fit_limit
            )
            plain_fits = peak(group, False) <= fit_limit
            if cached_fits or plain_fits:
                fitting.append((group, cached_fits))
        saturated = [
            item for item in fitting if item[0] == saturation_group
        ]
        selected = (
            saturated[0]
            if saturated
            else (fitting[0] if fitting else None)
        )
        if selected is not None:
            head_group = selected[0]
            cache_input = bool(selected[1])
    estimated = peak(head_group, cache_input)
    decision = AttentionDecision(
        mode,
        rows,
        heads,
        head_group,
        saturation_group,
        cache_input,
        available,
        estimated,
        reserve,
    )
    key = (
        x.device.index,
        "attention_heads",
        mode,
        rows,
        heads,
        head_group,
        saturation_group,
        cache_input,
        reserve // (256 * _MIB),
    )
    if _should_log(runtime_plan, key):
        LOG.info(
            "MiniMax H3 attention policy: mode=%s tier=%d rows=%d heads=%d "
            "head_group=%d saturation_group=%d input_cache=%s available=%.2f GiB "
            "reserve=%.2f GiB planned_available=%.2f GiB estimated_peak=%.2f GiB %s",
            mode,
            decision.tier,
            rows,
            heads,
            head_group,
            saturation_group,
            cache_input,
            available / 1024**3,
            reserve / 1024**3,
            planned_available / 1024**3,
            estimated / 1024**3,
            _log_memory_diagnostics(x.device, base_model),
        )
    return decision


def decide_ffn_channels(
    x: torch.Tensor,
    *,
    expanded_size: int,
    chunk_rows: int,
    runtime_plan: ActivationRuntimePlan | None = None,
    base_model=None,
) -> FFNChannelDecision:
    """Select an exact ConvRot-aligned FFN intermediate shard."""
    rows = int(x.shape[0])
    expanded_size = int(expanded_size)
    mode = _mode()
    if x.device.type != "cuda" or rows <= 0 or expanded_size <= 0:
        return FFNChannelDecision(
            mode, rows, expanded_size, chunk_rows, 0, 0, 0, 0
        )
    available, reserve, usable = _runtime_memory(x.device, base_model)
    planned_available = _planning_available(
        runtime_plan,
        x.device,
        rows,
        available,
        mode,
        "mlp",
    )
    safety = max(768 * _MIB, int(usable * 0.075))
    working = max(planned_available - safety - 512 * _MIB, 0)
    tile_rows = min(rows, chunk_rows or rows)
    hidden = int(x.shape[-1])
    persistent = rows * hidden * int(x.element_size())
    # Full fc1 BF16, fused INT8 activation, and the BF16 output tile.
    unsharded = persistent + tile_rows * (expanded_size * 5 + hidden * 2)
    override = _override_ffn_channels()
    if override is None and (mode == "throughput" or unsharded <= working):
        chunk_channels = 0
        estimated = unsharded
    else:
        # An explicit channel override without row streaming must not create a
        # sequence-sized INT32 accumulator.  This affects only the opt-in or
        # tight-memory channel path; the normal full path stays unchanged.
        if chunk_rows <= 0:
            tile_rows = min(rows, 4096)
            chunk_rows = tile_rows
        # Retain the complete compressed A8 row instead of a larger INT32
        # output accumulator. Current fc1-input A8 and final BF16 output are
        # also live; only the 2*C BF16 gate/up shard plus A8 scratch scales
        # with the selected channel width.
        fixed_per_row = expanded_size + hidden * 3
        budget = max(
            working - persistent - tile_rows * fixed_per_row, 0
        )
        automatic = max(
            (budget // max(tile_rows * 5, 1)) // 256 * 256,
            256,
        )
        saturation_channels = balanced_saturation_size(
            expanded_size,
            alignment=256,
            minimum=_MIN_SATURATED_GEMM_WIDTH,
        )
        requested = (
            min(automatic, saturation_channels)
            if override is None
            else override
        )
        chunk_channels = min(
            expanded_size,
            max((requested // 256) * 256, 256),
        )
        estimated = (
            persistent
            + tile_rows * (fixed_per_row + chunk_channels * 5)
        )
    decision = FFNChannelDecision(
        mode,
        rows,
        expanded_size,
        chunk_rows,
        chunk_channels,
        available,
        estimated,
        reserve,
    )
    key = (
        x.device.index,
        "ffn_channels",
        mode,
        rows,
        chunk_rows,
        chunk_channels,
        reserve // (256 * _MIB),
    )
    if _should_log(runtime_plan, key):
        LOG.info(
            "MiniMax H3 FFN policy: mode=%s tier=%d rows=%d row_chunk=%d "
            "channel_chunk=%d available=%.2f GiB reserve=%.2f GiB "
            "planned_available=%.2f GiB estimated_peak=%.2f GiB %s",
            mode,
            decision.tier,
            rows,
            chunk_rows,
            chunk_channels,
            available / 1024**3,
            reserve / 1024**3,
            planned_available / 1024**3,
            estimated / 1024**3,
            _log_memory_diagnostics(x.device, base_model),
        )
    return decision


__all__ = [
    "ActivationRuntimePlan",
    "ActivationDecision",
    "AttentionDecision",
    "FFNChannelDecision",
    "balanced_saturation_size",
    "decide_activation_chunks",
    "decide_attention_heads",
    "decide_ffn_channels",
    "ensure_dynamic_vram_headroom",
    "estimate_attention_lifecycle_peak",
]
