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


LOG = logging.getLogger("comfyui-turing-utils")
_MIB = 1024**2
_LOGGED_DECISIONS: set[tuple] = set()


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


@dataclasses.dataclass(frozen=True, slots=True)
class AttentionDecision:
    mode: str
    rows: int
    heads: int
    head_group: int
    cache_quantized_input: bool
    available_bytes: int
    estimated_peak_bytes: int
    reserve_bytes: int

    @property
    def sharded(self) -> bool:
        return 0 < self.head_group < self.heads


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


def _runtime_memory(device: torch.device) -> tuple[int, int, int]:
    """Return usable remaining bytes, total reserve, and usable ceiling."""
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


def _align_rows(value: int, alignment: int, minimum: int) -> int:
    value = value // alignment * alignment
    return max(value, minimum) if value >= minimum else 0


def decide_activation_chunks(
    x: torch.Tensor,
    *,
    operation: str,
    hidden_size: int,
    expanded_size: int,
) -> ActivationDecision:
    """Select the full or streamed H3 activation path.

    ``expanded_size`` is the SwiGLU post-split width for ``mlp`` and the QKV
    inner width (heads * head_dim) for ``qkv``.
    """
    rows = int(x.shape[0])
    mode = _mode()
    if x.device.type != "cuda" or rows <= 0:
        return ActivationDecision(operation, mode, rows, 0, 0, 0, 0, 0)

    available, reserve, usable = _runtime_memory(x.device)
    element = int(x.element_size())
    hidden_size = int(hidden_size)
    expanded_size = int(expanded_size)

    # Leave room for one streamed block's weights, allocator fragmentation,
    # CUDA graph/kernel scratch, and the desktop compositor.  The compositor's
    # long-lived allocation is separately represented by --reserve-vram.
    safety = max(768 * _MIB, int(usable * 0.075))
    weight_scratch = 512 * _MIB
    working = max(available - safety - weight_scratch, 0)

    if operation == "mlp":
        # Persistent returned hidden output plus fc1 BF16, fused fc2 A8 input,
        # and the current hidden output tile. Long H3 contractions use the
        # fixed-workspace fused fc2 rather than a full INT32 accumulator.
        persistent = rows * hidden_size * element
        per_row = 2 * expanded_size * element + expanded_size + hidden_size * element
        cap, alignment, minimum = 32768, 256, 2048
    elif operation == "qkv":
        # Streamed projection retains Q/K INT8 and V BF16.  One tile holds the
        # packed BF16 projection and temporary Q/K INT8 results.
        persistent = rows * (4 * expanded_size)
        per_row = 3 * expanded_size * element + 2 * expanded_size
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
    if key not in _LOGGED_DECISIONS:
        LOG.info(
            "MiniMax H3 activation policy: op=%s mode=%s rows=%d path=%s "
            "chunk_rows=%d available=%.2f GiB reserve=%.2f GiB "
            "estimated_peak(full/selected)=%.2f/%.2f GiB",
            operation,
            mode,
            rows,
            "streamed" if decision.streamed else "throughput",
            chunk_rows,
            available / 1024**3,
            reserve / 1024**3,
            full_peak / 1024**3,
            streamed_peak / 1024**3,
        )
        _LOGGED_DECISIONS.add(key)
    return decision


def decide_attention_heads(
    x: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    compact_qk: bool,
    quantized_input: bool,
) -> AttentionDecision:
    """Choose a whole-head group without changing global sequence attention."""
    rows = int(x.shape[0])
    heads = int(heads)
    head_dim = int(head_dim)
    mode = _mode()
    if x.device.type != "cuda" or rows <= 0 or heads <= 0:
        return AttentionDecision(mode, rows, heads, heads, False, 0, 0, 0)

    available, reserve, usable = _runtime_memory(x.device)
    element = int(x.element_size())
    safety = max(768 * _MIB, int(usable * 0.075))
    weight_scratch = 512 * _MIB
    working = max(available - safety - weight_scratch, 0)
    output = rows * heads * head_dim * element
    input_cache = (
        rows * (int(x.shape[-1]) + 4) if quantized_input else 0
    )

    def group_local(group: int) -> int:
        features = rows * group * head_dim
        if not compact_qk:
            # BF16 Q/K/V plus the BF16 attention result.
            return features * 8
        # Q8 + K8 + V16 remain live for the complete sequence. The fused
        # quantizers retain five FP32 scale values per 64-token/head block.
        compact = features * 4 + rows * group * 5 * 4 // 64
        result = features * element
        # Projection is streamed inside every head group. Q and K must overlap
        # for fused RMSNorm/RoPE, while V can be emitted separately.
        projected_tile = min(rows, 16_384) * group * head_dim * element * 2
        return compact + result + projected_tile

    def peak(group: int, cache_input: bool) -> int:
        # The unsharded path returns its backend result directly and therefore
        # does not allocate a second full attention-output buffer.
        destination = 0 if group == heads else output
        return (
            destination
            + group_local(group)
            + (input_cache if cache_input and group != heads else 0)
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

    full_fits = peak(heads, False) <= working
    if (mode == "throughput" or full_fits) and override is None:
        head_group = heads
        cache_input = False
    else:
        head_group = groups[-1]
        cache_input = False
        for group in groups:
            if peak(group, quantized_input) <= working:
                head_group = group
                cache_input = quantized_input
                break
            if peak(group, False) <= working:
                head_group = group
                cache_input = False
                break
    estimated = peak(head_group, cache_input)
    decision = AttentionDecision(
        mode,
        rows,
        heads,
        head_group,
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
        cache_input,
        reserve // (256 * _MIB),
    )
    if key not in _LOGGED_DECISIONS:
        LOG.info(
            "MiniMax H3 attention policy: mode=%s rows=%d heads=%d "
            "head_group=%d input_cache=%s available=%.2f GiB "
            "reserve=%.2f GiB estimated_peak=%.2f GiB",
            mode,
            rows,
            heads,
            head_group,
            cache_input,
            available / 1024**3,
            reserve / 1024**3,
            estimated / 1024**3,
        )
        _LOGGED_DECISIONS.add(key)
    return decision


def decide_ffn_channels(
    x: torch.Tensor,
    *,
    expanded_size: int,
    chunk_rows: int,
) -> FFNChannelDecision:
    """Select an exact ConvRot-aligned FFN intermediate shard."""
    rows = int(x.shape[0])
    expanded_size = int(expanded_size)
    mode = _mode()
    if x.device.type != "cuda" or rows <= 0 or expanded_size <= 0:
        return FFNChannelDecision(
            mode, rows, expanded_size, chunk_rows, 0, 0, 0, 0
        )
    available, reserve, usable = _runtime_memory(x.device)
    safety = max(768 * _MIB, int(usable * 0.075))
    working = max(available - safety - 512 * _MIB, 0)
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
        requested = automatic if override is None else override
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
    if key not in _LOGGED_DECISIONS:
        LOG.info(
            "MiniMax H3 FFN policy: mode=%s rows=%d row_chunk=%d "
            "channel_chunk=%d available=%.2f GiB reserve=%.2f GiB "
            "estimated_peak=%.2f GiB",
            mode,
            rows,
            chunk_rows,
            chunk_channels,
            available / 1024**3,
            reserve / 1024**3,
            estimated / 1024**3,
        )
        _LOGGED_DECISIONS.add(key)
    return decision


__all__ = [
    "ActivationDecision",
    "AttentionDecision",
    "FFNChannelDecision",
    "decide_activation_chunks",
    "decide_attention_heads",
    "decide_ffn_channels",
]
