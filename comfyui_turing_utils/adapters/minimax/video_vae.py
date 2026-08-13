"""Node-local MiniMax H3 video VAE execution paths."""

from __future__ import annotations

import logging
import math
import queue
import threading

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import comfy.memory_management
import comfy.model_management
import comfy.ops
import comfy.quant_ops
import comfy.rmsnorm
import comfy.utils
import comfy_aimdo.model_vbar
from comfy.ldm.minimax import vae as h3_vae
from comfy.ldm.modules.attention import AttentionTensorContainer

from ...attention import make_attention_override
from ...attention.integration import execute_projected_attention
from ...attention.protocol import (
    ATTENTION_EXECUTOR_KEY,
    QKTransformSpec,
    RMSNormSpec,
    RotaryEmbeddingSpec,
)


TILE_SIZE = 256
TILE_OVERLAP = 64
_AUTO_DECODE_TILE_BATCH_LIMIT = 4
_AUTO_ENCODE_TILE_BATCH_LIMIT = 2
_DECODER_BYTES_PER_FP16_TOKEN = 64 * 1024
_ENCODER_BYTES_PER_FP16_PIXEL_FRAME = 1800
_ENCODER_FIXED_WORKSPACE = 256 * 1024**2
_DECODE_SAFETY_FACTOR = 1.08
_ENCODE_SAFETY_FACTOR = 1.05
_MULTIBAND_DOWNSAMPLE = 8
_MULTIBAND_HIGH_WEIGHT_POWER = 4
_SHARED_CORE_FEATHER_TOKENS = 1


def require_h3_video_vae(vae):
    vae.throw_exception_if_invalid()
    model = vae.first_stage_model
    if not isinstance(model, h3_vae.MiniMaxH3VideoVAE):
        raise ValueError("This node requires a MiniMax H3 video VAE")
    return model


def split_tiles(input_len: int, tile_size: int, overlap_min: int, ratio: int):
    if tile_size >= input_len:
        return [0], [input_len], []

    count = math.ceil(input_len / tile_size)
    while True:
        overlaps = [overlap_min] * (count - 1)
        remaining = tile_size * count - sum(overlaps) - input_len
        if remaining >= 0:
            break
        count += 1

    for i in range(remaining // ratio):
        overlaps[i % (count - 1)] += ratio

    starts = [0]
    for i in range(count - 1):
        starts.append(starts[-1] + tile_size - overlaps[i])
    return starts, [tile_size] * count, overlaps


def _spatial_tile_count(height, width, tile_size=TILE_SIZE):
    return len(split_tiles(height, tile_size, TILE_OVERLAP, 16)[0]) * len(
        split_tiles(width, tile_size, TILE_OVERLAP, 16)[0]
    )


def _decode_memory_requirement(
    vae,
    latent_shape,
    tiles_per_batch,
    dtype,
):
    model = vae.first_stage_model
    height = latent_shape[-2] * model.vae_ratio
    width = latent_shape[-1] * model.vae_ratio
    tile_height = min(height, TILE_SIZE)
    tile_width = min(width, TILE_SIZE)
    bounded_shape = (
        latent_shape[0],
        latent_shape[1],
        latent_shape[2],
        tile_height // model.vae_ratio,
        tile_width // model.vae_ratio,
    )
    official_estimate = int(vae.memory_used_decode(bounded_shape, dtype))

    if latent_shape[2] == 1:
        resident_tokens = 1
        resident_frames = 1
    else:
        resident_tokens = model.tokens_chunk_size + model.token_overlap
        resident_frames = resident_tokens * model.vae_ratio_t
    sequence = (
        resident_tokens
        * (tile_height // model.vae_ratio)
        * (tile_width // model.vae_ratio)
        + 1
        + model.decoder.num_register_tokens
    )
    dtype_scale = comfy.model_management.dtype_size(dtype) / 2.0
    # Dense H3 decoder blocks peak around 64 KiB per FP16 token once QKV and
    # the gated MLP workspace overlap. The complete decoded chunk also exists
    # as a compute-dtype canvas while FP32 finalized pixels are copied out.
    transformer_workspace = (
        latent_shape[0]
        * sequence
        * _DECODER_BYTES_PER_FP16_TOKEN
        * dtype_scale
        * tiles_per_batch
    )
    decoder_dim = model.decoder.transformer_blocks[0].scale1.numel()
    unique_tokens = (
        latent_shape[0] * resident_tokens * latent_shape[-2] * latent_shape[-1]
    )
    # The fixed decoder keeps one global image state plus global QKV and
    # attention workspaces during the shared prefix.
    transformer_workspace += (
        unique_tokens * decoder_dim * comfy.model_management.dtype_size(dtype) * 7
    )
    # The final projection is always evaluated as independent 256px windows,
    # even when no transformer blocks are assigned to the independent tail.
    window_count = _spatial_tile_count(height, width)
    window_state_tokens = latent_shape[0] * window_count * sequence
    transformer_workspace += (
        window_state_tokens * decoder_dim * comfy.model_management.dtype_size(dtype)
    )
    pixel_elements = (
        latent_shape[0] * model.decoder.out_channels * resident_frames * height * width
    )
    pixel_workspace = pixel_elements * (comfy.model_management.dtype_size(dtype) + 4)
    # Multiband uses an FP32 accumulation canvas, one tile-local low-pass
    # tensor, and two FP32 scalar normalization maps.
    pixel_workspace += (
        pixel_elements * (4 - comfy.model_management.dtype_size(dtype))
        + latent_shape[0]
        * model.decoder.out_channels
        * resident_frames
        * tile_height
        * tile_width
        * comfy.model_management.dtype_size(dtype)
        + 2 * height * width * 4
    )
    structural_estimate = int(
        (transformer_workspace + pixel_workspace) * _DECODE_SAFETY_FACTOR
    )
    # ComfyUI's estimate describes one internally tiled sample. Preserve its
    # fixed allowance and scale only the structural per-tile workspace here.
    return max(official_estimate, structural_estimate)


def _encode_memory_requirement(
    vae,
    pixel_shape,
    tile_size,
    tiles_per_batch,
    dtype,
):
    model = vae.first_stage_model
    batch, channels, frames, height, width = pixel_shape
    clip_frames = min(frames, model.clip_length)
    tile_height = min(height, tile_size)
    tile_width = min(width, tile_size)
    bounded_shape = (
        batch,
        channels,
        clip_frames,
        tile_height,
        tile_width,
    )
    official_estimate = int(vae.memory_used_encode(bounded_shape, dtype))

    dtype_scale = comfy.model_management.dtype_size(dtype) / 2.0
    # The causal CNN's high-resolution feature pyramid dominates the encoder.
    # This coefficient is deliberately conservative and was measured against
    # the full 2.6B-parameter H3 VAE at 256, 400, and 480 pixel tiles.
    convolution_workspace = (
        batch
        * clip_frames
        * tile_height
        * tile_width
        * _ENCODER_BYTES_PER_FP16_PIXEL_FRAME
        * dtype_scale
        * tiles_per_batch
    )
    buffer_count = 2 if frames > model.clip_length else 1
    input_buffers = (
        buffer_count
        * batch
        * channels
        * clip_frames
        * height
        * width
        * comfy.model_management.dtype_size(dtype)
    )
    structural_estimate = int(
        (convolution_workspace + input_buffers + _ENCODER_FIXED_WORKSPACE)
        * _ENCODE_SAFETY_FACTOR
    )
    return max(official_estimate, structural_estimate)


def _tile_memory_budget(vae):
    try:
        available = vae.patcher.get_free_memory(vae.device)
    except (AttributeError, RuntimeError):
        available = comfy.model_management.get_free_memory(vae.device)
    return max(
        0,
        int(available - comfy.model_management.extra_reserved_memory()),
    )


def _select_tiles_per_batch(
    vae,
    tile_count,
    memory_estimator,
    auto_limit,
):
    budget = _tile_memory_budget(vae)
    selected = 1
    for candidate in range(2, min(tile_count, auto_limit) + 1):
        if memory_estimator(candidate) > budget:
            break
        selected = candidate
    estimate = memory_estimator(selected)
    log = logging.warning if estimate > budget else logging.info
    log(
        "MiniMax H3 VAE auto tile batch selected %d/%d: estimated %.0f MiB, available %.0f MiB%s",
        selected,
        tile_count,
        estimate / 1024**2,
        budget / 1024**2,
        " (one tile exceeds the current budget)" if estimate > budget else "",
    )
    return selected, estimate


class _TileProgress:
    def __init__(self, total, device=None, description="H3 VAE"):
        self.total = int(total)
        self.bar = comfy.utils.ProgressBar(self.total)
        self.terminal = tqdm(
            total=self.total,
            desc=description,
            disable=not comfy.utils.PROGRESS_BAR_ENABLED,
        )
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.pending = None
        self.worker = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.pending = queue.Queue()
            self.worker = threading.Thread(
                target=self._consume,
                name="h3-vae-tile-progress",
                daemon=True,
            )
            self.worker.start()

    def _consume(self):
        while True:
            item = self.pending.get()
            if item is None:
                return
            event, count = item
            try:
                event.synchronize()
                self.bar.update(count)
                self.terminal.update(count)
            except RuntimeError:
                logging.exception("H3 VAE tile progress event failed")

    def update(self, count):
        count = int(count)
        if self.worker is None:
            self.bar.update(count)
            self.terminal.update(count)
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self.pending.put((event, count))

    def finish(self):
        if self.worker is not None:
            self.pending.put(None)
            self.worker.join()
            self.worker = None
        self.terminal.close()


def _norm_weight(module, name, reference):
    norm = getattr(module, name)
    weight = norm.weight
    if weight is not None:
        return weight
    cache_name = f"_turing_utils_{name}_unit_norm"
    cached = getattr(module, cache_name, None)
    if (
        cached is None
        or cached.device != reference.device
        or cached.dtype != reference.dtype
        or cached.numel() != module.dim_head
    ):
        cached = torch.ones(
            module.dim_head,
            device=reference.device,
            dtype=reference.dtype,
        )
        setattr(module, cache_name, cached)
    return cached


def _projected_attention(
    module,
    query,
    key,
    value,
    rotary_pos_emb,
    transformer_options,
):
    executor = transformer_options.get(ATTENTION_EXECUTOR_KEY)
    if callable(executor):
        query_norm = _norm_weight(module, "norm_q", query)
        key_norm = _norm_weight(module, "norm_k", key)
        rot_dim = int(rotary_pos_emb.shape[-3] * 2) if rotary_pos_emb is not None else 0
        transform = QKTransformSpec(
            query_norm=RMSNormSpec(query_norm, float(module.norm_q.eps), "head"),
            key_norm=RMSNormSpec(key_norm, float(module.norm_k.eps), "head"),
            rotary=RotaryEmbeddingSpec(
                rotary_pos_emb,
                rot_dim,
                "split_half" if rotary_pos_emb is not None else "none",
            ),
        )
        outcome = execute_projected_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            heads=module.heads,
            qk_transform=transform,
            transformer_options=transformer_options,
            container_factory=AttentionTensorContainer,
        )
        if outcome.supported:
            return outcome.output.nan_to_num_(0.0)

    query = comfy.rmsnorm.rms_norm(query, module.norm_q.weight, module.norm_q.eps)
    key = comfy.rmsnorm.rms_norm(key, module.norm_k.weight, module.norm_k.eps)

    if rotary_pos_emb is not None:
        rot = rotary_pos_emb.shape[-3] * 2
        query[..., :rot], key[..., :rot] = comfy.quant_ops.ck.apply_rope_split_half(
            query[..., :rot], key[..., :rot], rotary_pos_emb
        )

    query = AttentionTensorContainer(query.transpose(1, 2))
    key = AttentionTensorContainer(key.transpose(1, 2))
    value = AttentionTensorContainer(value.transpose(1, 2))
    out = h3_vae.optimized_attention(
        query,
        key,
        value,
        module.heads,
        skip_reshape=True,
        transformer_options=transformer_options,
    )
    return out.nan_to_num_(0.0)


def _apply_split_half_rope(value, rotary_pos_emb):
    if rotary_pos_emb is None:
        return value
    rot = int(rotary_pos_emb.shape[-3] * 2)
    apply_one = getattr(comfy.quant_ops.ck, "apply_rope_split_half1", None)
    if callable(apply_one):
        rotated = apply_one(value[..., :rot], rotary_pos_emb)
    else:
        source = (
            value[..., :rot]
            .reshape(*value.shape[:-1], 2, -1)
            .movedim(-2, -1)
            .unsqueeze(-2)
            .to(rotary_pos_emb.dtype)
        )
        rotated = (
            rotary_pos_emb[..., 0] * source[..., 0]
            + rotary_pos_emb[..., 1] * source[..., 1]
        )
        rotated = rotated.movedim(-1, -2).reshape(*value.shape[:-1], rot)
        rotated = rotated.to(value.dtype)
    if rot == value.shape[-1]:
        return rotated
    return torch.cat((rotated, value[..., rot:]), dim=-1)


def _asymmetric_projected_attention(
    module,
    query,
    key,
    value,
    query_rotary,
    key_rotary,
    transformer_options,
):
    """Execute core-query/full-halo attention without padding either side.

    Prepared attention normally fuses a common Q/K RoPE table.  Shared-core Q
    is a subset of the window while K spans the full window, so normalize and
    rotate them independently before handing the true asymmetric tensors to
    the selected SDPA/Sage/W8A8 backend.
    """

    query = comfy.rmsnorm.rms_norm(query, module.norm_q.weight, module.norm_q.eps)
    key = comfy.rmsnorm.rms_norm(key, module.norm_k.weight, module.norm_k.eps)
    query = _apply_split_half_rope(query, query_rotary)
    key = _apply_split_half_rope(key, key_rotary)
    out = h3_vae.optimized_attention(
        AttentionTensorContainer(query.transpose(1, 2)),
        AttentionTensorContainer(key.transpose(1, 2)),
        AttentionTensorContainer(value.transpose(1, 2)),
        module.heads,
        skip_reshape=True,
        transformer_options=transformer_options,
    )
    return out.nan_to_num_(0.0)


def _attention_forward(module, x, rotary_pos_emb, transformer_options):
    batch_size, seq_len, _ = x.shape
    qkv = module.to_qkv(x).view(batch_size, seq_len, -1, 3 * module.dim_head)
    query, key, value = torch.chunk(qkv, 3, dim=-1)
    out = _projected_attention(
        module,
        query,
        key,
        value,
        rotary_pos_emb,
        transformer_options,
    )
    return module.to_out(out)


def _attention_options(attention, device):
    override = make_attention_override(attention, device)
    options = {"optimized_attention_override": override}
    executor = getattr(override, "prepared_attention_executor", None)
    if callable(executor):
        options[ATTENTION_EXECUTOR_KEY] = executor
    return options


def _clear_attention_caches(decoder):
    for block in decoder.transformer_blocks:
        attention = block.attn
        for name in (
            "_turing_utils_norm_q_unit_norm",
            "_turing_utils_norm_k_unit_norm",
        ):
            if hasattr(attention, name):
                delattr(attention, name)


def _vbar_modules(module):
    return [child for child in module.modules() if hasattr(child, "_v")]


def _decoder_weight_stages(model):
    stages = []
    for index, block in enumerate(model.decoder.transformer_blocks):
        modules = _vbar_modules(block)
        if index == 0:
            modules = (
                _vbar_modules(model.post_quant_conv)
                + _vbar_modules(model.decoder.x_embedder)
                + modules
            )
        if index == len(model.decoder.transformer_blocks) - 1:
            modules += _vbar_modules(model.decoder.norm_out)
            modules += _vbar_modules(model.decoder.proj_out)
        stages.append(modules)
    return stages


def _encoder_weight_stages(model):
    stages = [_vbar_modules(model.encoder.conv_in)]
    for down in model.encoder.down:
        stages.extend(_vbar_modules(block) for block in down.block)
        if hasattr(down, "downsample"):
            stages.append(_vbar_modules(down.downsample))
    stages.append(
        _vbar_modules(model.encoder.norm_out)
        + _vbar_modules(model.encoder.conv_out)
        + _vbar_modules(model.quant_conv)
    )
    return stages


class _RetainedWeights:
    def __init__(self, stages, device, enabled):
        self.device = device
        self.enabled = enabled and not comfy.model_management.is_device_cpu(device)
        self.non_blocking = (
            comfy.model_management.NUM_STREAMS > 0
            and comfy.model_management.device_supports_non_blocking(device)
        )
        self.stages = stages
        self.streams = []
        self.started = 0
        self.attempted = False
        if not self.enabled:
            return
        if not any(self.stages):
            self.enabled = False

    @staticmethod
    def _clear_modules(modules):
        for module in modules:
            prefetch = getattr(module, "_prefetch", None)
            if prefetch is None:
                continue
            for param_key in ("weight", "bias"):
                lowvram = getattr(module, param_key + "_lowvram_function", None)
                if lowvram is not None:
                    lowvram.clear_prepared()
            if prefetch["signature"] is not None:
                comfy_aimdo.model_vbar.vbar_unpin(module._v)
            delattr(module, "_prefetch")

    def _prefetch(self, index):
        if not self.enabled or index >= len(self.stages) or index < self.started:
            return
        self.attempted = True
        modules = self.stages[index]
        if not modules:
            self.streams.append(None)
            self.started += 1
            return
        registerable_size = sum(
            comfy.memory_management.vram_aligned_size([module.weight, module.bias])
            for module in modules
        )
        stream = comfy.ops.cast_modules_with_vbar(
            modules, None, self.device, None, self.non_blocking
        )
        if not comfy.model_management.args.fast_disk:
            comfy.model_management.ensure_pin_registerable(registerable_size)
        if any(module._prefetch["signature"] is None for module in modules):
            comfy.model_management.sync_stream(self.device, stream)
            comfy.model_management.current_stream(self.device).synchronize()
            self._clear_modules(modules)
            for retained in self.stages[: self.started]:
                self._clear_modules(retained)
            self.enabled = False
            logging.warning(
                "H3 VAE could not retain the complete weight cycle; released the retained prefix and switched to synchronous streaming"
            )
            return
        self.streams.append(stream)
        self.started += 1

    def start(self):
        self._prefetch(0)

    def before_stage(self, index):
        if not self.enabled:
            return
        comfy.model_management.sync_stream(self.device, self.streams[index])
        self.streams[index] = None
        self._prefetch(index + 1)

    def finish(self):
        if not self.attempted:
            return
        # An exception can interrupt cast_modules_with_vbar before its stream is
        # returned to us.  Synchronize the device before removing any VBAR
        # signature in that case as well as on the normal path.
        comfy.model_management.synchronize()
        for modules in self.stages:
            self._clear_modules(modules)


def _fused_swiglu_eligible(linear):
    weight = linear.weight
    return bool(
        not comfy.model_management.in_training
        and isinstance(weight, comfy.ops.QuantizedTensor)
        and weight._layout_cls == "TensorWiseINT8Layout"
        and not getattr(weight._params, "transposed", False)
    )


class _SharedWindowLayout:
    def __init__(self, model, latent_t, latent_h, latent_w, device):
        self.latent_t = int(latent_t)
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)
        y_idx, y_len, y_overlap = split_tiles(
            latent_h * model.vae_ratio,
            TILE_SIZE,
            TILE_OVERLAP,
            model.vae_ratio,
        )
        x_idx, x_len, x_overlap = split_tiles(
            latent_w * model.vae_ratio,
            TILE_SIZE,
            TILE_OVERLAP,
            model.vae_ratio,
        )
        self.y_idx = [value // model.vae_ratio for value in y_idx]
        self.y_len = [value // model.vae_ratio for value in y_len]
        self.y_overlap = [value // model.vae_ratio for value in y_overlap]
        self.x_idx = [value // model.vae_ratio for value in x_idx]
        self.x_len = [value // model.vae_ratio for value in x_len]
        self.x_overlap = [value // model.vae_ratio for value in x_overlap]
        self.descriptors = [
            (i, j, yi, yl, xi, xl)
            for i, (yi, yl) in enumerate(zip(self.y_idx, self.y_len))
            for j, (xi, xl) in enumerate(zip(self.x_idx, self.x_len))
        ]
        self.window_h = self.y_len[0]
        self.window_w = self.x_len[0]
        self.window_tokens = self.latent_t * self.window_h * self.window_w
        self.image_tokens = self.latent_t * self.latent_h * self.latent_w

        token_indices = []
        for _i, _j, yi, yl, xi, xl in self.descriptors:
            temporal = torch.arange(self.latent_t, device=device)[:, None, None]
            rows = torch.arange(yi, yi + yl, device=device)[None, :, None]
            columns = torch.arange(xi, xi + xl, device=device)[None, None, :]
            indices = (
                temporal * self.latent_h * self.latent_w
                + rows * self.latent_w
                + columns
            )
            token_indices.append(indices.reshape(-1))
        self.token_indices = torch.stack(token_indices)

    @staticmethod
    def _axis_feather_weights(length, starts, lengths, device):
        centers = torch.tensor(
            [start + extent / 2 for start, extent in zip(starts, lengths)],
            dtype=torch.float32,
            device=device,
        )
        positions = torch.arange(length, dtype=torch.float32, device=device).add_(0.5)
        owners = (positions[:, None] - centers[None, :]).abs().argmin(dim=1)
        weights = torch.zeros(len(starts), length, dtype=torch.float32, device=device)
        weights[owners, torch.arange(length, device=device)] = 1.0
        feather_tokens = _SHARED_CORE_FEATHER_TOKENS
        for index in range(len(starts) - 1):
            boundary = (centers[index] + centers[index + 1]) / 2
            phase = (
                (positions - (boundary - feather_tokens)) / (2 * feather_tokens)
            ).clamp(0.0, 1.0)
            transition = (phase > 0.0) & (phase < 1.0)
            if not bool(transition.any()):
                continue
            left = torch.cos(phase[transition] * (math.pi / 2)).square()
            right = 1.0 - left
            weights[:, transition] = 0.0
            weights[index, transition] = left
            weights[index + 1, transition] = right

        coordinates = torch.arange(length, device=device)
        for index, (start, extent) in enumerate(zip(starts, lengths)):
            covered = (coordinates >= start) & (coordinates < start + extent)
            weights[index].mul_(covered)
        return weights / weights.sum(dim=0, keepdim=True).clamp_min_(1e-12)

    def query_groups(self):
        device = self.token_indices.device
        y_weights = self._axis_feather_weights(
            self.latent_h,
            self.y_idx,
            self.y_len,
            device,
        )
        x_weights = self._axis_feather_weights(
            self.latent_w,
            self.x_idx,
            self.x_len,
            device,
        )
        global_indices_by_window = []
        local_indices_by_window = []
        weights_by_window = []
        spatial_weight_sum = torch.zeros(
            self.latent_h,
            self.latent_w,
            dtype=torch.float32,
            device=device,
        )
        for i, j, yi, _yl, xi, _xl in self.descriptors:
            spatial_weights = y_weights[i, :, None] * x_weights[j, None, :]
            coordinates = torch.nonzero(spatial_weights > 0.0, as_tuple=False)
            rows, columns = coordinates[:, 0], coordinates[:, 1]
            token_weights = spatial_weights[rows, columns]
            spatial_weight_sum.index_put_(
                (rows, columns), token_weights, accumulate=True
            )
            temporal = torch.arange(self.latent_t, device=device)[:, None]
            spatial_global = rows * self.latent_w + columns
            spatial_local = (rows - yi) * self.window_w + (columns - xi)
            global_indices_by_window.append(
                (
                    temporal * self.latent_h * self.latent_w + spatial_global[None, :]
                ).reshape(-1)
            )
            local_indices_by_window.append(
                (
                    temporal * self.window_h * self.window_w + spatial_local[None, :]
                ).reshape(-1)
            )
            weights_by_window.append(token_weights.repeat(self.latent_t))

        if not torch.allclose(
            spatial_weight_sum,
            torch.ones_like(spatial_weight_sum),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError("shared-core feather weights do not sum to one")

        windows_by_count = {}
        for window_index, global_indices in enumerate(global_indices_by_window):
            windows_by_count.setdefault(int(global_indices.numel()), []).append(
                window_index
            )
        grouped = {}
        for query_tokens, windows in windows_by_count.items():
            grouped[query_tokens] = (
                torch.tensor(windows, dtype=torch.long, device=device),
                torch.stack([global_indices_by_window[index] for index in windows]),
                torch.stack([local_indices_by_window[index] for index in windows]),
                torch.stack([weights_by_window[index] for index in windows]),
            )
        return grouped

    @property
    def window_count(self):
        return len(self.descriptors)


def _feed_forward(module, value):
    if _fused_swiglu_eligible(module.w2):
        return comfy.ops.linear_input_act(module.w2, module.w1(value), "swiglu")
    return module(value)


def _reshape_decoder_patches(
    decoder,
    output,
    batch,
    latent_t,
    latent_h,
    latent_w,
):
    output = output.view(
        batch,
        latent_t,
        latent_h,
        latent_w,
        decoder.out_channels,
        decoder.patch_size_t,
        decoder.patch_size,
        decoder.patch_size,
    )
    output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return output.reshape(
        batch,
        decoder.out_channels,
        latent_t * decoder.patch_size_t,
        latent_h * decoder.patch_size,
        latent_w * decoder.patch_size,
    )


def _shared_core_multiband_decoder_forward(
    model,
    x,
    transformer_options,
    block_session,
    tiles_per_batch,
    progress,
    independent_tail_blocks,
):
    decoder = model.decoder
    batch, _, latent_t, latent_h, latent_w = x.shape
    layout = _SharedWindowLayout(
        model,
        latent_t,
        latent_h,
        latent_w,
        x.device,
    )
    h = decoder.x_embedder(x.flatten(2).transpose(1, 2))
    dim = h.shape[-1]
    suffix_tokens = 1 + decoder.num_register_tokens
    suffix = torch.cat(
        (
            comfy.ops.cast_to_input(decoder.register_tokens, h)
            .view(1, 1, decoder.num_register_tokens, dim)
            .expand(batch, layout.window_count, -1, -1),
            torch.zeros(
                batch,
                layout.window_count,
                1,
                dim,
                dtype=h.dtype,
                device=h.device,
            ),
        ),
        dim=2,
    ).contiguous()
    image_ids = h3_vae.create_token_ids(
        (latent_t, layout.window_h, layout.window_w),
        x.device,
        x.dtype,
    )
    suffix_ids = torch.zeros(
        (1, suffix_tokens, 3),
        dtype=image_ids.dtype,
        device=image_ids.device,
    )
    rotary = decoder.pos_embed(torch.cat((image_ids, suffix_ids), dim=1))
    suffix_rotary_indices = torch.arange(
        layout.window_tokens,
        layout.window_tokens + suffix_tokens,
        device=x.device,
    )
    linear_chunk = max(layout.window_tokens, tiles_per_batch * layout.window_tokens)
    blocks = list(decoder.transformer_blocks)
    independent_tail_blocks = min(max(0, int(independent_tail_blocks)), len(blocks))
    shared_block_count = len(blocks) - independent_tail_blocks
    query_groups = {
        query_tokens: (
            windows,
            global_indices,
            local_indices,
            weights.to(h.dtype),
        )
        for query_tokens, (
            windows,
            global_indices,
            local_indices,
            weights,
        ) in layout.query_groups().items()
    }

    for block_index, block in enumerate(blocks[:shared_block_count]):
        if block_index > 0:
            block_session.before_stage(block_index)
        attention = block.attn
        qkv_image = torch.empty(
            batch,
            layout.image_tokens,
            attention.heads,
            3 * attention.dim_head,
            dtype=h.dtype,
            device=h.device,
        )
        for token_start in range(0, layout.image_tokens, linear_chunk):
            token_end = min(token_start + linear_chunk, layout.image_tokens)
            normed = comfy.rmsnorm.rms_norm(
                h[:, token_start:token_end],
                block.norm1.weight,
                block.norm1.eps,
            )
            projected = attention.to_qkv(normed).view(
                batch,
                token_end - token_start,
                attention.heads,
                3 * attention.dim_head,
            )
            qkv_image[:, token_start:token_end].copy_(projected)

        image_attention = torch.zeros_like(h)
        for core_tokens in sorted(query_groups):
            (
                grouped_windows,
                grouped_global_indices,
                grouped_local_indices,
                grouped_weights,
            ) = query_groups[core_tokens]
            for group_start in range(0, grouped_windows.numel(), tiles_per_batch):
                window_tensor = grouped_windows[
                    group_start : group_start + tiles_per_batch
                ]
                group_slice = slice(group_start, group_start + window_tensor.numel())
                window_count = window_tensor.numel()
                key_indices = layout.token_indices.index_select(0, window_tensor)
                query_global_indices = grouped_global_indices[group_slice]
                query_local_indices = grouped_local_indices[group_slice]
                query_weights = grouped_weights[group_slice]

                window_qkv = qkv_image[:, key_indices].reshape(
                    batch * window_count,
                    layout.window_tokens,
                    attention.heads,
                    3 * attention.dim_head,
                )
                query_q = qkv_image[:, query_global_indices][..., : attention.dim_head]
                query_q = query_q.reshape(
                    batch * window_count,
                    core_tokens,
                    attention.heads,
                    attention.dim_head,
                )
                suffix_group = (
                    suffix.index_select(1, window_tensor)
                    .reshape(
                        batch * window_count,
                        suffix_tokens,
                        dim,
                    )
                    .clone()
                )
                suffix_normed = comfy.rmsnorm.rms_norm(
                    suffix_group,
                    block.norm1.weight,
                    block.norm1.eps,
                )
                suffix_qkv = attention.to_qkv(suffix_normed).view(
                    batch * window_count,
                    suffix_tokens,
                    attention.heads,
                    3 * attention.dim_head,
                )
                suffix_q, suffix_k, suffix_v = torch.chunk(suffix_qkv, 3, dim=-1)
                query = torch.cat((query_q, suffix_q), dim=1)
                key = torch.cat(
                    (
                        window_qkv[..., attention.dim_head : 2 * attention.dim_head],
                        suffix_k,
                    ),
                    dim=1,
                )
                value = torch.cat(
                    (window_qkv[..., 2 * attention.dim_head :], suffix_v),
                    dim=1,
                )

                query_rotary_indices = torch.cat(
                    (
                        query_local_indices,
                        suffix_rotary_indices.expand(window_count, -1),
                    ),
                    dim=1,
                )
                query_rotary = rotary[0][query_rotary_indices]
                query_rotary = (
                    query_rotary.unsqueeze(0)
                    .expand(batch, *query_rotary.shape)
                    .reshape(
                        batch * window_count,
                        core_tokens + suffix_tokens,
                        *query_rotary.shape[2:],
                    )
                )
                key_rotary = rotary.expand(batch * window_count, *rotary.shape[1:])
                attended = _asymmetric_projected_attention(
                    attention,
                    query,
                    key,
                    value,
                    query_rotary,
                    key_rotary,
                    transformer_options,
                ).view(
                    batch,
                    window_count,
                    core_tokens + suffix_tokens,
                    dim,
                )

                for local_index, global_indices in enumerate(query_global_indices):
                    weighted = attended[:, local_index, :core_tokens].mul(
                        query_weights[local_index].view(1, -1, 1)
                    )
                    image_attention.index_add_(1, global_indices, weighted)

                suffix_attention = attended[:, :, core_tokens:].reshape(
                    batch * window_count, suffix_tokens, dim
                )
                suffix_group.addcmul_(
                    attention.to_out(suffix_attention),
                    comfy.ops.cast_to_input(block.scale1, suffix_group),
                )
                suffix_normed = comfy.rmsnorm.rms_norm(
                    suffix_group,
                    block.norm2.weight,
                    block.norm2.eps,
                )
                suffix_group.addcmul_(
                    _feed_forward(block.ff, suffix_normed),
                    comfy.ops.cast_to_input(block.scale2, suffix_group),
                )
                suffix[:, window_tensor] = suffix_group.view(
                    batch,
                    window_count,
                    suffix_tokens,
                    dim,
                )
                if progress is not None:
                    progress.update(window_count)

        del qkv_image
        for token_start in range(0, layout.image_tokens, linear_chunk):
            token_end = min(token_start + linear_chunk, layout.image_tokens)
            h_slice = h[:, token_start:token_end]
            h_slice.addcmul_(
                attention.to_out(image_attention[:, token_start:token_end]),
                comfy.ops.cast_to_input(block.scale1, h_slice),
            )
            normed = comfy.rmsnorm.rms_norm(
                h_slice,
                block.norm2.weight,
                block.norm2.eps,
            )
            h_slice.addcmul_(
                _feed_forward(block.ff, normed),
                comfy.ops.cast_to_input(block.scale2, h_slice),
            )
        del image_attention

    return _independent_window_tail(
        decoder,
        h,
        suffix,
        rotary,
        layout,
        blocks,
        shared_block_count,
        transformer_options,
        block_session,
        tiles_per_batch,
        progress,
    )


def _independent_window_tail(
    decoder,
    h,
    suffix,
    rotary,
    layout,
    blocks,
    first_block,
    transformer_options,
    block_session,
    tiles_per_batch,
    progress,
):
    batch = h.shape[0]
    dim = h.shape[-1]
    suffix_tokens = suffix.shape[2]
    window_states = torch.cat(
        (
            h[:, layout.token_indices],
            suffix,
        ),
        dim=2,
    ).contiguous()
    sequence = layout.window_tokens + suffix_tokens

    for block_index in range(first_block, len(blocks)):
        if block_index > 0:
            block_session.before_stage(block_index)
        block = blocks[block_index]
        for window_start in range(0, layout.window_count, tiles_per_batch):
            window_end = min(window_start + tiles_per_batch, layout.window_count)
            window_count = window_end - window_start
            state = window_states[:, window_start:window_end].reshape(
                batch * window_count, sequence, dim
            )
            normed = comfy.rmsnorm.rms_norm(
                state,
                block.norm1.weight,
                block.norm1.eps,
            )
            state.addcmul_(
                _attention_forward(
                    block.attn,
                    normed,
                    rotary.expand(batch * window_count, *rotary.shape[1:]),
                    transformer_options,
                ),
                comfy.ops.cast_to_input(block.scale1, state),
            )
            normed = comfy.rmsnorm.rms_norm(
                state,
                block.norm2.weight,
                block.norm2.eps,
            )
            state.addcmul_(
                _feed_forward(block.ff, normed),
                comfy.ops.cast_to_input(block.scale2, state),
            )
            if progress is not None:
                progress.update(window_count)

    pixel_y_idx = [value * decoder.patch_size for value in layout.y_idx]
    pixel_y_len = [value * decoder.patch_size for value in layout.y_len]
    pixel_x_idx = [value * decoder.patch_size for value in layout.x_idx]
    pixel_x_len = [value * decoder.patch_size for value in layout.x_len]
    assembler = _MultibandPixelAssembler(
        pixel_y_idx,
        pixel_y_len,
        pixel_x_idx,
        pixel_x_len,
        layout.latent_h * decoder.patch_size,
        layout.latent_w * decoder.patch_size,
        h.device,
    )
    for window_start in range(0, layout.window_count, tiles_per_batch):
        window_end = min(window_start + tiles_per_batch, layout.window_count)
        window_count = window_end - window_start
        image_states = window_states[
            :, window_start:window_end, : layout.window_tokens
        ].reshape(batch * window_count, layout.window_tokens, dim)
        projected = decoder.proj_out(decoder.norm_out(image_states))
        decoded = _reshape_decoder_patches(
            decoder,
            projected,
            batch * window_count,
            layout.latent_t,
            layout.window_h,
            layout.window_w,
        ).view(
            batch,
            window_count,
            decoder.out_channels,
            layout.latent_t * decoder.patch_size_t,
            layout.window_h * decoder.patch_size,
            layout.window_w * decoder.patch_size,
        )
        for local_index, window_index in enumerate(range(window_start, window_end)):
            assembler.add(window_index, decoded[:, local_index])
    return assembler.finish()


def _decode_spatial(
    model,
    z,
    transformer_options,
    block_session,
    tiles_per_batch,
    progress,
    independent_tail_blocks,
):
    block_session.before_stage(0)
    z = model.post_quant_conv(z)
    return _shared_core_multiband_decoder_forward(
        model,
        z,
        transformer_options,
        block_session,
        tiles_per_batch,
        progress,
        independent_tail_blocks,
    )


def _axis_multiband_weight(
    index,
    starts,
    lengths,
    dtype,
    device,
):
    length = lengths[index]
    weight = torch.ones(length, dtype=dtype, device=device)
    if index > 0:
        overlap = starts[index - 1] + lengths[index - 1] - starts[index]
        phase = (torch.arange(overlap, dtype=dtype, device=device) + 0.5) / overlap
        weight[:overlap].mul_(0.5 - 0.5 * torch.cos(math.pi * phase))
    if index + 1 < len(starts):
        overlap = starts[index] + lengths[index] - starts[index + 1]
        phase = (torch.arange(overlap, dtype=dtype, device=device) + 0.5) / overlap
        weight[-overlap:].mul_(
            torch.flip(0.5 - 0.5 * torch.cos(math.pi * phase), dims=(0,))
        )
    return weight


def _multiband_window_weights(
    y_index,
    x_index,
    y_starts,
    y_lengths,
    x_starts,
    x_lengths,
    dtype,
    device,
):
    y_weight = _axis_multiband_weight(y_index, y_starts, y_lengths, dtype, device)
    x_weight = _axis_multiband_weight(x_index, x_starts, x_lengths, dtype, device)
    low = y_weight[:, None] * x_weight[None, :]
    high = low.pow(_MULTIBAND_HIGH_WEIGHT_POWER)
    return low, high


def _multiband_denominators(
    descriptors,
    y_starts,
    y_lengths,
    x_starts,
    x_lengths,
    height,
    width,
    dtype,
    device,
):
    low_sum = torch.zeros(height, width, dtype=dtype, device=device)
    high_sum = torch.zeros_like(low_sum)
    for i, j, _zi, _zl, _zj, _zw in descriptors:
        low, high = _multiband_window_weights(
            i,
            j,
            y_starts,
            y_lengths,
            x_starts,
            x_lengths,
            dtype,
            device,
        )
        y = y_starts[i]
        x = x_starts[j]
        low_sum[y : y + y_lengths[i], x : x + x_lengths[j]].add_(low)
        high_sum[y : y + y_lengths[i], x : x + x_lengths[j]].add_(high)
    tiny = torch.finfo(dtype).tiny
    return low_sum.clamp_min_(tiny), high_sum.clamp_min_(tiny)


def _spatial_lowpass(tile):
    batch, channels, frames, height, width = tile.shape
    frame_batch = tile.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    low_height = max(1, math.ceil(height / _MULTIBAND_DOWNSAMPLE))
    low_width = max(1, math.ceil(width / _MULTIBAND_DOWNSAMPLE))
    low = F.interpolate(
        frame_batch,
        size=(low_height, low_width),
        mode="area",
    )
    low = F.interpolate(
        low,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return low.view(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


class _MultibandPixelAssembler:
    def __init__(
        self,
        y_starts,
        y_lengths,
        x_starts,
        x_lengths,
        height,
        width,
        device,
    ):
        self.y_starts = y_starts
        self.y_lengths = y_lengths
        self.x_starts = x_starts
        self.x_lengths = x_lengths
        self.height = height
        self.width = width
        self.columns = len(x_starts)
        descriptors = [
            (i, j, 0, 0, 0, 0)
            for i in range(len(y_starts))
            for j in range(len(x_starts))
        ]
        self.low_sum, self.high_sum = _multiband_denominators(
            descriptors,
            y_starts,
            y_lengths,
            x_starts,
            x_lengths,
            height,
            width,
            torch.float32,
            device,
        )
        self.canvas = None

    def add(self, window_index, tile):
        i, j = divmod(window_index, self.columns)
        y, x = self.y_starts[i], self.x_starts[j]
        if self.canvas is None:
            self.canvas = torch.zeros(
                *tile.shape[:-2],
                self.height,
                self.width,
                dtype=torch.float32,
                device=tile.device,
            )
        low_weight, high_weight = _multiband_window_weights(
            i,
            j,
            self.y_starts,
            self.y_lengths,
            self.x_starts,
            self.x_lengths,
            torch.float32,
            tile.device,
        )
        low_weight = (
            low_weight
            / self.low_sum[y : y + self.y_lengths[i], x : x + self.x_lengths[j]]
        )
        high_weight = (
            high_weight
            / self.high_sum[y : y + self.y_lengths[i], x : x + self.x_lengths[j]]
        )
        low = _spatial_lowpass(tile)
        high = tile.float() - low.float()
        contribution = low.float().mul(low_weight)
        contribution.addcmul_(high, high_weight)
        self.canvas[..., y : y + tile.shape[-2], x : x + tile.shape[-1]].add_(
            contribution
        )

    def finish(self):
        return self.canvas


def _encode_moments(model, x, module_session):
    if module_session is None:
        return model._encode_moments(x)

    stage = 0
    module_session.before_stage(stage)
    h = model.encoder.conv_in(x)
    stage += 1
    for down in model.encoder.down:
        for block in down.block:
            module_session.before_stage(stage)
            h = block(h)
            stage += 1
        if hasattr(down, "downsample"):
            module_session.before_stage(stage)
            h = down.downsample(h)
            stage += 1
    module_session.before_stage(stage)
    h = torch.nn.functional.silu(model.encoder.norm_out(h))
    return model.quant_conv(model.encoder.conv_out(h))


def _tiled_encode(
    model,
    x,
    tile_size,
    tile_overlap,
    module_session=None,
    tiles_per_batch=1,
    progress=None,
):
    height, width = x.shape[-2:]
    y_idx, y_len, y_overlap = split_tiles(
        height, tile_size, tile_overlap, model.vae_ratio
    )
    x_idx, x_len, x_overlap = split_tiles(
        width, tile_size, tile_overlap, model.vae_ratio
    )

    descriptors = [
        (i, j, i_pos, i_len, j_pos, j_len)
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len))
        for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len))
    ]
    rows = [[None] * len(x_idx) for _ in y_idx]
    source_batch = x.shape[0]
    for start in range(0, len(descriptors), tiles_per_batch):
        group = descriptors[start : start + tiles_per_batch]
        inputs = [
            x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
            for _i, _j, i_pos, i_len, j_pos, j_len in group
        ]
        batched = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=0)
        encoded = _encode_moments(model, batched, module_session)
        for (i, j, *_bounds), tile in zip(group, encoded.split(source_batch, dim=0)):
            rows[i][j] = tile
        if progress is not None:
            progress.update(len(group))

    latent_y_overlap = [value // model.vae_ratio for value in y_overlap]
    latent_x_overlap = [value // model.vae_ratio for value in x_overlap]
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = model.blend(
                    rows[i - 1][j], tile, latent_y_overlap[i - 1], dim=-2
                )
            if j > 0:
                tile = model.blend(row[j - 1], tile, latent_x_overlap[j - 1], dim=-1)
            if i < len(rows) - 1:
                tile = tile[..., : -latent_y_overlap[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, : -latent_x_overlap[j]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


class _PixelWriter:
    def __init__(self, output, model, device):
        self.output = output
        self.model = model
        self.write_pos = 0
        self.double_buffer = (
            output.device.type == "cpu"
            and device.type == "cuda"
            and torch.cuda.is_available()
        )
        self.copy_stream = None
        self.staging = [None, None]
        self.pending = [None, None]
        self.next_slot = 0
        if self.double_buffer:
            self.copy_stream = torch.cuda.Stream(device=device)

    def _flush(self, index):
        entry = self.pending[index]
        if entry is None:
            return
        staging, done, start, frames, _source = entry
        done.synchronize()
        self.output[:, :, start : start + frames].copy_(staging[:, :, :frames])
        self.pending[index] = None

    def write(self, part):
        part_frames = part.shape[2]
        copy_frames = min(part_frames, max(0, self.output.shape[2] - self.write_pos))
        if copy_frames <= 0:
            return
        part = self.model._finalize_pixels(part).to(self.output.dtype)
        start = self.write_pos
        self.write_pos += copy_frames
        part = part[:, :, :copy_frames]
        if not self.double_buffer:
            self.output[:, :, start : start + copy_frames].copy_(part)
            return

        index = self.next_slot
        self._flush(index)
        staging = self.staging[index]
        if (
            staging is None
            or staging.shape != part.shape
            or staging.dtype != part.dtype
        ):
            try:
                staging = torch.empty_like(part, device="cpu", pin_memory=True)
            except RuntimeError:
                self._flush(0)
                self._flush(1)
                self.double_buffer = False
                self.copy_stream = None
                self.staging = [None, None]
                logging.warning(
                    "H3 VAE could not allocate a pinned decoder buffer; using synchronous FP32 pixel copies"
                )
                self.output[:, :, start : start + copy_frames].copy_(part)
                return
            self.staging[index] = staging
        ready = torch.cuda.Event()
        done = torch.cuda.Event()
        torch.cuda.current_stream(part.device).record_event(ready)
        self.copy_stream.wait_event(ready)
        with torch.cuda.stream(self.copy_stream):
            staging.copy_(part, non_blocking=True)
            done.record(self.copy_stream)
        part.record_stream(self.copy_stream)
        # Keep the source alive until the side-stream D2H copy completes. This
        # also prevents the caching allocator from recycling its storage for
        # the next decoded chunk.
        self.pending[index] = (staging, done, start, copy_frames, part)
        self.next_slot = 1 - index
        self._flush(self.next_slot)

    def finish(self):
        self._flush(0)
        self._flush(1)
        return self.output


def _decode_temporal(
    model,
    z,
    transformer_options,
    block_session,
    tiles_per_batch,
    progress,
    independent_tail_blocks,
):
    chunk_dec = model.tokens_chunk_size * model.vae_ratio_t
    split_count = int(model.token_drop > 0) + 1
    output = torch.empty(
        model.decode_output_shape(z.shape),
        dtype=torch.float32,
        device=comfy.model_management.intermediate_device(),
    )
    writer = _PixelWriter(output, model, z.device)

    pad_tokens, num_chunks = model._decode_temporal_chunks(z.shape[2])
    if pad_tokens > 0:
        pad_z = z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)
        z = torch.cat((z, pad_z), dim=2)

    dec_overlap = None
    for i in range(num_chunks):
        start = i * model.tokens_chunk_size
        end = start + model.tokens_chunk_size + model.token_overlap
        clip_z = z[:, :, start:end]
        clip_dec = _decode_spatial(
            model,
            clip_z,
            transformer_options,
            block_session,
            tiles_per_batch,
            progress,
            independent_tail_blocks,
        )

        for j in range(split_count):
            frame_start = j * chunk_dec
            frame_end = min(frame_start + chunk_dec, clip_dec.shape[2])
            part = clip_dec[:, :, frame_start:frame_end]
            part = part[:, :, model.frame_pre_padding :]
            if j == 0:
                if dec_overlap is not None:
                    part = model.blend(dec_overlap, part, model.frame_overlap, dim=-3)
                    dec_overlap = None
                writer.write(part)
            else:
                dec_overlap = part.contiguous()

        if i == num_chunks - 1 and dec_overlap is not None:
            writer.write(dec_overlap)
            dec_overlap = None
    return writer.finish()


def decode_video(
    vae,
    latent,
    attention="sdpa",
    independent_tail_blocks=2,
):
    model = require_h3_video_vae(vae)
    # H3 advertises FP16/FP32 to ComfyUI and defaults to FP16 on supported
    # NVIDIA GPUs, including Turing.  Follow the dtype used to load this VAE so
    # an explicit global --fp32-vae override cannot create mixed-dtype modules.
    compute_dtype = vae.vae_dtype
    block_count = len(model.decoder.transformer_blocks)
    independent_tail_blocks = int(independent_tail_blocks)
    if not 0 <= independent_tail_blocks <= block_count:
        raise ValueError(
            "H3 VAE independent_tail_blocks must be between 0 and "
            f"{block_count}, got {independent_tail_blocks}"
        )
    tile_count = _spatial_tile_count(
        latent.shape[-2] * model.vae_ratio,
        latent.shape[-1] * model.vae_ratio,
    )
    tile_tokens = (
        min(latent.shape[-2] * model.vae_ratio, TILE_SIZE) // model.vae_ratio
    ) * (min(latent.shape[-1] * model.vae_ratio, TILE_SIZE) // model.vae_ratio)
    duplicate_ratio = tile_count * tile_tokens / (
        latent.shape[-2] * latent.shape[-1]
    )
    logging.info(
        "Experimental H3 VAE shared-core multiband decoder active: windows=%d duplicate_spatial_ratio=%.2fx independent_tail_blocks=%d",
        tile_count,
        duplicate_ratio,
        independent_tail_blocks,
    )
    batch_tiles, memory = _select_tiles_per_batch(
        vae,
        tile_count,
        lambda count: _decode_memory_requirement(
            vae,
            latent.shape,
            count,
            compute_dtype,
        ),
        _AUTO_DECODE_TILE_BATCH_LIMIT,
    )
    comfy.model_management.load_models_gpu(
        [vae.patcher], memory_required=memory, force_full_load=vae.disable_offload
    )

    block_session = None
    progress = None
    try:
        transformer_options = _attention_options(attention, vae.device)
        z = latent.to(device=vae.device, dtype=compute_dtype)
        mean = model.latents_mean.view(1, -1, 1, 1, 1).to(z)
        std = model.latents_std.view(1, -1, 1, 1, 1).to(z)
        z = z * std + mean
        temporal_chunks = (
            1 if z.shape[2] == 1 else model._decode_temporal_chunks(z.shape[2])[1]
        )
        progress_units = tile_count * temporal_chunks * block_count
        progress = _TileProgress(
            progress_units,
            z.device,
            "H3 VAE Decode Windows",
        )
        block_session = _RetainedWeights(_decoder_weight_stages(model), z.device, True)
        block_session.start()
        if z.shape[2] == 1:
            dec = _decode_spatial(
                model,
                z,
                transformer_options,
                block_session,
                batch_tiles,
                progress,
                independent_tail_blocks,
            )[:, :, -1:]
            dec = model._finalize_pixels(dec)
            return dec.to(vae.output_device).movedim(1, -1)
        dec = _decode_temporal(
            model,
            z,
            transformer_options,
            block_session,
            batch_tiles,
            progress,
            independent_tail_blocks,
        )
        return dec.movedim(1, -1)
    finally:
        if progress is not None:
            progress.finish()
        if block_session is not None:
            block_session.finish()
        _clear_attention_caches(model.decoder)
        vae.patcher.partially_unload(vae.patcher.offload_device, 1e30)


def _encode_clip(
    model,
    clip,
    tile_size,
    tile_overlap,
    module_session=None,
    tiles_per_batch=1,
    progress=None,
):
    return _tiled_encode(
        model,
        model._normalize_pixels(clip),
        tile_size,
        tile_overlap,
        module_session,
        tiles_per_batch,
        progress,
    )


def _encode_temporal(
    model,
    x,
    device,
    compute_dtype,
    tile_size,
    tile_overlap,
    module_session,
    tiles_per_batch,
    progress,
):
    z_list = []
    for start in range(0, x.shape[2], model.clip_length):
        clip = x[:, :, start : start + model.clip_length].to(device)
        if clip.dtype != compute_dtype:
            clip = clip.to(compute_dtype)
        if clip.shape[2] < model.clip_length:
            pad = clip[:, :, -1:].repeat(1, 1, model.clip_length - clip.shape[2], 1, 1)
            clip = torch.cat((clip, pad), dim=2)
        z_list.append(
            _encode_clip(
                model,
                clip,
                tile_size,
                tile_overlap,
                module_session,
                tiles_per_batch,
                progress,
            )
        )
    z = torch.cat(z_list, dim=2)
    return z[:, :, : -model.token_drop] if model.token_drop > 0 else z


def _encode_temporal_buffered(
    model,
    pixels,
    process_input,
    compute_dtype,
    device,
    tile_size,
    tile_overlap,
    module_session,
    tiles_per_batch,
    progress,
):
    copy_stream = torch.cuda.Stream(device=device)
    staging = [None, None]
    device_clips = [None, None]
    copy_done = [None, None]
    compute_done = [None, None]

    def prepare(clip_index, slot):
        if copy_done[slot] is not None:
            copy_done[slot].synchronize()
        start = clip_index * model.clip_length
        clip = pixels[:, :, start : start + model.clip_length]
        if clip.shape[2] < model.clip_length:
            pad = clip[:, :, -1:].repeat(1, 1, model.clip_length - clip.shape[2], 1, 1)
            clip = torch.cat((clip, pad), dim=2)
        # Keep the transport path in FP32.  Casting pixels on the host costs a
        # full extra pass and changes the transfer representation; the VAE
        # activation cast happens on the GPU immediately before compute.
        clip = process_input(clip).float()
        if (
            staging[slot] is None
            or staging[slot].shape != clip.shape
            or staging[slot].dtype != clip.dtype
        ):
            try:
                staging[slot] = torch.empty_like(clip, device="cpu", pin_memory=True)
            except RuntimeError as error:
                raise _PinnedBufferUnavailable from error
            device_clips[slot] = torch.empty(
                clip.shape,
                dtype=compute_dtype,
                device=device,
            )
        staging[slot].copy_(clip)
        if compute_done[slot] is not None:
            copy_stream.wait_event(compute_done[slot])
        event = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            device_clips[slot].copy_(staging[slot], non_blocking=True)
            event.record(copy_stream)
        copy_done[slot] = event

    count = math.ceil(pixels.shape[2] / model.clip_length)
    prepare(0, 0)
    z_list = []
    current_stream = torch.cuda.current_stream(device)
    for clip_index in range(count):
        slot = clip_index % 2
        current_stream.wait_event(copy_done[slot])
        clip = device_clips[slot]
        z_list.append(
            _encode_clip(
                model,
                clip,
                tile_size,
                tile_overlap,
                module_session,
                tiles_per_batch,
                progress,
            )
        )
        done = torch.cuda.Event()
        done.record(current_stream)
        compute_done[slot] = done
        if clip_index + 1 < count:
            prepare(clip_index + 1, 1 - slot)
    z = torch.cat(z_list, dim=2)
    return z[:, :, : -model.token_drop] if model.token_drop > 0 else z


class _PinnedBufferUnavailable(RuntimeError):
    pass


def encode_video(
    vae,
    pixels,
):
    model = require_h3_video_vae(vae)
    pixels = vae.vae_encode_crop_pixels(pixels).movedim(-1, 1)
    if pixels.ndim < 5:
        pixels = pixels.movedim(1, 0).unsqueeze(0)
    compute_dtype = vae.vae_dtype
    tile_size = TILE_SIZE
    tile_count = _spatial_tile_count(
        pixels.shape[-2],
        pixels.shape[-1],
        tile_size,
    )
    batch_tiles, memory = _select_tiles_per_batch(
        vae,
        tile_count,
        lambda count: _encode_memory_requirement(
            vae,
            pixels.shape,
            tile_size,
            count,
            compute_dtype,
        ),
        _AUTO_ENCODE_TILE_BATCH_LIMIT,
    )
    tile_overlap = TILE_OVERLAP
    comfy.model_management.load_models_gpu(
        [vae.patcher], memory_required=memory, force_full_load=vae.disable_offload
    )

    module_session = None
    progress = None
    try:
        module_session = _RetainedWeights(
            _encoder_weight_stages(model), vae.device, True
        )
        module_session.start()
        temporal_clips = (
            1
            if pixels.shape[2] == 1
            else math.ceil(pixels.shape[2] / model.clip_length)
        )
        progress = _TileProgress(
            tile_count * temporal_clips,
            vae.device,
            "H3 VAE Encode",
        )
        if pixels.shape[2] == 1:
            x = vae.process_input(pixels).float().to(vae.device)
            if x.dtype != compute_dtype:
                x = x.to(compute_dtype)
            moments = _encode_clip(
                model,
                x,
                tile_size,
                tile_overlap,
                module_session,
                batch_tiles,
                progress,
            )[:, :, -1:]
        elif pixels.device.type == "cpu" and torch.cuda.is_available():
            try:
                moments = _encode_temporal_buffered(
                    model,
                    pixels,
                    vae.process_input,
                    compute_dtype,
                    vae.device,
                    tile_size,
                    tile_overlap,
                    module_session,
                    batch_tiles,
                    progress,
                )
            except _PinnedBufferUnavailable:
                comfy.model_management.synchronize()
                logging.warning(
                    "H3 VAE could not allocate pinned encoder buffers; using synchronous FP32 pixel copies"
                )
                x = vae.process_input(pixels).float()
                moments = _encode_temporal(
                    model,
                    x,
                    vae.device,
                    compute_dtype,
                    tile_size,
                    tile_overlap,
                    module_session,
                    batch_tiles,
                    progress,
                )
        else:
            x = vae.process_input(pixels).float()
            moments = _encode_temporal(
                model,
                x,
                vae.device,
                compute_dtype,
                tile_size,
                tile_overlap,
                module_session,
                batch_tiles,
                progress,
            )
        mean = torch.chunk(moments.float(), 2, dim=1)[0]
        latent_mean = model.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        latent_std = model.latents_std.view(1, -1, 1, 1, 1).to(mean)
        return ((mean - latent_mean) / latent_std).to(
            device=vae.output_device, dtype=torch.float32
        )
    finally:
        if progress is not None:
            progress.finish()
        if module_session is not None:
            module_session.finish()
        vae.patcher.partially_unload(vae.patcher.offload_device, 1e30)
