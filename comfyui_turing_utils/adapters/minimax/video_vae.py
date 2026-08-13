"""Node-local MiniMax H3 video VAE execution paths."""

from __future__ import annotations

import logging
import math

import torch

import comfy.memory_management
import comfy.model_management
import comfy.ops
import comfy.quant_ops
import comfy.rmsnorm
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


TILE_PRESETS = ("auto", "256", "288", "320", "384", "480")
_FIXED_TILE_EDGES = {str(edge): edge for edge in (256, 288, 320, 384, 480)}
_AUTO_TILE_EDGES = tuple(range(256, 481, 16))
TILE_OVERLAP = 64
_DECODER_BYTES_PER_FP16_TOKEN = 64 * 1024
_ENCODER_BYTES_PER_FP16_PIXEL_FRAME = 1800
_ENCODER_FIXED_WORKSPACE = 256 * 1024**2
_DECODE_SAFETY_FACTOR = 1.08
_ENCODE_SAFETY_FACTOR = 1.05


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


def resolve_tile_preset(
    preset: str,
    height: int,
    width: int,
    *,
    memory_limit: int | None = None,
    memory_estimator=None,
) -> int:
    if preset in _FIXED_TILE_EDGES:
        return _FIXED_TILE_EDGES[preset]
    if preset != "auto":
        raise ValueError(f"Unknown H3 VAE tile preset: {preset}")
    if (memory_limit is None) != (memory_estimator is None):
        raise ValueError("memory_limit and memory_estimator must be supplied together")

    # Search aligned sizes and minimize repeated tile area. The upper bound
    # avoids admitting arbitrarily large per-tile activations.
    best = None
    for edge in _AUTO_TILE_EDGES:
        if (
            memory_estimator is not None
            and memory_estimator(edge) > memory_limit
            and edge != 256
        ):
            continue
        _y_starts, y_lengths, _y_overlaps = split_tiles(
            height, edge, TILE_OVERLAP, 16
        )
        _x_starts, x_lengths, _x_overlaps = split_tiles(
            width, edge, TILE_OVERLAP, 16
        )
        processed_area = sum(y_lengths) * sum(x_lengths)
        candidate = (processed_area, edge)
        if best is None or candidate < best:
            best = candidate
    return best[1]


def _decode_memory_requirement(vae, latent_shape, tile_size, dtype):
    model = vae.first_stage_model
    height = latent_shape[-2] * model.vae_ratio
    width = latent_shape[-1] * model.vae_ratio
    tile_height = min(height, tile_size)
    tile_width = min(width, tile_size)
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
        sequence * _DECODER_BYTES_PER_FP16_TOKEN * dtype_scale
    )
    pixel_elements = (
        latent_shape[0]
        * model.decoder.out_channels
        * resident_frames
        * height
        * width
    )
    pixel_workspace = pixel_elements * (
        comfy.model_management.dtype_size(dtype) + 4
    )
    structural_estimate = int(
        (transformer_workspace + pixel_workspace) * _DECODE_SAFETY_FACTOR
    )
    return max(official_estimate, structural_estimate)


def _encode_memory_requirement(vae, pixel_shape, tile_size, dtype):
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


def _select_tile(vae, preset, height, width, memory_estimator):
    if preset == "auto":
        budget = _tile_memory_budget(vae)
        tile_size = resolve_tile_preset(
            preset,
            height,
            width,
            memory_limit=budget,
            memory_estimator=memory_estimator,
        )
        estimate = memory_estimator(tile_size)
        log = logging.warning if estimate > budget else logging.info
        log(
            "MiniMax H3 VAE auto tile selected %dpx: estimated %.0f MiB, available %.0f MiB%s",
            tile_size,
            estimate / 1024**2,
            budget / 1024**2,
            " (minimum tile exceeds the current budget)" if estimate > budget else "",
        )
        return tile_size, estimate
    tile_size = resolve_tile_preset(preset, height, width)
    return tile_size, memory_estimator(tile_size)


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


def _attention_forward(module, x, rotary_pos_emb, transformer_options):
    batch_size, seq_len, _ = x.shape
    qkv = module.to_qkv(x).view(batch_size, seq_len, -1, 3 * module.dim_head)
    query, key, value = torch.chunk(qkv, 3, dim=-1)

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
            del qkv, query, key, value
            return module.to_out(outcome.output.nan_to_num_(0.0))

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
    del qkv
    out = h3_vae.optimized_attention(
        query,
        key,
        value,
        module.heads,
        skip_reshape=True,
        transformer_options=transformer_options,
    )
    return module.to_out(out.nan_to_num_(0.0))


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


def _decoder_forward(decoder, x, transformer_options, block_session):
    batch, _, latent_t, latent_h, latent_w = x.shape
    h = decoder.x_embedder(x.flatten(2).transpose(1, 2))
    num_patches = h.shape[1]
    num_suffix = 1 + decoder.num_register_tokens
    h = torch.cat(
        (
            h,
            comfy.ops.cast_to_input(decoder.register_tokens, h).expand(batch, -1, -1),
            torch.zeros_like(h[:, 0:1, :]),
        ),
        dim=1,
    )

    img_ids = h3_vae.create_token_ids(
        (latent_t, latent_h, latent_w), x.device, x.dtype
    ).expand(batch, -1, -1)
    suffix_ids = torch.zeros(
        (batch, num_suffix, 3), device=x.device, dtype=img_ids.dtype
    )
    rotary_pos_emb = decoder.pos_embed(torch.cat((img_ids, suffix_ids), dim=1))

    blocks = list(decoder.transformer_blocks)
    for index, block in enumerate(blocks):
        if index > 0:
            block_session.before_stage(index)
        normed = comfy.rmsnorm.rms_norm(h, block.norm1.weight, block.norm1.eps)
        h = h.addcmul_(
            _attention_forward(block.attn, normed, rotary_pos_emb, transformer_options),
            comfy.ops.cast_to_input(block.scale1, h),
        )
        normed = comfy.rmsnorm.rms_norm(h, block.norm2.weight, block.norm2.eps)
        if _fused_swiglu_eligible(block.ff.w2):
            mlp = comfy.ops.linear_input_act(block.ff.w2, block.ff.w1(normed), "swiglu")
        else:
            mlp = block.ff(normed)
        h = h.addcmul_(mlp, comfy.ops.cast_to_input(block.scale2, h))
    output = decoder.proj_out(decoder.norm_out(h))[:, :num_patches, :]
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


def _decode_pixels(model, z, transformer_options, block_session):
    block_session.before_stage(0)
    z = model.post_quant_conv(z)
    return _decoder_forward(model.decoder, z, transformer_options, block_session)


def _tiled_decode(
    model,
    z,
    tile_size,
    tile_overlap,
    transformer_options,
    block_session,
):
    height = z.shape[-2] * model.vae_ratio
    width = z.shape[-1] * model.vae_ratio
    y_idx, y_len, y_overlap = split_tiles(
        height, tile_size, tile_overlap, model.vae_ratio
    )
    x_idx, x_len, x_overlap = split_tiles(
        width, tile_size, tile_overlap, model.vae_ratio
    )

    canvas = None
    row_tails = []
    out_y = 0
    for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
        zi, zl = i_pos // model.vae_ratio, i_len // model.vae_ratio
        new_tails = []
        left_tail = None
        out_x = 0
        for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
            zj, zw = j_pos // model.vae_ratio, j_len // model.vae_ratio
            tile = _decode_pixels(
                model,
                z[..., zi : zi + zl, zj : zj + zw],
                transformer_options,
                block_session,
            )
            if i < len(y_idx) - 1:
                new_tails.append(tile[..., -y_overlap[i] :, :].clone())
            next_left_tail = (
                tile[..., :, -x_overlap[j] :].clone() if j < len(x_idx) - 1 else None
            )
            if i > 0:
                tile = model.blend(row_tails[j], tile, y_overlap[i - 1], dim=-2)
            if j > 0:
                tile = model.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
            left_tail = next_left_tail
            if i < len(y_idx) - 1:
                tile = tile[..., : -y_overlap[i], :]
            if j < len(x_idx) - 1:
                tile = tile[..., :, : -x_overlap[j]]
            if canvas is None:
                canvas = torch.empty(
                    *tile.shape[:-2],
                    height,
                    width,
                    dtype=tile.dtype,
                    device=tile.device,
                )
            canvas[
                ..., out_y : out_y + tile.shape[-2], out_x : out_x + tile.shape[-1]
            ].copy_(tile)
            out_x += tile.shape[-1]
        row_tails = new_tails
        out_y += tile.shape[-2]
    return canvas


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


def _tiled_encode(model, x, tile_size, tile_overlap, module_session=None):
    height, width = x.shape[-2:]
    y_idx, y_len, y_overlap = split_tiles(
        height, tile_size, tile_overlap, model.vae_ratio
    )
    x_idx, x_len, x_overlap = split_tiles(
        width, tile_size, tile_overlap, model.vae_ratio
    )

    rows = []
    for i_pos, i_len in zip(y_idx, y_len):
        row = []
        for j_pos, j_len in zip(x_idx, x_len):
            tile = x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
            row.append(_encode_moments(model, tile, module_session))
        rows.append(row)

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
    tile_size,
    tile_overlap,
    transformer_options,
    block_session,
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
        clip_dec = _tiled_decode(
            model,
            clip_z,
            tile_size,
            tile_overlap,
            transformer_options,
            block_session,
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
    tile_preset="auto",
    attention="sdpa",
):
    model = require_h3_video_vae(vae)
    # H3 advertises FP16/FP32 to ComfyUI and defaults to FP16 on supported
    # NVIDIA GPUs, including Turing.  Follow the dtype used to load this VAE so
    # an explicit global --fp32-vae override cannot create mixed-dtype modules.
    compute_dtype = vae.vae_dtype
    tile_size, memory = _select_tile(
        vae,
        tile_preset,
        latent.shape[-2] * model.vae_ratio,
        latent.shape[-1] * model.vae_ratio,
        lambda edge: _decode_memory_requirement(
            vae, latent.shape, edge, compute_dtype
        ),
    )
    tile_overlap = TILE_OVERLAP
    comfy.model_management.load_models_gpu(
        [vae.patcher], memory_required=memory, force_full_load=vae.disable_offload
    )

    block_session = None
    try:
        transformer_options = _attention_options(attention, vae.device)
        z = latent.to(device=vae.device, dtype=compute_dtype)
        mean = model.latents_mean.view(1, -1, 1, 1, 1).to(z)
        std = model.latents_std.view(1, -1, 1, 1, 1).to(z)
        z = z * std + mean
        block_session = _RetainedWeights(_decoder_weight_stages(model), z.device, True)
        block_session.start()
        if z.shape[2] == 1:
            dec = _tiled_decode(
                model,
                z,
                tile_size,
                tile_overlap,
                transformer_options,
                block_session,
            )[:, :, -1:]
            dec = model._finalize_pixels(dec)
            return dec.to(vae.output_device).movedim(1, -1)
        dec = _decode_temporal(
            model,
            z,
            tile_size,
            tile_overlap,
            transformer_options,
            block_session,
        )
        return dec.movedim(1, -1)
    finally:
        if block_session is not None:
            block_session.finish()
        _clear_attention_caches(model.decoder)
        vae.patcher.partially_unload(vae.patcher.offload_device, 1e30)


def _encode_clip(model, clip, tile_size, tile_overlap, module_session=None):
    return _tiled_encode(
        model,
        model._normalize_pixels(clip),
        tile_size,
        tile_overlap,
        module_session,
    )


def _encode_temporal(
    model,
    x,
    device,
    compute_dtype,
    tile_size,
    tile_overlap,
    module_session,
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
            _encode_clip(model, clip, tile_size, tile_overlap, module_session)
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
    tile_preset="auto",
):
    model = require_h3_video_vae(vae)
    pixels = vae.vae_encode_crop_pixels(pixels).movedim(-1, 1)
    if pixels.ndim < 5:
        pixels = pixels.movedim(1, 0).unsqueeze(0)
    compute_dtype = vae.vae_dtype
    tile_size, memory = _select_tile(
        vae,
        tile_preset,
        pixels.shape[-2],
        pixels.shape[-1],
        lambda edge: _encode_memory_requirement(
            vae, pixels.shape, edge, compute_dtype
        ),
    )
    tile_overlap = TILE_OVERLAP
    comfy.model_management.load_models_gpu(
        [vae.patcher], memory_required=memory, force_full_load=vae.disable_offload
    )

    module_session = None
    try:
        module_session = _RetainedWeights(
            _encoder_weight_stages(model), vae.device, True
        )
        module_session.start()
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
            )
        mean = torch.chunk(moments.float(), 2, dim=1)[0]
        latent_mean = model.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        latent_std = model.latents_std.view(1, -1, 1, 1, 1).to(mean)
        return ((mean - latent_mean) / latent_std).to(
            device=vae.output_device, dtype=torch.float32
        )
    finally:
        if module_session is not None:
            module_session.finish()
        vae.patcher.partially_unload(vae.patcher.offload_device, 1e30)
