"""H3 VAE adapters: native ComfyUI lifecycle with scoped operator overrides.

Do not implement tiling, batching, transfers, prefetch, memory estimates, or OOM
retries here. Those belong to comfy.sd.VAE and the upstream H3 model.
"""

from __future__ import annotations

import math
from contextlib import ExitStack, contextmanager
from functools import partial

import torch
from tqdm.auto import tqdm

import comfy.model_management
import comfy.ops
import comfy.quant_ops
import comfy.rmsnorm
import comfy.utils
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
from ...log import get_logger
from ...quantization.dispatch import register_backend
from ...quantization.operator_scope import use_turing_operator_backend


LOG = get_logger("minimax.vae")


def require_h3_video_vae(vae):
    vae.throw_exception_if_invalid()
    model = vae.first_stage_model
    if not isinstance(model, h3_vae.MiniMaxH3VideoVAE):
        raise ValueError("This node requires a MiniMax H3 video VAE")
    return model


@contextmanager
def _vae_operator_scope():
    if register_backend():
        with use_turing_operator_backend():
            yield
    else:
        yield


def _spatial_tile_count(model, height, width):
    if not model.tiling:
        return 1
    y_starts, _, _ = model.split_tiles(int(height))
    x_starts, _, _ = model.split_tiles(int(width))
    return len(y_starts) * len(x_starts)


def _encode_tile_total(vae, model, pixels):
    """Return the exact native H3 forward count for a normal IMAGE batch."""
    try:
        if not torch.is_tensor(pixels) or pixels.ndim != 4 or int(pixels.shape[0]) < 1:
            return None
        cropped = vae.vae_encode_crop_pixels(pixels)
        frames, height, width = map(int, cropped.shape[:3])
        temporal_chunks = 1 if frames == 1 else math.ceil(frames / model.clip_length)
        return temporal_chunks * _spatial_tile_count(model, height, width)
    except Exception:
        # Progress accounting must never change whether the official VAE accepts
        # an input. Leave unusual inputs open-ended and let vae.encode validate.
        LOG.debug("Could not precompute H3 VAE encode tile total", exc_info=True)
        return None


def _decode_tile_total(model, latent):
    """Return the exact native H3 forward count for the usual B=1 latent."""
    try:
        if (
            not torch.is_tensor(latent)
            or latent.ndim != 5
            or int(latent.shape[0]) != 1
            or int(latent.shape[2]) < 1
        ):
            return None
        temporal_tokens = int(latent.shape[2])
        temporal_chunks = (
            1
            if temporal_tokens == 1
            else int(model._decode_temporal_chunks(temporal_tokens)[1])
        )
        height = int(latent.shape[-2]) * model.vae_ratio
        width = int(latent.shape[-1]) * model.vae_ratio
        return temporal_chunks * _spatial_tile_count(model, height, width)
    except Exception:
        LOG.debug("Could not precompute H3 VAE decode tile total", exc_info=True)
        return None


@contextmanager
def _tile_progress(module, description, total=None):
    """Observe native tile forwards without retaining tensors or synchronizing.

    The counter reports submitted forwards, not GPU completion. Normal H3
    single-video inputs have an exact planned total. If an OOM retry submits
    more work than planned, the display falls back to an open-ended counter.
    No CUDA events, side streams, worker threads, or ComfyUI progress hooks are
    added to the model's execution path.
    """
    with tqdm(
        total=total,
        desc=description,
        unit="tile",
        disable=not comfy.utils.PROGRESS_BAR_ENABLED,
    ) as terminal:
        submitted = 0

        def update(_module, _inputs, _output):
            nonlocal submitted
            submitted += 1
            if total is not None and submitted > total and terminal.total is not None:
                LOG.warning(
                    "%s exceeded planned total=%d; switching to open-ended progress after a retry",
                    description,
                    total,
                )
                terminal.total = None
                terminal.refresh()
            terminal.update(1)

        handle = module.register_forward_hook(update)
        try:
            yield
        finally:
            handle.remove()


def _norm_weight(module, name, reference):
    norm = getattr(module, name)
    if norm.weight is not None:
        return norm.weight
    # Non-affine norms need a unit weight for the fused QK transform contract.
    # Keep it call-local; no device tensors are cached on the shared VAE.
    return torch.ones(
        module.dim_head, device=reference.device, dtype=reference.dtype
    )


def _projected_attention(module, query, key, value, rotary_pos_emb, options):
    executor = options.get(ATTENTION_EXECUTOR_KEY)
    if callable(executor):
        rot_dim = int(rotary_pos_emb.shape[-3] * 2) if rotary_pos_emb is not None else 0
        transform = QKTransformSpec(
            query_norm=RMSNormSpec(
                _norm_weight(module, "norm_q", query), float(module.norm_q.eps), "head"
            ),
            key_norm=RMSNormSpec(
                _norm_weight(module, "norm_k", key), float(module.norm_k.eps), "head"
            ),
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
            transformer_options=options,
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

    out = h3_vae.optimized_attention(
        AttentionTensorContainer(query.transpose(1, 2)),
        AttentionTensorContainer(key.transpose(1, 2)),
        AttentionTensorContainer(value.transpose(1, 2)),
        module.heads,
        skip_reshape=True,
        transformer_options=options,
    )
    return out.nan_to_num_(0.0)


def _attention_forward(module, x, rotary_pos_emb=None, *, options):
    batch_size, seq_len, _ = x.shape
    qkv = module.to_qkv(x).view(batch_size, seq_len, -1, 3 * module.dim_head)
    query, key, value = torch.chunk(qkv, 3, dim=-1)
    out = _projected_attention(
        module, query, key, value, rotary_pos_emb, options
    )
    return module.to_out(out)


def _attention_options(attention, device):
    override = make_attention_override(attention, device)
    options = {"optimized_attention_override": override}
    executor = getattr(override, "prepared_attention_executor", None)
    if callable(executor):
        options[ATTENTION_EXECUTOR_KEY] = executor
    return options


def _fused_swiglu_eligible(linear):
    weight = linear.weight
    return bool(
        not comfy.model_management.in_training
        and isinstance(weight, comfy.ops.QuantizedTensor)
        and weight._layout_cls == "TensorWiseINT8Layout"
        and not getattr(weight._params, "transposed", False)
    )


def _feed_forward(module, value, *, original_forward):
    if _fused_swiglu_eligible(module.w2):
        # Both projections still use ComfyUI's own cast/uncast weight lifecycle.
        output = comfy.ops.linear_input_act(module.w2, module.w1(value), "swiglu")
    else:
        output = original_forward(value)
    if output.shape != value.shape:
        raise RuntimeError(
            f"H3 VAE feed-forward returned {tuple(output.shape)} for input "
            f"{tuple(value.shape)}"
        )
    return output


@contextmanager
def _temporary_forward(module, forward):
    # Restore the exact instance state, including any pre-existing wrapper.
    # Never patch a class or a process-global attention function.
    had_override = "forward" in module.__dict__
    previous = module.__dict__.get("forward")
    module.forward = forward
    try:
        yield
    finally:
        if had_override:
            module.forward = previous
        else:
            delattr(module, "forward")


@contextmanager
def _decoder_overrides(decoder, attention, device):
    options = _attention_options(attention, device)
    with ExitStack() as stack:
        for block in decoder.transformer_blocks:
            stack.enter_context(_temporary_forward(
                block.attn,
                partial(_attention_forward, block.attn, options=options),
            ))
            stack.enter_context(_temporary_forward(
                block.ff,
                partial(_feed_forward, block.ff, original_forward=block.ff.forward),
            ))
        yield


def decode_video(vae, latent, attention="sdpa"):
    model = require_h3_video_vae(vae)
    tile_total = _decode_tile_total(model, latent)
    LOG.info(
        "H3 VAE decode: native ComfyUI lifecycle, attention=%s; "
        "tile progress counts submitted forwards, planned_total=%s",
        attention,
        tile_total if tile_total is not None else "dynamic",
    )
    with (
        _vae_operator_scope(),
        _decoder_overrides(model.decoder, attention, vae.device),
        _tile_progress(
            model.decoder,
            "H3 VAE Decode Tiles (submitted)",
            total=tile_total,
        ),
    ):
        # Includes official dtype/output handling, model loading, tiling and OOM
        # fallback. In particular, do not call first_stage_model.decode directly.
        return vae.decode(latent)


def encode_video(vae, pixels):
    """Compatibility facade for callers importing encode from this module."""
    from .video_vae_encode import encode_video as implementation

    return implementation(vae, pixels)
