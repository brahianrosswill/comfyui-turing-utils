"""Wan/Bernini integration for generic Turing kernels and memory planning."""

from __future__ import annotations

import logging
import math
import types
from collections import Counter

import torch

try:
    from .turing_fusions import convrot_weight_kind
    from .turing_ops import is_supported_turing_device, turing_int8_workspace_bytes
except ImportError:
    from turing_fusions import convrot_weight_kind
    from turing_ops import is_supported_turing_device, turing_int8_workspace_bytes


LOG = logging.getLogger("comfyui-turing-utils")
_CONTEXT_SHAPE_KEY = "context_latents"
_MEMORY_CONTEXT_ATTR = "_turing_utils_wan_memory_context"
_OUTER_SAMPLE_WRAPPER_KEY = "turing_utils_wan_memory_context"


def _context_latents_from_kwargs(kwargs):
    context_latents = kwargs.get(_CONTEXT_SHAPE_KEY)
    if context_latents is not None:
        return context_latents
    model_conds = kwargs.get("model_conds")
    if isinstance(model_conds, dict):
        cond = model_conds.get(_CONTEXT_SHAPE_KEY)
        return getattr(cond, "cond", None)
    return None


def _context_latents_shape(
    context_latents,
    patch_size=(1, 2, 2),
    estimate_batch_size: int | None = None,
) -> list[int] | None:
    if not isinstance(context_latents, (list, tuple)) or not context_latents:
        return None
    tensors = [latent for latent in context_latents if torch.is_tensor(latent) and latent.ndim >= 3]
    if not tensors:
        return None
    channels = int(tensors[0].shape[1])
    if channels <= 0 or any(int(latent.shape[1]) != channels for latent in tensors):
        return None
    patch_size = tuple(int(value) for value in patch_size)
    patch_volume = math.prod(patch_size)
    padded_per_sample = []
    for latent in tensors:
        spatial = tuple(int(value) for value in latent.shape[2:])
        if len(spatial) == len(patch_size):
            padded = math.prod(
                math.ceil(size / patch) * patch
                for size, patch in zip(spatial, patch_size)
            )
        else:
            padded = math.ceil(math.prod(spatial) / patch_volume) * patch_volume
        padded_per_sample.append(padded)

    if estimate_batch_size is not None:
        batch = max(int(estimate_batch_size), 1)
        return [batch, channels, sum(padded_per_sample)]

    # Without the sampler batch hint, preserve the total volume already present
    # in each tensor. CONDList.size() uses the same flattened representation.
    total = sum(
        max(int(latent.shape[0]), 1) * padded
        for latent, padded in zip(tensors, padded_per_sample)
    )
    return [1, channels, total]


def _shape_token_rows(shape, patch_size) -> int:
    if len(shape) < 3:
        return 0
    batch = int(shape[0])
    spatial = [int(value) for value in shape[2:]]
    if len(spatial) == len(patch_size):
        tokens = math.prod(
            math.ceil(size / patch) for size, patch in zip(spatial, patch_size)
        )
    else:
        tokens = math.ceil(math.prod(spatial) / math.prod(patch_size))
    return batch * tokens


def _make_extra_conds_shapes(base_model, patch_size):
    original = base_model.extra_conds_shapes

    def extra_conds_shapes(self, **kwargs):
        out = dict(original(**kwargs))
        context = getattr(self, _MEMORY_CONTEXT_ATTR, None)
        estimate_batch_size = (
            context.get("batch_size") if isinstance(context, dict) else None
        )
        shape = _context_latents_shape(
            _context_latents_from_kwargs(kwargs),
            patch_size,
            estimate_batch_size=estimate_batch_size,
        )
        if shape is not None:
            out[_CONTEXT_SHAPE_KEY] = shape
        return out

    return types.MethodType(extra_conds_shapes, base_model)


def _make_extra_conds(base_model, patch_size):
    original = base_model.extra_conds

    class PaddedContextLatents:
        def __init__(self, cond):
            self.cond = cond

        def _copy_with(self, cond):
            return PaddedContextLatents(cond)

        def process_cond(self, batch_size, **kwargs):
            import comfy.utils

            out = [
                comfy.utils.repeat_to_batch_size(latent, batch_size)
                for latent in self.cond
            ]
            return self._copy_with(out)

        def can_concat(self, other):
            return (
                isinstance(other, PaddedContextLatents)
                and len(self.cond) == len(other.cond)
                and all(
                    left.shape == right.shape
                    for left, right in zip(self.cond, other.cond)
                )
            )

        def concat(self, others):
            out = []
            for index in range(len(self.cond)):
                out.append(
                    torch.cat(
                        [self.cond[index]]
                        + [other.cond[index] for other in others]
                    )
                )
            return out

        def size(self):
            return _context_latents_shape(self.cond, patch_size)

    def extra_conds(self, **kwargs):
        out = original(**kwargs)
        context = out.get(_CONTEXT_SHAPE_KEY)
        values = getattr(context, "cond", None)
        if isinstance(values, list):
            out[_CONTEXT_SHAPE_KEY] = PaddedContextLatents(values)
        return out

    return types.MethodType(extra_conds, base_model)


def _make_memory_required(base_model, patch_size, w8_output_channels: tuple[int, ...]):
    original = base_model.memory_required

    def memory_required(self, input_shape, cond_shapes={}):
        required = original(input_shape, cond_shapes=cond_shapes)
        if not w8_output_channels:
            return required

        rows = _shape_token_rows(input_shape, patch_size)
        for shape in cond_shapes.get(_CONTEXT_SHAPE_KEY, ()):
            rows += _shape_token_rows(shape, patch_size)
        workspace = max(
            turing_int8_workspace_bytes(rows, output_channels)
            for output_channels in w8_output_channels
        )
        return required + workspace

    return types.MethodType(memory_required, base_model)


def _make_outer_sample_wrapper(base_model):
    def outer_sample_wrapper(executor, *args, **kwargs):
        noise = args[0] if args else kwargs.get("noise")
        batch_size = int(noise.shape[0]) if torch.is_tensor(noise) else 1
        previous = getattr(base_model, _MEMORY_CONTEXT_ATTR, None)
        setattr(base_model, _MEMORY_CONTEXT_ATTR, {"batch_size": batch_size})
        try:
            return executor(*args, **kwargs)
        finally:
            if previous is None:
                try:
                    delattr(base_model, _MEMORY_CONTEXT_ATTR)
                except AttributeError:
                    pass
            else:
                setattr(base_model, _MEMORY_CONTEXT_ATTR, previous)

    return outer_sample_wrapper


def _quantized_wan_summary(diffusion_model) -> tuple[Counter, tuple[int, ...]]:
    formats = Counter()
    w8_outputs = set()
    for module in diffusion_model.modules():
        weight = getattr(module, "weight", None)
        kind = convrot_weight_kind(weight)
        if kind is None:
            continue
        formats[kind] += 1
        if kind == "w8a8" and getattr(weight, "ndim", 0) == 2:
            w8_outputs.add(int(weight.shape[0]))
    return formats, tuple(sorted(w8_outputs))


def apply_wan_adapter(model, device: torch.device) -> int:
    """Install Wan-specific planning without imposing input-size restrictions."""
    if not is_supported_turing_device(device):
        return 0

    try:
        from comfy.ldm.wan.model import WanModel
    except ImportError:
        return 0

    base_model = getattr(model, "model", model)
    diffusion_model = getattr(base_model, "diffusion_model", None)
    if not isinstance(diffusion_model, WanModel):
        return 0
    if getattr(base_model, "_turing_utils_wan_adapter", False):
        return 0

    formats, w8_output_channels = _quantized_wan_summary(diffusion_model)
    if not formats:
        return 0

    patch_size = tuple(int(value) for value in diffusion_model.patch_size)
    base_model.extra_conds = _make_extra_conds(base_model, patch_size)
    base_model.extra_conds_shapes = _make_extra_conds_shapes(base_model, patch_size)
    factors = tuple(getattr(base_model, "memory_usage_factor_conds", ()))
    if _CONTEXT_SHAPE_KEY not in factors:
        base_model.memory_usage_factor_conds = (*factors, _CONTEXT_SHAPE_KEY)
    base_model.memory_required = _make_memory_required(
        base_model, patch_size, w8_output_channels
    )
    base_model._turing_utils_wan_adapter = True

    if hasattr(model, "add_wrapper_with_key"):
        import comfy.patcher_extension

        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            _OUTER_SAMPLE_WRAPPER_KEY,
            _make_outer_sample_wrapper(base_model),
        )

    LOG.info(
        "Enabled Wan Turing adapter: formats=[%s], context-aware VRAM planning, "
        "w8_outputs=[%s]",
        ",".join(f"{kind}:{count}" for kind, count in sorted(formats.items())),
        ",".join(map(str, w8_output_channels)) or "none",
    )
    return sum(formats.values())
