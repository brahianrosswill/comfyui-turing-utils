from __future__ import annotations

import logging
import types

import comfy.context_windows
import comfy.conds
import comfy.patcher_extension
import comfy.utils
import node_helpers
import torch
import torch.nn.functional as F
from comfy_api.latest import io

try:
    from .reference_nodes import (
        ImageReferenceSet,
        ImageReferences,
        SpatialOptions,
        VideoReferenceSet,
        VideoReferences,
        _spatial_transform,
        _validate_image,
    )
except ImportError:
    from reference_nodes import (
        ImageReferenceSet,
        ImageReferences,
        SpatialOptions,
        VideoReferenceSet,
        VideoReferences,
        _spatial_transform,
        _validate_image,
    )


LOG = logging.getLogger("comfyui-turing-utils")
_BERNINI_ROPE_WRAPPER_KEY = "turing_utils_bernini_context_rope"
_ABSOLUTE_INDEX_KEY = "turing_utils_bernini_absolute_latent_indices"
_CONTEXT_ROLES_KEY = "turing_utils_bernini_context_roles"


def _context_roles(value, count: int) -> tuple[str, ...] | None:
    value = getattr(value, "cond", value)
    if not isinstance(value, (list, tuple)) or len(value) != count:
        return None
    roles = tuple(str(role) for role in value)
    if any(role not in ("aligned", "global") for role in roles):
        return None
    return roles


def _slice_context_latents_for_estimate(value, full_length: int, estimate_length: int, dim: int, roles=None):
    if not isinstance(value, (list, tuple)):
        return value
    roles = _context_roles(roles, len(value))
    changed = False
    sliced = []
    for index, latent in enumerate(value):
        aligned = roles[index] == "aligned" if roles is not None else (
            torch.is_tensor(latent) and latent.ndim > dim and latent.shape[dim] == full_length
        )
        if aligned and torch.is_tensor(latent) and latent.ndim > dim:
            latent = latent.narrow(dim, 0, min(estimate_length, int(latent.shape[dim])))
            changed = True
        sliced.append(latent)
    if not changed:
        return value
    return tuple(sliced) if isinstance(value, tuple) else sliced


def _estimate_conditioning(conds, full_length: int, estimate_length: int, dim: int):
    changed = False
    estimated = {}
    for group, entries in conds.items():
        new_entries = []
        for entry in entries:
            new_entry = entry
            raw = entry.get("context_latents")
            raw_roles = entry.get(_CONTEXT_ROLES_KEY)
            sliced = _slice_context_latents_for_estimate(raw, full_length, estimate_length, dim, raw_roles)
            if sliced is not raw:
                new_entry = dict(entry)
                new_entry["context_latents"] = sliced
                changed = True

            model_conds = entry.get("model_conds")
            if isinstance(model_conds, dict) and "context_latents" in model_conds:
                cond = model_conds["context_latents"]
                value = getattr(cond, "cond", None)
                roles = model_conds.get(_CONTEXT_ROLES_KEY)
                sliced = _slice_context_latents_for_estimate(value, full_length, estimate_length, dim, roles)
                if sliced is not value:
                    if new_entry is entry:
                        new_entry = dict(entry)
                    new_model_conds = dict(model_conds)
                    new_model_conds["context_latents"] = cond._copy_with(sliced)
                    new_entry["model_conds"] = new_model_conds
                    changed = True
            new_entries.append(new_entry)
        estimated[group] = new_entries
    return estimated if changed else conds


def _bernini_prepare_sampling_wrapper(executor, model, noise_shape, conds, *args, **kwargs):
    model_options = kwargs.get("model_options")
    if model_options is None:
        raise RuntimeError("model_options not found in Bernini prepare-sampling wrapper")
    handler = model_options.get("context_handler")
    if handler is None or handler.dim >= len(noise_shape):
        return executor(model, noise_shape, conds, *args, **kwargs)

    # Match ComfyUI's conservative behavior for packed latents: latent_shapes
    # is not attached yet, so a flat per-window size cannot be derived safely.
    if len(noise_shape) == 3 and noise_shape[1] == 1:
        return executor(model, noise_shape, conds, *args, **kwargs)

    full_length = int(noise_shape[handler.dim])
    if full_length <= handler.context_length:
        return executor(model, noise_shape, conds, *args, **kwargs)

    # Later causal windows prepend one source frame. Budget for the largest
    # actual invocation, while leaving the real conditioning untouched.
    anchor = 1 if getattr(handler, "causal_window_fix", False) else 0
    estimate_length = min(full_length, int(handler.context_length) + anchor)
    estimated_shape = list(noise_shape)
    estimated_shape[handler.dim] = estimate_length
    estimated_conds = _estimate_conditioning(
        conds, full_length, estimate_length, handler.dim
    )
    result = executor(model, estimated_shape, estimated_conds, *args, **kwargs)
    if isinstance(result, tuple) and len(result) >= 2:
        return (result[0], conds, *result[2:])
    return result


def _validate_context_window_frames(context_length: int, context_overlap: int) -> tuple[int, int]:
    context_length = int(context_length)
    context_overlap = int(context_overlap)
    if context_length < 1:
        raise ValueError(f"context_length must be at least 1 real frame; got {context_length}.")
    if context_overlap < 0:
        raise ValueError(f"context_overlap must be non-negative; got {context_overlap}.")

    latent_context_length = max(((context_length - 1) // 4) + 1, 1)
    latent_context_overlap = max(context_overlap // 4, 0)
    if latent_context_overlap >= latent_context_length:
        raise ValueError(
            "context_overlap must be shorter than context_length after Wan latent conversion; "
            f"got overlap={context_overlap} -> {latent_context_overlap}, "
            f"length={context_length} -> {latent_context_length}."
        )
    return latent_context_length, latent_context_overlap


def _with_transformer_options(args, kwargs, transformer_options):
    if len(args) >= 6 and isinstance(args[5], dict):
        new_args = list(args)
        new_args[5] = transformer_options
        return tuple(new_args), kwargs
    new_kwargs = dict(kwargs)
    new_kwargs["transformer_options"] = transformer_options
    return args, new_kwargs


def _install_bernini_absolute_rope_patch() -> None:
    from comfy.ldm.wan import model as wan_model

    if hasattr(wan_model.WanModel, "_turing_utils_original_forward"):
        return

    wan_model.WanModel._turing_utils_original_forward = wan_model.WanModel._forward
    wan_model.WanModel._forward = _wan_forward_with_optional_absolute_indices


def _wan_forward_with_optional_absolute_indices(
    self,
    x,
    timestep,
    context,
    clip_fea=None,
    time_dim_concat=None,
    transformer_options={},
    **kwargs,
):
    import comfy.ldm.common_dit

    target_indices = transformer_options.get(_ABSOLUTE_INDEX_KEY, None)
    if target_indices is None:
        return self._turing_utils_original_forward(
            x,
            timestep,
            context,
            clip_fea=clip_fea,
            time_dim_concat=time_dim_concat,
            transformer_options=transformer_options,
            **kwargs,
        )

    if time_dim_concat is not None or (self.ref_conv is not None and "reference_latent" in kwargs):
        LOG.warning(
            "Bernini absolute context RoPE indices were ignored for a Wan path with "
            "time_dim_concat/reference_latent; falling back to ComfyUI RoPE."
        )
        return self._turing_utils_original_forward(
            x,
            timestep,
            context,
            clip_fea=clip_fea,
            time_dim_concat=time_dim_concat,
            transformer_options=transformer_options,
            **kwargs,
        )

    bs, c, t, h, w = x.shape
    del bs, c
    x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
    freqs = _rope_encode_with_absolute_indices(
        self,
        target_indices,
        t,
        h,
        w,
        device=x.device,
        dtype=x.dtype,
        transformer_options=transformer_options,
        source_id=0,
    )

    context_latents = kwargs.get("context_latents", None)
    if context_latents is not None:
        roles = _context_roles(kwargs.get(_CONTEXT_ROLES_KEY), len(context_latents))
        context_latents = [comfy.ldm.common_dit.pad_to_patch_size(lat, self.patch_size) for lat in context_latents]
        for i, lat in enumerate(context_latents):
            if roles is None:
                context_indices = target_indices if lat.shape[-3] == len(target_indices) else None
            else:
                context_indices = target_indices if roles[i] == "aligned" else None
            freqs = torch.cat(
                [
                    freqs,
                    _rope_encode_with_absolute_indices(
                        self,
                        context_indices,
                        lat.shape[-3],
                        lat.shape[-2],
                        lat.shape[-1],
                        device=x.device,
                        dtype=x.dtype,
                        transformer_options=transformer_options,
                        source_id=i + 1,
                    ),
                ],
                dim=1,
            )
        kwargs = {**kwargs, "context_latents": context_latents}

    return self.forward_orig(
        x,
        timestep,
        context,
        clip_fea=clip_fea,
        freqs=freqs,
        transformer_options=transformer_options,
        **kwargs,
    )[:, :, :t, :h, :w]


def _rope_encode_with_absolute_indices(
    model,
    indices,
    t,
    h,
    w,
    *,
    device,
    dtype,
    transformer_options,
    source_id: int,
):
    if indices is None:
        return model.rope_encode(
            t,
            h,
            w,
            device=device,
            dtype=dtype,
            transformer_options=transformer_options,
            source_id=source_id,
        )

    patch_size = model.patch_size
    steps_t = ((t + (patch_size[0] // 2)) // patch_size[0])
    steps_h = ((h + (patch_size[1] // 2)) // patch_size[1])
    steps_w = ((w + (patch_size[2] // 2)) // patch_size[2])
    temporal = _normalize_temporal_indices(indices, steps_t).to(device=device, dtype=dtype)

    h_len = steps_h
    w_len = steps_w
    h_start = 0.0
    w_start = 0.0
    rope_options = transformer_options.get("rope_options", None)
    if rope_options is not None:
        temporal = temporal * float(rope_options.get("scale_t", 1.0)) + float(rope_options.get("shift_t", 0.0))
        h_len = (h_len - 1.0) * float(rope_options.get("scale_y", 1.0)) + 1.0
        w_len = (w_len - 1.0) * float(rope_options.get("scale_x", 1.0)) + 1.0
        h_start += float(rope_options.get("shift_y", 0.0))
        w_start += float(rope_options.get("shift_x", 0.0))

    img_ids = torch.zeros((steps_t, steps_h, steps_w, 3), device=device, dtype=dtype)
    img_ids[:, :, :, 0] = temporal.reshape(-1, 1, 1)
    img_ids[:, :, :, 1] = torch.linspace(h_start, h_len - 1 + h_start, steps_h, device=device, dtype=dtype).reshape(1, -1, 1)
    img_ids[:, :, :, 2] = torch.linspace(w_start, w_len - 1 + w_start, steps_w, device=device, dtype=dtype).reshape(1, 1, -1)
    img_ids = img_ids.reshape(1, steps_t * steps_h * steps_w, img_ids.shape[-1])
    freqs = model.rope_embedder(img_ids).movedim(1, 2)

    if source_id:
        from comfy.ldm.flux.math import rope

        head_dim = model.dim // model.num_heads
        pos = torch.tensor([[float(source_id)]], device=freqs.device, dtype=torch.float32)
        id_rot = rope(pos, head_dim, model.rope_embedder.theta).reshape(
            1,
            1,
            1,
            head_dim // 2,
            2,
            2,
        ).to(freqs.dtype)
        freqs = torch.einsum("...ij,...jk->...ik", freqs, id_rot)
    return freqs


def _normalize_temporal_indices(indices, steps_t: int) -> torch.Tensor:
    if isinstance(indices, torch.Tensor):
        values = indices.detach().flatten().to(dtype=torch.float32, device="cpu")
    else:
        values = torch.tensor([float(v) for v in indices], dtype=torch.float32)
    if values.numel() == 0:
        values = torch.arange(steps_t, dtype=torch.float32)
    if values.numel() < steps_t:
        last = values[-1]
        pad = torch.arange(1, steps_t - values.numel() + 1, dtype=torch.float32) + last
        values = torch.cat([values, pad], dim=0)
    return values[:steps_t]


def _bernini_context_rope_wrapper(executor, *args, **kwargs):
    transformer_options = None
    if len(args) >= 6 and isinstance(args[5], dict):
        transformer_options = args[5]
    elif isinstance(kwargs.get("transformer_options"), dict):
        transformer_options = kwargs["transformer_options"]

    if transformer_options is None:
        return executor(*args, **kwargs)

    window = transformer_options.get("context_window")
    if window is None or not getattr(window, "index_list", None):
        return executor(*args, **kwargs)

    if getattr(window, "turing_utils_use_absolute_indices", False):
        indices = list(window.index_list)
        anchor_idx = getattr(window, "causal_anchor_index", None)
        if anchor_idx is not None and anchor_idx >= 0:
            indices = [int(anchor_idx)] + indices
        new_transformer_options = dict(transformer_options)
        new_transformer_options[_ABSOLUTE_INDEX_KEY] = tuple(int(index) for index in indices)
        args, kwargs = _with_transformer_options(args, kwargs, new_transformer_options)
        return executor(*args, **kwargs)
    return executor(*args, **kwargs)


class BerniniScheduledContextHandler(comfy.context_windows.IndexListContextHandler):
    def __init__(self, *args, turing_utils_absolute_positions: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.turing_utils_absolute_positions = bool(turing_utils_absolute_positions)

    def get_context_windows(self, model, x_in: torch.Tensor, model_options: dict[str]):
        windows = super().get_context_windows(model, x_in, model_options)
        for window in windows:
            window.turing_utils_use_absolute_indices = self.turing_utils_absolute_positions
        return windows


def _resize_bernini_context(cond_key, cond_value, window, x_in, device, new_cond_item):
    if cond_key != "context_latents" or not isinstance(getattr(cond_value, "cond", None), list):
        return None
    roles = _context_roles(new_cond_item.get(_CONTEXT_ROLES_KEY), len(cond_value.cond))
    if roles is None:
        return None

    resized = []
    for role, latent in zip(roles, cond_value.cond):
        if role == "aligned" and latent.ndim > window.dim and latent.shape[window.dim] > 1:
            resized.append(window.get_tensor(latent, device, dim=window.dim))
        else:
            resized.append(latent.to(device))
    return cond_value._copy_with(resized)


def _make_extra_conds_with_bernini_roles(base_model):
    original = base_model.extra_conds

    def extra_conds(self, **kwargs):
        out = original(**kwargs)
        roles = kwargs.get(_CONTEXT_ROLES_KEY)
        if roles is not None:
            out[_CONTEXT_ROLES_KEY] = comfy.conds.CONDConstant(tuple(roles))
        return out

    return types.MethodType(extra_conds, base_model)


def _align_source_video_and_mask(source_video, mask, length: int):
    source_video = _validate_image(source_video, "source_video")
    source_length = int(source_video.shape[0])
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3 or mask.shape[-2:] != source_video.shape[1:3]:
            raise ValueError(
                "MASK must match source_video spatial dimensions; expected "
                f"(*, {source_video.shape[1]}, {source_video.shape[2]}), got {tuple(mask.shape)}"
            )
        if mask.shape[0] == 1 and source_length > 1:
            mask = mask.repeat(source_length, 1, 1)
        elif mask.shape[0] != source_length:
            raise ValueError(
                f"MASK must contain one frame or match source_video ({source_length} frames); got {mask.shape[0]}"
            )

    if source_length > length:
        source_video = source_video[:length]
        if mask is not None:
            mask = mask[:length]
    elif source_length < length:
        pad = length - source_length
        source_video = torch.cat((source_video, source_video[-1:].repeat(pad, 1, 1, 1)), dim=0)
        if mask is not None:
            mask = torch.cat((mask, mask[-1:].repeat(pad, 1, 1)), dim=0)
    return source_video, mask


def _upper_bound_latent_mask(mask: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    target = tuple(int(size) for size in latent.shape[-3:])
    mask = mask.float().clamp(0.0, 1.0).reshape(1, 1, *mask.shape)
    mask = F.adaptive_max_pool3d(mask, target)
    return F.max_pool3d(mask, kernel_size=3, stride=1, padding=1).clamp_(0.0, 1.0)


class BerniniInpaintCondition(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsBerniniInpaintCondition",
            display_name="Bernini Inpaint Condition",
            category="Turing Utils/conditioning",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Image.Input("source_video"),
                io.Int.Input("width", default=832, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("length", default=81, min=1, max=8192, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Boolean.Input("source_as_context", default=False, tooltip="Also append the aligned source video as Bernini context tokens."),
                io.Mask.Input("mask", optional=True, tooltip="White is repainted and black is preserved. Omit for global repaint."),
                ImageReferences.Input("image_references", optional=True),
                VideoReferences.Input("video_references", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, source_video, width, height, length, batch_size,
                source_as_context=False, mask=None, image_references=None, video_references=None):
        length = int(length)
        if (length - 1) % 4 != 0:
            raise ValueError(f"Bernini length must be 4*n+1 real frames; got {length}")

        source_video, mask = _align_source_video_and_mask(source_video, mask, length)
        source_options = SpatialOptions(
            width=int(width),
            height=int(height),
            upscale_method="area",
            keep_proportion="crop",
            crop_position="center",
            divisible_by=1,
        )
        if mask is None:
            source_video = _spatial_transform(source_video, source_options)
        else:
            source_video, mask = _spatial_transform(source_video, source_options, mask)

        source_latent = vae.encode(source_video)
        source_latent = comfy.utils.repeat_to_batch_size(source_latent, int(batch_size))
        latent = {"samples": source_latent}
        if mask is not None:
            latent["noise_mask"] = _upper_bound_latent_mask(mask, source_latent)

        context = []
        roles = []
        if source_as_context:
            context.append(source_latent)
            roles.append("aligned")

        if isinstance(video_references, VideoReferenceSet):
            for video in video_references.materialize():
                context.append(vae.encode(video))
                roles.append("global")
        if isinstance(image_references, ImageReferenceSet):
            for image in image_references.materialize():
                context.append(vae.encode(image))
                roles.append("global")

        if context:
            values = {"context_latents": context, _CONTEXT_ROLES_KEY: tuple(roles)}
            positive = node_helpers.conditioning_set_values(positive, values)
            negative = node_helpers.conditioning_set_values(negative, values)
        return io.NodeOutput(positive, negative, latent)


class BerniniContextWindowsCore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "context_length": (
                    "INT",
                    {
                        "default": 81,
                        "min": 1,
                        "max": 16385,
                        "step": 4,
                        "tooltip": "The length of the context window in real frames. Must be 4*n + 1.",
                    },
                ),
                "context_overlap": (
                    "INT",
                    {
                        "default": 28,
                        "min": 0,
                        "max": 16384,
                        "step": 4,
                        "tooltip": "The overlap of the context window in real frames.",
                    },
                ),
                "position_mode": (
                    ["absolute", "relative"],
                    {
                        "default": "absolute",
                        "tooltip": "Absolute keeps global latent-frame RoPE positions; relative matches official ComfyUI window-local positions.",
                    },
                ),
                "context_schedule": (
                    [
                        comfy.context_windows.ContextSchedules.STATIC_STANDARD,
                        comfy.context_windows.ContextSchedules.UNIFORM_STANDARD,
                        comfy.context_windows.ContextSchedules.UNIFORM_LOOPED,
                        comfy.context_windows.ContextSchedules.BATCHED,
                    ],
                    {
                        "default": comfy.context_windows.ContextSchedules.UNIFORM_STANDARD,
                        "tooltip": "Step-dependent scheduling algorithm for context windows.",
                    },
                ),
                "context_stride": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 32,
                        "advanced": True,
                        "tooltip": "The stride of the context window; only applicable to uniform schedules.",
                    },
                ),
                "closed_loop": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "advanced": True,
                        "tooltip": "Whether to close the context window loop; only applicable to looped schedules.",
                    },
                ),
                "fuse_method": (
                    comfy.context_windows.ContextFuseMethods.LIST_STATIC,
                    {
                        "default": comfy.context_windows.ContextFuseMethods.PYRAMID,
                        "tooltip": "The method to use to fuse the context windows.",
                    },
                ),
                "freenoise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "advanced": True,
                        "tooltip": "Whether to apply FreeNoise noise shuffling, improves window blending.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Turing Utils/patches"
    TITLE = "Bernini Context Windows"

    def apply(
        self,
        model,
        context_length: int,
        context_overlap: int,
        context_schedule: str,
        position_mode: str = "absolute",
        context_stride: int = 1,
        closed_loop: bool = False,
        fuse_method: str = comfy.context_windows.ContextFuseMethods.PYRAMID,
        freenoise: bool = True,
    ):
        if position_mode not in ("absolute", "relative"):
            raise ValueError(f"position_mode must be absolute or relative; got {position_mode}")
        latent_context_length, latent_context_overlap = _validate_context_window_frames(
            context_length,
            context_overlap,
        )
        context_handler = BerniniScheduledContextHandler(
            context_schedule=comfy.context_windows.get_matching_context_schedule(context_schedule),
            fuse_method=comfy.context_windows.get_matching_fuse_method(fuse_method),
            context_length=latent_context_length,
            context_overlap=latent_context_overlap,
            context_stride=max(int(context_stride), 1),
            closed_loop=bool(closed_loop),
            dim=2,
            freenoise=bool(freenoise),
            cond_retain_index_list="",
            split_conds_to_windows=False,
            latent_retain_index_list="",
            causal_window_fix=True,
            turing_utils_absolute_positions=position_mode == "absolute",
        )
        comfy.patcher_extension.add_callback(
            comfy.context_windows.IndexListCallbacks.RESIZE_COND_ITEM,
            _resize_bernini_context,
            context_handler.callbacks,
        )

        patched = model.clone()
        patched.model_options["context_handler"] = context_handler
        patched.model_options.setdefault("transformer_options", {})
        base_model = getattr(patched, "model", None)
        if base_model is not None and callable(getattr(base_model, "extra_conds", None)) and hasattr(patched, "add_object_patch"):
            patched.add_object_patch("extra_conds", _make_extra_conds_with_bernini_roles(base_model))

        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
            "ContextWindows_prepare_sampling",
        )
        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
            "ContextWindows_sampler_sample",
        )
        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            _BERNINI_ROPE_WRAPPER_KEY,
        )

        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
            "ContextWindows_prepare_sampling",
            _bernini_prepare_sampling_wrapper,
        )
        if freenoise:
            comfy.context_windows.create_sampler_sample_wrapper(patched)
        if position_mode == "absolute":
            _install_bernini_absolute_rope_patch()
            patched.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                _BERNINI_ROPE_WRAPPER_KEY,
                _bernini_context_rope_wrapper,
            )

        LOG.info(
            "Bernini context windows enabled: schedule=%s, length=%s -> %s latent frames, "
            "overlap=%s -> %s latent frames, stride=%s, closed_loop=%s, "
            "position=%s, causal_window_fix=True, fuse=%s",
            context_schedule,
            context_length,
            latent_context_length,
            context_overlap,
            latent_context_overlap,
            context_stride,
            closed_loop,
            position_mode,
            fuse_method,
        )
        return (patched,)
