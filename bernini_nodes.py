from __future__ import annotations

import logging

import comfy.context_windows
import comfy.patcher_extension
import torch


LOG = logging.getLogger("comfyui-turing-utils")
_BERNINI_ROPE_WRAPPER_KEY = "turing_utils_bernini_context_rope"
_ABSOLUTE_INDEX_KEY = "turing_utils_bernini_absolute_latent_indices"


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


def _window_start_and_stride(window) -> tuple[float, float]:
    indices = list(getattr(window, "index_list", []) or [])
    if not indices:
        return 0.0, 1.0

    start = indices[0]
    stride = 1
    if len(indices) > 1:
        deltas = [b - a for a, b in zip(indices, indices[1:])]
        if deltas and all(delta == deltas[0] for delta in deltas) and deltas[0] > 0:
            stride = deltas[0]

    return float(start), float(stride)


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
        context_latents = [comfy.ldm.common_dit.pad_to_patch_size(lat, self.patch_size) for lat in context_latents]
        for i, lat in enumerate(context_latents):
            context_indices = target_indices if lat.shape[-3] == len(target_indices) else None
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

    start, stride = _window_start_and_stride(window)
    if start == 0.0 and stride == 1.0:
        return executor(*args, **kwargs)

    rope_options = dict(transformer_options.get("rope_options") or {})
    rope_options["shift_t"] = float(rope_options.get("shift_t", 0.0)) + start
    if stride != 1.0:
        rope_options["scale_t"] = float(rope_options.get("scale_t", 1.0)) * stride

    new_transformer_options = dict(transformer_options)
    new_transformer_options["rope_options"] = rope_options
    args, kwargs = _with_transformer_options(args, kwargs, new_transformer_options)
    return executor(*args, **kwargs)


class BerniniScheduledContextHandler(comfy.context_windows.IndexListContextHandler):
    def get_context_windows(self, model, x_in: torch.Tensor, model_options: dict[str]):
        windows = super().get_context_windows(model, x_in, model_options)
        for window in windows:
            window.turing_utils_use_absolute_indices = True
        return windows


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
                        "default": 30,
                        "min": 0,
                        "max": 16384,
                        "tooltip": "The overlap of the context window in real frames.",
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
                "retain_first_frame": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Retain the first I2V frame in every context window."},
                ),
                "split_conds_to_windows": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "advanced": True,
                        "tooltip": "Whether to split multiple conditionings to each window based on region index.",
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
        context_stride: int = 1,
        closed_loop: bool = False,
        fuse_method: str = comfy.context_windows.ContextFuseMethods.PYRAMID,
        freenoise: bool = True,
        retain_first_frame: bool = False,
        split_conds_to_windows: bool = False,
    ):
        latent_context_length, latent_context_overlap = _validate_context_window_frames(
            context_length,
            context_overlap,
        )
        retain_index_list = "0" if retain_first_frame else ""
        context_handler = BerniniScheduledContextHandler(
            context_schedule=comfy.context_windows.get_matching_context_schedule(context_schedule),
            fuse_method=comfy.context_windows.get_matching_fuse_method(fuse_method),
            context_length=latent_context_length,
            context_overlap=latent_context_overlap,
            context_stride=max(int(context_stride), 1),
            closed_loop=bool(closed_loop),
            dim=2,
            freenoise=bool(freenoise),
            cond_retain_index_list=retain_index_list,
            split_conds_to_windows=bool(split_conds_to_windows),
            latent_retain_index_list="",
            causal_window_fix=True,
        )

        patched = model.clone()
        patched.model_options["context_handler"] = context_handler
        patched.model_options.setdefault("transformer_options", {})

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

        comfy.context_windows.create_prepare_sampling_wrapper(patched)
        if freenoise:
            comfy.context_windows.create_sampler_sample_wrapper(patched)
        _install_bernini_absolute_rope_patch()
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            _BERNINI_ROPE_WRAPPER_KEY,
            _bernini_context_rope_wrapper,
        )

        LOG.info(
            "Bernini context windows enabled: schedule=%s, length=%s -> %s latent frames, "
            "overlap=%s -> %s latent frames, stride=%s, closed_loop=%s, "
            "retain_first_frame=%s, split_conds_to_windows=%s, causal_window_fix=True, fuse=%s",
            context_schedule,
            context_length,
            latent_context_length,
            context_overlap,
            latent_context_overlap,
            context_stride,
            closed_loop,
            retain_first_frame,
            split_conds_to_windows,
            fuse_method,
        )
        return (patched,)
