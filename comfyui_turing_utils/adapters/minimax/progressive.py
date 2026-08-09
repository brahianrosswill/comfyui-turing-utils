"""Experimental MiniMax H3 progressive-resolution sampling patch."""

from __future__ import annotations

import contextvars
import logging
import math
from dataclasses import dataclass

import torch

from .acceleration import (
    _MEMORY_CONTEXT_ATTR,
    _MEMORY_SHAPE_KEY,
    _MiniMaxMemoryCond,
    _MiniMaxMemoryShape,
)


LOG = logging.getLogger("comfyui-turing-utils")
_PROGRESSIVE_OUTER_WRAPPER_KEY = "turing_utils_h3_progressive_resolution_steps"
_PROGRESSIVE_COND_WRAPPER_KEY = "turing_utils_h3_progressive_resolution_cond"


@dataclass(frozen=True)
class _H3ProgressiveResolutionConfig:
    low_short_edge: int
    low_resolution_steps: int
    medium_short_edge: int
    medium_resolution_steps: int
    input_downscale: str
    output_upscale: str
    visual_condition_policy: str
    debug: bool = False


def _h3_latent_shapes(conds):
    """Find the processed H3 packed-stream shapes in a conditioning batch."""
    for cond_list in conds:
        if cond_list is None:
            continue
        for cond in cond_list:
            model_conds = cond.get("model_conds", {})
            shape_cond = model_conds.get("latent_shapes")
            shapes = getattr(shape_cond, "cond", None)
            if not isinstance(shapes, (list, tuple)) or len(shapes) < 2:
                continue
            video_shape = tuple(int(value) for value in shapes[0])
            audio_shape = tuple(int(value) for value in shapes[1])
            if (
                len(video_shape) == 5
                and len(audio_shape) == 4
                and video_shape[1] == 24
                and audio_shape[1] == 32
                and audio_shape[2] == 2
            ):
                return list(shapes)
    return None


def _h3_progressive_target_hw(video_shape, low_short_edge: int) -> tuple[int, int]:
    """Return an aspect-preserving H3 latent size aligned to 32-pixel canvas units."""
    final_h, final_w = int(video_shape[-2]), int(video_shape[-1])
    final_short_pixels = min(final_h, final_w) * 16
    if low_short_edge <= 0 or low_short_edge >= final_short_pixels:
        return final_h, final_w

    scale = float(low_short_edge) / float(final_short_pixels)

    def aligned(value: int) -> int:
        # H3 consumes 2x2 latent patches, corresponding to 32x32 pixel units.
        return min(value, max(2, int(round(value * scale / 2.0)) * 2))

    return aligned(final_h), aligned(final_w)


def _resize_h3_video(video: torch.Tensor, height: int, width: int, method: str) -> torch.Tensor:
    if tuple(video.shape[-2:]) == (int(height), int(width)):
        return video
    import comfy.utils

    return comfy.utils.common_upscale(
        video,
        int(width),
        int(height),
        method,
        "disabled",
    )


def _downsample_h3_video(
    video: torch.Tensor,
    height: int,
    width: int,
    config: _H3ProgressiveResolutionConfig,
    state: dict,
) -> torch.Tensor:
    if config.input_downscale != "sigma_blend":
        return _resize_h3_video(video, height, width, config.input_downscale)

    # Nearest-exact preserves the variance of the already-sampled high-resolution
    # noise. Area filtering preserves the emerging low-frequency composition. The
    # deterministic blend avoids injecting a second, unrelated noise trajectory.
    nearest = _resize_h3_video(video, height, width, "nearest-exact")
    area = _resize_h3_video(video, height, width, "area")
    progressive_steps = max(int(state.get("progressive_steps", 1)), 1)
    step = max(int(state.get("step", 0)), 0)
    progressive_sigmas = state.get("progressive_sigmas", ())
    if len(progressive_sigmas) >= 2:
        sigma_start = float(progressive_sigmas[0])
        sigma_end = float(progressive_sigmas[-1])
        sigma = float(progressive_sigmas[min(step, len(progressive_sigmas) - 1)])
        denominator = sigma_start - sigma_end
        nearest_weight = (
            (sigma - sigma_end) / denominator
            if abs(denominator) > 1e-12
            else 0.0
        )
        nearest_weight = min(max(nearest_weight, 0.0), 1.0)
    else:
        nearest_weight = max(0.0, 1.0 - float(step) / float(progressive_steps))
    return torch.lerp(area, nearest, nearest_weight)


def _resize_h3_keyframe_payload(
    payload,
    height: int,
    width: int,
    cache: dict,
):
    if not isinstance(payload, dict) or not payload.get("keyframes"):
        return payload

    resized_keyframes = []
    for keyframe in payload["keyframes"]:
        if not isinstance(keyframe, dict) or not torch.is_tensor(keyframe.get("latent")):
            resized_keyframes.append(keyframe)
            continue
        latent = keyframe["latent"]
        cache_key = (id(latent), int(height), int(width))
        resized = cache.get(cache_key)
        if resized is None:
            resized = _resize_h3_video(latent, height, width, "area")
            cache[cache_key] = resized
        new_keyframe = keyframe.copy()
        new_keyframe["latent"] = resized
        resized_keyframes.append(new_keyframe)

    new_payload = payload.copy()
    new_payload["keyframes"] = resized_keyframes
    original_cond_latents = list(payload.get("cond_video_latents", ()))
    resized_latents = [
        keyframe["latent"]
        for keyframe in resized_keyframes
        if isinstance(keyframe, dict) and torch.is_tensor(keyframe.get("latent"))
    ]
    # extra_conds orders keyframes before independent reference latents.
    new_payload["cond_video_latents"] = resized_latents + original_cond_latents[
        len(payload["keyframes"]):
    ]
    # Target and condition geometry both changed; force a fresh lightweight layout.
    new_payload.pop("layout", None)
    return new_payload


def _resize_h3_memory_condition(
    memory_cond,
    high_shapes,
    low_shapes,
    *,
    resized_keyframes: int = 0,
):
    if not isinstance(memory_cond, _MiniMaxMemoryCond):
        return memory_cond
    old = memory_cond.cond
    video_shape = tuple(int(value) for value in low_shapes[0])
    audio_shape = tuple(int(value) for value in low_shapes[1])
    latent_t = int(video_shape[2])
    frame_rows = math.ceil(video_shape[3] / 2) * math.ceil(video_shape[4] / 2)
    target_visual_rows = latent_t * frame_rows
    target_audio_rows = int(audio_shape[2]) * int(audio_shape[3])
    target_rows = target_visual_rows + target_audio_rows

    old_video_shape = tuple(int(value) for value in high_shapes[0])
    old_frame_rows = math.ceil(old_video_shape[3] / 2) * math.ceil(old_video_shape[4] / 2)
    visual_condition_rows = int(old.visual_condition_rows) + int(resized_keyframes) * (
        frame_rows - old_frame_rows
    )
    condition_rows = max(int(old.full_rows) - int(old.target_rows), 0)
    condition_rows += visual_condition_rows - int(old.visual_condition_rows)
    full_rows = target_rows + condition_rows
    target_area = math.prod(video_shape[1:]) + math.prod(audio_shape[1:])
    equivalent_area = math.ceil(target_area * condition_rows / max(target_rows, 1))
    return _MiniMaxMemoryCond(
        _MiniMaxMemoryShape(
            equivalent_area,
            full_rows=full_rows,
            target_rows=target_rows,
            target_visual_rows=target_visual_rows,
            target_audio_rows=target_audio_rows,
            visual_condition_rows=visual_condition_rows,
            audio_condition_rows=int(old.audio_condition_rows),
            hidden_size=int(old.hidden_size),
            video_row_width=int(old.video_row_width),
            audio_row_width=int(old.audio_row_width),
        )
    )


def _patch_h3_conds_for_shapes(
    conds,
    high_shapes,
    low_shapes,
    config: _H3ProgressiveResolutionConfig,
    state: dict,
):
    import comfy.conds

    shape_cond = comfy.conds.CONDConstant(low_shapes)
    output = []
    payload_memo = {}
    for cond_list in conds:
        if cond_list is None:
            output.append(None)
            continue
        patched_list = []
        for cond in cond_list:
            patched_cond = cond.copy()
            model_conds = cond.get("model_conds")
            if not isinstance(model_conds, dict):
                patched_list.append(patched_cond)
                continue
            patched_model_conds = model_conds.copy()
            if "latent_shapes" in patched_model_conds:
                patched_model_conds["latent_shapes"] = shape_cond
            keyframe_count = 0
            if config.visual_condition_policy == "resize_keyframes":
                payload_cond = patched_model_conds.get("minimax_payload")
                payload = getattr(payload_cond, "cond", None)
                if isinstance(payload, dict) and payload.get("keyframes"):
                    keyframe_count = len(payload["keyframes"])
                    memo_key = id(payload_cond)
                    patched_payload_cond = payload_memo.get(memo_key)
                    if patched_payload_cond is None:
                        patched_payload = _resize_h3_keyframe_payload(
                            payload,
                            int(low_shapes[0][-2]),
                            int(low_shapes[0][-1]),
                            state.setdefault("condition_cache", {}),
                        )
                        patched_payload_cond = comfy.conds.CONDConstant(patched_payload)
                        payload_memo[memo_key] = patched_payload_cond
                    patched_model_conds["minimax_payload"] = patched_payload_cond
            if _MEMORY_SHAPE_KEY in patched_model_conds:
                patched_model_conds[_MEMORY_SHAPE_KEY] = _resize_h3_memory_condition(
                    patched_model_conds[_MEMORY_SHAPE_KEY],
                    high_shapes,
                    low_shapes,
                    resized_keyframes=keyframe_count,
                )
            patched_cond["model_conds"] = patched_model_conds
            patched_list.append(patched_cond)
        output.append(patched_list)
    return output


def _h3_conds_support_progressive_resize(conds) -> bool:
    # Spatial areas, masks, and controls need their own shape transformations.
    # The official H3 text/keyframe/reference paths do not use these fields.
    unsupported = ("area", "mask", "control", "gligen")
    return all(
        not any(key in cond for key in unsupported)
        for cond_list in conds
        if cond_list is not None
        for cond in cond_list
    )


def _make_h3_progressive_wrappers(config: _H3ProgressiveResolutionConfig):
    runtime = contextvars.ContextVar(
        "turing_utils_h3_progressive_resolution_runtime",
        default=None,
    )

    def outer_sample_wrapper(executor, *args, **kwargs):
        sigmas = kwargs.get("sigmas")
        if sigmas is None and len(args) > 3:
            sigmas = args[3]
        total_steps = max(int(getattr(sigmas, "shape", (0,))[-1]) - 1, 0)
        low_steps = min(max(int(config.low_resolution_steps), 0), total_steps)
        medium_steps = min(
            max(int(config.medium_resolution_steps), 0),
            total_steps - low_steps,
        )
        progressive_steps = low_steps + medium_steps
        if progressive_steps <= 0:
            return executor(*args, **kwargs)

        state = {
            "step": 0,
            "low_steps": low_steps,
            "medium_steps": medium_steps,
            "progressive_steps": progressive_steps,
            "total_steps": total_steps,
            "progressive_sigmas": (
                sigmas[:progressive_steps + 1].detach().to("cpu", torch.float32).tolist()
                if torch.is_tensor(sigmas)
                else ()
            ),
            "condition_cache": {},
            "logged_stages": set(),
            "fallback_logged": False,
        }
        callback = kwargs.get("callback")
        callback_in_args = len(args) > 5
        if callback_in_args:
            callback = args[5]

        def progressive_callback(step, x0, x, callback_total_steps):
            state["step"] = max(int(state["step"]), int(step) + 1)
            if callback is not None:
                return callback(step, x0, x, callback_total_steps)
            return None

        if callback_in_args:
            args_list = list(args)
            args_list[5] = progressive_callback
            args = tuple(args_list)
        else:
            kwargs = kwargs.copy()
            kwargs["callback"] = progressive_callback

        token = runtime.set(state)
        try:
            return executor(*args, **kwargs)
        finally:
            runtime.reset(token)

    def calc_cond_batch_wrapper(executor, model, conds, x_in, timestep, model_options):
        state = runtime.get()
        if state is None or int(state["step"]) >= int(state["progressive_steps"]):
            return executor(model, conds, x_in, timestep, model_options)
        if not _h3_conds_support_progressive_resize(conds):
            if config.debug and not state["fallback_logged"]:
                LOG.warning(
                    "H3 progressive resolution skipped a staged step because spatial areas, masks, or controls are attached"
                )
                state["fallback_logged"] = True
            return executor(model, conds, x_in, timestep, model_options)

        high_shapes = _h3_latent_shapes(conds)
        if high_shapes is None:
            return executor(model, conds, x_in, timestep, model_options)
        video_shape = high_shapes[0]
        if int(state["step"]) < int(state["low_steps"]):
            stage_name = "low"
            stage_short_edge = config.low_short_edge
        else:
            stage_name = "medium"
            stage_short_edge = config.medium_short_edge
        low_h, low_w = _h3_progressive_target_hw(video_shape, stage_short_edge)
        if (low_h, low_w) == tuple(video_shape[-2:]):
            return executor(model, conds, x_in, timestep, model_options)

        import comfy.utils

        high_streams = list(comfy.utils.unpack_latents(x_in, high_shapes))
        low_video = _downsample_h3_video(
            high_streams[0],
            low_h,
            low_w,
            config,
            state,
        )
        low_streams = [low_video, *high_streams[1:]]
        low_x, low_shapes = comfy.utils.pack_latents(low_streams)
        low_conds = _patch_h3_conds_for_shapes(
            conds,
            high_shapes,
            low_shapes,
            config,
            state,
        )

        if config.debug and stage_name not in state["logged_stages"]:
            LOG.warning(
                "Experimental H3 progressive resolution active: stage=%s step_range=%d:%d total_processed=%d/%d video_latent=%sx%s -> %sx%s input=%s output=%s",
                stage_name,
                0 if stage_name == "low" else state["low_steps"],
                state["low_steps"] if stage_name == "low" else state["progressive_steps"],
                state["progressive_steps"],
                state["total_steps"],
                video_shape[-1],
                video_shape[-2],
                low_w,
                low_h,
                config.input_downscale,
                config.output_upscale,
            )
            state["logged_stages"].add(stage_name)

        previous_memory_context = (
            getattr(model, _MEMORY_CONTEXT_ATTR, None) if model is not None else None
        )
        if model is not None:
            setattr(model, _MEMORY_CONTEXT_ATTR, {"latent_shapes": low_shapes})
        try:
            low_outputs = executor(model, low_conds, low_x, timestep, model_options)
        finally:
            if model is None:
                pass
            elif previous_memory_context is None:
                try:
                    delattr(model, _MEMORY_CONTEXT_ATTR)
                except AttributeError:
                    pass
            else:
                setattr(model, _MEMORY_CONTEXT_ATTR, previous_memory_context)
        projected_low_video = _resize_h3_video(
            low_video,
            int(video_shape[-2]),
            int(video_shape[-1]),
            config.output_upscale,
        )
        high_outputs = []
        for low_output in low_outputs:
            output_streams = list(comfy.utils.unpack_latents(low_output, low_shapes))
            resized_denoised = _resize_h3_video(
                output_streams[0],
                int(video_shape[-2]),
                int(video_shape[-1]),
                config.output_upscale,
            )
            # BaseModel has already converted H3's flow velocity into
            # D_low = X_low - sigma * V_low. Transferring U(D_low) directly
            # creates a spurious projection derivative. Preserve the sampler's
            # high-resolution state and resize only the predicted velocity:
            # D_high = U(D_low) + X_high - U(X_low).
            output_dtype = resized_denoised.dtype
            output_streams[0] = resized_denoised + (
                high_streams[0].to(output_dtype)
                - projected_low_video.to(output_dtype)
            )
            packed_output, _ = comfy.utils.pack_latents(output_streams)
            high_outputs.append(packed_output)
        return high_outputs

    return outer_sample_wrapper, calc_cond_batch_wrapper


def apply_h3_progressive_resolution_patch(
    model,
    *,
    low_short_edge: int = 480,
    low_resolution_steps: int = 2,
    medium_short_edge: int = 720,
    medium_resolution_steps: int = 0,
    input_downscale: str = "sigma_blend",
    output_upscale: str = "bilinear",
    visual_condition_policy: str = "resize_keyframes",
    debug: bool = False,
):
    """Run early H3 DiT evaluations at lower spatial resolution while keeping one final-resolution sampler state."""
    input_methods = ("sigma_blend", "nearest-exact", "area")
    output_methods = ("bilinear", "bicubic", "nearest-exact")
    condition_policies = ("resize_keyframes", "keep_original")
    if input_downscale not in input_methods:
        raise ValueError(f"Unsupported input_downscale: {input_downscale}")
    if output_upscale not in output_methods:
        raise ValueError(f"Unsupported output_upscale: {output_upscale}")
    if visual_condition_policy not in condition_policies:
        raise ValueError(f"Unsupported visual_condition_policy: {visual_condition_policy}")
    if int(low_short_edge) < 32:
        raise ValueError("low_short_edge must be at least 32 pixels")
    if int(medium_short_edge) < 32:
        raise ValueError("medium_short_edge must be at least 32 pixels")
    if int(low_resolution_steps) < 0:
        raise ValueError("low_resolution_steps must be non-negative")
    if int(medium_resolution_steps) < 0:
        raise ValueError("medium_resolution_steps must be non-negative")

    import comfy.patcher_extension

    config = _H3ProgressiveResolutionConfig(
        low_short_edge=int(low_short_edge),
        low_resolution_steps=int(low_resolution_steps),
        medium_short_edge=int(medium_short_edge),
        medium_resolution_steps=int(medium_resolution_steps),
        input_downscale=input_downscale,
        output_upscale=output_upscale,
        visual_condition_policy=visual_condition_policy,
        debug=bool(debug),
    )
    outer_wrapper, cond_wrapper = _make_h3_progressive_wrappers(config)
    patched = model.clone()
    if callable(getattr(patched, "remove_wrappers_with_key", None)):
        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            _PROGRESSIVE_OUTER_WRAPPER_KEY,
        )
        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
            _PROGRESSIVE_COND_WRAPPER_KEY,
        )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        _PROGRESSIVE_OUTER_WRAPPER_KEY,
        outer_wrapper,
    )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
        _PROGRESSIVE_COND_WRAPPER_KEY,
        cond_wrapper,
    )
    LOG.info(
        "Enabled experimental H3 progressive resolution: low_edge=%d low_steps=%d medium_edge=%d medium_steps=%d input=%s output=%s visual_conditions=%s",
        config.low_short_edge,
        config.low_resolution_steps,
        config.medium_short_edge,
        config.medium_resolution_steps,
        config.input_downscale,
        config.output_upscale,
        config.visual_condition_policy,
    )
    return patched
