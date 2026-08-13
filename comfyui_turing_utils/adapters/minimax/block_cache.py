"""Trajectory-aware transformer-block caching for MiniMax H3."""

from __future__ import annotations

import inspect
import logging
import math

import torch

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
from comfy.ldm.minimax import model as minimax_model

from ..methods import weak_method


LOG = logging.getLogger("comfyui-turing-utils")
CACHE_KEY = "turing_utils_minimax_h3_block_cache"
PATCH_KEY = "turing_utils_minimax_h3_block_cache"
FORWARD_PATCH_KEY = "diffusion_model._forward"

_MIB = 1 << 20
_RESIDUAL_FORECAST_BLEND = 0.35
_RESIDUAL_FORECAST_LIMIT = 0.18
_RESIDUAL_FORECAST_MAX_BETA = 1.0
_RMS_DRIFT_LIMIT = 0.10
_RMS_GAIN_LIMIT = 0.18
_TURBO_DELTA_FLOOR = 0.30
_TURBO_END_MARGIN = 6
_TURBO_RMS_DRIFT_LIMIT = 0.16
_TURBO_RMS_GAIN_LIMIT = 0.30
_TURBO_SCALE_MAX = 1.15
_TURBO_SCALE_MIN = 0.85
_TURBO_SHORT_STEP_MAX = 10
_TURBO_START = 6

_FORWARD_PARAMETERS = (
    "x",
    "timestep",
    "context",
    "transformer_options",
    "minimax_payload",
    "kwargs",
)


def _sample_rms(hidden: torch.Tensor, span: tuple[int, int]) -> float:
    start, end = span
    if end <= start:
        return 0.0
    token_stride = max((end - start) // 1024, 1)
    sample = hidden[start:end:token_stride, ::16].float()
    return float(torch.sqrt(sample.square().mean() + 1e-12).item())


def _segment_stats(hidden: torch.Tensor, ranges) -> dict[str, float]:
    return {kind: _sample_rms(hidden, span) for kind, span in ranges}


def _relative_change(current: float, reference: float) -> float:
    reference = max(float(reference), 1e-8)
    return abs(float(current) - reference) / reference


def _denoise_step_count(transformer_options) -> int | None:
    sample_sigmas = transformer_options.get("sample_sigmas")
    if sample_sigmas is None:
        return None
    try:
        return max(len(sample_sigmas) - 1, 0)
    except TypeError:
        return None


def _turbo_ranges(block_count: int) -> tuple[int, int]:
    if block_count <= 1:
        return 0, max(block_count, 0)
    start = min(_TURBO_START, block_count - 1)
    return start, max(start, block_count - _TURBO_END_MARGIN)


class MiniMaxH3BlockCache:
    def __init__(
        self,
        threshold: float,
        start_percent: float,
        end_percent: float,
        max_consecutive_skips: int,
        cache_device: str,
        block_count: int,
        turbo_mode: bool = False,
    ):
        self.threshold = float(threshold)
        self.turbo_mode = bool(turbo_mode)
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.skip_start, self.skip_end = (
            _turbo_ranges(block_count) if turbo_mode else (0, block_count)
        )
        self.cache_device = cache_device
        self.block_count = int(block_count)
        self.rms_gain_limit = (
            _TURBO_RMS_GAIN_LIMIT if turbo_mode else _RMS_GAIN_LIMIT
        )
        self.rms_drift_limit = (
            _TURBO_RMS_DRIFT_LIMIT if turbo_mode else _RMS_DRIFT_LIMIT
        )
        self.reset()

    def reset(self):
        self.residual = None
        self.previous_residual = None
        self.residual_before_stats = {}
        self.residual_after_stats = {}
        self.previous_residual_before_stats = {}
        self.previous_residual_after_stats = {}
        self.residual_percent = None
        self.previous_residual_percent = None
        self.residual_delta_score = 0.0
        self.current_percent = -1.0
        self.current_step_index = -1
        self.step_count = 0
        self.previous_signature = None
        self.shape_key = None
        self.schedule = None
        self.last_sigma = None
        self.accumulated_delta = 0.0
        self.consecutive_skips = 0
        self.full_steps = 0
        self.cache_hits = 0
        self.skipped_blocks = 0
        self.cache_ranges = ()
        self.full_run = False
        self.base = None
        self.reject_counts = {}

    def finish(self):
        total = (self.full_steps + self.cache_hits) * self.block_count
        if total:
            LOG.info(
                "MiniMax H3 block cache: acceleration %.1f%%",
                100.0 * self.skipped_blocks / total,
            )
        if self.reject_counts:
            details = ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(self.reject_counts.items())
            )
            LOG.debug("MiniMax H3 block cache diagnostics: %s", details)
        self.reset()

    def _reject(self, reason: str):
        self.reject_counts[reason] = self.reject_counts.get(reason, 0) + 1

    def set_cache_ranges(self, layout):
        segments = getattr(layout, "segments", ()) if layout is not None else ()
        self.cache_ranges = tuple(
            (kind, (int(start), int(end)))
            for start, end, kind in segments
            if kind in ("audio", "video")
        )

    def _step_info(self, transformer_options):
        sigmas = transformer_options.get("sigmas")
        if sigmas is None:
            return None
        current_flat = sigmas.flatten()
        if not current_flat.numel():
            return None
        current_sigma = float(current_flat[0].detach().float().item())

        if self.schedule is None:
            sample_sigmas = transformer_options.get("sample_sigmas")
            if sample_sigmas is None:
                return None
            values = sample_sigmas.detach().float().cpu().flatten().tolist()
            if not values:
                return None
            self.schedule = tuple(float(value) for value in values)
            self.step_count = max(len(self.schedule) - 1, 1)

        count = min(self.step_count, len(self.schedule))
        self.current_step_index = min(
            range(count), key=lambda index: abs(self.schedule[index] - current_sigma)
        )
        denominator = max(self.step_count - 1, 1)
        return current_sigma, self.current_step_index / denominator

    @staticmethod
    def _shape_key(hidden: torch.Tensor, cache_ranges):
        return (
            tuple(hidden.shape),
            hidden.dtype,
            hidden.device.type,
            hidden.device.index,
            tuple(cache_ranges),
        )

    @staticmethod
    def _signature(hidden: torch.Tensor, cache_ranges):
        ranges = cache_ranges or (("all", (0, int(hidden.shape[0]))),)
        feature_count = int(hidden.shape[1])
        feature_stride = max(feature_count // 64, 1)
        parts = []
        for _, (start, end) in ranges:
            row_step = max(math.ceil((end - start) / 512), 1)
            part = hidden[start:end:row_step, ::feature_stride]
            parts.append(part[:, :64].reshape(-1))
        return torch.cat(parts).detach().float()

    def _clear_tensors(self):
        self.residual = None
        self.previous_residual = None
        self.residual_before_stats = {}
        self.residual_after_stats = {}
        self.previous_residual_before_stats = {}
        self.previous_residual_after_stats = {}
        self.residual_percent = None
        self.previous_residual_percent = None
        self.residual_delta_score = 0.0
        self.previous_signature = None
        self.accumulated_delta = 0.0
        self.consecutive_skips = 0
        self.full_run = False
        self.base = None

    @staticmethod
    def _relative_tensor_change(current, reference) -> float:
        if current is None or reference is None or current.shape != reference.shape:
            return math.inf
        current_flat = current.detach().reshape(-1)
        reference_flat = reference.detach().reshape(-1)
        stride = max(current_flat.numel() // 4096, 1)
        current_sample = current_flat[::stride].float()
        reference_sample = reference_flat[::stride].float()
        denominator = reference_sample.abs().mean().clamp_min(1e-6)
        return float(((current_sample - reference_sample).abs().mean() / denominator).item())

    def _cache_on_gpu(self, source: torch.Tensor) -> bool:
        if source.device.type != "cuda":
            return True
        if self.cache_device == "gpu":
            return True
        if self.cache_device == "cpu":
            return False
        required = _MIB * 1024 + source.numel() * source.element_size() * 4
        return comfy.model_management.get_free_memory(source.device) > required

    def _store_residual(self, residual, before_stats, after_stats, percent):
        source = residual.detach()
        if self._cache_on_gpu(source):
            self.residual = source
        else:
            cached = torch.empty_like(source, device="cpu", pin_memory=True)
            cached.copy_(source, non_blocking=cached.is_pinned())
            self.residual = cached
        self.residual_before_stats = dict(before_stats)
        self.residual_after_stats = dict(after_stats)
        self.residual_percent = percent

    def store_middle(self, after):
        if not self.full_run or self.base is None:
            return
        old_residual = self.residual
        old_before_stats = self.residual_before_stats
        old_after_stats = self.residual_after_stats
        old_percent = self.residual_percent

        before_stats = _segment_stats(self.base, self.cache_ranges)
        after_stats = _segment_stats(after, self.cache_ranges)
        self._store_residual(
            after - self.base,
            before_stats,
            after_stats,
            self.current_percent,
        )
        if (
            old_residual is not None
            and old_residual.shape == self.residual.shape
            and old_residual.device == self.residual.device
            and old_residual.dtype == self.residual.dtype
        ):
            self.previous_residual = old_residual
            self.previous_residual_before_stats = old_before_stats
            self.previous_residual_after_stats = old_after_stats
            self.previous_residual_percent = old_percent
            self.residual_delta_score = self._relative_tensor_change(
                self.residual, old_residual
            )
        else:
            self.previous_residual = None
            self.previous_residual_before_stats = {}
            self.previous_residual_after_stats = {}
            self.previous_residual_percent = None
            self.residual_delta_score = math.inf

        self.base = None
        self.full_run = False
        self.accumulated_delta = 0.0
        self.consecutive_skips = 0
        self.full_steps += 1

    def _turbo_segment_scales(self, before_stats):
        scales = {}
        for kind, _ in self.cache_ranges:
            current = before_stats.get(kind)
            reference = self.residual_before_stats.get(kind)
            if current is None or reference is None or reference <= 0.0:
                continue
            scales[kind] = min(
                max(current / reference, _TURBO_SCALE_MIN), _TURBO_SCALE_MAX
            )
        return scales

    def _apply_cached(self, hidden, cached, segment_scales, alpha=1.0):
        if cached.device != hidden.device or cached.dtype != hidden.dtype:
            non_blocking = cached.device.type == "cpu" and cached.is_pinned()
            cached = cached.to(
                device=hidden.device,
                dtype=hidden.dtype,
                non_blocking=non_blocking,
            )
        hidden.add_(cached, alpha=alpha)
        for kind, span in self.cache_ranges:
            scale = segment_scales.get(kind, 1.0)
            if scale != 1.0:
                start, end = span
                hidden[start:end].add_(cached[start:end], alpha=alpha * (scale - 1.0))

    def _forecast_residual(self):
        if (
            self.previous_residual is None
            or self.previous_residual_percent is None
            or self.residual_percent is None
            or not math.isfinite(self.residual_delta_score)
            or self.residual_delta_score > _RESIDUAL_FORECAST_LIMIT
        ):
            return self.residual
        denominator = self.residual_percent - self.previous_residual_percent
        if denominator <= 0.0:
            self._reject("forecast_fallback")
            return self.residual
        beta = min(
            max(
                (self.current_percent - self.residual_percent) / denominator,
                0.0,
            ),
            _RESIDUAL_FORECAST_MAX_BETA,
        )
        forecast_factor = _RESIDUAL_FORECAST_BLEND * beta
        return self.residual + (self.residual - self.previous_residual) * forecast_factor

    def prepare_middle(self, hidden, transformer_options, force_full=False) -> bool:
        step_info = self._step_info(transformer_options)
        if step_info is None:
            self._reject("missing_step_info")
            self.full_run = True
            self.base = hidden.clone()
            return False
        current_sigma, percent = step_info

        shape_key = self._shape_key(hidden, self.cache_ranges)
        if self.shape_key is not None and shape_key != self.shape_key:
            self._reject("shape_reset")
            self._clear_tensors()
        self.shape_key = shape_key
        if self.last_sigma is not None and current_sigma >= self.last_sigma:
            self._reject("discontinuity")
            self._clear_tensors()
        self.last_sigma = current_sigma
        self.current_percent = percent

        signature = self._signature(hidden, self.cache_ranges)
        delta = self._relative_tensor_change(signature, self.previous_signature)
        self.previous_signature = signature
        if math.isfinite(delta):
            self.accumulated_delta += delta

        before_stats = _segment_stats(hidden, self.cache_ranges)
        eligible = True
        if force_full:
            self._reject("patch_overlap")
            eligible = False
        if not self.start_percent <= percent <= self.end_percent:
            self._reject("outside_percent")
            eligible = False
        if self.residual is None or self.residual.shape != hidden.shape:
            self._reject(
                "missing_residual" if self.residual is None else "residual_shape"
            )
            eligible = False
        if not math.isfinite(delta):
            self._reject("missing_delta")
            eligible = False

        delta_threshold = (
            max(self.threshold, _TURBO_DELTA_FLOOR)
            if self.turbo_mode
            else self.threshold
        )
        if self.accumulated_delta > delta_threshold:
            self._reject("delta_threshold")
            eligible = False
        if self.consecutive_skips >= self.max_consecutive_skips:
            self._reject("mcs")
            eligible = False

        if self.residual is not None and self.residual_before_stats:
            gain_changes = [
                _relative_change(value, self.residual_before_stats[kind])
                for kind, value in before_stats.items()
                if kind in self.residual_before_stats
            ]
            drift_changes = [
                _relative_change(value, self.residual_after_stats[kind])
                for kind, value in before_stats.items()
                if kind in self.residual_after_stats
            ]
            if (
                not self.turbo_mode
                and gain_changes
                and max(gain_changes) > self.rms_gain_limit
            ):
                self._reject("rms_gain")
                eligible = False
            if drift_changes and max(drift_changes) > self.rms_drift_limit:
                self._reject("rms_drift")
                eligible = False

        if not eligible:
            self.full_run = True
            self.base = hidden.clone()
            return False

        cached = self._forecast_residual()
        segment_scales = (
            self._turbo_segment_scales(before_stats) if self.turbo_mode else {}
        )
        self._apply_cached(hidden, cached, segment_scales)
        self.full_run = False
        self.base = None
        self.consecutive_skips += 1
        self.cache_hits += 1
        self.skipped_blocks += self.skip_end - self.skip_start
        return True


class MiniMaxH3BlockCacheGroup:
    def __init__(
        self,
        threshold: float,
        start_percent: float,
        end_percent: float,
        max_consecutive_skips: int,
        cache_device: str,
        block_count: int,
        turbo_mode: bool = False,
    ):
        self.threshold = float(threshold)
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.cache_device = cache_device
        self.block_count = int(block_count)
        self.turbo_mode = bool(turbo_mode)
        self.states = {}

    def _strategy(self, transformer_options):
        if not self.turbo_mode:
            return "standard"
        step_count = _denoise_step_count(transformer_options)
        if step_count is not None and step_count <= _TURBO_SHORT_STEP_MAX:
            return "turbo_four"
        return "standard"

    @staticmethod
    def _branch_key(transformer_options):
        uuids = transformer_options.get("uuids")
        if uuids:
            return "uuid", tuple(uuids)
        branches = transformer_options.get("cond_or_uncond")
        if branches is not None and len(branches) == 1:
            return "branch", tuple(branches)
        return ("default",)

    def state_for(self, transformer_options):
        strategy = self._strategy(transformer_options)
        key = (strategy, *self._branch_key(transformer_options))
        state = self.states.get(key)
        if state is not None:
            return state
        if strategy == "turbo_four":
            state = MiniMaxH3BlockCache(
                _TURBO_DELTA_FLOOR,
                0.20,
                0.80,
                1,
                self.cache_device,
                self.block_count,
                True,
            )
        else:
            state = MiniMaxH3BlockCache(
                self.threshold,
                self.start_percent,
                self.end_percent,
                self.max_consecutive_skips,
                self.cache_device,
                self.block_count,
                False,
            )
        self.states[key] = state
        return state

    def reset(self):
        for state in self.states.values():
            state.reset()
        self.states.clear()

    def finish(self):
        for state in self.states.values():
            state.finish()
        self.states.clear()


class _SamplingScope:
    def __init__(self, cache: MiniMaxH3BlockCacheGroup):
        self.cache = cache

    def __call__(self, executor, *args, **kwargs):
        self.cache.reset()
        try:
            return executor(*args, **kwargs)
        finally:
            self.cache.finish()


def _run_block(
    model,
    hidden,
    index,
    t_emb,
    mod_segments,
    rope_freqs,
    transformer_options,
    blocks_replace,
):
    block = model.blocks[index]
    replacement = blocks_replace.get(("double_block", index))
    if replacement is None:
        return block(
            hidden,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options=transformer_options,
        )

    def block_wrap(args):
        return {
            "img": block(
                args["img"],
                args["t_emb"],
                args["mod_segments"],
                args["rope_freqs"],
                transformer_options=args["transformer_options"],
            )
        }

    return replacement(
        {
            "img": hidden,
            "t_emb": t_emb,
            "mod_segments": mod_segments,
            "rope_freqs": rope_freqs,
            "transformer_options": transformer_options,
        },
        {"original_block": block_wrap},
    )["img"]


def _run_span(
    model,
    hidden,
    start,
    end,
    t_emb,
    mod_segments,
    rope_freqs,
    transformer_options,
    blocks_replace,
    device,
):
    blocks = list(model.blocks[start:end])
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
        blocks, device, transformer_options
    )
    for index, block in zip(range(start, end), blocks):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
        hidden = _run_block(
            model,
            hidden,
            index,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options,
            blocks_replace,
        )
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
    return hidden


def _span_has_replacement(blocks_replace, start, end):
    return any(("double_block", index) in blocks_replace for index in range(start, end))


def _cached_forward(
    self,
    x,
    timestep,
    context,
    transformer_options={},
    minimax_payload=None,
    **kwargs,
):
    cache_group = transformer_options.get(CACHE_KEY)
    if not isinstance(cache_group, MiniMaxH3BlockCacheGroup):
        raise RuntimeError("MiniMax H3 cached forward was called without its cache state")

    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype

    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = minimax_model.PackedLayout(
            text_len,
            latent_t,
            lat_h,
            lat_w,
            audio_t,
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
            frame_count=payload.get("frame_count"),
        )

    shift_v = float(
        transformer_options.get(
            "minimax_h3_sigma_shift_video", self.sigma_shift_video
        )
    )
    shift_a = float(
        transformer_options.get(
            "minimax_h3_sigma_shift_audio", self.sigma_shift_audio
        )
    )
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - minimax_model.time_shift_sigma(sigma_v, shift_v, shift_a))

    vis_aug = float(
        payload.get("visual_cond_noise_aug", minimax_model.VISUAL_COND_TIMESTEP)
    )
    aud_aug = float(
        payload.get("audio_cond_noise_aug", minimax_model.AUDIO_COND_TIMESTEP)
    )
    has_vis_cond = any(kind in ("cond", "ref_img") for _, _, kind in layout.segments)
    has_aud_cond = any(kind == "ref_audio" for _, _, kind in layout.segments)
    seg_t = {
        "text": t_v,
        "video": t_v,
        "audio": t_a,
        "cond": max(t_v, vis_aug),
        "ref_img": max(t_v, vis_aug),
        "ref_audio": max(t_a, aud_aug),
    }
    unique_t = sorted(
        {t_v, t_a}
        | ({seg_t["cond"]} if has_vis_cond else set())
        | ({seg_t["ref_audio"]} if has_aud_cond else set())
    )
    t_row = {value: index for index, value in enumerate(unique_t)}
    seg_tag = {
        "text": 1,
        "video": 0,
        "audio": 2,
        "cond": 0,
        "ref_img": 0,
        "ref_audio": 2,
    }

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for start, end, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for index in range(1, end - start + 1):
                if index == end - start or tags[index] != tags[run_start]:
                    mod_segments.append(
                        (
                            start + run_start,
                            start + index,
                            row_base + int(tags[run_start]),
                        )
                    )
                    run_start = index
        else:
            mod_segments.append((start, end, row_base + seg_tag[kind]))

    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = minimax_model.patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = minimax_model.pack_audio(audio_x.to(torch.float32))
    cond_video_rows = self._cond_video_rows(payload, device)
    cond_audio_rows = self._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(
            img_update.shape[0],
            video_rows.shape[1],
            dtype=torch.float32,
            device=device,
        )
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(
            audio_update.shape[0],
            audio_rows.shape[1],
            dtype=torch.float32,
            device=device,
        )
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = self.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != self.hidden_size:
        text_states = self.token_refiner(
            self.condition_proj(text_states), transformer_options=transformer_options
        )

    hidden = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
    video_offset = 0
    audio_offset = 0
    for start, end, kind in layout.segments:
        count = end - start
        if kind == "text":
            hidden[start:end] = text_states
        elif kind in ("cond", "ref_img", "video"):
            hidden[start:end] = video_embed[video_offset : video_offset + count]
            video_offset += count
        else:
            hidden[start:end] = audio_embed[audio_offset : audio_offset + count]
            audio_offset += count

    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if self.use_adaln_curves:
        table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(
            table[i0], table[i0 + 1], (pos - i0).unsqueeze(1)
        )
    else:
        t_emb = self.time_embedder(t_vals).to(dtype)

    rope_freqs = minimax_model.rope_rotation_table(
        self.rope_freqs(layout.position_ids, device), dtype
    )

    cache = cache_group.state_for(transformer_options)
    cache.set_cache_ranges(layout)
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    hidden = _run_span(
        self,
        hidden,
        0,
        cache.skip_start,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options,
        blocks_replace,
        device,
    )
    force_full = _span_has_replacement(
        blocks_replace, cache.skip_start, cache.skip_end
    )
    if not cache.prepare_middle(hidden, transformer_options, force_full=force_full):
        hidden = _run_span(
            self,
            hidden,
            cache.skip_start,
            cache.skip_end,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options,
            blocks_replace,
            device,
        )
        cache.store_middle(hidden)
    hidden = _run_span(
        self,
        hidden,
        cache.skip_end,
        len(self.blocks),
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options,
        blocks_replace,
        device,
    )

    video_seg = next(
        (start, end, t_row[seg_t["video"]])
        for start, end, kind in layout.segments
        if kind == "video"
    )
    audio_seg = next(
        (start, end, t_row[seg_t["audio"]])
        for start, end, kind in layout.segments
        if kind == "audio"
    )
    video, audio = self.final_layer(hidden, t_emb, video_seg, audio_seg)
    video_out = minimax_model.unpatchify_video(
        video,
        latent_t,
        lat_h // 2,
        lat_w // 2,
        self.latents_dim,
        self.patch_size,
    )
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = minimax_model.unpack_audio(audio)
    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]


def _compatible_forward(forward) -> bool:
    try:
        parameters = tuple(inspect.signature(forward).parameters)
    except (TypeError, ValueError):
        return False
    if parameters and parameters[0] == "self":
        parameters = parameters[1:]
    if parameters != _FORWARD_PARAMETERS:
        return False
    code = getattr(forward, "__code__", None)
    if code is None:
        return False
    required = {
        "PackedLayout",
        "patchify_video",
        "pack_audio",
        "time_shift_sigma",
        "rope_rotation_table",
        "make_prefetch_queue",
        "final_layer",
        "unpatchify_video",
        "unpack_audio",
    }
    return required.issubset(code.co_names)


def _make_cached_forward(diffusion_model):
    def forward(
        self,
        x,
        timestep,
        context,
        transformer_options={},
        minimax_payload=None,
        **kwargs,
    ):
        return _cached_forward(
            self,
            x,
            timestep,
            context,
            transformer_options=transformer_options,
            minimax_payload=minimax_payload,
            **kwargs,
        )

    return weak_method(forward, diffusion_model)


def install_minimax_block_cache(
    model,
    threshold: float,
    start_percent: float,
    end_percent: float,
    max_consecutive_skips: int,
    cache_device: str,
    turbo_mode: bool,
):
    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
        raise ValueError("MiniMax H3 block cache only supports MiniMax H3 models")
    if len(diffusion_model.blocks) < 2:
        raise ValueError("MiniMax H3 block cache requires at least two transformer blocks")
    if not _compatible_forward(type(diffusion_model)._forward):
        raise RuntimeError(
            "MiniMax H3 block cache is disabled because MiniMaxH3Model._forward changed"
        )

    patched = model.clone()
    cache = MiniMaxH3BlockCacheGroup(
        threshold,
        start_percent,
        end_percent,
        max_consecutive_skips,
        cache_device,
        len(diffusion_model.blocks),
        turbo_mode,
    )
    transformer_options = patched.model_options.setdefault(
        "transformer_options", {}
    )
    transformer_options[CACHE_KEY] = cache
    patched.add_object_patch(
        FORWARD_PATCH_KEY, _make_cached_forward(diffusion_model)
    )
    if hasattr(patched, "remove_wrappers_with_key"):
        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, PATCH_KEY
        )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        PATCH_KEY,
        _SamplingScope(cache),
    )
    return patched


__all__ = [
    "CACHE_KEY",
    "MiniMaxH3BlockCache",
    "MiniMaxH3BlockCacheGroup",
    "install_minimax_block_cache",
]
