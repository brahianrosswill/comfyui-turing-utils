from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from collections import Counter
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

import comfy.model_management as model_management
import comfy.ops
import comfy.model_patcher
import comfy.sample
import comfy.samplers
import comfy.utils

from .loader import build_loader_state_dict, is_svdint4_file


LOG = logging.getLogger("comfyui-svdint4")

SEEDVR2_SCALING_FACTOR = 0.9152
SEEDVR2_T = 1000.0
_FLOAT8_DTYPES = tuple(
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if hasattr(torch, name)
)


class _LogShim:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def log(self, message: str, level: str = "INFO", **_: Any) -> None:
        if self.enabled or level in {"WARNING", "ERROR"}:
            getattr(LOG, level.lower(), LOG.info)("SeedVR2: %s", message)

    def start_timer(self, *_: Any, **__: Any) -> None:
        return

    def end_timer(self, *_: Any, **__: Any) -> None:
        return


def _compute_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability(device)
        if major >= 8:
            return torch.bfloat16
    return torch.float16


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _target_dimensions(height: int, width: int, resolution: int, max_resolution: int) -> tuple[int, int]:
    short = min(height, width)
    scale = max(float(resolution) / float(short), 1.0)
    out_h = height * scale
    out_w = width * scale
    if max_resolution > 0 and max(out_h, out_w) > max_resolution:
        limit = float(max_resolution) / max(out_h, out_w)
        out_h *= limit
        out_w *= limit
    return max(2, int(out_h) // 2 * 2), max(2, int(out_w) // 2 * 2)


def _pad_16(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = x.shape[-2:]
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return x, (height, width)


def _pad_4n1(video_tchw: torch.Tensor) -> tuple[torch.Tensor, int]:
    frames = int(video_tchw.shape[0])
    if frames % 4 == 1:
        return video_tchw, frames
    target = ((frames - 1) // 4 + 1) * 4 + 1
    need = target - frames
    if frames == 1:
        pad = video_tchw[-1:].repeat(need, 1, 1, 1)
    elif need >= frames:
        repeat = video_tchw[-1:].repeat(need - frames + 1, 1, 1, 1)
        reverse = video_tchw[1:].flip(0)
        pad = torch.cat([reverse, repeat], dim=0)[:need]
    else:
        pad = video_tchw[-need - 1 : -1].flip(0)
    return torch.cat([video_tchw, pad], dim=0), frames


def _resize_normalize(images: torch.Tensor, resolution: int, max_resolution: int) -> tuple[torch.Tensor, tuple[int, int]]:
    if images.ndim != 4 or int(images.shape[-1]) not in (3, 4):
        raise ValueError(f"SeedVR2 expects IMAGE [frames,h,w,3/4], got {tuple(images.shape)}")
    rgb = images[..., :3].to(torch.float32).clamp(0.0, 1.0)
    frames, height, width, _channels = rgb.shape
    target_h, target_w = _target_dimensions(height, width, resolution, max_resolution)
    video = rgb.permute(0, 3, 1, 2)
    if (height, width) != (target_h, target_w):
        video = F.interpolate(video, size=(target_h, target_w), mode="bicubic", align_corners=False)
        video = video.clamp(0.0, 1.0)
    video, true_dims = _pad_16(video)
    video = video.mul(2.0).sub(1.0)
    video, original_frames = _pad_4n1(video)
    if original_frames != frames:
        raise RuntimeError("internal SeedVR2 frame bookkeeping error")
    return video, true_dims


def _video_to_vae_input(video_tchw: torch.Tensor) -> torch.Tensor:
    return video_tchw.permute(1, 0, 2, 3).unsqueeze(0)


def _vae_latent_to_seedvr2(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    latent = latent.permute(0, 2, 3, 4, 1)
    return latent.squeeze(0)


def _seedvr2_latent_to_comfy(latent: torch.Tensor) -> torch.Tensor:
    return latent.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def _comfy_latent_to_seedvr2(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(f"SeedVR2 currently samples one 4n+1 window at a time, got {tuple(latent.shape)}")
    return latent.squeeze(0).permute(1, 2, 3, 0).contiguous()


def _seedvr2_condition(latent: torch.Tensor, latent_blur: torch.Tensor) -> torch.Tensor:
    cond = torch.zeros((*latent.shape[:-1], latent.shape[-1] + 1), device=latent.device, dtype=latent.dtype)
    cond[..., :-1] = latent_blur
    cond[..., -1:] = 1.0
    return cond


def _model_size_bytes(model: torch.nn.Module) -> int:
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor is not None:
            total += tensor.nelement() * tensor.element_size()
    return total


def _format_tensor_counter(counter: Counter[str]) -> str:
    return ", ".join(f"{key}={value / (1024 * 1024):.2f} MB" for key, value in counter.most_common())


def _log_module_placement(module: torch.nn.Module, label: str) -> None:
    device_bytes: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()

    for iterator in (module.parameters(), module.buffers()):
        for tensor in iterator:
            if tensor is None:
                continue
            size = tensor.nelement() * tensor.element_size()
            device_bytes[str(tensor.device)] += size
            dtype_bytes[str(tensor.dtype)] += size

    if not device_bytes:
        return
    LOG.info(
        "SeedVR2 %s tensor placement: devices=[%s], dtypes=[%s]",
        label,
        _format_tensor_counter(device_bytes),
        _format_tensor_counter(dtype_bytes),
    )


def _set_runtime_device(module: torch.nn.Module, device: torch.device) -> None:
    try:
        module.device = torch.device(device)
    except AttributeError:
        # Some imported modules expose device as a read-only property derived
        # from their tensors. ComfyUI moves those tensors through the patcher.
        return


def _ensure_finite(tensor: torch.Tensor, label: str) -> None:
    if torch.isfinite(tensor).all():
        return
    finite = torch.isfinite(tensor)
    bad = int((~finite).sum().item())
    valid = tensor[finite]
    if valid.numel() > 0:
        summary = f"valid_min={valid.min().item():.5g}, valid_max={valid.max().item():.5g}"
    else:
        summary = "no finite values"
    raise RuntimeError(
        f"SeedVR2 {label} produced {bad} non-finite value(s) in shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, device={tensor.device}; {summary}."
    )


def _device_label(device: torch.device) -> str:
    try:
        return model_management.get_torch_device_name(device)
    except Exception:
        return str(device)


def _require_accelerator_device(device: torch.device, label: str) -> torch.device:
    if model_management.is_device_cpu(device):
        raise RuntimeError(
            f"SeedVR2 {label} load device resolved to CPU. "
            "SeedVR2 DiT CPU inference is not supported by this node because it is too slow. "
            "Start ComfyUI with GPU support enabled and without --cpu, or free/select a GPU device before running SeedVR2."
        )
    return device


def _unload_patcher(patcher: comfy.model_patcher.ModelPatcher | None) -> None:
    if patcher is None:
        return
    try:
        model_management.unload_model_and_clones(patcher, unload_additional_models=False)
    except Exception:
        LOG.debug("SeedVR2 patcher managed unload failed", exc_info=True)
    try:
        patcher.detach()
    except Exception:
        LOG.debug("SeedVR2 patcher detach failed", exc_info=True)


def _infer_dit_template(path: str, model_name: str) -> str:
    lower = model_name.lower()
    if "7b" in lower:
        return "7b"
    if "3b" in lower:
        return "3b"

    max_block = -1
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not key.startswith("blocks."):
                    continue
                parts = key.split(".", 2)
                if len(parts) >= 2 and parts[1].isdigit():
                    max_block = max(max_block, int(parts[1]))
    except Exception as exc:
        raise ValueError(
            f"Could not inspect SeedVR2 DiT checkpoint architecture for {model_name}. "
            "Use a safetensors/.sft checkpoint whose filename or block keys identify a 3B or 7B SeedVR2 DiT."
        ) from exc

    if max_block >= 35:
        return "7b"
    if 0 <= max_block <= 31:
        return "3b"
    raise ValueError(
        f"Unsupported SeedVR2 DiT checkpoint architecture for {model_name}: highest block index is {max_block}. "
        "Only the SeedVR2 3B and 7B DiT templates are supported."
    )


def _instantiate_dit(template: str, attention_mode: str) -> tuple[torch.nn.Module, Any]:
    if template == "7b":
        from . import seedvr2_dit7b as na

        return (
            na.NaDiT(
                vid_in_channels=33,
                vid_out_channels=16,
                vid_dim=3072,
                txt_in_dim=5120,
                txt_dim=3072,
                emb_dim=18432,
                heads=24,
                head_dim=128,
                expand_ratio=4,
                norm="fusedrms",
                norm_eps=1e-5,
                ada="single",
                qk_bias=False,
                qk_rope=True,
                qk_norm="fusedrms",
                patch_size=(1, 2, 2),
                num_layers=36,
                block_type=["mmdit_sr"] * 36,
                shared_qkv=False,
                shared_mlp=False,
                mlp_type="normal",
                window=[(4, 3, 3)] * 36,
                window_method=["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * 18,
                attention_mode=attention_mode,
            ),
            na,
        )

    if template != "3b":
        raise ValueError(f"Unsupported SeedVR2 DiT template: {template}")

    from . import seedvr2_dit3b as na

    return (
        na.NaDiT(
            vid_in_channels=33,
            vid_out_channels=16,
            vid_dim=2560,
            vid_out_norm="fusedrms",
            txt_in_dim=5120,
            txt_in_norm="fusedln",
            txt_dim=2560,
            emb_dim=15360,
            heads=20,
            head_dim=128,
            expand_ratio=4,
            norm="fusedrms",
            norm_eps=1.0e-5,
            ada="single",
            qk_bias=False,
            qk_norm="fusedrms",
            patch_size=(1, 2, 2),
            num_layers=32,
            mm_layers=10,
            mlp_type="swiglu",
            msa_type=[None] * 32,
            block_type=["mmdit_sr"] * 32,
            window=[(4, 3, 3)] * 32,
            window_method=["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * 16,
            rope_type="mmrope3d",
            rope_dim=128,
            attention_mode=attention_mode,
        ),
        na,
    )


def _instantiate_vae() -> torch.nn.Module:
    from .seedvr2_vae import VideoAutoencoderKLWrapper

    model = VideoAutoencoderKLWrapper(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock3D",) * 4,
        up_block_types=("UpDecoderBlock3D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=16,
        norm_num_groups=32,
        spatial_downsample_factor=8,
        temporal_downsample_factor=4,
        freeze_encoder=False,
        temporal_scale_num=2,
        slicing_sample_min_size=4,
        inflation_mode="pad",
        use_quant_conv=False,
        use_post_quant_conv=False,
    )
    model.set_causal_slicing(split_size=4, memory_device="same")
    model.set_memory_limit(conv_max_mem=0.5, norm_max_mem=0.5)
    return model.eval()


def _replace_svdint4_linears(
    model: torch.nn.Module,
    packed_layer_tensors: dict[str, dict[str, torch.Tensor]],
    w4_layer_tensors: dict[str, dict[str, torch.Tensor]],
) -> torch.nn.Module:
    from .loader import SVDInt4LinearOp

    target_names = set(packed_layer_tensors) | set(w4_layer_tensors)
    if not target_names:
        return model
    linear_cls = type(
        f"SeedVR2SVDInt4Linear_{id(packed_layer_tensors):x}",
        (SVDInt4LinearOp,),
        {
            "packed_layer_names": frozenset(packed_layer_tensors),
            "packed_layer_tensors": packed_layer_tensors,
            "w4_layer_names": frozenset(w4_layer_tensors),
            "w4_layer_tensors": w4_layer_tensors,
        },
    )
    modules = dict(model.named_modules())
    missing = sorted(name for name in target_names if name not in modules)
    if missing:
        raise ValueError(f"SVDInt4 SeedVR2 checkpoint targets missing layer(s), first: {missing[:8]}")
    for name in sorted(target_names):
        module = modules[name]
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"SVDInt4 layer {name} targets {type(module).__name__}, expected Linear")
        parent, attr = _parent_module(model, name)
        replacement = linear_cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        replacement.train(module.training)
        setattr(parent, attr, replacement)
    return model


def _is_float8_dtype(dtype: torch.dtype) -> bool:
    return dtype in _FLOAT8_DTYPES


def _replace_fp8_linears_with_comfy_ops(model: torch.nn.Module, load_device: torch.device) -> tuple[int, str]:
    fp8_compute = load_device.type == "cuda" and model_management.supports_fp8_compute(load_device)
    linear_cls = comfy.ops.fp8_ops.Linear if fp8_compute else comfy.ops.manual_cast.Linear
    op_name = "fp8_ops" if fp8_compute else "manual_cast"
    replacements = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if isinstance(module, linear_cls):
            continue
        if module.weight is None or not _is_float8_dtype(module.weight.dtype):
            continue

        parent, attr = _parent_module(model, name)
        replacement = linear_cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device="meta",
            dtype=module.weight.dtype,
        )
        replacement.weight = module.weight
        replacement.bias = module.bias
        replacement.weight_function = list(getattr(module, "weight_function", ()))
        replacement.bias_function = list(getattr(module, "bias_function", ()))
        replacement.weight_comfy_model_dtype = module.weight.dtype
        if module.bias is not None:
            replacement.bias_comfy_model_dtype = module.bias.dtype
        replacement.train(module.training)
        setattr(parent, attr, replacement)
        replacements += 1
    return replacements, op_name


def _parent_module(model: torch.nn.Module, name: str) -> tuple[torch.nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _load_standard_state(
    model: torch.nn.Module,
    path: str,
    *,
    dtype: torch.dtype | None = None,
    assign: bool = True,
) -> torch.nn.Module:
    state = comfy.utils.load_torch_file(path, safe_load=True, device=torch.device("cpu"))
    if dtype is not None:
        for key, value in list(state.items()):
            if torch.is_tensor(value) and value.is_floating_point():
                state[key] = value.to(dtype=dtype)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=assign)
    if missing:
        LOG.warning("SeedVR2 missing %d key(s) while loading %s", len(missing), os.path.basename(path))
    if unexpected:
        LOG.warning("SeedVR2 unexpected %d key(s) while loading %s", len(unexpected), os.path.basename(path))
    return model


def _load_weights(
    model: torch.nn.Module,
    path: str,
    *,
    dtype: torch.dtype | None = None,
    load_device: torch.device | None = None,
) -> torch.nn.Module:
    svdint4 = is_svdint4_file(path)
    if svdint4:
        state, _metadata, packed, w4 = build_loader_state_dict(path)
        model = _replace_svdint4_linears(model, packed, w4)
        if dtype is not None:
            for key, value in list(state.items()):
                if torch.is_tensor(value) and value.is_floating_point():
                    state[key] = value.to(dtype=dtype)
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    else:
        model = _load_standard_state(model, path, dtype=dtype, assign=True)
        missing, unexpected = [], []
    if missing:
        LOG.warning("SeedVR2 missing %d key(s) while loading %s", len(missing), os.path.basename(path))
    if unexpected:
        LOG.warning("SeedVR2 unexpected %d key(s) while loading %s", len(unexpected), os.path.basename(path))
    if not svdint4:
        fp8_linears, op_name = _replace_fp8_linears_with_comfy_ops(
            model,
            load_device if load_device is not None else model_management.get_torch_device(),
        )
        if fp8_linears:
            LOG.info(
                "SeedVR2 routed %d FP8 Linear layer(s) through ComfyUI %s; FP8 weight storage is preserved.",
                fp8_linears,
                op_name,
            )
    return model


class SeedVR2VAE:
    def __init__(self, path: str, load_device: torch.device, offload_device: torch.device, dtype: torch.dtype):
        core = _instantiate_vae()
        core.to(dtype)
        _load_standard_state(core, path, dtype=None, assign=True)
        model_management.archive_model_dtypes(core)
        core.to(offload_device)
        self.model = _PatchableVAE(core, offload_device)
        self.dtype = dtype
        self.load_device = load_device
        self.offload_device = offload_device
        self.patcher = comfy.model_patcher.CoreModelPatcher(
            self.model,
            load_device=load_device,
            offload_device=offload_device,
            size=_model_size_bytes(self.model),
        )

    def _load(self, memory_required: int) -> None:
        model_management.load_models_gpu(
            [self.patcher],
            memory_required=memory_required,
        )
        self.model.device = self.patcher.load_device
        _set_runtime_device(self.model.core, self.patcher.load_device)
        _log_module_placement(self.model.core, "VAE")

    @torch.no_grad()
    def encode(self, video_tchw: torch.Tensor, seed: int, tiled: bool, tile_size: int, tile_overlap: int) -> torch.Tensor:
        _seed_everything(seed + 1_000_000)
        samples = _video_to_vae_input(video_tchw).to(device=self.load_device, dtype=self.dtype)
        memory = int(samples.nelement() * samples.element_size() * 60)
        self._load(memory)
        encoded = self.model.encode(
            samples,
            tiled=tiled,
            tile_size=(tile_size, tile_size),
            tile_overlap=(tile_overlap, tile_overlap),
        ).latent
        latent = _vae_latent_to_seedvr2(encoded).to(dtype=self.dtype)
        _ensure_finite(latent, "VAE encode")
        return (latent - 0.0) * SEEDVR2_SCALING_FACTOR

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, tiled: bool, tile_size: int, tile_overlap: int) -> torch.Tensor:
        latent = (latent / SEEDVR2_SCALING_FACTOR).to(device=self.load_device, dtype=self.dtype)
        z = latent.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        memory = int(z.nelement() * z.element_size() * 900)
        self._load(memory)
        decoded = self.model.decode(
            z,
            tiled=tiled,
            tile_size=(tile_size, tile_size),
            tile_overlap=(tile_overlap, tile_overlap),
        ).sample
        if decoded.ndim == 4:
            decoded = decoded.unsqueeze(2)
        decoded = decoded.squeeze(0).permute(1, 0, 2, 3).contiguous()
        _ensure_finite(decoded, "VAE decode")
        return decoded


class _PatchableVAE(torch.nn.Module):
    def __init__(self, core: torch.nn.Module, device: torch.device):
        super().__init__()
        self.core = core
        self.device = device
        _set_runtime_device(self.core, device)

    def encode(self, *args, **kwargs):
        _set_runtime_device(self.core, self.device)
        return self.core.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        _set_runtime_device(self.core, self.device)
        return self.core.decode(*args, **kwargs)


class SeedVR2ComfyModel(torch.nn.Module):
    def __init__(
        self,
        dit: torch.nn.Module,
        na_module: Any,
        text_pos: torch.Tensor,
        text_neg: torch.Tensor,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.diffusion_model = dit
        self.na = na_module
        self.dtype = dtype
        self.device = torch.device("cpu")
        self.current_patcher = None
        self.current_condition: torch.Tensor | None = None
        self.register_buffer("text_pos", text_pos.to(dtype=dtype), persistent=False)
        self.register_buffer("text_neg", text_neg.to(dtype=dtype), persistent=False)
        self.model_options = {"transformer_options": {}}
        self.model_sampling = _SeedVR2FlowSampling()
        self.latent_format = _SeedVR2LatentFormat()

    def get_model_object(self, name: str) -> Any:
        if name == "model_sampling":
            return self.model_sampling
        if name == "latent_format":
            return self.latent_format
        return getattr(self, name)

    def get_dtype(self) -> torch.dtype:
        try:
            return next(self.diffusion_model.parameters()).dtype
        except StopIteration:
            return self.dtype

    def model_dtype(self) -> torch.dtype:
        return self.get_dtype()

    def memory_required(self, input_shape, cond_shapes=None) -> int:
        del cond_shapes
        return int(math.prod(input_shape) * torch.tensor([], dtype=self.dtype).element_size() * 36)

    def extra_conds_shapes(self, **kwargs) -> dict[str, tuple[int, ...]]:
        del kwargs
        return {}

    def extra_conds(self, **kwargs) -> dict[str, Any]:
        del kwargs
        return {}

    def process_latent_in(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def process_latent_out(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def apply_model(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        if self.current_condition is None:
            raise RuntimeError("SeedVR2 condition latent was not prepared")
        sigma = t.reshape((-1,) + (1,) * (x.ndim - 1)).to(device=x.device, dtype=x.dtype)
        x_model = x.to(dtype=self.dtype)
        pred = self._predict_v(_comfy_latent_to_seedvr2(x_model), t)
        pred = _seedvr2_latent_to_comfy(pred).to(device=x.device, dtype=x.dtype)
        denoised = x - pred * sigma
        _ensure_finite(denoised, "DiT denoised")
        return denoised

    def _predict_v(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if model_management.is_device_cpu(x_t.device):
            raise RuntimeError(
                "SeedVR2 DiT received CPU latents. "
                "This indicates the model was not prepared on an accelerator device."
            )
        if x_t.device != self.device:
            raise RuntimeError(
                f"SeedVR2 DiT input is on {x_t.device}, but the model is registered on {self.device}. "
                "This indicates ComfyUI did not prepare the model on the expected load device."
            )
        try:
            model_dtype = next(self.diffusion_model.parameters()).dtype
        except StopIteration:
            model_dtype = self.dtype
        autocast_enabled = x_t.device.type == "cuda" and model_dtype != self.dtype
        cond = self.current_condition.to(device=x_t.device, dtype=x_t.dtype)
        vid = torch.cat([x_t, cond], dim=-1)
        vid_flat, vid_shape = self.na.flatten([vid])
        txt, txt_shape = self.na.flatten([self.text_pos.to(device=x_t.device, dtype=x_t.dtype)])
        timestep = (t[:1].to(device=x_t.device, dtype=torch.float32) * SEEDVR2_T).repeat(1)
        with torch.autocast(x_t.device.type, self.dtype, enabled=autocast_enabled):
            out = self.diffusion_model(
                vid=vid_flat,
                txt=txt,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                timestep=timestep,
            ).vid_sample
        out = self.na.unflatten(out, vid_shape)[0]
        _ensure_finite(out, "DiT forward")
        return out


class _SeedVR2LatentFormat:
    latent_channels = 16
    latent_dimensions = 3
    spacial_downscale_ratio = 8
    temporal_downscale_ratio = 4

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


class _SeedVR2FlowSampling(torch.nn.Module):
    sigma_min = torch.tensor(0.0)
    sigma_max = torch.tensor(1.0)

    def percent_to_sigma(self, percent: float) -> float:
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return 1.0 - float(percent)

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        return timestep

    def calculate_input(self, sigma: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        del sigma
        return noise

    def calculate_denoised(self, sigma: torch.Tensor, model_output: torch.Tensor, model_input: torch.Tensor) -> torch.Tensor:
        sigma = sigma.reshape(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma: torch.Tensor, noise: torch.Tensor, latent_image: torch.Tensor, max_denoise: bool = False) -> torch.Tensor:
        del max_denoise
        sigma = sigma.reshape(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return sigma * noise + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(self, sigma: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        del sigma
        return latent


class SeedVR2DiT:
    def __init__(
        self,
        path: str,
        model_name: str,
        load_device: torch.device,
        offload_device: torch.device,
        dtype: torch.dtype,
        attention_mode: str,
        debug: bool,
        script_dir: Path,
    ):
        load_device = _require_accelerator_device(load_device, "DiT")
        template = _infer_dit_template(path, model_name)
        LOG.info("SeedVR2 DiT template selected: %s for %s", template.upper(), os.path.basename(path))
        dit, na_module = _instantiate_dit(template, attention_mode)
        dit = _load_weights(dit, path, dtype=None, load_device=load_device)
        from .seedvr2_common import CompatibleDiT, validate_attention_mode

        requested_attention_mode = attention_mode
        attention_mode = validate_attention_mode(attention_mode, _LogShim(debug))
        if attention_mode != requested_attention_mode:
            LOG.info(
                "SeedVR2 attention backend %s resolved to %s.",
                requested_attention_mode,
                attention_mode,
            )
        else:
            LOG.info("SeedVR2 attention backend: %s", attention_mode)
        for module in dit.modules():
            if type(module).__name__ == "FlashAttentionVarlen":
                module.attention_mode = attention_mode
                module.compute_dtype = dtype
        dit = CompatibleDiT(dit, _LogShim(debug), compute_dtype=dtype, skip_conversion=False)
        text_pos = torch.load(script_dir / "pos_emb.pt", map_location="cpu", weights_only=True)
        text_neg = torch.load(script_dir / "neg_emb.pt", map_location="cpu", weights_only=True)
        model = SeedVR2ComfyModel(dit, na_module, text_pos, text_neg, dtype)
        model.to(offload_device)
        model.device = offload_device
        self.model = model
        self.patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=load_device,
            offload_device=offload_device,
            size=_model_size_bytes(model),
        )
        LOG.info(
            "SeedVR2 DiT patcher ready: load=%s, offload=%s, dtype=%s, size=%.2f MB",
            _device_label(load_device),
            _device_label(offload_device),
            dtype,
            _model_size_bytes(model) / (1024 * 1024),
        )

    def _load(self, latent: torch.Tensor) -> None:
        _require_accelerator_device(self.patcher.load_device, "DiT")
        noise_shape = _seedvr2_latent_to_comfy(latent).shape
        memory_required = self.patcher.memory_required(noise_shape)
        LOG.info(
            "SeedVR2 loading DiT: load=%s, offload=%s, memory_required=%.2f MB, latent_shape=%s",
            _device_label(self.patcher.load_device),
            _device_label(self.patcher.offload_device),
            memory_required / (1024 * 1024),
            tuple(noise_shape),
        )
        model_management.load_models_gpu(
            [self.patcher],
            memory_required=memory_required,
        )
        self.model.device = self.patcher.load_device
        _log_module_placement(self.model.diffusion_model, "DiT")

    def sample(self, latent: torch.Tensor, seed: int, *, show_pbar: bool = True) -> torch.Tensor:
        _seed_everything(seed)
        self._load(latent)
        latent = latent.to(device=self.patcher.load_device, dtype=self.model.dtype)
        base_noise = torch.randn_like(latent, dtype=self.model.dtype)
        condition = _seedvr2_condition(base_noise, latent)
        self.model.current_condition = condition
        noise = _seedvr2_latent_to_comfy(base_noise)
        latent_image = torch.zeros_like(noise)
        sigmas = torch.tensor([1.0, 0.0], device=self.patcher.load_device, dtype=torch.float32)
        sampler = comfy.samplers.sampler_object("euler")
        try:
            out = comfy.sample.sample_custom(
                self.patcher,
                noise,
                cfg=1.0,
                sampler=sampler,
                sigmas=sigmas,
                positive=[(None, {})],
                negative=[(None, {})],
                latent_image=latent_image,
                disable_pbar=not show_pbar,
                seed=seed,
            )
        finally:
            self.model.current_condition = None
        out = _comfy_latent_to_seedvr2(out.to(self.model.dtype))
        _ensure_finite(out, "DiT sample")
        return out

    def close(self) -> None:
        self.model.current_condition = None
        patcher = self.patcher
        if patcher is None:
            return
        self.patcher = None
        _unload_patcher(patcher)


class SeedVR2Pipeline:
    def __init__(
        self,
        *,
        model_dir: str,
        dit_model: str,
        vae_model: str,
        attention_backend: str = "sdpa",
        debug: bool = False,
    ):
        self.model_dir = Path(model_dir)
        self.dit_model = dit_model
        self.vae_model = vae_model
        self.dit_device = _require_accelerator_device(model_management.get_torch_device(), "DiT")
        self.vae_device = model_management.vae_device()
        self.offload_device = model_management.unet_offload_device()
        self.vae_offload_device = model_management.vae_offload_device()
        self.dtype = _compute_dtype(self.dit_device)
        LOG.info(
            "SeedVR2 devices: dit=%s, dit_offload=%s, vae=%s, vae_offload=%s, dtype=%s",
            _device_label(self.dit_device),
            _device_label(self.offload_device),
            _device_label(self.vae_device),
            _device_label(self.vae_offload_device),
            self.dtype,
        )
        script_dir = Path(__file__).resolve().parent / "seedvr2_assets"
        self.vae = SeedVR2VAE(str(self.model_dir / vae_model), self.vae_device, self.vae_offload_device, self.dtype)
        self.dit = SeedVR2DiT(
            str(self.model_dir / dit_model),
            dit_model,
            self.dit_device,
            self.offload_device,
            self.dtype,
            attention_backend,
            debug,
            script_dir,
        )

    def close(self) -> None:
        if self.dit is not None:
            self.dit.close()
            self.dit = None
        patcher = self.vae.patcher if self.vae is not None else None
        if patcher is None:
            return
        self.vae.patcher = None
        _unload_patcher(patcher)
        self.vae = None

    @torch.no_grad()
    def upscale(
        self,
        image: torch.Tensor,
        *,
        resolution: int,
        max_resolution: int,
        seed: int,
        batch_size: int,
        encode_tiled: bool,
        encode_tile_size: int,
        encode_tile_overlap: int,
        decode_tiled: bool,
        decode_tile_size: int,
        decode_tile_overlap: int,
    ) -> torch.Tensor:
        video, true_dims = _resize_normalize(image, resolution, max_resolution)
        batches = _split_batches(video, batch_size)
        pbar = comfy.utils.ProgressBar(max(1, len(batches) * 3))
        pbar.update_absolute(0)
        output_batches: list[torch.Tensor] = []
        original_lengths: list[int] = []
        for batch in batches:
            original_lengths.append(int(batch.shape[0]))
            latent = self.vae.encode(batch, seed, encode_tiled, encode_tile_size, encode_tile_overlap)
            pbar.update(1)
            upscaled = self.dit.sample(latent, seed, show_pbar=True)
            pbar.update(1)
            decoded = self.vae.decode(upscaled, decode_tiled, decode_tile_size, decode_tile_overlap)
            pbar.update(1)
            output_batches.append(decoded)
        pbar.update_absolute(pbar.total)
        out = _merge_batches(output_batches, original_lengths)
        out = out[: image.shape[0], :, : true_dims[0], : true_dims[1]]
        out = out.permute(0, 2, 3, 1).add(1.0).mul(0.5).clamp(0.0, 1.0)
        _ensure_finite(out, "final image")
        return out.to(torch.float32).cpu()


def _split_batches(video: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    frames = int(video.shape[0])
    if batch_size <= 0:
        batch_size = frames
    batch_size = max(1, ((batch_size - 1) // 4) * 4 + 1)
    if frames <= batch_size:
        return [video]
    batches = []
    for start in range(0, frames, batch_size):
        batch = video[start : min(start + batch_size, frames)]
        if batch.shape[0] == 0:
            continue
        batch, _ = _pad_4n1(batch)
        batches.append(batch)
    return batches


def _merge_batches(batches: list[torch.Tensor], original_lengths: list[int]) -> torch.Tensor:
    trimmed = [batch[:length] for batch, length in zip(batches, original_lengths)]
    return torch.cat(trimmed, dim=0)
