from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.sample
import comfy.samplers

from .loader import build_loader_state_dict, is_svdint4_file


LOG = logging.getLogger("comfyui-svdint4")

SEEDVR2_SCALING_FACTOR = 0.9152
SEEDVR2_T = 1000.0


class _LogShim:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def log(self, message: str, level: str = "INFO", **_: Any) -> None:
        if self.enabled or level in {"WARNING", "ERROR"}:
            getattr(LOG, level.lower(), LOG.info)("SeedVR2 native: %s", message)

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
        raise ValueError(f"SeedVR2 native expects IMAGE [frames,h,w,3/4], got {tuple(images.shape)}")
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
        raise ValueError(f"SeedVR2 native currently samples one 4n+1 window at a time, got {tuple(latent.shape)}")
    return latent.squeeze(0).permute(1, 2, 3, 0).contiguous()


def _seedvr2_condition(latent: torch.Tensor, latent_blur: torch.Tensor) -> torch.Tensor:
    cond = torch.zeros((*latent.shape[:-1], latent.shape[-1] + 1), device=latent.device, dtype=latent.dtype)
    cond[..., :-1] = latent_blur
    cond[..., -1:] = 1.0
    return cond


def _flow_add_noise(x: torch.Tensor, aug_noise: torch.Tensor, noise_scale: float) -> torch.Tensor:
    if noise_scale <= 0.0:
        return x
    t = torch.tensor([SEEDVR2_T * noise_scale], device=x.device, dtype=x.dtype)
    shape = torch.tensor(x.shape[:-1], device=x.device, dtype=torch.long)[None]
    t = _timestep_transform(t, shape)
    coeff = t.reshape((1,) * (x.ndim - 1) + (1,)) / SEEDVR2_T
    return (1.0 - coeff) * x + coeff * aug_noise


def _timestep_transform(timesteps: torch.Tensor, latent_shapes: torch.Tensor) -> torch.Tensor:
    vt = 4
    vs = 8
    frames = (latent_shapes[:, 0] - 1) * vt + 1
    heights = latent_shapes[:, 1] * vs
    widths = latent_shapes[:, 2] * vs

    def lin(x1: float, y1: float, x2: float, y2: float, x: torch.Tensor) -> torch.Tensor:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m * x + b

    img_shift = lin(256 * 256, 1.0, 1024 * 1024, 3.2, heights * widths)
    vid_shift = lin(256 * 256 * 37, 1.0, 1280 * 720 * 145, 5.0, heights * widths * frames)
    shift = torch.where(frames > 1, vid_shift, img_shift).to(timesteps.dtype)
    t = timesteps / SEEDVR2_T
    return (shift * t / (1 + (shift - 1) * t)) * SEEDVR2_T


def _model_size_bytes(model: torch.nn.Module) -> int:
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor is not None:
            total += tensor.nelement() * tensor.element_size()
    return total


def _instantiate_dit(model_name: str, attention_mode: str) -> tuple[torch.nn.Module, Any]:
    lower = model_name.lower()
    if "7b" in lower:
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
        f"SeedVR2NativeSVDInt4Linear_{id(packed_layer_tensors):x}",
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


def _parent_module(model: torch.nn.Module, name: str) -> tuple[torch.nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _load_weights(model: torch.nn.Module, path: str, *, dtype: torch.dtype | None = None) -> torch.nn.Module:
    if is_svdint4_file(path):
        state, _metadata, packed, w4 = build_loader_state_dict(path)
        model = _replace_svdint4_linears(model, packed, w4)
    else:
        state = load_safetensors_file(path, device="cpu")
    if dtype is not None:
        for key, value in list(state.items()):
            if torch.is_tensor(value) and value.is_floating_point():
                state[key] = value.to(dtype=dtype)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing:
        LOG.warning("SeedVR2 native missing %d key(s) while loading %s", len(missing), os.path.basename(path))
    if unexpected:
        LOG.warning("SeedVR2 native unexpected %d key(s) while loading %s", len(unexpected), os.path.basename(path))
    return model


class SeedVR2VAE:
    def __init__(self, path: str, load_device: torch.device, offload_device: torch.device, dtype: torch.dtype):
        core = _load_weights(_instantiate_vae(), path, dtype=dtype)
        core.to(offload_device)
        self.model = _PatchableVAE(core, offload_device)
        self.dtype = dtype
        self.load_device = load_device
        self.offload_device = offload_device
        self.patcher = comfy.model_patcher.ModelPatcher(
            self.model,
            load_device=load_device,
            offload_device=offload_device,
            size=_model_size_bytes(self.model),
        )

    def _load(self, memory_required: int) -> None:
        model_management.load_models_gpu([self.patcher], memory_required=memory_required)

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
        return decoded.squeeze(0).permute(1, 0, 2, 3).contiguous()


class _PatchableVAE(torch.nn.Module):
    def __init__(self, core: torch.nn.Module, device: torch.device):
        super().__init__()
        self.core = core
        self.device = device

    def encode(self, *args, **kwargs):
        return self.core.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
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
            raise RuntimeError("SeedVR2 native condition latent was not prepared")
        sigma = t.reshape((-1,) + (1,) * (x.ndim - 1)).to(device=x.device, dtype=x.dtype)
        x_model = x.to(dtype=self.dtype)
        pred = self._predict_v(_comfy_latent_to_seedvr2(x_model), t)
        pred = _seedvr2_latent_to_comfy(pred).to(device=x.device, dtype=x.dtype)
        return x - pred * sigma

    def _predict_v(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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
        return self.na.unflatten(out, vid_shape)[0]


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
        dit, na_module = _instantiate_dit(model_name, attention_mode)
        dit = _load_weights(dit, path, dtype=None)
        from .seedvr2_common import CompatibleDiT, validate_attention_mode

        attention_mode = validate_attention_mode(attention_mode, _LogShim(debug))
        for module in dit.modules():
            if type(module).__name__ == "FlashAttentionVarlen":
                module.attention_mode = attention_mode
                module.compute_dtype = dtype
        dit = CompatibleDiT(dit, _LogShim(debug), compute_dtype=dtype, skip_conversion=False)
        text_pos = torch.load(script_dir / "pos_emb.pt", map_location="cpu", weights_only=True)
        text_neg = torch.load(script_dir / "neg_emb.pt", map_location="cpu", weights_only=True)
        model = SeedVR2ComfyModel(dit, na_module, text_pos, text_neg, dtype)
        model.to(offload_device)
        self.model = model
        self.patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=load_device,
            offload_device=offload_device,
            size=_model_size_bytes(model),
        )

    def sample(self, latent: torch.Tensor, seed: int, latent_noise_scale: float) -> torch.Tensor:
        _seed_everything(seed)
        latent = latent.to(device=self.patcher.load_device, dtype=self.model.dtype)
        base_noise = torch.randn_like(latent, dtype=self.model.dtype)
        aug_noise = base_noise * 0.1 + torch.randn_like(base_noise) * 0.05
        latent_blur = _flow_add_noise(latent, aug_noise, latent_noise_scale)
        condition = _seedvr2_condition(base_noise, latent_blur)
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
                disable_pbar=True,
                seed=seed,
            )
        finally:
            self.model.current_condition = None
        return _comfy_latent_to_seedvr2(out.to(self.model.dtype))

    def close(self) -> None:
        self.model.current_condition = None
        patcher = self.patcher
        if patcher is None:
            return
        self.patcher = None
        try:
            patcher.detach()
        except Exception:
            LOG.debug("SeedVR2 native DiT patcher detach failed", exc_info=True)


class SeedVR2NativePipeline:
    def __init__(
        self,
        *,
        model_dir: str,
        dit_model: str,
        vae_model: str,
        dit_device: str,
        vae_device: str,
        attention_mode: str,
        debug: bool = False,
    ):
        self.model_dir = Path(model_dir)
        self.dit_model = dit_model
        self.vae_model = vae_model
        self.dit_device = torch.device(dit_device)
        self.vae_device = torch.device(vae_device)
        self.offload_device = model_management.unet_offload_device()
        self.dtype = _compute_dtype(self.dit_device)
        script_dir = Path(__file__).resolve().parent / "seedvr2_assets"
        self.vae = SeedVR2VAE(str(self.model_dir / vae_model), self.vae_device, self.offload_device, self.dtype)
        self.dit = SeedVR2DiT(
            str(self.model_dir / dit_model),
            dit_model,
            self.dit_device,
            self.offload_device,
            self.dtype,
            attention_mode,
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
        try:
            patcher.detach()
        except Exception:
            LOG.debug("SeedVR2 native VAE patcher detach failed", exc_info=True)
        finally:
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
        temporal_overlap: int,
        encode_tiled: bool,
        encode_tile_size: int,
        encode_tile_overlap: int,
        decode_tiled: bool,
        decode_tile_size: int,
        decode_tile_overlap: int,
        input_noise_scale: float,
        latent_noise_scale: float,
    ) -> torch.Tensor:
        del temporal_overlap
        video, true_dims = _resize_normalize(image, resolution, max_resolution)
        if input_noise_scale > 0:
            noise = torch.randn_like(video) * 0.05
            blend = float(input_noise_scale) * 0.5
            video = video * (1.0 - blend) + (video + noise) * blend
        batches = _split_batches(video, batch_size)
        output_batches: list[torch.Tensor] = []
        original_lengths: list[int] = []
        for batch in batches:
            original_lengths.append(int(batch.shape[0]))
            latent = self.vae.encode(batch, seed, encode_tiled, encode_tile_size, encode_tile_overlap)
            upscaled = self.dit.sample(latent, seed, latent_noise_scale)
            decoded = self.vae.decode(upscaled, decode_tiled, decode_tile_size, decode_tile_overlap)
            output_batches.append(decoded)
        out = _merge_batches(output_batches, original_lengths)
        out = out[: image.shape[0], :, : true_dims[0], : true_dims[1]]
        return out.permute(0, 2, 3, 1).add(1.0).mul(0.5).clamp(0.0, 1.0).to(torch.float32).cpu()


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
