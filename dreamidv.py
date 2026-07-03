from __future__ import annotations

import json
import logging
from pathlib import Path

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
import torch
from safetensors import safe_open


LOG = logging.getLogger("comfyui-svdint4")
FOLDER_NAME = "diffusion_models"
DREAMIDV_MODEL_EXTENSIONS = {".safetensors", ".sft", ".pth", ".pt", ".ckpt"}
DREAMIDV_MODEL_DIR_NAME = "DreamID-V"
DREAMIDV_CONTEXT_PATH = Path(__file__).resolve().parent / "assets" / "dreamidv" / "context.pth"
DREAMIDV_TEXT_LEN = 512
DREAMIDV_TRANSFORMER_CONFIG = {
    # DreamID-V uses Wan's reference_latent/ref_conv path, not CLIP image features.
    # Keep the base Wan block type t2v so ComfyUI does not instantiate unused img_emb weights.
    "model_type": "t2v",
    "patch_size": [1, 2, 2],
    "dim": 1536,
    "ffn_dim": 8960,
    "freq_dim": 256,
    "in_dim": 48,
    "num_heads": 12,
    "num_layers": 30,
    "window_size": [-1, -1],
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
    "in_dim_ref_conv": 16,
}
SUPPORTED_FORMATS = {"svdint4-dit-single-v2"}
_DREAMIDV_CONTEXT_CACHE: torch.Tensor | None = None


def _dreamidv_model_root() -> Path:
    return Path(folder_paths.models_dir) / DREAMIDV_MODEL_DIR_NAME


def _dreamidv_model_names() -> list[str]:
    names: set[str] = set()
    root = _dreamidv_model_root()
    if root.is_dir():
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in DREAMIDV_MODEL_EXTENSIONS:
                names.add(f"{DREAMIDV_MODEL_DIR_NAME}/{path.relative_to(root).as_posix()}")
    for name in folder_paths.get_filename_list(FOLDER_NAME):
        if Path(name).suffix.lower() not in DREAMIDV_MODEL_EXTENSIONS:
            continue
        path = folder_paths.get_full_path(FOLDER_NAME, name)
        if path is not None and _is_dreamidv_svdint4_file(path):
            names.add(f"{FOLDER_NAME}/{name}")
    return sorted(names)


def _is_svdint4_file(model_path: str | Path) -> bool:
    try:
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            return (handle.metadata() or {}).get("format") in SUPPORTED_FORMATS
    except Exception:
        return False


def _is_dreamidv_svdint4_file(model_path: str | Path) -> bool:
    try:
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception:
        return False
    if metadata.get("format") not in SUPPORTED_FORMATS:
        return False
    haystack = " ".join(
        str(metadata.get(key, ""))
        for key in ("model_family", "model_name", "model_variant", "source")
    ).lower()
    return "dreamid" in haystack or "dream_id" in haystack or "swapface" in haystack


def _resolve_dreamidv_model_path(model_name: str) -> str:
    if model_name.startswith(f"{DREAMIDV_MODEL_DIR_NAME}/"):
        path = _dreamidv_model_root() / model_name[len(DREAMIDV_MODEL_DIR_NAME) + 1:]
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"DreamID-V model not found: {path}")
    if model_name.startswith(f"{FOLDER_NAME}/"):
        return folder_paths.get_full_path_or_raise(FOLDER_NAME, model_name[len(FOLDER_NAME) + 1:])

    dreamidv_path = _dreamidv_model_root() / model_name
    if dreamidv_path.is_file():
        return str(dreamidv_path)
    return folder_paths.get_full_path_or_raise(FOLDER_NAME, model_name)


def _unwrap_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a DreamID-V state dict, got {type(state_dict).__name__}.")
    for key in ("state_dict", "model", "module"):
        value = state_dict.get(key)
        if isinstance(value, dict):
            return value
    return state_dict


def _find_state_key(state_dict: dict, suffix: str) -> str | None:
    if suffix in state_dict:
        return suffix
    for key in state_dict:
        if key.endswith(suffix):
            return key
    return None


def _validate_dreamidv_state_dict(state_dict: dict, model_path: str | Path) -> None:
    patch_key = _find_state_key(state_dict, "patch_embedding.weight")
    ref_key = _find_state_key(state_dict, "ref_conv.weight")
    layer_key = _find_state_key(state_dict, "blocks.29.ffn.2.weight")
    if patch_key is None or ref_key is None or layer_key is None:
        raise ValueError(
            f"{model_path} does not look like a DreamID-V Faster DiT checkpoint. "
            "Expected patch_embedding, ref_conv, and 30 Wan transformer blocks."
        )
    patch_shape = tuple(state_dict[patch_key].shape)
    ref_shape = tuple(state_dict[ref_key].shape)
    if len(patch_shape) < 2 or patch_shape[1] != DREAMIDV_TRANSFORMER_CONFIG["in_dim"]:
        raise ValueError(f"DreamID-V patch_embedding input channels must be 48, got {patch_shape}.")
    if len(ref_shape) < 2 or ref_shape[1] != DREAMIDV_TRANSFORMER_CONFIG["in_dim_ref_conv"]:
        raise ValueError(f"DreamID-V ref_conv input channels must be 16, got {ref_shape}.")


def _dreamidv_metadata(metadata: dict | None) -> dict:
    metadata = dict(metadata or {})
    config = {}
    if metadata.get("config"):
        try:
            config = json.loads(metadata["config"])
        except Exception:
            LOG.warning("Ignoring invalid DreamID-V checkpoint config metadata.")
            config = {}
    transformer = dict(config.get("transformer", {}))
    transformer.update(DREAMIDV_TRANSFORMER_CONFIG)
    config["transformer"] = transformer
    metadata["config"] = json.dumps(config)
    return metadata


def _load_dreamidv_dense_model(model_path: str):
    state_dict, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)
    state_dict = _unwrap_state_dict(state_dict)
    _validate_dreamidv_state_dict(state_dict, model_path)
    model = comfy.sd.load_diffusion_model_state_dict(
        state_dict,
        metadata=_dreamidv_metadata(metadata),
        disable_dynamic=False,
    )
    if model is None:
        raise RuntimeError(f"Could not load DreamID-V DiT checkpoint: {model_path}")
    model.cached_patcher_init = (_load_dreamidv_dense_model, (model_path,))
    return model


def _load_dreamidv_context() -> torch.Tensor:
    global _DREAMIDV_CONTEXT_CACHE
    if _DREAMIDV_CONTEXT_CACHE is None:
        if not DREAMIDV_CONTEXT_PATH.is_file():
            raise FileNotFoundError(f"DreamID-V context file is missing: {DREAMIDV_CONTEXT_PATH}")
        context = torch.load(DREAMIDV_CONTEXT_PATH, map_location="cpu")
        if not isinstance(context, (list, tuple)) or len(context) != 1 or not torch.is_tensor(context[0]):
            raise ValueError(f"Unexpected DreamID-V context format in {DREAMIDV_CONTEXT_PATH}.")
        tensor = context[0].detach().cpu()
        if tensor.ndim != 2 or tensor.shape[-1] != 4096:
            raise ValueError(f"DreamID-V context must have shape [tokens, 4096], got {tuple(tensor.shape)}.")
        if tensor.shape[0] > DREAMIDV_TEXT_LEN:
            raise ValueError(
                f"DreamID-V context has {tensor.shape[0]} tokens, which exceeds text_len={DREAMIDV_TEXT_LEN}."
            )
        if tensor.shape[0] < DREAMIDV_TEXT_LEN:
            tensor = torch.cat(
                (
                    tensor,
                    torch.zeros(
                        (DREAMIDV_TEXT_LEN - tensor.shape[0], tensor.shape[1]),
                        dtype=tensor.dtype,
                    ),
                ),
                dim=0,
            )
        _DREAMIDV_CONTEXT_CACHE = tensor.unsqueeze(0)
    return _DREAMIDV_CONTEXT_CACHE


def _is_wan_frame_count(frame_count: int) -> bool:
    return frame_count >= 1 and (frame_count - 1) % 4 == 0


def _fit_frames(image: torch.Tensor, length: int, name: str) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"{name} must be an IMAGE tensor shaped [frames, height, width, channels], got {tuple(image.shape)}.")
    if image.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one frame.")
    if image.shape[0] >= length:
        return image[:length]
    pad = image[-1:].repeat(length - image.shape[0], 1, 1, 1)
    return torch.cat((image, pad), dim=0)


def _fit_mask(mask: torch.Tensor, length: int) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"face_mask must be a MASK tensor shaped [frames, height, width], got {tuple(mask.shape)}.")
    if mask.shape[0] < 1:
        raise ValueError("face_mask must contain at least one frame.")
    if mask.shape[0] >= length:
        return mask[:length]
    pad = mask[-1:].repeat(length - mask.shape[0], 1, 1)
    return torch.cat((mask, pad), dim=0)


def _resize_image(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)


def _encode_vae_raw(
    vae,
    pixels: torch.Tensor,
    *,
    tiled: bool = False,
    tile_size: int = 256,
    tile_overlap: int = 64,
    temporal_size: int = 64,
    temporal_overlap: int = 8,
) -> torch.Tensor:
    vae.throw_exception_if_invalid()
    pixel_samples = vae.vae_encode_crop_pixels(pixels).movedim(-1, 1)
    if vae.latent_dim == 3 and pixel_samples.ndim < 5:
        if not vae.not_video:
            pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)
        else:
            pixel_samples = pixel_samples.unsqueeze(2)

    with comfy.model_management.cuda_device_context(vae.device):
        memory_used = vae.memory_used_encode(pixel_samples.shape, vae.vae_dtype)
        comfy.model_management.load_models_gpu(
            [vae.patcher],
            memory_required=memory_used,
            force_full_load=vae.disable_offload,
        )
        if tiled and vae.latent_dim == 3:
            encode_fn = lambda a: vae.first_stage_model.encode(a.to(vae.vae_dtype).to(vae.device)).to(  # noqa: E731
                dtype=vae.vae_output_dtype()
            )
            return comfy.utils.tiled_scale_multidim(
                pixel_samples,
                encode_fn,
                tile=(max(1, temporal_size), tile_size, tile_size),
                overlap=(max(1, temporal_overlap), tile_overlap, tile_overlap),
                upscale_amount=vae.downscale_ratio,
                out_channels=vae.latent_channels,
                downscale=True,
                index_formulas=vae.downscale_index_formula,
                output_device=vae.output_device,
            )

        free_memory = vae.patcher.get_free_memory(vae.device)
        batch_number = max(1, int(free_memory / max(1, memory_used)))
        samples = None
        for index in range(0, pixel_samples.shape[0], batch_number):
            pixels_in = pixel_samples[index:index + batch_number].to(device=vae.device, dtype=vae.vae_dtype)
            if getattr(vae.first_stage_model, "comfy_has_chunked_io", False):
                out = vae.first_stage_model.encode(pixels_in, device=vae.device)
            else:
                out = vae.first_stage_model.encode(pixels_in)
            out = out.to(vae.output_device).to(dtype=vae.vae_output_dtype())
            if samples is None:
                samples = torch.empty(
                    (pixel_samples.shape[0],) + tuple(out.shape[1:]),
                    device=vae.output_device,
                    dtype=vae.vae_output_dtype(),
                )
            samples[index:index + batch_number] = out
    return samples


class DreamIDVDiTLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit_name": (
                    _dreamidv_model_names(),
                    {
                        "tooltip": (
                            "DreamID-V Faster DiT checkpoint. Dense .pth/.safetensors files can live in "
                            "ComfyUI/models/DreamID-V; SVDInt4 single-file safetensors can live there or "
                            "in ComfyUI/models/diffusion_models."
                        )
                    },
                ),
                "backend": (
                    ["auto", "dense", "svdint4"],
                    {"default": "auto", "tooltip": "auto detects SVDInt4 safetensors metadata; .pth loads as dense."},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_dit"
    CATEGORY = "SVDInt4/loaders"
    TITLE = "Load DreamID-V DiT"

    def load_dit(self, dit_name: str, backend: str):
        model_path = _resolve_dreamidv_model_path(dit_name)
        is_svdint4 = _is_svdint4_file(model_path)
        if backend == "auto":
            backend = "svdint4" if is_svdint4 else "dense"
        if backend == "svdint4":
            if not is_svdint4:
                raise ValueError(f"{model_path} is not an SVDInt4 single-file safetensors checkpoint.")
            from .loader import load_svdint4_model

            return (load_svdint4_model(model_path),)
        if is_svdint4:
            raise ValueError(f"{model_path} is an SVDInt4 checkpoint; use backend=auto or backend=svdint4.")
        return (_load_dreamidv_dense_model(model_path),)


class DreamIDVConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "source_video": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "face_mask": ("MASK",),
                "width": (
                    "INT",
                    {
                        "default": 832,
                        "min": 16,
                        "max": 16384,
                        "step": 16,
                        "tooltip": "Output/model width. Source, reference, and mask inputs are resized to this size.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 480,
                        "min": 16,
                        "max": 16384,
                        "step": 16,
                        "tooltip": "Output/model height. Source, reference, and mask inputs are resized to this size.",
                    },
                ),
                "length": (
                    "INT",
                    {
                        "default": 81,
                        "min": 1,
                        "max": 16385,
                        "step": 4,
                        "tooltip": "Real frame count. Wan video VAEs require 4*n+1 frames.",
                    },
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "invert_mask": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable when the incoming MASK uses black for the edited face area.",
                    },
                ),
                "vae_tiling": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use tiled VAE encode for the raw mask latent path. Source/reference use ComfyUI VAE encode.",
                    },
                ),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "advanced": True}),
                "tile_overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32, "advanced": True}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "advanced": True}),
                "temporal_overlap": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1, "advanced": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "apply"
    CATEGORY = "SVDInt4/conditioning"
    TITLE = "DreamID-V Conditioning"

    def apply(
        self,
        vae,
        source_video,
        reference_image,
        face_mask,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        invert_mask: bool,
        vae_tiling: bool,
        tile_size: int,
        tile_overlap: int,
        temporal_size: int,
        temporal_overlap: int,
    ):
        width = int(width)
        height = int(height)
        length = int(length)
        batch_size = int(batch_size)
        if not _is_wan_frame_count(length):
            raise ValueError(f"DreamID-V length must be 4*n+1 real frames; got {length}.")
        if width <= 0 or height <= 0:
            raise ValueError(f"DreamID-V width/height must be positive; got {width}x{height}.")
        if vae_tiling:
            if tile_size <= 0:
                raise ValueError("tile_size must be positive.")
            if tile_overlap < 0 or tile_overlap >= tile_size:
                raise ValueError(f"tile_overlap must be non-negative and smaller than tile_size; got {tile_overlap}/{tile_size}.")
            if temporal_size <= 0:
                raise ValueError("temporal_size must be positive.")
            if temporal_overlap < 0 or temporal_overlap >= temporal_size:
                raise ValueError(
                    f"temporal_overlap must be non-negative and smaller than temporal_size; got {temporal_overlap}/{temporal_size}."
                )

        spacial_scale = vae.spacial_compression_encode()
        if width % spacial_scale != 0 or height % spacial_scale != 0:
            raise ValueError(f"DreamID-V width/height must be divisible by VAE scale {spacial_scale}; got {width}x{height}.")

        source = _resize_image(_fit_frames(source_video, length, "source_video")[:, :, :, :3], width, height)
        reference = _resize_image(_fit_frames(reference_image, 1, "reference_image")[:, :, :, :3], width, height)
        mask = _fit_mask(face_mask, length).float().clamp_(0.0, 1.0)
        mask = comfy.utils.common_upscale(mask.unsqueeze(1), width, height, "nearest-exact", "center").squeeze(1).clamp_(0.0, 1.0)
        if invert_mask:
            mask = 1.0 - mask
        mask_rgb = mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        source_latent = vae.encode(source)
        reference_latent = vae.encode(reference)
        mask_latent = _encode_vae_raw(
            vae,
            mask_rgb,
            tiled=bool(vae_tiling),
            tile_size=int(tile_size),
            tile_overlap=int(tile_overlap),
            temporal_size=int(temporal_size),
            temporal_overlap=int(temporal_overlap),
        )

        latent_t = ((length - 1) // 4) + 1
        if source_latent.shape[2] != latent_t:
            raise ValueError(f"Source VAE latent length mismatch: expected {latent_t}, got {source_latent.shape[2]}.")
        if mask_latent.shape != source_latent.shape:
            raise ValueError(
                "Mask VAE latent shape must match source VAE latent shape; "
                f"got mask={tuple(mask_latent.shape)} source={tuple(source_latent.shape)}."
            )

        concat_latent = torch.cat((source_latent, mask_latent), dim=1)
        concat_latent = comfy.utils.resize_to_batch_size(concat_latent, batch_size)
        reference_latent = comfy.utils.resize_to_batch_size(reference_latent, batch_size)

        latent = torch.zeros(
            [
                batch_size,
                vae.latent_channels,
                latent_t,
                height // spacial_scale,
                width // spacial_scale,
            ],
            device=comfy.model_management.intermediate_device(),
            dtype=source_latent.dtype,
        )

        context = _load_dreamidv_context().to(device=comfy.model_management.intermediate_device())
        positive = [[context, {"concat_latent_image": concat_latent, "reference_latents": [reference_latent]}]]
        negative = [[context, {"concat_latent_image": concat_latent, "reference_latents": [torch.zeros_like(reference_latent)]}]]
        return (positive, negative, {"samples": latent})
