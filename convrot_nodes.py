from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import folder_paths
import torch
from safetensors import safe_open

import comfy.model_management
import comfy.sd
import comfy.utils

try:
    from .attention_backends import (
        apply_attention_backend,
        attention_backend_choices,
        normalize_attention_backend,
    )
except ImportError:
    from attention_backends import (
        apply_attention_backend,
        attention_backend_choices,
        normalize_attention_backend,
    )


LOG = logging.getLogger("comfyui-svdint4")
DIFFUSION_FOLDER_NAME = "diffusion_models"
CLIP_FOLDER_NAME = "text_encoders"
MODEL_EXTENSIONS = {".safetensors", ".sft"}
CLIP_TYPES = (
    "stable_diffusion",
    "stable_cascade",
    "sd3",
    "stable_audio",
    "mochi",
    "ltxv",
    "pixart",
    "cosmos",
    "lumina2",
    "wan",
    "hidream",
    "chroma",
    "ace",
    "omnigen2",
    "qwen_image",
    "hunyuan_image",
    "flux2",
    "ovis",
    "longcat_image",
    "cogvideox",
    "lens",
    "pixeldit",
    "ideogram4",
    "boogu",
    "krea2",
    "joyimage",
)
W4_FORMAT = "convrot_w4a4"
W8_FORMAT = "int8_tensorwise"


@dataclasses.dataclass(frozen=True)
class ConvRotSummary:
    w4a4: int = 0
    w4a8: int = 0
    w8a8: int = 0


def _params(config: dict, layer_name: str) -> dict:
    params = config.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"ConvRot layer {layer_name} has non-object params metadata")
    return params


def _config_value(config: dict, params: dict, name: str, default):
    return config[name] if name in config else params.get(name, default)


def _classify_config(config: dict, layer_name: str, force_int8_gemm: bool) -> tuple[str, str] | None:
    if not isinstance(config, dict):
        raise ValueError(f"Quantization metadata for layer {layer_name} must be an object")

    quant_format = config.get("format")
    params = _params(config, layer_name)
    if quant_format == W4_FORMAT:
        activation_dtype = _config_value(config, params, "linear_dtype", "int4")
        if activation_dtype not in {"int4", "int8"}:
            raise ValueError(
                f"ConvRot W4 layer {layer_name} declares unsupported linear_dtype={activation_dtype!r}; "
                "expected 'int4' or 'int8'"
            )
        return "w4", "int8" if force_int8_gemm else activation_dtype

    convrot = _config_value(config, params, "convrot", False)
    if not isinstance(convrot, bool):
        raise ValueError(f"Quantized layer {layer_name} has non-boolean convrot={convrot!r}")
    if not convrot:
        return None
    if quant_format != W8_FORMAT:
        raise ValueError(f"ConvRot layer {layer_name} uses unsupported weight format {quant_format!r}")

    activation_dtype = _config_value(config, params, "linear_dtype", "int8")
    if activation_dtype not in {"int4", "int8"}:
        raise ValueError(
            f"ConvRot W8 layer {layer_name} declares unsupported linear_dtype={activation_dtype!r}; "
            "expected 'int8'"
        )
    if activation_dtype != "int8" and not force_int8_gemm:
        raise ValueError(
            f"ConvRot W8 layer {layer_name} cannot use linear_dtype={activation_dtype!r}; "
            "W8 ConvRot supports INT8 activations only"
        )
    return "w8", "int8"


def _decode_quant_tensor(value: torch.Tensor, key: str) -> dict:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{key} must be a JSON byte tensor")
    try:
        config = json.loads(value.detach().cpu().numpy().tobytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{key} contains invalid quantization JSON") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{key} quantization JSON must be an object")
    return config


def _encode_quant_tensor(config: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)


def _convrot_skip_reason(model_path: str | Path) -> str | None:
    model_path = Path(model_path)
    if model_path.suffix.lower() not in MODEL_EXTENSIONS:
        return f"unsupported file extension {model_path.suffix!r}"

    try:
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            raw_header = metadata.get("_quantization_metadata")
            if raw_header is not None:
                header = json.loads(raw_header)
                if not isinstance(header, dict) or not isinstance(header.get("layers"), dict):
                    return "safetensors _quantization_metadata must contain a layers object"
                for layer_name, config in header["layers"].items():
                    if _classify_config(config, layer_name, True) is not None:
                        return None

            for key in handle.keys():
                if not key.endswith(".comfy_quant"):
                    continue
                config = _decode_quant_tensor(handle.get_tensor(key), key)
                if _classify_config(config, key[: -len(".comfy_quant")], True) is not None:
                    return None
    except Exception as exc:
        return f"could not read ConvRot metadata ({exc})"

    return "does not contain supported ConvRot quantization metadata"


def _convrot_model_names(folder_name: str) -> list[str]:
    names = []
    for name in folder_paths.get_filename_list(folder_name):
        if Path(name).suffix.lower() not in MODEL_EXTENSIONS:
            continue
        model_path = folder_paths.get_full_path(folder_name, name)
        if model_path is None:
            LOG.debug("Skipping ConvRot candidate %s: folder_paths could not resolve it", name)
            continue
        skip_reason = _convrot_skip_reason(model_path)
        if skip_reason is None:
            names.append(name)
        else:
            LOG.debug("Skipping ConvRot candidate %s: %s", name, skip_reason)
    return names


def _resolve_convrot_model_path(folder_name: str, model_name: str) -> str:
    model_path = folder_paths.get_full_path_or_raise(folder_name, model_name)
    skip_reason = _convrot_skip_reason(model_path)
    if skip_reason is not None:
        raise ValueError(f"{model_path} is not a supported ConvRot model: {skip_reason}")
    return model_path


def configure_convrot_activation(
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, str] | None,
    force_int8_gemm: bool,
) -> tuple[dict[str, str], ConvRotSummary]:
    if not isinstance(force_int8_gemm, bool):
        raise ValueError(f"force_int8_gemm must be boolean, got {force_int8_gemm!r}")

    metadata = dict(metadata or {})
    records: list[tuple[str, dict]] = []
    header_quantization = None
    raw_header = metadata.get("_quantization_metadata")
    if raw_header is not None:
        try:
            header_quantization = json.loads(raw_header)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Safetensors _quantization_metadata contains invalid JSON") from exc
        if not isinstance(header_quantization, dict) or not isinstance(header_quantization.get("layers"), dict):
            raise ValueError("Safetensors _quantization_metadata must contain a layers object")
        records.extend((name, config) for name, config in header_quantization["layers"].items())

    tensor_configs: dict[str, dict] = {}
    for key, value in state_dict.items():
        if not key.endswith(".comfy_quant"):
            continue
        layer_name = key[: -len(".comfy_quant")]
        config = _decode_quant_tensor(value, key)
        tensor_configs[key] = config
        records.append((layer_name, config))

    classified_records: list[tuple[str, dict, str, str]] = []
    layer_types: dict[str, tuple[str, str]] = {}
    for layer_name, config in records:
        classification = _classify_config(config, layer_name, force_int8_gemm)
        if classification is None:
            continue
        previous = layer_types.get(layer_name)
        if previous is not None:
            if previous != classification:
                raise ValueError(
                    f"Conflicting ConvRot metadata for layer {layer_name}: "
                    f"{previous[0]}/{previous[1]} versus {classification[0]}/{classification[1]}"
                )
        else:
            layer_types[layer_name] = classification
        classified_records.append((layer_name, config, *classification))

    if not classified_records:
        raise ValueError("The selected model does not contain supported ConvRot quantization metadata")

    if force_int8_gemm:
        for _, config, _, _ in classified_records:
            config["linear_dtype"] = "int8"

    if header_quantization is not None:
        metadata["_quantization_metadata"] = json.dumps(header_quantization)
    for key, config in tensor_configs.items():
        state_dict[key] = _encode_quant_tensor(config)

    summary = ConvRotSummary(
        w4a4=sum(1 for weight_dtype, act_dtype in layer_types.values() if (weight_dtype, act_dtype) == ("w4", "int4")),
        w4a8=sum(1 for weight_dtype, act_dtype in layer_types.values() if (weight_dtype, act_dtype) == ("w4", "int8")),
        w8a8=sum(1 for weight_dtype, act_dtype in layer_types.values() if (weight_dtype, act_dtype) == ("w8", "int8")),
    )
    return metadata, summary


def _summarize_convrot_modules(root: torch.nn.Module) -> ConvRotSummary:
    w4a4 = 0
    w4a8 = 0
    w8a8 = 0
    for _, module in root.named_modules():
        quant_format = getattr(module, "quant_format", None)
        weight = getattr(module, "weight", None)
        params = getattr(weight, "_params", None)
        if quant_format == W4_FORMAT:
            activation_dtype = getattr(params, "linear_dtype", None)
            if activation_dtype == "int4":
                w4a4 += 1
            elif activation_dtype == "int8":
                w4a8 += 1
            else:
                raise RuntimeError(
                    f"Loaded ConvRot W4 layer has unsupported linear_dtype={activation_dtype!r}; "
                    "update ComfyUI and comfy-kitchen"
                )
        elif quant_format == W8_FORMAT and getattr(params, "convrot", False):
            w8a8 += 1
    return ConvRotSummary(w4a4=w4a4, w4a8=w4a8, w8a8=w8a8)


def _loaded_convrot_summary(model) -> ConvRotSummary:
    return _summarize_convrot_modules(model.model)


def _loaded_convrot_clip_summary(clip) -> ConvRotSummary:
    return _summarize_convrot_modules(clip.cond_stage_model)


def _validate_runtime_support(expected: ConvRotSummary, device: torch.device | None = None) -> None:
    if expected.w4a8 == 0:
        return

    try:
        import comfy_kitchen
    except ImportError as exc:
        raise RuntimeError("ConvRot W4A8 requires comfy-kitchen") from exc

    cuda_backend = comfy_kitchen.list_backends().get("cuda", {})
    if not cuda_backend.get("available", False) or cuda_backend.get("disabled", False):
        reason = cuda_backend.get("unavailable_reason")
        detail = f": {reason}" if reason else ""
        raise RuntimeError(
            "ConvRot W4A8 requires the comfy-kitchen CUDA backend, but it is not enabled"
            f"{detail}. The eager ConvRot W4 path always computes A4, so this loader will not silently accept A8."
        )

    if device is None:
        device = comfy.model_management.get_torch_device()
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError(f"ConvRot W4A8 requires an NVIDIA CUDA load device, got {device}")
    capability = torch.cuda.get_device_capability(device)
    if capability < (7, 5):
        raise RuntimeError(
            f"ConvRot W4A8 requires NVIDIA Turing/sm75 or newer, got sm{capability[0]}{capability[1]}"
        )


def load_convrot_model(
    model_path: str | Path,
    force_int8_gemm: bool = False,
    attention_backend: str | None = "auto",
    *,
    disable_dynamic: bool = False,
):
    attention_backend = normalize_attention_backend(attention_backend)
    model_path = Path(model_path)
    state_dict, metadata = comfy.utils.load_torch_file(str(model_path), return_metadata=True)
    metadata, expected = configure_convrot_activation(state_dict, metadata, force_int8_gemm)
    _validate_runtime_support(expected)
    model = comfy.sd.load_diffusion_model_state_dict(
        state_dict,
        metadata=metadata,
        disable_dynamic=disable_dynamic,
    )
    if model is None:
        raise RuntimeError(f"ComfyUI could not detect a supported model config from {model_path}")

    loaded = _loaded_convrot_summary(model)
    if loaded != expected:
        raise RuntimeError(
            "ConvRot activation selection was not applied to every layer: "
            f"expected {expected}, loaded {loaded}. Update ComfyUI and comfy-kitchen."
        )

    LOG.info(
        "Loaded ConvRot model with force_int8_gemm=%s: W4A4=%d, W4A8=%d, W8A8=%d",
        force_int8_gemm,
        loaded.w4a4,
        loaded.w4a8,
        loaded.w8a8,
    )
    apply_attention_backend(model, attention_backend)
    model.cached_patcher_init = (
        load_convrot_model,
        (str(model_path), force_int8_gemm, attention_backend),
    )
    return model


def load_convrot_clip_patcher(
    model_path: str | Path,
    embedding_directory,
    clip_type,
    force_int8_gemm: bool,
    model_options: dict,
    disable_dynamic: bool = False,
):
    return load_convrot_clip(
        model_path,
        embedding_directory=embedding_directory,
        clip_type=clip_type,
        force_int8_gemm=force_int8_gemm,
        model_options=model_options,
        disable_dynamic=disable_dynamic,
    ).patcher


def load_convrot_clip(
    model_path: str | Path,
    *,
    embedding_directory=None,
    clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
    force_int8_gemm: bool = False,
    model_options: dict | None = None,
    disable_dynamic: bool = False,
):
    model_path = Path(model_path)
    state_dict, metadata = comfy.utils.load_torch_file(
        str(model_path),
        safe_load=True,
        return_metadata=True,
    )
    metadata, expected = configure_convrot_activation(state_dict, metadata, force_int8_gemm)

    model_options = dict(model_options or {})
    load_device = model_options.get("load_device", comfy.model_management.text_encoder_device())
    _validate_runtime_support(expected, load_device)

    state_dict, metadata = comfy.utils.convert_old_quants(state_dict, model_prefix="", metadata=metadata)
    model_options["quantization_metadata"] = {"mixed_ops": True}
    clip = comfy.sd.load_text_encoder_state_dicts(
        [state_dict],
        embedding_directory=embedding_directory,
        clip_type=clip_type,
        model_options=model_options,
        disable_dynamic=disable_dynamic,
    )

    loaded = _loaded_convrot_clip_summary(clip)
    if loaded != expected:
        raise RuntimeError(
            "ConvRot activation selection was not applied to every CLIP layer: "
            f"expected {expected}, loaded {loaded}. Check the CLIP type and update ComfyUI and comfy-kitchen."
        )

    LOG.info(
        "Loaded ConvRot CLIP with force_int8_gemm=%s: W4A4=%d, W4A8=%d, W8A8=%d",
        force_int8_gemm,
        loaded.w4a4,
        loaded.w4a8,
        loaded.w8a8,
    )
    clip.patcher.cached_patcher_init = (
        load_convrot_clip_patcher,
        (
            str(model_path),
            embedding_directory,
            clip_type,
            force_int8_gemm,
            model_options,
        ),
    )
    return clip


class ConvRotDiffusionModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (
                    _convrot_model_names(DIFFUSION_FOLDER_NAME),
                    {
                        "tooltip": (
                            "ConvRot DiT file from ComfyUI/models/diffusion_models. "
                            "Files without supported ConvRot quantization metadata are hidden."
                        )
                    },
                ),
                "force_int8_gemm": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "False follows each layer's activation format. "
                            "True forces INT8 GEMM activations."
                        ),
                    },
                ),
            },
            "optional": {
                "patch_attention": (
                    attention_backend_choices(),
                    {
                        "default": "auto",
                        "tooltip": (
                            "Select this ConvRot model's attention backend. "
                            "auto tries sage_attn, then flash_attn, then PyTorch SDPA."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_diffusion_model"
    CATEGORY = "SVDInt4/loaders"
    TITLE = "Load ConvRot DiT"

    def load_diffusion_model(
        self,
        unet_name: str,
        force_int8_gemm: bool = False,
        patch_attention: str = "auto",
    ):
        model_path = _resolve_convrot_model_path(DIFFUSION_FOLDER_NAME, unet_name)
        return (load_convrot_model(model_path, force_int8_gemm, attention_backend=patch_attention),)


class ConvRotCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (
                    _convrot_model_names(CLIP_FOLDER_NAME),
                    {
                        "tooltip": (
                            "ConvRot CLIP file from ComfyUI/models/text_encoders. "
                            "Files without supported ConvRot quantization metadata are hidden."
                        )
                    },
                ),
                "type": (CLIP_TYPES,),
                "force_int8_gemm": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "False follows each layer's activation format. "
                            "True forces INT8 GEMM activations."
                        ),
                    },
                ),
            },
            "optional": {
                "device": (("default", "cpu"), {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "SVDInt4/loaders"
    TITLE = "Load ConvRot CLIP"

    def load_clip(
        self,
        clip_name: str,
        type: str = "stable_diffusion",
        force_int8_gemm: bool = False,
        device: str = "default",
    ):
        if type not in CLIP_TYPES:
            raise ValueError(f"Unsupported ConvRot CLIP type {type!r}; expected one of {CLIP_TYPES}")
        if device not in {"default", "cpu"}:
            raise ValueError(f"Unsupported ConvRot CLIP device {device!r}; expected 'default' or 'cpu'")
        clip_type = getattr(comfy.sd.CLIPType, type.upper())
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        model_path = _resolve_convrot_model_path(CLIP_FOLDER_NAME, clip_name)
        clip = load_convrot_clip(
            model_path,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            force_int8_gemm=force_int8_gemm,
            model_options=model_options,
        )
        return (clip,)
