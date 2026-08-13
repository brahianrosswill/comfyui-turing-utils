"""ConvRot metadata, model loading services, and loader nodes."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import struct
import sys
from pathlib import Path

import folder_paths
import torch

import comfy.model_detection
import comfy.model_management
import comfy.sd
import comfy.utils

from ..attention import (
    apply_attention_backend,
    normalize_attention_backend,
)
from ..adapters.registry import apply_model_adapters
from ..precision import (
    normalize_turing_convrot_weight_dtypes,
    prepare_turing_runtime,
    select_compute_dtype,
)


LOG = logging.getLogger("comfyui-turing-utils")
DIFFUSION_FOLDER_NAME = "diffusion_models"
CLIP_FOLDER_NAME = "text_encoders"
MODEL_EXTENSIONS = {".safetensors", ".sft"}
W4_FORMAT = "convrot_w4a4"
CODEBOOK_W4_FORMAT = "asym_w4a8_int8"
W8_FORMAT = "int8_tensorwise"
LEGACY_W8_FORMAT = "int8_rowwise"
MAX_SAFETENSORS_HEADER_SIZE = 128 * 1024 * 1024
MAX_QUANT_CONFIG_SIZE = 1024 * 1024


@dataclasses.dataclass(frozen=True)
class ConvRotSummary:
    w4a4: int = 0
    w4a8: int = 0
    codebook_w4a8: int = 0
    w8a8: int = 0


def _official_clip_loader_inputs() -> dict:
    comfy_nodes = sys.modules.get("nodes")
    if comfy_nodes is None or not hasattr(comfy_nodes, "CLIPLoader"):
        raise RuntimeError("ComfyUI's official CLIPLoader is not available")
    inputs = comfy_nodes.CLIPLoader.INPUT_TYPES()
    if not isinstance(inputs, dict) or not isinstance(inputs.get("required"), dict):
        raise RuntimeError("ComfyUI's official CLIPLoader returned invalid input definitions")
    return inputs


def _official_clip_types() -> tuple[str, ...]:
    required = _official_clip_loader_inputs()["required"]
    if "type" not in required or not required["type"]:
        raise RuntimeError("ComfyUI's official CLIPLoader does not expose a type input")
    choices = required["type"][0]
    return tuple(choices)


def _params(config: dict, layer_name: str) -> dict:
    params = config.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"ConvRot layer {layer_name} has non-object params metadata")
    return params


def _config_value(config: dict, params: dict, name: str, default):
    return config[name] if name in config else params.get(name, default)


def _normalize_legacy_config(config: dict, layer_name: str) -> None:
    params = _params(config, layer_name)
    quant_format = config.get("format")
    convrot = _config_value(config, params, "convrot", False)
    per_row = _config_value(config, params, "per_row", False)
    if (quant_format is None or quant_format == LEGACY_W8_FORMAT) and convrot is True and per_row is True:
        config["format"] = W8_FORMAT


def _classify_config(config: dict, layer_name: str, force_int8_gemm: bool) -> tuple[str, str] | None:
    if not isinstance(config, dict):
        raise ValueError(f"Quantization metadata for layer {layer_name} must be an object")

    _normalize_legacy_config(config, layer_name)
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

    if quant_format == CODEBOOK_W4_FORMAT:
        convrot = _config_value(config, params, "convrot", True)
        if convrot is not True:
            raise ValueError(
                f"Grouped-codebook W4A8 layer {layer_name} must declare convrot=true"
            )
        group_size = _config_value(config, params, "group_size", 16)
        convrot_groupsize = _config_value(config, params, "convrot_groupsize", 256)
        if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 4:
            raise ValueError(
                f"Grouped-codebook W4A8 layer {layer_name} has invalid group_size={group_size!r}"
            )
        if convrot_groupsize != 256:
            raise ValueError(
                f"Grouped-codebook W4A8 layer {layer_name} requires convrot_groupsize=256"
            )
        return "w4_codebook", "int8"

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


def _read_safetensors_header(model_path: Path) -> tuple[dict, int, int]:
    file_size = model_path.stat().st_size
    with model_path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError("file is too short to contain a safetensors header")
        header_size = struct.unpack("<Q", raw_length)[0]
        if header_size == 0 or header_size > MAX_SAFETENSORS_HEADER_SIZE:
            raise ValueError(f"invalid safetensors header size {header_size}")
        data_start = 8 + header_size
        if data_start > file_size:
            raise ValueError(
                f"safetensors header ends at byte {data_start}, beyond file size {file_size}"
            )
        raw_header = handle.read(header_size)

    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("safetensors header contains invalid JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    return header, data_start, file_size


def _read_quant_config(
    handle,
    tensor_info: dict,
    key: str,
    data_start: int,
    file_size: int,
) -> dict:
    if not isinstance(tensor_info, dict):
        raise ValueError(f"{key} has an invalid safetensors tensor descriptor")
    if tensor_info.get("dtype") != "U8":
        raise ValueError(f"{key} must use U8 storage")

    shape = tensor_info.get("shape")
    offsets = tensor_info.get("data_offsets")
    if (
        not isinstance(shape, list)
        or any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or any(not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets)
    ):
        raise ValueError(f"{key} has invalid shape or data offsets")

    relative_start, relative_end = offsets
    config_size = relative_end - relative_start
    if config_size != math.prod(shape):
        raise ValueError(f"{key} byte length does not match its U8 shape")
    if config_size > MAX_QUANT_CONFIG_SIZE:
        raise ValueError(f"{key} is too large to be quantization JSON")
    absolute_start = data_start + relative_start
    absolute_end = data_start + relative_end
    if relative_start < 0 or absolute_end < absolute_start or absolute_end > file_size:
        raise ValueError(f"{key} data offsets are outside the safetensors file")

    handle.seek(absolute_start)
    raw_config = handle.read(config_size)
    if len(raw_config) != config_size:
        raise ValueError(f"{key} tensor data is truncated")
    try:
        config = json.loads(raw_config)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{key} contains invalid quantization JSON") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{key} quantization JSON must be an object")
    return config


def _convrot_skip_reason(model_path: str | Path) -> str | None:
    model_path = Path(model_path)
    if model_path.suffix.lower() not in MODEL_EXTENSIONS:
        return f"unsupported file extension {model_path.suffix!r}"

    try:
        tensor_index, data_start, file_size = _read_safetensors_header(model_path)
        metadata = tensor_index.get("__metadata__", {})
        if not isinstance(metadata, dict):
            return "safetensors __metadata__ must be an object"

        raw_quantization = metadata.get("_quantization_metadata")
        if raw_quantization is not None:
            quantization = json.loads(raw_quantization)
            if not isinstance(quantization, dict) or not isinstance(quantization.get("layers"), dict):
                return "safetensors _quantization_metadata must contain a layers object"
            for layer_name, config in quantization["layers"].items():
                if _classify_config(config, layer_name, True) is not None:
                    return None

        with model_path.open("rb") as handle:
            for key, tensor_info in tensor_index.items():
                if key == "__metadata__" or not key.endswith(".comfy_quant"):
                    continue
                config = _read_quant_config(handle, tensor_info, key, data_start, file_size)
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
            log = LOG.warning if "convrot" in name.lower() else LOG.debug
            log("Skipping ConvRot candidate %s: %s", name, skip_reason)
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
    model_prefix: str | None = None,
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
        records.extend(
            (name, config)
            for name, config in header_quantization["layers"].items()
            if model_prefix is None or name.startswith(model_prefix)
        )

    tensor_configs: dict[str, dict] = {}
    for key, value in state_dict.items():
        if not key.endswith(".comfy_quant"):
            continue
        layer_name = key[: -len(".comfy_quant")]
        if model_prefix is not None and not layer_name.startswith(model_prefix):
            continue
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
        for _, config, weight_dtype, _ in classified_records:
            if weight_dtype != "w4_codebook":
                config["linear_dtype"] = "int8"

    if header_quantization is not None:
        metadata["_quantization_metadata"] = json.dumps(header_quantization)
    for key, config in tensor_configs.items():
        state_dict[key] = _encode_quant_tensor(config)

    summary = ConvRotSummary(
        w4a4=sum(1 for weight_dtype, act_dtype in layer_types.values() if (weight_dtype, act_dtype) == ("w4", "int4")),
        w4a8=sum(1 for weight_dtype, act_dtype in layer_types.values() if (weight_dtype, act_dtype) == ("w4", "int8")),
        codebook_w4a8=sum(
            1
            for weight_dtype, act_dtype in layer_types.values()
            if (weight_dtype, act_dtype) == ("w4_codebook", "int8")
        ),
        w8a8=sum(1 for weight_dtype, act_dtype in layer_types.values() if (weight_dtype, act_dtype) == ("w8", "int8")),
    )
    return metadata, summary


def _summarize_convrot_modules(root: torch.nn.Module) -> ConvRotSummary:
    w4a4 = 0
    w4a8 = 0
    codebook_w4a8 = 0
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
        elif quant_format == CODEBOOK_W4_FORMAT:
            if getattr(params, "codebook", None) is None:
                raise RuntimeError(
                    "Loaded grouped-codebook W4A8 layer is missing its 16-entry codebook"
                )
            if getattr(params, "correction", None) is not None:
                raise RuntimeError(
                    "Loaded grouped-codebook W4A8 layer uses asymmetric correction, "
                    "which is not supported by the production Turing path"
                )
            codebook_w4a8 += 1
        elif quant_format == W8_FORMAT and getattr(params, "convrot", False):
            w8a8 += 1
    return ConvRotSummary(
        w4a4=w4a4,
        w4a8=w4a8,
        codebook_w4a8=codebook_w4a8,
        w8a8=w8a8,
    )


def _loaded_convrot_summary(model) -> ConvRotSummary:
    return _summarize_convrot_modules(model.model)


def _loaded_convrot_clip_summary(clip) -> ConvRotSummary:
    return _summarize_convrot_modules(clip.cond_stage_model)


def _validate_runtime_support(expected: ConvRotSummary, device: torch.device | None = None) -> None:
    if expected.w4a8 == 0 and expected.codebook_w4a8 == 0:
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
    attention_backend: str | None = "w8a8",
    *,
    disable_dynamic: bool = False,
):
    attention_backend = normalize_attention_backend(attention_backend)
    model_path = Path(model_path)
    state_dict, metadata = comfy.utils.load_torch_file(str(model_path), return_metadata=True)
    diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
    if not any(key.startswith(diffusion_model_prefix) for key in state_dict):
        diffusion_model_prefix = ""
    metadata, expected = configure_convrot_activation(
        state_dict,
        metadata,
        force_int8_gemm,
        model_prefix=diffusion_model_prefix,
    )
    load_device = comfy.model_management.get_torch_device()
    _validate_runtime_support(expected, load_device)
    prepare_turing_runtime(expected, load_device, attention_backend)
    model_config = comfy.model_detection.model_config_from_unet(
        state_dict,
        diffusion_model_prefix,
        metadata=metadata,
    )
    compute_dtype = select_compute_dtype(
        model_config,
        load_device,
    )
    model = comfy.sd.load_diffusion_model_state_dict(
        state_dict,
        model_options={"dtype": compute_dtype} if compute_dtype is not None else {},
        metadata=metadata,
        disable_dynamic=disable_dynamic,
    )
    if model is None:
        raise RuntimeError(f"ComfyUI could not detect a supported model config from {model_path}")

    if compute_dtype is not None:
        model.set_model_compute_dtype(compute_dtype)
        normalize_turing_convrot_weight_dtypes(model, load_device, compute_dtype)

    loaded = _loaded_convrot_summary(model)
    if loaded != expected:
        raise RuntimeError(
            "ConvRot activation selection was not applied to every layer: "
            f"expected {expected}, loaded {loaded}. Update ComfyUI and comfy-kitchen."
        )

    LOG.info(
        "Loaded ConvRot model with force_int8_gemm=%s: "
        "W4A4=%d, legacy_W4A8=%d, codebook_W4A8=%d, W8A8=%d",
        force_int8_gemm,
        loaded.w4a4,
        loaded.w4a8,
        loaded.codebook_w4a8,
        loaded.w8a8,
    )
    apply_model_adapters(model, load_device)
    apply_attention_backend(model, attention_backend, device=load_device)
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
    prepare_turing_runtime(expected, load_device)

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
        "Loaded ConvRot CLIP with force_int8_gemm=%s: "
        "W4A4=%d, legacy_W4A8=%d, codebook_W4A8=%d, W8A8=%d",
        force_int8_gemm,
        loaded.w4a4,
        loaded.w4a8,
        loaded.codebook_w4a8,
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
