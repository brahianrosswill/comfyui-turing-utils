from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import folder_paths
import torch

import comfy.model_management
import comfy.sd
import comfy.utils


LOG = logging.getLogger("comfyui-svdint4")
FOLDER_NAME = "diffusion_models"
ACTIVATION_DTYPES = ("auto", "int4", "int8")
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


def _classify_config(config: dict, layer_name: str) -> tuple[str, str] | None:
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
        return "w4", activation_dtype

    convrot = _config_value(config, params, "convrot", False)
    if not isinstance(convrot, bool):
        raise ValueError(f"Quantized layer {layer_name} has non-boolean convrot={convrot!r}")
    if not convrot:
        return None
    if quant_format != W8_FORMAT:
        raise ValueError(f"ConvRot layer {layer_name} uses unsupported weight format {quant_format!r}")

    activation_dtype = _config_value(config, params, "linear_dtype", "int8")
    if activation_dtype != "int8":
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


def configure_convrot_activation(
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, str] | None,
    activation_dtype: str,
) -> tuple[dict[str, str], ConvRotSummary]:
    if activation_dtype not in ACTIVATION_DTYPES:
        raise ValueError(
            f"Unsupported ConvRot activation_dtype={activation_dtype!r}; expected one of {ACTIVATION_DTYPES}"
        )

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
        classification = _classify_config(config, layer_name)
        if classification is None:
            continue
        previous = layer_types.get(layer_name)
        if previous is not None:
            same_weight_format = previous[0] == classification[0]
            same_auto_activation = previous[1] == classification[1]
            if not same_weight_format or (activation_dtype == "auto" and not same_auto_activation):
                raise ValueError(
                    f"Conflicting ConvRot metadata for layer {layer_name}: "
                    f"{previous[0]}/{previous[1]} versus {classification[0]}/{classification[1]}"
                )
        else:
            layer_types[layer_name] = classification
        classified_records.append((layer_name, config, *classification))

    if not classified_records:
        raise ValueError("The selected model does not contain supported ConvRot quantization metadata")

    w8_layers = sorted(name for name, (weight_dtype, _) in layer_types.items() if weight_dtype == "w8")
    if activation_dtype == "int4" and w8_layers:
        sample = ", ".join(w8_layers[:5])
        suffix = "" if len(w8_layers) <= 5 else ", ..."
        raise ValueError(
            f"Cannot run ConvRot W8 weights with INT4 activations. "
            f"The model contains {len(w8_layers)} W8 ConvRot layer(s): {sample}{suffix}"
        )

    if activation_dtype != "auto":
        for _, config, weight_dtype, _ in classified_records:
            if weight_dtype == "w4":
                config["linear_dtype"] = activation_dtype
        layer_types = {
            name: (weight_dtype, activation_dtype if weight_dtype == "w4" else current_activation)
            for name, (weight_dtype, current_activation) in layer_types.items()
        }

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


def _loaded_convrot_summary(model) -> ConvRotSummary:
    w4a4 = 0
    w4a8 = 0
    w8a8 = 0
    for _, module in model.model.named_modules():
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


def _validate_runtime_support(expected: ConvRotSummary) -> None:
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
    activation_dtype: str = "auto",
    *,
    disable_dynamic: bool = False,
):
    model_path = Path(model_path)
    state_dict, metadata = comfy.utils.load_torch_file(str(model_path), return_metadata=True)
    metadata, expected = configure_convrot_activation(state_dict, metadata, activation_dtype)
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
        "Loaded ConvRot model with activation_dtype=%s: W4A4=%d, W4A8=%d, W8A8=%d",
        activation_dtype,
        loaded.w4a4,
        loaded.w4a8,
        loaded.w8a8,
    )
    model.cached_patcher_init = (load_convrot_model, (str(model_path), activation_dtype))
    return model


class ConvRotDiffusionModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list(FOLDER_NAME),),
                "activation_dtype": (
                    ACTIVATION_DTYPES,
                    {
                        "default": "auto",
                        "tooltip": (
                            "auto follows each layer's file metadata. int4 selects W4A4 and requires W4 ConvRot weights. "
                            "int8 selects W4A8 for W4 weights and keeps W8 ConvRot layers at W8A8."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_diffusion_model"
    CATEGORY = "SVDInt4/loaders"
    TITLE = "Load ConvRot DiT"

    def load_diffusion_model(self, unet_name: str, activation_dtype: str = "auto"):
        model_path = folder_paths.get_full_path_or_raise(FOLDER_NAME, unet_name)
        return (load_convrot_model(model_path, activation_dtype),)
