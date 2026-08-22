"""Thin ComfyUI nodes for ConvRot model and CLIP loading."""

from __future__ import annotations

import folder_paths
import torch

import comfy.sd

from ..attention import attention_backend_choices
from ..quantization import convrot as service


class ConvRotDiffusionModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (
                    service._convrot_model_names(service.DIFFUSION_FOLDER_NAME),
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
                        "default": "w8a8",
                        "tooltip": (
                            "Select w8a8, sage, or sdpa. W8A8 uses the bundled sm75+ path with a native cubin "
                            "for the installed GPU; exact-sm75 Sage is bundled and newer GPUs use installed SageAttention. "
                            "Turing BF16 SDPA inputs are stored as FP16 for the attention call. "
                            "Sol sparse attention is configured with the separate Sol patch node."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_diffusion_model"
    CATEGORY = "Turing Utils/loaders"
    TITLE = "Load ConvRot DiT"

    def load_diffusion_model(
        self,
        unet_name: str,
        force_int8_gemm: bool = False,
        patch_attention: str = "w8a8",
    ):
        model_path = service._resolve_convrot_model_path(
            service.DIFFUSION_FOLDER_NAME, unet_name
        )
        return (
            service.load_convrot_model(
                model_path,
                force_int8_gemm,
                attention_backend=patch_attention,
            ),
        )


class ConvRotCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        official_inputs = service._official_clip_loader_inputs()
        official_required = official_inputs["required"]
        if "clip_name" not in official_required or "type" not in official_required:
            raise RuntimeError(
                "ComfyUI's official CLIPLoader must expose clip_name and type inputs"
            )
        official_optional = official_inputs.get("optional", {})
        return {
            "required": {
                "clip_name": official_required["clip_name"],
                "type": official_required["type"],
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
            "optional": (
                {"device": official_optional["device"]}
                if "device" in official_optional
                else {}
            ),
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "Turing Utils/loaders"
    TITLE = "Load ConvRot CLIP"

    def load_clip(
        self,
        clip_name: str,
        type: str = "stable_diffusion",
        force_int8_gemm: bool = False,
        device: str = "default",
    ):
        clip_types = service._official_clip_types()
        if type not in clip_types:
            raise ValueError(f"Unsupported ConvRot CLIP type {type!r}; expected one of {clip_types}")
        if device not in {"default", "cpu"}:
            raise ValueError(f"Unsupported ConvRot CLIP device {device!r}; expected 'default' or 'cpu'")
        try:
            clip_type = comfy.sd.CLIPType[type.upper()]
        except KeyError as exc:
            raise RuntimeError(f"ComfyUI CLIPLoader exposes type {type!r} without a matching CLIPType") from exc
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        model_path = service._resolve_convrot_model_path(
            service.CLIP_FOLDER_NAME, clip_name
        )
        clip = service.load_convrot_clip(
            model_path,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            force_int8_gemm=force_int8_gemm,
            model_options=model_options,
        )
        return (clip,)
