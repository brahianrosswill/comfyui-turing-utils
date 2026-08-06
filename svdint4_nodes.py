from __future__ import annotations

import logging
import os
from pathlib import Path

import folder_paths
from safetensors import safe_open

try:
    from .attention import attention_backend_choices
    from .model_format import validate_svdint4_metadata
except ImportError:
    from attention import attention_backend_choices
    from model_format import validate_svdint4_metadata


LOG = logging.getLogger("comfyui-svdint4")
FOLDER_NAME = "diffusion_models"
MODEL_EXTENSIONS = {".safetensors", ".sft"}
ENV_PATHS = ("SVDINT4_DIT_PATHS",)


def _model_dirs() -> list[str]:
    return folder_paths.get_folder_paths(FOLDER_NAME)


def _register_extra_model_dirs() -> None:
    changed = False
    for env_name in ENV_PATHS:
        for item in os.environ.get(env_name, "").split(os.pathsep):
            if not item:
                continue
            path = Path(item).expanduser()
            if not path.is_dir():
                LOG.warning("Ignoring %s entry because it is not a directory: %s", env_name, item)
                continue
            before = _model_dirs()
            folder_paths.add_model_folder_path(FOLDER_NAME, str(path))
            changed = changed or before != _model_dirs()
    if changed:
        folder_paths.filename_list_cache.pop(FOLDER_NAME, None)


def _model_names() -> list[str]:
    _register_extra_model_dirs()
    names: list[str] = []
    for name in folder_paths.get_filename_list(FOLDER_NAME):
        if Path(name).suffix.lower() not in MODEL_EXTENSIONS:
            continue
        path = folder_paths.get_full_path(FOLDER_NAME, name)
        if path is None:
            LOG.debug("Skipping SVDInt4 candidate %s: folder_paths could not resolve it", name)
            continue
        skip_reason = _svdint4_skip_reason(path)
        if skip_reason is None:
            names.append(name)
        else:
            LOG.debug("Skipping SVDInt4 candidate %s: %s", name, skip_reason)
    return names


def _svdint4_skip_reason(model_path: str | Path) -> str | None:
    try:
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception as exc:
        return f"could not read safetensors metadata ({exc})"
    try:
        validate_svdint4_metadata(metadata, model_path)
    except ValueError as exc:
        return str(exc)
    return None


def _resolve_model_path(model_name: str) -> str:
    _register_extra_model_dirs()
    return folder_paths.get_full_path_or_raise(FOLDER_NAME, model_name)


class SVDInt4DiffusionModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (
                    _model_names(),
                    {
                        "tooltip": (
                            "SVDInt4 DiT file from ComfyUI/models/diffusion_models. "
                            "Only supported SVDInt4 single-file safetensors assets are shown."
                        )
                    },
                ),
            },
            "optional": {
                "patch_attention": (
                    attention_backend_choices(),
                    {
                        "default": "auto",
                        "tooltip": (
                            "Select this SVDInt4 model's attention backend. "
                            "On Turing, auto uses the stable bundled sage backend. Elsewhere auto tries installed "
                            "sage_attn, then flash_attn, then PyTorch SDPA."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_diffusion_model"
    CATEGORY = "SVDInt4/loaders"
    TITLE = "Load SVDInt4 DiT"

    def load_diffusion_model(
        self,
        unet_name: str,
        patch_attention: str = "auto",
    ):
        from .loader import load_svdint4_model

        return (load_svdint4_model(_resolve_model_path(unet_name), attention_backend=patch_attention),)
