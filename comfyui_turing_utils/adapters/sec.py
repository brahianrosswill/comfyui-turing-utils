"""ComfyUI-managed integration for SeC visual-concept video tracking."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.model_patcher
import comfy.storage
import comfy.utils
import folder_paths

from ..log import get_logger


LOG = get_logger("sec")
_MISSING_MODEL = "(No SeC models found in models/sams)"
_SINGLE_FILE_SUFFIXES = {".safetensors"}


def register_sec_model_folder() -> None:
    """Register the established SeC/SAM model directory with ComfyUI."""

    folder_paths.add_model_folder_path(
        "sams",
        str(Path(folder_paths.models_dir) / "sams"),
    )


register_sec_model_folder()


@dataclass(frozen=True)
class SeCModelSpec:
    name: str
    path: Path
    config_path: Path
    single_file: bool


class _ManagedSeCModule(nn.Module):
    """Give Hugging Face's read-only ``device`` model a Comfy patcher shell."""

    def __init__(self, sec_model: nn.Module):
        super().__init__()
        self.sec_model = sec_model
        self.device = torch.device("cpu")


@dataclass
class SeCModelHandle:
    patcher: comfy.model_patcher.ModelPatcher
    dtype: torch.dtype
    source: str

    @property
    def model(self):
        return self.patcher.model.sec_model


@dataclass(frozen=True)
class SeCVisualPrompt:
    mask: np.ndarray | None
    points: np.ndarray | None
    labels: np.ndarray | None
    box: np.ndarray | None


def _bundled_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "vendor" / "sec" / "model_config"


def _model_roots() -> tuple[Path, ...]:
    try:
        return tuple(Path(path) for path in folder_paths.get_folder_paths("sams"))
    except KeyError:
        return ()


def available_sec_models() -> tuple[SeCModelSpec, ...]:
    """Find single-file and Hugging Face directory-format SeC checkpoints."""

    found: dict[str, SeCModelSpec] = {}
    bundled_config = _bundled_config_path()
    for root in _model_roots():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in _SINGLE_FILE_SUFFIXES:
                if "sec" not in path.relative_to(root).as_posix().lower():
                    continue
                # Shard files belong to their parent directory entry.
                if path.name.startswith("model-") and (path.parent / "config.json").is_file():
                    continue
                name = path.relative_to(root).as_posix()
                found.setdefault(
                    name,
                    SeCModelSpec(name, path, bundled_config, True),
                )
                continue
            if not path.is_dir() or not (path / "config.json").is_file():
                continue
            if "sec" not in path.relative_to(root).as_posix().lower():
                continue
            has_weights = any(
                (path / filename).is_file()
                for filename in (
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "pytorch_model.bin",
                    "pytorch_model.bin.index.json",
                )
            )
            if has_weights and (path / "tokenizer_config.json").is_file():
                name = f"{path.relative_to(root).as_posix()}/"
                found.setdefault(name, SeCModelSpec(name, path, path, False))
    return tuple(found[name] for name in sorted(found, key=str.casefold))


def sec_model_choices() -> list[str]:
    choices = [spec.name for spec in available_sec_models()]
    return choices or [_MISSING_MODEL]


def _resolve_model(model_name: str) -> SeCModelSpec:
    for spec in available_sec_models():
        if spec.name == model_name:
            return spec
    raise FileNotFoundError(
        f"SeC model {model_name!r} was not found. Put a SeC single-file checkpoint "
        "or Hugging Face model directory under ComfyUI/models/sams."
    )


def _dominant_float_dtype(state_dict: dict[str, Any]) -> torch.dtype:
    counts: dict[torch.dtype, int] = {}
    for value in state_dict.values():
        if torch.is_tensor(value) and value.is_floating_point():
            counts[value.dtype] = counts.get(value.dtype, 0) + value.numel()
    if not counts:
        raise ValueError("The SeC checkpoint contains no floating-point weights")
    dtype = max(counts, key=counts.__getitem__)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.float16
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported SeC checkpoint dtype: {dtype}")
    return dtype


def _convert_float8_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    # Replace tensors incrementally so a 4B checkpoint does not coexist with a
    # second, fully materialized fp16 state dict in host memory.
    for key in list(state_dict):
        value = state_dict[key]
        if torch.is_tensor(value) and value.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            state_dict[key] = value.to(torch.float16)
    return state_dict


def _config_dtype(config) -> torch.dtype:
    value = getattr(config, "torch_dtype", None)
    if isinstance(value, torch.dtype):
        return value
    value = str(value or "").lower()
    return {
        "torch.float16": torch.float16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "torch.float32": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(value, torch.float16)


def _register_parameter_dtype_input_hooks(model: nn.Module) -> int:
    """Align floating inputs with each parameterized module's weight dtype.

    SeC/SAM2 creates several positional and memory tensors in fp32 even when
    the released checkpoint is fp16 or bf16.  PyTorch's Linear and LayerNorm
    kernels require those inputs to match their parameters.  Keep the repair
    scoped to this model instance: the hooks change dtype only, while ComfyUI
    remains responsible for device placement and model residency.
    """

    handles = []
    for module in model.modules():
        parameter = next(module.parameters(recurse=False), None)
        if parameter is None or not parameter.is_floating_point():
            continue
        target_dtype = parameter.dtype

        def align_dtype(current_module, args, kwargs, *, dtype=target_dtype):
            del current_module

            def convert(value):
                if torch.is_tensor(value) and value.is_floating_point() and value.dtype != dtype:
                    return value.to(dtype=dtype)
                return value

            return tuple(convert(value) for value in args), {
                key: convert(value) for key, value in kwargs.items()
            }

        handles.append(
            module.register_forward_pre_hook(align_dtype, with_kwargs=True)
        )
    # Retain handles for the lifetime of the model and make the installation
    # explicit for diagnostics.  Module hook dictionaries also own them, but
    # this gives us one place to inspect or remove them in future revisions.
    model._sec_dtype_input_hook_handles = handles
    return len(handles)


def load_sec_model(
    model_name: str,
    use_flash_attention: bool,
    allow_mask_overlap: bool,
) -> SeCModelHandle:
    """Load SeC on the offload device and register it with ComfyUI."""

    try:
        from accelerate import init_empty_weights
        from transformers import Qwen2Tokenizer

        from ..vendor.sec.configuration_sec import SeCConfig
        from ..vendor.sec.modeling_sec import SeCModel
    except ImportError as error:
        raise ImportError(
            "SeC dependencies are incomplete. Reinstall this custom node's "
            "requirements.txt before loading a SeC checkpoint."
        ) from error

    spec = _resolve_model(model_name)
    config = SeCConfig.from_pretrained(str(spec.config_path))
    config.hydra_overrides_extra = [
        f"++model.non_overlap_masks={'false' if allow_mask_overlap else 'true'}"
    ]
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    cpu_inference = comfy.model_management.is_device_cpu(load_device)

    fast_disk = False
    if spec.single_file:
        state_dict = comfy.utils.load_torch_file(str(spec.path), safe_load=True)
        dtype = _dominant_float_dtype(state_dict)
        if any(
            torch.is_tensor(value)
            and value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            for value in state_dict.values()
        ):
            LOG.warning(
                "SeC float8 checkpoint %s is expanded to fp16 because upstream SeC "
                "float8 inference is numerically unstable",
                spec.name,
            )
            state_dict = _convert_float8_state_dict(state_dict)
        if cpu_inference and dtype != torch.float32:
            LOG.warning(
                "SeC checkpoint %s will use fp32 because ComfyUI is running on CPU",
                spec.name,
            )
            dtype = torch.float32
        enable_flash = bool(use_flash_attention and not cpu_inference and dtype != torch.float32)
        with init_empty_weights(include_buffers=False):
            model = SeCModel(config, use_flash_attn=enable_flash)
        incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
        missing = [key for key in incompatible.missing_keys if not key.endswith("num_batches_tracked")]
        if missing or incompatible.unexpected_keys:
            raise ValueError(
                "SeC checkpoint does not match the bundled architecture: "
                f"missing={missing[:20]}, unexpected={incompatible.unexpected_keys[:20]}"
            )
        fast_disk = comfy.storage.state_dict_fast_disk(state_dict)
        del state_dict
    else:
        dtype = _config_dtype(config)
        enable_flash = bool(use_flash_attention and not cpu_inference and dtype != torch.float32)
        model = SeCModel.from_pretrained(
            str(spec.path),
            config=config,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            use_flash_attn=enable_flash,
        )
        dtype = next(
            (parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()),
            dtype,
        )
        if cpu_inference and dtype != torch.float32:
            LOG.warning(
                "SeC checkpoint %s will use fp32 because ComfyUI is running on CPU",
                spec.name,
            )
            dtype = torch.float32

    tokenizer = Qwen2Tokenizer.from_pretrained(
        str(spec.config_path),
        local_files_only=True,
    )
    model.eval()
    model.preparing_for_generation(tokenizer=tokenizer, torch_dtype=dtype)
    dtype_hook_count = 0
    if not cpu_inference and dtype != torch.float32:
        dtype_hook_count = _register_parameter_dtype_input_hooks(model)

    managed = _ManagedSeCModule(model).eval()
    # DynamicVRAM intentionally keeps parameter storage off-device and moves the
    # active workset at op time.  SeC and SAM2 otherwise infer their compute
    # device from the first stored parameter and incorrectly choose CPU.
    model._comfy_load_device = load_device
    model.grounding_encoder._comfy_load_device = load_device
    if offload_device.type != "cpu":
        managed.to(offload_device)
        managed.device = offload_device
    patcher = comfy.model_patcher.CoreModelPatcher(
        managed,
        load_device=load_device,
        offload_device=offload_device,
        fast_disk=fast_disk,
    )
    LOG.info(
        "Loaded SeC model %s on the ComfyUI offload device: dtype=%s flash_attention=%s "
        "allow_mask_overlap=%s dtype_hooks=%d size=%.2f GiB",
        spec.name,
        dtype,
        enable_flash,
        allow_mask_overlap,
        dtype_hook_count,
        patcher.model_size() / (1024**3),
    )
    return SeCModelHandle(patcher=patcher, dtype=dtype, source=spec.name)


def parse_points(value: str | None, *, width: int, height: int, label: int) -> tuple[np.ndarray, np.ndarray]:
    if value is None or not str(value).strip():
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid point JSON: {error}") from error
    if not isinstance(decoded, list):
        raise ValueError("Point prompts must be a JSON list")
    points = []
    for index, item in enumerate(decoded):
        if not isinstance(item, dict) or "x" not in item or "y" not in item:
            raise ValueError(f"Point {index} must contain numeric x and y fields")
        try:
            x, y = float(item["x"]), float(item["y"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Point {index} has non-numeric coordinates") from error
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"Point {index} ({x:g}, {y:g}) is outside the {width}x{height} frame"
            )
        points.append((x, y))
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    labels = np.full((len(array),), int(label), dtype=np.int32)
    return array, labels


def parse_bbox(value, *, width: int, height: int) -> np.ndarray | None:
    if value is None:
        return None
    current = value
    while isinstance(current, (list, tuple)) and len(current) == 1:
        current = current[0]
    if isinstance(current, dict):
        if {"startX", "startY", "endX", "endY"} <= current.keys():
            coords = (
                current["startX"],
                current["startY"],
                current["endX"],
                current["endY"],
            )
        elif {"x", "y", "width", "height"} <= current.keys():
            coords = (
                current["x"],
                current["y"],
                float(current["x"]) + float(current["width"]),
                float(current["y"]) + float(current["height"]),
            )
        else:
            raise ValueError("BBOX dictionaries must use startX/startY/endX/endY or x/y/width/height")
    elif isinstance(current, (list, tuple)) and len(current) == 4:
        coords = current
    else:
        raise ValueError(f"Unsupported BBOX value: {type(value).__name__}")
    try:
        x1, y1, x2, y2 = (float(item) for item in coords)
    except (TypeError, ValueError) as error:
        raise ValueError("BBOX coordinates must be numeric") from error
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(width), x2), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"BBOX is empty after clipping to the {width}x{height} frame")
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _select_mask(input_mask: torch.Tensor, frames: torch.Tensor, annotation_frame_idx: int) -> np.ndarray:
    if input_mask.ndim == 2:
        selected = input_mask
    elif input_mask.ndim == 3:
        if input_mask.shape[0] == 1:
            selected = input_mask[0]
        elif input_mask.shape[0] == frames.shape[0]:
            selected = input_mask[annotation_frame_idx]
        else:
            raise ValueError(
                "input_mask must contain one mask or one mask per video frame; "
                f"got {input_mask.shape[0]} masks for {frames.shape[0]} frames"
            )
    else:
        raise ValueError(f"input_mask must be [H,W] or [N,H,W], got {tuple(input_mask.shape)}")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    selected = selected.detach().to(device="cpu", dtype=torch.float32)
    if tuple(selected.shape) != (height, width):
        selected = F.interpolate(
            selected[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    mask = selected.numpy() >= 0.5
    if not mask.any():
        raise ValueError("input_mask contains no foreground on the annotation frame")
    return mask


def prepare_visual_prompt(
    frames: torch.Tensor,
    *,
    annotation_frame_idx: int,
    positive_points: str | None,
    negative_points: str | None,
    bbox,
    input_mask: torch.Tensor | None,
) -> SeCVisualPrompt:
    height, width = int(frames.shape[1]), int(frames.shape[2])
    positive, positive_labels = parse_points(
        positive_points, width=width, height=height, label=1
    )
    negative, negative_labels = parse_points(
        negative_points, width=width, height=height, label=0
    )
    box = parse_bbox(bbox, width=width, height=height)

    if input_mask is not None:
        mask = _select_mask(input_mask, frames, annotation_frame_idx)
        if box is not None:
            x1, y1, x2, y2 = box
            roi = np.zeros_like(mask)
            roi[int(np.floor(y1)) : int(np.ceil(y2)), int(np.floor(x1)) : int(np.ceil(x2))] = True
            mask &= roi
            if not mask.any():
                raise ValueError("BBOX does not overlap the input_mask foreground")
        for point in positive:
            x, y = int(point[0]), int(point[1])
            if not mask[y, x]:
                raise ValueError(f"Positive point ({x}, {y}) is outside the authoritative input_mask")
        for point in negative:
            x, y = int(point[0]), int(point[1])
            if mask[y, x]:
                raise ValueError(f"Negative point ({x}, {y}) falls inside the authoritative input_mask")
        # SAM2 stores mask and point inputs as mutually exclusive prompt states.
        # Keeping one authoritative mask avoids the silent last-prompt-wins behavior.
        return SeCVisualPrompt(mask=mask, points=None, labels=None, box=None)

    if len(positive) == 0 and box is None:
        if len(negative):
            raise ValueError("Negative points require at least one positive point or a BBOX")
        raise ValueError("Provide input_mask, positive_points, or bbox")
    points = np.concatenate((positive, negative), axis=0)
    labels = np.concatenate((positive_labels, negative_labels), axis=0)
    if len(points) == 0:
        points = labels = None
    return SeCVisualPrompt(mask=None, points=points, labels=labels, box=box)


def estimate_sec_activation_memory(frames: torch.Tensor, mllm_memory_size: int) -> int:
    """Conservative reservation for SAM state plus scene-change MLLM activations."""

    frame_bytes = int(frames.shape[0] * frames.shape[1] * frames.shape[2] * 12)
    base = 1024**3
    semantic = int(max(1, mllm_memory_size) * 128 * 1024**2)
    return base + semantic + frame_bytes


def _seed_predictor(predictor, state, prompt: SeCVisualPrompt, frame_idx: int, object_id: int):
    if prompt.mask is not None:
        _, _, logits = predictor.add_new_mask(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=object_id,
            mask=prompt.mask,
        )
        return prompt.mask
    _, _, logits = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=frame_idx,
        obj_id=object_id,
        points=prompt.points,
        labels=prompt.labels,
        box=prompt.box,
    )
    return (logits[0].detach().float().cpu().numpy().squeeze() > 0.0)


def track_visual_concept(
    handle: SeCModelHandle,
    frames: torch.Tensor,
    *,
    positive_points: str | None = "",
    negative_points: str | None = "",
    bbox=None,
    input_mask: torch.Tensor | None = None,
    tracking_direction: str = "forward",
    annotation_frame_idx: int = 0,
    object_id: int = 1,
    max_frames_to_track: int = -1,
    mllm_memory_size: int = 12,
) -> torch.Tensor:
    if not isinstance(handle, SeCModelHandle):
        raise TypeError("model must come from Load SeC Model")
    if not torch.is_tensor(frames) or frames.ndim != 4 or frames.shape[-1] < 3:
        shape = tuple(frames.shape) if hasattr(frames, "shape") else type(frames).__name__
        raise ValueError(f"frames must be a ComfyUI IMAGE batch [N,H,W,C], got {shape}")
    frame_count = int(frames.shape[0])
    if frame_count < 1:
        raise ValueError("frames must contain at least one image")
    if not 0 <= int(annotation_frame_idx) < frame_count:
        raise ValueError(
            f"annotation_frame_idx must be in [0,{frame_count - 1}], got {annotation_frame_idx}"
        )
    if tracking_direction not in {"forward", "backward", "bidirectional"}:
        raise ValueError(f"Unsupported tracking_direction: {tracking_direction}")

    prompt = prepare_visual_prompt(
        frames,
        annotation_frame_idx=int(annotation_frame_idx),
        positive_points=positive_points,
        negative_points=negative_points,
        bbox=bbox,
        input_mask=input_mask,
    )
    memory_required = estimate_sec_activation_memory(frames, int(mllm_memory_size))
    comfy.model_management.load_models_gpu(
        [handle.patcher],
        memory_required=memory_required,
        force_full_load=True,
    )

    model = handle.model
    predictor = model.grounding_encoder
    state = None
    output_device = comfy.model_management.intermediate_device()
    masks = torch.zeros(
        (frame_count, int(frames.shape[1]), int(frames.shape[2])),
        dtype=torch.float32,
        device=output_device,
    )
    limit = frame_count if int(max_frames_to_track) < 0 else int(max_frames_to_track)
    progress = comfy.utils.ProgressBar(frame_count)
    completed: set[int] = set()

    def propagate(reverse: bool, init_mask: np.ndarray) -> None:
        for out_frame_idx, _out_obj_ids, mask_logits in model.propagate_in_video(
            state,
            start_frame_idx=int(annotation_frame_idx),
            max_frame_num_to_track=limit,
            reverse=reverse,
            init_mask=init_mask,
            mllm_memory_size=int(mllm_memory_size),
        ):
            comfy.model_management.throw_exception_if_processing_interrupted()
            mask = (mask_logits[0].detach().float().cpu().squeeze() > 0.0).to(torch.float32)
            masks[int(out_frame_idx)].copy_(mask.to(output_device))
            if int(out_frame_idx) not in completed:
                completed.add(int(out_frame_idx))
                progress.update(1)

    try:
        # Frames and tracking state remain on CPU; only the current model workset is
        # transferred to the device.  ComfyUI owns the model's residency lifecycle.
        state = predictor.init_state(
            video_path=frames,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        predictor.reset_state(state)
        init_mask = _seed_predictor(
            predictor,
            state,
            prompt,
            int(annotation_frame_idx),
            int(object_id),
        )
        if tracking_direction == "bidirectional":
            propagate(False, init_mask)
            predictor.reset_state(state)
            init_mask = _seed_predictor(
                predictor,
                state,
                prompt,
                int(annotation_frame_idx),
                int(object_id),
            )
            propagate(True, init_mask)
        else:
            propagate(tracking_direction == "backward", init_mask)
        return masks
    finally:
        if state is not None:
            try:
                predictor.reset_state(state)
            finally:
                state.clear()


__all__ = [
    "SeCModelHandle",
    "available_sec_models",
    "estimate_sec_activation_memory",
    "load_sec_model",
    "parse_bbox",
    "parse_points",
    "prepare_visual_prompt",
    "sec_model_choices",
    "track_visual_concept",
]
