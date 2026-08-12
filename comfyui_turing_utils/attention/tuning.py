"""Explicit experimental controls for the bundled SM75 attention kernels."""

from __future__ import annotations

from dataclasses import dataclass


ATTENTION_TUNING_KEY = "turing_utils_attention_tuning"


@dataclass(frozen=True, slots=True)
class AttentionKernelTuning:
    key_tile_tokens: int = 0
    rotate_qk: bool = True
    stabilize_k: bool = True


def attention_kernel_tuning(transformer_options) -> AttentionKernelTuning:
    if not isinstance(transformer_options, dict):
        return AttentionKernelTuning()
    raw = transformer_options.get(ATTENTION_TUNING_KEY)
    if not isinstance(raw, dict):
        return AttentionKernelTuning()
    key_tile_tokens = int(raw.get("key_tile_tokens", 0))
    if key_tile_tokens not in (0, 64, 128):
        key_tile_tokens = 0
    rotate_qk = bool(raw.get("rotate_qk", True))
    return AttentionKernelTuning(
        key_tile_tokens=key_tile_tokens,
        rotate_qk=rotate_qk,
        stabilize_k=rotate_qk and bool(raw.get("stabilize_k", True)),
    )


def apply_attention_kernel_tuning_patch(
    model,
    *,
    key_tile: str = "auto",
    rotate_qk: bool = True,
    stabilize_k: bool = True,
):
    choices = {"auto": 0, "64": 64, "128": 128}
    key_tile = str(key_tile).strip().lower()
    if key_tile not in choices:
        raise ValueError("key_tile must be auto, 64, or 128")
    patched = model.clone()
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    transformer_options[ATTENTION_TUNING_KEY] = {
        "key_tile_tokens": choices[key_tile],
        "rotate_qk": bool(rotate_qk),
        "stabilize_k": bool(rotate_qk and stabilize_k),
    }
    return patched
