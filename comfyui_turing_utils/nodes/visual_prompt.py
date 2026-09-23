"""Derive reusable point, box, and mask prompts from one guidance frame."""

from __future__ import annotations

import json
import math

import numpy as np
import torch
from scipy import ndimage

from comfy_api.latest import io


def _select_guidance(source: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not torch.is_tensor(source) or source.ndim not in (3, 4):
        shape = tuple(source.shape) if hasattr(source, "shape") else type(source).__name__
        raise ValueError(
            "guidance must be a MASK [frames,height,width] or IMAGE "
            f"[frames,height,width,channels], got {shape}"
        )
    frames = int(source.shape[0])
    if frames < 1:
        raise ValueError("guidance must contain at least one frame")
    if index < 0 or index >= frames:
        raise ValueError(f"index {index} is outside the guidance batch range 0-{frames - 1}")

    selected = source[index : index + 1]
    if source.ndim == 3:
        mask = selected
    else:
        channels = int(source.shape[-1])
        if channels < 1:
            raise ValueError("IMAGE guidance must contain at least one channel")
        # Guidance images are scalar maps represented as IMAGE. Max preserves
        # coloured label maps as foreground without imposing RGB luminance semantics.
        mask = selected[..., : min(channels, 3)].amax(dim=-1)
    if not mask.is_floating_point():
        mask = mask.float()
    return selected, mask, index


def _spread_points(score: np.ndarray, count: int) -> list[tuple[int, int]]:
    """Select high-score points while progressively favouring spatial coverage."""

    count = max(0, int(count))
    ys, xs = np.nonzero(score > 0.0)
    if count == 0 or not len(xs):
        return []

    candidates = np.column_stack((ys, xs)).astype(np.float64, copy=False)
    weights = score[ys, xs].astype(np.float64, copy=False)
    selected: list[int] = []
    min_distance_sq: np.ndarray | None = None
    diagonal_sq = max(float(score.shape[0] ** 2 + score.shape[1] ** 2), 1.0)

    for _ in range(min(count, len(candidates))):
        if min_distance_sq is None:
            choice = int(np.argmax(weights))
        else:
            # Interior depth keeps points away from uncertain boundaries; the
            # distance factor prevents all points collapsing into one thick area.
            coverage = np.sqrt(min_distance_sq / diagonal_sq)
            ranking = weights * (0.25 + coverage)
            if selected:
                ranking[np.asarray(selected, dtype=np.int64)] = -1.0
            choice = int(np.argmax(ranking))
            if ranking[choice] < 0.0:
                break
        selected.append(choice)
        delta = candidates - candidates[choice]
        distance_sq = np.sum(delta * delta, axis=1)
        min_distance_sq = distance_sq if min_distance_sq is None else np.minimum(min_distance_sq, distance_sq)

    return [(int(xs[item]), int(ys[item])) for item in selected]


def visual_prompts_from_mask(
    mask: torch.Tensor,
    mask_threshold: float,
    positive_point_count: int,
    negative_point_count: int,
    bbox_padding: float,
) -> tuple[str, str, list[dict], list[list[dict]]]:
    if mask.ndim == 3:
        if int(mask.shape[0]) != 1:
            raise ValueError(f"expected one selected mask, got {int(mask.shape[0])}")
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"selected guidance must resolve to [height,width], got {tuple(mask.shape)}")
    if not 0.0 < mask_threshold <= 1.0:
        raise ValueError("mask_threshold must be in (0, 1]")
    if bbox_padding < 0.0:
        raise ValueError("bbox_padding must be non-negative")

    values = mask.detach().float().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("guidance contains non-finite values")
    binary = values >= float(mask_threshold)
    ys, xs = np.nonzero(binary)
    if not len(xs):
        raise ValueError("the selected guidance contains no foreground at mask_threshold")

    height, width = binary.shape
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    pad_x = int(math.ceil((x1 - x0) * float(bbox_padding)))
    pad_y = int(math.ceil((y1 - y0) * float(bbox_padding)))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)

    interior_depth = ndimage.distance_transform_edt(binary)
    positive = _spread_points(interior_depth, positive_point_count)

    negative: list[tuple[int, int]] = []
    if negative_point_count > 0 and not bool(binary.all()):
        outside_depth = ndimage.distance_transform_edt(~binary)
        object_scale = max(x1 - x0, y1 - y0)
        ring_radius = max(2.0, min(50.0, object_scale * 0.15))
        ring_score = np.where(
            (outside_depth > 0.0) & (outside_depth <= ring_radius),
            outside_depth,
            0.0,
        )
        negative = _spread_points(ring_score, negative_point_count)

    def encode(points: list[tuple[int, int]]) -> str:
        return json.dumps(
            [{"x": x, "y": y} for x, y in points],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    # SeC currently consumes the legacy KJNodes BBOX contract.
    legacy_bbox = [{"startX": x0, "startY": y0, "endX": x1, "endY": y1}]
    # ComfyUI's built-in SAM3 uses the canonical BOUNDING_BOX contract, nested
    # once because this node always outputs exactly one selected frame.
    canonical_bbox = [[{"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}]]
    return encode(positive), encode(negative), legacy_bbox, canonical_bbox


class MaskToVisualPrompts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        guidance_type = io.MatchType.Template("guidance", allowed_types=[io.Image, io.Mask])
        return io.Schema(
            node_id="TuringUtilsMaskToVisualPrompts",
            display_name="Mask to Visual Prompts",
            category="Turing Utils/mask",
            description=(
                "Select one frame from a MASK or scalar guidance IMAGE and derive reusable "
                "positive/negative point JSON, legacy SeC/KJ BBOX, and ComfyUI BOUNDING_BOX prompts. "
                "An RGB input is treated as a guidance map, not semantically segmented."
            ),
            inputs=[
                io.MatchType.Input("guidance", template=guidance_type),
                io.Int.Input("index", default=0, min=0, max=0x7FFFFFFF, step=1),
                io.Float.Input("mask_threshold", default=0.5, min=0.001, max=1.0, step=0.01, advanced=True),
                io.Int.Input("positive_point_count", default=3, min=1, max=32, step=1, advanced=True),
                io.Int.Input("negative_point_count", default=4, min=0, max=32, step=1, advanced=True),
                io.Float.Input("bbox_padding", default=0.05, min=0.0, max=1.0, step=0.01, advanced=True, tooltip="Padding on each side as a fraction of the tight mask bounding-box size."),
            ],
            outputs=[
                io.MatchType.Output(template=guidance_type, id="guidance"),
                io.Mask.Output("mask"),
                io.String.Output("positive_coords"),
                io.String.Output("negative_coords"),
                io.BBOX.Output("bbox"),
                io.BoundingBox.Output("bounding_box"),
                io.Int.Output("index"),
            ],
        )

    @classmethod
    def execute(
        cls,
        guidance,
        index,
        mask_threshold,
        positive_point_count,
        negative_point_count,
        bbox_padding,
    ) -> io.NodeOutput:
        selected, mask, selected_index = _select_guidance(guidance, int(index))
        prompts = visual_prompts_from_mask(
            mask,
            float(mask_threshold),
            int(positive_point_count),
            int(negative_point_count),
            float(bbox_padding),
        )
        return io.NodeOutput(selected, mask, *prompts, selected_index)
