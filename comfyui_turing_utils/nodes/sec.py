"""SeC visual-concept video tracking nodes."""

from __future__ import annotations

from comfy_api.latest import io

from ..adapters.sec import load_sec_model, sec_model_choices, track_visual_concept


SeCModelType = io.Custom("TURING_UTILS_SEC_MODEL")


class SeCModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        choices = sec_model_choices()
        return io.Schema(
            node_id="TuringUtilsSeCModelLoader",
            display_name="Load SeC Model",
            category="Turing Utils/SeC",
            description=(
                "Load a SeC visual-concept tracking model through ComfyUI's model "
                "lifecycle. Device placement, residency, and unloading are managed by ComfyUI."
            ),
            inputs=[
                io.Combo.Input("model_name", options=choices, default=choices[0]),
                io.Boolean.Input(
                    "use_flash_attention",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "Use Flash Attention when the installed runtime and checkpoint dtype support it; "
                        "otherwise SeC falls back to its standard attention implementation."
                    ),
                ),
                io.Boolean.Input(
                    "allow_mask_overlap",
                    default=True,
                    advanced=True,
                    tooltip="Allow masks from multiple tracked objects to overlap.",
                ),
            ],
            outputs=[SeCModelType.Output("model")],
        )

    @classmethod
    def execute(
        cls,
        model_name: str,
        use_flash_attention: bool = True,
        allow_mask_overlap: bool = True,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            load_sec_model(
                model_name,
                bool(use_flash_attention),
                bool(allow_mask_overlap),
            )
        )


class SeCTrackVisualConcept(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsSeCTrackVisualConcept",
            display_name="SeC Track Visual Concept",
            category="Turing Utils/SeC",
            description=(
                "Track one visual concept through a video. With input_mask connected, the mask is "
                "authoritative and points must agree with it; the BBOX limits its region. Without a "
                "mask, BBOX and positive/negative points are submitted together as one SAM2 prompt."
            ),
            inputs=[
                SeCModelType.Input("model"),
                io.Image.Input("frames"),
                io.String.Input(
                    "positive_points",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip='JSON point list such as [{"x": 120, "y": 240}].',
                ),
                io.String.Input(
                    "negative_points",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip='JSON exclusion-point list such as [{"x": 80, "y": 200}].',
                ),
                io.BBOX.Input(
                    "bbox",
                    optional=True,
                    tooltip="Legacy BBOX prompt, including the output from Mask to Visual Prompts.",
                ),
                io.Mask.Input(
                    "input_mask",
                    optional=True,
                    tooltip=(
                        "One mask or one mask per video frame. If batched, the annotation-frame mask "
                        "is selected. Mask prompt state takes precedence over points because SAM2 "
                        "cannot retain mask and click state simultaneously."
                    ),
                ),
                io.Combo.Input(
                    "tracking_direction",
                    options=["forward", "backward", "bidirectional"],
                    default="forward",
                ),
                io.Int.Input("annotation_frame_idx", default=0, min=0, max=1_000_000, step=1),
                io.Int.Input("object_id", default=1, min=1, max=1_000_000, step=1, advanced=True),
                io.Int.Input(
                    "max_frames_to_track",
                    default=-1,
                    min=-1,
                    max=1_000_000,
                    step=1,
                    advanced=True,
                    tooltip="-1 tracks every reachable frame in the selected direction.",
                ),
                io.Int.Input(
                    "mllm_memory_size",
                    default=12,
                    min=1,
                    max=20,
                    step=1,
                    advanced=True,
                    tooltip=(
                        "Maximum semantic keyframes retained for scene-change recovery. Larger values "
                        "can improve concept recovery but also increase MLLM activation memory."
                    ),
                ),
            ],
            outputs=[io.Mask.Output("masks")],
        )

    @classmethod
    def execute(
        cls,
        model,
        frames,
        positive_points="",
        negative_points="",
        bbox=None,
        input_mask=None,
        tracking_direction="forward",
        annotation_frame_idx=0,
        object_id=1,
        max_frames_to_track=-1,
        mllm_memory_size=12,
    ) -> io.NodeOutput:
        masks = track_visual_concept(
            model,
            frames,
            positive_points=positive_points,
            negative_points=negative_points,
            bbox=bbox,
            input_mask=input_mask,
            tracking_direction=tracking_direction,
            annotation_frame_idx=int(annotation_frame_idx),
            object_id=int(object_id),
            max_frames_to_track=int(max_frames_to_track),
            mllm_memory_size=int(mllm_memory_size),
        )
        return io.NodeOutput(masks)


__all__ = ["SeCModelLoader", "SeCTrackVisualConcept"]
