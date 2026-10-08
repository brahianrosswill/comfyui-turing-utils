"""Public cards have canvas-only ports and cannot execute as ordinary nodes."""

import folder_paths
from comfy_api.latest import io

from .graph import H3, MASK, SETTINGS
from ..adapters.sec import sec_model_choices


ImageAsset = io.Custom("TURING_CANVAS_IMAGE_ASSET")
VideoAsset = io.Custom("TURING_CANVAS_VIDEO_ASSET")
AudioAsset = io.Custom("TURING_CANVAS_AUDIO_ASSET")
MaskAsset = io.Custom("TURING_CANVAS_MASK_ASSET")
CATEGORY = "Turing Utils/Canvas"


class CanvasCard(io.ComfyNode):
    @classmethod
    def execute(cls, **kwargs):
        raise ValueError("Use the material canvas card buttons; ordinary workflow execution is disabled")


def model_input(name, folder):
    return io.Combo.Input(name, options=["", *folder_paths.get_filename_list(folder)])


class CanvasSettings(CanvasCard):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=SETTINGS, display_name="Canvas Settings", category=CATEGORY,
            inputs=[
                io.String.Input("work_directory", default="canvas/project", tooltip="Relative to this ComfyUI instance's output directory."),
                io.String.Input("cache_directory", default="canvas/project", advanced=True,
                    tooltip="Relative to this ComfyUI instance's .cache directory; only rebuildable previews, not weights."),
                io.Combo.Input("import_mode", options=["browser_upload", "local_copy"]),
                model_input("dit", "diffusion_models"), model_input("clip", "text_encoders"),
                model_input("video_vae", "vae"), model_input("audio_vae", "vae"),
                io.String.Input("loras", default="[]", multiline=True, advanced=True,
                    tooltip='JSON list: [{"name":"model.safetensors","strength":1.0}]'),
                io.Combo.Input("attention", options=["w8a8", "sage", "sdpa"], advanced=True),
                io.Boolean.Input("sol", default=False, advanced=True),
                io.Int.Input("steps", default=8, min=1, max=100, advanced=True),
                io.String.Input("sigmas", default="", advanced=True,
                    tooltip="Optional explicit sigma sequence, already shifted. Empty uses simple scheduler and steps."),
                io.Float.Input("shift_video", default=12, min=0.01, advanced=True),
                io.Float.Input("shift_audio", default=6, min=0.01, advanced=True),
                io.String.Input("chat_url", default="http://127.0.0.1:9200", advanced=True),
                io.String.Input("chat_model", default="", advanced=True),
                io.String.Input("chat_api_key_env", default="", advanced=True,
                    tooltip="Environment variable name only. No API secret is saved in the workflow."),
                io.String.Input("chat_system_prompt", default="Rewrite the user's intent into a precise video generation prompt. Preserve reference numbering. Return only the prompt.", multiline=True, advanced=True),
                io.Combo.Input("sec_model", options=["", *sec_model_choices()], advanced=True),
            ], outputs=[])


def asset_schema(node_id, title, output):
    inputs = [io.String.Input("asset_id", default="", advanced=True),
              io.String.Input("local_path", default="", tooltip="Local-copy mode: path relative to this server instance's input directory.")]
    if output != ImageAsset:
        inputs += [io.Float.Input("start_seconds", default=0, min=0),
                   io.Float.Input("duration_seconds", default=0, min=0, tooltip="0: to end")]
    outputs = [output.Output("material")]
    if output == VideoAsset:
        outputs.append(AudioAsset.Output("soundtrack"))
    return io.Schema(node_id=node_id, display_name=title, category=CATEGORY, inputs=inputs, outputs=outputs)


class CanvasImage(CanvasCard):
    @classmethod
    def define_schema(cls):
        return asset_schema("TuringCanvasImage", "Canvas Image", ImageAsset)


class CanvasVideo(CanvasCard):
    @classmethod
    def define_schema(cls):
        return asset_schema("TuringCanvasVideo", "Canvas Video", VideoAsset)


class CanvasAudio(CanvasCard):
    @classmethod
    def define_schema(cls):
        return asset_schema("TuringCanvasAudio", "Canvas Audio", AudioAsset)


class CanvasH3(CanvasCard):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=H3, display_name="Canvas H3 Generate", category=CATEGORY,
            inputs=[
                ImageAsset.Input("reference_images", optional=True),
                io.Autogrow.Input("extra_images", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=ImageAsset.Input("image"), prefix="image_", min=0, max=31)),
                VideoAsset.Input("reference_video", optional=True),
                AudioAsset.Input("reference_audio", optional=True),
                ImageAsset.Input("first_frame", optional=True), ImageAsset.Input("last_frame", optional=True),
                VideoAsset.Input("target", optional=True), MaskAsset.Input("mask", optional=True),
                AudioAsset.Input("soundtrack", optional=True),
                io.DynamicCombo.Input("mode", options=[
                    io.DynamicCombo.Option("reference", []), io.DynamicCombo.Option("edit", []),
                    io.DynamicCombo.Option("extend", [io.Int.Input("prefix_frames", default=22, min=1)]),
                    io.DynamicCombo.Option("outpaint", [io.Float.Input("expand_ratio", default=0.5, min=0),
                        io.Float.Input("offset_x", default=0), io.Float.Input("offset_y", default=0)]),
                ]),
                io.String.Input("user_prompt", default="", multiline=True),
                io.String.Input("model_prompt", default="", multiline=True),
                io.Int.Input("width", default=832, min=32, step=32),
                io.Int.Input("height", default=480, min=32, step=32),
                io.Int.Input("frames", default=124, min=5, max=3600),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Boolean.Input("random_seed", default=True),
                io.Boolean.Input("preserve_audio", default=True, advanced=True),
            ], outputs=[VideoAsset.Output("video"), AudioAsset.Output("audio")])


class CanvasMask(CanvasCard):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=MASK, display_name="Canvas Video Mask", category=CATEGORY,
            inputs=[VideoAsset.Input("target"), io.String.Input("positive_coords", default=""),
                io.String.Input("negative_coords", default=""),
                io.Int.Input("annotation_frame_idx", default=0, min=-100000, max=100000),
                io.Combo.Input("tracking_direction", options=["forward", "backward", "bidirectional"]),
            ], outputs=[MaskAsset.Output("mask")])


PUBLIC_NODES = {SETTINGS: CanvasSettings, "TuringCanvasImage": CanvasImage,
                "TuringCanvasVideo": CanvasVideo, "TuringCanvasAudio": CanvasAudio,
                H3: CanvasH3, MASK: CanvasMask}
