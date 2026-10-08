"""Hidden execution graph stages. They reuse existing model and VAE contracts."""

import json
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import InputImpl, Types
from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV
from comfy_extras.nodes_audio import VAEEncodeAudio
from comfy_extras.nodes_lt import LTXVConcatAVLatent

from ..nodes.latent import SetVideoLatentNoiseMask
from ..nodes.minimax_vae import MiniMaxH3VideoVAEEncode
from ..nodes.video_padding import VideoFramesPadding
from ..nodes.video_roi import pad_video_for_outpaint
from .media import read_material
from .store import Project
from ..nodes.attention import _ATTENTION_STRATEGIES


class ReadAsset:
    CATEGORY = ""
    FUNCTION = "read"
    RETURN_TYPES = ("IMAGE", "AUDIO", "MASK")
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"directory": ("STRING",), "reference": ("STRING",),
            "width": ("INT", {"default": 0}), "height": ("INT", {"default": 0})}}

    def read(self, directory, reference, width=0, height=0):
        return read_material(Project(directory), json.loads(reference), width, height)


class PrepareH3:
    CATEGORY = ""
    FUNCTION = "prepare"
    RETURN_TYPES = ("LATENT", "INT", "INT", "AUDIO")
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"settings": ("STRING",), "vae": ("VAE",), "audio_vae": ("VAE",)},
                "optional": {"images": ("IMAGE",), "mask": ("MASK",), "audio": ("AUDIO",)}}

    def prepare(self, settings, vae, audio_vae, images=None, mask=None, audio=None):
        cfg = json.loads(settings)
        width, height, count = cfg["width"], cfg["height"], cfg["frames"]
        mode, prefix = cfg["mode"], 0
        if mode != "reference" and images is None:
            raise ValueError(f"{mode} requires target video")
        if mode == "outpaint":
            images, mask = pad_video_for_outpaint(images, {
                "layout": "relative_frame", "expand_ratio": cfg.get("mode.expand_ratio", 0.5),
                "offset_x": cfg.get("mode.offset_x", 0), "offset_y": cfg.get("mode.offset_y", 0),
                "allow_image_outside_frame": False}, width * height / 1e6, 32, "edge")
        if mode == "extend":
            original_count = len(images)
            prefix = min(len(images), int(cfg.get("mode.prefix_frames", 22)))
            context = images[-prefix:]
            images = torch.cat([context, context[-1:].repeat(count, 1, 1, 1)])
            mask = torch.ones(images.shape[:3], dtype=images.dtype, device=images.device)
            mask[:prefix] = 0
            if audio is not None:
                rate = audio["sample_rate"]
                start = round((original_count - prefix) * rate / 24)
                audio = {**audio, "waveform": audio["waveform"][..., start:round(original_count * rate / 24)]}
        if mode != "reference":
            count, height, width = images.shape[:3]
            if mask is None:
                mask = torch.ones(images.shape[:3], dtype=images.dtype, device=images.device)
            if mask.shape[0] == 1:
                mask = mask.expand(count, -1, -1)
            if mask.shape[0] != count:
                raise ValueError("Mask and target video intervals do not have matching frame counts")
            if mask.shape[1:] != (height, width):
                mask = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="nearest").squeeze(1)
            padded = VideoFramesPadding.execute(type="minimax", image=images, mask=mask).result
            video = MiniMaxH3VideoVAEEncode().encode(padded[0], vae)[0]
            video = SetVideoLatentNoiseMask.execute(video, padded[1], "minimax").result[0]
            length = padded[4]
        else:
            length = count
        empty = EmptyMiniMaxH3LatentAV.execute(width, height, length).result[0]
        empty_video, empty_audio = empty["samples"].unbind()
        if mode == "reference":
            video = {"samples": empty_video}
        preserved = audio if cfg.get("preserve_audio", True) else None
        if preserved is not None:
            waveform = preserved["waveform"]
            samples = round((5 + 17 * max(0, (length - 5 + 16) // 17)) * preserved["sample_rate"] / 24)
            waveform = F.pad(waveform[..., :samples], (0, max(0, samples - waveform.shape[-1])))
            sound = VAEEncodeAudio.execute(audio_vae, {**preserved, "waveform": waveform}).result[0]
            sound["noise_mask"] = torch.zeros_like(sound["samples"])
            if mode == "extend":
                boundary = round(prefix / 24 * preserved["sample_rate"] * sound["samples"].shape[-1] / samples)
                sound["noise_mask"][..., boundary:] = 1
                preserved = None  # Decode the generated body, not the source prefix soundtrack.
        else:
            sound = {"samples": empty_audio}
        latent = LTXVConcatAVLatent.execute(video, sound).result[0]
        return latent, count, prefix, preserved


class Publish:
    CATEGORY = ""
    FUNCTION = "publish"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"directory": ("STRING",), "task": ("STRING",),
            "signature": ("STRING",), "snapshot": ("STRING",), "run_id": ("STRING",)},
            "optional": {"images": ("IMAGE",), "audio": ("AUDIO",), "original_audio": ("AUDIO",),
                         "mask": ("MASK",), "length": ("INT",), "prefix": ("INT",)}}

    def publish(self, directory, task, signature, snapshot, run_id, images=None,
                audio=None, original_audio=None, mask=None, length=0, prefix=0):
        project = Project(directory)
        cfg = json.loads(snapshot)
        if mask is not None:
            asset_id, path = project.reserve(".npy")
            with path.open("wb") as file:
                np.save(file, mask.detach().cpu().numpy(), allow_pickle=False)
            kind = "mask"
        else:
            images = images[prefix:length or None]
            audio = original_audio if original_audio is not None else audio
            if audio is not None:
                rate = audio["sample_rate"]
                waveform = audio["waveform"][..., round(prefix * rate / 24):round((prefix + len(images)) * rate / 24)]
                expected = round(len(images) * rate / 24)
                waveform = F.pad(waveform, (0, max(0, expected - waveform.shape[-1])))
                audio = {**audio, "waveform": waveform}
            asset_id, path = project.reserve(".mp4")
            video = InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=audio,
                frame_rate=Fraction(24)), bit_depth=8, color_space="sRGB")
            video.save_to(str(path), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264, crf=19)
            kind = "video"
        project.register(asset_id, path, kind, f"{task}-{run_id}", {"inputs": cfg["materials"]})
        project.publish(task, asset_id, signature, cfg)
        return {"ui": {"canvas_result": [{"task": task, "asset": asset_id}]}, "result": (asset_id,)}


class RunNoise:
    CATEGORY = ""
    FUNCTION = "run"
    RETURN_TYPES = ("NOISE",)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"noise": ("NOISE",), "run_id": ("STRING",)}}

    def run(self, noise, run_id):
        return (noise,)


class Sol:
    CATEGORY = ""
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL",)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "settings": ("STRING",)}}

    def apply(self, model, settings):
        return (_ATTENTION_STRATEGIES["sol"](model, **json.loads(settings)),)


class RunFrames:
    CATEGORY = ""
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames": ("IMAGE",), "run_id": ("STRING",)}}

    def run(self, frames, run_id):
        return (frames,)


INTERNAL_NODES = {"_TuringCanvasRead": ReadAsset, "_TuringCanvasPrepare": PrepareH3,
                  "_TuringCanvasPublish": Publish, "_TuringCanvasRunNoise": RunNoise, "_TuringCanvasSol": Sol,
                  "_TuringCanvasRunFrames": RunFrames}
for _node in INTERNAL_NODES.values():
    _node.DEV_ONLY = True
