"""One selected card becomes a private ordinary ComfyUI execution graph."""

import json
import secrets
import uuid

from .graph import H3, MASK, material_inputs, signature, validate_canvas, image_ports
from ..nodes.attention import sol_inputs


def compile_task(graph, task, project):
    nodes, settings = validate_canvas(graph)
    task = str(task)
    node = nodes[task]
    if node["type"] not in {H3, MASK}:
        raise ValueError("Only generation/mask cards can be queued")
    materials = material_inputs(node, nodes, project.state(), project)
    values = dict(node["values"])
    if values.get("random_seed", False):
        values["seed"] = secrets.randbits(63)
    prompt = {}

    def add(key, cls, **inputs):
        prompt[key] = {"class_type": cls, "inputs": inputs}
        return [key, 0]

    reads = {}
    for name, ref in materials.items():
        reads[name] = add("read_" + name, "_TuringCanvasRead", directory=project.directory,
            reference=json.dumps(ref), width=int(values.get("width", 0)) if name == "target" and node["type"] == H3 else 0,
            height=int(values.get("height", 0)) if name == "target" and node["type"] == H3 else 0)
        if name == "mask":
            binding = project.asset(ref["asset"])["metadata"].get("inputs", {}).get("target")
            if binding is not None and binding != materials.get("target"):
                raise ValueError("Mask belongs to a different target material or time range")

    if node["type"] == MASK:
        if "target" not in reads:
            raise ValueError("Mask generation needs a target video")
        model = add("sec", "_TuringUtilsSeCLoader", model_name=settings["sec_model"], attention="auto")
        frames = add("run_frames", "_TuringCanvasRunFrames", frames=reads["target"], run_id=uuid.uuid4().hex)
        result = add("mask", "_TuringUtilsSeCApply", model=model, frames=frames,
            positive_coords=values.get("positive_coords", ""), negative_coords=values.get("negative_coords", ""),
            tracking_direction=values.get("tracking_direction", "forward"),
            annotation_frame_idx=values.get("annotation_frame_idx", 0), max_frames_to_track=-1, semantic_keyframes=7)
        output = {"mask": result}
    else:
        for name in ("dit", "clip", "video_vae", "audio_vae"):
            if not settings.get(name):
                raise ValueError(f"Select {name} in Canvas Settings")
        model = add("dit", "TuringUtilsConvRotDiffusionModelLoader", unet_name=settings["dit"],
            force_int8_gemm=False, patch_attention=settings.get("attention", "w8a8"))
        loras = json.loads(settings.get("loras", "[]"))
        if not isinstance(loras, list):
            raise ValueError("LoRAs must be a JSON list")
        for index, lora in enumerate(loras):
            model = add(f"lora_{index}", "LoraLoaderModelOnly", model=model,
                lora_name=lora["name"], strength_model=float(lora.get("strength", 1)))
        model = add("shift", "MiniMaxH3SigmaShift", model=model,
            shift_video=settings.get("shift_video", 12), shift_audio=settings.get("shift_audio", 6))
        if settings.get("sol"):
            # Keep the backend's defaults, not a second independent strategy implementation.
            specs = sol_inputs()
            defaults = {key: spec[1]["default"] for fields in specs.values() for key, spec in fields.items()
                        if len(spec) > 1 and "default" in spec[1]}
            # Invocation is deferred to a private strategy node in the execution graph.
            model = add("sol", "_TuringCanvasSol", model=model, settings=json.dumps(defaults))
        clip = add("clip", "TuringUtilsConvRotCLIPLoader", clip_name=settings["clip"], type="minimax", force_int8_gemm=False, device="default")
        vae = add("vae", "VAELoader", vae_name=settings["video_vae"])
        audio_vae = add("audio_vae", "VAELoader", vae_name=settings["audio_vae"])
        prepare_values = {k: v for k, v in values.items() if k in
            {"width", "height", "frames", "mode", "preserve_audio"} or k.startswith("mode.")}
        prep = {"settings": json.dumps(prepare_values), "vae": vae, "audio_vae": audio_vae}
        if "target" in reads:
            prep["images"] = reads["target"]
            prep["audio"] = [reads["target"][0], 1]
        if "soundtrack" in reads:
            prep["audio"] = [reads["soundtrack"][0], 1]
        if "mask" in reads:
            prep["mask"] = [reads["mask"][0], 2]
        latent = add("prepare", "_TuringCanvasPrepare", **prep)
        refs = {}
        if image_ports(reads):
            refs["image_reference"] = add("image_ref", "TuringUtilsH3ImageReference", vae=vae, megapixels=1,
                latent=latent, **{f"images.image_{i}": reads[p] for i, p in enumerate(image_ports(reads))})
        if "reference_video" in reads:
            refs["video_reference"] = add("video_ref", "TuringUtilsH3VideoReference", video_vae=vae,
                audio_vae=audio_vae, latent=latent, megapixels=1,
                **{"videos.video_0": reads["reference_video"], "video_audios.video_audio_0": [reads["reference_video"][0], 1]})
        if "reference_audio" in reads:
            refs["audio_reference"] = add("audio_ref", "TuringUtilsH3AudioReference", audio_vae=audio_vae,
                **{"audios.audio_0": [reads["reference_audio"][0], 1]})
        for name in ("first_frame", "last_frame"):
            if name in reads:
                refs[name] = add(name, "TuringUtilsH3KeyframeReference", vae=vae, latent=latent,
                    **{"images.image_0": reads[name]})
        text = values.get("model_prompt", "").strip() or values.get("user_prompt", "")
        semantic = add("semantic", "TuringUtilsH3SemanticReference", clip=clip, prompt=text, **refs)
        conditioning = add("conditioning", "TuringUtilsH3BuildConditioning", semantic_reference=semantic, latent=latent, **refs)
        guider = add("guider", "BasicGuider", model=model, conditioning=conditioning)
        if settings.get("sigmas", "").strip():
            sigmas = add("sigmas", "ManualSigmas", sigmas=settings["sigmas"])
        else:
            sigmas = add("sigmas", "BasicScheduler", model=model, scheduler="simple", steps=settings.get("steps", 8), denoise=1.0)
        noise = add("noise", "RandomNoise", noise_seed=values.get("seed", 0))
        sampler = add("sampler", "KSamplerSelect", sampler_name="euler")
        # A run nonce forces inference, without invalidating material reads or encoders.
        noise = add("run_noise", "_TuringCanvasRunNoise", noise=noise, run_id=uuid.uuid4().hex)
        sampled = add("sample", "SamplerCustomAdvanced", noise=noise, guider=guider,
            sampler=sampler, sigmas=sigmas, latent_image=latent)
        streams = add("separate", "LTXVSeparateAVLatent", av_latent=sampled)
        images = add("decode_video", "TuringUtilsMiniMaxH3VideoVAEDecode", samples=streams, vae=vae, attention=settings.get("attention", "w8a8"))
        audio = add("decode_audio", "VAEDecodeAudio", samples=[streams[0], 1], vae=audio_vae)
        output = {"images": images, "audio": audio, "length": ["prepare", 1], "prefix": ["prepare", 2], "original_audio": ["prepare", 3]}
    snapshot = {"materials": materials, "values": values,
                "settings": {k: v for k, v in settings.items() if not k.startswith("chat_")}, "template": 1}
    add("publish", "_TuringCanvasPublish", directory=project.directory, task=task,
        signature=signature(materials), snapshot=json.dumps(snapshot), run_id=uuid.uuid4().hex, **output)
    return prompt
