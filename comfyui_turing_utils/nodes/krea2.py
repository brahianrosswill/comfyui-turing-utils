"""Thin Krea2 Identity Edit conditioning node."""

from __future__ import annotations

from comfy_api.latest import io

from ..adapters.krea2 import build_identity_edit_conditioning


class Krea2IdentityEditConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsKrea2IdentityEditConditioning",
            display_name="Krea2 Identity Edit Conditioning",
            category="Turing Utils/conditioning",
            description=(
                "Builds Krea2 Identity Edit conditioning from one required character "
                "reference and one optional background/edit canvas."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Latent.Input(
                    "target_latent",
                    tooltip=(
                        "Connect the same latent that feeds KSampler. It is used only "
                        "to pre-encode references at the sampling resolution."
                    ),
                ),
                io.Image.Input(
                    "character_image",
                    tooltip="Required character identity reference.",
                ),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                ),
                io.Float.Input(
                    "character_strength",
                    default=4.0,
                    min=0.0,
                    max=1000.0,
                    step=0.01,
                    tooltip=(
                        "Character reference-fidelity bias. Set to 1 for no extra "
                        "attention bias and the fastest native attention path."
                    ),
                ),
                io.Float.Input(
                    "background_strength",
                    default=1.0,
                    min=0.0,
                    max=1000.0,
                    step=0.01,
                    tooltip="Background reference-fidelity bias; 1 disables it.",
                ),
                io.Int.Input(
                    "grounding_px",
                    default=768,
                    min=0,
                    max=4096,
                    step=64,
                    tooltip="Longest side presented to Qwen3-VL; 0 keeps native size.",
                ),
                io.Image.Input(
                    "background_image",
                    optional=True,
                    tooltip=(
                        "Optional scene or edit canvas. Internally it is always ordered "
                        "before the character reference."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        vae,
        target_latent,
        character_image,
        prompt,
        character_strength=4.0,
        background_strength=1.0,
        grounding_px=768,
        background_image=None,
    ) -> io.NodeOutput:
        patched, conditioning = build_identity_edit_conditioning(
            model,
            clip,
            vae,
            target_latent,
            character_image,
            prompt,
            background_image=background_image,
            grounding_px=int(grounding_px),
            character_strength=float(character_strength),
            background_strength=float(background_strength),
        )
        return io.NodeOutput(patched, conditioning)
