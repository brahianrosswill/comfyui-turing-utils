"""ComfyUI nodes for explicit experimental attention patches."""

from __future__ import annotations

from ..attention import apply_frame_sparse_attention_patch, apply_sparse_attention_patch


class SolSparseAttentionPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "routing_threshold": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -4.0,
                        "max": 4.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "Route blocks whose input-adaptive proxy score exceeds mean + threshold × standard deviation. Lower values preserve more exact blocks; 1.0 matches the official Sol policy.",
                    },
                ),
                "prefix_policy": (
                    ["auto", "none", "manual"],
                    {
                        "default": "auto",
                        "tooltip": "Auto applies the model's semantic segments and the three reference switches; none protects no modality ranges; manual protects only the leading token count below.",
                    },
                ),
                "manual_prefix_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 262144,
                        "step": 64,
                        "tooltip": "Leading Query tokens kept dense and leading K/V tokens kept exact only for manual policy. Boundaries round outward to 64-token blocks.",
                    },
                ),
                "skipped_residual": (
                    ["1x64", "2x32"],
                    {
                        "default": "1x64",
                        "tooltip": "Official-style 1x64 uses one K/V centroid per skipped block. 2x32 keeps two residual centroids for higher approximation quality without changing routing.",
                    },
                ),
                "sparse_reference_image": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Allow reference-image Query/KV interactions to use Sol routing. Disabled protects keyframes and reference images with dense Query and exact KV blocks.",
                    },
                ),
                "sparse_reference_video": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Allow long reference-video Query/KV interactions to use Sol routing instead of protecting the complete reference video.",
                    },
                ),
                "sparse_reference_audio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Allow reference-audio Query/KV interactions to use Sol routing. Disabled preserves reference audio and dialogue conditioning exactly.",
                    },
                ),
                "dense_prefix_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of early denoising steps that use the selected dense backend across every transformer layer (stable Sage, or W8A8 when enabled).",
                    },
                ),
                "dense_suffix_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of final denoising steps that use the selected dense backend across every transformer layer (stable Sage, or W8A8 when enabled).",
                    },
                ),
                "dense_prefix_layers": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Keep this many transformer layers at the beginning of every sparse step on the selected dense backend.",
                    },
                ),
                "dense_suffix_layers": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Keep this many transformer layers at the end of every sparse step on the selected dense backend. Requires layer-count metadata.",
                    },
                ),
            },
            "optional": {
                "use_w8a8": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use signed INT8 V and unsigned INT8 probability Tensor Cores for Sol exact blocks and protected dense steps/layers. Experimental; disabled keeps the stable FP16/BF16 PV path.",
                    },
                ),
                "debug_route_density": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Log min/mean/max route density once per denoising step. Disabled by default; enabling it adds tiny reductions and one synchronization per step.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Turing Utils/patches"
    TITLE = "Patch Sol Sparse Attention (Experimental)"

    def patch(
        self,
        model,
        routing_threshold: float = 1.0,
        prefix_policy: str = "auto",
        manual_prefix_tokens: int = 0,
        skipped_residual: str = "1x64",
        sparse_reference_image: bool = False,
        sparse_reference_video: bool = True,
        sparse_reference_audio: bool = False,
        dense_prefix_steps: int = 0,
        dense_suffix_steps: int = 0,
        dense_prefix_layers: int = 2,
        dense_suffix_layers: int = 0,
        use_w8a8: bool = False,
        debug_route_density: bool = False,
    ):
        return (
            apply_sparse_attention_patch(
                model,
                routing_threshold=routing_threshold,
                prefix_policy=prefix_policy,
                manual_prefix_tokens=manual_prefix_tokens,
                skipped_residual=skipped_residual,
                sparse_reference_image=sparse_reference_image,
                sparse_reference_video=sparse_reference_video,
                sparse_reference_audio=sparse_reference_audio,
                dense_prefix_steps=dense_prefix_steps,
                dense_suffix_steps=dense_suffix_steps,
                dense_prefix_layers=dense_prefix_layers,
                dense_suffix_layers=dense_suffix_layers,
                use_w8a8=use_w8a8,
                debug_route_density=debug_route_density,
            ),
        )


class FrameSparseAttentionPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "quality_profile": (
                    ["custom", "conservative", "balanced", "fast"],
                    {
                        "default": "custom",
                        "tooltip": "Custom preserves every control below. Presets replace the pattern, temporal/spatial coverage, anchors, sinks, and protected layer counts with coherent quality/speed settings.",
                    },
                ),
                "sparse_pattern": (
                    ["frame_window", "radial"],
                    {
                        "default": "frame_window",
                        "tooltip": "Frame window selects complete nearby/anchor frames. Radial also samples distant frames with 2D spatial locality and logarithmically increasing temporal stride.",
                    },
                ),
                "prefix_policy": (
                    ["auto", "none", "manual"],
                    {
                        "default": "auto",
                        "tooltip": "Auto keeps adapter-provided text, reference, and audio K/V tokens exact for every video query. Non-video queries always remain dense.",
                    },
                ),
                "manual_prefix_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 262144,
                        "step": 64,
                        "tooltip": "Leading K/V tokens kept exact only when prefix_policy is manual. Rounded up to 64-token blocks.",
                    },
                ),
                "temporal_window_frames": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Attend every spatial token in this many neighboring latent frames on each side, including the complete current frame.",
                    },
                ),
                "global_anchor_stride": (
                    "INT",
                    {
                        "default": 12,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Attend one complete global anchor frame every N latent frames. Zero disables periodic anchors.",
                    },
                ),
                "rotate_global_anchors": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Shift periodic full-frame anchors, or the radial temporal sampling phase, by transformer layer so distant information propagates without a fixed grid.",
                    },
                ),
                "sink_frames": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Always attend this many initial target-video latent frames as stable scene anchors.",
                    },
                ),
                "radial_spatial_radius": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 8,
                        "step": 1,
                        "tooltip": "For radial distant frames, expand around matching 8x8 spatial token tiles by this many tiles. Ignored by frame_window.",
                    },
                ),
                "radial_max_temporal_stride": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Maximum temporal subsampling stride for distant radial bands. Lower values preserve more distant frames; ignored by frame_window.",
                    },
                ),
                "dense_prefix_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of early denoising steps that use stable dense Sage across every transformer layer.",
                    },
                ),
                "dense_suffix_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of final denoising steps that use stable dense Sage across every transformer layer.",
                    },
                ),
                "dense_prefix_layers": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Keep this many first transformer layers on stable dense Sage.",
                    },
                ),
                "dense_suffix_layers": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Keep this many final transformer layers on stable dense Sage.",
                    },
                ),
            },
            "optional": {
                "debug_route_density": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Log the static block density once per distinct shape and anchor offset. It does not synchronize the GPU.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Turing Utils/patches"
    TITLE = "Patch Sage Frame Sparse Attention (Experimental)"

    def patch(
        self,
        model,
        quality_profile: str = "custom",
        sparse_pattern: str = "frame_window",
        prefix_policy: str = "auto",
        manual_prefix_tokens: int = 0,
        temporal_window_frames: int = 2,
        global_anchor_stride: int = 12,
        rotate_global_anchors: bool = True,
        sink_frames: int = 1,
        radial_spatial_radius: int = 1,
        radial_max_temporal_stride: int = 16,
        dense_prefix_steps: int = 0,
        dense_suffix_steps: int = 0,
        dense_prefix_layers: int = 1,
        dense_suffix_layers: int = 1,
        debug_route_density: bool = False,
    ):
        return (
            apply_frame_sparse_attention_patch(
                model,
                quality_profile=quality_profile,
                sparse_pattern=sparse_pattern,
                prefix_policy=prefix_policy,
                manual_prefix_tokens=manual_prefix_tokens,
                temporal_window_frames=temporal_window_frames,
                global_anchor_stride=global_anchor_stride,
                rotate_global_anchors=rotate_global_anchors,
                sink_frames=sink_frames,
                radial_spatial_radius=radial_spatial_radius,
                radial_max_temporal_stride=radial_max_temporal_stride,
                dense_prefix_steps=dense_prefix_steps,
                dense_suffix_steps=dense_suffix_steps,
                dense_prefix_layers=dense_prefix_layers,
                dense_suffix_layers=dense_suffix_layers,
                debug_route_density=debug_route_density,
            ),
        )
