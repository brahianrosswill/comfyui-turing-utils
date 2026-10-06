"""Branch-local H3 attention strategy; predictor bundles are not LoRAs."""

from __future__ import annotations

from pathlib import Path

from ....attention.orchestration import install_attention_strategy
from ....attention.patches import (
    _attention_layer_metadata, _prepared_qk_transform, attention_base_runtime,
)
from ....attention.protocol import AttentionExecutionOutcome
from ....attention.sparse_runtime import SparseSchedule
from ....kernel_api import kernel_extension_has_symbol
from ..layout import is_minimax_h3_model
from .engine import VedaConfig, attend
from .predictor import load_bundle


def register_predictor_folder():
    import folder_paths

    folder_paths.add_model_folder_path("veda", str(Path(folder_paths.models_dir) / "veda"))
    return folder_paths


def predictor_choices():
    folders = register_predictor_folder()
    return [name for name in folders.get_filename_list("veda") if name.endswith(".safetensors")]


def make_override(config: VedaConfig, dense_override, schedule: SparseSchedule):
    def dit_attention(options, query_tokens):
        options = options or {}
        packed = options.get("minimax_h3_layout")
        layer, _ = _attention_layer_metadata(options)
        # H3's text token-refiner also uses optimized_attention. It has no
        # Veda-trained layer/projection and must keep the original dense path.
        return layer is not None and packed is not None and packed.seq_len == query_tokens

    def run(q, k, v, transformer_options):
        options = transformer_options or {}
        layer, _ = _attention_layer_metadata(options)
        if layer is None or "minimax_h3_layout" not in options:
            raise RuntimeError("Veda requires the H3 packed layout and layer index")
        import comfy.model_prefetch

        # Predictor staging and data-dependent Top-K must not become persistent
        # allocations owned by the DiT allocation graph.
        with comfy.model_prefetch.pause_malloc_graph():
            return attend(q, k, v, config=config, packed_layout=options["minimax_h3_layout"],
                          layer=layer, head_start=int(options.get("turing_utils_attention_head_start", 0)),
                          cache=options.get("turing_utils_veda_forward_cache"),
                          heterogeneous=options.get("turing_utils_veda_heterogeneous", False))

    def override(original, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        options = kwargs.get("transformer_options")
        tokens = q.shape[2] if skip_reshape else q.shape[1]
        if mask is not None or not dit_attention(options, tokens) or schedule.is_dense(options):
            return dense_override(original, q, k, v, heads, mask=mask,
                                  attn_precision=attn_precision, skip_reshape=skip_reshape,
                                  skip_output_reshape=skip_output_reshape, **kwargs)
        if not skip_reshape:
            q, k, v = (x.reshape(x.shape[0], x.shape[1], heads, -1).transpose(1, 2)
                       for x in (q, k, v))
        result = run(q, k, v, options)
        return result if skip_output_reshape else result.transpose(1, 2).flatten(2)

    def prepared(request):
        if (request.mask is not None or request.is_causal
                or not dit_attention(request.transformer_options, request.query_tokens)
                or schedule.is_dense(request.transformer_options)):
            return dense_override.prepared_attention_executor(request)
        if request.tensor_layout != "HND" or request.observer_requirements:
            return AttentionExecutionOutcome.unsupported("Veda requires unobserved HND attention")
        if request.scale is not None and abs(request.scale - request.head_dim ** -0.5) > 1e-8:
            return AttentionExecutionOutcome.unsupported("Veda requires the model's standard attention scale")
        q, k, v = request.consume_qkv()
        q, k = _prepared_qk_transform(q, k, request.qk_transform)
        result = run(q, k, v, request.transformer_options)
        if not request.skip_output_reshape:
            result = result.transpose(1, 2).flatten(2)
        return AttentionExecutionOutcome(result)

    override.prepared_attention_executor = prepared
    return override


def forward_scope(executor, x, timestep, context, transformer_options, **kwargs):
    if transformer_options.get("turing_utils_attention_strategy") != "veda":
        return executor(x, timestep, context, transformer_options, **kwargs)
    cache = {}
    options = dict(transformer_options, turing_utils_veda_forward_cache=cache)
    try:
        return executor(x, timestep, context, options, **kwargs)
    finally:
        cache.clear()


def configure(model, *, predictor_name: str, predictor_precision: str = "w8a8",
              keep_ratio: float = 0.1, reference_keep_ratio: float = 1.0,
              plan_policy: str = "nearest", dense_prefix_steps: int = 0,
              dense_suffix_steps: int = 0, dense_prefix_layers: int = 0,
              dense_suffix_layers: int = 0, debug: bool = False,
              execution_mode: str = "auto", projection_chunk_tiles: int = 16):
    if not is_minimax_h3_model(model):
        raise ValueError("Configure H3 Veda Sparse Attention requires MiniMax H3")
    if plan_policy not in ("nearest", "strict"):
        raise ValueError("Veda plan_policy must be nearest or strict")
    if execution_mode not in ("auto", "serial", "heterogeneous", "compact_qkv", "compact_qkv_heterogeneous"):
        raise ValueError("Unsupported Veda execution_mode")
    if "heterogeneous" in execution_mode and not kernel_extension_has_symbol(
            "veda_sparse_ragged_attn", "_sage_qattn_sm75"):
        raise RuntimeError("Rebuild the Turing Utils kernel for heterogeneous Veda")
    if not 1 <= projection_chunk_tiles <= 128:
        raise ValueError("Veda projection_chunk_tiles must be in [1, 128]")
    if not 0 < keep_ratio <= 1 or not 0 < reference_keep_ratio <= 1:
        raise ValueError("Veda keep ratios must be in (0, 1]")
    if not kernel_extension_has_symbol("veda_sparse_online_attn", "_sage_qattn_sm75"):
        raise RuntimeError("Rebuild the Turing Utils CUDA kernel to enable H3 Veda")
    folders = register_predictor_folder()
    path = folders.get_full_path_or_raise("veda", predictor_name)
    bundle = load_bundle(path, predictor_precision)
    blocks = model.model.diffusion_model.blocks
    if len(blocks) != bundle.num_layers or any(
        block.attn.heads != bundle.num_heads or block.attn.head_dim != bundle.head_dim
        for block in blocks
    ):
        raise ValueError("Veda predictor layer/head dimensions do not match this H3 model")
    runtime = attention_base_runtime(model, use_w8a8=None)
    schedule = SparseSchedule(dense_prefix_steps=dense_prefix_steps,
                              dense_suffix_steps=dense_suffix_steps,
                              dense_prefix_layers=dense_prefix_layers,
                              dense_suffix_layers=dense_suffix_layers)
    config = VedaConfig(bundle, keep_ratio, reference_keep_ratio, plan_policy, debug)
    override = make_override(config, runtime.dense_override, schedule)
    installed = install_attention_strategy(model, override, strategy="Veda", backend="veda",
                                          implementation="veda:sm75_int8_qk_fp16_pv",
                                          runtime_config=runtime)
    if not installed.layout.installed:
        raise RuntimeError(f"H3 Veda layout provider unavailable: {installed.layout.reason}")
    installed.model.model_options["transformer_options"]["turing_utils_veda"] = config
    installed.model.model_options["transformer_options"]["turing_utils_veda_schedule"] = schedule
    options = installed.model.model_options["transformer_options"]
    options["turing_utils_veda_projected_qkv"] = execution_mode.startswith("compact_qkv")
    options["turing_utils_veda_heterogeneous"] = "heterogeneous" in execution_mode
    options["turing_utils_veda_projection_chunk_tiles"] = projection_chunk_tiles
    options["turing_utils_veda_auto_schedule"] = execution_mode == "auto"
    import comfy.patcher_extension

    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    installed.model.remove_wrappers_with_key(wrapper_type, "turing_utils_veda_scope")
    installed.model.add_wrapper_with_key(wrapper_type, "turing_utils_veda_scope", forward_scope)
    return installed.model
