from .. import _sage_qattn_sm75 as _qattn_sm75


def qk_int8_sv_f16_accum_f32_attn(
    query,
    key,
    value,
    output,
    query_scale,
    key_scale,
    tensor_layout,
    is_causal,
    qk_quant_gran,
    sm_scale,
    return_lse,
):
    return _qattn_sm75.qk_int8_sv_f16_accum_f32_attn(
        query,
        key,
        value,
        output,
        query_scale,
        key_scale,
        tensor_layout,
        is_causal,
        qk_quant_gran,
        sm_scale,
        return_lse,
    )

def qk_int8_sv_f16_varlen_accum_f32_attn(
    query,
    key,
    value,
    output,
    query_scale,
    key_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    is_causal,
    sm_scale,
):
    return _qattn_sm75.qk_int8_sv_f16_varlen_accum_f32_attn(
        query,
        key,
        value,
        output,
        query_scale,
        key_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal,
        sm_scale,
    )


def sol_sparse_online_int8_f16_attn(
    query_int8,
    key_int8,
    value,
    value_int8,
    value_scale,
    output,
    query_scale,
    key_scale,
    sparse_query_blocks,
    exact_kv_blocks,
    threshold_sigma,
    residual_subblocks,
    sm_scale,
    return_stats,
    use_w8a8,
    force_dense,
):
    if not hasattr(_qattn_sm75, "sol_sparse_online_int8_f16_attn"):
        raise RuntimeError(
            "online Sol routing requires comfyui-turing-utils-kernel 0.17.0 or newer"
        )
    return _qattn_sm75.sol_sparse_online_int8_f16_attn(
        query_int8,
        key_int8,
        value,
        value_int8,
        value_scale,
        output,
        query_scale,
        key_scale,
        sparse_query_blocks,
        exact_kv_blocks,
        threshold_sigma,
        residual_subblocks,
        sm_scale,
        return_stats,
        use_w8a8,
        force_dense,
    )


def quantize_v_int8(value, quantized, scale):
    if not hasattr(_qattn_sm75, "quantize_v_int8_sm75"):
        raise RuntimeError(
            "W8A8 attention requires comfyui-turing-utils-kernel 0.18.0 or newer"
        )
    _qattn_sm75.quantize_v_int8_sm75(value, quantized, scale)


def sol_w8a8_precompute_summaries(
    key_int8,
    key_scale,
    value,
    value_scale,
    residual_subblocks,
):
    if not hasattr(_qattn_sm75, "sol_w8a8_precompute_summaries"):
        raise RuntimeError(
            "split Sol W8A8 requires comfyui-turing-utils-kernel 0.19.0 or newer"
        )
    return _qattn_sm75.sol_w8a8_precompute_summaries(
        key_int8,
        key_scale,
        value,
        value_scale,
        residual_subblocks,
    )


def sol_sparse_online_w8a8_prequantized_attn(
    query_int8,
    key_int8,
    value_int8,
    value_scale,
    output,
    query_scale,
    key_scale,
    summaries,
    sparse_query_blocks,
    exact_kv_blocks,
    threshold_sigma,
    residual_subblocks,
    sm_scale,
    return_stats,
    force_dense,
):
    if not hasattr(_qattn_sm75, "sol_sparse_online_w8a8_prequantized_attn"):
        raise RuntimeError(
            "split Sol W8A8 requires comfyui-turing-utils-kernel 0.19.0 or newer"
        )
    (
        key_summary,
        key_score_summary,
        value_mean,
        key_summary_mean,
        key_summary_variance,
    ) = summaries
    return _qattn_sm75.sol_sparse_online_w8a8_prequantized_attn(
        query_int8,
        key_int8,
        value_int8,
        value_scale,
        output,
        query_scale,
        key_scale,
        key_summary,
        key_score_summary,
        value_mean,
        key_summary_mean,
        key_summary_variance,
        sparse_query_blocks,
        exact_kv_blocks,
        threshold_sigma,
        residual_subblocks,
        sm_scale,
        return_stats,
        force_dense,
    )
