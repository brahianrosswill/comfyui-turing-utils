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
    query,
    key,
    query_int8,
    key_int8,
    value,
    output,
    query_scale,
    key_scale,
    sparse_query_blocks,
    exact_kv_blocks,
    threshold_sigma,
    residual_subblocks,
    sm_scale,
    return_stats,
):
    if not hasattr(_qattn_sm75, "sol_sparse_online_int8_f16_attn"):
        raise RuntimeError(
            "online Sol routing requires comfyui-turing-utils-kernel 0.16.0 or newer"
        )
    return _qattn_sm75.sol_sparse_online_int8_f16_attn(
        query,
        key,
        query_int8,
        key_int8,
        value,
        output,
        query_scale,
        key_scale,
        sparse_query_blocks,
        exact_kv_blocks,
        threshold_sigma,
        residual_subblocks,
        sm_scale,
        return_stats,
    )


def frame_sparse_int8_f16_attn(
    query_int8,
    key_int8,
    value,
    output,
    query_scale,
    key_scale,
    row_offsets,
    key_blocks,
    sm_scale,
):
    if not hasattr(_qattn_sm75, "frame_sparse_int8_f16_attn"):
        raise RuntimeError(
            "frame-sparse attention requires comfyui-turing-utils-kernel 0.15.0 or newer"
        )
    return _qattn_sm75.frame_sparse_int8_f16_attn(
        query_int8,
        key_int8,
        value,
        output,
        query_scale,
        key_scale,
        row_offsets,
        key_blocks,
        sm_scale,
    )
