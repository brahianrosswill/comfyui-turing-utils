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
        query, key, value, output, query_scale, key_scale,
        tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse
    )


def qk_int8_sv_f16_accum_f16_attn(
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
    return _qattn_sm75.qk_int8_sv_f16_accum_f16_attn(
        query, key, value, output, query_scale, key_scale,
        tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse
    )


def qk_int8_sv_f16_accum_f16_attn_inst_buf(
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
    return _qattn_sm75.qk_int8_sv_f16_accum_f16_attn_inst_buf(
        query, key, value, output, query_scale, key_scale,
        tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse
    )


def qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(
    query,
    key,
    value,
    output,
    query_scale,
    key_scale,
    value_mean,
    tensor_layout,
    is_causal,
    qk_quant_gran,
    sm_scale,
    return_lse,
):
    return _qattn_sm75.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(
        query, key, value, output, query_scale, key_scale, value_mean,
        tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse
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
        query, key, value, output, query_scale, key_scale,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        is_causal, sm_scale
    )


def qk_int4_sv_f16_accum_f16_f32_attn(
    query,
    key,
    value,
    output,
    query_scale,
    key_scale,
    key_original,
    query_mean,
    key_mean,
    tensor_layout,
    is_causal,
    sm_scale,
    return_lse,
    smooth_q,
    smooth_k,
):
    return _qattn_sm75.qk_int4_sv_f16_accum_f16_f32_attn(
        query, key, value, output, query_scale, key_scale,
        key_original, query_mean, key_mean, tensor_layout, is_causal,
        sm_scale, return_lse, smooth_q, smooth_k
    )


def qk_int4_sv_f16_accum_f16_f32_precomputed_attn(
    query,
    key,
    value,
    output,
    query_scale,
    key_scale,
    score_correction,
    tensor_layout,
    is_causal,
    sm_scale,
    return_lse,
    q_block_start,
    q_block_count,
):
    return _qattn_sm75.qk_int4_sv_f16_accum_f16_f32_precomputed_attn(
        query, key, value, output, query_scale, key_scale,
        score_correction, tensor_layout, is_causal, sm_scale, return_lse,
        q_block_start, q_block_count
    )


# Compatibility for 0.6.0 callers. The unsafe sequence-long FP16 accumulator
# is intentionally no longer selected by the public Sage2 facade.
qk_int4_sv_f16_accum_f16_attn = qk_int4_sv_f16_accum_f16_f32_attn
