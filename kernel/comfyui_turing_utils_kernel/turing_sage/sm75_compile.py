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


def sol_sparse_threshold_int8_f16_attn(
    query,
    key,
    query_int8,
    key_int8,
    value,
    output,
    query_scale,
    key_scale,
    prefix_tokens,
    threshold_sigma,
    local_block_radius,
    topology_start_tokens,
    topology_tokens,
    tokens_per_frame,
    temporal_neighbor_frames,
    query_token_offset,
    sm_scale,
):
    return _qattn_sm75.sol_sparse_threshold_int8_f16_attn(
        query,
        key,
        query_int8,
        key_int8,
        value,
        output,
        query_scale,
        key_scale,
        prefix_tokens,
        threshold_sigma,
        local_block_radius,
        topology_start_tokens,
        topology_tokens,
        tokens_per_frame,
        temporal_neighbor_frames,
        query_token_offset,
        sm_scale,
    )


def sol_sparse_route_selected(route):
    if not hasattr(_qattn_sm75, "sol_sparse_route_selected"):
        raise RuntimeError(
            "route-density debug requires comfyui-turing-utils-kernel 0.12.1 or newer"
        )
    return _qattn_sm75.sol_sparse_route_selected(route)
