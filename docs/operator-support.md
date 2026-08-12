# Operator and runtime support

This document is the source of truth for the production operator surface. It
separates public custom operators from Python dispatch and model-specific
adapters; supporting a quantization format does not imply that every GEMM is
implemented locally.

## Production stack

```text
ComfyUI nodes
    -> generic attention / quantization services
    -> optional MiniMax, Wan, or Bernini adapter
    -> kernel_api (the only independent-kernel import boundary)
    -> comfyui-turing-utils-kernel
```

The plugin and kernel package remain independently installable. A Python-only
plugin update does not rebuild CUDA. Sparse attention and W8A8 attention remain
explicit opt-ins; loader `auto` selects the stable bundled Sage path on a
supported Turing GPU.

## Public custom operators

Every operator below is registered with `torch.library.custom_op` and a fake
implementation.

| Family | `turing_utils::` operator | Purpose |
|---|---|---|
| Linear | `w4a8_linear` | Packed INT4 weights with INT8 activations on SM75 Tensor Cores; BF16 output |
| Epilogue | `dequantize_int8_bf16` | INT32 GEMM workspace to packed BF16 output |
| Activation quantization | `swiglu_int8_convrot_quantize`, `swiglu_int4_convrot_quantize` | Fused SwiGLU and ConvRot activation quantization |
| Activation quantization | `gelu_int8_convrot_quantize`, `gelu_int4_convrot_quantize` | Fused tanh-GELU and ConvRot activation quantization |
| Activation quantization | `bf16_int8_convrot_quantize`, `bf16_int4_convrot_quantize` | BF16 row-buffer ConvRot quantization, optionally with SwiGLU |
| Activation quantization | `bf16_gelu_int8_convrot_quantize`, `bf16_gelu_int4_convrot_quantize` | BF16 row-buffer GELU and ConvRot quantization |
| Normalization | `segmented_rms_adaln` | RMSNorm plus segmented AdaLN modulation |
| Normalization | `layer_norm_adaln` | LayerNorm plus AdaLN modulation |
| Attention preprocessing | `qk_rms_rope_int8` | Fused RMSNorm, RoPE and production Q/K INT8 quantization |
| Attention | `sage_attention` | Stable dense INT8-QK, FP16/BF16-PV SM75 attention |
| Attention | `sage_attention_varlen` | Packed variable-length stable Sage attention |
| Attention | `w8a8_attention` | Dense INT8-QK and INT8-PV SM75 attention |
| Attention | `w8a8_attention_varlen` | Packed variable-length dense W8A8 attention |
| Attention | `sol_attention` | Online Sol routing with FP16/BF16-PV or INT8-PV |

W4A4 and the main W8A8 linear contraction deliberately reuse Comfy Kitchen or
cuBLAS. The local package supplies the Turing-specific quantization, BF16
epilogue, activation fusions, dispatch, and W4A8 contraction. It does not carry
duplicate full W4A4 and W8A8 GEMM implementations.

## Attention feature matrix

| Feature | Stable Sage | Dense W8A8 | Sol FP16-PV | Sol W8A8 |
|---|---:|---:|---:|---:|
| Production status | default on Turing | explicit | experimental patch | experimental patch option |
| Input storage | FP16/BF16 | FP16/BF16 | FP16/BF16 | FP16/BF16 |
| QK score domain | INT8, FP32 accumulation | INT8, FP32 accumulation | INT8-consistent routing and score | INT8-consistent routing and score |
| PV path | FP16 Tensor Core, FP32 accumulation | U8 x S8 Tensor Core | FP16 Tensor Core | U8 x S8 Tensor Core |
| Output dtype | input V dtype | input V dtype | input V dtype | input V dtype |
| Head dimension | 1--128, native D64/D128 | 1--128, native D64/D128 | 1--128, native D64/D128 | 1--128, native D64/D128 |
| HND / NHD | yes / yes | yes / yes | HND only | HND only |
| GQA | yes | yes | yes | yes |
| Unequal Q/K length | yes | yes | yes | yes |
| Causal | yes | yes (upper-left) | no | no |
| Arbitrary attention mask | no | no | no | no |
| Variable length | yes | yes (packed) | no | no |
| Split prequantization | Q/K; V retained | Q/K/V | Q/K; V retained | Q/K/V plus summaries |
| Q/K Hadamard rotation | no | optional, default on | optional, default on | optional, default on |
| Adaptive K anchor | no | optional, default on | optional, default on | optional, default on |
| Exact modal KV ranges | n/a | n/a | yes | yes |

FP32 is accepted at the plugin boundary only for ComfyUI's Turing BF16
fallback case; it is converted to BF16 storage before entering these kernels.
The kernels do not advertise native FP32 attention.

The D64 attention specializations use 16 KiB shared memory and D128 uses 32
KiB. The long-key 128-token schedule reuses a 64-token shared tile rather than
doubling the CTA allocation. Exact occupancy and throughput still require a
real SM75 profile.

## Quantized linear support

| Format | Weight storage | Activation | Contraction owner | Local Turing additions |
|---|---|---|---|---|
| W8A8 | INT8 | INT8 | Kitchen/cuBLAS | ConvRot quantization, fused SwiGLU/GELU, BF16 epilogue, workspace policy |
| W4A4 | packed INT4 | INT4 | Kitchen | Turing BF16 input/output compatibility and fused activation quantization |
| W4A8 | packed INT4 | INT8 | local exact-SM75 kernel | packed-weight load, INT8 Tensor Core MMA, BF16 output |

ConvRot group size 256 is the optimized H3 path. Unsupported layouts, group
sizes, devices, or dtypes are rejected or delegated to Kitchen according to the
format contract; dense checkpoint weights are never silently quantized at
runtime.

## Model-specific integration

- MiniMax H3 publishes packed text/image/video/audio token ranges, estimates
  packed dynamic-VRAM allocations, fuses segmented RMSNorm+AdaLN, and fuses
  fc1 -> SwiGLU -> fc2-input quantization. Supported attention calls also fuse
  per-head RMSNorm+split-half RoPE directly into INT8 Q/K. Progressive
  resolution remains an explicit experiment.
- Wan publishes packed-context memory estimates and fuses its whole-row Q/K
  RMSNorm plus interleaved RoPE into Sage/W8A8/Sol Q/K quantization.
- Bernini provides conditioning, context-window memory estimation, and
  absolute-position integration. Its Wan self-attention inherits the same
  generic fused preprocessing and tensor-lifetime path, including explicitly
  selected Sol; it does not own
  quantized CUDA kernels.
- Reference hubs, padding, resizing, and H3 AV latent utilities are media or
  conditioning operations, not CUDA kernel families.

## Optimization status

| Work item | Status | Decision |
|---|---|---|
| AttentionTensorContainer lifetime | complete | Dense Sage, dense W8A8, and Sol consume containers only after preflight; quantized Q/K replace floating Q/K before output allocation, W8A8 also releases floating V, while FP16-PV correctly retains V until the attention CTA consumes it |
| Fused Q/K RMSNorm + RoPE + INT8 quantization | complete | H3, Wan and Bernini publish model semantics through one adapter contract; dense Sage/W8A8 and explicit Sol share the same D64/D128 custom op |
| fc2/out_proj gate + residual epilogue | not implemented | high-value H3 adapter work, but requires a contraction-owned epilogue rather than another post-GEMM kernel |
| Head/layer/step-specific Sol policy | not implemented | useful only behind calibration and visual/audio gates; do not replace the stable global policy without H3 data |
| Route reuse and hysteresis | not implemented | potentially useful, but must justify route-state memory and avoid cross-prompt state leakage |
| Kitchen versus bundled SM75 A/B | directionally tested on A40 | final backend policy still requires exact-SM75 profiling |
| First-block or trajectory cache | not integrated | not a default fit for 6--8-step H3; leave to dedicated high-step nodes |
| Spectrum-style transformer skipping | not integrated | replay memory and audio workflow cost conflict with the Turing/dynamic-VRAM target |

The next production-oriented order is: gated fc2/out-projection epilogues,
calibration tooling for head/layer/step Sol
budgets, then real-Turing backend A/B. The first two target tensor traffic in
the MLP/projection-heavy part of H3 and therefore have a higher whole-model
ceiling than another dense attention variant.
