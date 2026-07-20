# SVDInt4 Model Format

SVDInt4 has one public safetensors format identifier:

```text
format = svdint4
```

There are no version, container, model-family, or layout suffixes. Loaders must
reject every other value. Breaking layout changes replace this specification
and its implementation together; this project does not provide aliases for
older identifiers.

## Required metadata

```text
format = svdint4
architecture = wan
```

`architecture` selects the model key mapping independently from the storage
format. The current loader implements `wan`, which covers supported Wan,
Wan2.2, Bernini, and SCAIL-derived DiT checkpoints. New architecture mappings
must add a new `architecture` value without creating a new `format` value.

Release name, branch, recipe, calibration method, rank, provenance, and model
version are not format identity. They may be stored as additional metadata or
in a sidecar JSON.

## Quantized Linear tensors

For a logical Linear weight with shape `[N, K]` and padded SVD rank `R`, the
published tensor names and storage are:

| Suffix | Shape | Dtype | Meaning |
|---|---:|---:|---|
| `.qweight` | `[N, K / 2]` | INT8 | Two signed INT4 residual weights per byte |
| `.wscales` | `[K / 64, N]` or an equivalent packed 2D view | FP16 | Group-64 weight scales |
| `.smooth` | `[K]` | FP16 | Packed per-input-channel smooth factors |
| `.svd_down` | `[K, R]` | FP16 | Packed low-rank down projection |
| `.svd_up` | `[N, R]` | FP16 | Packed low-rank up projection |
| `.bias_packed` | `[N]` | FP16 | Optional packed bias |

`.qweight`, `.wscales`, `.svd_down`, and `.svd_up` are required for every
quantized Linear. `.smooth` and `.bias_packed` are optional at the storage
level; the loader supplies identity smooth or zero bias behavior when absent.
`N` is padded to a multiple of 128, `K` to a multiple of 64, and `R` to a
multiple of 16.

Non-quantized tensors retain the normal architecture keys and dtypes expected
by ComfyUI. One file contains one DiT branch; high-noise and low-noise branches
remain separate files.

## Architecture: `wan`

The public quantized Linear bases use storage-style Wan names:

```text
blocks.N.self_attn.{q,k,v,o}
blocks.N.cross_attn.{q,k,v,o}
blocks.N.ffn.{0,2}
```

All other tensors use their normal Wan/ComfyUI keys. The loader validates every
packed shape and rejects packed layers that the selected ComfyUI architecture
does not consume.

## Data-free and calibrated weights

Both paths produce exactly the same format. Calibration status does not alter
`format` or `architecture`:

- Data-free conversion derives a weight-only low-rank correction and writes
  unit smooth factors.
- Calibrated conversion collects real activation statistics, derives smooth
  factors, then quantizes the residual with those factors.

Calibration and release eligibility belong in metadata such as `policy`,
`calibration_mode`, and `calibration_sample_count`, not in the format string.
