# Plugin architecture

The repository has two intentionally independent lifecycles:

- `comfyui_turing_utils/` is the ComfyUI plugin implementation. Python-only
  updates do not compile or JIT CUDA code.
- `kernel/` builds and installs `comfyui-turing-utils-kernel`. Its extension
  names and public Python API are an ABI boundary for the plugin.

## Dependency direction

```text
ComfyUI registration and nodes
        ↓
attention / quantization services   ←   model adapters
        ↓                                    ↓
        └──────── kernel_api / hardware ─────┘
                         ↓
          comfyui-turing-utils-kernel
```

Concrete model adapters never belong in the generic attention or quantization
layers. `comfyui_turing_utils.__init__` is the composition root: it registers
built-in adapters without importing ComfyUI node definitions.

## Python packages

| Package | Responsibility |
|---|---|
| `attention/stable.py` | backend registry, stable bundled Sage, dtype/layout facade |
| `attention/sparse.py` | explicit experimental Sol and frame-sparse policies |
| `attention/layout.py` | model-independent packed-video topology contract and provider registry |
| `attention/patches.py` | attention overrides and loader-independent ModelPatcher installation |
| `quantization/convrot.py` | ConvRot metadata parsing and model/CLIP loading services |
| `quantization/dispatch.py` | W8A8/W4A8/W4A4 activation quantization and GEMM dispatch |
| `quantization/fusions.py` | model-independent fused activation and normalization operations |
| `adapters/minimax/` | H3 layout publication, VRAM planning, block fusions, and explicit progressive experiment |
| `adapters/wan.py` | Wan/Bernini packed-context memory planning |
| `adapters/bernini.py` | Bernini context-window and absolute-RoPE integration |
| `media/` | reference sets, resizing transforms, and shared video padding |
| `nodes/` | thin ComfyUI schemas and calls into the implementation packages |

`hardware.py` owns architecture predicates. `kernel_api.py` is the only module
allowed to import the independently installed kernel package. `registration.py`
is the only node mapping table.

## Compatibility

The legacy top-level Python modules are temporary aliases to the new package so
third-party imports continue to work. ComfyUI workflow compatibility is governed
by the stable `NODE_CLASS_MAPPINGS` keys, input names, and defaults rather than
Python filenames. New code and tests should import `comfyui_turing_utils` paths.

Sparse attention remains explicit and is never selected by loader `auto`.
Model-specific topology is installed through the attention-layout provider
registry, so the official ComfyUI loader and ConvRot loader follow the same path.
