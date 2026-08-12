# Plugin architecture

The repository has two intentionally independent lifecycles:

- `comfyui_turing_utils/` is the ComfyUI plugin implementation. Python-only
  updates do not compile or JIT CUDA code.
- `kernel/` builds and installs `comfyui-turing-utils-kernel`. Its extension
  names and public Python API are an ABI boundary for the plugin.

The maintained source tree is intentionally shallow:

```text
comfyui-turing-utils/
├── __init__.py                  # ComfyUI entry point
├── attention.py                 # sole legacy Python compatibility facade
├── comfyui_turing_utils/
│   ├── attention/               # dense/sparse backends, layout and patches
│   ├── quantization/            # ConvRot loading, dispatch and fusions
│   ├── adapters/                # MiniMax, Wan and Bernini integration
│   ├── media/                   # references, resize and padding
│   ├── nodes/                   # thin ComfyUI schemas
│   ├── hardware.py
│   ├── kernel_api.py            # independent-kernel boundary
│   ├── precision.py
│   ├── profiling.py             # disabled-by-default bounded CUDA timing
│   └── registration.py          # sole node mapping table
├── kernel/
│   ├── comfyui_turing_utils_kernel/
│   │   ├── ops.py               # linear/fusion custom ops
│   │   └── turing_sage/         # attention API, quantization and scheduling
│   ├── csrc/turing/             # exact-SM75 CUDA/C++ sources
│   ├── scripts/                 # validation and benchmarks
│   └── setup.py
├── docs/
└── tests/
```

There are no model or node implementation modules at repository root. New
functionality belongs in one existing package above unless it introduces a
genuinely new responsibility.

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
| `attention/stable.py` | backend registry, stable bundled Sage/W8A8, dtype/layout facade |
| `attention/sparse.py` | explicit experimental Sol policy, including optional W8A8 PV dispatch |
| `attention/protocol.py` | versioned tensor ownership, transform, capability, and execution contract |
| `attention/integration.py` | model-neutral projected-QKV handoff and attention-site registry |
| `attention/layout.py` | versioned Query/KV modality topology contract and provider registry |
| `attention/patches.py` | attention overrides and loader-independent ModelPatcher installation |
| `attention/tuning.py` | explicit experimental SM75 launch/quantization policy metadata |
| `quantization/convrot.py` | ConvRot metadata parsing and model/CLIP loading services |
| `quantization/dispatch.py` | W8A8/W4A8/W4A4 activation quantization and GEMM dispatch |
| `quantization/fusions.py` | model-independent fused activation and normalization operations |
| `adapters/minimax/` | H3 layout publication, VRAM planning, block fusions, and explicit progressive experiment |
| `adapters/wan.py` | Wan/Bernini packed-context planning and supported self-attention preprocessing |
| `adapters/wan_layout.py` | loader-independent Wan/Bernini self-attention sequence semantics |
| `adapters/bernini.py` | Bernini context-window and absolute-RoPE integration |
| `media/` | reference sets, resizing transforms, and shared video padding |
| `nodes/` | thin ComfyUI schemas and calls into the implementation packages |

`hardware.py` owns architecture predicates. `kernel_api.py` is the only module
allowed to import the independently installed kernel package. `registration.py`
is the only node mapping table.

## Compatibility

ComfyUI workflow compatibility is governed by the stable
`NODE_CLASS_MAPPINGS` keys, input names, and defaults rather than Python
filenames. Implementation and tests import canonical `comfyui_turing_utils`
paths. The sole top-level compatibility module is `attention.py`; it preserves
the old monkey-patchable attention facade while downstream integrations migrate
to `comfyui_turing_utils.attention`.

Sparse attention remains explicit and is never selected by loader `auto`.
Model-specific topology is installed through the attention-layout provider
registry, so the official ComfyUI loader and ConvRot loader follow the same path.
Model-side fused Q/K handoff is installed through a separate attention-site
registry. This lets dense and Sol backends request the same H3 or Wan/Bernini
integration without importing either model family into generic attention code.
