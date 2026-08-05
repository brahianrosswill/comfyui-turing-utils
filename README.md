# ComfyUI SVDInt4

ComfyUI custom node for loading SVDInt4-quantized Wan/Bernini DiT models and
ConvRot-quantized models, including MiniMax H3.

The plugin keeps the base model in packed INT4 form and runs the built-in
SVD residual correction tensors inside the CUDA kernel. User LoRAs are separate
adapter overlays; to make a LoRA part of the quantized base, repack the model.

## Requirements

- NVIDIA GPU with CUDA support
- Turing, Ampere, Ada, or newer GPU architecture
- Python 3.10 or newer
- PyTorch with CUDA
- CUDA toolkit with `nvcc` if installing from source
- ComfyUI
- The comfy-kitchen version pinned by ComfyUI

The default source build targets `sm_75`, `sm_80`, `sm_86`, and `sm_89`.

## Installation

Clone the plugin into ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wjie98/comfyui-svdint4.git
```

The ComfyUI plugin itself has no heavy Python dependency. This lets ComfyUI
Manager install or update the custom node without compiling CUDA code.
The CUDA package remains a separate installation: Python-only plugin updates
never invoke a compiler or JIT. Reinstall `svdint4-kernel` only when its CUDA
sources or published kernel version change.

Install the CUDA kernel in the same Python environment that runs ComfyUI:

```bash
python -m pip install -v --no-build-isolation --no-cache-dir --no-deps --upgrade --force-reinstall \
  "git+https://github.com/wjie98/comfyui-svdint4.git@main#subdirectory=kernel"
```

For local development from an already cloned checkout:

```bash
cd ComfyUI/custom_nodes/comfyui-svdint4
python -m pip install -v --no-build-isolation -e ./kernel
```

To limit the architectures built for your machine:

```bash
SVDINT4_ARCH_LIST="8.0;8.6" \
python -m pip install -v --no-build-isolation --no-cache-dir --no-deps --upgrade --force-reinstall \
  "git+https://github.com/wjie98/comfyui-svdint4.git@main#subdirectory=kernel"
```

Use `--no-build-isolation` so pip builds against the PyTorch already installed
in your ComfyUI environment. Use `--no-deps` to avoid reinstalling or
downloading PyTorch when rebuilding the kernel.

If you must use the SSH URL, initialize GitHub's SSH host key in the same
Windows account first:

```powershell
New-Item -ItemType Directory -Force $env:USERPROFILE\.ssh | Out-Null
ssh-keyscan github.com | Out-File -Append -Encoding ascii $env:USERPROFILE\.ssh\known_hosts
ssh -T git@github.com
```

Verify the printed fingerprint against GitHub's published SSH key
fingerprints before trusting the host key.

## Verify The Kernel

```bash
python - <<'PY'
import torch
import svdint4
from svdint4.ops import svd_int4_linear

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("svdint4:", svdint4.__file__)
print("kernel api:", callable(svd_int4_linear))
print("Turing SageAttention2:", svdint4.turing_sage2.available())
PY
```

The installed pip distribution is named `svdint4-kernel`; the Python package is
imported as `svdint4`.

The plugin-side split is intentionally small: `bf16_policy.py` owns generic
dtype selection, `attention_backends.py` owns backend choice,
`turing_attention.py` adapts bundled SageAttention2 to ComfyUI, and
`turing_ops.py` is the exact-sm75 comfy-kitchen bridge. None is H3-specific.

## Model Files

Place SVDInt4 DiT model files under ComfyUI's normal diffusion model folder:

```text
ComfyUI/models/diffusion_models/<model-name>.safetensors
```

Each `.safetensors` file contains one DiT branch. Wan2.2/Bernini workflows that
use separate high-noise and low-noise DiTs should use two loader nodes, one for
each file.

The file must use the SVDInt4 single-file layout:

```text
metadata:
  format = svdint4

tensors:
  blocks.N.self_attn.q.qweight
  blocks.N.self_attn.q.wscales
  blocks.N.self_attn.q.svd_down
  blocks.N.self_attn.q.svd_up
  blocks.N.self_attn.q.smooth
  blocks.N.self_attn.q.bias_packed
  ...
  non-quantized model tensors use their normal ComfyUI/Diffusers keys
```

`format` is required and has exactly one accepted value, `svdint4`.
`architecture` is optional informational metadata and is not a loader
whitelist. See [`docs/format.md`](docs/format.md) for the complete tensor
contract. Keep provenance, calibration notes, source paths, and experiment
notes in a sidecar JSON if you need them.

The data-free converter consumes a dense checkpoint whose desired LoRAs are
already fused:

```bash
python scripts/convert_diffusion_model.py \
  --input path/to/fused-high.safetensors \
  --output path/to/svdint4-high.safetensors \
  --architecture wan \
  --branch high
```

This path uses unit smooth factors and performs no activation calibration. It
is useful for conversion experiments, but it is not a substitute for the
Bernini calibration and held-out validation gate. The calibrated flow and its
1024/256 manifest contract are documented in
[`docs/calibration.md`](docs/calibration.md).

The node scans ComfyUI's `diffusion_models` paths and only shows supported
SVDInt4 files. For custom model locations, set `SVDINT4_DIT_PATHS` before
starting ComfyUI. Separate multiple paths with `:` on Linux/macOS or `;` on
Windows.

## Converting Shard Packs

Older SVDInt4 development packs may be stored as directories with one branch per
folder:

```text
packed-model/
  high/
    manifest.json
    kept_mixed.safetensors
    block_00.safetensors
    ...
  low/
    manifest.json
    kept_mixed.safetensors
    block_00.safetensors
    ...
```

Convert each branch into one `.safetensors` file before using it in ComfyUI:

1. Start with all tensors from `kept_mixed.safetensors` (`kept_fp16.safetensors`
   in older packs).
2. Read every `block_XX.safetensors` listed by `manifest.json`.
3. Copy packed tensors into the same output file.
4. Rename old SVD correction tensor keys:
   - `.lora_down` -> `.svd_down`
   - `.lora_up` -> `.svd_up`
5. Save with minimal metadata: `format=svdint4` and `architecture=wan`.

High-noise and low-noise branches should become two separate files, for example:

```text
bernini-high.safetensors
bernini-low.safetensors
```

To repack an existing single-file asset into the canonical metadata layout:

```bash
python custom_nodes/comfyui-svdint4/scripts/repack_single_file.py \
  --input old-high.safetensors \
  --output bernini-high.safetensors \
  --architecture wan
```

The script writes original metadata and basic provenance to
`bernini-high.safetensors.json` instead of embedding it in the weight file.

## Nodes

- `Load ConvRot DiT`
  Loads an official ComfyUI ConvRot diffusion model. `force_int8_gemm=false`
  follows each layer's activation format, while `true` forces INT8 GEMM
  activations. W4A8 requires an enabled comfy-kitchen CUDA backend because its
  eager fallback always computes W4A4. `patch_attention=auto` selects
  SageAttention first, Flash Attention second, and PyTorch SDPA as the fallback.
  BF16 activation storage is selected automatically when the detected model
  declares BF16 inference support; there is no loader-only dtype switch.
  Legacy per-row W8 ConvRot descriptors with a missing format or
  `format=int8_rowwise` are normalized in memory to native
  `int8_tensorwise`; the packed weights are not rewritten.

- `Load ConvRot CLIP`
  Loads one ConvRot text encoder from `models/text_encoders` with the same
  `force_int8_gemm` behavior. Its type list is read dynamically from ComfyUI's
  single CLIP loader, so newly added official types appear without a plugin
  update. It verifies that every declared ConvRot layer was loaded through
  mixed-precision ops and rejects W4A8 on a CPU load device.

- `Load SVDInt4 DiT`
  Selects one SVDInt4 DiT `.safetensors` file from `diffusion_models` and
  returns a ComfyUI `MODEL`. `patch_attention=auto` uses the same SageAttention,
  Flash Attention, then PyTorch SDPA priority on non-Turing GPUs. On a supported
  Turing GPU, `auto` selects the bundled SageAttention2 implementation.

- `Bernini Context Windows`
  Uses the same controls and defaults as ComfyUI's Wan Context Windows node,
  including `standard_uniform`, `context_overlap=30`, `retain_first_frame`,
  `split_conds_to_windows`, FreeNoise, looped schedules, and the official
  causal window fix. Internally it adds Bernini-compatible absolute temporal
  RoPE indices for each context window, including the causal fix frame.

### BF16 activation storage and Turing

Both diffusion loaders select BF16 by default when the detected ComfyUI model
configuration declares it as a supported inference dtype. Explicit ComfyUI
command-line dtype selections still win. The policy sets only the model
inference/compute boundary through the model patcher; it does not recursively
cast modules or replace normalization, RoPE, modulation, input, or output
logic. Their existing FP32 reductions and other intentional precision islands
therefore remain under ComfyUI/model control.

On exact-sm75 Tensor Core GPUs (T4 and RTX 20-series), this avoids ComfyUI's
normal BF16-to-FP32 fallback after the bundled operators pass startup
self-tests. GTX 16-series cards remain on ComfyUI's policy because they cannot
run the tensor-core path. On Ampere, Ada, and newer GPUs the plugin requests
BF16 but continues to use the normal installed backends.

The Turing linear dispatch reuses comfy-kitchen's official sm75 W4A4 and W8A8
GEMM kernels. W4A8 is supplied by the local backend: it consumes the original
packed W4 weight directly, fuses nibble unpacking, INT8 dot products, scaling,
bias, and BF16 storage, and never materializes a full W8 copy of the weight. Its
static shared-memory footprint is 2 KiB. W8A8 and W4A8 activation rotations use
Kitchen's staged BF16 INT8 quantizer whenever the fused row would reach the
48 KiB limit. For a staged SwiGLU input, the bundled quantizer folds SwiGLU into
its first ConvRot pass, uses 16 KiB of dynamic shared memory, and avoids the
activated BF16 intermediate. Its scale and INT8 rounding order match Kitchen's
fused ConvRot path. W4A4 keeps fused A4 quantization while it fits and otherwise
uses Kitchen's grouped FHT rotation followed by row-wise INT4 quantization. None
of the three paths falls back to a dense Hadamard matmul on supported sm75
devices. On a non-sm75 tensor the local backend constraint does not match, so
comfy-kitchen selects its official backend.

The bundled attention backend is derived from wjie98's full SM75
SageAttention2 path and does not import or require the standalone
`sageattention` package. It accepts FP16 or BF16 input/output, HND or NHD
layout, causal attention, GQA, different Q/KV lengths, variable-length batches,
and head dimensions up to 128 (smaller dimensions are padded internally). Q/K
are quantized to INT8 per warp; V is staged as FP16 for SM75 tensor cores;
online softmax and PV accumulation use FP32; the result is written in the
original FP16/BF16 dtype. Dynamic shared memory tops out at 32 KiB, below the
48 KiB default target. If the calling model supplies FP32 Q/K/V, only the
attention boundary is narrowed to BF16 and the result is restored to FP32;
the rest of the model remains under its existing dtype policy.

ComfyUI attention masks, disabled low-precision attention, and head dimensions
outside that contract use the original ComfyUI attention function without a
dtype conversion. The standalone Sage backend on non-Turing GPUs likewise
hands FP32 back to ComfyUI. Failure of a required SM75 kernel is reported before
model allocation instead of silently reverting the whole model to FP32.

Packed SVDInt4 weights are represented as ComfyUI QuantizedTensor weights so
ComfyUI can account for and move their qweight, scales, smooth factors, and SVD
correction tensors together. The public `state_dict()` does not expose packed
weights as normal `.weight` tensors, so standard ComfyUI LoRA patching does not
accidentally treat them as dense fp16 weights.

SVDInt4 DiT weights are loaded through ComfyUI's normal model patcher path.
ComfyUI may keep the packed branch fully loaded or use its DynamicVRAM/offload
logic depending on the current workflow and available VRAM. The loader only
defines the QuantizedTensor layout and packed Linear execution; it does not
force a separate loading policy.

The node category is:

```text
SVDInt4/loaders
```

## LoRA

The packed SVD residual correction tensors inside the model are part of the
base quantized model and are not LoRA adapters. Standard LoRA patches targeting
packed SVDInt4 Linear weights are kept out of ComfyUI's dense weight patch
table. Compatible adapter LoRAs run automatically as fp16 overlays on top of
the packed model. Adapter overlay tensors stay in CPU-owned storage and are
staged into a small per-model GPU buffer layer by layer. The overlay reuses
ComfyUI's `WeightAdapter` h/g tensors; it is still a separate fp16 matmul path,
not fused SVDInt4 weight. Dense `diff`/`set` weight patches are intentionally
not supported for packed SVDInt4 weights. Repack the model when a LoRA is meant
to become part of the quantized base.

## Smoke Tests

Local load and single-layer CUDA forward:

```bash
python custom_nodes/comfyui-svdint4/scripts/smoke_test.py \
  --model ComfyUI/models/diffusion_models/your-model.safetensors
```

Real denoise smoke on a running ComfyUI server with an API-format workflow:

```bash
python custom_nodes/comfyui-svdint4/scripts/smoke_test.py \
  --workflow smoke-workflow-api.json \
  --server http://127.0.0.1:8188 \
  --steps 3
```

On Windows/Turing, run both tests after installing the kernel in the same
environment that starts ComfyUI. The first test catches loader/kernel import
and single-kernel issues; the workflow test catches DynamicVRAM, high/low DiT,
VAE/text encoder, and scheduler integration issues.

## Troubleshooting

`ModuleNotFoundError: No module named 'svdint4'`

Install the kernel in the same environment that launches ComfyUI:

```bash
python -m pip install -v --no-build-isolation --no-cache-dir --no-deps --upgrade --force-reinstall \
  "git+https://github.com/wjie98/comfyui-svdint4.git@main#subdirectory=kernel"
```

`CUDA version mismatches the version that was used to compile PyTorch`

Make sure the CUDA toolkit used by `nvcc` matches your PyTorch CUDA version.
For source builds, also make sure `--no-build-isolation` is present.

Windows runtime

The SVDInt4 CUDA kernel requires Turing/sm75 or newer. Models that declare BF16
use BF16 activation storage by default; the packed SVDInt4 linear kernel still
uses its native FP16 compute path internally and restores the surrounding
activation dtype on output.

`fatal error C1083: cannot open include file: 'nv/target'`

CUDA 12.x obtains this header from CCCL. In an NVIDIA Conda CUDA 12.8
environment, install the matching package and retry the independent kernel
installation:

```bat
conda install -n comfyui -c nvidia cuda-cccl=12.8.90
set SVDINT4_ARCH_LIST=7.5
python -m pip install -v --no-build-isolation -e .\kernel
```

The build automatically discovers Conda's
`Library\include\targets\x64` layout. For a custom CCCL installation, set
`SVDINT4_CCCL_INCLUDE_DIR` to the directory that directly contains
`nv\target`.

`fatal error C1083: ... cusparse.h: No such file or directory`

Update to the latest `comfyui-svdint4` commit. Older builds included heavy
PyTorch extension headers from the binding file, which could pull in PyTorch
CUDA sparse headers on Windows. The current binding uses lighter ATen/pybind
headers and does not require `cusparse.h` directly.

`Error checking compiler version for cl`

This warning can appear when MSVC prints localized diagnostics and PyTorch
cannot decode them with the active Windows code page. The build script sets
`VSLANG=1033` automatically on Windows. If you still see this warning, set it
before running pip:

```powershell
$env:VSLANG = "1033"
python -m pip install -v --no-build-isolation --no-cache-dir --no-deps --upgrade --force-reinstall `
  "git+https://github.com/wjie98/comfyui-svdint4.git@main#subdirectory=kernel"
```

The model dropdown is empty

No valid SVDInt4 DiT files were found. Put the single-file assets in
`ComfyUI/models/diffusion_models` and make sure their metadata contains
`format=svdint4`.

ComfyUI starts, but generation fails when sampling

The plugin can be loaded without the CUDA extension, but inference requires the
kernel to be installed and importable from the ComfyUI environment.

## License

This project is distributed under Apache-2.0. See `kernel/LICENSE`,
`kernel/NOTICE`, and `kernel/LICENSES/` for kernel license details.
