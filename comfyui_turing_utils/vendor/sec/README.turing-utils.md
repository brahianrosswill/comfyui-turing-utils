Vendored SeC inference runtime
==============================

This directory contains the inference implementation and model configuration
files derived from `9nate-drake/Comfyui-SecNodes` at commit
`af39e6260bc414873fbb2dc8df355904d1609754` (Apache-2.0).

Turing Utils changes the integration and frame-input paths. ComfyUI IMAGE
tensors are consumed directly instead of being encoded to temporary JPEG files;
package-relative imports avoid a global `inference` package; Hydra's process
state is left untouched; forward and reverse frame caches are bounded; and the
`mllm_memory_size=1` edge case is handled explicitly. A ComfyUI runtime-device
hint is honored when DynamicVRAM keeps stored parameters on CPU. The vendored
code also avoids changing the process-wide Python RNG and warnings filters.

The original license is retained in `LICENSE`.  Individual source files also
retain their upstream copyright and attribution notices.
