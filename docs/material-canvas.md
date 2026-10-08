# Material Canvas (experimental)

This is a material workspace inside the existing ComfyUI editor, not an ordinary
workflow. It does not require changes to ComfyUI core or a separate inference
server. Python lives in `comfyui_turing_utils/canvas/`; UI lives in `web/canvas/`.

## Cards

All six public cards are in **Turing Utils / Canvas**:

- Canvas Settings: one per canvas; project/cache directories, upload mode,
  default DiT/CLIP/VAEs, LoRA JSON, attention/Sol, steps or explicit sigmas,
  shifts, Chat configuration and SeC model.
- Canvas Image, Canvas Video, Canvas Audio: import and preview material.
  Video/audio allow an interval in seconds; duration zero means through the end.
  Video's second output references its soundtrack, not a separately copied file.
- Canvas H3 Generate: reference generation, masked editing, continuation and
  relative-frame outpainting; no upscale/second pass. Extra reference images
  use dynamic inputs. Generation uses the nonblank model prompt, otherwise the
  user prompt. Enhance User Prompt is a separate explicit network operation,
  never an automatic dependency of generation. API secrets are referenced by
  environment variable name, not stored in workflow JSON.
- Canvas Video Mask: existing SeC point-guided tracking, publishing a mask
  bound to a particular target asset/interval. This is not a new text-grounding
  implementation. Select the model in Settings and supply positive points.

Ports use independent `TURING_CANVAS_*_ASSET` labels. Cards cannot consume normal
IMAGE/AUDIO/LATENT ports, and cannot execute through ordinary `/prompt` requests.
Internal nodes compile a selected task into the existing ComfyUI execution queue;
all output publication is done after successful encoding/saving.

## Getting started

1. Open an empty workflow and add Canvas Settings. It receives a unique default
   project directory. Select the actual H3 DiT, CLIP and both VAEs. Use a model
   supported by the ConvRot loaders; these controls do not convert arbitrary
   QuantFunc/SVD checkpoints into supported models.
2. Add material cards and drag files onto them, or click Load / Refresh.
3. Add H3, connect material roles, enter a user/model prompt and select a mode.
   Reference, target and soundtrack are distinct roles, not interchangeable.
4. Click Enhance User Prompt only if wanted. It replaces model prompt text after
   success, offers an overwrite confirmation for concurrent edits and an Undo
   prompt enhancement menu entry. Visual enhancement inputs currently use a
   bounded preview (up to the first 240 frames at 24 FPS), not full long-video
   analysis. Audio is not sent to the Chat image/video interface.
5. Click Generate New Result. History is retained and the new result is published.
   The result can directly feed another H3 card. Select Published Version changes
   the active output without deleting the other versions.

Use ComfyUI's normal workflow save/open or Settings' Save Canvas Project / Open
Saved Project. Use a separate work directory for a separate project. A copied
workflow pointing at the same directory intentionally opens the same result
registry; change its directory to fork a project.

## Material update semantics

Model/LoRA/steps/seed/prompt edits do **not** invalidate existing material or mark
downstream tasks for automatic refresh. Only source asset version, connections
and selected time intervals participate in update detection. ComfyUI tensor
caches are opportunistic; published results survive process restarts.

Generate New Result explicitly reruns inference even with unchanged inputs.
Material reads/encoding remain independent of the run nonce. Failure keeps the
last successful output. The Pin / Follow input menu can bind a specific source
asset version. Time selection still belongs to the source material card.

Global queue actions are locked by default in canvas mode. Prepare / Lock Global
Refresh previews the plan. Refresh Changed Materials Once asks for confirmation
and processes the frozen plan sequentially. Yellow means changed material;
dashed yellow means affected downstream; blue means running; red means blocked
or failed. Lock is restored before the run starts, including on errors. Changed
parameters alone do not enter this plan. A failed task stops the remaining chain.

## Storage and import modes

- `work_directory`: relative to this instance's ComfyUI output directory. Assets,
  immutable result files, result history and saved project JSON are stored here.
- `cache_directory`: relative to this instance's `.cache`. Currently used for
  rebuildable mask thumbnails; not a model-weight or tensor cache.
- `browser_upload`: multipart file data is streamed to disk. Reverse proxies may
  still impose their own upload limits. This is not a resumable upload protocol.
- `local_copy`: copy an explicitly selected file from this server instance's
  **input directory**, entered as a relative `local_path`. Nothing is uploaded
  from the browser. Native browser file drops do not expose the server's real
  filesystem path, so the UI explains this and requests a path instead. A file
  on the browser's machine is not necessarily present on the server.

No production/development directory sharing, arbitrary server-directory browser,
automatic file deletion, or file relocation is added. Imported files are copied;
changing a source file does nothing until Load / Refresh is explicitly used.
Every refresh imports a new version, even if the source bytes happen to match.

## Editor isolation and validation

With Settings present, ordinary node creation and right-click entries are
disabled. Search filtering uses the frontend's node-definition filter API through
an optional Pinia bridge (verified on frontend 1.53.6). If that access changes,
creation guards and backend rejection remain active. Removing Settings restores
ordinary menus. Existing mixed canvases are preserved on opening, but execution
fails instead of silently deleting or executing their ordinary nodes. Subgraphs,
mute and bypass are not part of this first material-canvas contract.

Tests cover material-only update detection, publication/version selection,
path containment, CPU material decode/save and compilation against actual core
node schemas with model loaders substituted. UI checks cover creation/search
filtering, prompt controls, upload, material delivery and save/reopen. Full H3
and SeC inference quality/performance still needs testing with complete model
weights; the development instance lacks the H3 CLIP/video/audio VAE set.

Verification on 2026-10-08: 703 selected Python tests and 1178 subtests passed,
including 17 canvas tests; two existing JS tests passed. CUDA/quantization
selections and the previously failing strict VAE INT8 FP16 equality file were
excluded. Browser checks additionally passed actual video-file drop/upload,
locked global queue behavior and mixed-canvas rejection. Dev runtime diagnostics
passed with kernel 0.43.0 / PyTorch 2.9.1+cu130 on A40. This is not a full H3
generation or target-GPU numerical benchmark.
