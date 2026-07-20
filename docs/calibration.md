# Bernini SVDInt4 Calibration

Bernini release weights require multi-resolution, multi-frame,
context-window-aware calibration. A data-free conversion is an experimental
candidate and must not be presented as a calibrated Bernini release.

## Required coverage

- 1024 calibration windows and 256 disjoint held-out validation windows
- resolution buckets 480, 540, 720, and 1080
- frame buckets 5, 9, 17, 33, 49, 65, 81, 97, and 121
- source video, up to one reference video, and zero to five reference images
- `single_full`, `first`, `middle`, `last`, `tail_padded`, and `short_video`
  context-window roles
- landscape, portrait, and square or near-square inputs
- low, medium, high-subject, camera, and occlusion-heavy motion

Logical 97/121-frame cases normally contribute first, middle, last, or padded
windows. The window cap is 81 frames for 480/540/720 and 33 frames for 1080 on
44 GB calibration GPUs.

### Resolution quotas

| Bucket | Calibration | Held out |
|---:|---:|---:|
| 480 | 256 | 64 |
| 540 | 192 | 48 |
| 720 | 320 | 80 |
| 1080 | 256 | 64 |

### Frame quotas

| Frames | Calibration |
|---:|---:|
| 5 | 64 |
| 9 | 64 |
| 17 | 96 |
| 33 | 128 |
| 49 | 128 |
| 65 | 128 |
| 81 | 160 |
| 97 | 128 |
| 121 | 128 |

### Conditioning quotas

| Conditioning | Calibration |
|---|---:|
| source + 0 images | 128 |
| source + 1 image | 192 |
| source + 2 images | 160 |
| source + 3 images | 128 |
| source + 5 images | 96 |
| source + reference video + 0 images | 96 |
| source + reference video + 1 image | 96 |
| source + reference video + 3 images | 80 |
| source + reference video + 5 images | 48 |

### Context-window quotas

| Role | Calibration |
|---|---:|
| `single_full` | 128 |
| `first` | 192 |
| `middle` | 256 |
| `last` | 192 |
| `tail_padded` | 192 |
| `short_video` | 64 |

## Tool flow

1. Build exact 1024/256 manifests with
   `scripts/calibration/build_bernini_manifest.py`.
2. Fuse the intended LoRA recipe into dense high/low sources.
3. Collect high/low Linear input statistics through the real Bernini pipeline
   with `scripts/calibration/collect_bernini_stats.py`.
4. Merge distributed shards with `scripts/calibration/merge_stats.py` when
   needed.
5. Derive per-branch smooth factors with
   `scripts/calibration/derive_smooth.py`, using the cross-bucket upper-envelope
   p99.9 statistic by default.
6. Run `scripts/convert_diffusion_model.py --smooth ...` independently for high
   and low branches.
7. Validate all 256 held-out windows and report metrics per resolution, frame,
   aspect, window role, reference-image count, reference-video presence, and
   motion bucket. Visually inspect short clips, padded tails, 1080/121 logical
   clips, five-image conditioning, portrait inputs, and high-motion cases.

Every LoRA recipe must collect its own statistics after fusion. Do not reuse
smooth factors across recipes. Do not publish a Bernini SVDInt4 pair when any
required bucket, held-out report, or targeted visual check is missing.

## Example commands

Build the mandatory calibration, held-out, and high-risk manifests:

```bash
python scripts/calibration/build_bernini_manifest.py \
  --base-manifest path/to/source_cases.jsonl \
  --calib-data path/to/calibration_media \
  --output-root work/manifests
```

Collect statistics after the exact LoRA recipe has been fused into both dense
branches. The collector rejects a calibration manifest unless all 1024 rows
and every required marginal quota are present:

```bash
torchrun --nproc-per-node=4 scripts/calibration/collect_bernini_stats.py \
  --bernini-repo path/to/Bernini \
  --shim-root path/to/veomni_shim \
  --model-dir path/to/Bernini-Diffusers \
  --high-source work/fused-high.safetensors \
  --low-source work/fused-low.safetensors \
  --manifest work/manifests/bernini_multires_calib1024.jsonl \
  --output-prefix work/stats/bernini \
  --ulysses 4
```

Merge the rank outputs, then derive branch-specific smooth factors:

```bash
python scripts/calibration/merge_stats.py \
  --output-prefix work/stats/bernini-merged \
  work/stats/bernini_rank00 work/stats/bernini_rank01 \
  work/stats/bernini_rank02 work/stats/bernini_rank03

python scripts/calibration/derive_smooth.py \
  --source work/fused-high.safetensors \
  --stats work/stats/bernini-merged_high.pt \
  --output work/smooth-high.pt

python scripts/calibration/derive_smooth.py \
  --source work/fused-low.safetensors \
  --stats work/stats/bernini-merged_low.pt \
  --output work/smooth-low.pt
```

Quantize the branches into the same canonical format used by data-free
conversion:

```bash
python scripts/convert_diffusion_model.py \
  --input work/fused-high.safetensors \
  --output work/bernini-high.safetensors \
  --architecture wan --branch high --smooth work/smooth-high.pt

python scripts/convert_diffusion_model.py \
  --input work/fused-low.safetensors \
  --output work/bernini-low.safetensors \
  --architecture wan --branch low --smooth work/smooth-low.pt
```

These commands produce calibration candidates, not a release decision. The
256 held-out rows still need real output metrics grouped by every required
bucket and the targeted visual checks listed above.
