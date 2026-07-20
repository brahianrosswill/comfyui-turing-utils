from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from safetensors import safe_open


DEFAULT_GUIDANCE_MODE = "vae_txt_vit_wapg"
LINEAR_MAP = {
    "attn1.to_q": "self_attn.q",
    "attn1.to_k": "self_attn.k",
    "attn1.to_v": "self_attn.v",
    "attn1.to_out.0": "self_attn.o",
    "attn2.to_q": "cross_attn.q",
    "attn2.to_k": "cross_attn.k",
    "attn2.to_v": "cross_attn.v",
    "attn2.to_out.0": "cross_attn.o",
    "ffn.net.0.proj": "ffn.0",
    "ffn.net.2": "ffn.2",
}
BLOCK_LINEAR_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")
SOURCE_PREFIXES = ("", "model.diffusion_model.", "diffusion_model.")


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _with_source_prefixes(name: str) -> list[str]:
    return [prefix + name for prefix in SOURCE_PREFIXES]


def _replace_prefix(name: str, prefix: str, target: str) -> str | None:
    if not name.startswith(prefix):
        return None
    return target + name[len(prefix) :]


def source_key_candidates(module_key: str) -> list[str]:
    candidates = [module_key]
    replacements = (
        ("condition_embedder.time_embedder.linear_1.", "time_embedding.0."),
        ("condition_embedder.time_embedder.linear_2.", "time_embedding.2."),
        ("condition_embedder.time_proj.", "time_projection.1."),
        ("condition_embedder.text_embedder.linear_1.", "text_embedding.0."),
        ("condition_embedder.text_embedder.linear_2.", "text_embedding.2."),
        ("proj_out.", "head.head."),
    )
    for prefix, target in replacements:
        mapped = _replace_prefix(module_key, prefix, target)
        if mapped is not None:
            candidates.append(mapped)

    if module_key == "scale_shift_table":
        candidates.append("head.modulation")

    block_match = re.match(r"^blocks\.(\d+)\.(.+)$", module_key)
    if block_match is not None:
        block, suffix = block_match.groups()
        if suffix == "scale_shift_table":
            candidates.append(f"blocks.{block}.modulation")
        elif suffix.startswith("norm2."):
            candidates.append(f"blocks.{block}.norm3.{suffix[len('norm2.'):]}")
        else:
            for module_suffix, storage_suffix in LINEAR_MAP.items():
                if suffix.startswith(module_suffix + "."):
                    candidates.append(f"blocks.{block}.{storage_suffix}.{suffix[len(module_suffix) + 1:]}")
                    break
            for attn_prefix, storage_prefix in (
                ("attn1.norm_q.", "self_attn.norm_q."),
                ("attn1.norm_k.", "self_attn.norm_k."),
                ("attn2.norm_q.", "cross_attn.norm_q."),
                ("attn2.norm_k.", "cross_attn.norm_k."),
            ):
                if suffix.startswith(attn_prefix):
                    candidates.append(f"blocks.{block}.{storage_prefix}{suffix[len(attn_prefix):]}")
                    break

    expanded = []
    for candidate in _dedupe(candidates):
        expanded.extend(_with_source_prefixes(candidate))
    return _dedupe(expanded)


@torch.no_grad()
def load_storage_style_transformer_weights(module: nn.Module, source_path: Path, branch: str) -> dict[str, int]:
    if module is None:
        raise ValueError(f"{branch}: transformer module is missing")
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    named_tensors = dict(module.named_parameters())
    named_tensors.update(module.named_buffers())
    loaded = 0
    missing = []
    linear_weight_loaded = 0
    linear_bias_loaded = 0

    with safe_open(source_path, framework="pt", device="cpu") as f:
        source_keys = set(f.keys())
        for module_key, target in named_tensors.items():
            source_key = next((key for key in source_key_candidates(module_key) if key in source_keys), None)
            if source_key is None:
                missing.append(module_key)
                continue
            tensor = f.get_tensor(source_key)
            if tuple(tensor.shape) != tuple(target.shape):
                raise ValueError(
                    f"{branch}: {source_key} shape {tuple(tensor.shape)} does not match "
                    f"{module_key} {tuple(target.shape)}"
                )
            target.copy_(tensor.to(device=target.device, dtype=target.dtype, non_blocking=True))
            loaded += 1
            linear_base = None
            if module_key.endswith(".weight"):
                linear_base = module_key[: -len(".weight")]
            elif module_key.endswith(".bias"):
                linear_base = module_key[: -len(".bias")]
            if linear_base is not None and HookCollector.storage_name(linear_base) is not None:
                if module_key.endswith(".weight"):
                    linear_weight_loaded += 1
                elif module_key.endswith(".bias"):
                    linear_bias_loaded += 1

    if linear_weight_loaded < 400:
        raise RuntimeError(
            f"{branch}: loaded only {linear_weight_loaded} quantized Linear weights from {source_path}; expected 400"
        )
    print(
        f"[weights] {branch}: loaded {loaded}/{len(named_tensors)} tensors from {source_path}; "
        f"Linear weights={linear_weight_loaded}, Linear bias={linear_bias_loaded}, missing={len(missing)}",
        flush=True,
    )
    if missing:
        print("[weights] missing target tensors: " + ", ".join(missing[:20]), flush=True)
    return {
        "loaded": loaded,
        "target_tensors": len(named_tensors),
        "linear_weight_loaded": linear_weight_loaded,
        "linear_bias_loaded": linear_bias_loaded,
        "missing": len(missing),
    }


class RunningStats:
    def __init__(self):
        self.amax: dict[str, torch.Tensor] = {}
        self.sum_abs: dict[str, torch.Tensor] = {}
        self.quantiles: dict[str, dict[str, torch.Tensor]] = {"p99": {}, "p999": {}, "p9999": {}}
        self.count: dict[str, int] = {}
        self.calls: dict[str, int] = {}
        self.shapes: dict[str, set[tuple[int, ...]]] = {}
        self.sample_rows = 64

    @torch.no_grad()
    def add(self, name: str, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or x.numel() == 0:
            return
        v = x.detach()
        if v.dim() == 1:
            v = v.reshape(1, -1)
        else:
            v = v.reshape(-1, v.shape[-1])
        if v.shape[-1] <= 0:
            return
        if 0 < self.sample_rows < v.shape[0]:
            idx = torch.linspace(0, v.shape[0] - 1, self.sample_rows, device=v.device).long()
            sample = v.index_select(0, idx).float().abs()
        else:
            sample = v.float().abs()
        if sample.numel() == 0:
            return
        cur_amax = sample.amax(dim=0)
        cur_sum = sample.sum(dim=0)
        sample_count = int(sample.shape[0])

        def sampled_quantile(q: float) -> torch.Tensor:
            k = min(sample_count, max(1, math.ceil(q * sample_count)))
            return sample.kthvalue(k, dim=0).values

        cur_quantiles = {
            "p99": sampled_quantile(0.99),
            "p999": sampled_quantile(0.999),
            "p9999": sampled_quantile(0.9999),
        }
        if name not in self.amax:
            self.amax[name] = cur_amax
            self.sum_abs[name] = cur_sum
            for label, value in cur_quantiles.items():
                self.quantiles[label][name] = value
            self.count[name] = sample_count
            self.calls[name] = 1
            self.shapes[name] = {tuple(x.shape)}
            return
        if self.amax[name].numel() != cur_amax.numel():
            raise ValueError(f"{name}: channel mismatch {self.amax[name].numel()} vs {cur_amax.numel()}")
        self.amax[name] = torch.maximum(self.amax[name], cur_amax)
        self.sum_abs[name] += cur_sum
        for label, value in cur_quantiles.items():
            self.quantiles[label][name] = torch.maximum(self.quantiles[label][name], value)
        self.count[name] += sample_count
        self.calls[name] += 1
        if len(self.shapes[name]) < 16:
            self.shapes[name].add(tuple(x.shape))

    def as_tensors(self) -> dict[str, torch.Tensor]:
        out = {}
        for name in sorted(self.amax):
            out[f"{name}.amax"] = self.amax[name].detach().cpu().to(torch.float16)
            for label in ("p99", "p999", "p9999"):
                out[f"{name}.{label}"] = self.quantiles[label][name].detach().cpu().to(torch.float16)
            out[f"{name}.mean_abs"] = (
                self.sum_abs[name] / max(1, self.count[name])
            ).detach().cpu().to(torch.float16)
        return out

    @torch.inference_mode()
    def merge(self, other: "RunningStats") -> None:
        for name, other_amax in other.amax.items():
            if name not in self.amax:
                self.amax[name] = other_amax
                self.sum_abs[name] = other.sum_abs[name]
                for label in self.quantiles:
                    self.quantiles[label][name] = other.quantiles[label][name]
                self.count[name] = other.count[name]
                self.calls[name] = other.calls[name]
                self.shapes[name] = set(other.shapes[name])
                continue
            if self.amax[name].numel() != other_amax.numel():
                raise ValueError(f"{name}: channel mismatch while merging attempt stats")
            target_device = self.amax[name].device
            other_amax = other_amax.to(target_device)
            self.amax[name] = torch.maximum(self.amax[name], other_amax)
            self.sum_abs[name] += other.sum_abs[name].to(target_device)
            for label in self.quantiles:
                self.quantiles[label][name] = torch.maximum(
                    self.quantiles[label][name], other.quantiles[label][name].to(target_device)
                )
            self.count[name] += other.count[name]
            self.calls[name] += other.calls[name]
            if len(self.shapes[name]) < 16:
                remaining = 16 - len(self.shapes[name])
                self.shapes[name].update(sorted(other.shapes[name])[:remaining])

    @torch.inference_mode()
    def to(self, device: torch.device | str) -> "RunningStats":
        self.amax = {name: value.to(device) for name, value in self.amax.items()}
        self.sum_abs = {name: value.to(device) for name, value in self.sum_abs.items()}
        self.quantiles = {
            label: {name: value.to(device) for name, value in values.items()}
            for label, values in self.quantiles.items()
        }
        return self

    @classmethod
    def from_saved(
        cls,
        tensors: dict[str, torch.Tensor],
        summary_rows: list[dict],
        device: torch.device,
    ) -> "RunningStats":
        out = cls()
        for row in summary_rows:
            name = row["name"]
            tokens = int(row.get("tokens", 0))
            calls = int(row.get("calls", 0))
            if tokens <= 0 or calls <= 0:
                raise ValueError(f"{name}: invalid saved tokens/calls")
            required = [f"{name}.{label}" for label in ("amax", "p99", "p999", "p9999", "mean_abs")]
            missing = [key for key in required if key not in tensors]
            if missing:
                raise KeyError(f"{name}: missing saved stats {missing}")
            out.amax[name] = tensors[f"{name}.amax"].to(device=device, dtype=torch.float32)
            for label in out.quantiles:
                out.quantiles[label][name] = tensors[f"{name}.{label}"].to(device=device, dtype=torch.float32)
            mean_abs = tensors[f"{name}.mean_abs"].to(device=device, dtype=torch.float32)
            out.sum_abs[name] = mean_abs * tokens
            out.count[name] = tokens
            out.calls[name] = calls
            out.shapes[name] = {tuple(int(value) for value in shape) for shape in row.get("shapes", [])}
        return out

    def summary(self) -> list[dict]:
        rows = []
        for name in sorted(self.amax):
            mean_abs = self.sum_abs[name] / max(1, self.count[name])
            rows.append(
                {
                    "name": name,
                    "channels": int(self.amax[name].numel()),
                    "tokens": int(self.count[name]),
                    "calls": int(self.calls[name]),
                    "amax_mean": float(self.amax[name].mean().item()),
                    "amax_max": float(self.amax[name].max().item()),
                    "p99_mean": float(self.quantiles["p99"][name].mean().item()),
                    "p99_max": float(self.quantiles["p99"][name].max().item()),
                    "p999_mean": float(self.quantiles["p999"][name].mean().item()),
                    "p999_max": float(self.quantiles["p999"][name].max().item()),
                    "p9999_mean": float(self.quantiles["p9999"][name].mean().item()),
                    "p9999_max": float(self.quantiles["p9999"][name].max().item()),
                    "mean_abs_mean": float(mean_abs.mean().item()),
                    "mean_abs_max": float(mean_abs.max().item()),
                    "shapes": [list(s) for s in sorted(self.shapes[name])],
                }
            )
        return rows


class HookCollector:
    def __init__(self):
        self.stats = {"high": RunningStats(), "low": RunningStats()}
        self.handles = []
        self.registered = {"high": [], "low": []}

    @staticmethod
    def storage_name(module_name: str) -> str | None:
        match = BLOCK_LINEAR_RE.match(module_name)
        if match is None:
            return None
        block, suffix = match.groups()
        mapped = LINEAR_MAP.get(suffix)
        if mapped is None:
            return None
        return f"blocks.{block}.{mapped}"

    def register_branch(self, branch: str, module: nn.Module | None) -> None:
        if module is None:
            return
        for module_name, child in module.named_modules():
            storage = self.storage_name(module_name)
            if storage is None or not isinstance(child, nn.Linear):
                continue
            self.registered[branch].append({"module": module_name, "storage": storage})

            def pre_hook(_mod, inputs, branch=branch, storage=storage):
                if inputs:
                    self.stats[branch].add(storage, inputs[0])

            self.handles.append(child.register_forward_pre_hook(pre_hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def save(self, prefix: Path, meta: dict) -> None:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        payload = {
        "artifact_type": "bernini_svdint4_linear_input_stats",
            "meta": meta,
            "high": self.stats["high"].as_tensors(),
            "low": self.stats["low"].as_tensors(),
        }
        torch.save(payload["high"], prefix.with_name(prefix.name + "_high.pt"))
        torch.save(payload["low"], prefix.with_name(prefix.name + "_low.pt"))
        torch.save(payload, prefix.with_suffix(".pt"))
        summary = {
            "artifact_type": payload["artifact_type"],
            "meta": meta,
            "registered": self.registered,
            "high": self.stats["high"].summary(),
            "low": self.stats["low"].summary(),
        }
        prefix.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def setup_paths(bernini_repo: Path, shim_root: Path) -> None:
    sys.path.insert(0, str(shim_root))
    sys.path.insert(0, str(bernini_repo))



def jsonable_args(args) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def setup_ulysses_parallel(args) -> tuple[torch.device, object | None, dict[str, object]]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ulysses = int(getattr(args, "ulysses", 1))
    if world_size == 1 and ulysses <= 1:
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return device, None, {
            "enabled": False,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "ulysses_size": 1,
            "dp_size": 1,
            "dp_rank": 0,
            "ulysses_rank": 0,
        }

    if world_size == 1:
        raise RuntimeError("--ulysses > 1 requires launching with torchrun")
    if ulysses <= 1:
        raise RuntimeError("distributed calibration requires --ulysses > 1")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="cuda:nccl,cpu:gloo",
        timeout=timedelta(seconds=3600),
        rank=rank,
        world_size=world_size,
    )
    from bernini.parallel import init_parallel_state

    ps = init_parallel_state(ulysses_size=ulysses)
    return device, ps, {
        "enabled": True,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "ulysses_size": ps.ulysses_size,
        "dp_size": ps.dp_size,
        "dp_rank": ps.dp_rank,
        "ulysses_rank": ps.ulysses_rank,
    }


def rank_output_prefix(prefix: Path, parallel_meta: dict[str, object]) -> Path:
    if not parallel_meta.get("enabled"):
        return prefix
    rank = int(parallel_meta["rank"])
    return prefix.with_name(f"{prefix.name}_rank{rank:02d}")


def configure_vae_memory(vae, args) -> dict[str, object]:
    config: dict[str, object] = {}
    if not getattr(args, "disable_vae_tiling", False) and hasattr(vae, "enable_tiling"):
        vae.enable_tiling(
            tile_sample_min_height=getattr(args, "vae_tile_size", 1024),
            tile_sample_min_width=getattr(args, "vae_tile_size", 1024),
            tile_sample_stride_height=getattr(args, "vae_tile_stride", 768),
            tile_sample_stride_width=getattr(args, "vae_tile_stride", 768),
        )
        config["tiling"] = True
        config["tile_size"] = getattr(args, "vae_tile_size", 1024)
        config["tile_stride"] = getattr(args, "vae_tile_stride", 768)
    else:
        config["tiling"] = False
    if not getattr(args, "disable_vae_slicing", False) and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
        config["slicing"] = True
    else:
        config["slicing"] = False
    return config


def _module_device(module: nn.Module | None) -> str | None:
    if module is None:
        return None
    try:
        return str(next(module.parameters()).device)
    except StopIteration:
        return None


def _module_device_obj(module: nn.Module | None) -> torch.device | None:
    if module is None:
        return None
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def _cuda_empty_cache(device: torch.device | None = None) -> None:
    if device is not None and device.type != "cuda":
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _forward_hook_always(module: nn.Module, hook):
    try:
        return module.register_forward_hook(hook, always_call=True)
    except TypeError:
        return module.register_forward_hook(hook)


def _transformer_blocks(module: nn.Module | None) -> list[nn.Module]:
    if module is None or not hasattr(module, "blocks"):
        return []
    return list(getattr(module, "blocks"))


def _move_transformer_blocks(module: nn.Module | None, device: torch.device | str) -> int:
    blocks = _transformer_blocks(module)
    for block in blocks:
        block.to(device)
    return len(blocks)


def _move_direct_tensors(module: nn.Module, device: torch.device | str) -> None:
    for name, param in list(module._parameters.items()):
        if param is None:
            continue
        module._parameters[name] = nn.Parameter(param.to(device), requires_grad=param.requires_grad)
    for name, buf in list(module._buffers.items()):
        if buf is not None:
            module._buffers[name] = buf.to(device)


def _move_transformer_non_block_modules(module: nn.Module, device: torch.device) -> int:
    moved = 0
    _move_direct_tensors(module, device)
    for name, child in module.named_children():
        if name == "blocks":
            continue
        child.to(device)
        moved += 1
    return moved


def _install_offloaded_block_hooks(
    module: nn.Module,
    blocks: list[nn.Module],
    indices: set[int],
    device: torch.device,
) -> int:
    hooked = getattr(module, "_svdquant_block_offload_hooked_indices", set())
    handles = getattr(module, "_svdquant_block_offload_handles", [])
    installed = 0
    for index in sorted(indices - hooked):
        block = blocks[index]

        def pre_hook(block_module, _inputs, target_device=device):
            block_module.to(target_device)

        def post_hook(block_module, _inputs, output):
            block_module.to("cpu")
            return output

        handles.append(block.register_forward_pre_hook(pre_hook))
        handles.append(_forward_hook_always(block, post_hook))
        hooked.add(index)
        installed += 1
    setattr(module, "_svdquant_block_offload_hooked_indices", hooked)
    setattr(module, "_svdquant_block_offload_handles", handles)
    return installed


def _move_resident_transformer_blocks(module: nn.Module | None, device: torch.device | str) -> int:
    if module is None:
        return 0
    offloaded = getattr(module, "_svdquant_block_offload_indices", set())
    moved = 0
    for index, block in enumerate(_transformer_blocks(module)):
        if index not in offloaded:
            block.to(device)
            moved += 1
    return moved


def install_transformer_block_offload(
    module: nn.Module | None,
    device: torch.device,
    block_count: int,
    peer: nn.Module | None = None,
) -> dict[str, object]:
    if module is None:
        return {"enabled": False, "blocks": 0}
    non_block_modules = _move_transformer_non_block_modules(module, device)
    blocks = _transformer_blocks(module)
    requested = len(blocks) if block_count <= 0 else min(len(blocks), block_count)
    target_indices = set(range(len(blocks) - requested, len(blocks)))
    current_indices = getattr(module, "_svdquant_block_offload_indices", set())
    offloaded_indices = current_indices | target_indices
    setattr(module, "_svdquant_block_offload_indices", offloaded_indices)
    installed = _install_offloaded_block_hooks(module, blocks, offloaded_indices, device)

    if not getattr(module, "_svdquant_block_offload_enabled", False):
        def branch_pre_hook(_module, _inputs, target_device=device, peer_module=peer):
            _move_resident_transformer_blocks(peer_module, "cpu")
            _move_resident_transformer_blocks(_module, target_device)

        handles = getattr(module, "_svdquant_block_offload_handles", [])
        handles.append(module.register_forward_pre_hook(branch_pre_hook))
        setattr(module, "_svdquant_block_offload_handles", handles)
        setattr(module, "_svdquant_block_offload_enabled", True)

    # Start with both branches on CPU. The branch pre-hook moves only the
    # resident subset, while the selected blocks page in for their own call.
    _move_transformer_blocks(module, "cpu")
    _cuda_empty_cache(device)
    return {
        "enabled": True,
        "blocks": len(blocks),
        "offloaded_blocks": len(offloaded_indices),
        "resident_blocks": len(blocks) - len(offloaded_indices),
        "new_block_hooks": installed,
        "non_block_modules": non_block_modules,
    }


def cleanup_transformer_memory(diff_dec, args, device: torch.device) -> None:
    high = getattr(diff_dec, "transformer", None)
    low = getattr(diff_dec, "transformer_2", None)
    if getattr(args, "block_offload_transformers", False):
        _move_transformer_blocks(high, "cpu")
        _move_transformer_blocks(low, "cpu")
    elif not getattr(args, "keep_transformers_on_gpu", False):
        if high is not None:
            high.to("cpu")
        if low is not None:
            low.to("cpu")
    _cuda_empty_cache(device)


def enable_transformer_block_offload(
    diff_dec,
    args,
    device: torch.device,
    block_count: int | None = None,
) -> dict[str, object]:
    args.keep_transformers_on_gpu = False
    args.block_offload_transformers = True
    if block_count is not None:
        args.active_block_offload_count = block_count
    return configure_transformer_memory(diff_dec, args, device)


def disable_transformer_block_offload(diff_dec, args, device: torch.device) -> dict[str, object]:
    high = getattr(diff_dec, "transformer", None)
    low = getattr(diff_dec, "transformer_2", None)
    _move_transformer_blocks(high, "cpu")
    _move_transformer_blocks(low, "cpu")
    for module in (high, low):
        if module is None:
            continue
        for handle in getattr(module, "_svdquant_block_offload_handles", []):
            handle.remove()
        for attr in (
            "_svdquant_block_offload_enabled",
            "_svdquant_block_offload_handles",
            "_svdquant_block_offload_hooked_indices",
            "_svdquant_block_offload_indices",
        ):
            if hasattr(module, attr):
                delattr(module, attr)
    args.block_offload_transformers = False
    if hasattr(args, "active_block_offload_count"):
        delattr(args, "active_block_offload_count")
    _cuda_empty_cache(device)
    return configure_transformer_memory(diff_dec, args, device)


def is_cuda_oom(exc: BaseException) -> bool:
    text = repr(exc)
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in text or "OutOfMemoryError" in text


def configure_transformer_memory(diff_dec, args, device: torch.device) -> dict[str, object]:
    high = getattr(diff_dec, "transformer", None)
    low = getattr(diff_dec, "transformer_2", None)
    config: dict[str, object] = {
        "initial_high_device": _module_device(high),
        "initial_low_device": _module_device(low),
    }
    if getattr(args, "block_offload_transformers", False):
        block_count = int(getattr(args, "active_block_offload_count", getattr(args, "block_offload_count", 16)))
        high_state = install_transformer_block_offload(high, device, block_count, peer=low)
        low_state = install_transformer_block_offload(low, device, block_count, peer=high)
        config.update(
            {
                "mode": "block_offload",
                "high_device": _module_device(high),
                "low_device": _module_device(low),
                "high_blocks": high_state,
                "low_blocks": low_state,
            }
        )
        return config

    if getattr(args, "keep_transformers_on_gpu", False):
        config.update(
            {
                "mode": "resident",
                "high_device": _module_device(high),
                "low_device": _module_device(low),
            }
        )
        return config

    if high is not None:
        high.to("cpu")
    if low is not None:
        low.to("cpu")
    _cuda_empty_cache(device)
    config.update(
        {
            "mode": "bernini_local_device_moves",
            "high_device": _module_device(high),
            "low_device": _module_device(low),
        }
    )
    return config
