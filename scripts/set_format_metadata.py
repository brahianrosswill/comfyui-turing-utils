from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from model_format import SUPPORTED_ARCHITECTURES, SVDINT4_FORMAT, validate_svdint4_metadata  # noqa: E402


def tensor_signature(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {
            key: (tuple(int(dim) for dim in handle.get_slice(key).get_shape()), str(handle.get_slice(key).get_dtype()))
            for key in handle.keys()
        }


def file_and_tensor_payload_sha256(path: Path) -> tuple[str, str]:
    """Hash the complete file and immutable tensor payload in one pass."""
    file_digest = hashlib.sha256()
    payload_digest = hashlib.sha256()
    with path.open("rb") as handle:
        header_size_bytes = handle.read(8)
        if len(header_size_bytes) != 8:
            raise RuntimeError(f"invalid safetensors header in {path}")
        file_digest.update(header_size_bytes)
        header_size = int.from_bytes(header_size_bytes, "little")
        header = handle.read(header_size)
        if len(header) != header_size:
            raise RuntimeError(f"truncated safetensors header in {path}")
        file_digest.update(header)
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            file_digest.update(chunk)
            payload_digest.update(chunk)
    return file_digest.hexdigest(), payload_digest.hexdigest()


def update_sidecar(path: Path, architecture: str, size: int, digest: str) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["format"] = SVDINT4_FORMAT
    data["architecture"] = architecture
    if "bytes" in data:
        data["bytes"] = size
    if "sha256" in data:
        data["sha256"] = digest
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        metadata["format"] = SVDINT4_FORMAT
        metadata["architecture"] = architecture
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def rewrite(path: Path, architecture: str, write_sidecar: bool) -> dict[str, object]:
    before = tensor_signature(path)
    _before_file_digest, before_payload = file_and_tensor_payload_sha256(path)
    temporary = path.with_name(path.name + ".format-tmp")
    if temporary.exists():
        temporary.unlink()
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        metadata["format"] = SVDINT4_FORMAT
        metadata["architecture"] = architecture
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        save_file(tensors, temporary, metadata=metadata)

    after = tensor_signature(temporary)
    if after != before:
        temporary.unlink()
        raise RuntimeError(f"tensor signature changed while updating {path}")
    digest, after_payload = file_and_tensor_payload_sha256(temporary)
    if after_payload != before_payload:
        temporary.unlink()
        raise RuntimeError(f"tensor payload changed while updating {path}")
    with safe_open(temporary, framework="pt", device="cpu") as handle:
        validate_svdint4_metadata(handle.metadata() or {}, temporary)
    os.replace(temporary, path)

    size = path.stat().st_size
    if write_sidecar:
        update_sidecar(path.with_name(path.name + ".json"), architecture, size, digest)
    return {
        "path": str(path),
        "format": SVDINT4_FORMAT,
        "architecture": architecture,
        "tensor_count": len(before),
        "tensor_payload_sha256": before_payload,
        "bytes": size,
        "sha256": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite SVDInt4 metadata in place while preserving every tensor.")
    parser.add_argument("--architecture", required=True, choices=sorted(SUPPORTED_ARCHITECTURES))
    parser.add_argument("--no-sidecar", action="store_true")
    parser.add_argument("models", nargs="+", type=Path)
    args = parser.parse_args()
    reports = []
    for index, path in enumerate(args.models, 1):
        print(f"[{index}/{len(args.models)}] rewriting {path}", file=sys.stderr, flush=True)
        reports.append(rewrite(path, args.architecture, not args.no_sidecar))
        print(f"[{index}/{len(args.models)}] verified {path}", file=sys.stderr, flush=True)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
