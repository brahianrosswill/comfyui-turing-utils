"""Instance-local immutable assets and published task versions (no tensor cache)."""

import json
import os
import re
import shutil
import threading
import uuid
from pathlib import Path

import folder_paths
import av
from PIL import Image


LOCK = threading.RLock()
KINDS = {"image", "video", "audio", "mask"}
EXTENSIONS = {"image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"},
              "video": {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"},
              "audio": {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}}


def probe_import(path, kind):
    if path.suffix.lower() not in EXTENSIONS.get(kind, set()):
        raise ValueError(f"Unsupported {kind} file extension")
    if kind == "image":
        with Image.open(path) as image:
            metadata = {"width": image.width, "height": image.height}
            image.verify()
            return metadata
    with av.open(str(path)) as container:
        streams = container.streams.video if kind == "video" else container.streams.audio
        if not streams:
            raise ValueError(f"File does not contain {kind}")
        stream = streams[0]
        metadata = {"audio": bool(container.streams.audio),
                    "duration": float(container.duration or 0) / av.time_base}
        if kind == "video":
            metadata.update(width=stream.width, height=stream.height, fps=float(stream.average_rate or 24))
        return metadata


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Canvas paths must be non-empty relative paths inside their owning directory")
    return path


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Project:
    def __init__(self, directory):
        self.directory = directory
        self.root = contained(folder_paths.get_output_directory(), directory)
        self.assets = contained(self.root, "assets")
        self.assets.mkdir(parents=True, exist_ok=True)

    def state(self):
        path = contained(self.root, "state.json")
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"tasks": {}}

    def configure_cache(self, relative):
        cache_directory(relative)
        path = contained(self.root, "cache.json")
        value = {"directory": relative}
        if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != value:
            atomic_json(path, value)

    def cache(self):
        path = contained(self.root, "cache.json")
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"directory": self.directory}
        return cache_directory(value["directory"])

    def asset(self, asset_id):
        if not isinstance(asset_id, str) or not re.fullmatch(r"[0-9a-f]{32}", asset_id):
            raise ValueError("Invalid canvas asset ID")
        data = json.loads(contained(self.assets, f"{asset_id}.json").read_text(encoding="utf-8"))
        contained(self.assets, data["file"])
        return data

    def path(self, asset_id):
        return contained(self.assets, self.asset(asset_id)["file"])

    def reserve(self, extension):
        if not re.fullmatch(r"\.[a-zA-Z0-9]{1,10}", extension):
            raise ValueError("Unsupported filename extension")
        asset_id = uuid.uuid4().hex
        return asset_id, self.assets / f"{asset_id}{extension.lower()}"

    def register(self, asset_id, path, kind, name, metadata=None):
        if kind not in KINDS or path.parent.resolve() != self.assets.resolve():
            raise ValueError("Invalid asset kind or location")
        data = {"id": asset_id, "file": path.name, "kind": kind, "name": name,
                "metadata": metadata or {}}
        atomic_json(self.assets / f"{asset_id}.json", data)
        return data

    def copy(self, source, kind):
        asset_id, target = self.reserve(source.suffix)
        temporary = target.with_suffix(target.suffix + ".partial")
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
            metadata = probe_import(target, kind)
            return self.register(asset_id, target, kind, source.name, metadata)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)

    def publish(self, task, asset, signature, snapshot):
        with LOCK:
            state = self.state()
            item = state["tasks"].setdefault(str(task), {"history": []})
            record = {"asset": asset, "signature": signature, "snapshot": snapshot}
            item["history"].append(record)
            item.update(asset=asset, signature=signature)
            atomic_json(self.root / "state.json", state)
        return record

    def select(self, task, asset):
        with LOCK:
            state = self.state()
            item = state["tasks"][str(task)]
            record = next(r for r in item["history"] if r["asset"] == asset)
            item.update(asset=asset, signature=record["signature"])
            atomic_json(self.root / "state.json", state)


def cache_directory(relative):
    # Never use ComfyUI/temp: startup removes it.
    path = contained(Path(folder_paths.base_path) / ".cache", relative)
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_source(relative):
    # A browser cannot reveal a dragged file's server filesystem path.
    # Local-copy imports deliberately resolve only inside this instance's input.
    path = contained(folder_paths.get_input_directory(), relative)
    if not path.is_file():
        raise ValueError("Local-copy source must be a file in this instance's input directory")
    return path
