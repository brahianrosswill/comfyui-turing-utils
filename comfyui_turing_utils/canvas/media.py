"""Decode selected material only when a task actually executes."""

import av
import numpy as np
import torch
from PIL import Image, ImageOps


def read_material(project, ref, width=0, height=0, max_frames=0):
    item = project.asset(ref["asset"])
    path = project.path(ref["asset"])
    start, duration = float(ref.get("start", 0)), float(ref.get("duration", 0))
    if not np.isfinite(start) or not np.isfinite(duration) or start < 0 or duration < 0:
        raise ValueError("Material time range must be finite and non-negative")
    end = start + duration if duration else float("inf")
    if item["kind"] == "mask":
        return None, None, torch.from_numpy(np.load(path, allow_pickle=False))
    if item["kind"] == "image":
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if width and height:
                image = ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS)
            pixels = torch.from_numpy(np.asarray(image).copy()).float().unsqueeze(0) / 255
        return pixels, None, pixels[..., 0]
    frames = []
    if item["kind"] == "video" and not ref.get("slot", 0):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            origin = float(stream.start_time * stream.time_base) if stream.start_time else 0
            container.seek(int((start + origin) / stream.time_base), stream=stream)
            next_time = start
            previous = None
            for frame in container.decode(stream):
                time = float(frame.time or 0) - origin
                if time < start:
                    continue
                if time >= end:
                    break
                rgb = frame.reformat(width=width or frame.width, height=height or frame.height, format="rgb24").to_ndarray()
                while next_time < time - 1e-7:
                    frames.append(previous if previous is not None else rgb)
                    next_time += 1 / 24
                    if max_frames and len(frames) >= max_frames:
                        break
                if max_frames and len(frames) >= max_frames:
                    break
                if next_time <= time + 1e-7:
                    frames.append(rgb)
                    next_time += 1 / 24
                previous = rgb
                if max_frames and len(frames) >= max_frames:
                    break
        if not frames:
            raise ValueError("Selected video interval contains no frames")
    chunks = []
    rate = 44100
    with av.open(str(path)) as container:
        if container.streams.audio:
            stream = container.streams.audio[0]
            origin = float(stream.start_time * stream.time_base) if stream.start_time else 0
            container.seek(int((start + origin) / stream.time_base), stream=stream)
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
            audio_end = min(end, start + len(frames) / 24) if frames else end
            cursor = start

            def collect(converted):
                nonlocal cursor
                t = float(converted.time) - origin if converted.time is not None else cursor
                data = converted.to_ndarray()
                cursor = t + data.shape[1] / rate
                lo = max(0, round((start - t) * rate))
                hi = min(data.shape[1], round((audio_end - t) * rate)) if np.isfinite(audio_end) else data.shape[1]
                if hi > lo:
                    chunks.append((max(0, round((t - start) * rate) + lo), data[:, lo:hi]))

            for frame in container.decode(stream):
                time = float(frame.time or 0) - origin
                for converted in resampler.resample(frame):
                    collect(converted)
                if time >= audio_end:
                    break
            for converted in resampler.resample(None):
                collect(converted)
    audio = None
    if chunks:
        length = round((audio_end - start) * rate) if np.isfinite(audio_end) else max(pos + data.shape[1] for pos, data in chunks)
        waveform = np.zeros((2, length), dtype=np.float32)
        for pos, data in chunks:
            waveform[:, pos:pos + data.shape[1]] = data[:, :max(0, length - pos)]
        audio = {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": rate}
    if ref.get("slot", 0) and audio is None:
        raise ValueError("The selected material has no audio in this interval")
    images = torch.from_numpy(np.stack(frames)).float() / 255 if frames else None
    return images, audio, images[..., 0] if images is not None else None
