"""MJPEG-AVI clip playback for browsers.

Browsers cannot play MJPEG-AVI in a <video> tag, so clips are re-served the
same way as the live view: a multipart MJPEG stream of the clip's frames,
paced at the frame rate recorded in the AVI header. Frames are stored in the
container as encoded JPEGs, so playback is parse-and-copy — no decode, no
re-encode, no ffmpeg dependency on the device.

Seeking: a clip is scanned once into a frame-offset index (header reads and
seeks only — no frame data, so it's fast even on a 300-second clip) and the
index is cached, making /play?start=N and single-frame reads cheap. Clips
can be hundreds of MB, so the browser never downloads one just to scrub it.

The parser is deliberately minimal: it reads the finalized files that the
tracker's MjpegAviWriter produces (and plain MJPG AVIs generally), not
arbitrary video.
"""
from __future__ import annotations

import struct
import time
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Iterator

from crittercam.web.live import mjpeg_part

FALLBACK_FPS = 20.0
MAX_PLAYBACK_FPS = 60.0

FrameIndex = tuple[tuple[int, int], ...]  # (byte offset, size) per frame


def _scan(f: BinaryIO, name: str) -> tuple[float, FrameIndex]:
    """Walk the container and index the frames without reading their data."""
    riff = f.read(12)
    if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:] != b"AVI ":
        raise ValueError(f"{name}: not an AVI file")
    fps = FALLBACK_FPS
    frames: list[tuple[int, int]] = []
    while True:
        header = f.read(8)
        if len(header) < 8:
            raise ValueError(f"{name}: no movi list (unfinished clip?)")
        fourcc, size = header[:4], struct.unpack("<I", header[4:])[0]
        if fourcc != b"LIST":
            f.seek(size + (size & 1), 1)
            continue
        list_type = f.read(4)
        if list_type == b"hdrl":
            fps = _hdrl_fps(f, size - 4) or fps
        elif list_type == b"movi":
            end = f.tell() + size - 4
            while f.tell() + 8 <= end:
                chunk = f.read(8)
                if len(chunk) < 8:
                    break
                cc, csize = chunk[:4], struct.unpack("<I", chunk[4:])[0]
                if cc == b"00dc" and csize:
                    frames.append((f.tell(), csize))
                f.seek(csize + (csize & 1), 1)
            return fps, tuple(frames)
        else:
            f.seek(size - 4, 1)
        if size & 1:
            f.seek(1, 1)


def _hdrl_fps(f: BinaryIO, size: int) -> float | None:
    """Read the frame rate from the avih chunk inside the hdrl list,
    consuming the whole list either way."""
    end = f.tell() + size
    fps = None
    while f.tell() + 8 <= end:
        header = f.read(8)
        fourcc, csize = header[:4], struct.unpack("<I", header[4:])[0]
        if fourcc == b"avih" and csize >= 4:
            us_per_frame = struct.unpack("<I", f.read(4))[0]
            if us_per_frame:
                fps = 1_000_000 / us_per_frame
            f.seek(csize - 4 + (csize & 1), 1)
        else:
            f.seek(csize + (csize & 1), 1)
    f.seek(end)
    return fps


@lru_cache(maxsize=32)
def _cached_index(path_str: str, size: int, mtime_ns: int) -> tuple[float, FrameIndex]:
    with open(path_str, "rb") as f:
        return _scan(f, Path(path_str).name)


def clip_index(path: Path) -> tuple[float, FrameIndex]:
    """(fps, frame index) for a clip; cached, invalidated by size/mtime.

    Raises ValueError if the file is not a finalized MJPEG-AVI.
    """
    st = path.stat()
    return _cached_index(str(path), st.st_size, st.st_mtime_ns)


def clip_info(path: Path) -> tuple[float, int]:
    fps, frames = clip_index(path)
    return fps, len(frames)


def read_frame(path: Path, n: int) -> bytes:
    """A single frame's JPEG bytes. Raises IndexError if n is out of range."""
    fps, frames = clip_index(path)
    if not 0 <= n < len(frames):
        raise IndexError(n)
    offset, size = frames[n]
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(size)


def open_clip(path: Path, start: int = 0) -> tuple[float, Iterator[bytes]]:
    """(fps, iterator of JPEG frames) beginning at frame `start`.

    Raises ValueError if the file is not a finalized MJPEG-AVI.
    """
    fps, frames = clip_index(path)
    return fps, _read_frames(path, frames[max(start, 0):])


def _read_frames(path: Path, frames: FrameIndex) -> Iterator[bytes]:
    with open(path, "rb") as f:
        for offset, size in frames:
            f.seek(offset)
            data = f.read(size)
            if len(data) < size:
                return
            yield data


def clip_mjpeg_stream(fps: float, frames: Iterator[bytes]) -> Iterator[bytes]:
    """Wrap clip frames as a paced multipart MJPEG stream."""
    interval = 1.0 / max(min(fps, MAX_PLAYBACK_FPS), 1.0)
    next_due = time.monotonic()
    for jpeg in frames:
        yield mjpeg_part(jpeg)
        next_due += interval
        delay = next_due - time.monotonic()
        if delay > 0:
            time.sleep(delay)
