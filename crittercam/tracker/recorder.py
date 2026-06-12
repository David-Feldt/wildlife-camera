"""Clip recording: preroll ring buffer + minimal MJPEG-in-AVI muxer.

The ring buffer stores the encoded JPEGs that FramePublisher.publish returns
(that contract exists for us) and the muxer writes them into the container
as-is — no decode, no re-encode. Buffering raw 1280x720 BGR instead would
cost ~28x the memory and re-encoding would burn CPU the detector needs.
"""
from __future__ import annotations

import logging
import struct
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Used when a clip is too short to measure its own frame rate.
FALLBACK_FPS = 20.0


def _jpeg_dims(jpeg: bytes) -> tuple[int, int]:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    return w, h


class MjpegAviWriter:
    """AVI container around an MJPG stream. Header fields that depend on the
    final frame count and rate are written as placeholders and patched on
    close, so a crash mid-clip leaves a recognizably unfinished file."""

    def __init__(self, path: Path, width: int, height: int):
        self.width = width
        self.height = height
        self._index: list[tuple[int, int]] = []  # (offset within movi, size)
        self._frames = 0
        self._f = open(path, "wb")
        self._write_headers()

    def _write_headers(self) -> None:
        f = self._f
        f.write(b"RIFF\x00\x00\x00\x00AVI ")  # riff size patched on close
        f.write(b"LIST" + struct.pack("<I", 4 + 8 + 56 + 8 + 4 + 8 + 56 + 8 + 40) + b"hdrl")
        f.write(b"avih" + struct.pack("<I", 56))
        self._avih_off = f.tell()
        f.write(struct.pack(
            "<14I",
            0,                # us per frame (patched)
            0, 0,
            0x10,             # AVIF_HASINDEX
            0,                # total frames (patched)
            0, 1,
            self.width * self.height * 3,
            self.width, self.height,
            0, 0, 0, 0,
        ))
        f.write(b"LIST" + struct.pack("<I", 4 + 8 + 56 + 8 + 40) + b"strl")
        f.write(b"strh" + struct.pack("<I", 56))
        self._strh_off = f.tell()
        f.write(struct.pack(
            "<4s4sIHH8I4H",
            b"vids", b"MJPG",
            0, 0, 0, 0,
            1,                # dwScale (patched with rate on close)
            0,                # dwRate (patched)
            0,
            0,                # dwLength in frames (patched)
            self.width * self.height * 3,
            0xFFFFFFFF,       # quality: default
            0,
            0, 0, 0, 0,
        ))
        f.write(b"strf" + struct.pack("<I", 40))
        f.write(struct.pack(
            "<IiiHH4sIiiII",
            40, self.width, self.height, 1, 24, b"MJPG",
            self.width * self.height * 3, 0, 0, 0, 0,
        ))
        f.write(b"LIST\x00\x00\x00\x00movi")
        self._movi_size_off = f.tell() - 8
        self._movi_start = f.tell() - 4  # offset of the 'movi' fourcc

    def add(self, jpeg: bytes) -> None:
        f = self._f
        self._index.append((f.tell() - self._movi_start, len(jpeg)))
        f.write(b"00dc" + struct.pack("<I", len(jpeg)))
        f.write(jpeg)
        if len(jpeg) % 2:
            f.write(b"\x00")  # RIFF chunks are word-aligned
        self._frames += 1

    def close(self, fps: float) -> None:
        f = self._f
        movi_end = f.tell()
        f.write(b"idx1" + struct.pack("<I", 16 * len(self._index)))
        for offset, size in self._index:
            f.write(b"00dc" + struct.pack("<III", 0x10, offset, size))  # AVIIF_KEYFRAME
        riff_end = f.tell()

        fps = max(fps, 1.0)
        f.seek(4)
        f.write(struct.pack("<I", riff_end - 8))
        f.seek(self._avih_off)
        f.write(struct.pack("<I", int(1_000_000 / fps)))
        f.seek(self._avih_off + 16)
        f.write(struct.pack("<I", self._frames))
        f.seek(self._strh_off + 20)
        f.write(struct.pack("<II", 1000, round(fps * 1000)))  # scale, rate
        f.seek(self._strh_off + 32)
        f.write(struct.pack("<I", self._frames))
        f.seek(self._movi_size_off)
        f.write(struct.pack("<I", movi_end - self._movi_start))
        f.close()


@dataclass
class ClipResult:
    path: Path
    frame_count: int
    duration_s: float


class ClipRecorder:
    """Holds the last preroll_seconds of published JPEGs; on start() they are
    flushed into the clip so the recording begins before the trigger."""

    def __init__(self, preroll_seconds: float):
        self.preroll_s = preroll_seconds
        self._buffer: deque[tuple[float, bytes]] = deque()
        self._recording = False
        self._writer: MjpegAviWriter | None = None
        self._path: Path | None = None
        self._first_ts = 0.0
        self._last_ts = 0.0
        self._frames = 0

    @property
    def recording(self) -> bool:
        return self._recording

    def add_frame(self, ts_monotonic: float, jpeg: bytes) -> None:
        if not jpeg:
            return
        if self._recording:
            self._write(ts_monotonic, jpeg)
            return
        self._buffer.append((ts_monotonic, jpeg))
        horizon = ts_monotonic - self.preroll_s
        while self._buffer and self._buffer[0][0] < horizon:
            self._buffer.popleft()

    def start(self, path: Path) -> None:
        if self._recording:
            return
        self._path = path
        self._frames = 0
        self._writer = None
        self._recording = True
        for ts, jpeg in self._buffer:
            self._write(ts, jpeg)
        self._buffer.clear()

    def _write(self, ts: float, jpeg: bytes) -> None:
        if self._writer is None:
            width, height = _jpeg_dims(jpeg)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = MjpegAviWriter(self._path, width, height)
            self._first_ts = ts
        self._writer.add(jpeg)
        self._last_ts = ts
        self._frames += 1

    def stop(self) -> ClipResult | None:
        if not self._recording:
            return None
        self._recording = False
        writer, path, frames = self._writer, self._path, self._frames
        self._writer = None
        self._path = None
        if writer is None:
            return None
        duration = max(self._last_ts - self._first_ts, 0.0)
        fps = (frames - 1) / duration if frames > 1 and duration > 0 else FALLBACK_FPS
        writer.close(fps)
        log.info("clip closed %s frames=%d duration=%.1fs", path.name, frames, duration)
        return ClipResult(path=path, frame_count=frames, duration_s=duration)
