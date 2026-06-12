"""CameraSource implementations.

Everything hardware-specific stays behind the CameraSource protocol so the
pipeline runs identically off a USB camera or a looping video file.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator, Protocol

import cv2

from crittercam.config import CameraConfig
from crittercam.models import Frame

log = logging.getLogger(__name__)


class CameraSource(Protocol):
    def frames(self) -> Iterator[Frame]: ...


class UsbCamera:
    """V4L2 UVC camera via OpenCV. Reopens with backoff on failure."""

    REOPEN_DELAY_S = 2.0

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._cap: cv2.VideoCapture | None = None

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.cfg.device, cv2.CAP_V4L2)
        # Order matters on UVC bridges: MJPG fourcc must be set before the
        # resolution, or the camera silently stays in YUYV and caps at ~5 FPS.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.cfg.device}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        log.info("camera open device=%s %dx%d@%.0f", self.cfg.device, w, h, fps)
        return cap

    def frames(self) -> Iterator[Frame]:
        index = 0
        while True:
            if self._cap is None:
                try:
                    self._cap = self._open()
                except Exception:
                    log.exception("camera open failed, retrying in %.0fs", self.REOPEN_DELAY_S)
                    time.sleep(self.REOPEN_DELAY_S)
                    continue
            ok, image = self._cap.read()
            if not ok:
                log.warning("camera read failed, reopening")
                self._cap.release()
                self._cap = None
                time.sleep(self.REOPEN_DELAY_S)
                continue
            yield Frame(image=image, ts_monotonic=time.monotonic(), ts_wall=time.time(), index=index)
            index += 1


class FileCamera:
    """Loops a video file at its native FPS. Used in dev and tests."""

    def __init__(self, cfg: CameraConfig, loop: bool = True):
        self.path = Path(cfg.device).expanduser()
        self.loop = loop
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def frames(self) -> Iterator[Frame]:
        index = 0
        while True:
            cap = cv2.VideoCapture(str(self.path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            interval = 1.0 / fps
            next_due = time.monotonic()
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                now = time.monotonic()
                if now < next_due:
                    time.sleep(next_due - now)
                next_due += interval
                yield Frame(image=image, ts_monotonic=time.monotonic(), ts_wall=time.time(), index=index)
                index += 1
            cap.release()
            if not self.loop:
                return


def open_camera(cfg: CameraConfig) -> CameraSource:
    if cfg.kind == "usb":
        return UsbCamera(cfg)
    if cfg.kind == "file":
        return FileCamera(cfg)
    raise NotImplementedError(f"camera kind {cfg.kind!r} not implemented yet")
