"""Shared dataclasses passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    image: np.ndarray          # BGR, as captured
    ts_monotonic: float        # time.monotonic() at capture
    ts_wall: float             # time.time() at capture
    index: int                 # capture sequence number


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # pixel x1, y1, x2, y2

    def bbox_normalized(self, width: int, height: int) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height)


@dataclass
class Track:
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # pixel x1, y1, x2, y2
    age_frames: int = 1


@dataclass
class FrameResult:
    """A frame plus everything the pipeline derived from it."""
    frame: Frame
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    inferred: bool = False     # False when detections were reused from a prior frame
