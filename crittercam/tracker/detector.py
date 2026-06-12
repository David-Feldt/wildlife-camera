"""Detector interface + backends."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from crittercam.config import DetectorConfig
from crittercam.models import Detection, Frame

log = logging.getLogger(__name__)


class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]: ...


class CpuDetector:
    """Ultralytics YOLO .pt on CPU. Slow (~1-3 FPS on Orin's CPU) but correct;
    the pipeline compensates with detector.infer_every_n."""

    def __init__(self, cfg: DetectorConfig, model_path: Path):
        from ultralytics import YOLO  # heavy import, keep local

        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            log.info("model %s missing, downloading", model_path)
            from ultralytics.utils.downloads import attempt_download_asset
            attempt_download_asset(str(model_path))
        self.model = YOLO(str(model_path), task="detect")
        self.confidence = cfg.confidence
        names = self.model.names  # {id: name}
        self.class_ids: list[int] | None = None
        if cfg.classes:
            wanted = set(cfg.classes)
            self.class_ids = [i for i, n in names.items() if n in wanted]
            missing = wanted - set(names.values())
            if missing:
                raise ValueError(f"unknown class names in config: {sorted(missing)}")
        self.names = names

    def detect(self, frame: Frame) -> list[Detection]:
        results = self.model.predict(
            frame.image,
            conf=self.confidence,
            classes=self.class_ids,
            verbose=False,
        )
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_name=self.names[int(box.cls)],
                    confidence=float(box.conf),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections


def open_detector(cfg: DetectorConfig, model_path: Path) -> Detector:
    if cfg.backend == "cpu":
        return CpuDetector(cfg, model_path)
    raise NotImplementedError(f"detector backend {cfg.backend!r} not implemented yet")
