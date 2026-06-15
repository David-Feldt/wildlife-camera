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


def _resolve_class_ids(names: dict[int, str], classes: list[str]) -> list[int] | None:
    """Map configured class names to the model's class ids, or None for all.

    Class names come from the model file, so a fine-tuned model (milestone 5)
    just works once detector.classes is updated to match — but a typo or a stale
    config fails loudly here rather than silently detecting nothing."""
    if not classes:
        return None
    wanted = set(classes)
    missing = wanted - set(names.values())
    if missing:
        raise ValueError(f"unknown class names in config: {sorted(missing)}")
    return [i for i, n in names.items() if n in wanted]


def _extract_detections(results, names: dict[int, str]) -> list[Detection]:
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            Detection(
                class_name=names[int(box.cls)],
                confidence=float(box.conf),
                bbox=(x1, y1, x2, y2),
            )
        )
    return detections


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
        self.names = self.model.names  # {id: name}
        self.class_ids = _resolve_class_ids(self.names, cfg.classes)

    def detect(self, frame: Frame) -> list[Detection]:
        results = self.model.predict(
            frame.image,
            conf=self.confidence,
            classes=self.class_ids,
            device="cpu",  # pin to CPU: ultralytics auto-selects the GPU once CUDA
            verbose=False,  # is available, but this backend must stay off it (the
        )                   # GPU is for the tensorrt backend / on-device training)
        return _extract_detections(results, self.names)


def _ensure_engine(cfg: DetectorConfig, model_path: Path) -> Path:
    """Return a path to a TensorRT engine, building it on this device if needed.

    Engines are device-specific (they bake in the GPU's compute capability), so
    they are never shipped or committed — they are exported here from the .pt on
    first run. detector.model normally points at the .pt; we cache the engine as
    a sibling .engine and rebuild it whenever the .pt is newer (e.g. a freshly
    deployed milestone-5 best.pt). If detector.model already points at a .engine,
    we use it as-is and never try to rebuild."""
    if model_path.suffix == ".engine":
        if not model_path.exists():
            raise FileNotFoundError(
                f"tensorrt engine {model_path} not found; point detector.model "
                f"at the .pt so it can be exported on this device"
            )
        return model_path

    engine_path = model_path.with_suffix(".engine")
    if engine_path.exists() and (
        not model_path.exists()
        or engine_path.stat().st_mtime >= model_path.stat().st_mtime
    ):
        log.info("using cached tensorrt engine %s", engine_path)
        return engine_path

    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("model %s missing, downloading", model_path)
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(str(model_path))

    from ultralytics import YOLO  # heavy import, keep local

    log.info(
        "exporting tensorrt engine from %s (fp16=%s imgsz=%d); this can take "
        "several minutes on the Orin",
        model_path, cfg.trt_fp16, cfg.trt_imgsz,
    )
    exported = Path(
        YOLO(str(model_path), task="detect").export(
            format="engine", half=cfg.trt_fp16, imgsz=cfg.trt_imgsz, device=0,
        )
    )
    if exported != engine_path:
        exported.replace(engine_path)
    return engine_path


class TrtDetector:
    """Ultralytics YOLO over a TensorRT engine. Much faster than CpuDetector
    (the engine runs on the Orin's GPU), so it is the production backend. The
    engine is built from the .pt on this device the first time it is needed; see
    _ensure_engine. Inference and class handling are otherwise identical to the
    CPU backend — ultralytics presents the same results interface for engines."""

    def __init__(self, cfg: DetectorConfig, model_path: Path):
        from ultralytics import YOLO  # heavy import, keep local

        engine_path = _ensure_engine(cfg, model_path)
        self.model = YOLO(str(engine_path), task="detect")
        self.confidence = cfg.confidence
        self.names = self.model.names  # {id: name}, read from engine metadata
        self.class_ids = _resolve_class_ids(self.names, cfg.classes)

    def detect(self, frame: Frame) -> list[Detection]:
        results = self.model.predict(
            frame.image,
            conf=self.confidence,
            classes=self.class_ids,
            verbose=False,
        )
        return _extract_detections(results, self.names)


def open_detector(cfg: DetectorConfig, model_path: Path) -> Detector:
    if cfg.backend == "cpu":
        return CpuDetector(cfg, model_path)
    if cfg.backend == "tensorrt":
        return TrtDetector(cfg, model_path)
    raise NotImplementedError(f"detector backend {cfg.backend!r} not implemented yet")
