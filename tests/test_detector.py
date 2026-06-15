"""Detector backend selection, class resolution, and TensorRT engine caching.

No YOLO/TensorRT here: ultralytics is mocked so these run on any machine. The
real engine export is exercised on the Orin (it bakes in the GPU's compute
capability and takes minutes), which no unit test can stand in for.
"""
import sys
import types

import pytest

from crittercam.config import DetectorConfig
from crittercam.models import Frame
from crittercam.tracker.detector import (
    CpuDetector,
    TrtDetector,
    _resolve_class_ids,
    open_detector,
)

COCO = {0: "person", 15: "cat", 16: "dog"}


# --- class resolution -------------------------------------------------------

def test_resolve_class_ids_empty_means_all():
    assert _resolve_class_ids(COCO, []) is None


def test_resolve_class_ids_filters_to_wanted():
    assert sorted(_resolve_class_ids(COCO, ["cat", "dog"])) == [15, 16]


def test_resolve_class_ids_rejects_unknown_name():
    with pytest.raises(ValueError, match="squirrel"):
        _resolve_class_ids(COCO, ["cat", "squirrel"])


# --- fake ultralytics -------------------------------------------------------

class _FakeBox:
    def __init__(self, cls, conf, xyxy):
        import numpy as np
        self.cls = cls
        self.conf = conf
        self.xyxy = np.array([xyxy], dtype=np.float32)  # tensor-like: has .tolist()


class _FakeResults:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    """Stands in for ultralytics.YOLO. Records the path it was loaded from and
    returns one scripted cat detection from predict()."""

    loaded = []
    exported = []

    def __init__(self, path, task="detect"):
        self.path = str(path)
        self.names = COCO
        FakeYOLO.loaded.append(self.path)

    def predict(self, image, conf, classes, verbose, device=None):
        self.last_device = device
        box = _FakeBox(15, 0.9, [10.0, 20.0, 110.0, 220.0])
        return [_FakeResults([box])]

    def export(self, format, half, imgsz, device):
        out = self.path.replace(".pt", ".engine")
        FakeYOLO.exported.append((out, format, half, imgsz, device))
        # ultralytics writes the engine next to the .pt; mimic that so the
        # caller's rename path is a no-op.
        from pathlib import Path
        Path(out).write_bytes(b"ENGINE")
        return out


@pytest.fixture
def fake_yolo(monkeypatch):
    FakeYOLO.loaded = []
    FakeYOLO.exported = []
    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    return FakeYOLO


def _frame():
    import numpy as np
    return Frame(image=np.zeros((4, 4, 3), dtype=np.uint8),
                 ts_monotonic=0.0, ts_wall=0.0, index=0)


# --- backend selection ------------------------------------------------------

def test_open_detector_cpu(fake_yolo, tmp_path):
    model = tmp_path / "yolo11n.pt"
    model.write_bytes(b"PT")
    d = open_detector(DetectorConfig(backend="cpu"), model)
    assert isinstance(d, CpuDetector)


def test_open_detector_tensorrt(fake_yolo, tmp_path):
    model = tmp_path / "yolo11n.pt"
    model.write_bytes(b"PT")
    d = open_detector(DetectorConfig(backend="tensorrt"), model)
    assert isinstance(d, TrtDetector)


def test_open_detector_unknown_backend_raises(tmp_path):
    cfg = DetectorConfig.model_construct(backend="bogus", classes=[])
    with pytest.raises(NotImplementedError):
        open_detector(cfg, tmp_path / "x.pt")


# --- detection extraction ---------------------------------------------------

def test_cpu_detect_extracts_boxes(fake_yolo, tmp_path):
    model = tmp_path / "yolo11n.pt"
    model.write_bytes(b"PT")
    d = CpuDetector(DetectorConfig(backend="cpu"), model)
    out = d.detect(_frame())
    assert d.model.last_device == "cpu"  # must not auto-grab the GPU
    assert len(out) == 1
    assert out[0].class_name == "cat"
    assert out[0].confidence == pytest.approx(0.9)
    assert out[0].bbox == (10.0, 20.0, 110.0, 220.0)


# --- engine export / caching ------------------------------------------------

def test_tensorrt_exports_engine_from_pt(fake_yolo, tmp_path):
    model = tmp_path / "yolo11n.pt"
    model.write_bytes(b"PT")
    TrtDetector(DetectorConfig(backend="tensorrt", trt_fp16=True, trt_imgsz=512), model)
    # exported once, with our config knobs
    assert len(fake_yolo.exported) == 1
    out, fmt, half, imgsz, device = fake_yolo.exported[0]
    assert fmt == "engine" and half is True and imgsz == 512
    assert (tmp_path / "yolo11n.engine").exists()
    # and the engine — not the .pt — is what got loaded for inference
    assert fake_yolo.loaded[-1].endswith("yolo11n.engine")


def test_tensorrt_reuses_fresh_cached_engine(fake_yolo, tmp_path):
    model = tmp_path / "yolo11n.pt"
    model.write_bytes(b"PT")
    engine = tmp_path / "yolo11n.engine"
    engine.write_bytes(b"ENGINE")
    # make the engine newer than the .pt
    import os
    os.utime(engine, (model.stat().st_mtime + 10, model.stat().st_mtime + 10))

    TrtDetector(DetectorConfig(backend="tensorrt"), model)
    assert fake_yolo.exported == []  # cache hit, no rebuild


def test_tensorrt_rebuilds_when_pt_is_newer(fake_yolo, tmp_path):
    model = tmp_path / "yolo11n.pt"
    model.write_bytes(b"PT")
    engine = tmp_path / "yolo11n.engine"
    engine.write_bytes(b"OLD")
    import os
    # .pt is newer than the stale engine (e.g. a freshly deployed best.pt)
    os.utime(model, (engine.stat().st_mtime + 10, engine.stat().st_mtime + 10))

    TrtDetector(DetectorConfig(backend="tensorrt"), model)
    assert len(fake_yolo.exported) == 1


def test_tensorrt_engine_path_used_directly(fake_yolo, tmp_path):
    engine = tmp_path / "best.engine"
    engine.write_bytes(b"ENGINE")
    TrtDetector(DetectorConfig(backend="tensorrt", model="best.engine"), engine)
    assert fake_yolo.exported == []
    assert fake_yolo.loaded[-1].endswith("best.engine")


def test_tensorrt_missing_engine_path_raises(fake_yolo, tmp_path):
    missing = tmp_path / "best.engine"
    with pytest.raises(FileNotFoundError):
        TrtDetector(DetectorConfig(backend="tensorrt", model="best.engine"), missing)
