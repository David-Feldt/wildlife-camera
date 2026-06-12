"""YAML config loading and validation.

Defaults ship in config/default.yaml; user overrides live at
<data_root>/config.yaml and are deep-merged over the defaults.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class CameraConfig(BaseModel):
    kind: Literal["usb", "file", "csi", "rtsp"] = "usb"
    device: str = "/dev/video0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"


class DetectorConfig(BaseModel):
    backend: Literal["cpu", "tensorrt"] = "cpu"
    model: str = "yolo11n.pt"
    confidence: float = Field(0.45, ge=0.0, le=1.0)
    classes: list[str] = []
    infer_every_n: int = Field(1, ge=1)


class EventsConfig(BaseModel):
    min_track_frames: int = 5
    linger_seconds: float = 5.0
    preroll_seconds: float = 10.0
    max_clip_seconds: float = 300.0


class StorageConfig(BaseModel):
    disk_high_watermark: float = 0.85
    disk_low_watermark: float = 0.75

    @field_validator("disk_low_watermark")
    @classmethod
    def low_below_high(cls, v, info):
        if "disk_high_watermark" in info.data and v >= info.data["disk_high_watermark"]:
            raise ValueError("disk_low_watermark must be below disk_high_watermark")
        return v


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    live_fps: float = 10.0


class Config(BaseModel):
    data_root: Path
    camera: CameraConfig = CameraConfig()
    detector: DetectorConfig = DetectorConfig()
    events: EventsConfig = EventsConfig()
    storage: StorageConfig = StorageConfig()
    web: WebConfig = WebConfig()
    zmq_frame_endpoint: str = "tcp://127.0.0.1:5555"
    log_level: str = "INFO"

    @field_validator("data_root")
    @classmethod
    def expand_data_root(cls, v: Path) -> Path:
        return v.expanduser()

    @property
    def model_path(self) -> Path:
        return self.data_root / "models" / self.detector.model

    @property
    def db_path(self) -> Path:
        return self.data_root / "critters.db"

    @property
    def clips_dir(self) -> Path:
        return self.data_root / "clips"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_root / "thumbs"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(default_path: Path = DEFAULT_CONFIG_PATH) -> Config:
    raw = yaml.safe_load(default_path.read_text())
    data_root = Path(raw["data_root"]).expanduser()
    user_path = data_root / "config.yaml"
    if user_path.exists():
        user_raw = yaml.safe_load(user_path.read_text()) or {}
        raw = _deep_merge(raw, user_raw)
    cfg = Config.model_validate(raw)
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    return cfg


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
