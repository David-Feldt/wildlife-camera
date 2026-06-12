# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Crittercam: fully local wildlife detection, recording, and logging running on a Jetson Orin Nano (this machine). Currently at milestone 2 (IoU tracking, sighting rows, MJPEG-AVI clip recording with preroll, watermark pruning); TensorRT backend is milestone 4, a fine-tuned backyard wildlife model is milestone 5 — `NotImplementedError` branches in `capture.py`/`detector.py` mark the seams where csi/rtsp cameras and the tensorrt backend land.

## Milestone 5: fine-tuned wildlife model (plan)

COCO has no squirrel/raccoon/opossum/deer/fox classes, so stock YOLO mislabels most backyard wildlife. Plan: fine-tune `yolo11n.pt` on camera-trap datasets (ENA24-detection has boxes; NACTI boxed via MegaDetector) plus empty-yard background frames, validating only on this camera's real sightings (split by day/clip, never by frame). Constraints that affect code along the way:

- **M2 recorder must save clean (un-annotated) frames** for sightings — they become training/validation images.
- Training happens off-device (desktop GPU/Colab); the Orin Nano is inference-only.
- A fine-tuned model is a drop-in: class names come from the model file (`CpuDetector` reads `model.names`), so deploying is copying `best.pt` to `<data_root>/models/` and setting `detector.model` — but `detector.classes` in config must be updated to match the new class list or startup validation fails.
- TensorRT engines (M4) must be exported on the Jetson itself; engine files are device-specific.
- The sightings DB doubles as the active-learning queue: low-confidence detections are the labeling targets for the next training round.

## Environment constraints

- **Never pip-install `opencv-python`.** The venv is created with `--system-site-packages` specifically to pick up the system OpenCV (GStreamer-enabled JetPack build). It is intentionally absent from `pyproject.toml`.
- Python 3.10, venv at `.venv/`.
- `numpy<2` is pinned (JetPack compatibility).

## Commands

```bash
source .venv/bin/activate
pip install -e ".[dev]"            # install with dev deps (pytest, httpx)

python -m crittercam.tracker.main  # run the tracker process
python -m crittercam.web.main      # run the web process (port 8080)

pytest                             # run tests (testpaths = tests/)
pytest tests/test_foo.py::test_bar # run a single test
```

To develop without camera hardware, set `camera.kind: file` and point `camera.device` at a video file (e.g. `tests/fixtures/sample.mp4`) in the user config override.

## Architecture

Two independent processes that never import each other's modules and share only two channels:

1. **Tracker** (`crittercam/tracker/`) — capture → detect → track → record → publish pipeline. A capture thread feeds a `maxsize=2` queue with drop-oldest semantics (`put_latest`) so the pipeline never falls behind real time; the main loop runs YOLO every `detector.infer_every_n` frames (CPU inference is ~1–3 FPS) and reuses the last detections in between, publishing annotated JPEGs every frame. On inference frames, `tracking.IouTracker` associates detections to tracks (class-agnostic on purpose — COCO labels flicker on unfamiliar animals) and `events.EventManager` owns the sighting lifecycle: open at `min_track_frames`, close after `linger_seconds` with no tracks, split at `max_clip_seconds`. `recorder.ClipRecorder` keeps a `preroll_seconds` ring buffer of clean JPEGs and muxes them into MJPEG-AVI (`recorder.MjpegAviWriter`, no re-encode) under `<data_root>/clips/`.
2. **Web** (`crittercam/web/`) — FastAPI app that re-serves frames as MJPEG and exposes `/api/status` plus the static UI.

The two channels:

- **ZMQ PUB/SUB** (`tcp://127.0.0.1:5555`): tracker PUBs JPEG frames; each browser connection gets its own SUB socket with CONFLATE so slow clients only ever see the newest frame. Messages must stay **single-part** — CONFLATE does not support multipart.
- **SQLite** (`<data_root>/critters.db`, WAL mode): **single-writer discipline** — the tracker owns all writes to `sightings`; the web process only writes favorites/deletes/config. Tracker liveness is a heartbeat row in `config_kv` that the web process reads.

Hardware specifics stay behind protocols: `CameraSource` (`capture.py`) and `Detector` (`detector.py`), each selected by an `open_*` factory from config. New backends extend those factories.

## Config

`config/default.yaml` ships defaults; user overrides at `<data_root>/config.yaml` (default `~/wildlife-camera-data/config.yaml`) are deep-merged over it at startup, then validated by the pydantic models in `crittercam/config.py`. New config keys need both the YAML default and the pydantic field. Runtime/mutable state goes in the `config_kv` table, not YAML.

## Migrations

Numbered SQL files in `crittercam/migrations/` (`001_init.sql`, …), applied in order at every DB open and tracked via `PRAGMA user_version`. Add a new numbered file for schema changes; never edit an applied migration.

## Gotchas

- On UVC cameras the MJPG fourcc must be set **before** width/height (see `UsbCamera._open`), or the camera silently stays in YUYV capped at ~5 FPS.
- `FramePublisher.publish` returns a JPEG of the **clean** frame for the recorder's ring buffer; boxes/status text are drawn only on the live-stream copy (clips feed milestone 5 training and must stay un-annotated) — keep that contract.
- `pip install -e ".[dev]"` will pull in `opencv-python` (ultralytics depends on it, and the system cv2 has no pip metadata so pip thinks it's missing). If that happens, `pip uninstall opencv-python` to restore the GStreamer-enabled system build.
- YOLO model files (`*.pt`, `*.engine`) are gitignored and auto-downloaded to `<data_root>/models/` on first run.
