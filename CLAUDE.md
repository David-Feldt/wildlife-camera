# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Crittercam: fully local wildlife detection, recording, and logging running on a Jetson Orin Nano (this machine). Milestone 3 is the last fully-deployed milestone (IoU tracking, sighting rows, MJPEG-AVI clip recording with preroll, count-cap + watermark clip pruning, gallery UI with clip playback/favorites/delete and detection-box thumbnail zoom). A fine-tuned backyard wildlife model is milestone 5 and is now the active focus; the remaining `NotImplementedError` branch in `capture.py` marks the seam where csi/rtsp cameras land.

**Milestone 4 (TensorRT backend) — implemented, deferred by choice (env now unblocked).** The `tensorrt` backend (`TrtDetector` in `detector.py`) is written and unit-tested (it exports a device-specific engine from the `.pt` on first run, rebuilding when the `.pt` is newer). The default config stays on `backend: cpu` by decision (2026-06-14): CPU YOLO at `infer_every_n: 5` is adequate for current slow-moving backyard scenes. To use M4 now just set `detector.backend: tensorrt` — the GPU stack is fixed (see below), no code change needed; it's still unvalidated on real hardware.

**GPU stack (fixed 2026-06-14).** The venv originally had a generic `torch 2.12.0+cu130` wheel (CUDA 13) so `torch.cuda` was dead on this CUDA-12.6 device. Working recipe: `sudo apt install nvidia-jetpack` (completes the CUDA 12.6 runtime + cuDNN + `tensorrt` 10.3 python bindings, picked up via the venv's `--system-site-packages`), then `pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch==2.8.0 torchvision==0.23.0`. **Pin torch to 2.8.0** — 2.9–2.11 from that index drag in `libcudss`/`libcusparseLt` and have a broken `cublasCreate` (CUBLAS_STATUS_ALLOC_FAILED) on this JetPack. `CpuDetector` passes `device="cpu"` explicitly so the CPU backend doesn't auto-grab the GPU now that CUDA is live.

## Milestone 5: fine-tuned wildlife model (plan)

COCO has no squirrel/raccoon/opossum/deer/fox classes, so stock YOLO mislabels most backyard wildlife. Plan: fine-tune `yolo11n.pt` on camera-trap datasets (ENA24-detection has boxes; NACTI boxed via MegaDetector) plus empty-yard background frames, validating only on this camera's real sightings (split by day/clip, never by frame). Constraints that affect code along the way:

- **M2 recorder must save clean (un-annotated) frames** for sightings — they become training/validation images.
- **Training now runs on-device** (decision 2026-06-14, revising the original off-device plan — it fits the "fully local" ethos). The `crittercam/training/` subpackage does it: `dataset.build_yolo_dataset` (COCO→YOLO, used today on the ENA24 starter pool) then `finetune.finetune`. Run `python -m crittercam.training.finetune` **with the tracker stopped** — the 8 GB is shared, and training OOMs (`NvMapMem error 12`) against the live capture pipeline. Smoke-validated on the Orin GPU (yolo11n, 1 epoch, ~0.23 GB at imgsz 320). It's slow (hours for a real run), so it's an occasional batch job. Training is a third concern that, like tracker/web, imports neither of them — only `config` for paths.
- A fine-tuned model is a drop-in: class names come from the model file (`CpuDetector` reads `model.names`), so deploying is copying `best.pt` to `<data_root>/models/` and setting `detector.model` — but `detector.classes` in config must be updated to match the new class list or startup validation fails. `finetune` prints the exact `detector.classes` list to paste.
- TensorRT engines (M4) must be exported on the Jetson itself; engine files are device-specific.
- The sightings DB doubles as the active-learning queue: low-confidence detections are the labeling targets for the next training round.

## Mock data (dashboard development)

A balanced ENA24-detection subset lives at `~/wildlife-camera-mock/source/ena24/` (outside the repo): `subset.json` is a 420-image manifest (20 per class, seed 42, Vehicle excluded, bboxes included) sampled from `ena24_public.json`, with the JPEGs in `images/`. Source: `https://storage.googleapis.com/public-datasets-lila/ena24/images/<file_name>` (CDLA-permissive). It feeds mock sightings for UI work against a separate data root (`~/wildlife-camera-mock/`) — never seed the real DB (single-writer discipline) — and doubles as a starter pool for milestone 5. Note: capture timestamps are burned into the image pixels, so they won't match synthesized sighting times.

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
2. **Web** (`crittercam/web/`) — FastAPI app (`create_app(cfg)` factory in `web/main.py`) that re-serves frames as MJPEG, serves the gallery API (`/api/sightings` list/thumb/play/clip/favorite/delete), and the static UI. Clip playback re-streams the stored JPEGs as paced multipart MJPEG (`web/clips.py` parses the AVI; browsers can't play MJPEG-AVI natively) — no re-encode, no ffmpeg. Timeline scrubbing is server-side seeking: `web/clips.py` scans a clip once into an lru-cached frame-offset index, so `/play?start=N` and `/frame/{n}` are O(1) in clip size (clips reach ~465 MB; the browser never downloads one to scrub it). Deleting is refused while a sighting is still `recording` (the tracker holds the file open).

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
