# Crittercam

Fully local wildlife detection, recording, and logging on an NVIDIA Jetson Orin Nano. A USB camera watches the yard, YOLO spots the critters, and a small web UI shows a live annotated stream — no cloud, no subscriptions, everything stays on the device.

**Status: milestone 3** — live detection boxes, IoU tracking, sighting events logged to SQLite, clip recording with preroll, and a gallery UI: recent sightings with thumbnails, click-to-play clip playback in the browser, favorites (exempt from disk pruning), and delete. TensorRT inference comes in milestone 4.

## How it works

Two independent processes that share only a ZMQ socket and a SQLite database:

```
┌──────────────────────────────────────┐         ┌──────────────────────────┐
│ Tracker                              │         │ Web (FastAPI, port 8080) │
│ capture → detect → track → record ───┼─ ZMQ ──▶│ MJPEG stream + /api      │
│            │              → publish  │ PUB/SUB │            │             │
└────────────┼─────────────────────────┘         └────────────┼─────────────┘
             ▼                                                ▼
            SQLite (critters.db, WAL) ◀───────────────────────┘
```

- **Tracker** (`crittercam/tracker/`) — a capture thread feeds a small drop-oldest queue so the pipeline never falls behind real time. YOLO runs every Nth frame (CPU inference is ~1–3 FPS); annotated JPEGs are published every frame over ZMQ. On inference frames a greedy IoU tracker associates detections to tracks (class-agnostic, because COCO labels flicker on unfamiliar animals), and a track that persists long enough opens a **sighting**: a row in SQLite plus an MJPEG-AVI clip that starts with a preroll buffer, so the clip includes the seconds *before* the animal was first confirmed. Sightings close after a quiet period and clips are muxed from the already-encoded JPEGs — no re-encode. Recorded frames are kept clean (no boxes burned in); they double as training data for a future fine-tuned model. When the disk crosses a high watermark, the oldest non-favorite clips are pruned until usage falls below a low watermark — the sighting rows survive, so the log of what visited remains intact.
- **Web** (`crittercam/web/`) — re-serves frames as MJPEG with a per-client conflating subscriber, so slow browsers only ever see the newest frame. The same page shows a gallery of recent sightings; clicking one plays its clip with a scrubbable timeline. Browsers can't play MJPEG-AVI in a `<video>` tag, so clips are re-served the same way as the live view — a paced multipart MJPEG stream of the stored JPEGs, no re-encode and no ffmpeg — and seeking is server-side: each clip is scanned once into a cached frame index, so jumping anywhere in even a multi-hundred-MB clip is a few milliseconds. The API covers `/api/status`, `/api/sightings`, and per-sighting thumb/play/clipinfo/frame/clip-download/favorite/delete.

Camera and detector backends live behind `CameraSource` and `Detector` protocols, so CSI/RTSP cameras and TensorRT slot in later without touching the pipeline.

## Hardware

- NVIDIA Jetson Orin Nano (JetPack)
- USB UVC camera (developed against an Arducam; MJPG fourcc required for full frame rate over USB2)

## Setup

Requires Python 3.10 and the system OpenCV from JetPack (the GStreamer-enabled build):

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

> **Note:** never `pip install opencv-python` here — it would shadow the JetPack OpenCV build. The venv uses `--system-site-packages` specifically to pick it up, and `numpy<2` is pinned for JetPack compatibility.

## Running

```bash
python -m crittercam.tracker.main   # detection pipeline
python -m crittercam.web.main       # web UI on http://<jetson>:8080
```

The YOLO model is auto-downloaded to `<data_root>/models/` on first run.

## Configuration

Defaults ship in [`config/default.yaml`](config/default.yaml). Put overrides in `~/wildlife-camera-data/config.yaml` — they're deep-merged over the defaults at startup and validated. Useful knobs:

| Key | Default | What it does |
| --- | --- | --- |
| `camera.kind` | `usb` | `usb` or `file` (video file playback for development) |
| `camera.device` | `/dev/video0` | V4L2 device, or a video file path when `kind: file` |
| `detector.model` | `yolo11n.pt` | YOLO weights, resolved under `<data_root>/models/` |
| `detector.confidence` | `0.45` | Detection threshold |
| `detector.infer_every_n` | `5` | Run inference every Nth frame, reuse boxes in between |
| `events.min_track_frames` | `5` | Inference rounds a track must persist before a sighting opens |
| `events.linger_seconds` | `5` | Quiet time with no tracks before a sighting closes |
| `events.preroll_seconds` | `10` | Footage kept from before the sighting opened |
| `events.max_clip_seconds` | `300` | Split long sightings into clips of at most this length |
| `storage.disk_high_watermark` | `0.85` | Disk usage that triggers pruning of oldest non-favorite clips |
| `web.port` | `8080` | Web UI port |

Clips land under `<data_root>/clips/` and sightings are queryable at `/api/sightings`.

## Developing without a camera

Point the config at a video file:

```yaml
# ~/wildlife-camera-data/config.yaml
camera:
  kind: file
  device: tests/fixtures/sample.mp4
```

Run tests with:

```bash
pytest
```

## Roadmap

- [x] **M1** — live detection boxes, MJPEG stream, heartbeat
- [x] **M2** — tracking, sighting events, clip recording with preroll, disk-watermark pruning
- [ ] **M3** — gallery UI, favorites, retention/cleanup
- [ ] **M4** — TensorRT backend for real-time inference
- [ ] **M5** — fine-tuned backyard wildlife model (stock COCO has no squirrel/raccoon/deer classes)
