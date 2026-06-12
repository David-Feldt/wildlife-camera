"""Web entrypoint: FastAPI app serving the live stream, status, and static UI.

Run: python -m crittercam.web.main
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from crittercam import db
from crittercam.config import load_config, setup_logging
from crittercam.web.live import MJPEG_MEDIA_TYPE, mjpeg_stream

STATIC_DIR = Path(__file__).parent / "static"

cfg = load_config()
app = FastAPI(title="crittercam")


@app.get("/live.mjpg")
def live():
    return StreamingResponse(
        mjpeg_stream(cfg.zmq_frame_endpoint, cfg.web.live_fps),
        media_type=MJPEG_MEDIA_TYPE,
    )


@app.get("/api/sightings")
def sightings(limit: int = 50):
    conn = db.open_db(cfg.db_path)
    try:
        rows = db.recent_sightings(conn, limit)
    finally:
        conn.close()
    return [dict(row) for row in rows]


@app.get("/api/status")
def status():
    conn = db.open_db(cfg.db_path)
    try:
        age = db.heartbeat_age(conn)
        infer_fps = db.kv_get(conn, "tracker_infer_fps")
    finally:
        conn.close()
    usage = shutil.disk_usage(cfg.data_root)
    return {
        "tracker_alive": age is not None and age < 10,
        "heartbeat_age_s": age,
        "infer_fps": float(infer_fps) if infer_fps else None,
        "disk_used_fraction": round(usage.used / usage.total, 3),
        "camera": cfg.camera.kind,
        "detector": cfg.detector.backend,
        "model": cfg.detector.model,
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main() -> None:
    import uvicorn

    setup_logging(cfg.log_level)
    uvicorn.run(app, host=cfg.web.host, port=cfg.web.port, log_level="warning")


if __name__ == "__main__":
    main()
