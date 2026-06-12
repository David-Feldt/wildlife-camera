"""Web entrypoint: FastAPI app serving the live stream, sighting gallery,
and static UI.

Run: python -m crittercam.web.main
"""
from __future__ import annotations

import logging
import shutil
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from crittercam import db
from crittercam.config import Config, load_config, setup_logging
from crittercam.web import clips
from crittercam.web.live import MJPEG_MEDIA_TYPE, mjpeg_stream

STATIC_DIR = Path(__file__).parent / "static"


class FavoriteBody(BaseModel):
    favorite: bool


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="crittercam")

    def fetch_sighting(conn: sqlite3.Connection, sighting_id: int) -> sqlite3.Row:
        row = db.get_sighting(conn, sighting_id)
        if row is None:
            raise HTTPException(404, "no such sighting")
        return row

    def load_sighting(sighting_id: int) -> sqlite3.Row:
        conn = db.open_db(cfg.db_path)
        try:
            return fetch_sighting(conn, sighting_id)
        finally:
            conn.close()

    def data_file(relpath: str) -> Path:
        path = (cfg.data_root / relpath).resolve()
        if not path.is_relative_to(cfg.data_root.resolve()):
            raise HTTPException(404, "file missing")
        if not path.is_file():
            raise HTTPException(404, "file missing")
        return path

    def playable_clip(row: sqlite3.Row) -> Path:
        if row["status"] == "recording":
            raise HTTPException(409, "sighting still recording")
        if not row["clip_path"]:
            raise HTTPException(404, "clip was pruned")
        return data_file(row["clip_path"])

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

    @app.get("/api/sightings/{sighting_id}/thumb")
    def thumb(sighting_id: int):
        row = load_sighting(sighting_id)
        if not row["thumb_path"]:
            raise HTTPException(404, "no thumbnail")
        return FileResponse(data_file(row["thumb_path"]), media_type="image/jpeg")

    @app.get("/api/sightings/{sighting_id}/play")
    def play(sighting_id: int, start: int = 0):
        path = playable_clip(load_sighting(sighting_id))
        try:
            fps, frames = clips.open_clip(path, start)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return StreamingResponse(
            clips.clip_mjpeg_stream(fps, frames), media_type=MJPEG_MEDIA_TYPE
        )

    @app.get("/api/sightings/{sighting_id}/clipinfo")
    def clipinfo(sighting_id: int):
        path = playable_clip(load_sighting(sighting_id))
        try:
            fps, frame_count = clips.clip_info(path)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {"fps": fps, "frames": frame_count}

    @app.get("/api/sightings/{sighting_id}/frame/{n}")
    def frame(sighting_id: int, n: int):
        path = playable_clip(load_sighting(sighting_id))
        try:
            jpeg = clips.read_frame(path, n)
        except IndexError:
            raise HTTPException(404, "no such frame")
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return Response(jpeg, media_type="image/jpeg")

    @app.get("/api/sightings/{sighting_id}/clip")
    def clip(sighting_id: int):
        path = playable_clip(load_sighting(sighting_id))
        return FileResponse(path, media_type="video/x-msvideo", filename=path.name)

    @app.post("/api/sightings/{sighting_id}/favorite")
    def favorite(sighting_id: int, body: FavoriteBody):
        conn = db.open_db(cfg.db_path)
        try:
            fetch_sighting(conn, sighting_id)
            db.set_favorite(conn, sighting_id, body.favorite)
        finally:
            conn.close()
        return {"id": sighting_id, "favorite": body.favorite}

    @app.delete("/api/sightings/{sighting_id}")
    def delete(sighting_id: int):
        conn = db.open_db(cfg.db_path)
        try:
            row = fetch_sighting(conn, sighting_id)
            if row["status"] == "recording":
                raise HTTPException(409, "sighting still recording")
            db.delete_sighting(conn, sighting_id)
        finally:
            conn.close()
        for relpath in (row["clip_path"], row["thumb_path"]):
            if relpath:
                (cfg.data_root / relpath).unlink(missing_ok=True)
        return {"deleted": sighting_id}

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
    return app


def _open_browser_when_ready(port: int) -> None:
    """Open the UI on the local display once the server accepts connections.

    Runs in a daemon thread: a missing browser or display logs a warning but
    never blocks the server (e.g. headless/SSH runs).
    """
    import os
    import socket
    import time
    import webbrowser

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            time.sleep(0.2)
    # When started over SSH/tmux there is no DISPLAY; the attached screen is :0.
    os.environ.setdefault("DISPLAY", ":0")
    if not webbrowser.open(f"http://127.0.0.1:{port}/"):
        logging.getLogger(__name__).warning(
            "web.open_browser is set but no browser could be opened"
        )


def main() -> None:
    import threading

    import uvicorn

    cfg = load_config()
    setup_logging(cfg.log_level)
    if cfg.web.open_browser:
        threading.Thread(
            target=_open_browser_when_ready, args=(cfg.web.port,), daemon=True
        ).start()
    uvicorn.run(create_app(cfg), host=cfg.web.host, port=cfg.web.port, log_level="warning")


if __name__ == "__main__":
    main()
