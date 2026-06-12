"""End-to-end: file camera -> scripted detector -> tracker -> sighting + clip.

The detector is injected through the Detector protocol so the test is
deterministic and needs no YOLO model.
"""
import threading
import time
from pathlib import Path

import cv2

from crittercam import db
from crittercam.config import CameraConfig, Config, DetectorConfig, EventsConfig
from crittercam.models import Detection
from crittercam.tracker.main import run

FIXTURE = Path(__file__).parent / "fixtures" / "sample.mp4"


class ScriptedDetector:
    """A cat drifts through frames 15-120 (0.5s-4s of the 30fps fixture)."""

    def detect(self, frame):
        if 15 <= frame.index <= 120:
            x = 100.0 + frame.index
            return [Detection("cat", 0.9, (x, 100.0, x + 200.0, 300.0))]
        return []


def test_pipeline_end_to_end(tmp_path):
    cfg = Config(
        data_root=tmp_path,
        camera=CameraConfig(kind="file", device=str(FIXTURE)),
        detector=DetectorConfig(infer_every_n=2),
        events=EventsConfig(min_track_frames=3, linger_seconds=1.0,
                            preroll_seconds=1.0, max_clip_seconds=30.0),
        zmq_frame_endpoint="tcp://127.0.0.1:5601",
    )
    # Create and migrate the DB up front so the tracker thread and the test
    # are not both running first-time migrations on the same fresh file.
    conn = db.open_db(cfg.db_path)

    stop = threading.Event()
    thread = threading.Thread(target=run, args=(cfg, stop),
                              kwargs={"detector": ScriptedDetector()}, daemon=True)
    thread.start()
    row = None
    deadline = time.monotonic() + 15
    try:
        while time.monotonic() < deadline:
            row = conn.execute("SELECT * FROM sightings WHERE status='complete'").fetchone()
            if row:
                break
            time.sleep(0.5)
    finally:
        stop.set()
        thread.join(timeout=10)
        conn.close()

    assert row is not None, "no completed sighting within deadline"
    assert row["dominant_class"] == "cat"
    assert row["duration_s"] > 1.0
    assert row["track_count"] >= 1

    clip = cfg.data_root / row["clip_path"]
    assert clip.exists()
    cap = cv2.VideoCapture(str(clip))
    assert cap.isOpened()
    frames = 0
    while cap.read()[0]:
        frames += 1
    # ~1s preroll + ~3.5s of sighting + 1s linger at ~30fps
    assert frames > 60
    assert (cfg.data_root / row["thumb_path"]).exists()
