"""Tracker entrypoint: wires capture -> detect -> publish.

Milestone 1 scope: live detection boxes on a ZMQ stream, heartbeat in the DB.
Tracking, events, and recording arrive in milestone 2.

Run: python -m crittercam.tracker.main
"""
from __future__ import annotations

import logging
import queue
import signal
import threading
import time

from crittercam import db
from crittercam.config import Config, load_config, setup_logging
from crittercam.models import Frame, FrameResult
from crittercam.tracker.capture import open_camera
from crittercam.tracker.detector import open_detector
from crittercam.tracker.publisher import FramePublisher

log = logging.getLogger("tracker")

HEARTBEAT_INTERVAL_S = 3.0
STATS_INTERVAL_S = 60.0


def put_latest(q: queue.Queue, item) -> bool:
    """Put without blocking; on a full queue, drop the oldest entry so the
    pipeline never falls behind real time. Returns True if a frame was dropped."""
    dropped = False
    while True:
        try:
            q.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                q.get_nowait()
                dropped = True
            except queue.Empty:
                pass


class Stats:
    def __init__(self):
        self.captured = 0
        self.inferred = 0
        self.dropped = 0
        self.lock = threading.Lock()

    def snapshot_and_reset(self) -> tuple[int, int, int]:
        with self.lock:
            out = (self.captured, self.inferred, self.dropped)
            self.captured = self.inferred = self.dropped = 0
        return out


def capture_loop(cfg: Config, frame_q: queue.Queue, stats: Stats, stop: threading.Event) -> None:
    camera = open_camera(cfg.camera)
    for frame in camera.frames():
        if stop.is_set():
            return
        with stats.lock:
            stats.captured += 1
        if put_latest(frame_q, frame):
            with stats.lock:
                stats.dropped += 1


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.log_level)
    conn = db.open_db(cfg.db_path)
    detector = open_detector(cfg.detector, cfg.model_path)
    publisher = FramePublisher(cfg.zmq_frame_endpoint)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    stats = Stats()
    frame_q: queue.Queue[Frame] = queue.Queue(maxsize=2)
    capture_thread = threading.Thread(
        target=capture_loop, args=(cfg, frame_q, stats, stop), daemon=True, name="capture"
    )
    capture_thread.start()

    last_detections = []
    last_heartbeat = 0.0
    last_stats = time.monotonic()
    infer_fps = 0.0
    frames_seen = 0

    log.info("tracker started camera=%s detector=%s", cfg.camera.kind, cfg.detector.backend)
    while not stop.is_set():
        try:
            frame = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue

        inferred = frames_seen % cfg.detector.infer_every_n == 0
        if inferred:
            t0 = time.monotonic()
            last_detections = detector.detect(frame)
            infer_fps = 1.0 / max(time.monotonic() - t0, 1e-6)
            with stats.lock:
                stats.inferred += 1
        frames_seen += 1

        result = FrameResult(frame=frame, detections=last_detections, inferred=inferred)
        status = f"infer {infer_fps:.1f}/s  det {len(last_detections)}"
        publisher.publish(result, status_line=status)

        now = time.monotonic()
        if now - last_heartbeat > HEARTBEAT_INTERVAL_S:
            db.heartbeat(conn)
            db.kv_set(conn, "tracker_infer_fps", f"{infer_fps:.2f}")
            last_heartbeat = now
        if now - last_stats > STATS_INTERVAL_S:
            cap, inf, drop = stats.snapshot_and_reset()
            elapsed = now - last_stats
            log.info("stats capture_fps=%.1f infer_fps=%.1f queue_drops=%d",
                     cap / elapsed, inf / elapsed, drop)
            last_stats = now

    log.info("shutting down")
    publisher.close()
    conn.close()


if __name__ == "__main__":
    main()
