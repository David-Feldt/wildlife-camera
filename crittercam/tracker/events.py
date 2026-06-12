"""Sighting lifecycle: a track that persists for min_track_frames opens a
sighting (the recorder starts with its preroll), and the sighting closes
after linger_seconds with no tracks, or splits at max_clip_seconds.

All writes to the sightings tables happen here, in the tracker process
(single-writer discipline; see db.py).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from sqlite3 import Connection

from crittercam import db
from crittercam.config import Config
from crittercam.models import Frame, Track
from crittercam.tracker.recorder import ClipRecorder
from crittercam.tracker.storage import prune_clips

log = logging.getLogger(__name__)

# detections_sample is a sketch of the sighting, not a full trace: at CPU
# inference rates a full trace would be ~900 rows per 5-minute sighting.
SAMPLE_INTERVAL_S = 2.0


def _iso_utc(ts_wall: float) -> str:
    return datetime.fromtimestamp(ts_wall, tz=timezone.utc).isoformat(timespec="seconds")


class EventManager:
    """Consumes one tracker update per inference round and owns the sighting
    rows plus the clip/thumbnail files derived from them."""

    def __init__(self, cfg: Config, conn: Connection, recorder: ClipRecorder):
        self.cfg = cfg
        self.conn = conn
        self.recorder = recorder
        self.sighting_id: int | None = None
        db.recover_stale_recordings(conn, cfg.data_root)

    @property
    def recording(self) -> bool:
        return self.sighting_id is not None

    def update(self, tracks: list[Track], frame: Frame, jpeg: bytes) -> None:
        if self.sighting_id is None:
            if any(t.age_frames >= self.cfg.events.min_track_frames for t in tracks):
                self._open(tracks, frame, jpeg)
            return
        if tracks:
            self._observe(tracks, frame, jpeg)
        if frame.ts_monotonic - self._last_seen_m >= self.cfg.events.linger_seconds:
            self._close()
        elif frame.ts_monotonic - self._opened_m >= self.cfg.events.max_clip_seconds:
            log.info("sighting %d hit max_clip_seconds, splitting", self.sighting_id)
            self._close()

    def close_if_open(self) -> None:
        """Shutdown path: finalize the clip instead of orphaning a 'recording' row."""
        if self.sighting_id is not None:
            self._close()

    def _open(self, tracks: list[Track], frame: Frame, jpeg: bytes) -> None:
        now_w, now_m = frame.ts_wall, frame.ts_monotonic
        best = max(tracks, key=lambda t: t.confidence)
        self._started_w = now_w
        self._opened_m = now_m
        self._last_seen_m = now_m
        self._last_seen_w = now_w
        self._last_sample_m = now_m - SAMPLE_INTERVAL_S  # sample immediately
        self._class_scores: dict[str, float] = {}
        self._track_ids: set[int] = set()
        self._max_conf = 0.0
        self._best_thumb: tuple[float, bytes] = (0.0, b"")

        self.sighting_id = db.insert_sighting(
            self.conn, _iso_utc(now_w), best.class_name, best.confidence)
        stamp = datetime.fromtimestamp(now_w, tz=timezone.utc).strftime("%Y%m%d/%H%M%S")
        self._clip_relpath = f"clips/{stamp}_{self.sighting_id}.avi"
        # Recorded in the row up front so a crash mid-recording can be cleaned
        # up on the next start (see db.recover_stale_recordings).
        db.set_sighting_clip(self.conn, self.sighting_id, self._clip_relpath)
        self.recorder.start(self.cfg.data_root / self._clip_relpath)
        self._observe(tracks, frame, jpeg)
        log.info("sighting %d opened class=%s conf=%.2f",
                 self.sighting_id, best.class_name, best.confidence)

    def _observe(self, tracks: list[Track], frame: Frame, jpeg: bytes) -> None:
        now_m, now_w = frame.ts_monotonic, frame.ts_wall
        self._last_seen_m, self._last_seen_w = now_m, now_w
        for t in tracks:
            self._class_scores[t.class_name] = self._class_scores.get(t.class_name, 0.0) + t.confidence
            self._track_ids.add(t.track_id)
            self._max_conf = max(self._max_conf, t.confidence)
        top = max(t.confidence for t in tracks)
        if jpeg and top > self._best_thumb[0]:
            self._best_thumb = (top, jpeg)
        if now_m - self._last_sample_m >= SAMPLE_INTERVAL_S:
            self._last_sample_m = now_m
            h, w = frame.image.shape[:2]
            ts = _iso_utc(now_w)
            for t in tracks:
                x1, y1, x2, y2 = t.bbox
                bbox = json.dumps([round(x1 / w, 4), round(y1 / h, 4),
                                   round((x2 - x1) / w, 4), round((y2 - y1) / h, 4)])
                db.insert_detection_sample(
                    self.conn, self.sighting_id, ts, t.class_name, t.confidence, bbox)

    def _close(self) -> None:
        clip = self.recorder.stop()
        thumb_relpath = None
        if self._best_thumb[1]:
            thumb_relpath = f"thumbs/{self.sighting_id}.jpg"
            thumb_path = self.cfg.data_root / thumb_relpath
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            thumb_path.write_bytes(self._best_thumb[1])
        dominant = max(self._class_scores, key=self._class_scores.get)
        duration = max(self._last_seen_w - self._started_w, 0.0)
        db.close_sighting(
            self.conn, self.sighting_id,
            ended_at=_iso_utc(self._last_seen_w),
            duration_s=round(duration, 2),
            dominant_class=dominant,
            max_confidence=round(self._max_conf, 4),
            track_count=len(self._track_ids),
            clip_path=self._clip_relpath if clip else None,
            thumb_path=thumb_relpath,
            status="complete" if clip else "clip_missing",
        )
        log.info("sighting %d closed class=%s duration=%.1fs frames=%d",
                 self.sighting_id, dominant, duration, clip.frame_count if clip else 0)
        self.sighting_id = None
        prune_clips(self.conn, self.cfg)
