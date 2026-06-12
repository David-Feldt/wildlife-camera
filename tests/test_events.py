import pytest

from crittercam import db
from crittercam.models import Frame, Track
from crittercam.tracker.events import EventManager
from crittercam.tracker.recorder import ClipRecorder
from util import make_image, make_jpeg


@pytest.fixture
def conn(cfg):
    c = db.open_db(cfg.db_path)
    yield c
    c.close()


def frame(t: float) -> Frame:
    return Frame(image=make_image(), ts_monotonic=t,
                 ts_wall=1_700_000_000.0 + t, index=int(t * 10))


def track(tid=1, age=5, name="cat", conf=0.8, bbox=(10.0, 10.0, 60.0, 60.0)):
    return Track(track_id=tid, class_name=name, confidence=conf,
                 bbox=bbox, age_frames=age)


def feed(mgr, rec, t, tracks):
    jpeg = make_jpeg()
    rec.add_frame(t, jpeg)
    mgr.update(tracks, frame(t), jpeg)


def test_opens_after_min_track_frames_then_closes_after_linger(cfg, conn):
    rec = ClipRecorder(cfg.events.preroll_seconds)
    mgr = EventManager(cfg, conn, rec)

    feed(mgr, rec, 0.0, [track(age=1)])
    feed(mgr, rec, 0.5, [track(age=2)])
    assert not mgr.recording
    feed(mgr, rec, 1.0, [track(age=3)])  # min_track_frames=3
    assert mgr.recording
    assert conn.execute("SELECT status FROM sightings").fetchone()["status"] == "recording"

    feed(mgr, rec, 1.5, [])  # linger_seconds=1.0 not yet elapsed
    assert mgr.recording
    feed(mgr, rec, 2.1, [])
    assert not mgr.recording

    row = conn.execute("SELECT * FROM sightings").fetchone()
    assert row["status"] == "complete"
    assert row["dominant_class"] == "cat"
    assert row["max_confidence"] == pytest.approx(0.8)
    assert row["track_count"] == 1
    assert row["ended_at"] and row["duration_s"] >= 0
    assert (cfg.data_root / row["clip_path"]).exists()
    assert (cfg.data_root / row["thumb_path"]).exists()
    samples = conn.execute("SELECT COUNT(*) AS c FROM detections_sample").fetchone()["c"]
    assert samples >= 1


def test_max_clip_seconds_splits_long_sighting(cfg, conn):
    rec = ClipRecorder(cfg.events.preroll_seconds)
    mgr = EventManager(cfg, conn, rec)
    t = 0.0
    feed(mgr, rec, t, [track(age=3)])
    assert mgr.recording
    while t < cfg.events.max_clip_seconds + 1.0:
        t += 0.5
        feed(mgr, rec, t, [track(age=4)])
    first = conn.execute("SELECT status FROM sightings ORDER BY id LIMIT 1").fetchone()
    assert first["status"] == "complete"


def test_dominant_class_is_confidence_weighted(cfg, conn):
    rec = ClipRecorder(cfg.events.preroll_seconds)
    mgr = EventManager(cfg, conn, rec)
    feed(mgr, rec, 0.0, [track(age=3, name="dog", conf=0.5)])
    feed(mgr, rec, 0.5, [track(age=4, name="squirrel", conf=0.9)])
    feed(mgr, rec, 1.0, [track(age=5, name="squirrel", conf=0.9)])
    feed(mgr, rec, 3.0, [])  # past linger -> close
    row = conn.execute("SELECT dominant_class FROM sightings").fetchone()
    assert row["dominant_class"] == "squirrel"


def test_recovers_stale_recording_rows(cfg, conn):
    sid = db.insert_sighting(conn, "2026-06-11T00:00:00+00:00", "cat", 0.9)
    db.set_sighting_clip(conn, sid, "clips/x.avi")
    (cfg.data_root / "clips").mkdir(parents=True)
    (cfg.data_root / "clips" / "x.avi").write_bytes(b"partial garbage")

    EventManager(cfg, conn, ClipRecorder(1.0))

    row = conn.execute("SELECT status, clip_path FROM sightings").fetchone()
    assert row["status"] == "clip_missing"
    assert row["clip_path"] is None
    assert not (cfg.data_root / "clips" / "x.avi").exists()
