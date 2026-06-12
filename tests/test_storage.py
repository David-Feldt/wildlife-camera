import pytest

from crittercam import db
from crittercam.tracker import storage


@pytest.fixture
def conn(cfg):
    c = db.open_db(cfg.db_path)
    yield c
    c.close()


def make_sighting(cfg, conn, day: int, favorite: bool = False):
    """Insert a completed sighting on 2026-06-<day> with a clip file on disk."""
    sid = db.insert_sighting(conn, f"2026-06-{day:02d}T00:00:00+00:00", "cat", 0.9)
    rel = f"clips/{sid}.avi"
    path = cfg.data_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"avi")
    db.close_sighting(conn, sid, ended_at=f"2026-06-{day:02d}T00:01:00+00:00",
                      duration_s=60.0, dominant_class="cat", max_confidence=0.9,
                      track_count=1, clip_path=rel, thumb_path=f"thumbs/{sid}.jpg",
                      status="complete")
    if favorite:
        with conn:
            conn.execute("UPDATE sightings SET favorite=1 WHERE id=?", (sid,))
    return sid, path


def fake_disk(cfg, monkeypatch):
    """Each existing clip file 'uses' 5% of the disk on top of a 70% base, so
    with the default 0.85/0.75 watermarks: 4 clips = 90% (over high), and
    pruning must delete 3 to reach 75% (low)."""
    clips = cfg.data_root / "clips"
    monkeypatch.setattr(
        storage, "disk_used_fraction",
        lambda _cfg: 0.70 + 0.05 * len(list(clips.glob("*.avi"))))


def test_noop_below_high_watermark(cfg, conn, monkeypatch):
    fake_disk(cfg, monkeypatch)
    paths = [make_sighting(cfg, conn, day)[1] for day in (1, 2)]  # 80% < high

    assert storage.prune_clips(conn, cfg) == 0
    assert all(p.exists() for p in paths)


def test_prunes_oldest_until_low_watermark(cfg, conn, monkeypatch):
    fake_disk(cfg, monkeypatch)
    sightings = [make_sighting(cfg, conn, day) for day in (1, 2, 3, 4)]  # 90%

    assert storage.prune_clips(conn, cfg) == 3

    for sid, path in sightings[:3]:
        assert not path.exists()
        row = conn.execute("SELECT * FROM sightings WHERE id=?", (sid,)).fetchone()
        assert row["status"] == "clip_missing"
        assert row["clip_path"] is None
        assert row["thumb_path"] is not None  # the log of the visit survives
    newest_id, newest_path = sightings[3]
    assert newest_path.exists()
    row = conn.execute("SELECT status FROM sightings WHERE id=?", (newest_id,)).fetchone()
    assert row["status"] == "complete"


def test_favorites_are_never_pruned(cfg, conn, monkeypatch):
    fake_disk(cfg, monkeypatch)
    fav_id, fav_path = make_sighting(cfg, conn, 1, favorite=True)  # oldest
    others = [make_sighting(cfg, conn, day) for day in (2, 3, 4)]

    storage.prune_clips(conn, cfg)

    assert fav_path.exists()
    row = conn.execute("SELECT status FROM sightings WHERE id=?", (fav_id,)).fetchone()
    assert row["status"] == "complete"
    assert all(not p.exists() for _, p in others)
