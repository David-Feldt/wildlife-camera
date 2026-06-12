"""Gallery API tests: a seeded data root served through the FastAPI app.

The seeded clip is written with the tracker's MjpegAviWriter so playback is
exercised against the real on-disk format, not a fixture blob.
"""
import pytest
from fastapi.testclient import TestClient

from crittercam import db
from crittercam.tracker.recorder import MjpegAviWriter
from crittercam.web.clips import open_clip
from crittercam.web.main import create_app
from util import make_jpeg

JPEGS = [make_jpeg(seed=i) for i in range(3)]


@pytest.fixture
def seeded(cfg):
    """Three sightings: complete (clip+thumb), pruned, and mid-recording."""
    conn = db.open_db(cfg.db_path)

    complete = db.insert_sighting(conn, "2026-06-12T01:00:00+00:00", "cat", 0.8)
    clip_rel = f"clips/test_{complete}.avi"
    clip_path = cfg.data_root / clip_rel
    clip_path.parent.mkdir(parents=True)
    writer = MjpegAviWriter(clip_path, 160, 120)
    for jpeg in JPEGS:
        writer.add(jpeg)
    writer.close(fps=10.0)
    thumb_rel = f"thumbs/{complete}.jpg"
    (cfg.data_root / "thumbs").mkdir()
    (cfg.data_root / thumb_rel).write_bytes(JPEGS[0])
    db.close_sighting(conn, complete, ended_at="2026-06-12T01:00:02+00:00",
                      duration_s=2.0, dominant_class="cat", max_confidence=0.8,
                      track_count=1, clip_path=clip_rel, thumb_path=thumb_rel,
                      status="complete")

    pruned = db.insert_sighting(conn, "2026-06-12T02:00:00+00:00", "dog", 0.7)
    db.close_sighting(conn, pruned, ended_at="2026-06-12T02:00:05+00:00",
                      duration_s=5.0, dominant_class="dog", max_confidence=0.7,
                      track_count=1, clip_path=None, thumb_path=None,
                      status="clip_missing")

    recording = db.insert_sighting(conn, "2026-06-12T03:00:00+00:00", "bird", 0.6)
    conn.close()
    return {"complete": complete, "pruned": pruned, "recording": recording}


@pytest.fixture
def client(cfg, seeded):
    return TestClient(create_app(cfg))


def test_open_clip_roundtrip(cfg, seeded):
    path = cfg.data_root / f"clips/test_{seeded['complete']}.avi"
    fps, frames = open_clip(path)
    assert fps == pytest.approx(10.0)
    assert list(frames) == JPEGS
    _, frames = open_clip(path, start=2)
    assert list(frames) == JPEGS[2:]
    _, frames = open_clip(path, start=99)
    assert list(frames) == []


def test_sightings_newest_first(client, seeded):
    rows = client.get("/api/sightings").json()
    assert [r["id"] for r in rows] == [
        seeded["recording"], seeded["pruned"], seeded["complete"]]


def test_thumb(client, seeded):
    r = client.get(f"/api/sightings/{seeded['complete']}/thumb")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == JPEGS[0]
    assert client.get(f"/api/sightings/{seeded['pruned']}/thumb").status_code == 404
    assert client.get("/api/sightings/9999/thumb").status_code == 404


def test_play_streams_every_frame(client, seeded):
    r = client.get(f"/api/sightings/{seeded['complete']}/play")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
    for jpeg in JPEGS:
        assert jpeg in r.content
    assert r.content.count(b"--crittercam-frame") == len(JPEGS)


def test_play_with_start_skips_frames(client, seeded):
    r = client.get(f"/api/sightings/{seeded['complete']}/play?start=1")
    assert r.status_code == 200
    assert JPEGS[0] not in r.content
    assert JPEGS[1] in r.content and JPEGS[2] in r.content
    assert r.content.count(b"--crittercam-frame") == 2


def test_clipinfo(client, seeded):
    r = client.get(f"/api/sightings/{seeded['complete']}/clipinfo")
    assert r.status_code == 200
    assert r.json() == {"fps": pytest.approx(10.0), "frames": len(JPEGS)}
    assert client.get(f"/api/sightings/{seeded['pruned']}/clipinfo").status_code == 404
    assert client.get(f"/api/sightings/{seeded['recording']}/clipinfo").status_code == 409


def test_frame_by_index(client, seeded):
    sid = seeded["complete"]
    for n, jpeg in enumerate(JPEGS):
        r = client.get(f"/api/sightings/{sid}/frame/{n}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert r.content == jpeg
    assert client.get(f"/api/sightings/{sid}/frame/{len(JPEGS)}").status_code == 404
    assert client.get(f"/api/sightings/{sid}/frame/-1").status_code == 404


def test_play_unavailable(client, seeded):
    assert client.get(f"/api/sightings/{seeded['pruned']}/play").status_code == 404
    assert client.get(f"/api/sightings/{seeded['recording']}/play").status_code == 409
    assert client.get("/api/sightings/9999/play").status_code == 404


def test_clip_download(client, cfg, seeded):
    r = client.get(f"/api/sightings/{seeded['complete']}/clip")
    assert r.status_code == 200
    assert r.content == (cfg.data_root / f"clips/test_{seeded['complete']}.avi").read_bytes()


def test_favorite_roundtrip(client, seeded):
    sid = seeded["complete"]
    assert client.post(f"/api/sightings/{sid}/favorite",
                       json={"favorite": True}).status_code == 200
    rows = {r["id"]: r for r in client.get("/api/sightings").json()}
    assert rows[sid]["favorite"] == 1
    client.post(f"/api/sightings/{sid}/favorite", json={"favorite": False})
    rows = {r["id"]: r for r in client.get("/api/sightings").json()}
    assert rows[sid]["favorite"] == 0
    assert client.post("/api/sightings/9999/favorite",
                       json={"favorite": True}).status_code == 404


def test_delete_removes_row_and_files(client, cfg, seeded):
    sid = seeded["complete"]
    assert client.delete(f"/api/sightings/{sid}").status_code == 200
    assert all(r["id"] != sid for r in client.get("/api/sightings").json())
    assert not (cfg.data_root / f"clips/test_{sid}.avi").exists()
    assert not (cfg.data_root / f"thumbs/{sid}.jpg").exists()


def test_delete_refuses_while_recording(client, seeded):
    assert client.delete(f"/api/sightings/{seeded['recording']}").status_code == 409
    ids = [r["id"] for r in client.get("/api/sightings").json()]
    assert seeded["recording"] in ids
