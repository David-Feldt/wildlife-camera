import cv2

from crittercam.tracker.recorder import ClipRecorder
from util import make_jpeg


def test_preroll_trim_clip_write_and_readback(tmp_path):
    rec = ClipRecorder(preroll_seconds=2.0)
    jpeg = make_jpeg()
    # 10s of frames at 10 fps before the trigger: only ~2s should survive.
    for i in range(100):
        rec.add_frame(i * 0.1, jpeg)

    path = tmp_path / "clips" / "a.avi"
    rec.start(path)
    assert rec.recording
    for i in range(100, 120):
        rec.add_frame(i * 0.1, jpeg)
    clip = rec.stop()

    assert not rec.recording
    assert 35 <= clip.frame_count <= 45  # ~21 preroll + 20 recorded
    assert clip.duration_s > 3.0

    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    frames = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        assert image.shape[:2] == (120, 160)
        frames += 1
    assert frames == clip.frame_count
    fps = cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FPS)
    assert 8.0 < fps < 12.0


def test_stop_without_frames_returns_none(tmp_path):
    rec = ClipRecorder(preroll_seconds=1.0)
    rec.start(tmp_path / "clips" / "empty.avi")
    assert rec.stop() is None
    assert not (tmp_path / "clips" / "empty.avi").exists()


def test_empty_jpeg_ignored(tmp_path):
    rec = ClipRecorder(preroll_seconds=1.0)
    rec.add_frame(0.0, b"")
    rec.start(tmp_path / "clips" / "b.avi")
    rec.add_frame(0.1, make_jpeg())
    clip = rec.stop()
    assert clip.frame_count == 1
