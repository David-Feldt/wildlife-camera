from crittercam.models import Detection
from crittercam.tracker.tracking import IouTracker, iou

BOX = (100.0, 100.0, 200.0, 200.0)


def det(bbox=BOX, name="cat", conf=0.8):
    return Detection(class_name=name, confidence=conf, bbox=bbox)


def test_iou_identical():
    assert iou(BOX, BOX) == 1.0


def test_iou_disjoint():
    assert iou(BOX, (300, 300, 400, 400)) == 0.0


def test_track_continuity_and_age():
    tr = IouTracker(expiry_s=5.0)
    first = tr.update([det()], now=0.0)
    second = tr.update([det(bbox=(110.0, 105.0, 210.0, 205.0))], now=0.5)
    assert second[0].track_id == first[0].track_id
    assert second[0].age_frames == 2


def test_separate_objects_get_separate_ids():
    tracks = IouTracker(expiry_s=5.0).update(
        [det(), det(bbox=(500.0, 100.0, 600.0, 200.0))], now=0.0)
    assert len({t.track_id for t in tracks}) == 2


def test_track_survives_brief_miss():
    tr = IouTracker(expiry_s=5.0)
    a = tr.update([det()], now=0.0)[0]
    assert tr.update([], now=1.0) == []
    b = tr.update([det()], now=2.0)[0]
    assert b.track_id == a.track_id
    assert b.age_frames == 2


def test_track_expires_after_long_gap():
    tr = IouTracker(expiry_s=1.0)
    a = tr.update([det()], now=0.0)[0]
    tr.update([], now=2.5)
    b = tr.update([det()], now=3.0)[0]
    assert b.track_id != a.track_id
    assert b.age_frames == 1


def test_dominant_class_smooths_label_flicker():
    # A squirrel that COCO calls dog twice and cat once stays a dog track.
    tr = IouTracker(expiry_s=5.0)
    tr.update([det(name="dog", conf=0.6)], now=0.0)
    tr.update([det(name="dog", conf=0.6)], now=0.5)
    out = tr.update([det(name="cat", conf=0.5)], now=1.0)
    assert out[0].class_name == "dog"
