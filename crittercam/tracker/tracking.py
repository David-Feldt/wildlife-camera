"""Greedy IoU tracking across inference rounds.

CPU inference runs at ~2-3 FPS, so consecutive looks at a moving animal can
be ~0.5s apart. The IoU threshold is deliberately loose to keep association
working across those gaps, and matching ignores the class label on purpose:
COCO flickers between e.g. dog/cat on animals it was never trained on, and a
class-strict matcher would split one critter into many tracks. Each track
instead accumulates per-class confidence and reports the running winner.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from crittercam.models import Detection, Track

IOU_MATCH_THRESHOLD = 0.2


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union


@dataclass
class _TrackState:
    track_id: int
    bbox: tuple[float, float, float, float]
    last_seen: float
    age_frames: int = 1
    confidence: float = 0.0
    class_scores: dict[str, float] = field(default_factory=dict)

    def dominant_class(self) -> str:
        return max(self.class_scores, key=self.class_scores.get)


class IouTracker:
    """Associates detections to tracks greedily by best IoU first. Unmatched
    tracks survive expiry_s of missed rounds before being dropped, so a brief
    occlusion or confidence dip does not split one animal into two tracks."""

    def __init__(self, expiry_s: float, iou_threshold: float = IOU_MATCH_THRESHOLD):
        self.expiry_s = expiry_s
        self.iou_threshold = iou_threshold
        self._tracks: list[_TrackState] = []
        self._next_id = 1

    def update(self, detections: list[Detection], now: float) -> list[Track]:
        """Consume one inference round; returns only the tracks seen this round."""
        pairs = []
        for ti, t in enumerate(self._tracks):
            for di, d in enumerate(detections):
                score = iou(t.bbox, d.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        matched_t: set[int] = set()
        matched_d: set[int] = set()
        seen: list[_TrackState] = []
        for score, ti, di in pairs:
            if ti in matched_t or di in matched_d:
                continue
            matched_t.add(ti)
            matched_d.add(di)
            t, d = self._tracks[ti], detections[di]
            t.bbox = d.bbox
            t.last_seen = now
            t.age_frames += 1
            t.confidence = d.confidence
            t.class_scores[d.class_name] = t.class_scores.get(d.class_name, 0.0) + d.confidence
            seen.append(t)

        for di, d in enumerate(detections):
            if di in matched_d:
                continue
            t = _TrackState(self._next_id, d.bbox, now, confidence=d.confidence,
                            class_scores={d.class_name: d.confidence})
            self._next_id += 1
            self._tracks.append(t)
            seen.append(t)

        self._tracks = [t for t in self._tracks if now - t.last_seen <= self.expiry_s]
        return [Track(t.track_id, t.dominant_class(), t.confidence, t.bbox, t.age_frames)
                for t in seen]
