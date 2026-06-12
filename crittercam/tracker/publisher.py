"""Annotates frames and publishes them as JPEG on a ZMQ PUB socket.

PUB/SUB drops frames for slow subscribers instead of backpressuring the
pipeline — exactly what we want for a live view.
"""
from __future__ import annotations

import cv2
import zmq

from crittercam.models import FrameResult

_BOX_COLOR = (80, 220, 80)
_TEXT_COLOR = (0, 0, 0)


class FramePublisher:
    def __init__(self, endpoint: str, jpeg_quality: int = 80):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 2)
        self._sock.bind(endpoint)
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    def publish(self, result: FrameResult, status_line: str = "") -> bytes:
        """Publish an annotated frame for the live view; returns a JPEG of the
        *clean* frame for the recorder. Clips feed milestone 5 training, so
        boxes and status text must never be burned into recorded frames."""
        ok, clean = cv2.imencode(".jpg", result.frame.image, self._encode_params)
        clean_jpeg = clean.tobytes() if ok else b""
        image = result.frame.image.copy()
        if result.tracks:
            for trk in result.tracks:
                self._draw_box(image, trk.bbox,
                               f"#{trk.track_id} {trk.class_name} {trk.confidence:.2f}")
        else:
            for det in result.detections:
                self._draw_box(image, det.bbox, f"{det.class_name} {det.confidence:.2f}")
        if status_line:
            cv2.putText(image, status_line, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ok, jpeg = cv2.imencode(".jpg", image, self._encode_params)
        if ok:
            # Single-part on purpose: subscribers use CONFLATE, which does not
            # support multipart messages.
            self._sock.send(jpeg.tobytes(), flags=zmq.NOBLOCK)
        return clean_jpeg

    @staticmethod
    def _draw_box(image, bbox, label: str) -> None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), _BOX_COLOR, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _TEXT_COLOR, 2)

    def close(self) -> None:
        self._sock.close(linger=0)
