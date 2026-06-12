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
        """Annotate, encode, publish. Returns the JPEG so the recorder's ring
        buffer can reuse it without a second encode."""
        image = result.frame.image.copy()
        for det in result.detections:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), _BOX_COLOR, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), _BOX_COLOR, -1)
            cv2.putText(image, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _TEXT_COLOR, 2)
        if status_line:
            cv2.putText(image, status_line, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ok, jpeg = cv2.imencode(".jpg", image, self._encode_params)
        if not ok:
            return b""
        data = jpeg.tobytes()
        # Single-part on purpose: subscribers use CONFLATE, which does not
        # support multipart messages.
        self._sock.send(data, flags=zmq.NOBLOCK)
        return data

    def close(self) -> None:
        self._sock.close(linger=0)
