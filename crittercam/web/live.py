"""ZMQ SUB -> MJPEG re-serving for browsers.

Each connected browser gets its own SUB socket with CONFLATE so it always
receives the newest frame and never builds a backlog. Output is capped at
web.live_fps regardless of the tracker's publish rate.
"""
from __future__ import annotations

import time
from typing import Iterator

import zmq

_BOUNDARY = b"--crittercam-frame"


def mjpeg_stream(endpoint: str, max_fps: float) -> Iterator[bytes]:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.connect(endpoint)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    min_interval = 1.0 / max_fps
    try:
        while True:
            try:
                jpeg = sock.recv()
            except zmq.Again:
                continue  # tracker quiet; keep the connection open
            yield (
                _BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                + jpeg + b"\r\n"
            )
            time.sleep(min_interval)
    finally:
        sock.close(linger=0)


MJPEG_MEDIA_TYPE = f"multipart/x-mixed-replace; boundary={_BOUNDARY[2:].decode()}"
