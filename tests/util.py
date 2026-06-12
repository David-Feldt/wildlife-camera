"""Shared helpers for synthetic frames."""
import cv2
import numpy as np


def make_image(w: int = 160, h: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def make_jpeg(w: int = 160, h: int = 120, seed: int = 0) -> bytes:
    ok, buf = cv2.imencode(".jpg", make_image(w, h, seed))
    assert ok
    return buf.tobytes()
