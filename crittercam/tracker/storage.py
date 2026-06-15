"""Clip retention: count cap + disk-watermark safety net.

Two rules prune the oldest non-favorite clips:
  1. Count cap: keep at most storage.max_clips non-favorite clips on disk.
  2. Disk watermark: if the data disk is still over the high watermark, keep
     deleting the next-oldest until usage falls under the low watermark.
Either way only the clip file goes: the sighting row survives as
status='clip_missing' and keeps its thumbnail, so the log of what visited
remains intact. Favorites are never candidates.
"""
from __future__ import annotations

import logging
import shutil
from sqlite3 import Connection

from crittercam import db
from crittercam.config import Config

log = logging.getLogger(__name__)


def disk_used_fraction(cfg: Config) -> float:
    usage = shutil.disk_usage(cfg.data_root)
    return usage.used / usage.total


def _drop_clip(conn: Connection, cfg: Config, row) -> None:
    (cfg.data_root / row["clip_path"]).unlink(missing_ok=True)
    db.mark_clip_missing(conn, row["id"])


def prune_clips(conn: Connection, cfg: Config) -> int:
    candidates = db.prunable_sightings(conn)  # oldest first; never favorites

    # 1. Count cap: drop the oldest beyond max_clips.
    over = max(len(candidates) - cfg.storage.max_clips, 0)
    for row in candidates[:over]:
        _drop_clip(conn, cfg, row)
    if over:
        log.info("clip cap %d exceeded: pruned %d oldest clips",
                 cfg.storage.max_clips, over)

    # 2. Disk-watermark safety net on whatever the cap left behind.
    disk_pruned = 0
    if disk_used_fraction(cfg) >= cfg.storage.disk_high_watermark:
        for row in candidates[over:]:
            if disk_used_fraction(cfg) <= cfg.storage.disk_low_watermark:
                break
            _drop_clip(conn, cfg, row)
            disk_pruned += 1
        if disk_pruned:
            log.warning("disk over %.0f%% watermark: pruned %d more oldest clips",
                        cfg.storage.disk_high_watermark * 100, disk_pruned)

    return over + disk_pruned
