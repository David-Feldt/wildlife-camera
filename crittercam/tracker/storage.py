"""Disk-watermark pruning.

When the data disk crosses the high watermark, the oldest non-favorite clips
are deleted until usage falls under the low watermark. Only the clip file
goes: the sighting row survives as status='clip_missing' and keeps its
thumbnail, so the log of what visited remains intact.
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


def prune_clips(conn: Connection, cfg: Config) -> int:
    if disk_used_fraction(cfg) < cfg.storage.disk_high_watermark:
        return 0
    pruned = 0
    for row in db.prunable_sightings(conn):
        if disk_used_fraction(cfg) <= cfg.storage.disk_low_watermark:
            break
        (cfg.data_root / row["clip_path"]).unlink(missing_ok=True)
        db.mark_clip_missing(conn, row["id"])
        pruned += 1
    if pruned:
        log.warning("disk over %.0f%% watermark: pruned %d oldest clips",
                    cfg.storage.disk_high_watermark * 100, pruned)
    return pruned
