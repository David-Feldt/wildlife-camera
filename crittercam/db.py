"""SQLite schema, migrations, and typed query helpers.

Single-writer discipline: the tracker process owns all writes to sightings;
the web process only writes favorites/deletes/config. WAL mode makes the
concurrent reader + single writer pattern safe.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    migrations = sorted(MIGRATIONS_DIR.glob("[0-9]*.sql"))
    for path in migrations:
        version = int(re.match(r"(\d+)", path.name).group(1))
        if version <= current:
            continue
        log.info("applying migration %s", path.name)
        with conn:
            conn.executescript(path.read_text())
            conn.execute(f"PRAGMA user_version={version}")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = connect(db_path)
    migrate(conn)
    return conn


# --- config_kv helpers -------------------------------------------------------

def kv_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO config_kv(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


def kv_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM config_kv WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def heartbeat(conn: sqlite3.Connection) -> None:
    kv_set(conn, "tracker_heartbeat", str(time.time()))


def heartbeat_age(conn: sqlite3.Connection) -> float | None:
    raw = kv_get(conn, "tracker_heartbeat")
    return time.time() - float(raw) if raw else None


# --- sightings helpers (tracker-owned writes) --------------------------------

def insert_sighting(conn: sqlite3.Connection, started_at: str,
                    dominant_class: str, max_confidence: float) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO sightings(started_at, dominant_class, max_confidence, status) "
            "VALUES(?, ?, ?, 'recording')",
            (started_at, dominant_class, max_confidence),
        )
    return cur.lastrowid


def set_sighting_clip(conn: sqlite3.Connection, sighting_id: int, clip_path: str) -> None:
    with conn:
        conn.execute("UPDATE sightings SET clip_path=? WHERE id=?", (clip_path, sighting_id))


def close_sighting(conn: sqlite3.Connection, sighting_id: int, *, ended_at: str,
                   duration_s: float, dominant_class: str, max_confidence: float,
                   track_count: int, clip_path: str | None, thumb_path: str | None,
                   status: str) -> None:
    with conn:
        conn.execute(
            "UPDATE sightings SET ended_at=?, duration_s=?, dominant_class=?, "
            "max_confidence=?, track_count=?, clip_path=?, thumb_path=?, status=? "
            "WHERE id=?",
            (ended_at, duration_s, dominant_class, max_confidence, track_count,
             clip_path, thumb_path, status, sighting_id),
        )


def insert_detection_sample(conn: sqlite3.Connection, sighting_id: int, ts: str,
                            class_name: str, confidence: float, bbox_json: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO detections_sample(sighting_id, ts, class, confidence, bbox) "
            "VALUES(?, ?, ?, ?, ?)",
            (sighting_id, ts, class_name, confidence, bbox_json),
        )


def recover_stale_recordings(conn: sqlite3.Connection, data_root: Path) -> None:
    """Clean up sightings left in 'recording' by a crash: the clip file has an
    unpatched header and is unreadable, so drop it and mark the row."""
    rows = conn.execute("SELECT id, clip_path FROM sightings WHERE status='recording'").fetchall()
    for row in rows:
        log.warning("recovering interrupted sighting %d", row["id"])
        if row["clip_path"]:
            (data_root / row["clip_path"]).unlink(missing_ok=True)
    if rows:
        with conn:
            conn.execute(
                "UPDATE sightings SET status='clip_missing', clip_path=NULL "
                "WHERE status='recording'"
            )


def recent_sightings(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM sightings ORDER BY started_at DESC, id DESC LIMIT ?", (limit,)
    ).fetchall()


def prunable_sightings(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Oldest first; favorites and already-pruned rows are never candidates."""
    return conn.execute(
        "SELECT id, clip_path FROM sightings "
        "WHERE favorite=0 AND clip_path IS NOT NULL AND status='complete' "
        "ORDER BY started_at"
    ).fetchall()


def mark_clip_missing(conn: sqlite3.Connection, sighting_id: int) -> None:
    with conn:
        conn.execute(
            "UPDATE sightings SET status='clip_missing', clip_path=NULL WHERE id=?",
            (sighting_id,),
        )


# --- sightings helpers (web-owned writes) ------------------------------------

def get_sighting(conn: sqlite3.Connection, sighting_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM sightings WHERE id=?", (sighting_id,)
    ).fetchone()


def set_favorite(conn: sqlite3.Connection, sighting_id: int, favorite: bool) -> None:
    with conn:
        conn.execute(
            "UPDATE sightings SET favorite=? WHERE id=?",
            (int(favorite), sighting_id),
        )


def delete_sighting(conn: sqlite3.Connection, sighting_id: int) -> None:
    """Deletes the row (detection samples cascade); the caller unlinks files."""
    with conn:
        conn.execute("DELETE FROM sightings WHERE id=?", (sighting_id,))
