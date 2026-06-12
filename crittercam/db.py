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
