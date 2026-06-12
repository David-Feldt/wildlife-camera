CREATE TABLE sightings (
  id             INTEGER PRIMARY KEY,
  started_at     TEXT NOT NULL,        -- ISO 8601 UTC
  ended_at       TEXT,
  duration_s     REAL,
  dominant_class TEXT NOT NULL,
  species        TEXT,                 -- reserved for v2 classifier
  max_confidence REAL NOT NULL,
  track_count    INTEGER NOT NULL DEFAULT 1,
  clip_path      TEXT,                 -- relative to data root
  thumb_path     TEXT,
  favorite       INTEGER NOT NULL DEFAULT 0,
  status         TEXT NOT NULL         -- 'recording' | 'complete' | 'clip_missing'
);
CREATE INDEX idx_sightings_started ON sightings(started_at);
CREATE INDEX idx_sightings_class   ON sightings(dominant_class);

CREATE TABLE detections_sample (
  id           INTEGER PRIMARY KEY,
  sighting_id  INTEGER NOT NULL REFERENCES sightings(id) ON DELETE CASCADE,
  ts           TEXT NOT NULL,
  class        TEXT NOT NULL,
  confidence   REAL NOT NULL,
  bbox         TEXT NOT NULL           -- JSON [x,y,w,h] normalized
);
CREATE INDEX idx_detections_sighting ON detections_sample(sighting_id);

CREATE TABLE config_kv (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
