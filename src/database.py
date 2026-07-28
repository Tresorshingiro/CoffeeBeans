"""SQLite persistence. Stdlib sqlite3, no ORM.

WAL mode is enabled because the retrain background thread writes concurrently
with request handlers.
"""
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path

from . import config

_db_path: Path = config.DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id    TEXT NOT NULL,
    filename    TEXT NOT NULL,
    class_label TEXT NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    uploaded_at TEXT NOT NULL,
    used_in_run INTEGER
);

CREATE TABLE IF NOT EXISTS retrain_runs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at         TEXT NOT NULL,
    finished_at        TEXT,
    status             TEXT NOT NULL,
    n_pending          INTEGER NOT NULL,
    n_replay           INTEGER NOT NULL,
    epochs             INTEGER NOT NULL,
    candidate_accuracy REAL,
    champion_accuracy  REAL,
    promoted           INTEGER NOT NULL DEFAULT 0,
    model_path         TEXT,
    log                TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence      REAL NOT NULL,
    latency_ms      REAL NOT NULL,
    model_version   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_registry (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    version     TEXT NOT NULL,
    path        TEXT NOT NULL,
    accuracy    REAL NOT NULL,
    promoted_at TEXT NOT NULL,
    is_champion INTEGER NOT NULL DEFAULT 0
);
"""


def set_db_path(path) -> None:
    global _db_path
    _db_path = Path(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_db_path, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def record_upload(batch_id, filename, class_label, sha256) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO uploads "
            "(batch_id, filename, class_label, sha256, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (batch_id, filename, class_label, sha256, _now()),
        )


def sha256_exists(sha256) -> bool:
    with connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM uploads WHERE sha256 = ?", (sha256,)
        ).fetchone()
    return row is not None


def create_retrain_run(n_pending, n_replay, epochs) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO retrain_runs "
            "(started_at, status, n_pending, n_replay, epochs) "
            "VALUES (?, 'running', ?, ?, ?)",
            (_now(), n_pending, n_replay, epochs),
        )
        return int(cur.lastrowid)


def finish_retrain_run(run_id, status, candidate_accuracy,
                       champion_accuracy, promoted, model_path, log) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE retrain_runs SET finished_at = ?, status = ?, "
            "candidate_accuracy = ?, champion_accuracy = ?, promoted = ?, "
            "model_path = ?, log = ? WHERE id = ?",
            (_now(), status, candidate_accuracy, champion_accuracy,
             int(promoted), model_path, log, run_id),
        )


def get_retrain_run(run_id):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM retrain_runs WHERE id = ?", (run_id,)
        ).fetchone()
    return dict(row) if row else None


def list_retrain_runs(limit=20):
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM retrain_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def record_prediction(predicted_class, confidence, latency_ms,
                      model_version) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO predictions "
            "(ts, predicted_class, confidence, latency_ms, model_version) "
            "VALUES (?, ?, ?, ?, ?)",
            (_now(), predicted_class, confidence, latency_ms, model_version),
        )


def prediction_stats():
    with connect() as conn:
        latencies = [r[0] for r in conn.execute(
            "SELECT latency_ms FROM predictions").fetchall()]
        counts = conn.execute(
            "SELECT predicted_class, COUNT(*) FROM predictions "
            "GROUP BY predicted_class").fetchall()
    if not latencies:
        return {"total": 0, "mean_latency_ms": 0.0,
                "p95_latency_ms": 0.0, "class_counts": {}}
    ordered = sorted(latencies)
    idx = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "total": len(latencies),
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "p95_latency_ms": round(ordered[idx], 2),
        "class_counts": {row[0]: row[1] for row in counts},
    }


def register_model(version, path, accuracy) -> None:
    with connect() as conn:
        conn.execute("UPDATE model_registry SET is_champion = 0")
        conn.execute(
            "INSERT INTO model_registry "
            "(version, path, accuracy, promoted_at, is_champion) "
            "VALUES (?, ?, ?, ?, 1)",
            (version, str(path), accuracy, _now()),
        )


def get_champion():
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM model_registry WHERE is_champion = 1 "
            "ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None
