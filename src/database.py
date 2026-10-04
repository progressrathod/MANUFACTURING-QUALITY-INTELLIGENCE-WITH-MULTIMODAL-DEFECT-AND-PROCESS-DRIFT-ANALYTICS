"""Step 21 - SQL database (SQLite). Creates the tables and gives a connection helper."""

import sqlite3
from contextlib import contextmanager
from src.config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id         TEXT PRIMARY KEY,
    line_id          TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    ended_at         TEXT,
    status           TEXT NOT NULL CHECK (status IN
                     ('RUNNING','PAUSED','COMPLETED','ON_HOLD','RELEASED','SCRAPPED')),
    planned_size     INTEGER NOT NULL,
    ok_count         INTEGER DEFAULT 0,
    defective_count  INTEGER DEFAULT 0,
    review_count     INTEGER DEFAULT 0,
    drift_count      INTEGER DEFAULT 0,
    decision         TEXT,
    decision_by      TEXT,
    decision_note    TEXT
);

CREATE TABLE IF NOT EXISTS products (
    product_id        TEXT PRIMARY KEY,
    batch_id          TEXT NOT NULL REFERENCES batches(batch_id),
    position          INTEGER NOT NULL,
    inspected_at      TEXT NOT NULL,
    temperature REAL,
    pressure REAL,
    machine_speed REAL,
    vibration REAL,
    humidity REAL,
    material_thickness REAL,
    cycle_time REAL,
    tool_wear REAL,
    image_file        TEXT,
    image_label       TEXT,
    image_confidence  REAL,
    process_label     TEXT,
    defect_probability REAL,
    risk_level        TEXT,
    drift             INTEGER NOT NULL DEFAULT 0,
    anomaly_score     REAL,
    final_status      TEXT NOT NULL CHECK (final_status IN ('OK','REVIEW','DEFECTIVE')),
    reasons           TEXT NOT NULL,
    top_factors       TEXT NOT NULL,
    UNIQUE (batch_id, position)
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id        TEXT NOT NULL REFERENCES batches(batch_id),
    product_id      TEXT,
    created_at      TEXT NOT NULL,
    level           TEXT NOT NULL CHECK (level IN ('WARNING','STOP')),
    code            TEXT NOT NULL,
    message         TEXT NOT NULL,
    acknowledged_by TEXT,
    acknowledged_at TEXT
);

CREATE TABLE IF NOT EXISTS batch_reports (
    report_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id     TEXT NOT NULL REFERENCES batches(batch_id),
    generated_at TEXT NOT NULL,
    report_json  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT,
    at TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    note TEXT
);

CREATE INDEX IF NOT EXISTS idx_products_batch ON products(batch_id);
CREATE INDEX IF NOT EXISTS idx_alerts_batch   ON alerts(batch_id);
"""


@contextmanager
def get_conn():
    """Open the database, commit on success, roll back on error, always close."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _add_column(c, table, col, ddl):
    """Add a column only if it does not already exist."""

    cols = [
        r["name"]
        for r in c.execute(f"PRAGMA table_info({table})")
    ]

    if col not in cols:
        c.execute(
            f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"
        )


def init_db():
    """Create tables and safely migrate older databases."""

    with get_conn() as c:

        # Create tables if they don't exist
        c.executescript(SCHEMA)

        # Migration for existing databases
        # These columns are required by the product review system.
        for col in (
            "review_result",
            "reviewed_by",
            "reviewed_at",
            "review_note"
        ):
            _add_column(c, "products", col, "TEXT")


if __name__ == "__main__":
    init_db()
    print("Database ready:", DB_PATH)