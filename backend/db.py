"""
db.py — SQLite database layer for VisionQC (v3)
"""
import sqlite3
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.getenv(
    "VISIONQC_DB_PATH",
    os.path.join(os.getenv("VISIONQC_DATA_DIR", os.path.dirname(__file__)), "visionqc.db"),
)


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS inspections (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            image_path    TEXT,
            heatmap_path  TEXT,
            anomaly_score REAL NOT NULL,
            confidence    REAL NOT NULL,
            result        TEXT NOT NULL,
            threshold     REAL NOT NULL,
            filename      TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS config (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        INSERT OR IGNORE INTO config (key, value) VALUES ('threshold', '0.5');
        INSERT OR IGNORE INTO config (key, value) VALUES ('model_trained', 'false');
        INSERT OR IGNORE INTO config (key, value) VALUES ('product_name', 'Product');
        INSERT OR IGNORE INTO config (key, value) VALUES ('trained_at', '');
        INSERT OR IGNORE INTO config (key, value) VALUES ('train_image_count', '0');
    """)
    # Add filename column to existing DBs (migration)
    try:
        cur.execute("ALTER TABLE inspections ADD COLUMN filename TEXT DEFAULT ''")
        conn.commit()
    except Exception:
        pass  # column already exists
    conn.commit()
    conn.close()


def log_inspection(
    image_path: str,
    heatmap_path: str,
    score: float,
    confidence: float,
    result: str,
    threshold: float,
    filename: str = "",
) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO inspections
            (timestamp, image_path, heatmap_path, anomaly_score, confidence, result, threshold, filename)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        image_path,
        heatmap_path,
        score,
        confidence,
        result,
        threshold,
        filename,
    ))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_history(
    limit: int = 200,
    result_filter: Optional[str] = None,
    page: int = 1,
) -> list:
    conn = get_conn()
    cur = conn.cursor()
    offset = (page - 1) * limit
    where = ""
    params = []
    if result_filter in ("PASS", "FAIL"):
        where = "WHERE result = ?"
        params.append(result_filter)
    params.extend([limit, offset])
    cur.execute(f"""
        SELECT id, timestamp, anomaly_score, confidence, result,
               threshold, heatmap_path, filename
        FROM inspections
        {where}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    """, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def delete_inspection(inspection_id: int) -> Optional[tuple]:
    """Delete one inspection; returns its (image_path, heatmap_path), or None if not found."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT image_path, heatmap_path FROM inspections WHERE id = ?", (inspection_id,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        return None
    cur.execute("DELETE FROM inspections WHERE id = ?", (inspection_id,))
    conn.commit()
    conn.close()
    return (row["image_path"], row["heatmap_path"])


def clear_history() -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM inspections")
    conn.commit()
    count = cur.rowcount
    conn.close()
    return count


def get_total_inspections(result_filter: Optional[str] = None) -> int:
    conn = get_conn()
    cur = conn.cursor()
    if result_filter in ("PASS", "FAIL"):
        cur.execute("SELECT COUNT(*) as cnt FROM inspections WHERE result = ?", (result_filter,))
    else:
        cur.execute("SELECT COUNT(*) as cnt FROM inspections")
    row = cur.fetchone()
    conn.close()
    return row["cnt"] if row else 0


def get_today_stats() -> dict:
    conn = get_conn()
    cur = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    cur.execute("""
        SELECT
            COUNT(*)  as total,
            SUM(CASE WHEN result = 'PASS' THEN 1 ELSE 0 END) as passed,
            SUM(CASE WHEN result = 'FAIL' THEN 1 ELSE 0 END) as failed,
            AVG(anomaly_score) as avg_score,
            AVG(confidence)    as avg_confidence
        FROM inspections
        WHERE timestamp LIKE ?
    """, (f"{today}%",))
    row = dict(cur.fetchone())
    conn.close()
    total      = row["total"]      or 0
    passed     = row["passed"]     or 0
    failed     = row["failed"]     or 0
    avg_score  = round(row["avg_score"]      or 0.0, 4)
    avg_conf   = round(row["avg_confidence"] or 0.0, 1)
    rej_rate   = round((failed / total * 100), 1) if total > 0 else 0.0
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "rejection_rate": rej_rate,
        "avg_score": avg_score,
        "avg_confidence": avg_conf,
    }


def get_hourly_stats() -> list:
    """Return hourly pass/fail counts for today."""
    conn = get_conn()
    cur = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    cur.execute("""
        SELECT
            strftime('%H', timestamp) as hour,
            SUM(CASE WHEN result = 'PASS' THEN 1 ELSE 0 END) as passed,
            SUM(CASE WHEN result = 'FAIL' THEN 1 ELSE 0 END) as failed
        FROM inspections
        WHERE timestamp LIKE ?
        GROUP BY hour
        ORDER BY hour
    """, (f"{today}%",))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_weekly_stats() -> list:
    """Return daily pass/fail/total for last 7 days."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            date(timestamp) as day,
            COUNT(*) as total,
            SUM(CASE WHEN result = 'PASS' THEN 1 ELSE 0 END) as passed,
            SUM(CASE WHEN result = 'FAIL' THEN 1 ELSE 0 END) as failed
        FROM inspections
        WHERE timestamp >= date('now', 'localtime', '-6 days')
        GROUP BY day
        ORDER BY day
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_config(key: str) -> Optional[str]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM config WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else None


def set_config(key: str, value: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, value)
    )
    conn.commit()
    conn.close()


def get_today_inspections() -> list:
    """Every inspection logged today, oldest first (for the daily report)."""
    conn = get_conn()
    cur = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    cur.execute("""
        SELECT id, timestamp, anomaly_score, confidence, result, threshold, heatmap_path, filename
        FROM inspections
        WHERE timestamp LIKE ?
        ORDER BY id ASC
    """, (f"{today}%",))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
