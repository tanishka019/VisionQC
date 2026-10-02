"""
db.py — SQLite database layer for VisionQC
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "visionqc.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS inspections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            image_path  TEXT,
            heatmap_path TEXT,
            anomaly_score REAL NOT NULL,
            confidence  REAL NOT NULL,
            result      TEXT NOT NULL,
            threshold   REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS config (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        INSERT OR IGNORE INTO config (key, value) VALUES ('threshold', '0.5');
        INSERT OR IGNORE INTO config (key, value) VALUES ('model_trained', 'false');
        INSERT OR IGNORE INTO config (key, value) VALUES ('product_name', 'Screw');
    """)
    conn.commit()
    conn.close()


def log_inspection(image_path: str, heatmap_path: str, score: float,
                   confidence: float, result: str, threshold: float) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO inspections (timestamp, image_path, heatmap_path,
                                 anomaly_score, confidence, result, threshold)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        image_path,
        heatmap_path,
        score,
        confidence,
        result,
        threshold,
    ))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_history(limit: int = 200):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, timestamp, anomaly_score, confidence, result, threshold, heatmap_path
        FROM inspections
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_today_stats():
    conn = get_conn()
    cur = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    cur.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN result = 'PASS' THEN 1 ELSE 0 END) as passed,
            SUM(CASE WHEN result = 'FAIL' THEN 1 ELSE 0 END) as failed
        FROM inspections
        WHERE timestamp LIKE ?
    """, (f"{today}%",))
    row = dict(cur.fetchone())
    conn.close()
    total = row["total"] or 0
    passed = row["passed"] or 0
    failed = row["failed"] or 0
    rejection_rate = round((failed / total * 100), 1) if total > 0 else 0.0
    return {"total": total, "passed": passed, "failed": failed, "rejection_rate": rejection_rate}


def get_hourly_stats():
    """Return hourly pass/fail counts for today for charting."""
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


def get_config(key: str) -> str:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM config WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else None


def set_config(key: str, value: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()
