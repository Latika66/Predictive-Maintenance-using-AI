import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "predictions.db")


def init_db():
    """Creates the predictions table if it doesn't already exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            machine_type TEXT,
            sensor_data TEXT NOT NULL,
            failure_risk REAL NOT NULL,
            risk_level TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _risk_level(risk, threshold):
    if risk > threshold:
        return "High"
    elif risk > 0.3:
        return "Medium"
    return "Low"


def log_prediction(source, machine_type, sensor_data: dict, risk: float, threshold: float):
    """Logs a single prediction to the database. Fails silently (non-critical feature)."""
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO predictions (timestamp, source, machine_type, sensor_data, failure_risk, risk_level) VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                source,
                machine_type,
                json.dumps(sensor_data, default=str),
                float(risk),
                _risk_level(risk, threshold),
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # history is a nice-to-have, never block a prediction on it


def log_predictions_batch(source, rows: list, threshold: float):
    """Logs multiple predictions at once (e.g. from a CSV upload)."""
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        ts = datetime.now().isoformat(timespec="seconds")
        records = [
            (
                ts,
                source,
                row.get("Type", "N/A"),
                json.dumps(row, default=str),
                float(row["Failure Risk"]),
                _risk_level(row["Failure Risk"], threshold),
            )
            for row in rows
        ]
        conn.executemany(
            "INSERT INTO predictions (timestamp, source, machine_type, sensor_data, failure_risk, risk_level) VALUES (?, ?, ?, ?, ?, ?)",
            records,
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_history(limit=500):
    """Returns prediction history as a list of dicts, most recent first."""
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def clear_history():
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False
