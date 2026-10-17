import sqlite3
from datetime import datetime
import pandas as pd

DB_PATH = "data/churn.db"


def _connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create the predictions table if it does not exist."""
    conn = _connect()
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        customer TEXT,
        source TEXT,
        tenure INTEGER,
        contract TEXT,
        internet TEXT,
        monthly REAL,
        risk_score REAL,
        risk_level TEXT)""")
    conn.commit()
    conn.close()


def save_predictions(records):
    """records = list of (customer, source, tenure, contract, internet,
    monthly, risk_score, risk_level)."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [(now,) + tuple(r) for r in records]
    conn = _connect()
    conn.executemany(
        "INSERT INTO predictions (created_at, customer, source, tenure, "
        "contract, internet, monthly, risk_score, risk_level) "
        "VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def load_history(limit=1000):
    conn = _connect()
    df = pd.read_sql_query(
        "SELECT created_at, customer, source, tenure, contract, internet, "
        "monthly, risk_score, risk_level FROM predictions "
        "ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    conn.close()
    return df


def clear_history():
    conn = _connect()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()