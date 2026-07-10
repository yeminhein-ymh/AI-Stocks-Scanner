from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
 id INTEGER PRIMARY KEY AUTOINCREMENT, signal_date TEXT NOT NULL, ticker TEXT NOT NULL,
 strategy TEXT NOT NULL, prediction TEXT NOT NULL, entry REAL NOT NULL, target REAL,
 stop REAL, confidence REAL, horizon_days INTEGER DEFAULT 20, exit_date TEXT, exit REAL,
 max_gain REAL, max_loss REAL, actual_result REAL, outcome TEXT,
 UNIQUE(signal_date, ticker, strategy, horizon_days)
)
"""


class SignalStore:
    def __init__(self, path: str | Path = "data/signals.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as conn:
            conn.execute(SCHEMA)
            conn.commit()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def record(self, values: dict[str, Any]) -> bool:
        columns = ["signal_date", "ticker", "strategy", "prediction", "entry", "target", "stop", "confidence", "horizon_days"]
        with closing(self.connect()) as conn:
            cursor = conn.execute(
                f"INSERT OR IGNORE INTO signals ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                [values.get(c) for c in columns],
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_outcome(self, signal_id: int, **values: Any) -> None:
        allowed = {"exit_date", "exit", "max_gain", "max_loss", "actual_result", "outcome"}
        clean = {k: v for k, v in values.items() if k in allowed}
        if not clean:
            return
        clause = ",".join(f"{key}=?" for key in clean)
        with closing(self.connect()) as conn:
            conn.execute(f"UPDATE signals SET {clause} WHERE id=?", [*clean.values(), signal_id])
            conn.commit()

    def frame(self) -> pd.DataFrame:
        with closing(self.connect()) as conn:
            return pd.read_sql_query("SELECT * FROM signals ORDER BY signal_date DESC, id DESC", conn)


def performance_metrics(frame: pd.DataFrame) -> dict[str, float]:
    closed = frame.dropna(subset=["actual_result"]).copy() if not frame.empty else frame
    if closed.empty:
        return {"Signals": float(len(frame)), "Closed": 0.0, "Win Rate": float("nan"),
                "Profit Factor": float("nan"), "Expectancy": float("nan"), "Sharpe": float("nan")}
    returns = closed["actual_result"].astype(float)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() else float("inf")
    sharpe = returns.mean() / returns.std() * (252 / 20) ** .5 if returns.std() else float("nan")
    return {"Signals": float(len(frame)), "Closed": float(len(closed)), "Win Rate": float((returns > 0).mean() * 100),
            "Profit Factor": float(profit_factor), "Expectancy": float(returns.mean()), "Sharpe": float(sharpe)}
