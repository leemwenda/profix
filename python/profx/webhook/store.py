"""Replay/duplicate protection backed by SQLite (survives restarts)."""
from __future__ import annotations

import sqlite3
import threading


class SignalStore:
    def __init__(self, path: str):
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS seen (signal_id TEXT PRIMARY KEY, body_sha256 TEXT UNIQUE NOT NULL,"
                " received_at REAL NOT NULL, status TEXT NOT NULL)")

    def register(self, signal_id: str, body_sha256: str, now: float) -> bool:
        """Atomically claim a signal. False if signal_id OR identical body was seen before."""
        with self._lock:
            try:
                self._db.execute("INSERT INTO seen VALUES (?,?,?,?)", (signal_id, body_sha256, now, "accepted"))
                return True
            except sqlite3.IntegrityError:
                return False

    def set_status(self, signal_id: str, status: str) -> None:
        with self._lock:
            self._db.execute("UPDATE seen SET status=? WHERE signal_id=?", (status, signal_id))

    def purge(self, older_than_ts: float) -> int:
        with self._lock:
            return self._db.execute("DELETE FROM seen WHERE received_at < ?", (older_than_ts,)).rowcount

    def close(self) -> None:
        self._db.close()
