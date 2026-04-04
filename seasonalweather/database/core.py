from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .migrations import apply_pending_migrations

_ALLOWED_JOURNAL_MODES = {"DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"}


class SeasonalDatabase:
    def __init__(self, *, path: str, busy_timeout_ms: int = 5000, journal_mode: str = "WAL") -> None:
        self.path = str(path)
        self.busy_timeout_ms = max(100, int(busy_timeout_ms))
        mode = str(journal_mode or "WAL").strip().upper()
        self.journal_mode = mode if mode in _ALLOWED_JOURNAL_MODES else "WAL"
        self._bootstrapped = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=max(1.0, self.busy_timeout_ms / 1000.0))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        conn.execute(f"PRAGMA journal_mode = {self.journal_mode}")
        return conn

    def bootstrap(self) -> None:
        if self._bootstrapped:
            return
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            apply_pending_migrations(conn)
            conn.commit()
        self._bootstrapped = True

    @contextmanager
    def connect(self):
        self.bootstrap()
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self):
        self.bootstrap()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
