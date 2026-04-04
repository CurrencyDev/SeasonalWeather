from __future__ import annotations

import sqlite3

from .schema import MIGRATIONS, SCHEMA_VERSION


def _ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
        """
    )


def current_version(conn: sqlite3.Connection) -> int:
    _ensure_schema_migrations(conn)
    row = conn.execute("SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations").fetchone()
    return int(row[0] if row is not None else 0)


def apply_pending_migrations(conn: sqlite3.Connection) -> int:
    _ensure_schema_migrations(conn)
    applied = 0
    start_version = current_version(conn)
    for version in range(start_version + 1, SCHEMA_VERSION + 1):
        statements = MIGRATIONS.get(version)
        if not statements:
            raise RuntimeError(f"Missing SQLite migration for schema version {version}")
        for statement in statements:
            conn.execute(statement)
        conn.execute("INSERT INTO schema_migrations(version) VALUES (?)", (version,))
        applied += 1
    return applied
