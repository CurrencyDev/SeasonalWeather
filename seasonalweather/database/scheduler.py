from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .core import SeasonalDatabase


class SchedulerStateRepository:
    def __init__(self, db: SeasonalDatabase) -> None:
        self.db = db

    def get_state(self, scheduler_name: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT scheduler_name, last_run_at, next_run_at, state_json FROM scheduler_state WHERE scheduler_name = ?",
                (scheduler_name,),
            ).fetchone()
        if row is None:
            return None
        state = json.loads(row["state_json"] or "{}")
        if not isinstance(state, dict):
            state = {}
        return {
            "scheduler_name": str(row["scheduler_name"]),
            "last_run_at": row["last_run_at"],
            "next_run_at": row["next_run_at"],
            "state": state,
        }

    def upsert_state(
        self,
        scheduler_name: str,
        *,
        last_run_at: str | None = None,
        next_run_at: str | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        payload = dict(state or {})
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO scheduler_state (scheduler_name, last_run_at, next_run_at, state_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(scheduler_name) DO UPDATE SET
                    last_run_at = excluded.last_run_at,
                    next_run_at = excluded.next_run_at,
                    state_json = excluded.state_json
                """,
                (
                    scheduler_name,
                    last_run_at,
                    next_run_at,
                    json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                ),
            )
