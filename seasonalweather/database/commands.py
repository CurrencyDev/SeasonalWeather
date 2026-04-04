from __future__ import annotations

import json
from typing import Any

from .core import SeasonalDatabase


class CommandRepository:
    def __init__(self, db: SeasonalDatabase) -> None:
        self.db = db

    def get_by_command_id(self, command_id: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM api_commands WHERE command_id = ?", (command_id,)).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def get_by_idempotency_key(self, idempotency_key: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM api_commands WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def insert(self, record: dict[str, Any]) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO api_commands (
                    command_id, command_type, status, accepted_at, idempotency_key,
                    actor, payload_hash, payload_json, request_id, started_at,
                    finished_at, idempotent_replay_count, result_json, error_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["command_id"],
                    record["command_type"],
                    record["status"],
                    record["accepted_at"],
                    record["idempotency_key"],
                    record["actor"],
                    record["payload_hash"],
                    json.dumps(record.get("payload") or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                    record["request_id"],
                    record.get("started_at"),
                    record.get("finished_at"),
                    int(record.get("idempotent_replay_count", 0) or 0),
                    json.dumps(record.get("result") or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    if record.get("result") is not None else None,
                    json.dumps(record.get("error") or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    if record.get("error") is not None else None,
                ),
            )

    def update(self, record: dict[str, Any]) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE api_commands
                   SET status = ?,
                       started_at = ?,
                       finished_at = ?,
                       idempotent_replay_count = ?,
                       result_json = ?,
                       error_json = ?
                 WHERE command_id = ?
                """,
                (
                    record["status"],
                    record.get("started_at"),
                    record.get("finished_at"),
                    int(record.get("idempotent_replay_count", 0) or 0),
                    json.dumps(record.get("result") or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    if record.get("result") is not None else None,
                    json.dumps(record.get("error") or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    if record.get("error") is not None else None,
                    record["command_id"],
                ),
            )


    def prune_terminal_before(self, cutoff_iso: str) -> int:
        with self.db.transaction() as conn:
            cur = conn.execute(
                """
                DELETE FROM api_commands
                 WHERE status IN ('succeeded', 'failed')
                   AND COALESCE(finished_at, accepted_at) < ?
                """,
                (cutoff_iso,),
            )
        return int(cur.rowcount or 0)

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        return {
            "command_id": str(row["command_id"]),
            "command_type": str(row["command_type"]),
            "status": str(row["status"]),
            "accepted_at": str(row["accepted_at"]),
            "idempotency_key": str(row["idempotency_key"]),
            "actor": str(row["actor"]),
            "payload_hash": str(row["payload_hash"]),
            "payload": json.loads(row["payload_json"] or "{}"),
            "request_id": str(row["request_id"]),
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "idempotent_replay_count": int(row["idempotent_replay_count"] or 0),
            "result": json.loads(row["result_json"] or "null"),
            "error": json.loads(row["error_json"] or "null"),
        }
