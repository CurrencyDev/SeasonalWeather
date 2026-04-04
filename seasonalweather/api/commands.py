from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..database.commands import CommandRepository
from ..database.core import SeasonalDatabase
from .models import CommandStatus


class IdempotencyConflictError(Exception):
    pass


class CommandNotFoundError(KeyError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class CommandRecord:
    command_id: str
    command_type: str
    status: str
    accepted_at: str
    idempotency_key: str
    actor: str
    payload_hash: str
    payload: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:16]}")
    started_at: str | None = None
    finished_at: str | None = None
    idempotent_replay_count: int = 0
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "command_type": self.command_type,
            "status": self.status,
            "accepted_at": self.accepted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "idempotency_key": self.idempotency_key,
            "actor": self.actor,
            "idempotent_replay_count": self.idempotent_replay_count,
            "result": self.result,
            "error": self.error,
        }


class EventBroker:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        item = {
            "event": event_type,
            "data": payload,
            "emitted_at": utc_now_iso(),
        }
        async with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(item)
                except asyncio.QueueFull:
                    pass


class CommandStore:
    def __init__(self, broker: EventBroker | None = None, database: SeasonalDatabase | None = None) -> None:
        self._broker = broker or EventBroker()
        self._lock = asyncio.Lock()
        self._commands_by_id: dict[str, CommandRecord] = {}
        self._commands_by_idempotency_key: dict[str, str] = {}
        self._repo = CommandRepository(database) if database is not None else None

    @property
    def broker(self) -> EventBroker:
        return self._broker

    @staticmethod
    def _payload_hash(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def _record_from_dict(raw: dict[str, Any]) -> CommandRecord:
        return CommandRecord(
            command_id=str(raw["command_id"]),
            command_type=str(raw["command_type"]),
            status=str(raw["status"]),
            accepted_at=str(raw["accepted_at"]),
            idempotency_key=str(raw["idempotency_key"]),
            actor=str(raw["actor"]),
            payload_hash=str(raw["payload_hash"]),
            payload=dict(raw.get("payload") or {}),
            request_id=str(raw.get("request_id") or f"req_{uuid.uuid4().hex[:16]}"),
            started_at=raw.get("started_at"),
            finished_at=raw.get("finished_at"),
            idempotent_replay_count=int(raw.get("idempotent_replay_count", 0) or 0),
            result=raw.get("result"),
            error=raw.get("error"),
        )

    def _persist_record(self, record: CommandRecord, *, insert: bool = False) -> None:
        if self._repo is None:
            return
        raw = asdict(record)
        if insert:
            self._repo.insert(raw)
        else:
            self._repo.update(raw)


    async def create_or_replay(
        self,
        *,
        command_type: str,
        idempotency_key: str,
        actor: str,
        payload: dict[str, Any],
    ) -> tuple[CommandRecord, bool]:
        payload_hash = self._payload_hash(payload)
        async with self._lock:
            existing_id = self._commands_by_idempotency_key.get(idempotency_key)
            existing: CommandRecord | None = None
            if existing_id is not None:
                existing = self._commands_by_id[existing_id]
            elif self._repo is not None:
                raw = self._repo.get_by_idempotency_key(idempotency_key)
                if raw is not None:
                    existing = self._record_from_dict(raw)
                    self._commands_by_id[existing.command_id] = existing
                    self._commands_by_idempotency_key[idempotency_key] = existing.command_id

            if existing is not None:
                if existing.payload_hash != payload_hash or existing.command_type != command_type:
                    raise IdempotencyConflictError("idempotency key was reused with a different request payload")
                existing.idempotent_replay_count += 1
                self._persist_record(existing)
                await self._broker.publish(
                    "command.replayed",
                    {
                        "command_id": existing.command_id,
                        "command_type": existing.command_type,
                        "request_id": existing.request_id,
                    },
                )
                return existing, True

            record = CommandRecord(
                command_id=f"cmd_{uuid.uuid4().hex[:20]}",
                command_type=command_type,
                status=CommandStatus.PENDING.value,
                accepted_at=utc_now_iso(),
                idempotency_key=idempotency_key,
                actor=actor,
                payload_hash=payload_hash,
                payload=payload,
            )
            self._commands_by_id[record.command_id] = record
            self._commands_by_idempotency_key[idempotency_key] = record.command_id
            self._persist_record(record, insert=True)

        await self._broker.publish(
            "command.accepted",
            {
                "command_id": record.command_id,
                "command_type": record.command_type,
                "request_id": record.request_id,
                "actor": record.actor,
            },
        )
        return record, False

    async def mark_running(self, command_id: str) -> CommandRecord:
        async with self._lock:
            record = self._commands_by_id.get(command_id)
            if record is None and self._repo is not None:
                raw = self._repo.get_by_command_id(command_id)
                if raw is not None:
                    record = self._record_from_dict(raw)
                    self._commands_by_id[command_id] = record
                    self._commands_by_idempotency_key[record.idempotency_key] = record.command_id
            if record is None:
                raise CommandNotFoundError(command_id)
            record.status = CommandStatus.RUNNING.value
            record.started_at = utc_now_iso()
            self._persist_record(record)
        await self._broker.publish(
            "command.running",
            {"command_id": record.command_id, "command_type": record.command_type, "request_id": record.request_id},
        )
        return record

    async def mark_succeeded(self, command_id: str, result: dict[str, Any]) -> CommandRecord:
        async with self._lock:
            record = self._commands_by_id.get(command_id)
            if record is None and self._repo is not None:
                raw = self._repo.get_by_command_id(command_id)
                if raw is not None:
                    record = self._record_from_dict(raw)
                    self._commands_by_id[command_id] = record
                    self._commands_by_idempotency_key[record.idempotency_key] = record.command_id
            if record is None:
                raise CommandNotFoundError(command_id)
            record.status = CommandStatus.SUCCEEDED.value
            record.finished_at = utc_now_iso()
            record.result = result
            record.error = None
            self._persist_record(record)
        await self._broker.publish(
            "command.succeeded",
            {
                "command_id": record.command_id,
                "command_type": record.command_type,
                "request_id": record.request_id,
                "result": result,
            },
        )
        return record

    async def mark_failed(self, command_id: str, error: dict[str, Any]) -> CommandRecord:
        async with self._lock:
            record = self._commands_by_id.get(command_id)
            if record is None and self._repo is not None:
                raw = self._repo.get_by_command_id(command_id)
                if raw is not None:
                    record = self._record_from_dict(raw)
                    self._commands_by_id[command_id] = record
                    self._commands_by_idempotency_key[record.idempotency_key] = record.command_id
            if record is None:
                raise CommandNotFoundError(command_id)
            record.status = CommandStatus.FAILED.value
            record.finished_at = utc_now_iso()
            record.error = error
            self._persist_record(record)
        await self._broker.publish(
            "command.failed",
            {
                "command_id": record.command_id,
                "command_type": record.command_type,
                "request_id": record.request_id,
                "error": error,
            },
        )
        return record

    async def get(self, command_id: str) -> CommandRecord:
        async with self._lock:
            record = self._commands_by_id.get(command_id)
            if record is None and self._repo is not None:
                raw = self._repo.get_by_command_id(command_id)
                if raw is not None:
                    record = self._record_from_dict(raw)
                    self._commands_by_id[command_id] = record
                    self._commands_by_idempotency_key[record.idempotency_key] = record.command_id
            if record is None:
                raise CommandNotFoundError(command_id)
            return record
