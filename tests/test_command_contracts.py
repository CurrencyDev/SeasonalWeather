from __future__ import annotations

import asyncio
import datetime as dt

import pytest
from pydantic import ValidationError

from seasonalweather.commands import (
    CommandAuditContext,
    CommandError,
    CommandRecord,
    CommandRelationshipPolicy,
    CommandResult,
    CommandStatus,
    CommandStore,
    CommandType,
    IdempotencyConflictError,
    RelationshipCompletion,
    request_command_cancellation,
    transition_command,
)
from seasonalweather.commands.contracts import CommandTransitionError
from seasonalweather.database.core import SeasonalDatabase
from seasonalweather.lifecycle import AdmissionClosedError, Lifecycle

NOW = dt.datetime(2026, 7, 24, 12, tzinfo=dt.UTC)


def _command(**updates: object) -> CommandRecord:
    values: dict[str, object] = {
        "command_id": "cmd_0123456789abcdef",
        "command_type": CommandType.CYCLE_REBUILD,
        "actor": "operator",
        "reason": "refresh",
        "idempotency_key": "request-1",
        "request_id": "req_0123456789abcdef",
        "payload_hash": "a" * 64,
        "created_at": NOW,
        "accepted_at": NOW,
        "audit_context": CommandAuditContext(channel="api"),
        "relationship": CommandRelationshipPolicy(
            completion=RelationshipCompletion.CONTROLLER_FINALIZATION,
            controller_finalization_required=True,
        ),
    }
    values.update(updates)
    return CommandRecord.model_validate(values)


def test_command_state_machine_is_immutable_and_idempotent() -> None:
    accepted = _command()
    running = transition_command(accepted, CommandStatus.RUNNING, at=NOW + dt.timedelta(seconds=1))
    result = CommandResult(code="cycle_ready", message="Cycle was rebuilt.")
    succeeded = transition_command(
        running,
        CommandStatus.SUCCEEDED,
        at=NOW + dt.timedelta(seconds=2),
        result=result,
    )

    assert accepted.status is CommandStatus.ACCEPTED
    assert running.status is CommandStatus.RUNNING
    assert succeeded.status is CommandStatus.SUCCEEDED
    assert transition_command(succeeded, CommandStatus.SUCCEEDED, at=NOW + dt.timedelta(seconds=3)) is succeeded
    with pytest.raises(CommandTransitionError):
        transition_command(succeeded, CommandStatus.FAILED, at=NOW + dt.timedelta(seconds=3))


def test_command_cancellation_request_is_distinct_from_completion() -> None:
    command = request_command_cancellation(_command(), at=NOW + dt.timedelta(seconds=1))
    assert command.status is CommandStatus.ACCEPTED
    assert command.cancel_requested_at == NOW + dt.timedelta(seconds=1)

    cancelled = transition_command(command, CommandStatus.CANCELLED, at=NOW + dt.timedelta(seconds=2))
    assert cancelled.status is CommandStatus.CANCELLED
    with pytest.raises(CommandTransitionError):
        request_command_cancellation(cancelled, at=NOW + dt.timedelta(seconds=3))


def test_command_validation_rejects_naive_time_secret_material_and_unbounded_results() -> None:
    with pytest.raises(ValidationError):
        _command(created_at=NOW.replace(tzinfo=None))
    with pytest.raises(ValidationError):
        _command(actor="Bearer top-secret-value")
    with pytest.raises(ValidationError):
        CommandResult(code="bad", message="bad", details={"authorization": "secret"})
    with pytest.raises(ValidationError):
        CommandResult(code="bad", message="bad", details={"items": ["x"] * 33})


def test_command_relationships_distinguish_required_optional_and_finalization() -> None:
    policy = CommandRelationshipPolicy(
        completion=RelationshipCompletion.ALL_REQUIRED_JOBS,
        required_job_ids=("job_required",),
        optional_job_ids=("job_optional",),
    )
    assert policy.required_job_ids == ("job_required",)
    with pytest.raises(ValidationError):
        CommandRelationshipPolicy(
            completion=RelationshipCompletion.ALL_REQUIRED_JOBS,
            required_job_ids=("job_same",),
            optional_job_ids=("job_same",),
        )


def test_command_store_persists_typed_shape_without_raw_payload(tmp_path) -> None:
    async def exercise() -> None:
        database = SeasonalDatabase(path=str(tmp_path / "commands.sqlite3"))
        store = CommandStore(database=database, clock=lambda: NOW)
        record, replayed = await store.create_or_replay(
            command_type=CommandType.ORIGINATE_TEXT.value,
            idempotency_key="request-typed",
            actor="operator",
            reason="bounded test",
            payload={"text": "raw synthesis text must not be stored"},
        )
        assert replayed is False
        assert "payload" not in record.snapshot()

        with database.connect() as connection:
            row = connection.execute(
                "SELECT status, payload_json, reason, created_at FROM api_commands WHERE command_id = ?",
                (record.command_id,),
            ).fetchone()
        assert row is not None
        assert tuple(row[:3]) == ("accepted", "{}", "bounded test")
        assert dt.datetime.fromisoformat(row[3].replace("Z", "+00:00")) == NOW

        replay, replayed = await store.create_or_replay(
            command_type=CommandType.ORIGINATE_TEXT.value,
            idempotency_key="request-typed",
            actor="operator",
            payload={"text": "raw synthesis text must not be stored"},
        )
        assert replayed is True
        assert replay.idempotent_replay_count == 1

        with pytest.raises(IdempotencyConflictError):
            await store.create_or_replay(
                command_type=CommandType.ORIGINATE_TEXT.value,
                idempotency_key="request-typed",
                actor="operator",
                payload={"text": "different"},
            )

    asyncio.run(exercise())


def test_command_store_uses_lifecycle_admission_gate() -> None:
    async def exercise() -> None:
        lifecycle = Lifecycle()
        lifecycle.mark_running()
        store = CommandStore(lifecycle=lifecycle, clock=lambda: NOW)
        await store.create_or_replay(
            command_type=CommandType.CYCLE_REBUILD.value,
            idempotency_key="before-drain",
            actor="operator",
            payload={},
        )
        lifecycle.request_shutdown()
        with pytest.raises(AdmissionClosedError):
            await store.create_or_replay(
                command_type=CommandType.CYCLE_REBUILD.value,
                idempotency_key="after-drain",
                actor="operator",
                payload={},
            )

    asyncio.run(exercise())


def test_failed_command_requires_typed_bounded_error() -> None:
    error = CommandError(code="validation_failed", message="Candidate was invalid.")
    failed = transition_command(_command(), CommandStatus.FAILED, at=NOW, error=error)
    assert failed.error == error
    with pytest.raises(CommandTransitionError):
        transition_command(_command(), CommandStatus.FAILED, at=NOW)
