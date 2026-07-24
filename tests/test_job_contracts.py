from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from seasonalweather.jobs.contracts import (
    AttemptOutcome,
    JobError,
    JobRecord,
    JobResult,
    JobStatus,
    JobTransitionError,
    ReplayEvidence,
    expire_job,
    lease_job,
    request_job_cancellation,
    resolve_attempt,
    should_replay,
    start_job,
    supersede_job,
)
from seasonalweather.jobs.policies import (
    ExecutorClass,
    FailureCategory,
    JobPriority,
    JobType,
    QueueClass,
    ReplayPolicy,
)
from seasonalweather.jobs.registry import policy_for

NOW = dt.datetime(2026, 7, 24, 12, tzinfo=dt.UTC)


def _job(**updates: object) -> JobRecord:
    policy = policy_for(JobType.TTS_SYNTHESIZE)
    values: dict[str, object] = {
        "job_id": "job_0123456789abcdef",
        "command_id": "cmd_0123456789abcdef",
        "job_type": JobType.TTS_SYNTHESIZE,
        "queue": QueueClass.ROUTINE,
        "executor": ExecutorClass.ROUTINE_WORKER,
        "priority": JobPriority.NORMAL,
        "payload_schema_version": 1,
        "result_schema_version": 1,
        "payload": {
            "content_ref": "content_01234567",
            "voice_profile_ref": "voice_01234567",
            "output_format": "wav",
            "config_generation": 4,
        },
        "created_at": NOW,
        "not_before": NOW,
        "deadline_at": NOW + dt.timedelta(minutes=3),
        "max_attempts": policy.retry.max_attempts,
        "dedupe_key": "tts:content_01234567",
        "config_generation": 4,
        "replay_policy": policy.replay,
    }
    values.update(updates)
    return JobRecord.model_validate(values)


def _leased() -> JobRecord:
    return lease_job(
        _job(),
        attempt_id="attempt_0001",
        lease_owner="worker_0001",
        at=NOW,
        lease_expires_at=NOW + dt.timedelta(seconds=30),
    )


def test_job_lease_start_and_success_are_attempt_fenced() -> None:
    leased = _leased()
    running = start_job(leased, attempt_id="attempt_0001", lease_owner="worker_0001", at=NOW)
    result = JobResult(code="artifact_ready", result_ref="artifact_01234567")
    succeeded = resolve_attempt(
        running,
        attempt_id="attempt_0001",
        lease_owner="worker_0001",
        outcome=AttemptOutcome.SUCCEEDED,
        at=NOW + dt.timedelta(seconds=1),
        retry_policy=policy_for(JobType.TTS_SYNTHESIZE).retry,
        result=result,
    )

    assert leased.attempt == 1
    assert running.status is JobStatus.RUNNING
    assert succeeded.status is JobStatus.SUCCEEDED
    with pytest.raises(JobTransitionError):
        resolve_attempt(
            running,
            attempt_id="attempt_stale",
            lease_owner="worker_0001",
            outcome=AttemptOutcome.SUCCEEDED,
            at=NOW,
            retry_policy=policy_for(JobType.TTS_SYNTHESIZE).retry,
            result=result,
        )


def test_retry_returns_to_pending_and_next_attempt_is_monotonic() -> None:
    leased = _leased()
    error = JobError(
        category=FailureCategory.TRANSIENT_TRANSPORT,
        code="transport_unavailable",
        message="Dependency was unavailable.",
    )
    pending = resolve_attempt(
        leased,
        attempt_id="attempt_0001",
        lease_owner="worker_0001",
        outcome=AttemptOutcome.RETRYABLE_FAILURE,
        at=NOW,
        retry_policy=policy_for(JobType.TTS_SYNTHESIZE).retry,
        error=error,
    )
    assert pending.status is JobStatus.PENDING
    assert pending.attempt == 1
    leased_again = lease_job(
        pending,
        attempt_id="attempt_0002",
        lease_owner="worker_0002",
        at=pending.not_before,
        lease_expires_at=pending.not_before + dt.timedelta(seconds=30),
    )
    assert leased_again.attempt == 2


def test_timeout_is_not_automatically_retryable_and_attempts_exhaust() -> None:
    policy = policy_for(JobType.TTS_SYNTHESIZE).retry
    failed = resolve_attempt(
        _leased(),
        attempt_id="attempt_0001",
        lease_owner="worker_0001",
        outcome=AttemptOutcome.TIMED_OUT,
        at=NOW + dt.timedelta(seconds=2),
        retry_policy=policy,
        error=JobError(category=FailureCategory.TIMED_OUT, code="attempt_timeout", message="Timed out."),
    )
    assert failed.status is JobStatus.FAILED


def test_cancellation_request_completion_expiry_and_supersession_are_distinct() -> None:
    requested = request_job_cancellation(_leased())
    cancelled = resolve_attempt(
        requested,
        attempt_id="attempt_0001",
        lease_owner="worker_0001",
        outcome=AttemptOutcome.CANCELLED,
        at=NOW,
        retry_policy=policy_for(JobType.TTS_SYNTHESIZE).retry,
    )
    assert cancelled.status is JobStatus.CANCELLED
    assert supersede_job(_job(), at=NOW).status is JobStatus.SUPERSEDED
    assert expire_job(_job(), at=NOW + dt.timedelta(minutes=3)).status is JobStatus.EXPIRED
    with pytest.raises(JobTransitionError):
        supersede_job(cancelled, at=NOW)


def test_replay_requires_policy_specific_authoritative_evidence() -> None:
    common = {
        "before_deadline": True,
        "command_active": True,
        "not_superseded": True,
        "prior_result_commit_certain_absent": True,
    }
    assert should_replay(
        ReplayPolicy.REVALIDATE,
        ReplayEvidence(authoritative_revalidation=True, **common),
    )
    assert not should_replay(ReplayPolicy.NEVER, ReplayEvidence(authoritative_revalidation=True, **common))
    assert not should_replay(
        ReplayPolicy.IDEMPOTENT_FENCED,
        ReplayEvidence(operation_idempotent=True, config_generation_matches=True, **common),
    )


def test_job_contract_rejects_naive_times_invalid_dedupe_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _job(created_at=NOW.replace(tzinfo=None))
    with pytest.raises(ValidationError):
        _job(dedupe_key="raw payload with spaces")
    with pytest.raises(ValidationError):
        _job(unknown="value")


def test_job_snapshot_is_deterministic_json_compatible() -> None:
    snapshot = _job().snapshot()
    restored = JobRecord.model_validate(snapshot)
    assert restored == _job()
