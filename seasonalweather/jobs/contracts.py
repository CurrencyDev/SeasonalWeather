from __future__ import annotations

import datetime as dt
import re
from enum import StrEnum
from typing import Any, Self

from ..validation.modeling import BaseModel, ConfigDict, Field, field_validator, model_validator
from .policies import (
    ExecutorClass,
    FailureCategory,
    JobPriority,
    JobType,
    QueueClass,
    ReplayPolicy,
    RetryPolicy,
    validate_dedupe_key,
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_CODE_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,63}$")


class JobContractError(ValueError):
    pass


class JobTransitionError(JobContractError):
    pass


class JobStatus(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


TERMINAL_JOB_STATUSES = frozenset(
    {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.EXPIRED,
        JobStatus.SUPERSEDED,
    }
)


class AttemptOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    RETRYABLE_FAILURE = "retryable_failure"
    PERMANENT_FAILURE = "permanent_failure"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    LOST = "lost"


class JobModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


def _utc(value: dt.datetime | None, name: str) -> dt.datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(dt.UTC)


def _required_utc(value: dt.datetime, name: str) -> dt.datetime:
    normalized = _utc(value, name)
    if normalized is None:
        raise ValueError(f"{name} is required")
    return normalized


def _identifier(value: str, name: str) -> str:
    if not _ID_RE.fullmatch(value):
        raise ValueError(f"{name} must be a bounded opaque identifier")
    return value


class JobResult(JobModel):
    code: str = Field(min_length=2, max_length=64)
    result_ref: str
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if not _CODE_RE.fullmatch(value):
            raise ValueError("result code must be a bounded declared key")
        return value

    @field_validator("result_ref")
    @classmethod
    def validate_result_ref(cls, value: str) -> str:
        return _identifier(value, "result_ref")

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 16:
            raise ValueError("result metadata is bounded")
        for key, item in value.items():
            if not _CODE_RE.fullmatch(key):
                raise ValueError("result metadata keys must be bounded")
            if isinstance(item, str) and len(item) > 256:
                raise ValueError("result metadata value is overlong")
        return dict(sorted(value.items()))


class JobError(JobModel):
    category: FailureCategory
    code: str = Field(min_length=2, max_length=64)
    message: str = Field(min_length=1, max_length=512)

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if not _CODE_RE.fullmatch(value):
            raise ValueError("error code must be a bounded declared key")
        return value


class JobRecord(JobModel):
    job_id: str
    command_id: str | None = None
    job_type: JobType
    queue: QueueClass
    executor: ExecutorClass
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority
    payload_schema_version: int = Field(ge=1, le=255)
    result_schema_version: int = Field(ge=1, le=255)
    payload: dict[str, Any]
    created_at: dt.datetime
    not_before: dt.datetime
    deadline_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    attempt: int = Field(default=0, ge=0, le=10)
    attempt_id: str | None = None
    max_attempts: int = Field(ge=1, le=10)
    dedupe_key: str | None = None
    config_generation: int | None = Field(default=None, ge=0)
    replay_policy: ReplayPolicy
    cancel_requested: bool = False
    lease_owner: str | None = None
    lease_expires_at: dt.datetime | None = None
    result: JobResult | None = None
    error: JobError | None = None

    @field_validator("job_id", "command_id", "attempt_id", "lease_owner")
    @classmethod
    def validate_ids(cls, value: str | None, info: Any) -> str | None:
        return _identifier(value, info.field_name) if value is not None else None

    @field_validator("dedupe_key")
    @classmethod
    def validate_dedupe(cls, value: str | None) -> str | None:
        return validate_dedupe_key(value) if value is not None else None

    @field_validator("created_at", "not_before", "deadline_at", "started_at", "finished_at", "lease_expires_at")
    @classmethod
    def validate_timestamp(cls, value: dt.datetime | None, info: Any) -> dt.datetime | None:
        return _utc(value, info.field_name)

    @field_validator("payload")
    @classmethod
    def validate_payload_shape(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 32:
            raise ValueError("job payload is bounded")
        return value

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        _validate_job_schedule(self)
        _validate_job_attempt_shape(self)
        _validate_job_terminal_shape(self)
        return self

    def snapshot(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ReplayEvidence(JobModel):
    authoritative_revalidation: bool = False
    operation_idempotent: bool = False
    config_generation_matches: bool = False
    source_identity_matches: bool = False
    event_identity_matches: bool = False
    content_identity_matches: bool = False
    before_deadline: bool = False
    command_active: bool = False
    not_superseded: bool = False
    prior_result_commit_certain_absent: bool = False


def _validate_job_schedule(job: JobRecord) -> None:
    if job.not_before < job.created_at:
        raise ValueError("not_before cannot precede created_at")
    if job.deadline_at <= job.created_at or job.deadline_at < job.not_before:
        raise ValueError("deadline must follow admission and not_before")
    if job.started_at is not None and job.started_at < job.created_at:
        raise ValueError("started_at cannot precede created_at")


def _validate_job_attempt_shape(job: JobRecord) -> None:
    if job.attempt > job.max_attempts:
        raise ValueError("attempt cannot exceed max_attempts")
    identified = all(
        (
            job.attempt >= 1,
            job.attempt_id is not None,
            job.lease_owner is not None,
            job.lease_expires_at is not None,
        )
    )
    if job.status in {JobStatus.LEASED, JobStatus.RUNNING} and not identified:
        raise ValueError("leased/running jobs require a positive identified attempt and lease")
    if job.status is JobStatus.RUNNING and job.started_at is None:
        raise ValueError("running jobs require started_at")


def _validate_job_terminal_shape(job: JobRecord) -> None:
    if (job.status in TERMINAL_JOB_STATUSES) != (job.finished_at is not None):
        raise ValueError("finished_at must be present exactly for terminal jobs")
    if job.status is JobStatus.SUCCEEDED and job.result is None:
        raise ValueError("succeeded jobs require a typed result")
    if job.status is JobStatus.FAILED and job.error is None:
        raise ValueError("failed jobs require a typed error")


def should_replay(policy: ReplayPolicy, evidence: ReplayEvidence) -> bool:
    common = all(
        (
            evidence.before_deadline,
            evidence.command_active,
            evidence.not_superseded,
            evidence.prior_result_commit_certain_absent,
        )
    )
    if policy is ReplayPolicy.NEVER:
        return False
    if policy is ReplayPolicy.REVALIDATE:
        return common and evidence.authoritative_revalidation
    return common and all(
        (
            evidence.operation_idempotent,
            evidence.config_generation_matches,
            evidence.source_identity_matches,
            evidence.event_identity_matches,
            evidence.content_identity_matches,
        )
    )


def _replace(job: JobRecord, **updates: Any) -> JobRecord:
    return JobRecord.model_validate(job.model_dump() | updates)


def _validate_attempt(job: JobRecord, *, attempt_id: str, lease_owner: str) -> None:
    if job.attempt_id != attempt_id or job.lease_owner != lease_owner:
        raise JobTransitionError("stale attempt or lease owner")


def lease_job(
    job: JobRecord,
    *,
    attempt_id: str,
    lease_owner: str,
    lease_expires_at: dt.datetime,
    at: dt.datetime,
) -> JobRecord:
    if job.status in TERMINAL_JOB_STATUSES:
        raise JobTransitionError("terminal job status is immutable")
    if job.status is not JobStatus.PENDING:
        raise JobTransitionError("only pending jobs may be leased")
    now = _required_utc(at, "lease timestamp")
    expires = _required_utc(lease_expires_at, "lease_expires_at")
    if now < job.not_before:
        raise JobTransitionError("job is not eligible before not_before")
    if now >= job.deadline_at:
        raise JobTransitionError("job deadline has expired")
    if expires <= now or expires > job.deadline_at:
        raise JobTransitionError("lease must be positive and bounded by the deadline")
    if job.attempt >= job.max_attempts:
        raise JobTransitionError("maximum attempts are exhausted")
    return _replace(
        job,
        status=JobStatus.LEASED,
        attempt=job.attempt + 1,
        attempt_id=_identifier(attempt_id, "attempt_id"),
        lease_owner=_identifier(lease_owner, "lease_owner"),
        lease_expires_at=expires,
        started_at=None,
        error=None,
    )


def start_job(job: JobRecord, *, attempt_id: str, lease_owner: str, at: dt.datetime) -> JobRecord:
    if job.status is not JobStatus.LEASED:
        raise JobTransitionError("only leased jobs may start")
    _validate_attempt(job, attempt_id=attempt_id, lease_owner=lease_owner)
    now = _required_utc(at, "start timestamp")
    if job.cancel_requested:
        raise JobTransitionError("cancel-requested job cannot start")
    if now >= job.deadline_at:
        raise JobTransitionError("job deadline has expired")
    if job.lease_expires_at is None or now >= job.lease_expires_at:
        raise JobTransitionError("lease expired before start")
    return _replace(job, status=JobStatus.RUNNING, started_at=now)


def request_job_cancellation(job: JobRecord) -> JobRecord:
    if job.status in TERMINAL_JOB_STATUSES:
        raise JobTransitionError("terminal job cancellation cannot be requested")
    if job.cancel_requested:
        return job
    return _replace(job, cancel_requested=True)


def supersede_job(job: JobRecord, *, at: dt.datetime) -> JobRecord:
    if job.status in TERMINAL_JOB_STATUSES:
        raise JobTransitionError("terminal job status is immutable")
    now = _required_utc(at, "supersession timestamp")
    return _replace(job, status=JobStatus.SUPERSEDED, finished_at=now)


def expire_job(job: JobRecord, *, at: dt.datetime) -> JobRecord:
    if job.status in TERMINAL_JOB_STATUSES:
        raise JobTransitionError("terminal job status is immutable")
    now = _required_utc(at, "expiration timestamp")
    if now < job.deadline_at:
        raise JobTransitionError("job cannot expire before its deadline")
    return _replace(job, status=JobStatus.EXPIRED, finished_at=now)


def _retry_delay(policy: RetryPolicy, attempt: int) -> int:
    if policy.backoff_strategy.value == "none":
        return 0
    if policy.backoff_strategy.value == "fixed":
        return policy.initial_backoff_seconds
    delay = policy.initial_backoff_seconds * (2 ** max(0, attempt - 1))
    return int(min(delay, policy.maximum_backoff_seconds))


def resolve_attempt(
    job: JobRecord,
    *,
    attempt_id: str,
    lease_owner: str,
    outcome: AttemptOutcome,
    at: dt.datetime,
    retry_policy: RetryPolicy,
    result: JobResult | None = None,
    error: JobError | None = None,
    replay_permitted: bool = False,
) -> JobRecord:
    if job.status not in {JobStatus.LEASED, JobStatus.RUNNING}:
        raise JobTransitionError("attempt resolution requires leased or running state")
    _validate_attempt(job, attempt_id=attempt_id, lease_owner=lease_owner)
    now = _required_utc(at, "attempt completion timestamp")
    resolved = _resolve_success_or_cancellation(job, outcome=outcome, at=now, result=result, error=error)
    if resolved is not None:
        return resolved

    failure = error or JobError(
        category=FailureCategory.TIMED_OUT if outcome is AttemptOutcome.TIMED_OUT else FailureCategory.UNSUPPORTED,
        code="attempt_failed",
        message="Job attempt did not complete.",
    )
    can_retry = all(
        (
            _outcome_allows_retry(outcome, failure, retry_policy, replay_permitted),
            job.attempt < min(job.max_attempts, retry_policy.max_attempts),
            not job.cancel_requested,
        )
    )
    not_before = now + dt.timedelta(seconds=_retry_delay(retry_policy, job.attempt))
    if can_retry and not_before < job.deadline_at:
        return _replace(
            job,
            status=JobStatus.PENDING,
            not_before=not_before,
            started_at=None,
            attempt_id=None,
            lease_owner=None,
            lease_expires_at=None,
            error=failure,
        )
    terminal = JobStatus.EXPIRED if now >= job.deadline_at else JobStatus.FAILED
    return _replace(job, status=terminal, finished_at=now, error=failure)


def _resolve_success_or_cancellation(
    job: JobRecord,
    *,
    outcome: AttemptOutcome,
    at: dt.datetime,
    result: JobResult | None,
    error: JobError | None,
) -> JobRecord | None:
    if outcome is AttemptOutcome.SUCCEEDED:
        if result is None:
            raise JobTransitionError("successful attempt requires a typed result")
        return _replace(job, status=JobStatus.SUCCEEDED, finished_at=at, result=result, error=None)
    if outcome is AttemptOutcome.CANCELLED:
        if not job.cancel_requested:
            raise JobTransitionError("cancelled outcome requires a cancellation request")
        return _replace(job, status=JobStatus.CANCELLED, finished_at=at, error=error)
    return None


def _outcome_allows_retry(
    outcome: AttemptOutcome,
    failure: JobError,
    policy: RetryPolicy,
    replay_permitted: bool,
) -> bool:
    if failure.category not in policy.retryable_categories:
        return False
    if outcome is AttemptOutcome.LOST:
        return replay_permitted
    if outcome is AttemptOutcome.TIMED_OUT:
        return FailureCategory.TIMED_OUT in policy.retryable_categories
    return outcome is AttemptOutcome.RETRYABLE_FAILURE
