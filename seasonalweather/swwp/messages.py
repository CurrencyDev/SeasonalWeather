"""Strict typed SWWP/1 envelope and message payloads."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, ClassVar, Self

from ..jobs.contracts import AttemptOutcome
from ..jobs.policies import ExecutorClass, FailureCategory, JobType, QueueClass
from ..validation.modeling import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)
from .constants import PROTOCOL_NAME, PROTOCOL_VERSION, ProtocolErrorCategory, ReconcileDisposition

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_KEY_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,63}$")


def _utc(value: dt.datetime, name: str) -> dt.datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(dt.UTC)


def _identifier(value: str, name: str) -> str:
    if not _ID_RE.fullmatch(value):
        raise ValueError(f"{name} must be a bounded opaque identifier")
    return value


def _key(value: str, name: str) -> str:
    if not _KEY_RE.fullmatch(value):
        raise ValueError(f"{name} must be a bounded declared key")
    return value


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Payload(WireModel):
    message_type: ClassVar[str]


class VersionSupport(WireModel):
    swwp: tuple[int, ...] = Field(min_length=1, max_length=8)
    job_payloads: dict[JobType, tuple[int, ...]] = Field(default_factory=dict, max_length=16)
    job_results: dict[JobType, tuple[int, ...]] = Field(default_factory=dict, max_length=16)
    diagnostics: tuple[int, ...] = Field(min_length=1, max_length=8)
    capability_manifest: tuple[int, ...] = Field(min_length=1, max_length=8)
    configuration_schema: tuple[int, ...] = Field(min_length=1, max_length=8)

    @field_validator(
        "swwp",
        "diagnostics",
        "capability_manifest",
        "configuration_schema",
    )
    @classmethod
    def validate_versions(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(item < 1 or item > 255 for item in value):
            raise ValueError("schema versions must be between 1 and 255")
        if len(set(value)) != len(value):
            raise ValueError("schema versions must be unique")
        return tuple(sorted(value))

    @field_validator("job_payloads", "job_results")
    @classmethod
    def validate_job_versions(cls, value: dict[JobType, tuple[int, ...]]) -> dict[JobType, tuple[int, ...]]:
        normalized: dict[JobType, tuple[int, ...]] = {}
        for job_type, versions in value.items():
            if not versions or any(item < 1 or item > 255 for item in versions):
                raise ValueError("job schema versions must be non-empty and bounded")
            if len(set(versions)) != len(versions):
                raise ValueError("job schema versions must be unique")
            normalized[job_type] = tuple(sorted(versions))
        return dict(sorted(normalized.items(), key=lambda item: item[0].value))


class SelectedVersions(WireModel):
    swwp: int = Field(ge=1, le=255)
    job_payloads: dict[JobType, int] = Field(default_factory=dict, max_length=16)
    job_results: dict[JobType, int] = Field(default_factory=dict, max_length=16)
    diagnostics: int = Field(ge=1, le=255)
    capability_manifest: int = Field(ge=1, le=255)
    configuration_schema: int = Field(ge=1, le=255)


class CapabilityManifest(WireModel):
    schema_version: int = Field(ge=1, le=255)
    epoch: int = Field(ge=0)
    digest: str = Field(min_length=3, max_length=128)
    names: tuple[str, ...] = Field(default_factory=tuple, max_length=64)

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _identifier(value, "capability digest")

    @field_validator("names")
    @classmethod
    def validate_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _KEY_RE.fullmatch(item) for item in value):
            raise ValueError("capability names must be bounded declared keys")
        return tuple(sorted(set(value)))


class LeaseRef(WireModel):
    job_id: str
    lease_id: str
    attempt_id: str
    attempt: int = Field(ge=1, le=10)

    @field_validator("job_id", "lease_id", "attempt_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _identifier(value, info.field_name)


class Register(Payload):
    message_type = "register"
    worker_id: str
    worker_instance_id: str
    worker_epoch: int = Field(ge=1)
    software_version: str = Field(min_length=1, max_length=64)
    build_identity: str = Field(min_length=1, max_length=128)
    requested_queues: tuple[QueueClass, ...] = Field(min_length=1, max_length=8)
    requested_slots: int = Field(ge=1, le=128)
    capability_manifest: CapabilityManifest
    supported_versions: VersionSupport

    @field_validator("worker_id", "worker_instance_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _identifier(value, info.field_name)


class Registered(Payload):
    message_type = "registered"
    session_id: str
    controller_epoch: int = Field(ge=1)
    selected_subprotocol: str = Field(min_length=1, max_length=128)
    heartbeat_interval_seconds: int = Field(ge=1, le=300)
    heartbeat_timeout_seconds: int = Field(ge=2, le=900)
    lease_seconds: int = Field(ge=1, le=3600)
    assignment_ack_seconds: int = Field(ge=1, le=300)
    accepted_queues: tuple[QueueClass, ...] = Field(max_length=8)
    authorized_job_types: tuple[JobType, ...] = Field(max_length=16)
    authorized_capabilities: tuple[str, ...] = Field(max_length=64)
    selected_versions: SelectedVersions
    max_message_bytes: int = Field(ge=1024, le=16_777_216)
    max_active_assignments: int = Field(ge=1, le=128)

    @field_validator("session_id")
    @classmethod
    def validate_session(cls, value: str) -> str:
        return _identifier(value, "session_id")


class RegistrationRejected(Payload):
    message_type = "registration_rejected"
    category: ProtocolErrorCategory
    summary: str = Field(min_length=1, max_length=256)
    supported_subprotocols: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    supported_swwp_versions: tuple[int, ...] = Field(default_factory=tuple, max_length=8)


class Heartbeat(Payload):
    message_type = "heartbeat"
    active_leases: tuple[LeaseRef, ...] = Field(default_factory=tuple, max_length=32)
    capability_epoch: int | None = Field(default=None, ge=0)
    capability_digest: str | None = Field(default=None, max_length=128)


class HeartbeatAck(Payload):
    message_type = "heartbeat_ack"
    renewed: tuple[LeaseRef, ...] = Field(default_factory=tuple, max_length=32)
    reconcile: tuple[LeaseRef, ...] = Field(default_factory=tuple, max_length=32)


class CapabilityUpdate(Payload):
    message_type = "capability_update"
    manifest: CapabilityManifest


class CapabilityUpdateAck(Payload):
    message_type = "capability_update_ack"
    epoch: int = Field(ge=0)
    digest: str = Field(min_length=3, max_length=128)


class CapabilityProbe(Payload):
    message_type = "capability_probe"
    probe_id: str
    names: tuple[str, ...] = Field(default_factory=tuple, max_length=64)

    _probe_id = field_validator("probe_id")(lambda value: _identifier(value, "probe_id"))


class CapabilityReport(Payload):
    message_type = "capability_report"
    probe_id: str
    manifest: CapabilityManifest
    evidence: dict[str, str | int | float | bool] = Field(default_factory=dict, max_length=32)

    _probe_id = field_validator("probe_id")(lambda value: _identifier(value, "probe_id"))


class JobAssignmentPayload(Payload):
    message_type = "job"
    lease: LeaseRef
    deadline_at: dt.datetime
    lease_expires_at: dt.datetime
    acknowledgment_deadline_at: dt.datetime
    job_type: JobType
    queue: QueueClass
    executor: ExecutorClass
    payload_schema_version: int = Field(ge=1, le=255)
    result_schema_version: int = Field(ge=1, le=255)
    configuration_generation: int | None = Field(default=None, ge=0)
    payload: dict[str, Any] = Field(max_length=32)
    capability_requirements: tuple[str, ...] = Field(default_factory=tuple, max_length=64)

    @field_validator("deadline_at", "lease_expires_at", "acknowledgment_deadline_at")
    @classmethod
    def validate_times(cls, value: dt.datetime, info: Any) -> dt.datetime:
        return _utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_timing(self) -> Self:
        if not (self.acknowledgment_deadline_at <= self.lease_expires_at <= self.deadline_at):
            raise ValueError("assignment deadlines must be ordered")
        return self


class JobAccepted(Payload):
    message_type = "job_accepted"
    lease: LeaseRef


class JobRejected(Payload):
    message_type = "job_rejected"
    lease: LeaseRef
    category: str = Field(min_length=2, max_length=64)
    summary: str = Field(min_length=1, max_length=256)

    _category = field_validator("category")(lambda value: _key(value, "category"))


class JobProgress(Payload):
    message_type = "job_progress"
    lease: LeaseRef
    stage: str = Field(min_length=2, max_length=64)
    reason: str | None = Field(default=None, max_length=64)
    numeric: dict[str, int | float] = Field(default_factory=dict, max_length=16)

    _stage = field_validator("stage")(lambda value: _key(value, "stage"))


class JobResult(Payload):
    message_type = "job_result"
    lease: LeaseRef
    result_schema_version: int = Field(ge=1, le=255)
    result: dict[str, Any] = Field(max_length=32)
    artifact_refs: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    completion_id: str

    _completion_id = field_validator("completion_id")(lambda value: _identifier(value, "completion_id"))

    @field_validator("artifact_refs")
    @classmethod
    def validate_artifacts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_identifier(item, "artifact_ref") for item in value)


class JobFailed(Payload):
    message_type = "job_failed"
    lease: LeaseRef
    outcome: AttemptOutcome
    category: FailureCategory
    error_code: str = Field(min_length=2, max_length=64)
    summary: str = Field(min_length=1, max_length=256)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict, max_length=16)

    _error_code = field_validator("error_code")(lambda value: _key(value, "error_code"))

    @model_validator(mode="after")
    def validate_failure(self) -> Self:
        if self.outcome is AttemptOutcome.SUCCEEDED:
            raise ValueError("job_failed cannot report success")
        return self


class Cancel(Payload):
    message_type = "cancel"
    lease: LeaseRef
    reason: str = Field(min_length=2, max_length=64)
    deadline_at: dt.datetime

    _reason = field_validator("reason")(lambda value: _key(value, "reason"))
    _deadline = field_validator("deadline_at")(lambda value: _utc(value, "deadline_at"))


class CancelAcknowledged(Payload):
    message_type = "cancel_acknowledged"
    lease: LeaseRef
    observed_at: dt.datetime

    _observed = field_validator("observed_at")(lambda value: _utc(value, "observed_at"))


class Drain(Payload):
    message_type = "drain"
    deadline_at: dt.datetime
    reason: str = Field(min_length=2, max_length=64)

    _deadline = field_validator("deadline_at")(lambda value: _utc(value, "deadline_at"))
    _reason = field_validator("reason")(lambda value: _key(value, "reason"))


class Drained(Payload):
    message_type = "drained"
    active: tuple[LeaseRef, ...] = Field(default_factory=tuple, max_length=32)
    unacknowledged_completions: tuple[str, ...] = Field(default_factory=tuple, max_length=32)


class ReconcileItem(WireModel):
    lease: LeaseRef
    prior_session_id: str | None = None
    accepted: bool = False
    cancellation_observed: bool = False
    completion_id: str | None = None
    result_schema_version: int | None = Field(default=None, ge=1, le=255)
    result: dict[str, Any] | None = Field(default=None, max_length=32)

    @field_validator("prior_session_id", "completion_id")
    @classmethod
    def validate_optional_ids(cls, value: str | None, info: Any) -> str | None:
        return _identifier(value, info.field_name) if value is not None else None


class Reconcile(Payload):
    message_type = "reconcile"
    prior_session_id: str | None = None
    prior_controller_epoch: int | None = Field(default=None, ge=1)
    items: tuple[ReconcileItem, ...] = Field(default_factory=tuple, max_length=64)

    @field_validator("prior_session_id")
    @classmethod
    def validate_prior_session(cls, value: str | None) -> str | None:
        return _identifier(value, "prior_session_id") if value is not None else None


class ReconcileDecision(WireModel):
    lease: LeaseRef
    disposition: ReconcileDisposition
    summary: str = Field(min_length=1, max_length=256)


class ReconcileResult(Payload):
    message_type = "reconcile_result"
    decisions: tuple[ReconcileDecision, ...] = Field(default_factory=tuple, max_length=64)


class ResultCommitted(Payload):
    message_type = "result_committed"
    lease: LeaseRef
    completion_id: str
    result_hash: str = Field(min_length=16, max_length=128)
    committed_at: dt.datetime

    _completion_id = field_validator("completion_id")(lambda value: _identifier(value, "completion_id"))
    _committed = field_validator("committed_at")(lambda value: _utc(value, "committed_at"))


class ProtocolErrorPayload(Payload):
    message_type = "protocol_error"
    category: ProtocolErrorCategory
    summary: str = Field(min_length=1, max_length=256)
    correlated_message_id: str | None = None
    fatal: bool

    @field_validator("correlated_message_id")
    @classmethod
    def validate_correlation(cls, value: str | None) -> str | None:
        return _identifier(value, "correlated_message_id") if value is not None else None


PAYLOAD_TYPES: tuple[type[Payload], ...] = (
    Register,
    Registered,
    RegistrationRejected,
    Heartbeat,
    HeartbeatAck,
    CapabilityUpdate,
    CapabilityUpdateAck,
    CapabilityProbe,
    CapabilityReport,
    JobAssignmentPayload,
    JobAccepted,
    JobRejected,
    JobProgress,
    JobResult,
    JobFailed,
    Cancel,
    CancelAcknowledged,
    Drain,
    Drained,
    Reconcile,
    ReconcileResult,
    ResultCommitted,
    ProtocolErrorPayload,
)
PAYLOAD_BY_TYPE = {model.message_type: model for model in PAYLOAD_TYPES}


class Envelope(WireModel):
    protocol: str = PROTOCOL_NAME
    protocol_version: int = PROTOCOL_VERSION
    message_type: str = Field(min_length=2, max_length=64)
    message_id: str
    sent_at: dt.datetime
    session_id: str | None = None
    worker_id: str | None = None
    worker_instance_id: str | None = None
    controller_epoch: int | None = Field(default=None, ge=1)
    worker_epoch: int | None = Field(default=None, ge=1)
    payload: SerializeAsAny[Payload]

    @field_validator("message_id", "session_id", "worker_id", "worker_instance_id")
    @classmethod
    def validate_ids(cls, value: str | None, info: Any) -> str | None:
        return _identifier(value, info.field_name) if value is not None else None

    @field_validator("sent_at")
    @classmethod
    def validate_sent_at(cls, value: dt.datetime) -> dt.datetime:
        return _utc(value, "sent_at")

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.protocol != PROTOCOL_NAME:
            raise ValueError("unsupported protocol identity")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("unsupported SWWP wire version")
        if self.message_type != self.payload.message_type:
            raise ValueError("envelope message_type does not match payload")
        return self
