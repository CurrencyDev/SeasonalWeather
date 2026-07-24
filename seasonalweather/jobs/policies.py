from __future__ import annotations

import re
from enum import IntEnum, StrEnum
from typing import Any, Self

from ..validation.modeling import BaseModel, ConfigDict, Field, field_validator, model_validator

_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,63}$")
_DEDUPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,191}$")


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class JobType(StrEnum):
    SEGMENT_BUILD = "routine.segment.build"
    TTS_SYNTHESIZE = "routine.tts.synthesize"
    AUDIO_CONVERT = "routine.audio.convert"
    CYCLE_REGENERATE = "routine.cycle.regenerate"
    MAINTENANCE_RECONCILE = "maintenance.reconcile"
    CONFIG_VALIDATE = "control.config.validate"
    CONFIG_COMMIT = "control.config.commit"
    ALERT_ARTIFACT_GENERATE = "alert.artifact.generate"


class QueueClass(StrEnum):
    ROUTINE = "routine"
    MAINTENANCE = "maintenance"
    CONTROL = "control"


class ExecutorClass(StrEnum):
    ROUTINE_WORKER = "routine_worker"
    MAINTENANCE_WORKER = "maintenance_worker"
    CONTROLLER = "controller"


class JobPriority(IntEnum):
    SAFETY_CRITICAL = 0
    HIGH = 10
    NORMAL = 20
    LOW = 30


class FailureCategory(StrEnum):
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    TRANSIENT_TRANSPORT = "transient_transport"
    TIMED_OUT = "timed_out"
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED = "unsupported"
    SIDE_EFFECT_UNCERTAIN = "side_effect_uncertain"
    CANCELLED = "cancelled"


class BackoffStrategy(StrEnum):
    NONE = "none"
    FIXED = "fixed"
    EXPONENTIAL = "exponential"


class ReplayPolicy(StrEnum):
    NEVER = "never"
    REVALIDATE = "authoritative_revalidation"
    IDEMPOTENT_FENCED = "idempotent_all_fences"


class DedupeMode(StrEnum):
    NONE = "none"
    EXACT = "exact"
    COALESCE_LATEST = "coalesce_latest"


class CancellationMode(StrEnum):
    BEFORE_START = "before_start"
    COOPERATIVE = "cooperative"
    CONTROLLER_FENCED = "controller_fenced"


class ConfigFence(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    NOT_APPLICABLE = "not_applicable"


class FinalCommitAuthority(StrEnum):
    CONTROLLER = "controller"
    COMMAND_SERVICE = "command_service"


class RetryPolicy(PolicyModel):
    max_attempts: int = Field(ge=1, le=10)
    attempt_timeout_seconds: int = Field(ge=1, le=3600)
    retryable_categories: frozenset[FailureCategory] = Field(default_factory=frozenset, max_length=8)
    backoff_strategy: BackoffStrategy = BackoffStrategy.NONE
    initial_backoff_seconds: int = Field(default=0, ge=0, le=600)
    maximum_backoff_seconds: int = Field(default=0, ge=0, le=3600)
    new_attempt_identity_required: bool = True

    @model_validator(mode="after")
    def validate_backoff(self) -> Self:
        if self.backoff_strategy is BackoffStrategy.NONE:
            if self.initial_backoff_seconds or self.maximum_backoff_seconds:
                raise ValueError("no-backoff retry policy cannot declare backoff values")
        elif self.initial_backoff_seconds < 1 or self.maximum_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("bounded backoff requires ordered positive limits")
        if FailureCategory.SIDE_EFFECT_UNCERTAIN in self.retryable_categories:
            raise ValueError("uncertain side effects cannot be retried blindly")
        return self


class DeadlinePolicy(PolicyModel):
    required: bool
    default_seconds: int | None = Field(default=None, ge=1, le=86400)

    @model_validator(mode="after")
    def validate_deadline(self) -> Self:
        if not self.required and self.default_seconds is None:
            raise ValueError("job policy must require a deadline or provide a bounded default")
        return self


class DedupePolicy(PolicyModel):
    mode: DedupeMode
    scope: str = Field(min_length=2, max_length=64)
    supersedes_older: bool = False

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, value: str) -> str:
        if not _CAPABILITY_RE.fullmatch(value):
            raise ValueError("dedupe scope must be a bounded declared key")
        return value

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if self.mode is DedupeMode.NONE and self.supersedes_older:
            raise ValueError("non-deduplicated work cannot supersede by dedupe policy")
        return self


class CapabilityRequirement(PolicyModel):
    name: str = Field(min_length=2, max_length=64)
    required: bool = True
    parameters: dict[str, str | int | bool] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _CAPABILITY_RE.fullmatch(value) or "*" in value:
            raise ValueError("capability name must be a stable bounded key")
        return value

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 16:
            raise ValueError("capability parameters are bounded")
        for key, item in value.items():
            if not _CAPABILITY_RE.fullmatch(key) or "*" in key:
                raise ValueError("capability parameter name must be bounded")
            if isinstance(item, str) and len(item) > 128:
                raise ValueError("capability parameter value is overlong")
        return dict(sorted(value.items()))


class FenceRequirements(PolicyModel):
    config_generation: ConfigFence
    source_identity: bool
    event_identity: bool
    content_identity: bool


class CommandRelationship(StrEnum):
    OPTIONAL = "optional_child"
    REQUIRED = "required_child"
    REQUIRED_WITH_CONTROLLER_FINALIZATION = "required_with_controller_finalization"
    INTERNAL_ONLY = "internal_only"


def validate_dedupe_key(value: str) -> str:
    normalized = value.strip()
    if not _DEDUPE_RE.fullmatch(normalized):
        raise ValueError("dedupe key must be canonical and bounded")
    return normalized


def queue_executor_compatible(queue: QueueClass, executor: ExecutorClass) -> bool:
    return {
        QueueClass.ROUTINE: ExecutorClass.ROUTINE_WORKER,
        QueueClass.MAINTENANCE: ExecutorClass.MAINTENANCE_WORKER,
        QueueClass.CONTROL: ExecutorClass.CONTROLLER,
    }[queue] is executor
