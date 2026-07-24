"""Immutable bounded capability records shared by qualification components."""

from __future__ import annotations

import datetime as dt
import re
from enum import StrEnum
from typing import Any, Self, TypeAlias, cast

from ..validation.modeling import BaseModel, ConfigDict, Field, field_validator, model_validator

_KEY_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,63}$")
_MAX_STRING = 128
_MAX_COLLECTION = 32
_PARAMETER_NAMES = frozenset(
    {
        "channels",
        "extensions",
        "feature_extensions",
        "format",
        "job_classes",
        "max_input_bytes",
        "max_output_bytes",
        "media_types",
        "profiles",
        "sample_rates",
        "schema_versions",
        "voices",
    }
)

ParameterScalar: TypeAlias = str | int | float | bool
ParameterValue: TypeAlias = ParameterScalar | tuple[str, ...] | tuple[int, ...] | tuple[float, ...] | tuple[bool, ...]


class CapabilityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class OperationalState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DRAINING = "draining"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


class CompatibilityState(StrEnum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class DependencyState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


def _utc(value: dt.datetime, name: str) -> dt.datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(dt.UTC)


def capability_key(value: str, name: str = "capability name") -> str:
    if not _KEY_RE.fullmatch(value) or "*" in value:
        raise ValueError(f"{name} must be a stable bounded key")
    return value


def _bounded_integer(value: int) -> int:
    if not -(2**31) <= value <= 2**31 - 1:
        raise ValueError("capability parameter integer is out of bounds")
    return value


def _bounded_number(value: float) -> float:
    if value != value or value in {float("inf"), float("-inf")}:
        raise ValueError("capability parameter number must be finite")
    if not (-1_000_000_000.0 <= value <= 1_000_000_000.0):
        raise ValueError("capability parameter number is out of bounds")
    return value


def _bounded_string(value: str) -> str:
    if not value or len(value) > _MAX_STRING:
        raise ValueError("capability parameter string is empty or overlong")
    if "\x00" in value or value.startswith(("/", "\\", "http:", "https:")):
        raise ValueError("capability parameters cannot contain paths or URLs")
    return value


def _normalize_scalar(value: Any) -> ParameterScalar:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return _bounded_integer(value)
    if isinstance(value, float):
        return _bounded_number(value)
    if isinstance(value, str):
        return _bounded_string(value)
    raise ValueError("capability parameters must be JSON scalar values")


def _validate_capacity(record: CapabilityRecord) -> None:
    if record.reported_available > record.total_capacity:
        raise ValueError("reported available capacity cannot exceed total capacity")
    if not record.implemented and (record.accepting_new_jobs or record.total_capacity or record.reported_available):
        raise ValueError("unimplemented capability cannot accept work or report capacity")
    if record.accepting_new_jobs and record.total_capacity == 0:
        raise ValueError("accepting capability requires positive total capacity")


def _validate_admission(record: CapabilityRecord) -> None:
    inactive = {
        OperationalState.UNAVAILABLE,
        OperationalState.DRAINING,
        OperationalState.DISABLED,
        OperationalState.UNKNOWN,
    }
    if record.operational_state in inactive and record.accepting_new_jobs:
        raise ValueError("inactive capability state cannot accept new work")
    if (
        record.operational_state is OperationalState.DEGRADED
        and record.accepting_new_jobs
        and record.reported_available == 0
    ):
        raise ValueError("degraded accepting capability requires available capacity")


def normalize_parameter_value(value: Any) -> ParameterValue:
    if isinstance(value, (list, tuple, frozenset, set)):
        if not value or len(value) > _MAX_COLLECTION:
            raise ValueError("capability parameter collection is empty or overlong")
        normalized = tuple(_normalize_scalar(item) for item in value)
        kinds = {bool if isinstance(item, bool) else type(item) for item in normalized}
        if len(kinds) != 1:
            raise ValueError("capability parameter collections must be homogeneous")
        ordered = tuple(
            sorted(
                set(normalized),
                key=lambda item: (type(item).__name__, repr(item)),
            )
        )
        return cast(ParameterValue, ordered)
    return _normalize_scalar(value)


def normalize_parameters(value: dict[str, Any]) -> dict[str, ParameterValue]:
    if len(value) > 16:
        raise ValueError("capability parameters are bounded")
    normalized: dict[str, ParameterValue] = {}
    for key, item in value.items():
        capability_key(key, "capability parameter name")
        if key not in _PARAMETER_NAMES:
            raise ValueError(f"unknown capability parameter: {key}")
        normalized[key] = normalize_parameter_value(item)
    return dict(sorted(normalized.items()))


class CapabilityRecord(CapabilityModel):
    name: str = Field(min_length=2, max_length=64)
    implemented: bool
    compatibility: CompatibilityState = CompatibilityState.UNKNOWN
    operational_state: OperationalState
    accepting_new_jobs: bool
    total_capacity: int = Field(ge=0, le=128)
    reported_available: int = Field(ge=0, le=128)
    job_restrictions: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    parameters: dict[str, ParameterValue] = Field(default_factory=dict, max_length=16)
    validity_seconds: int = Field(ge=1, le=900)
    observed_at: dt.datetime
    published_at: dt.datetime
    dependency_health: dict[str, DependencyState] = Field(default_factory=dict, max_length=8)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return capability_key(value)

    @field_validator("observed_at", "published_at")
    @classmethod
    def validate_time(cls, value: dt.datetime, info: Any) -> dt.datetime:
        return _utc(value, info.field_name)

    @field_validator("job_restrictions")
    @classmethod
    def validate_restrictions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({capability_key(item, "job restriction") for item in value}))

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, Any]) -> dict[str, ParameterValue]:
        return normalize_parameters(value)

    @field_validator("dependency_health")
    @classmethod
    def validate_dependencies(cls, value: dict[str, DependencyState]) -> dict[str, DependencyState]:
        normalized = {capability_key(key, "dependency name"): state for key, state in value.items()}
        return dict(sorted(normalized.items()))

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        _validate_capacity(self)
        _validate_admission(self)
        if self.observed_at > self.published_at:
            raise ValueError("observed time cannot follow published time")
        return self

    @property
    def state_capacity(self) -> int:
        if (
            not self.implemented
            or self.compatibility is not CompatibilityState.COMPATIBLE
            or not self.accepting_new_jobs
            or self.operational_state not in {OperationalState.HEALTHY, OperationalState.DEGRADED}
        ):
            return 0
        return self.reported_available

    def unknown(self) -> CapabilityRecord:
        return self.model_copy(
            update={
                "operational_state": OperationalState.UNKNOWN,
                "accepting_new_jobs": False,
            }
        )
