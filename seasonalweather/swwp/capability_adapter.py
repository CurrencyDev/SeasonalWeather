"""Narrow conversion between SWWP capability payloads and domain records."""

from __future__ import annotations

from ..capabilities.manifest import CapabilityManifest, CapabilityUpdate
from ..capabilities.models import (
    CapabilityRecord,
    CompatibilityState,
    DependencyState,
    OperationalState,
)
from .messages import (
    CapabilityDependencyState,
    CapabilityOperationalState,
    CapabilityRecordPayload,
)
from .messages import CapabilityManifest as WireCapabilityManifest
from .messages import CapabilityUpdate as WireCapabilityUpdate


def record_from_wire(payload: CapabilityRecordPayload) -> CapabilityRecord:
    return CapabilityRecord(
        name=payload.name,
        implemented=payload.implemented,
        compatibility=CompatibilityState.UNKNOWN,
        operational_state=OperationalState(payload.operational_state.value),
        accepting_new_jobs=payload.accepting_new_jobs,
        total_capacity=payload.total_capacity,
        reported_available=payload.reported_available,
        job_restrictions=payload.job_restrictions,
        parameters=payload.parameters,
        validity_seconds=payload.validity_seconds,
        observed_at=payload.observed_at,
        published_at=payload.published_at,
        dependency_health={key: DependencyState(value.value) for key, value in payload.dependency_health.items()},
    )


def record_to_wire(record: CapabilityRecord) -> CapabilityRecordPayload:
    return CapabilityRecordPayload(
        name=record.name,
        implemented=record.implemented,
        operational_state=CapabilityOperationalState(record.operational_state.value),
        accepting_new_jobs=record.accepting_new_jobs,
        total_capacity=record.total_capacity,
        reported_available=record.reported_available,
        job_restrictions=record.job_restrictions,
        parameters=record.parameters,
        validity_seconds=record.validity_seconds,
        observed_at=record.observed_at,
        published_at=record.published_at,
        dependency_health={
            key: CapabilityDependencyState(value.value) for key, value in record.dependency_health.items()
        },
    )


def manifest_from_wire(payload: WireCapabilityManifest) -> CapabilityManifest:
    return CapabilityManifest(
        schema_version=payload.schema_version,
        epoch=payload.epoch,
        records=tuple(record_from_wire(item) for item in payload.records),
        digest=payload.digest,
    )


def update_from_wire(payload: WireCapabilityUpdate) -> CapabilityUpdate:
    return CapabilityUpdate(
        epoch=payload.epoch,
        changed=tuple(record_from_wire(item) for item in payload.changed),
        removed=payload.removed,
        resulting_digest=payload.full_digest,
        validity_seconds=payload.validity_seconds,
    )
