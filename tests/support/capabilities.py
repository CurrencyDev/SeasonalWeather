from __future__ import annotations

import datetime as dt
from typing import Any

from seasonalweather.capabilities.manifest import manifest_digest
from seasonalweather.swwp.capability_adapter import record_from_wire
from seasonalweather.swwp.messages import (
    CapabilityManifest,
    CapabilityOperationalState,
    CapabilityRecordPayload,
)


def wire_record(
    name: str,
    *,
    now: dt.datetime,
    state: CapabilityOperationalState = CapabilityOperationalState.HEALTHY,
    accepting: bool = True,
    total: int = 2,
    available: int = 2,
    parameters: dict[str, Any] | None = None,
    validity_seconds: int = 60,
) -> CapabilityRecordPayload:
    return CapabilityRecordPayload(
        name=name,
        implemented=True,
        operational_state=state,
        accepting_new_jobs=accepting,
        total_capacity=total,
        reported_available=available,
        parameters=parameters or {},
        validity_seconds=validity_seconds,
        observed_at=now,
        published_at=now,
    )


def wire_manifest(
    records: tuple[CapabilityRecordPayload, ...],
    *,
    epoch: int = 1,
) -> CapabilityManifest:
    normalized = tuple(sorted(records, key=lambda item: item.name))
    return CapabilityManifest(
        schema_version=1,
        epoch=epoch,
        digest=manifest_digest(
            schema_version=1,
            records=tuple(record_from_wire(item) for item in normalized),
        ),
        records=normalized,
    )
