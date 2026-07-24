"""Canonical complete manifests, epochs, digests, and atomic partial updates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

from .models import CapabilityRecord

MANIFEST_SCHEMA_VERSION = 1


class EpochDisposition(StrEnum):
    ACCEPTED = "accepted"
    IDEMPOTENT = "idempotent"
    STALE = "stale"
    CONFLICT = "conflict"
    GAP = "gap"


@dataclass(frozen=True)
class CapabilityManifest:
    schema_version: int
    epoch: int
    records: tuple[CapabilityRecord, ...]
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported capability manifest schema")
        if self.epoch < 1:
            raise ValueError("capability epoch must be positive")
        names = tuple(record.name for record in self.records)
        if names != tuple(sorted(set(names))):
            raise ValueError("capability records must be unique and sorted")
        if self.digest != manifest_digest(
            schema_version=self.schema_version,
            records=self.records,
        ):
            raise ValueError("capability manifest digest does not match records")

    @classmethod
    def create(
        cls,
        *,
        epoch: int,
        records: tuple[CapabilityRecord, ...],
        schema_version: int = MANIFEST_SCHEMA_VERSION,
    ) -> CapabilityManifest:
        normalized = tuple(sorted(records, key=lambda item: item.name))
        return cls(
            schema_version=schema_version,
            epoch=epoch,
            records=normalized,
            digest=manifest_digest(schema_version=schema_version, records=normalized),
        )

    def by_name(self) -> dict[str, CapabilityRecord]:
        return {record.name: record for record in self.records}


@dataclass(frozen=True)
class CapabilityUpdate:
    epoch: int
    changed: tuple[CapabilityRecord, ...]
    removed: tuple[str, ...]
    resulting_digest: str
    validity_seconds: int

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise ValueError("capability epoch must be positive")
        if not 1 <= self.validity_seconds <= 900:
            raise ValueError("update validity is out of bounds")
        names = tuple(item.name for item in self.changed)
        if names != tuple(sorted(set(names))):
            raise ValueError("changed capability records must be unique and sorted")
        if self.removed != tuple(sorted(set(self.removed))):
            raise ValueError("removed capability names must be unique and sorted")
        if set(names).intersection(self.removed):
            raise ValueError("capability cannot be changed and removed together")
        _validate_digest(self.resulting_digest)


@dataclass(frozen=True)
class ManifestApplyResult:
    disposition: EpochDisposition
    manifest: CapabilityManifest | None
    requires_full_report: bool


def _record_json(record: CapabilityRecord) -> dict[str, object]:
    return record.model_dump(mode="json", exclude={"compatibility"})


def canonical_manifest_bytes(
    *,
    schema_version: int,
    records: tuple[CapabilityRecord, ...],
) -> bytes:
    payload = {
        "records": [_record_json(item) for item in sorted(records, key=lambda item: item.name)],
        "schema_version": schema_version,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def manifest_digest(
    *,
    schema_version: int,
    records: tuple[CapabilityRecord, ...],
) -> str:
    return (
        f"sha256:{hashlib.sha256(canonical_manifest_bytes(schema_version=schema_version, records=records)).hexdigest()}"
    )


def _validate_digest(value: str) -> None:
    if len(value) != 71 or not value.startswith("sha256:"):
        raise ValueError("capability digest must use sha256")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ValueError("capability digest must contain lowercase hexadecimal") from exc
    if value[7:] != value[7:].lower():
        raise ValueError("capability digest must contain lowercase hexadecimal")


def compare_epoch(
    current: CapabilityManifest,
    *,
    epoch: int,
    digest: str,
) -> EpochDisposition:
    _validate_digest(digest)
    if epoch < current.epoch:
        return EpochDisposition.STALE
    if epoch == current.epoch:
        return EpochDisposition.IDEMPOTENT if digest == current.digest else EpochDisposition.CONFLICT
    if epoch != current.epoch + 1:
        return EpochDisposition.GAP
    return EpochDisposition.ACCEPTED


def apply_update(
    current: CapabilityManifest,
    update: CapabilityUpdate,
) -> ManifestApplyResult:
    disposition = compare_epoch(
        current,
        epoch=update.epoch,
        digest=update.resulting_digest,
    )
    if disposition is not EpochDisposition.ACCEPTED:
        return ManifestApplyResult(
            disposition=disposition,
            manifest=current if disposition is EpochDisposition.IDEMPOTENT else None,
            requires_full_report=disposition in {EpochDisposition.CONFLICT, EpochDisposition.GAP},
        )
    records = current.by_name()
    for name in update.removed:
        records.pop(name, None)
    for record in update.changed:
        records[record.name] = record
    normalized = tuple(sorted(records.values(), key=lambda item: item.name))
    digest = manifest_digest(schema_version=current.schema_version, records=normalized)
    if digest != update.resulting_digest:
        return ManifestApplyResult(
            disposition=EpochDisposition.CONFLICT,
            manifest=None,
            requires_full_report=True,
        )
    return ManifestApplyResult(
        disposition=EpochDisposition.ACCEPTED,
        manifest=CapabilityManifest(
            schema_version=current.schema_version,
            epoch=update.epoch,
            records=normalized,
            digest=digest,
        ),
        requires_full_report=False,
    )
