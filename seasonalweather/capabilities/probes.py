"""Bounded full and targeted capability probe correlation."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum

from .models import capability_key


class ProbeMode(StrEnum):
    FULL = "full"
    TARGETED = "targeted"


class ProbeReason(StrEnum):
    REGISTRATION = "registration"
    RECONNECT = "reconnect"
    PERIODIC = "periodic"
    EPOCH_GAP = "epoch_gap"
    DIGEST_MISMATCH = "digest_mismatch"
    STALE = "stale"
    JOB_REJECTED = "job_rejected"
    INTERNAL = "internal"


def _validate_probe_identity(probe: CapabilityProbe) -> None:
    if not 3 <= len(probe.probe_id) <= 128:
        raise ValueError("probe identity is out of bounds")
    if probe.requested_at.tzinfo is None or probe.deadline_at.tzinfo is None:
        raise ValueError("probe timestamps must be timezone-aware")
    if probe.deadline_at <= probe.requested_at:
        raise ValueError("probe deadline must follow request time")


def _validate_probe_targets(probe: CapabilityProbe) -> None:
    normalized = tuple(sorted({capability_key(item) for item in probe.target_names}))
    if normalized != probe.target_names or len(normalized) > 64:
        raise ValueError("probe targets must be unique, sorted, and bounded")
    if probe.mode is ProbeMode.FULL and probe.target_names:
        raise ValueError("full probe cannot declare targets")
    if probe.mode is ProbeMode.TARGETED and not probe.target_names:
        raise ValueError("targeted probe requires targets")


@dataclass(frozen=True)
class CapabilityProbe:
    probe_id: str
    mode: ProbeMode
    target_names: tuple[str, ...]
    reason: ProbeReason
    requested_at: dt.datetime
    deadline_at: dt.datetime
    session_id: str
    worker_instance_id: str

    def __post_init__(self) -> None:
        _validate_probe_identity(self)
        _validate_probe_targets(self)


@dataclass(frozen=True)
class ProbeMatch:
    accepted: bool
    reason: str
    probe: CapabilityProbe | None


class ProbeTracker:
    def __init__(self, *, maximum_outstanding: int = 16) -> None:
        self._maximum = max(1, min(maximum_outstanding, 64))
        self._outstanding: dict[str, CapabilityProbe] = {}

    def add(self, probe: CapabilityProbe) -> None:
        if probe.probe_id in self._outstanding:
            if self._outstanding[probe.probe_id] != probe:
                raise ValueError("conflicting probe identity")
            return
        if len(self._outstanding) >= self._maximum:
            raise OverflowError("too many outstanding capability probes")
        self._outstanding[probe.probe_id] = probe

    def match(
        self,
        probe_id: str,
        *,
        session_id: str,
        worker_instance_id: str,
        now: dt.datetime,
        consume: bool = True,
    ) -> ProbeMatch:
        probe = self._outstanding.get(probe_id)
        if probe is None:
            return ProbeMatch(False, "unsolicited_probe_response", None)
        if probe.session_id != session_id or probe.worker_instance_id != worker_instance_id:
            return ProbeMatch(False, "probe_identity_mismatch", probe)
        if now >= probe.deadline_at:
            self._outstanding.pop(probe_id, None)
            return ProbeMatch(False, "probe_expired", probe)
        if consume:
            self._outstanding.pop(probe_id, None)
        return ProbeMatch(True, "probe_correlated", probe)

    def expire(self, now: dt.datetime) -> tuple[CapabilityProbe, ...]:
        expired = tuple(
            sorted(
                (item for item in self._outstanding.values() if now >= item.deadline_at),
                key=lambda item: item.probe_id,
            )
        )
        for item in expired:
            self._outstanding.pop(item.probe_id, None)
        return expired

    def close(self) -> tuple[CapabilityProbe, ...]:
        retained = self.snapshot()
        self._outstanding.clear()
        return retained

    def snapshot(self) -> tuple[CapabilityProbe, ...]:
        return tuple(sorted(self._outstanding.values(), key=lambda item: item.probe_id))
