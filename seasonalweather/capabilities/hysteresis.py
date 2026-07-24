"""Deterministic worker observation-to-publication hysteresis."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass

from .models import CapabilityRecord, OperationalState


@dataclass(frozen=True)
class HysteresisPolicy:
    failure_threshold: int = 3
    recovery_success_threshold: int = 2
    minimum_degraded_dwell_seconds: int = 5
    minimum_unavailable_dwell_seconds: int = 10
    publication_debounce_seconds: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.failure_threshold <= 20:
            raise ValueError("failure threshold is out of bounds")
        if not 1 <= self.recovery_success_threshold <= 20:
            raise ValueError("recovery threshold is out of bounds")
        for value in (
            self.minimum_degraded_dwell_seconds,
            self.minimum_unavailable_dwell_seconds,
            self.publication_debounce_seconds,
        ):
            if not 0 <= value <= 3600:
                raise ValueError("hysteresis duration is out of bounds")


@dataclass(frozen=True)
class CapabilityObservation:
    successful: bool
    hard_failure: bool = False
    state: OperationalState | None = None
    accepting_new_jobs: bool | None = None
    available_capacity: int | None = None

    def __post_init__(self) -> None:
        if self.state is OperationalState.UNKNOWN:
            raise ValueError("unknown is controller-owned freshness state")
        if self.available_capacity is not None and not 0 <= self.available_capacity <= 128:
            raise ValueError("observed capacity is out of bounds")
        if self.hard_failure and self.successful:
            raise ValueError("hard failure cannot be successful")


class CapabilityHysteresis:
    """Bounded state for one worker-instance capability."""

    def __init__(
        self,
        record: CapabilityRecord,
        *,
        policy: HysteresisPolicy | None = None,
        clock: Callable[[], dt.datetime],
    ) -> None:
        self._record = record
        self._policy = policy or HysteresisPolicy()
        self._clock = clock
        self._failures = 0
        self._successes = 0
        self._state_since = record.published_at
        self._last_publication = record.published_at

    @property
    def current(self) -> CapabilityRecord:
        return self._record

    def observe(self, observation: CapabilityObservation) -> CapabilityRecord | None:
        now = self._clock().astimezone(dt.UTC)
        self._record_observation(observation)

        target = self._target_state(observation, now)
        accepting, available = self._published_values(observation, target)
        if not (
            target is not self._record.operational_state
            or accepting != self._record.accepting_new_jobs
            or available != self._record.reported_available
        ):
            return None
        elapsed = (now - self._last_publication).total_seconds()
        if not self._immediate(observation, target, accepting, available) and (
            elapsed < self._policy.publication_debounce_seconds
        ):
            return None
        return self._publish(target, accepting, available, now)

    def _record_observation(self, observation: CapabilityObservation) -> None:
        if observation.successful:
            self._successes = min(self._successes + 1, self._policy.recovery_success_threshold)
            self._failures = 0
            return
        self._failures = min(self._failures + 1, self._policy.failure_threshold)
        self._successes = 0

    def _published_values(
        self,
        observation: CapabilityObservation,
        target: OperationalState,
    ) -> tuple[bool, int]:
        accepting = (
            observation.accepting_new_jobs
            if observation.accepting_new_jobs is not None
            else self._record.accepting_new_jobs
        )
        available = observation.available_capacity
        if available is None:
            available = self._record.reported_available
        inactive = {
            OperationalState.UNAVAILABLE,
            OperationalState.DRAINING,
            OperationalState.DISABLED,
        }
        return (False if target in inactive else accepting), min(available, self._record.total_capacity)

    def _immediate(
        self,
        observation: CapabilityObservation,
        target: OperationalState,
        accepting: bool,
        available: int,
    ) -> bool:
        return any(
            (
                observation.hard_failure,
                target in {OperationalState.DRAINING, OperationalState.DISABLED},
                available < self._record.reported_available,
                self._record.accepting_new_jobs and not accepting,
            )
        )

    def _publish(
        self,
        target: OperationalState,
        accepting: bool,
        available: int,
        now: dt.datetime,
    ) -> CapabilityRecord:
        if target is not self._record.operational_state:
            self._state_since = now
        self._record = self._record.model_copy(
            update={
                "operational_state": target,
                "accepting_new_jobs": accepting,
                "reported_available": available,
                "observed_at": now,
                "published_at": now,
            }
        )
        self._last_publication = now
        return self._record

    def _target_state(
        self,
        observation: CapabilityObservation,
        now: dt.datetime,
    ) -> OperationalState:
        explicit = observation.state
        if explicit in {OperationalState.DRAINING, OperationalState.DISABLED}:
            return explicit
        if observation.hard_failure:
            return explicit or OperationalState.UNAVAILABLE
        current = self._record.operational_state
        if not observation.successful:
            return current if self._failures < self._policy.failure_threshold else explicit or OperationalState.DEGRADED
        if self._successes < self._policy.recovery_success_threshold:
            return current
        dwell = (now - self._state_since).total_seconds()
        dwell_required = {
            OperationalState.DEGRADED: self._policy.minimum_degraded_dwell_seconds,
            OperationalState.UNAVAILABLE: self._policy.minimum_unavailable_dwell_seconds,
        }.get(current, 0)
        if dwell < dwell_required:
            return current
        return explicit or OperationalState.HEALTHY
