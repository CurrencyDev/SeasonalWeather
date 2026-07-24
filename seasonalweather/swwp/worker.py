"""Deterministic worker-side SWWP/1 session state machine without execution."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any

from ..jobs.contracts import AttemptOutcome
from ..jobs.policies import FailureCategory
from .constants import ProtocolErrorCategory, WorkerState
from .messages import (
    Cancel,
    CancelAcknowledged,
    Drain,
    Drained,
    Envelope,
    Heartbeat,
    JobAccepted,
    JobAssignmentPayload,
    JobFailed,
    JobProgress,
    JobRejected,
    JobResult,
    LeaseRef,
    ProtocolErrorPayload,
    Reconcile,
    ReconcileItem,
    ReconcileResult,
    Register,
    Registered,
    RegistrationRejected,
    ResultCommitted,
)
from .session import SessionMachine

_TERMINAL = {WorkerState.CLOSED, WorkerState.FAILED}


class WorkerSession(SessionMachine):
    def __init__(
        self,
        *,
        registration: Register,
        id_factory: Callable[[str], str],
        clock: Callable[[], dt.datetime],
        accept_assignments: bool = True,
    ) -> None:
        super().__init__(clock=clock, id_factory=id_factory)
        self.registration = registration
        self.accept_assignments = accept_assignments
        self.state = WorkerState.DISCONNECTED
        self.session_id: str | None = None
        self.controller_epoch: int | None = None
        self.assignments: dict[tuple[str, str, str, int], JobAssignmentPayload] = {}
        self.completions: dict[tuple[str, str, str, int], JobResult] = {}
        self.cancelled: set[tuple[str, str, str, int]] = set()

    def _out(self, payload: object) -> Envelope:
        return self.envelope(
            payload,  # type: ignore[arg-type]
            session_id=self.session_id,
            worker_id=self.registration.worker_id,
            worker_instance_id=self.registration.worker_instance_id,
            controller_epoch=self.controller_epoch,
            worker_epoch=self.registration.worker_epoch,
        )

    def connect(self) -> Envelope:
        if self.state is not WorkerState.DISCONNECTED:
            raise ValueError("worker can connect only while disconnected")
        self.state = WorkerState.REGISTERING
        return self._out(self.registration)

    def receive(self, incoming: Envelope) -> tuple[Envelope, ...]:
        if self.state in _TERMINAL:
            return ()
        try:
            replay = self.replay(incoming)
            if replay is not None:
                return replay
            responses = self._receive(incoming)
        except ValueError:
            self.state = WorkerState.FAILED
            responses = (
                self._out(
                    ProtocolErrorPayload(
                        category=ProtocolErrorCategory.STATE_VIOLATION,
                        summary="controller message is invalid for worker session state",
                        correlated_message_id=incoming.message_id,
                        fatal=True,
                    )
                ),
            )
        self.remember(incoming, responses)
        return responses

    def _receive(self, incoming: Envelope) -> tuple[Envelope, ...]:
        payload = incoming.payload
        if self.state is WorkerState.REGISTERING:
            return self._registration_response(payload)
        self._validate_session(incoming)
        handlers: dict[type[object], Callable[[Any], tuple[Envelope, ...]]] = {
            JobAssignmentPayload: self._assignment,
            Cancel: self._cancel,
            Drain: self._drain,
            ResultCommitted: self._result_committed,
            ReconcileResult: self._reconciled,
            ProtocolErrorPayload: self._protocol_error,
        }
        handler = handlers.get(type(payload))
        return handler(payload) if handler is not None else ()

    def _registration_response(self, payload: object) -> tuple[Envelope, ...]:
        if isinstance(payload, RegistrationRejected):
            self.state = WorkerState.CLOSED
            return ()
        if not isinstance(payload, Registered):
            raise ValueError("registered must be first controller message")
        if payload.selected_subprotocol != "seasonalweather.worker.v1":
            raise ValueError("controller selected unexpected subprotocol")
        self.session_id = payload.session_id
        self.controller_epoch = payload.controller_epoch
        self.state = WorkerState.ACTIVE
        return ()

    def _assignment(self, payload: JobAssignmentPayload) -> tuple[Envelope, ...]:
        if self.state is not WorkerState.ACTIVE:
            raise ValueError("assignment received while not active")
        key = self._key(payload.lease)
        prior = self.assignments.get(key)
        if prior is not None and prior != payload:
            raise ValueError("conflicting duplicate assignment")
        if not self.accept_assignments:
            return (
                self._out(
                    JobRejected(
                        lease=payload.lease,
                        category="capacity_unavailable",
                        summary="simulated worker rejected assignment",
                    )
                ),
            )
        self.assignments[key] = payload
        return (self._out(JobAccepted(lease=payload.lease)),)

    def _cancel(self, payload: Cancel) -> tuple[Envelope, ...]:
        self.cancelled.add(self._key(payload.lease))
        return (self._out(CancelAcknowledged(lease=payload.lease, observed_at=self.clock())),)

    def _drain(self, _: Drain) -> tuple[Envelope, ...]:
        self.state = WorkerState.DRAINING
        return (
            self._out(
                Drained(
                    active=tuple(item.lease for item in self.assignments.values()),
                    unacknowledged_completions=tuple(result.completion_id for result in self.completions.values()),
                )
            ),
        )

    def _result_committed(self, payload: ResultCommitted) -> tuple[Envelope, ...]:
        key = self._key(payload.lease)
        retained = self.completions.get(key)
        if retained is None or retained.completion_id != payload.completion_id:
            raise ValueError("result commitment does not match retained completion")
        del self.completions[key]
        self.assignments.pop(key, None)
        return ()

    def _reconciled(self, _: ReconcileResult) -> tuple[Envelope, ...]:
        self.state = WorkerState.ACTIVE
        return ()

    def _protocol_error(self, payload: ProtocolErrorPayload) -> tuple[Envelope, ...]:
        if payload.fatal:
            self.state = WorkerState.FAILED
        return ()

    def _validate_session(self, incoming: Envelope) -> None:
        if (
            incoming.session_id != self.session_id
            or incoming.controller_epoch != self.controller_epoch
            or incoming.worker_id != self.registration.worker_id
            or incoming.worker_instance_id != self.registration.worker_instance_id
            or incoming.worker_epoch != self.registration.worker_epoch
        ):
            raise ValueError("stale session identity")

    @staticmethod
    def _key(lease: LeaseRef) -> tuple[str, str, str, int]:
        return (lease.job_id, lease.lease_id, lease.attempt_id, lease.attempt)

    def heartbeat(self) -> Envelope:
        if self.state not in {WorkerState.ACTIVE, WorkerState.DRAINING}:
            raise ValueError("worker is not active")
        return self._out(Heartbeat(active_leases=tuple(item.lease for item in self.assignments.values())))

    def progress(
        self,
        lease: LeaseRef,
        *,
        stage: str,
        reason: str | None = None,
        numeric: dict[str, int | float] | None = None,
    ) -> Envelope:
        if self._key(lease) not in self.assignments:
            raise KeyError("unknown assignment")
        return self._out(JobProgress(lease=lease, stage=stage, reason=reason, numeric=numeric or {}))

    def result(
        self,
        lease: LeaseRef,
        *,
        result_schema_version: int,
        result: dict[str, object],
        completion_id: str,
        artifact_refs: tuple[str, ...] = (),
    ) -> Envelope:
        if self._key(lease) not in self.assignments:
            raise KeyError("unknown assignment")
        payload = JobResult(
            lease=lease,
            result_schema_version=result_schema_version,
            result=result,
            completion_id=completion_id,
            artifact_refs=artifact_refs,
        )
        self.completions[self._key(lease)] = payload
        return self._out(payload)

    def failure(
        self,
        lease: LeaseRef,
        *,
        outcome: AttemptOutcome,
        category: FailureCategory,
        error_code: str,
        summary: str,
    ) -> Envelope:
        if self._key(lease) not in self.assignments:
            raise KeyError("unknown assignment")
        return self._out(
            JobFailed(
                lease=lease,
                outcome=outcome,
                category=category,
                error_code=error_code,
                summary=summary,
            )
        )

    def reconnect_report(self, *, prior_session_id: str | None, prior_controller_epoch: int | None) -> Envelope:
        if self.state is not WorkerState.ACTIVE:
            raise ValueError("worker must register new session before reconciliation")
        self.state = WorkerState.RECONCILING
        items = tuple(
            ReconcileItem(
                lease=assignment.lease,
                prior_session_id=prior_session_id,
                accepted=True,
                cancellation_observed=key in self.cancelled,
                completion_id=self.completions[key].completion_id if key in self.completions else None,
                result_schema_version=(
                    self.completions[key].result_schema_version if key in self.completions else None
                ),
                result=self.completions[key].result if key in self.completions else None,
            )
            for key, assignment in self.assignments.items()
        )
        return self._out(
            Reconcile(
                prior_session_id=prior_session_id,
                prior_controller_epoch=prior_controller_epoch,
                items=items,
            )
        )

    def transport_lost(self) -> None:
        if self.state not in _TERMINAL:
            self.state = WorkerState.DISCONNECTED
            self.session_id = None
            self.controller_epoch = None
            self.reset_message_sequence()
