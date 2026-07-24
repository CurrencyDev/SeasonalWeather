from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable

from ..jobs.contracts import AttemptOutcome, JobError, JobRecord
from ..jobs.policies import ExecutorClass, QueueClass
from ..lifecycle import Lifecycle, WorkClass
from .models import JobAssignment, ResultCommitReceipt
from .repository import JobRepository


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(microsecond=0)


class JobScheduler:
    """Controller queue authority; returns assignments and never executes work."""

    def __init__(
        self,
        repository: JobRepository,
        lifecycle: Lifecycle,
        *,
        lease_seconds: int,
        acknowledgment_seconds: int,
        clock: Callable[[], dt.datetime] = _utc_now,
    ) -> None:
        self.repository = repository
        self.lifecycle = lifecycle
        self.lease_seconds = int(lease_seconds)
        self.acknowledgment_seconds = int(acknowledgment_seconds)
        self.clock = clock

    def assign(
        self,
        *,
        owner: str,
        queues: Iterable[QueueClass] | None = None,
        executors: Iterable[ExecutorClass] | None = None,
        capabilities: Iterable[str] = (),
    ) -> JobAssignment | None:
        self.lifecycle.require(WorkClass.JOB_LEASE)
        return self.repository.acquire_next(
            owner=owner,
            now=self.clock(),
            lease_seconds=self.lease_seconds,
            acknowledgment_seconds=self.acknowledgment_seconds,
            queues=queues,
            executors=executors,
            capabilities=capabilities,
        )

    def acknowledge(self, assignment: JobAssignment) -> JobRecord:
        return self.repository.acknowledge(
            job_id=assignment.job.job_id,
            lease_id=assignment.lease_id,
            attempt_id=assignment.attempt_id,
            owner=assignment.lease_owner,
            at=self.clock(),
        )

    def renew(self, assignment: JobAssignment) -> JobRecord:
        return self.repository.renew(
            job_id=assignment.job.job_id,
            lease_id=assignment.lease_id,
            attempt_id=assignment.attempt_id,
            owner=assignment.lease_owner,
            at=self.clock(),
            lease_seconds=self.lease_seconds,
        )

    def progress(
        self,
        assignment: JobAssignment,
        *,
        stage: str,
        reason: str | None = None,
        numeric: dict[str, int | float] | None = None,
    ) -> None:
        self.repository.append_progress(
            job_id=assignment.job.job_id,
            lease_id=assignment.lease_id,
            attempt_id=assignment.attempt_id,
            owner=assignment.lease_owner,
            stage=stage,
            reason=reason,
            numeric=numeric or {},
            at=self.clock(),
        )

    def outcome(
        self,
        assignment: JobAssignment,
        *,
        outcome: AttemptOutcome,
        result_payload: dict[str, object] | None = None,
        error: JobError | None = None,
        replay_permitted: bool = False,
    ) -> JobRecord | ResultCommitReceipt:
        return self.repository.record_outcome(
            job_id=assignment.job.job_id,
            lease_id=assignment.lease_id,
            attempt_id=assignment.attempt_id,
            owner=assignment.lease_owner,
            outcome=outcome,
            at=self.clock(),
            result_payload=result_payload,
            error=error,
            replay_permitted=replay_permitted,
        )
