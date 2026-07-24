"""Serialized controller coordination across registry, scheduler, and SWWP."""

from __future__ import annotations

import datetime as dt
import threading
from collections.abc import Callable

from ..job_store.models import ResultCommitReceipt
from ..jobs.contracts import JobRecord
from ..jobs.policies import ExecutorClass, JobType, QueueClass
from ..jobs.registry import policy_for
from ..swwp.adapter import JobStoreSwwpAdapter
from ..swwp.messages import (
    JobAssignmentPayload,
    JobFailed,
    JobProgress,
    JobResult,
    LeaseRef,
    ReconcileDecision,
    ReconcileItem,
)
from .manifest import CapabilityManifest, CapabilityUpdate, EpochDisposition
from .models import CapabilityRecord
from .probes import CapabilityProbe, ProbeMode, ProbeReason
from .qualification import QualificationResult, qualify
from .registry import CapabilityRegistry, WorkerCapabilitySnapshot


def declared_capability_names() -> frozenset[str]:
    return frozenset(requirement.name for job_type in JobType for requirement in policy_for(job_type).capabilities)


class CapabilitySchedulerService:
    """Narrow owner of qualification-before-lease and capacity reconciliation."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        durable: JobStoreSwwpAdapter,
        *,
        clock: Callable[[], dt.datetime],
        id_factory: Callable[[str], str],
        probe_timeout_seconds: int = 15,
    ) -> None:
        if not 1 <= probe_timeout_seconds <= 300:
            raise ValueError("probe timeout is out of bounds")
        self.registry = registry
        self.durable = durable
        self.clock = clock
        self.id_factory = id_factory
        self.probe_timeout_seconds = probe_timeout_seconds
        self._reservation_by_lease: dict[tuple[str, str, str, int], str] = {}
        self._worker_by_lease: dict[tuple[str, str, str, int], str] = {}
        self._outbound_probes: dict[str, list[CapabilityProbe]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _lease_key(lease: LeaseRef) -> tuple[str, str, str, int]:
        return (lease.job_id, lease.lease_id, lease.attempt_id, lease.attempt)

    def register(
        self,
        *,
        worker_id: str,
        worker_instance_id: str,
        session_id: str,
        manifest: CapabilityManifest,
        authorized_capabilities: frozenset[str],
        authorized_job_types: frozenset[JobType],
        payload_versions: dict[JobType, int],
        result_versions: dict[JobType, int],
    ):
        snapshot = self.registry.register(
            worker_id=worker_id,
            worker_instance_id=worker_instance_id,
            session_id=session_id,
            manifest=manifest,
            authorized_capabilities=authorized_capabilities,
            authorized_job_types=authorized_job_types,
            payload_versions=payload_versions,
            result_versions=result_versions,
            now=self.clock(),
        )
        if snapshot.probe_required:
            self.request_probe(worker_id, mode=ProbeMode.FULL, reason=ProbeReason.RECONNECT)
        return snapshot

    def update(
        self,
        worker_id: str,
        *,
        session_id: str,
        worker_instance_id: str,
        update: CapabilityUpdate,
    ) -> EpochDisposition:
        disposition = self.registry.apply_update(
            worker_id,
            session_id=session_id,
            worker_instance_id=worker_instance_id,
            update=update,
            now=self.clock(),
        )
        if disposition in {EpochDisposition.CONFLICT, EpochDisposition.GAP}:
            reason = ProbeReason.DIGEST_MISMATCH if disposition is EpochDisposition.CONFLICT else ProbeReason.EPOCH_GAP
            self.request_probe(worker_id, mode=ProbeMode.FULL, reason=reason)
        return disposition

    def heartbeat(
        self,
        worker_id: str,
        *,
        session_id: str,
        worker_instance_id: str,
        epoch: int,
        digest: str,
    ) -> EpochDisposition:
        disposition = self.registry.heartbeat(
            worker_id,
            session_id=session_id,
            worker_instance_id=worker_instance_id,
            epoch=epoch,
            digest=digest,
            now=self.clock(),
        )
        if disposition in {
            EpochDisposition.CONFLICT,
            EpochDisposition.GAP,
        }:
            reason = ProbeReason.DIGEST_MISMATCH if disposition is EpochDisposition.CONFLICT else ProbeReason.EPOCH_GAP
            self.request_probe(worker_id, mode=ProbeMode.FULL, reason=reason)
        return disposition

    def report(
        self,
        worker_id: str,
        *,
        session_id: str,
        worker_instance_id: str,
        probe_id: str,
        schema_version: int,
        epoch: int,
        records: tuple[CapabilityRecord, ...],
        full_digest: str,
        validity_seconds: int,
    ) -> EpochDisposition:
        match = self.registry.match_probe(
            worker_id,
            probe_id,
            session_id=session_id,
            worker_instance_id=worker_instance_id,
            now=self.clock(),
        )
        if not match.accepted or match.probe is None:
            raise ValueError(match.reason)
        probe = match.probe
        names = tuple(record.name for record in records)
        if probe.mode is ProbeMode.FULL:
            manifest = CapabilityManifest(
                schema_version=schema_version,
                epoch=epoch,
                records=records,
                digest=full_digest,
            )
            return self.registry.apply_full_report(
                worker_id,
                session_id=session_id,
                worker_instance_id=worker_instance_id,
                manifest=manifest,
                now=self.clock(),
            )
        if names != probe.target_names:
            raise ValueError("targeted report does not match probe targets")
        current = self.registry.snapshot(worker_id, self.clock())
        if current is None or not current.trusted:
            raise ValueError("targeted report cannot restore an untrusted baseline")
        return self.registry.apply_update(
            worker_id,
            session_id=session_id,
            worker_instance_id=worker_instance_id,
            update=CapabilityUpdate(
                epoch=epoch,
                changed=records,
                removed=(),
                resulting_digest=full_digest,
                validity_seconds=validity_seconds,
            ),
            now=self.clock(),
        )

    def qualified_workers(
        self,
        *,
        job: JobRecord,
    ) -> tuple[QualificationResult, ...]:
        policy = policy_for(job.job_type)
        results = [
            qualify(
                snapshot.qualification_view(),
                job_type=job.job_type,
                payload_schema_version=job.payload_schema_version,
                result_schema_version=job.result_schema_version,
                requirements=policy.capabilities,
            )
            for snapshot in self.registry.snapshots(self.clock())
        ]
        return tuple(
            sorted(
                (item for item in results if item.qualified),
                key=lambda item: (-item.effective_capacity, item.worker_id),
            )
        )

    def acquire(
        self,
        *,
        owner: str,
        queues: tuple[QueueClass, ...],
        executors: tuple[ExecutorClass, ...],
        capabilities: tuple[str, ...],
    ) -> JobAssignmentPayload | None:
        """Qualify and reserve before asking P1-07 to acquire one durable lease."""

        del capabilities
        with self._lock:
            self.registry.tick(self.clock())
            snapshot = self.registry.snapshot(owner, self.clock())
            if snapshot is None:
                return None
            for job in self.durable.scheduler.pending_candidates():
                result = self._candidate_qualification(job, snapshot, queues, executors)
                if result is None:
                    continue
                assignment = self._reserve_and_acquire(
                    owner,
                    snapshot,
                    job,
                    result,
                    queues,
                    executors,
                )
                if assignment is not None:
                    return assignment
            return None

    @staticmethod
    def _candidate_qualification(
        job: JobRecord,
        snapshot: WorkerCapabilitySnapshot,
        queues: tuple[QueueClass, ...],
        executors: tuple[ExecutorClass, ...],
    ) -> QualificationResult | None:
        if job.queue not in queues or job.executor not in executors:
            return None
        if job.job_type not in snapshot.authorized_job_types:
            return None
        policy = policy_for(job.job_type)
        result = qualify(
            snapshot.qualification_view(),
            job_type=job.job_type,
            payload_schema_version=job.payload_schema_version,
            result_schema_version=job.result_schema_version,
            requirements=policy.capabilities,
        )
        return result if result.qualified else None

    def _reserve_and_acquire(
        self,
        owner: str,
        snapshot: WorkerCapabilitySnapshot,
        job: JobRecord,
        result: QualificationResult,
        queues: tuple[QueueClass, ...],
        executors: tuple[ExecutorClass, ...],
    ) -> JobAssignmentPayload | None:
        policy = policy_for(job.job_type)
        names = tuple(sorted({item.name for item in policy.capabilities if item.required}))
        reservation_id = self.id_factory("reservation")
        now = self.clock()
        self.registry.reserve(
            worker_id=owner,
            worker_instance_id=snapshot.worker_instance_id,
            reservation_id=reservation_id,
            job_id=job.job_id,
            capability_names=names,
            snapshot_token=result.snapshot_token,
            expected_epoch=snapshot.epoch,
            expected_digest=snapshot.digest,
            now=now,
            expires_at=min(
                now + dt.timedelta(seconds=self.durable.scheduler.acknowledgment_seconds),
                job.deadline_at,
            ),
        )
        try:
            assignment = self.durable.acquire_job(
                owner=owner,
                queues=queues,
                executors=executors,
                capabilities=names,
                job_id=job.job_id,
            )
        except Exception:
            self.registry.release_reservation(owner, reservation_id)
            raise
        if assignment is None:
            self.registry.release_reservation(owner, reservation_id)
            return None
        key = self._lease_key(assignment.lease)
        self.registry.bind(owner, reservation_id, lease_key=key)
        self._reservation_by_lease[key] = reservation_id
        self._worker_by_lease[key] = owner
        return assignment

    def acknowledge(self, lease: LeaseRef) -> JobRecord:
        with self._lock:
            key = self._lease_key(lease)
            worker_id = self._worker_by_lease[key]
            result = self.durable.acknowledge(lease)
            self.registry.activate(
                worker_id,
                self._reservation_by_lease.pop(key),
                lease_key=key,
            )
            return result

    def reject_unacknowledged(
        self,
        lease: LeaseRef,
        *,
        category: str,
        capability_names: tuple[str, ...],
    ) -> JobRecord:
        with self._lock:
            key = self._lease_key(lease)
            worker_id = self._worker_by_lease.pop(key)
            self._reservation_by_lease.pop(key, None)
            result = self.durable.reject_unacknowledged(lease)
            self.registry.reject_assignment(
                worker_id,
                lease_key=key,
                capability_names=capability_names,
                reason=category,
            )
            mode = ProbeMode.TARGETED if capability_names else ProbeMode.FULL
            self.request_probe(
                worker_id,
                mode=mode,
                reason=ProbeReason.JOB_REJECTED,
                names=capability_names,
            )
            return result

    def renew(self, lease: LeaseRef) -> JobRecord:
        return self.durable.renew(lease)

    def progress(self, progress: JobProgress) -> None:
        self.durable.progress(progress)

    def result(self, result: JobResult) -> ResultCommitReceipt:
        receipt = self.durable.result(result)
        self._release_active(result.lease)
        return receipt

    def failure(self, failure: JobFailed):
        result = self.durable.failure(failure)
        self._release_active(failure.lease)
        return result

    def request_cancellation(self, job_id: str) -> JobRecord:
        return self.durable.request_cancellation(job_id)

    def reconcile(self, item: ReconcileItem) -> ReconcileDecision:
        decision = self.durable.reconcile(item)
        if decision.disposition.value in {
            "already_committed",
            "discard_stale",
            "unknown",
            "revalidation_required",
        }:
            self._release_active(item.lease)
        return decision

    def reconcile_repository(self) -> None:
        self.durable.reconcile_repository()

    def reconcile_active(
        self,
        worker_id: str,
        *,
        worker_instance_id: str,
        assignments: tuple[tuple[LeaseRef, tuple[str, ...]], ...],
    ) -> None:
        observed = {self._lease_key(lease): tuple(sorted(set(names))) for lease, names in assignments}
        self.registry.reconcile_active(
            worker_id,
            worker_instance_id=worker_instance_id,
            assignments=observed,
        )
        for lease, _ in assignments:
            self._worker_by_lease[self._lease_key(lease)] = worker_id

    def _release_active(self, lease: LeaseRef) -> None:
        key = self._lease_key(lease)
        worker_id = self._worker_by_lease.pop(key, None)
        self._reservation_by_lease.pop(key, None)
        if worker_id is not None:
            self.registry.release_active(worker_id, key)

    def request_probe(
        self,
        worker_id: str,
        *,
        mode: ProbeMode,
        reason: ProbeReason,
        names: tuple[str, ...] = (),
    ) -> CapabilityProbe:
        snapshot = self.registry.snapshot(worker_id, self.clock())
        if snapshot is None:
            raise KeyError("unknown worker")
        now = self.clock()
        targets = tuple(sorted(set(names)))
        probe = CapabilityProbe(
            probe_id=self.id_factory("probe"),
            mode=mode,
            target_names=targets,
            reason=reason,
            requested_at=now,
            deadline_at=now + dt.timedelta(seconds=self.probe_timeout_seconds),
            session_id=snapshot.session_id,
            worker_instance_id=snapshot.worker_instance_id,
        )
        self.registry.add_probe(worker_id, probe)
        self._outbound_probes.setdefault(worker_id, []).append(probe)
        return probe

    def take_probes(self, worker_id: str) -> tuple[CapabilityProbe, ...]:
        with self._lock:
            return tuple(self._outbound_probes.pop(worker_id, ()))

    def tick(self) -> tuple[str, ...]:
        changed = self.registry.tick(self.clock())
        for worker_id in changed:
            self.request_probe(
                worker_id,
                mode=ProbeMode.FULL,
                reason=ProbeReason.STALE,
            )
        return changed

    def disconnect(self, worker_id: str, *, session_id: str) -> None:
        self.registry.disconnect(worker_id, session_id=session_id)

    def drain(self, worker_id: str) -> None:
        self.registry.drain(worker_id)

    def close(self) -> None:
        self.registry.close()
