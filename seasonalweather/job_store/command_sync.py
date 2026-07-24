from __future__ import annotations

from ..commands.contracts import (
    TERMINAL_COMMAND_STATUSES,
    CommandStatus,
    RelationshipCompletion,
)
from ..commands.service import CommandNotFoundError, CommandStore
from ..jobs.contracts import TERMINAL_JOB_STATUSES, JobRecord, JobStatus
from .repository import JobRepository


class CommandJobCoordinator:
    """Idempotently repairs the non-atomic job-DB/command-DB boundary."""

    def __init__(
        self,
        repository: JobRepository,
        command_store: CommandStore,
    ) -> None:
        self.repository = repository
        self.command_store = command_store

    async def repair(self, *, limit: int = 100) -> int:
        repaired = 0
        for job in self.repository.jobs_pending_command_sync(limit=limit):
            repaired += await self._repair_job(job)
        return repaired

    async def _repair_job(self, job: JobRecord) -> int:
        if job.command_id is None:
            return 0
        try:
            command = await self.command_store.get(job.command_id)
        except CommandNotFoundError:
            return 0
        if command.status not in TERMINAL_COMMAND_STATUSES:
            siblings = tuple(
                candidate
                for candidate in self.repository.list_jobs(limit=500)
                if candidate.command_id == job.command_id
            )
            if command.status is CommandStatus.ACCEPTED:
                command = await self.command_store.mark_running(command.command_id)
            await self._finalize_aggregate(command, siblings)
        self.repository.mark_command_synced(
            job.job_id,
            expected_status=job.status,
        )
        return 1

    async def _finalize_aggregate(self, command, siblings) -> None:
        if not siblings or not all(sibling.status in TERMINAL_JOB_STATUSES for sibling in siblings):
            return
        failures = [
            sibling
            for sibling in siblings
            if sibling.status in {JobStatus.FAILED, JobStatus.EXPIRED, JobStatus.CANCELLED}
        ]
        if failures:
            await self.command_store.mark_failed(
                command.command_id,
                {
                    "code": "required_job_failed",
                    "message": "A required durable job did not succeed.",
                    "details": {"failed_count": len(failures)},
                },
            )
            return
        if command.relationship.completion is RelationshipCompletion.ALL_REQUIRED_JOBS:
            await self.command_store.mark_succeeded(
                command.command_id,
                {"job_count": len(siblings)},
            )
