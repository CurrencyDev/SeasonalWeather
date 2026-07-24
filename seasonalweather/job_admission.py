from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from .jobs.contracts import JobRecord
from .jobs.policies import ConfigFence, JobType
from .jobs.registry import policy_for
from .lifecycle import Lifecycle, WorkClass


class JobAdmissionService:
    """Lifecycle-gated, persistence-neutral construction of immutable job specifications."""

    def __init__(self, lifecycle: Lifecycle, *, clock: Any | None = None) -> None:
        self._lifecycle = lifecycle
        self._clock = clock or (lambda: dt.datetime.now(dt.UTC).replace(microsecond=0))

    def admit(
        self,
        *,
        job_type: JobType,
        payload: dict[str, Any],
        command_id: str | None = None,
        deadline_at: dt.datetime | None = None,
        not_before: dt.datetime | None = None,
        dedupe_key: str | None = None,
        config_generation: int | None = None,
    ) -> JobRecord:
        self._lifecycle.require(WorkClass.COMMAND)
        policy = policy_for(job_type)
        typed_payload = policy.payload_schema.model_validate(payload)
        if policy.fences.config_generation is ConfigFence.REQUIRED and config_generation is None:
            raise ValueError("job type requires an admitted configuration generation")
        now = self._clock()
        eligible_at = not_before or now
        if deadline_at is None:
            if policy.deadline.default_seconds is None:
                raise ValueError("job type requires an explicit deadline")
            deadline_at = now + dt.timedelta(seconds=policy.deadline.default_seconds)
        return JobRecord(
            job_id=f"job_{uuid.uuid4().hex[:20]}",
            command_id=command_id,
            job_type=job_type,
            queue=policy.queue,
            executor=policy.executor,
            priority=policy.default_priority,
            payload_schema_version=policy.payload_schema_version,
            result_schema_version=policy.result_schema_version,
            payload=typed_payload.model_dump(mode="json"),
            created_at=now,
            not_before=eligible_at,
            deadline_at=deadline_at,
            max_attempts=policy.retry.max_attempts,
            dedupe_key=dedupe_key,
            config_generation=config_generation,
            replay_policy=policy.replay,
        )
