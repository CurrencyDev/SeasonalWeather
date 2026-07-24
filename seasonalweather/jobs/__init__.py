"""Persistence-neutral contracts for bounded SeasonalWeather jobs."""

from .contracts import (
    AttemptOutcome,
    JobError,
    JobRecord,
    JobResult,
    JobStatus,
    ReplayEvidence,
    ReplayPolicy,
    lease_job,
    request_job_cancellation,
    resolve_attempt,
    should_replay,
    start_job,
    supersede_job,
)
from .policies import ExecutorClass, JobPriority, JobType, QueueClass
from .registry import JOB_TYPE_POLICIES, JobTypePolicy, policy_for

__all__ = [
    "AttemptOutcome",
    "ExecutorClass",
    "JOB_TYPE_POLICIES",
    "JobError",
    "JobPriority",
    "JobRecord",
    "JobResult",
    "JobStatus",
    "JobType",
    "JobTypePolicy",
    "QueueClass",
    "ReplayEvidence",
    "ReplayPolicy",
    "lease_job",
    "policy_for",
    "request_job_cancellation",
    "resolve_attempt",
    "should_replay",
    "start_job",
    "supersede_job",
]
