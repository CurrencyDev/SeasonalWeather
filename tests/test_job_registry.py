from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from pydantic import ValidationError

from seasonalweather.job_admission import JobAdmissionService
from seasonalweather.jobs.policies import (
    ConfigFence,
    DedupeMode,
    ExecutorClass,
    FinalCommitAuthority,
    JobPriority,
    JobType,
    QueueClass,
    queue_executor_compatible,
)
from seasonalweather.jobs.registry import JOB_TYPE_POLICIES
from seasonalweather.lifecycle import AdmissionClosedError, Lifecycle

NOW = dt.datetime(2026, 7, 24, 12, tzinfo=dt.UTC)


def test_registry_declares_complete_policy_for_each_initial_job_type() -> None:
    assert set(JOB_TYPE_POLICIES) == set(JobType)
    for job_type, policy in JOB_TYPE_POLICIES.items():
        assert policy.job_type is job_type
        assert policy.payload_schema_version == 1
        assert policy.result_schema_version == 1
        assert policy.retry.attempt_timeout_seconds > 0
        assert policy.retry.max_attempts > 0
        assert policy.deadline.required or policy.deadline.default_seconds
        assert policy.final_commit_authority is FinalCommitAuthority.CONTROLLER
        assert queue_executor_compatible(policy.queue, policy.executor)


def test_alert_policy_has_strict_deadline_nonblind_replay_and_identity_fences() -> None:
    policy = JOB_TYPE_POLICIES[JobType.ALERT_ARTIFACT_GENERATE]
    assert policy.queue is QueueClass.ROUTINE
    assert policy.executor is ExecutorClass.ROUTINE_WORKER
    assert policy.default_priority is JobPriority.SAFETY_CRITICAL
    assert policy.deadline.required
    assert policy.retry.max_attempts == 1
    assert policy.dedupe.mode is DedupeMode.EXACT
    assert policy.fences.config_generation is ConfigFence.REQUIRED
    assert policy.fences.source_identity
    assert policy.fences.event_identity
    assert policy.fences.content_identity


def test_payload_schemas_reject_raw_text_paths_and_unknown_fields() -> None:
    schema = JOB_TYPE_POLICIES[JobType.TTS_SYNTHESIZE].payload_schema
    with pytest.raises(ValidationError):
        schema.model_validate(
            {
                "content_ref": "content_01234567",
                "voice_profile_ref": "voice_01234567",
                "output_format": "wav",
                "config_generation": 1,
                "synthesis_text": "do not embed me",
            }
        )


def test_job_admission_requires_lifecycle_generation_and_deadline() -> None:
    lifecycle = Lifecycle()
    lifecycle.mark_running()
    admission = JobAdmissionService(lifecycle, clock=lambda: NOW)
    payload = {
        "content_ref": "content_01234567",
        "voice_profile_ref": "voice_01234567",
        "output_format": "wav",
        "config_generation": 3,
    }
    with pytest.raises(ValueError, match="configuration generation"):
        admission.admit(job_type=JobType.TTS_SYNTHESIZE, payload=payload)

    job = admission.admit(
        job_type=JobType.TTS_SYNTHESIZE,
        payload=payload,
        config_generation=3,
        dedupe_key="tts:content_01234567",
    )
    assert job.deadline_at == NOW + dt.timedelta(seconds=180)
    assert job.config_generation == 3

    lifecycle.request_shutdown()
    with pytest.raises(AdmissionClosedError):
        admission.admit(
            job_type=JobType.TTS_SYNTHESIZE,
            payload=payload,
            config_generation=3,
        )


def test_long_running_sources_are_not_job_types() -> None:
    forbidden = {"nwws", "cap", "ipaws", "ern", "poller", "source_client"}
    assert not any(term in job_type.value for job_type in JobType for term in forbidden)


def test_control_module_does_not_duplicate_command_or_job_policy() -> None:
    source = (Path(__file__).parents[1] / "seasonalweather" / "control.py").read_text(encoding="utf-8")
    for authority in ("class CommandStatus", "class JobStatus", "class JobTypePolicy", "JOB_TYPE_POLICIES"):
        assert authority not in source
