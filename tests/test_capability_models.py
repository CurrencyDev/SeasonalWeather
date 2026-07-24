from __future__ import annotations

import datetime as dt

import pytest

from seasonalweather.capabilities.hysteresis import (
    CapabilityHysteresis,
    CapabilityObservation,
    HysteresisPolicy,
)
from seasonalweather.capabilities.manifest import (
    CapabilityManifest,
    CapabilityUpdate,
    EpochDisposition,
    apply_update,
    manifest_digest,
)
from seasonalweather.capabilities.models import (
    CapabilityRecord,
    CompatibilityState,
    OperationalState,
)
from seasonalweather.capabilities.qualification import (
    QualificationReason,
    WorkerQualificationView,
    qualify,
)
from seasonalweather.jobs.policies import CapabilityRequirement, JobType

NOW = dt.datetime(2026, 7, 24, 12, tzinfo=dt.UTC)


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += dt.timedelta(seconds=seconds)


def record(
    *,
    name: str = "tts.synthesis.v1",
    state: OperationalState = OperationalState.HEALTHY,
    accepting: bool = True,
    compatibility: CompatibilityState = CompatibilityState.UNKNOWN,
    total: int = 2,
    available: int = 2,
    parameters: dict | None = None,
) -> CapabilityRecord:
    return CapabilityRecord(
        name=name,
        implemented=True,
        compatibility=compatibility,
        operational_state=state,
        accepting_new_jobs=accepting,
        total_capacity=total,
        reported_available=available,
        parameters=parameters or {"format": "wav"},
        validity_seconds=60,
        observed_at=NOW,
        published_at=NOW,
    )


def test_record_invariants_and_bounded_parameter_normalization() -> None:
    normalized = record(
        parameters={
            "format": "wav",
            "sample_rates": [48000, 24000, 48000],
        }
    )

    assert normalized.parameters["sample_rates"] == (24000, 48000)
    with pytest.raises(ValueError, match="inactive"):
        record(state=OperationalState.UNAVAILABLE, accepting=True)
    with pytest.raises(ValueError, match="exceed"):
        record(total=1, available=2)
    with pytest.raises(ValueError, match="unknown capability parameter"):
        record(parameters={"filesystem_path": "/tmp/sentinel"})
    with pytest.raises(ValueError):
        record(parameters={"profiles": [{"nested": True}]})


def test_manifest_digest_is_canonical_and_changes_semantically() -> None:
    first = record(parameters={"format": "wav", "sample_rates": [48000, 24000]})
    reordered = record(parameters={"sample_rates": [24000, 48000], "format": "wav"})
    changed = record(parameters={"format": "ogg", "sample_rates": [24000, 48000]})

    first_digest = manifest_digest(schema_version=1, records=(first,))

    assert first_digest == manifest_digest(schema_version=1, records=(reordered,))
    assert first_digest != manifest_digest(schema_version=1, records=(changed,))
    assert first_digest.startswith("sha256:")


def test_partial_update_is_atomic_and_epoch_rules_fail_closed() -> None:
    original = CapabilityManifest.create(epoch=1, records=(record(),))
    changed = record(available=1)
    expected = CapabilityManifest.create(epoch=2, records=(changed,))
    update = CapabilityUpdate(
        epoch=2,
        changed=(changed,),
        removed=(),
        resulting_digest=expected.digest,
        validity_seconds=60,
    )

    accepted = apply_update(original, update)
    duplicate = apply_update(accepted.manifest, update)  # type: ignore[arg-type]
    corrupt = CapabilityUpdate(
        epoch=3,
        changed=(record(available=2),),
        removed=(),
        resulting_digest=expected.digest,
        validity_seconds=60,
    )
    gap = CapabilityUpdate(
        epoch=5,
        changed=(),
        removed=(),
        resulting_digest=expected.digest,
        validity_seconds=60,
    )

    assert accepted.disposition is EpochDisposition.ACCEPTED
    assert duplicate.disposition is EpochDisposition.IDEMPOTENT
    assert apply_update(expected, corrupt).disposition is EpochDisposition.CONFLICT
    assert apply_update(expected, gap).disposition is EpochDisposition.GAP


def _view(capability: CapabilityRecord, **updates) -> WorkerQualificationView:
    values = {
        "worker_id": "worker_00000001",
        "worker_instance_id": "instance_00000001",
        "session_id": "session_00000001",
        "epoch": 1,
        "digest": "sha256:" + "0" * 64,
        "records": (capability,),
        "authorized_capabilities": frozenset({"tts.synthesis.v1"}),
        "authorized_job_types": frozenset({JobType.TTS_SYNTHESIZE}),
        "payload_versions": {JobType.TTS_SYNTHESIZE: 1},
        "result_versions": {JobType.TTS_SYNTHESIZE: 1},
        "effective_capacity": {"tts.synthesis.v1": 1},
    }
    values.update(updates)
    return WorkerQualificationView(**values)


@pytest.mark.parametrize(
    ("capability", "updates", "reason"),
    [
        (
            record(compatibility=CompatibilityState.INCOMPATIBLE),
            {},
            QualificationReason.INCOMPATIBLE,
        ),
        (
            record(
                state=OperationalState.DEGRADED,
                accepting=False,
                compatibility=CompatibilityState.COMPATIBLE,
            ),
            {},
            QualificationReason.DEGRADED_NOT_ACCEPTING,
        ),
        (
            record(
                state=OperationalState.UNKNOWN,
                accepting=False,
                compatibility=CompatibilityState.COMPATIBLE,
            ),
            {},
            QualificationReason.UNKNOWN_OR_STALE,
        ),
        (
            record(compatibility=CompatibilityState.COMPATIBLE),
            {"effective_capacity": {"tts.synthesis.v1": 0}},
            QualificationReason.NO_CAPACITY,
        ),
        (
            record(compatibility=CompatibilityState.COMPATIBLE),
            {"authorized_capabilities": frozenset()},
            QualificationReason.UNAUTHORIZED,
        ),
    ],
)
def test_qualification_reasons_are_bounded(
    capability: CapabilityRecord,
    updates: dict,
    reason: QualificationReason,
) -> None:
    result = qualify(
        _view(capability, **updates),
        job_type=JobType.TTS_SYNTHESIZE,
        payload_schema_version=1,
        result_schema_version=1,
        requirements=(
            CapabilityRequirement(
                name="tts.synthesis.v1",
                parameters={"format": "wav"},
            ),
        ),
    )

    assert result.reason is reason
    assert result.qualified is False


def test_hysteresis_threshold_dwell_hard_failure_and_capacity_reduction() -> None:
    clock = Clock()
    machine = CapabilityHysteresis(
        record(compatibility=CompatibilityState.UNKNOWN),
        policy=HysteresisPolicy(
            failure_threshold=2,
            recovery_success_threshold=2,
            minimum_degraded_dwell_seconds=5,
        ),
        clock=clock,
    )

    assert machine.observe(CapabilityObservation(successful=False)) is None
    degraded = machine.observe(CapabilityObservation(successful=False))
    assert degraded is not None
    assert degraded.operational_state is OperationalState.DEGRADED
    assert machine.observe(CapabilityObservation(successful=True)) is None
    assert machine.observe(CapabilityObservation(successful=True)) is None
    clock.advance(5)
    recovered = machine.observe(CapabilityObservation(successful=True))
    assert recovered is not None
    assert recovered.operational_state is OperationalState.HEALTHY
    reduced = machine.observe(CapabilityObservation(successful=True, available_capacity=1))
    assert reduced is not None
    assert reduced.reported_available == 1
    unavailable = machine.observe(
        CapabilityObservation(
            successful=False,
            hard_failure=True,
            state=OperationalState.UNAVAILABLE,
        )
    )
    assert unavailable is not None
    assert unavailable.operational_state is OperationalState.UNAVAILABLE
