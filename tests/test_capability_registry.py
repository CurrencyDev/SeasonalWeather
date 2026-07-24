from __future__ import annotations

import datetime as dt

from seasonalweather.capabilities.manifest import CapabilityManifest, EpochDisposition
from seasonalweather.capabilities.models import (
    CapabilityRecord,
    CompatibilityState,
    OperationalState,
)
from seasonalweather.capabilities.probes import (
    CapabilityProbe,
    ProbeMode,
    ProbeReason,
)
from seasonalweather.capabilities.registry import CapabilityRegistry
from seasonalweather.jobs.policies import JobType

NOW = dt.datetime(2026, 7, 24, 12, tzinfo=dt.UTC)


def record(*, validity: int = 60, available: int = 2) -> CapabilityRecord:
    return CapabilityRecord(
        name="tts.synthesis.v1",
        implemented=True,
        compatibility=CompatibilityState.UNKNOWN,
        operational_state=OperationalState.HEALTHY,
        accepting_new_jobs=True,
        total_capacity=2,
        reported_available=available,
        parameters={"format": "wav"},
        validity_seconds=validity,
        observed_at=NOW,
        published_at=NOW,
    )


def registry_and_manifest(*, validity: int = 60):
    registry = CapabilityRegistry(
        allowed_capabilities=frozenset({"tts.synthesis.v1"}),
        maximum_validity_seconds=120,
    )
    manifest = CapabilityManifest.create(epoch=1, records=(record(validity=validity),))
    snapshot = registry.register(
        worker_id="worker_00000001",
        worker_instance_id="instance_00000001",
        session_id="session_00000001",
        manifest=manifest,
        authorized_capabilities=frozenset({"tts.synthesis.v1"}),
        authorized_job_types=frozenset({JobType.TTS_SYNTHESIZE}),
        payload_versions={JobType.TTS_SYNTHESIZE: 1},
        result_versions={JobType.TTS_SYNTHESIZE: 1},
        now=NOW,
    )
    return registry, manifest, snapshot


def test_registry_expiry_becomes_unknown_and_matching_heartbeat_refreshes() -> None:
    registry, manifest, _ = registry_and_manifest(validity=10)

    assert (
        registry.heartbeat(
            "worker_00000001",
            session_id="session_00000001",
            worker_instance_id="instance_00000001",
            epoch=1,
            digest=manifest.digest,
            now=NOW + dt.timedelta(seconds=5),
        )
        is EpochDisposition.IDEMPOTENT
    )
    assert registry.tick(NOW + dt.timedelta(seconds=11)) == ()
    assert registry.tick(NOW + dt.timedelta(seconds=16)) == ("worker_00000001",)
    stale = registry.snapshot("worker_00000001", NOW + dt.timedelta(seconds=16))
    assert stale is not None
    assert stale.records[0].operational_state is OperationalState.UNKNOWN
    assert stale.effective_capacity["tts.synthesis.v1"] == 0


def test_digest_mismatch_blocks_assignment_until_full_report() -> None:
    registry, manifest, _ = registry_and_manifest()

    disposition = registry.heartbeat(
        "worker_00000001",
        session_id="session_00000001",
        worker_instance_id="instance_00000001",
        epoch=1,
        digest="sha256:" + "f" * 64,
        now=NOW,
    )
    blocked = registry.snapshot("worker_00000001", NOW)

    assert disposition is EpochDisposition.CONFLICT
    assert blocked is not None
    assert blocked.probe_required is True
    assert (
        registry.apply_full_report(
            "worker_00000001",
            session_id="session_00000001",
            worker_instance_id="instance_00000001",
            manifest=manifest,
            now=NOW,
        )
        is EpochDisposition.ACCEPTED
    )
    restored = registry.snapshot("worker_00000001", NOW)
    assert restored is not None
    assert restored.trusted is True


def test_reservation_bound_active_and_release_accounting() -> None:
    registry, manifest, snapshot = registry_and_manifest()
    reservation = registry.reserve(
        worker_id=snapshot.worker_id,
        worker_instance_id=snapshot.worker_instance_id,
        reservation_id="reservation_00000001",
        job_id="job_00000001",
        capability_names=("tts.synthesis.v1",),
        snapshot_token="q:" + "0" * 32,
        expected_epoch=manifest.epoch,
        expected_digest=manifest.digest,
        now=NOW,
        expires_at=NOW + dt.timedelta(seconds=5),
    )
    lease_key = (
        "job_00000001",
        "lease_00000001",
        "attempt_00000001",
        1,
    )

    assert reservation.capability_names == ("tts.synthesis.v1",)
    assert registry.snapshot(snapshot.worker_id, NOW).effective_capacity["tts.synthesis.v1"] == 1
    registry.bind(snapshot.worker_id, reservation.reservation_id, lease_key=lease_key)
    registry.activate(
        snapshot.worker_id,
        reservation.reservation_id,
        lease_key=lease_key,
    )
    assert registry.snapshot(snapshot.worker_id, NOW).active_assignments == 1
    registry.release_active(snapshot.worker_id, lease_key)
    assert registry.snapshot(snapshot.worker_id, NOW).effective_capacity["tts.synthesis.v1"] == 2


def test_probe_correlation_rejects_wrong_identity_and_timeout() -> None:
    registry, _, snapshot = registry_and_manifest()
    probe = CapabilityProbe(
        probe_id="probe_00000001",
        mode=ProbeMode.TARGETED,
        target_names=("tts.synthesis.v1",),
        reason=ProbeReason.INTERNAL,
        requested_at=NOW,
        deadline_at=NOW + dt.timedelta(seconds=5),
        session_id=snapshot.session_id,
        worker_instance_id=snapshot.worker_instance_id,
    )
    registry.add_probe(snapshot.worker_id, probe)

    mismatch = registry.match_probe(
        snapshot.worker_id,
        probe.probe_id,
        session_id=snapshot.session_id,
        worker_instance_id="instance_wrong_0001",
        now=NOW,
    )
    matched = registry.match_probe(
        snapshot.worker_id,
        probe.probe_id,
        session_id=snapshot.session_id,
        worker_instance_id=snapshot.worker_instance_id,
        now=NOW,
    )

    assert mismatch.accepted is False
    assert matched.accepted is True


def test_reconnect_active_use_is_rebuilt_from_matching_evidence() -> None:
    registry, _, snapshot = registry_and_manifest()
    lease_key = (
        "job_00000001",
        "lease_00000001",
        "attempt_00000001",
        1,
    )

    registry.reconcile_active(
        snapshot.worker_id,
        worker_instance_id=snapshot.worker_instance_id,
        assignments={lease_key: ("tts.synthesis.v1",)},
    )
    rebuilt = registry.snapshot(snapshot.worker_id, NOW)

    assert rebuilt is not None
    assert rebuilt.active_assignments == 1
    assert rebuilt.effective_capacity["tts.synthesis.v1"] == 1
