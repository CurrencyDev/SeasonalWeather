import asyncio
from types import SimpleNamespace

from seasonalweather.health_state import HealthStateMachine


def test_health_state_enters_detached_when_all_alert_sources_fail() -> None:
    cfg = SimpleNamespace(
        enabled=True,
        check_interval_seconds=30,
        min_hold_seconds=0,
        detached_loop_only=True,
        source_impaired_message="reduced redundancy",
        degraded_message="degraded",
        critical_message="critical degraded",
        detached_message="detached message",
        sources=[
            SimpleNamespace(name="nwws_oi", enabled=True, role="alert_redundant", stale_after_seconds=600, failure_threshold=1, critical=False),
            SimpleNamespace(name="cap_api", enabled=True, role="alert", stale_after_seconds=600, failure_threshold=1, critical=True),
        ],
    )
    health = HealthStateMachine(cfg)
    health.mark_failure("nwws_oi", "xmpp failed")
    health.mark_failure("cap_api", "http failed")

    ctx = asyncio.run(health.evaluate())

    assert ctx.mode == "detached"
    assert ctx.detached_loop_only is True
    assert ctx.notice == "detached message"


def test_health_state_source_impaired_for_redundant_alert_feed_only() -> None:
    cfg = SimpleNamespace(
        enabled=True,
        check_interval_seconds=30,
        min_hold_seconds=0,
        detached_loop_only=True,
        source_impaired_message="reduced redundancy",
        degraded_message="degraded",
        critical_message="critical degraded",
        detached_message="detached message",
        sources=[
            SimpleNamespace(name="nwws_oi", enabled=True, role="alert_redundant", stale_after_seconds=600, failure_threshold=1, critical=False),
            SimpleNamespace(name="cap_api", enabled=True, role="alert", stale_after_seconds=600, failure_threshold=1, critical=True),
        ],
    )
    health = HealthStateMachine(cfg)
    health.mark_success("cap_api")
    health.mark_failure("nwws_oi", "xmpp failed")

    ctx = asyncio.run(health.evaluate())

    assert ctx.mode == "source_impaired"
    assert ctx.notice == "reduced redundancy"
    assert ctx.detached_loop_only is False


def test_health_source_snapshot_does_not_expose_raw_error() -> None:
    cfg = SimpleNamespace(
        enabled=True,
        check_interval_seconds=30,
        min_hold_seconds=0,
        detached_loop_only=True,
        sources=[
            SimpleNamespace(
                name="nwws_oi",
                enabled=True,
                role="alert_redundant",
                stale_after_seconds=600,
                failure_threshold=1,
                critical=False,
            )
        ],
    )
    health = HealthStateMachine(cfg)
    sentinel = "SENTINEL-RAW-SOURCE-ERROR"
    health.mark_failure("nwws_oi", sentinel)

    snapshot = health.source_snapshot("nwws_oi")

    assert snapshot is not None
    assert snapshot["state"] == "degraded"
    assert snapshot["reason"] == "failure_threshold_reached"
    assert sentinel not in repr(snapshot)
