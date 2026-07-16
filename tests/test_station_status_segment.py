import asyncio
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from seasonalweather.alerts.active import ActiveAlert
from seasonalweather.broadcast.cycle import CycleBuilder, CycleContext, build_station_status_text
from seasonalweather.broadcast.segment_refresher import SegmentRefresher


def _alert(
    alert_id: str,
    event: str,
    *,
    source: str = "CAP",
    code: str = "SVR",
    watch_number: int | None = None,
) -> ActiveAlert:
    return ActiveAlert(
        id=alert_id,
        source=source,
        event=event,
        code=code,
        vtec=[],
        headline=event,
        script_text=event,
        audio_path=None,
        expires="2099-01-01T00:00:00+00:00",
        issued="2026-07-16T00:00:00+00:00",
        same_locs=["024003"],
        cycle_only=True,
        watch_number=watch_number,
    )


def test_station_status_uses_clear_normal_mode_wording() -> None:
    ctx = CycleContext(
        mode="normal",
        last_heightened_ago="two hours",
        last_product_desc=None,
    )

    text = build_station_status_text(ctx, ())

    assert text == (
        "And now, the station status and active alerts. "
        "SeasonalWeather is currently operating in normal broadcast mode. "
        "No active alerts are currently being tracked for the service area."
    )
    assert "two hours" not in text


def test_station_status_summarizes_local_alerts_in_tracker_order() -> None:
    alerts = (
        _alert("tor", "Tornado Warning", code="TOR"),
        _alert("svr-1", "Severe Thunderstorm Warning"),
        _alert("svr-2", "Severe Thunderstorm Warning"),
        _alert("toa", "Tornado Watch", code="TOA", watch_number=123),
        _alert("pns", "Severe Weather Safety Rules", source="PNS_CYCLE", code="PNS"),
    )
    ctx = CycleContext(
        mode="heightened",
        last_heightened_ago="twelve minutes",
        last_product_desc=None,
        active_alerts=alerts,
    )

    text = build_station_status_text(ctx, ctx.active_alerts)

    assert "Heightened mode was activated twelve minutes ago." in text
    assert (
        "The active alerts in the service area are: Tornado Warning; "
        "two Severe Thunderstorm Warnings; Tornado Watch number 123."
    ) in text
    assert "Severe Weather Safety Rules" not in text


def test_segment_refresher_builds_status_without_full_cycle_fetch() -> None:
    class Builder:
        def __init__(self) -> None:
            self.status_calls = 0

        def build_status_text(self, ctx: CycleContext) -> str:
            self.status_calls += 1
            return "local station status"

        async def build_segments(self, *args, **kwargs):
            raise AssertionError("status refresh must not invoke the full network-backed cycle builder")

    builder = Builder()
    refresher = SegmentRefresher(
        store=SimpleNamespace(),
        cycle_builder=builder,  # type: ignore[arg-type]
        tts=SimpleNamespace(),
        alert_tracker=SimpleNamespace(),
        ctx_fn=lambda: CycleContext(mode="normal", last_heightened_ago=None, last_product_desc=None),
        station_name="Test",
        service_area_name="Test area",
        disclaimer="Test disclaimer.",
        tz=ZoneInfo("America/New_York"),
        sample_rate=48000,
    )
    captured: dict[str, object] = {}

    async def capture_synth(**kwargs) -> None:
        captured.update(kwargs)

    refresher._synth = capture_synth  # type: ignore[method-assign]

    asyncio.run(refresher._refresh_one("status"))

    assert builder.status_calls == 1
    assert captured["key"] == "status"
    assert captured["text"] == "local station status"


def test_cycle_builder_status_method_does_not_touch_nws_api() -> None:
    class NoNetworkApi:
        def __getattr__(self, name: str):
            raise AssertionError(f"unexpected NWS API access: {name}")

    builder = CycleBuilder(
        api=NoNetworkApi(),  # type: ignore[arg-type]
        tz_name="America/New_York",
        obs_stations=[],
        reference_points=[],
        same_fips_all=["024003"],
        cycle_cfg=None,
    )
    ctx = CycleContext(
        mode="normal",
        last_heightened_ago=None,
        last_product_desc=None,
        active_alerts=(_alert("tor", "Tornado Warning", code="TOR"),),
    )

    text = builder.build_status_text(ctx)

    assert "The active alert in the service area is Tornado Warning." in text
