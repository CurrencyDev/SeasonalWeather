import datetime as dt

from seasonalweather.alerts.active import ActiveAlert, AlertTracker


def _alert(id_, source, event, code, vtec=None, issued="2026-06-06T00:00:00+00:00", same_locs=None):
    return ActiveAlert(
        id=id_,
        source=source,
        event=event,
        code=code,
        vtec=list(vtec or []),
        headline=event,
        script_text=event,
        audio_path=None,
        expires="2026-06-06T12:00:00+00:00",
        issued=issued,
        same_locs=list(same_locs or ["024003"]),
        cycle_only=True,
    )


def test_active_alerts_sort_by_significance_not_arrival_order(tmp_path):
    tracker = AlertTracker(tmp_path / "ignored.json", database=None)
    tracker.add_or_update(_alert("pns", "PNS_CYCLE", "Severe Weather Safety Rules", "PNS", issued="2026-06-06T00:00:00+00:00"))
    tracker.add_or_update(_alert("advisory", "CAP", "Special Weather Statement", "SPS", issued="2026-06-06T00:01:00+00:00"))
    tracker.add_or_update(_alert("watch", "CAP", "Severe Thunderstorm Watch", "SVA", issued="2026-06-06T00:02:00+00:00"))
    tracker.add_or_update(_alert("warning", "CAP", "Severe Thunderstorm Warning", "SVR", issued="2026-06-06T00:03:00+00:00"))

    assert [a.id for a in tracker.get_cycle_alerts(dt.datetime(2026, 6, 6, 1, tzinfo=dt.timezone.utc))] == [
        "warning",
        "watch",
        "advisory",
        "pns",
    ]


def test_authoritative_alert_removes_shadowed_ern_relay(tmp_path):
    tracker = AlertTracker(tmp_path / "ignored.json", database=None)
    tracker.add_or_update(_alert("ern", "ERN", "Severe Thunderstorm Watch", "SVA", same_locs=["024003"]))
    tracker.add_or_update(_alert("cap", "CAP", "Severe Thunderstorm Watch", "SVA", same_locs=["024003", "024031"]))

    assert [a.id for a in tracker.get_cycle_alerts(dt.datetime(2026, 6, 6, 1, tzinfo=dt.timezone.utc))] == ["cap"]
