from __future__ import annotations

import datetime as dt
from pathlib import Path

from seasonalweather.alerts.active import ActiveAlert
from seasonalweather.config import load_config
from seasonalweather.broadcast import station_feed_runtime as sfr


class _Tracker:
    def __init__(self, items):
        self._items = list(items)

    def get_cycle_alerts(self):
        return list(self._items)


def test_station_feed_seed_from_alert_tracker_uses_runtime_state(monkeypatch, tmp_path):
    monkeypatch.setenv("ICECAST_SOURCE_PASSWORD", "test-source")
    cfg = load_config("config/config.yaml")
    sfr.set_app_config(cfg)
    sfr.set_repository(None)
    sfr._STATION_FEED_STATE.clear()

    expires = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
    issued = expires - dt.timedelta(minutes=30)
    tracker = _Tracker([
        ActiveAlert(
            id="NWWS:KLWX.SV.W.0123",
            source="NWWS",
            event="Severe Thunderstorm Warning",
            code="SVR",
            vtec=["/O.NEW.KLWX.SV.W.0123.260614T1200Z-260614T1300Z/"],
            headline="Severe Thunderstorm Warning",
            script_text="Severe Thunderstorm Warning for Anne Arundel County.",
            audio_path=str(tmp_path / "svr.wav"),
            expires=expires.isoformat(),
            issued=issued.isoformat(),
            same_locs=["024003"],
            cycle_only=False,
        )
    ])

    try:
        assert sfr.seed_from_alert_tracker(tracker) == 1
        assert "NWWS:KLWX.SV.W.0123" in sfr._STATION_FEED_STATE
    finally:
        sfr._STATION_FEED_STATE.clear()
        sfr.set_app_config(None)
        sfr.set_repository(None)
