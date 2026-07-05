from __future__ import annotations

import datetime as dt
import inspect

from seasonalweather.config import load_config
from seasonalweather.broadcast import station_feed_runtime as sfr
from seasonalweather.broadcast.service_runtime import SeasonalWeatherServiceRuntime
from seasonalweather.database.core import SeasonalDatabase
from seasonalweather.database.station_feed import StationFeedRepository


def _payload(*, alert_id: str, sender_name: str, sender_kind: str, area: str) -> dict:
    expires = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
    return {
        "id": alert_id,
        "event": "Air Quality Alert",
        "headline": "Air Quality Alert issued July 4 at 4:45PM EDT",
        "severity": "Unknown",
        "urgency": "Unknown",
        "certainty": "Unknown",
        "area": area,
        "effective": expires.isoformat(),
        "ends": expires.isoformat(),
        "expires": expires.isoformat(),
        "sent": expires.isoformat(),
        "sameCodes": [],
        "from": {"name": sender_name, "kind": sender_kind},
        "links": {},
    }


def test_station_feed_cleanup_removes_only_legacy_cap_restore(monkeypatch, tmp_path):
    monkeypatch.setenv("ICECAST_SOURCE_PASSWORD", "test-source")
    cfg = load_config("config/config.yaml")
    db = SeasonalDatabase(path=str(tmp_path / "state.sqlite3"))
    repo = StationFeedRepository(db)
    sfr.set_app_config(cfg)
    sfr.set_repository(repo)
    sfr._STATION_FEED_STATE.clear()

    expires = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
    repo.upsert_alert(
        alert_id="legacy-cap-restore",
        payload=_payload(
            alert_id="legacy-cap-restore",
            sender_name="CAP restore",
            sender_kind="relay",
            area="",
        ),
        expires_at=expires,
    )
    repo.upsert_alert(
        alert_id="real-cap-origin",
        payload=_payload(
            alert_id="real-cap-origin",
            sender_name="NWS CAP",
            sender_kind="origin",
            area="District of Columbia",
        ),
        expires_at=expires,
    )
    repo.upsert_alert(
        alert_id="real-nwws-relay",
        payload=_payload(
            alert_id="real-nwws-relay",
            sender_name="NWWS-OI",
            sender_kind="relay",
            area="Anne Arundel County",
        ),
        expires_at=expires,
    )

    try:
        assert sfr.purge_legacy_synthetic_alerts() == 1
        assert sfr.purge_legacy_synthetic_alerts() == 0
        assert sfr.hydrate_persisted_alerts() == 2
        remaining = {item["id"]: item for item in repo.load_alerts()}
        assert set(remaining) == {"real-cap-origin", "real-nwws-relay"}
        assert remaining["real-cap-origin"]["area"] == "District of Columbia"
        assert remaining["real-cap-origin"]["from"] == {
            "name": "NWS CAP",
            "kind": "origin",
        }
        hydrated = sfr._STATION_FEED_STATE["real-cap-origin"][0]
        assert hydrated.area == "District of Columbia"
        assert hydrated.from_.name == "NWS CAP"
        assert hydrated.from_.kind == "origin"
    finally:
        sfr._STATION_FEED_STATE.clear()
        sfr.set_app_config(None)
        sfr.set_repository(None)


def test_service_startup_does_not_rebuild_station_feed_from_alert_tracker() -> None:
    source = inspect.getsource(SeasonalWeatherServiceRuntime.run)
    assert "seed_from_alert_tracker" not in source
    assert "restored %d alerts from AlertTracker" not in source
    assert "purge_legacy_synthetic_alerts" in source
    assert "hydrate_persisted_alerts" in source
