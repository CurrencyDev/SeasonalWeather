from __future__ import annotations

import datetime as dt

from seasonalweather.database.core import SeasonalDatabase
from seasonalweather.database.station_feed import StationFeedRepository
from seasonalweather.database.schema import SCHEMA_VERSION


def test_station_feed_repository_persists_and_prunes(tmp_path):
    db = SeasonalDatabase(path=str(tmp_path / "state.sqlite3"))
    repo = StationFeedRepository(db)

    now = dt.datetime(2026, 5, 6, 16, 0, tzinfo=dt.timezone.utc)
    future = now + dt.timedelta(minutes=30)
    past = now - dt.timedelta(minutes=30)

    repo.upsert_alert(
        alert_id="future-alert",
        payload={"id": "future-alert", "event": "Test", "expires": future.isoformat()},
        expires_at=future,
    )
    repo.upsert_alert(
        alert_id="past-alert",
        payload={"id": "past-alert", "event": "Old", "expires": past.isoformat()},
        expires_at=past,
    )

    assert [item["id"] for item in repo.load_alerts(now=now)] == ["future-alert"]
    assert repo.prune_expired(now=now) == 1
    assert [item["id"] for item in repo.load_alerts()] == ["future-alert"]


def test_database_schema_includes_station_feed_table(tmp_path):
    db = SeasonalDatabase(path=str(tmp_path / "state.sqlite3"))
    db.bootstrap()

    with db.connect() as conn:
        version = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='station_feed_alerts'"
        ).fetchone()

    assert version == SCHEMA_VERSION
    assert table is not None
