from __future__ import annotations

import datetime as dt
from pathlib import Path

from seasonalweather.database.core import SeasonalDatabase
from seasonalweather.database.inserts import CycleInsertRepository
from seasonalweather.database.housekeeping import DatabaseHousekeeper


def _iso(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()


def _record(tmp_path: Path, *, insert_id: str = "ins_test", defer: bool = True) -> dict:
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    audio = tmp_path / f"{insert_id}.wav"
    audio.write_bytes(b"not-a-real-wav-for-repository-test")
    return {
        "insert_id": insert_id,
        "kind": "text",
        "title": "Maintenance notice.",
        "text": "Maintenance notice.",
        "audio_path": str(audio),
        "audio_asset_id": None,
        "placement": "after_time",
        "start_after": _iso(now - dt.timedelta(minutes=1)),
        "expires_at": _iso(now + dt.timedelta(hours=1)),
        "repeat_mode": "every_n_rotations",
        "repeat_every_rotations": 2,
        "max_airings": 3,
        "defer_during_active_alerts": defer,
        "status": "active",
        "actor": "test",
        "created_at": _iso(now),
        "updated_at": _iso(now),
        "last_aired_at": None,
        "airing_count": 0,
        "last_aired_rotation": None,
        "duration_seconds": 1.0,
        "meta": {"source": "test"},
    }


def test_cycle_insert_repository_due_repeat_and_focus_defer(tmp_path) -> None:
    db = SeasonalDatabase(path=str(tmp_path / "state.sqlite3"))
    repo = CycleInsertRepository(db)
    repo.upsert_insert(_record(tmp_path))

    now_iso = _iso(dt.datetime.now(dt.timezone.utc))
    assert [item["insert_id"] for item in repo.list_due(
        placement="after_time",
        rotation_count=1,
        now_iso=now_iso,
        active_alert_focus=False,
    )] == ["ins_test"]
    assert repo.list_due(
        placement="after_time",
        rotation_count=1,
        now_iso=now_iso,
        active_alert_focus=True,
    ) == []

    aired = repo.mark_aired(insert_id="ins_test", aired_at=now_iso, rotation_count=1)
    assert aired is not None
    assert aired["airing_count"] == 1
    assert repo.list_due(
        placement="after_time",
        rotation_count=2,
        now_iso=now_iso,
        active_alert_focus=False,
    ) == []
    assert [item["insert_id"] for item in repo.list_due(
        placement="after_time",
        rotation_count=3,
        now_iso=now_iso,
        active_alert_focus=False,
    )] == ["ins_test"]


def test_cycle_insert_repository_marks_complete_at_max_airings(tmp_path) -> None:
    db = SeasonalDatabase(path=str(tmp_path / "state.sqlite3"))
    repo = CycleInsertRepository(db)
    repo.upsert_insert(_record(tmp_path, insert_id="ins_once", defer=False) | {
        "repeat_mode": "once",
        "repeat_every_rotations": 1,
        "max_airings": 1,
    })

    now_iso = _iso(dt.datetime.now(dt.timezone.utc))
    marked = repo.mark_aired(insert_id="ins_once", aired_at=now_iso, rotation_count=10)
    assert marked is not None
    assert marked["status"] == "completed"
    assert repo.list_due(
        placement="after_time",
        rotation_count=11,
        now_iso=now_iso,
        active_alert_focus=False,
    ) == []
