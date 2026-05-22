from __future__ import annotations

import datetime as dt
import json
import os
from types import SimpleNamespace

from seasonalweather.database.core import SeasonalDatabase
from seasonalweather.database.housekeeping import DatabaseHousekeeper
from seasonalweather.database.segments import SegmentRepository


def _cfg(tmp_path, *, retention_seconds: int = 3600, max_bytes: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(
            work_dir=str(tmp_path),
            audio_dir=str(tmp_path / "audio"),
        ),
        database=SimpleNamespace(
            housekeeping=SimpleNamespace(
                enabled=True,
                interval_seconds=60,
                startup_delay_seconds=0,
                api_command_retention_days=14,
                audio_asset_grace_seconds=60,
                generated_audio_retention_seconds=retention_seconds,
                generated_audio_max_bytes=max_bytes,
                tmp_file_grace_seconds=60,
                wal_checkpoint=False,
            )
        ),
    )


def _touch(path, *, age_seconds: int, size: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    ts = dt.datetime.now(dt.timezone.utc).timestamp() - age_seconds
    os.utime(path, (ts, ts))


def test_housekeeper_prunes_old_unreferenced_generated_audio(tmp_path) -> None:
    cfg = _cfg(tmp_path, retention_seconds=3600)
    db = SeasonalDatabase(path=str(tmp_path / "state.sqlite3"))
    audio_dir = tmp_path / "audio"

    stale = audio_dir / "alert_20260520-110034.wav"
    fresh = audio_dir / "alert_20260522-110034.wav"
    unknown = audio_dir / "operator-kept.wav"
    _touch(stale, age_seconds=7200)
    _touch(fresh, age_seconds=120)
    _touch(unknown, age_seconds=7200)

    stats = DatabaseHousekeeper(cfg, db).run_once()

    assert stats["generated_audio_deleted"] == 1
    assert not stale.exists()
    assert fresh.exists()
    assert unknown.exists()


def test_housekeeper_keeps_db_referenced_audio_even_when_old(tmp_path) -> None:
    cfg = _cfg(tmp_path, retention_seconds=3600)
    db = SeasonalDatabase(path=str(tmp_path / "state.sqlite3"))
    audio_dir = tmp_path / "audio"

    active_wav = audio_dir / "alert_20260520-110034.wav"
    segment_wav = audio_dir / "cycle_seg_fcst.wav"
    stale_wav = audio_dir / "rebcast_20260520-110034.wav"
    _touch(active_wav, age_seconds=7200)
    _touch(segment_wav, age_seconds=7200)
    _touch(stale_wav, age_seconds=7200)

    now = dt.datetime.now(dt.timezone.utc)
    future = (now + dt.timedelta(hours=1)).replace(microsecond=0).isoformat()
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO active_alerts (
                alert_id, source, event, code, headline, script_text, audio_path,
                expires_at, issued_at, cycle_only, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "alert-1", "CAP", "Test", "SVR", "Test", "Test script", str(active_wav),
                future, future, 0, future, future,
            ),
        )
        conn.execute(
            """
            INSERT INTO station_feed_alerts (
                alert_id, expires_at, payload_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "feed-1", future,
                json.dumps({"id": "feed-1", "links": {"wav": str(active_wav)}}),
                future, future,
            ),
        )

    SegmentRepository(db).upsert_entry(
        {
            "key": "fcst",
            "title": "Forecast",
            "text": "Forecast text",
            "audio_path": str(segment_wav),
            "duration_s": 1.0,
            "last_updated_ts": now.timestamp(),
            "refresh_interval_s": 300,
        }
    )

    stats = DatabaseHousekeeper(cfg, db).run_once()

    assert stats["generated_audio_deleted"] == 1
    assert active_wav.exists()
    assert segment_wav.exists()
    assert not stale_wav.exists()


def test_housekeeper_size_cap_deletes_oldest_unreferenced_audio(tmp_path) -> None:
    cfg = _cfg(tmp_path, retention_seconds=86400, max_bytes=10)
    db = SeasonalDatabase(path=str(tmp_path / "state.sqlite3"))
    audio_dir = tmp_path / "audio"

    oldest = audio_dir / "capvoice_20260520-100000.wav"
    newer = audio_dir / "capvoice_20260520-110000.wav"
    newest = audio_dir / "capvoice_20260520-120000.wav"
    _touch(oldest, age_seconds=300, size=6)
    _touch(newer, age_seconds=200, size=6)
    _touch(newest, age_seconds=100, size=6)

    stats = DatabaseHousekeeper(cfg, db).run_once()

    assert stats["generated_audio_deleted"] == 2
    assert not oldest.exists()
    assert not newer.exists()
    assert newest.exists()
