from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .core import SeasonalDatabase


class SegmentRepository:
    def __init__(self, db: SeasonalDatabase) -> None:
        self.db = db

    def replace_entries(self, entries: Iterable[Mapping[str, Any]]) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM cycle_segments")
            for entry in entries:
                self._upsert_unlocked(conn, entry)

    def upsert_entry(self, entry: Mapping[str, Any]) -> None:
        with self.db.transaction() as conn:
            self._upsert_unlocked(conn, entry)

    def load_entries(self) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM cycle_segments ORDER BY segment_key"
            ).fetchall()
        return [
            {
                "key": str(row["segment_key"]),
                "title": str(row["title"]),
                "text": str(row["text"]),
                "audio_path": str(row["audio_path"]),
                "duration_s": float(row["duration_s"] or 0.0),
                "last_updated_ts": float(row["last_updated_ts"] or 0.0),
                "refresh_interval_s": int(row["refresh_interval_s"] or 0),
                "is_placeholder": bool(row["is_placeholder"]),
            }
            for row in rows
        ]

    @staticmethod
    def _upsert_unlocked(conn: Any, entry: Mapping[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO cycle_segments (
                segment_key, title, text, audio_path, duration_s,
                last_updated_ts, refresh_interval_s, is_placeholder
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(segment_key) DO UPDATE SET
                title = excluded.title,
                text = excluded.text,
                audio_path = excluded.audio_path,
                duration_s = excluded.duration_s,
                last_updated_ts = excluded.last_updated_ts,
                refresh_interval_s = excluded.refresh_interval_s,
                is_placeholder = excluded.is_placeholder
            """,
            (
                str(entry["key"]),
                str(entry.get("title") or ""),
                str(entry.get("text") or ""),
                str(entry.get("audio_path") or ""),
                float(entry.get("duration_s") or 0.0),
                float(entry.get("last_updated_ts") or 0.0),
                int(entry.get("refresh_interval_s") or 0),
                1 if bool(entry.get("is_placeholder", False)) else 0,
            ),
        )
