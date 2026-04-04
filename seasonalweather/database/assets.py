from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .core import SeasonalDatabase


class AudioAssetRepository:
    def __init__(self, db: SeasonalDatabase) -> None:
        self.db = db

    def upsert_asset(self, record: Mapping[str, Any]) -> None:
        meta = dict(record)
        meta_json = json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO audio_assets (
                    asset_id, wav_path, original_filename, content_type, sha256,
                    headline, event_code, actor, created_at, expires_at, meta_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id) DO UPDATE SET
                    wav_path = excluded.wav_path,
                    original_filename = excluded.original_filename,
                    content_type = excluded.content_type,
                    sha256 = excluded.sha256,
                    headline = excluded.headline,
                    event_code = excluded.event_code,
                    actor = excluded.actor,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at,
                    meta_json = excluded.meta_json
                """,
                (
                    str(record["asset_id"]),
                    str(record.get("path") or record.get("wav_path") or ""),
                    str(record.get("filename") or record.get("original_filename") or "") or None,
                    str(record.get("content_type") or "") or None,
                    str(record.get("sha256") or "") or None,
                    str(record.get("headline") or "") or None,
                    str(record.get("event_code") or "") or None,
                    str(record.get("actor") or "") or None,
                    str(record.get("uploaded_at") or record.get("created_at") or ""),
                    str(record.get("expires_at") or "") or None,
                    meta_json,
                ),
            )


    def list_live_assets(self, now_iso: str) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT asset_id, wav_path, expires_at FROM audio_assets WHERE expires_at IS NULL OR expires_at > ? ORDER BY created_at",
                (now_iso,),
            ).fetchall()
        return [
            {
                "asset_id": str(row["asset_id"]),
                "wav_path": str(row["wav_path"]),
                "expires_at": row["expires_at"],
            }
            for row in rows
        ]

    def list_expired_assets(self, now_iso: str) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT asset_id, wav_path, expires_at FROM audio_assets WHERE expires_at IS NOT NULL AND expires_at <= ? ORDER BY expires_at, asset_id",
                (now_iso,),
            ).fetchall()
        return [
            {
                "asset_id": str(row["asset_id"]),
                "wav_path": str(row["wav_path"]),
                "expires_at": row["expires_at"],
            }
            for row in rows
        ]

    def delete_assets(self, asset_ids: list[str]) -> int:
        ids = [str(v).strip() for v in asset_ids if str(v).strip()]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self.db.transaction() as conn:
            cur = conn.execute(
                f"DELETE FROM audio_assets WHERE asset_id IN ({placeholders})",
                tuple(ids),
            )
        return int(cur.rowcount or 0)

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM audio_assets WHERE asset_id = ?", (asset_id,)).fetchone()
        if row is None:
            return None
        payload = json.loads(row["meta_json"] or "{}")
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("asset_id", str(row["asset_id"]))
        payload.setdefault("path", str(row["wav_path"]))
        if row["original_filename"] is not None:
            payload.setdefault("filename", str(row["original_filename"]))
        if row["content_type"] is not None:
            payload.setdefault("content_type", str(row["content_type"]))
        if row["sha256"] is not None:
            payload.setdefault("sha256", str(row["sha256"]))
        if row["headline"] is not None:
            payload.setdefault("headline", str(row["headline"]))
        if row["event_code"] is not None:
            payload.setdefault("event_code", str(row["event_code"]))
        if row["actor"] is not None:
            payload.setdefault("actor", str(row["actor"]))
        payload.setdefault("uploaded_at", str(row["created_at"]))
        if row["expires_at"] is not None:
            payload.setdefault("expires_at", str(row["expires_at"]))
        return payload
