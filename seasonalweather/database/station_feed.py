from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Mapping
from typing import Any

from .core import SeasonalDatabase


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _to_utc_iso(value: Any) -> str:
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        raw = str(value or "").strip()
        if not raw:
            parsed = _utc_now()
        else:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            try:
                parsed = dt.datetime.fromisoformat(raw)
            except Exception:
                parsed = _utc_now()

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    else:
        parsed = parsed.astimezone(dt.timezone.utc)
    return parsed.replace(microsecond=0).isoformat()


class StationFeedRepository:
    """SQLite-backed read model for the public station handled-alerts feed."""

    def __init__(self, db: SeasonalDatabase) -> None:
        self.db = db

    def upsert_alert(self, *, alert_id: str, payload: Mapping[str, Any], expires_at: Any) -> None:
        aid = str(alert_id or "").strip()
        if not aid:
            return
        now_iso = _to_utc_iso(_utc_now())
        expires_iso = _to_utc_iso(expires_at)
        payload_json = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO station_feed_alerts (
                    alert_id, expires_at, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(alert_id) DO UPDATE SET
                    expires_at = excluded.expires_at,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (aid, expires_iso, payload_json, now_iso, now_iso),
            )

    def import_alerts(self, alerts: Iterable[Mapping[str, Any]]) -> int:
        count = 0
        now_iso = _to_utc_iso(_utc_now())
        with self.db.transaction() as conn:
            for payload in alerts:
                aid = str(payload.get("id") or "").strip()
                if not aid:
                    continue
                expires_raw = payload.get("expires") or payload.get("ends")
                expires_iso = _to_utc_iso(expires_raw)
                payload_json = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                conn.execute(
                    """
                    INSERT INTO station_feed_alerts (
                        alert_id, expires_at, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(alert_id) DO UPDATE SET
                        expires_at = excluded.expires_at,
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (aid, expires_iso, payload_json, now_iso, now_iso),
                )
                count += 1
        return count

    def delete_alerts(self, alert_ids: Iterable[str]) -> int:
        ids = [str(x or "").strip() for x in alert_ids]
        ids = [x for x in ids if x]
        if not ids:
            return 0
        with self.db.transaction() as conn:
            cur = conn.executemany("DELETE FROM station_feed_alerts WHERE alert_id = ?", [(x,) for x in ids])
        # sqlite3.executemany() rowcount may be -1 on some builds; calculate conservatively.
        return max(0, int(getattr(cur, "rowcount", 0) or 0))

    def prune_expired(self, *, now: Any | None = None, grace_seconds: int = 0) -> int:
        cutoff = _utc_now() if now is None else (now if isinstance(now, dt.datetime) else None)
        if cutoff is None:
            cutoff_iso = _to_utc_iso(now)
        else:
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=dt.timezone.utc)
            cutoff = cutoff.astimezone(dt.timezone.utc) - dt.timedelta(seconds=max(0, int(grace_seconds or 0)))
            cutoff_iso = _to_utc_iso(cutoff)
        with self.db.transaction() as conn:
            cur = conn.execute("DELETE FROM station_feed_alerts WHERE expires_at < ?", (cutoff_iso,))
        return int(cur.rowcount or 0)

    def trim_to_max(self, max_items: int) -> int:
        limit = max(0, int(max_items or 0))
        if limit <= 0:
            return 0
        with self.db.transaction() as conn:
            rows = conn.execute(
                """
                SELECT alert_id
                FROM station_feed_alerts
                ORDER BY expires_at DESC, updated_at DESC, alert_id DESC
                LIMIT -1 OFFSET ?
                """,
                (limit,),
            ).fetchall()
            ids = [str(row["alert_id"]) for row in rows]
            if not ids:
                return 0
            conn.executemany("DELETE FROM station_feed_alerts WHERE alert_id = ?", [(x,) for x in ids])
        return len(ids)

    def load_alerts(self, *, now: Any | None = None, max_items: int | None = None) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if now is not None:
            where = "WHERE expires_at >= ?"
            params.append(_to_utc_iso(now))
        limit_clause = ""
        if max_items is not None and int(max_items or 0) > 0:
            limit_clause = "LIMIT ?"
            params.append(int(max_items))
        with self.db.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload_json
                FROM station_feed_alerts
                {where}
                ORDER BY expires_at DESC, updated_at DESC, alert_id DESC
                {limit_clause}
                """,
                params,
            ).fetchall()
        alerts: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"] or "{}"))
            except Exception:
                continue
            if isinstance(payload, dict):
                alerts.append(payload)
        return alerts
