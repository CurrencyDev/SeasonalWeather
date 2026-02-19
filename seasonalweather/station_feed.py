from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Literal, Optional


SenderKind = Literal["relay", "origin", "unknown"]


@dataclass(frozen=True)
class FeedSender:
    name: str
    kind: SenderKind = "unknown"


@dataclass(frozen=True)
class StationFeedAlert:
    id: str
    event: str
    headline: str
    severity: str
    urgency: str
    certainty: str
    area: str
    effective: Optional[str]
    ends: Optional[str]
    expires: Optional[str]
    sent: Optional[str]
    sameCodes: List[str]
    from_: Optional[FeedSender] = None
    links: Optional[Dict[str, str]] = None


def _clamp(s: Any, max_len: int) -> str:
    t = s if isinstance(s, str) else ""
    return t if len(t) <= max_len else t[:max_len]


def _uniq_strs(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen = set()
    for v in values:
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """
    Writes JSON atomically: write temp file, fsync, rename over target.
    Safe against partial writes and power loss mid-write.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp = f"{path}.tmp.{os.getpid()}.{int(time.time()*1000)}"
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def build_station_feed_payload(
    *,
    station_id: str,
    source: str,
    generated_at_iso: str,
    alerts: Iterable[StationFeedAlert],
) -> Dict[str, Any]:
    out_alerts: List[Dict[str, Any]] = []
    for a in alerts:
        d = asdict(a)
        # rename from_ -> from
        sender = d.pop("from_")
        if sender is not None:
            d["from"] = sender
        # hard clamp a few strings to keep payload sane
        d["id"] = _clamp(d.get("id"), 300)
        d["event"] = _clamp(d.get("event"), 120) or "Alert"
        d["headline"] = _clamp(d.get("headline"), 220)
        d["severity"] = _clamp(d.get("severity"), 24) or "Unknown"
        d["urgency"] = _clamp(d.get("urgency"), 24) or "Unknown"
        d["certainty"] = _clamp(d.get("certainty"), 24) or "Unknown"
        d["area"] = _clamp(d.get("area"), 320)
        d["sameCodes"] = _uniq_strs(d.get("sameCodes"))
        out_alerts.append(d)

    return {
        "stationId": station_id,
        "generatedAt": generated_at_iso,
        "source": source,
        "alerts": out_alerts,
    }
