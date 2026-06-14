from __future__ import annotations

import datetime as dt
import re
from typing import Any, Callable

from ..alerts.active import _vtec_track_id
from ..alerts.vtec import (
    same_codes_for_vtec as _vtec_same_codes_for_vtec,
    toneout_policy as _vtec_toneout_policy,
)


def cap_vtec_list(ev: Any) -> list[str]:
    """Return normalized, deduplicated VTEC strings from a CAP event."""
    vals: list[str] = []

    # Prefer explicit ev.vtec when cap_nws populates it.
    v0 = getattr(ev, "vtec", None)
    if isinstance(v0, (list, tuple)):
        vals.extend(str(x).strip() for x in v0 if str(x).strip())
    elif isinstance(v0, str) and v0.strip():
        vals.append(v0.strip())

    # Back-compat: still accept alternative attributes if present.
    for attr in ("vtec_codes", "vtec_code", "vtecList"):
        v = getattr(ev, attr, None)
        if not v:
            continue
        if isinstance(v, str):
            vals.append(v.strip())
        elif isinstance(v, (list, tuple)):
            vals.extend(str(x).strip() for x in v if str(x).strip())

    params = getattr(ev, "parameters", None)
    if isinstance(params, dict):
        for k, v in params.items():
            if str(k).strip().upper() != "VTEC":
                continue
            if isinstance(v, str):
                vals.append(v.strip())
            elif isinstance(v, (list, tuple)):
                vals.extend(str(x).strip() for x in v if str(x).strip())

    out: list[str] = []
    seen: set[str] = set()
    for x in vals:
        x2 = "".join(str(x).split()).strip()
        if not x2 or x2 in seen:
            continue
        seen.add(x2)
        out.append(x2)
    return out[:12]


def cap_is_actionable(ev: Any) -> bool:
    try:
        if str(ev.status or "").strip().lower() != "actual":
            return False
        mt = str(ev.message_type or "").strip().lower()
        if mt and mt not in {"alert", "update", "cancel"}:
            return False
    except Exception:
        return False
    return True


def cap_severity_str(ev: Any) -> str:
    return str(getattr(ev, "severity", None) or "").strip().lower()


_CAP_EVENT_TO_SAME_CODE: dict[str, str] = {
    "Tornado Warning": "TOR",
    "Tornado Watch": "TOA",
    "Severe Thunderstorm Warning": "SVR",
    "Severe Thunderstorm Watch": "SVA",
    "Flash Flood Warning": "FFW",
    "Flash Flood Watch": "FFA",
    "Flood Warning": "FLW",
    "Flood Watch": "FLA",
    "Flood Advisory": "FLA",
    "Winter Storm Warning": "WSW",
    "Winter Storm Watch": "WSA",
    "Blizzard Warning": "BZW",
    "Blizzard Watch": "BZA",
    "Ice Storm Warning": "ISW",
    "Ice Storm Watch": "ISA",
    "Freeze Warning": "FZW",
    "Freeze Watch": "FZA",
    "Flash Freeze Warning": "FSW",
    "Winter Weather Advisory": "SPS",
    "Hurricane Warning": "HUW",
    "Hurricane Watch": "HUA",
    "Tropical Storm Warning": "TRW",
    "Tropical Storm Watch": "TRA",
    "Storm Surge Warning": "SSW",
    "Storm Surge Watch": "SSA",
    "High Wind Warning": "HWW",
    "High Wind Watch": "HWA",
    "Extreme Wind Warning": "EWW",
    "Wind Chill Warning": "WCW",
    "Wind Chill Watch": "WCA",
    "Snow Squall Warning": "SQW",
    "Special Weather Statement": "SPS",
    "Severe Weather Statement": "SVS",
    "Flood Statement": "FLS",
    "Flash Flood Statement": "FFS",
    "Hurricane Statement": "HLS",
}


def _parse_vtec_dt_utc(token: str) -> dt.datetime | None:
    """Parse a VTEC UTC timestamp token like ``260614T0930Z`` or ``20260614T0930Z``."""
    token = (token or "").strip().upper()
    m = re.match(r"^(\d{6}|\d{8})T(\d{4})Z$", token)
    if not m:
        return None

    d = m.group(1)
    hm = m.group(2)
    try:
        if len(d) == 8:
            year = int(d[0:4])
            month = int(d[4:6])
            day = int(d[6:8])
        else:
            year = 2000 + int(d[0:2])
            month = int(d[2:4])
            day = int(d[4:6])

        hour = int(hm[0:2])
        minute = int(hm[2:4])
        return dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc)
    except Exception:
        return None


def best_expiry_from_vtec(vtec_list: list[str]) -> dt.datetime | None:
    """Return the latest END time found across VTEC codes, in UTC."""
    ends: list[dt.datetime] = []
    for raw in vtec_list or []:
        s = "".join(str(raw).split()).strip()
        if not s:
            continue

        # Pull the END token from the VTEC time pair: ...-YYYYMMDDThhmmZ/
        m = re.search(r"-((?:\d{8}|\d{6})T\d{4}Z)", s)
        if not m:
            continue

        parsed = _parse_vtec_dt_utc(m.group(1))
        if parsed is not None:
            ends.append(parsed)

    if not ends:
        return None
    return max(ends)


def cap_event_to_same_code(event: str) -> str:
    e = (event or "").strip()
    if e in _CAP_EVENT_TO_SAME_CODE:
        return _CAP_EVENT_TO_SAME_CODE[e]

    words = [w for w in re.split(r"\s+", e) if w]
    if words:
        code = "".join(ch for ch in "".join(w[0] for w in words[:3]) if ch.isalnum()).upper()
        if len(code) >= 3:
            return code[:3]
    return "SPS"


def vtec_matches_configured_toneout_code(cfg: Any, vtec: list[str]) -> bool:
    """True when VTEC maps to a configured toneout event code."""
    if not vtec:
        return False
    allowed_codes = {
        str(x).strip().upper()
        for x in getattr(cfg.policy, "toneout_product_types", [])
        if str(x).strip()
    }
    if not allowed_codes:
        return False
    return bool(set(_vtec_same_codes_for_vtec(vtec)) & allowed_codes)


def cap_should_full(cfg: Any, ev: Any) -> bool:
    if not cfg.cap.full.enabled:
        return False
    if not cap_is_actionable(ev):
        return False

    # CAP Update may carry VTEC CON/EXT/COR/ROU. Do not FULL-tone those
    # unless VTEC policy says this lifecycle action is FULL-worthy.
    try:
        mt = str(ev.message_type or "").strip().lower()
    except Exception:
        mt = ""
    if mt == "update":
        try:
            if _vtec_toneout_policy(cap_vtec_list(ev)).mode != "FULL":
                return False
        except Exception:
            return False

    event = (ev.event or "").strip()
    full_events = {str(x).strip() for x in cfg.cap.full.events if str(x).strip()}
    if event and event in full_events:
        return True

    sev = cap_severity_str(ev)
    full_severities = {str(x).strip().lower() for x in cfg.cap.full.severities if str(x).strip()}
    if sev and sev in full_severities:
        return True

    return False


def cap_should_voice(cfg: Any, ev: Any) -> bool:
    if not cfg.cap.voice.enabled:
        return False
    if not cap_is_actionable(ev):
        return False
    allow_events = {str(x).strip() for x in cfg.cap.voice.events if str(x).strip()}
    if allow_events and (ev.event or "").strip() not in allow_events:
        return False
    return True


def cap_should_update(cfg: Any, ev: Any, vtec_tracks: Callable[[list[str]], list[tuple[str, str]]]) -> bool:
    """True for CAP Update/Cancel with CON/EXT/CAN/EXP actions on watched events."""
    if not cfg.cap.full.enabled:
        return False
    if not cap_is_actionable(ev):
        return False
    mt = str(ev.message_type or "").strip().lower()
    if mt not in {"update", "cancel"}:
        return False

    event = (ev.event or "").strip()
    vtec = cap_vtec_list(ev)
    full_events = {str(x).strip() for x in cfg.cap.full.events if str(x).strip()}
    if event not in full_events and not vtec_matches_configured_toneout_code(cfg, vtec):
        return False

    tracks = vtec_tracks(vtec)
    update_actions = {"CON", "EXT", "CAN", "EXP"}
    vtec_actions = {act for (_track, act) in tracks} if tracks else set()
    return bool(vtec_actions & update_actions)


def alert_tracker_id_for_cap(ev: Any, same_code: str) -> str:
    """Return a stable AlertTracker ID for a CAP event."""
    for v in cap_vtec_list(ev):
        tid = _vtec_track_id(v)
        if tid:
            return f"CAP:{tid}"
    return f"CAP:{(ev.alert_id or '').strip()}"


def alert_expires_from_cap(
    ev: Any,
    vtec: list[str],
    *,
    best_expiry_from_vtec: Callable[[list[str]], dt.datetime | None],
) -> str:
    """Best-effort expiry ISO string from VTEC end time or CAP expires field."""
    exp_utc = best_expiry_from_vtec(vtec)
    if exp_utc:
        return exp_utc.isoformat()
    raw = getattr(ev, "expires", None) or getattr(ev, "ends", None)
    if raw:
        return str(raw).strip()
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=6)).isoformat()
