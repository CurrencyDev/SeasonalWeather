from __future__ import annotations
import json

# =========================================================================================
#      MP"""""`MM                                                       dP              MM'"""'YMM
#      M  mmmmm..M                                                       88              M' .mmm. `M
#      M.      `YM .d8888b. .d8888b. .d8888b. .d8888b. 88d888b. .d8888b. 88              M  MMMMMooM dP    dP 88d888b. 88d888b. .d8888b. 88d888b. .d8888b. dP    dP
#      MMMMMMM.  M 88ooood8 88'  `88 Y8ooooo. 88'  `88 88'  `88 88'  `88 88              M  MMMMMMMM 88    88 88'  `88 88'  `88 88ooood8 88'  `88 88'  `"" 88    88
#      M. .MMM'  M 88.  ... 88.  .88       88 88.  .88 88    88 88.  .88 88              M. `MMM' .M 88.  .88 88       88       88.  ... 88    88 88.  ... 88.  .88
#      Mb.     .dM `88888P' `88888P8 `88888P' `88888P' dP    dP `88888P8 dP              MM.     .dM `88888P' dP       dP       `88888P' dP    dP `88888P' `8888P88
#      MMMMMMMMMMM                                                Seasonal_Currency      MMMMMMMMMMM                                                            .88
#                                                                                                                                                           d8888P.
# =========================================================================================

import argparse
import asyncio
import datetime as dt
import hashlib
import logging
import math
import os
import shutil
import time

import re
import sys
import uuid
import wave
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx

from .config import load_config, AppConfig

# Module-level config reference — set once at startup before Orchestrator is created.
_APP_CFG: "AppConfig | None" = None
from .nws_api import NWSApi
from .nwws_client import NWWSClient
from .product import parse_product_text, ParsedProduct
from .alert_builder import build_spoken_alert, strip_nws_product_headers
from .tts import TTS
from .audio import write_sine_wav, write_silence_wav, concat_wavs, wav_duration_seconds
from .liquidsoap_telnet import LiquidsoapTelnet
from .cycle import CycleBuilder, CycleContext, CycleSegment

# Active alert tracker (persistent cycle state across restarts)
from .active_alerts import ActiveAlert, AlertTracker, _vtec_track_id
from .discord_log import DiscordLogger

# VTEC policy + SAME event code libraries — Orchestrator defers to these.
from .vtec import toneout_policy as _vtec_toneout_policy
from .same_events import label_or_code as _same_label_or_code

# RWT/RMT scheduler
from .rwt_rmt import RwtRmtSchedule, RwtRmtScheduler

# Optional SAME
try:
    from .same import SameHeader, chunk_locations, render_same_bursts_wav, render_same_eom_wav
except Exception:  # pragma: no cover
    SameHeader = None  # type: ignore
    chunk_locations = None  # type: ignore
    render_same_bursts_wav = None  # type: ignore
    render_same_eom_wav = None  # type: ignore

# Optional CAP (api.weather.gov/alerts/active)
try:
    from .cap_nws import NwsCapPoller, CapAlertEvent
except Exception:  # pragma: no cover
    NwsCapPoller = None  # type: ignore
    CapAlertEvent = None  # type: ignore


# Optional Station Alert Feed (handled alerts JSON for radio UI)
try:
    from .station_feed import FeedSender, StationFeedAlert, atomic_write_json, build_station_feed_payload
except Exception:
    FeedSender = None  # type: ignore
    StationFeedAlert = None  # type: ignore
    atomic_write_json = None  # type: ignore
    build_station_feed_payload = None  # type: ignore

# Optional ERN/GWES SAME monitor (Level 3 source)
try:
    from .ern_gwes import ErnGwesMonitor, ErnSameEvent
except Exception:  # pragma: no cover
    ErnGwesMonitor = None  # type: ignore
    ErnSameEvent = None  # type: ignore
    defaults_from_env = None  # type: ignore


log = logging.getLogger("seasonalweather")

# --- Station Alert Feed (handled alerts JSON for radio UI) ---
# SeasonalWeather writes a tiny JSON file with the most recently *handled* alerts.
# nginx serves it from /api/station/ so the radio site can display “what’s being aired”.

_STATION_FEED_STATE = {}  # id -> (StationFeedAlert, expires_ts)
_STATION_FEED_LAST_WRITE_TS = 0.0


def _sf_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _sf_parse_dt(value):
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return dt.datetime.fromisoformat(s)
        except Exception:
            return None
    return None


def _sf_iso(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return str(value)


def _sf_sha1_12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:12]


def _sf_enabled() -> bool:
    if StationFeedAlert is None or atomic_write_json is None or build_station_feed_payload is None:
        return False
    if _APP_CFG is None:
        return False
    return _APP_CFG.station_feed.enabled


def _sf_cfg():
    if _APP_CFG is None:
        return "seasonalweather", "/srv/seasonalweather/api/station/handled-alerts.json", "seasonalweather", 24, 7200, 0.5
    sf = _APP_CFG.station_feed
    return sf.station_id, sf.path, sf.source, sf.max_items, sf.ttl_seconds, sf.min_write_seconds


def _sf_prune(now_ts: float, *, max_items: int) -> None:
    expired = [k for k, (_, exp) in _STATION_FEED_STATE.items() if exp <= now_ts]
    for k in expired:
        _STATION_FEED_STATE.pop(k, None)
    if len(_STATION_FEED_STATE) > max_items:
        items = sorted(_STATION_FEED_STATE.items(), key=lambda kv: kv[1][1], reverse=True)
        keep = dict(items[:max_items])
        _STATION_FEED_STATE.clear()
        _STATION_FEED_STATE.update(keep)


def _sf_write(now_ts: float) -> None:
    global _STATION_FEED_LAST_WRITE_TS
    station_id, path, source, max_items, ttl_s, min_write_s = _sf_cfg()
    if now_ts - _STATION_FEED_LAST_WRITE_TS < min_write_s:
        return
    _sf_prune(now_ts, max_items=max_items)
    alerts = [a for (a, _) in _STATION_FEED_STATE.values()]
    payload = build_station_feed_payload(
        station_id=station_id,
        source=source,
        generated_at_iso=_sf_now_iso(),
        alerts=alerts,
    )
    atomic_write_json(path, payload)
    _STATION_FEED_LAST_WRITE_TS = now_ts


def _sf_emit(alert, *, expires_at=None) -> None:
    if not _sf_enabled():
        return
    try:
        _station_id, _path, _source, max_items, ttl_s, _min_write_s = _sf_cfg()
        _sf_station_feed_hk_start()
        now_ts = time.time()
        exp_dt = _sf_parse_dt(expires_at)
        exp_ts = exp_dt.timestamp() if exp_dt else (now_ts + ttl_s)
        _STATION_FEED_STATE[alert.id] = (alert, exp_ts)
        _sf_write(now_ts)
    except Exception:
        log.exception("Station feed: failed to write handled-alerts.json")


def _sf_remove_ids(ids) -> int:
    if not _sf_enabled():
        return 0
    removed = 0
    try:
        now_ts = time.time()
        for raw in ids or []:
            sid = str(raw or "").strip()
            if not sid:
                continue
            if _STATION_FEED_STATE.pop(sid, None) is not None:
                removed += 1
        if removed:
            _sf_write(now_ts)
    except Exception:
        log.exception("Station feed: failed removing ids=%s", ids)
    return removed


def _sf_remove_by_vtec_tracks(tracks) -> int:
    track_ids = {(t[0] if isinstance(t, tuple) else t) for t in (tracks or []) if (t[0] if isinstance(t, tuple) else t)}
    return _sf_remove_ids(track_ids)



def _sf_cap_reference_ids(ev) -> list[str]:
    refs = getattr(ev, "references", None)
    if not isinstance(refs, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in refs:
        s = str(raw or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _sf_vtec_track_ids(vtec_list) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in (vtec_list or []):
        tid = _vtec_track_id(str(raw))
        if not tid or tid in seen:
            continue
        seen.add(tid)
        out.append(tid)
    return out


def _sf_vtec_tracks(vtec_list) -> list[tuple[str, str]]:
    """
    Module-level VTEC parser for station-feed helpers.
    Returns [(track_id, action)] where track_id := OFFICE.PHEN.SIG.ETN.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in (vtec_list or []):
        s = "".join(str(raw).split()).strip()
        if not s:
            continue
        m = _VTEC_PARSE_RE.search(s)
        if not m:
            continue
        track = f"{m.group('office')}.{m.group('phen')}.{m.group('sig')}.{m.group('etn')}"
        act = (m.group('act') or '').upper()
        key = f"{track}|{act}"
        if key in seen:
            continue
        seen.add(key)
        out.append((track, act))
    return out


def _sf_eas_article(word: str) -> str:
    w = (word or "").strip()
    return "an" if (w[:1].lower() in "aeiou") else "a"


_SF_EAS_ORG_PREFIX = {
    "WXR": "The National Weather Service has issued",
    "CIV": "A civil authority has issued",
    "EAS": "An EAS participant has issued",
    "PEP": "A primary entry point station has issued",
}

# Minimal event map; unknown codes fall back to the code itself
_SF_EAS_EVENT_LABELS = {
    "RWT": "Required Weekly Test",
    "RMT": "Required Monthly Test",
    "DMO": "Practice/Demo Warning",
    "FFW": "Flash Flood Warning",
    "FFA": "Flash Flood Watch",
    "FLW": "Flood Warning",
    "FLS": "Flood Statement",
    "SVR": "Severe Thunderstorm Warning",
    "SVA": "Severe Thunderstorm Watch",
    "TOR": "Tornado Warning",
    "TOA": "Tornado Watch",
    "SMW": "Special Marine Warning",
    "SPS": "Special Weather Statement",
}

# ZCZC-ORG-EEE-LLLLLL-LLLLLL+TTTT-JJJHHMM-SENDER-
_SF_ZCZC_RE = re.compile(
    r"^ZCZC-"
    r"(?P<org>[A-Z]{3})-"
    r"(?P<event>[A-Z0-9]{3})-"
    r"(?P<locs>\d{6}(?:-\d{6})*)"
    r"\+(?P<dur>\d{4})-"
    r"(?P<jday>\d{3})(?P<hh>\d{2})(?P<mm>\d{2})-"
    r"(?P<sender>[^-]{1,16})-?$"
)

def _sf_same_jday_to_utc(jday: int, hh: int, mm: int):
    now = dt.datetime.now(dt.timezone.utc)
    base = dt.datetime(now.year, 1, 1, tzinfo=dt.timezone.utc)
    cand = base + dt.timedelta(days=jday - 1, hours=hh, minutes=mm)

    # Year rollover sanity: choose the closest plausible year
    cands = [cand]
    try:
        cands.append(cand.replace(year=cand.year - 1))
        cands.append(cand.replace(year=cand.year + 1))
    except Exception:
        pass
    return min(cands, key=lambda x: abs((x - now).total_seconds()))

def _sf_parse_same_header(zczc_text):
    s = str(zczc_text or "").strip()
    # Sometimes downstream strings may include extra whitespace/newlines
    s = "".join(s.split())
    m = _SF_ZCZC_RE.match(s)
    if not m:
        return None
    org = m.group("org")
    event_code = m.group("event")
    same_codes = [x for x in m.group("locs").split("-") if x]
    dur = m.group("dur")
    jday = int(m.group("jday"))
    hh = int(m.group("hh"))
    mm = int(m.group("mm"))
    sender = m.group("sender")

    start_utc = _sf_same_jday_to_utc(jday, hh, mm)
    end_utc = start_utc + dt.timedelta(hours=int(dur[:2]), minutes=int(dur[2:]))

    return {
        "org": org,
        "event_code": event_code,
        "same_codes": same_codes,
        "sender": sender,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "raw": s,
    }

def _sf_fmt_local(dt_obj):
    try:
        return dt_obj.astimezone().strftime("%-I:%M %p on %b %-d, %Y")
    except Exception:
        try:
            return dt_obj.isoformat()
        except Exception:
            return str(dt_obj)

def _sf_make_eas_headline(*, org, event_text, area_text, start_utc, end_utc, sender):
    prefix = _SF_EAS_ORG_PREFIX.get(str(org or "").upper(), "An alert originator has issued")
    event_text = str(event_text or "Alert").strip() or "Alert"
    area_text = str(area_text or "").strip() or "Unknown area"
    article = _sf_eas_article(event_text)
    return (
        f"{prefix} {article.upper()} {event_text.upper()} for the following counties or areas: "
        f"{area_text}; at {_sf_fmt_local(start_utc)} Effective until {_sf_fmt_local(end_utc)}. "
        f"Message from {sender}."
    )



_DEFAULT_SF_NWWS_VTEC_EVENT_LABELS: dict[str, str] = {
    "TO.W": "Tornado Warning",
    "TO.A": "Tornado Watch",
    "SV.W": "Severe Thunderstorm Warning",
    "SV.A": "Severe Thunderstorm Watch",
    "FF.W": "Flash Flood Warning",
    "FF.A": "Flash Flood Watch",
    "FA.Y": "Flood Advisory",
    "FA.W": "Flood Warning",
    "FA.A": "Flood Watch",
    "FL.Y": "Flood Advisory",
    "FL.W": "Flood Warning",
    "FL.A": "Flood Watch",
    "CF.Y": "Coastal Flood Advisory",
    "CF.W": "Coastal Flood Warning",
    "CF.A": "Coastal Flood Watch",
    "HU.W": "Hurricane Warning",
    "HU.A": "Hurricane Watch",
    "TR.W": "Tropical Storm Warning",
    "TR.A": "Tropical Storm Watch",
    "BZ.W": "Blizzard Warning",
    "BZ.A": "Blizzard Watch",
    "IS.W": "Ice Storm Warning",
    "HW.W": "High Wind Warning",
    "HW.A": "High Wind Watch",
    "WI.Y": "Wind Advisory",
    "EH.W": "Excessive Heat Warning",
    "EH.A": "Excessive Heat Watch",
    "HT.Y": "Heat Advisory",
    "FR.Y": "Frost Advisory",
    "HZ.W": "Hard Freeze Warning",
    "HZ.A": "Hard Freeze Watch",
    "FW.W": "Red Flag Warning",
    "LE.W": "Lake Effect Snow Warning",
    "LE.A": "Lake Effect Snow Watch",
    "LE.Y": "Lake Effect Snow Advisory",
    "WS.W": "Winter Storm Warning",
    "WS.A": "Winter Storm Watch",
    "WW.Y": "Winter Weather Advisory",
}

_DEFAULT_SF_NWWS_TZ_OFFSETS: dict[str, dt.tzinfo] = {
    "UTC": dt.timezone.utc,
    "GMT": dt.timezone.utc,
    "EST": dt.timezone(dt.timedelta(hours=-5)),
    "EDT": dt.timezone(dt.timedelta(hours=-4)),
    "CST": dt.timezone(dt.timedelta(hours=-6)),
    "CDT": dt.timezone(dt.timedelta(hours=-5)),
    "MST": dt.timezone(dt.timedelta(hours=-7)),
    "MDT": dt.timezone(dt.timedelta(hours=-6)),
    "PST": dt.timezone(dt.timedelta(hours=-8)),
    "PDT": dt.timezone(dt.timedelta(hours=-7)),
    "AKST": dt.timezone(dt.timedelta(hours=-9)),
    "AKDT": dt.timezone(dt.timedelta(hours=-8)),
    "HST": dt.timezone(dt.timedelta(hours=-10)),
    "AST": dt.timezone(dt.timedelta(hours=-4)),
    "ADT": dt.timezone(dt.timedelta(hours=-3)),
}


def _sf_nwws_tzinfo_from_override(value):
    if value is None:
        return None
    if isinstance(value, dt.tzinfo):
        return value
    s = str(value).strip()
    if not s:
        return None
    su = s.upper()
    if su in {"UTC", "GMT", "Z"}:
        return dt.timezone.utc
    m = re.fullmatch(r"([+-])(\d{1,2}):(\d{2})", s)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        hours = int(m.group(2))
        minutes = int(m.group(3))
        return dt.timezone(sign * dt.timedelta(hours=hours, minutes=minutes))
    if re.fullmatch(r"[+-]?\d+", s):
        try:
            mins = int(s)
            return dt.timezone(dt.timedelta(minutes=mins))
        except Exception:
            return None
    return None


def _sf_nwws_tz_offsets() -> dict[str, dt.tzinfo]:
    out = dict(_DEFAULT_SF_NWWS_TZ_OFFSETS)
    try:
        cfg = getattr(getattr(_APP_CFG, "station_feed", None), "nwws", None)
        overrides = getattr(cfg, "tz_abbrev_overrides", {}) if cfg is not None else {}
        if isinstance(overrides, dict):
            for raw_key, raw_val in overrides.items():
                key = str(raw_key or "").strip().upper()
                if not key:
                    continue
                tzinfo = _sf_nwws_tzinfo_from_override(raw_val)
                if tzinfo is not None:
                    out[key] = tzinfo
    except Exception:
        pass
    return out


def _sf_nwws_vtec_event_labels() -> dict[str, str]:
    out = dict(_DEFAULT_SF_NWWS_VTEC_EVENT_LABELS)
    try:
        cfg = getattr(getattr(_APP_CFG, "station_feed", None), "nwws", None)
        overrides = getattr(cfg, "vtec_event_labels", {}) if cfg is not None else {}
        if isinstance(overrides, dict):
            for raw_key, raw_val in overrides.items():
                key = str(raw_key or "").strip().upper()
                val = str(raw_val or "").strip()
                if key and val:
                    out[key] = val
    except Exception:
        pass
    return out


def _sf_nwws_titlecase_event(text: str) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip(" .")
    if not s:
        return ""
    if s.upper() == s:
        s = s.title().replace("Nws", "NWS")
    return s


def _sf_nwws_parse_header_issued_dt(text: str):
    tz_map = _sf_nwws_tz_offsets()
    for ln in (text or "").splitlines()[:120]:
        s = (ln or "").strip()
        m = _NWS_HEADER_ISSUED_RE.match(s)
        if not m:
            continue
        hhmm = m.group("hhmm")
        if len(hhmm) == 3:
            hour = int(hhmm[0]); minute = int(hhmm[1:])
        else:
            hour = int(hhmm[:2]); minute = int(hhmm[2:])
        ampm = m.group("ampm").upper()
        if ampm == "AM":
            hour = 0 if hour == 12 else hour
        else:
            hour = 12 if hour == 12 else hour + 12
        month = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}.get(m.group("mon").strip().upper())
        tzinfo = tz_map.get(m.group("tz").strip().upper())
        if month is None or tzinfo is None:
            continue
        try:
            return dt.datetime(int(m.group("year")), month, int(m.group("day")), hour, minute, tzinfo=tzinfo)
        except Exception:
            continue
    return None


def _sf_nwws_best_issued_dt(parsed, official_text: str):
    issued = _sf_parse_dt(getattr(parsed, "issued", None))
    if issued is not None:
        if issued.tzinfo is None:
            issued = issued.replace(tzinfo=dt.timezone.utc)
        return issued
    return _sf_nwws_parse_header_issued_dt(official_text)


def _sf_nwws_extract_issuer(text: str, fallback_wfo: str = "") -> str:
    for ln in (text or "").splitlines()[:80]:
        s = re.sub(r"\s+", " ", (ln or "").strip())
        if s.lower().startswith("national weather service "):
            return "NWS " + s[len("National Weather Service "):].strip()
    f = (fallback_wfo or "").strip()
    return f"NWS {f}".strip() if f else "NWS"


def _sf_nwws_event_from_text(text: str) -> str:
    for ln in (strip_nws_product_headers(text or "") or "").splitlines()[:80]:
        s = re.sub(r"\s+", " ", (ln or "").strip())
        if not s:
            continue
        m = re.match(r"^\.\.\.(?P<ev>.+?)(?:\s+(?:NOW\s+)?IN EFFECT.*)?\.\.\.$", s, flags=re.IGNORECASE)
        if m:
            ev = _sf_nwws_titlecase_event(m.group("ev"))
            if re.search(r"\b(?:warning|watch|advisory|statement|emergency|message)\b", ev, flags=re.IGNORECASE):
                return ev
        if re.search(r"\b(?:warning|watch|advisory|statement|emergency|message)\b$", s, flags=re.IGNORECASE):
            return _sf_nwws_titlecase_event(s)
    return ""


def _sf_nwws_event_label(prod_type: str, *, vtec_list=None, text: str = "") -> str:
    label_map = _sf_nwws_vtec_event_labels()
    for raw in (vtec_list or []):
        m = _VTEC_PARSE_RE.search(str(raw or ""))
        if not m:
            continue
        label = label_map.get(f"{m.group('phen')}.{m.group('sig')}")
        if label:
            return label
    text_label = _sf_nwws_event_from_text(text)
    if text_label:
        return text_label
    return _sf_eas_event_label_full(prod_type)


def _sf_nwws_area_from_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", (ln or "").strip()) for ln in (strip_nws_product_headers(text or "") or "").splitlines()]
    lines = [ln for ln in lines if ln]
    for i, ln in enumerate(lines):
        if re.match(r"^\*\s*WHERE\.\.\.", ln, flags=re.IGNORECASE):
            parts = [re.sub(r"^\*\s*WHERE\.\.\.\s*", "", ln, flags=re.IGNORECASE).strip()]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if re.match(r"^\*\s*[A-Z][A-Z /-]*\.\.\.", nxt) or nxt.startswith("*"):
                    break
                parts.append(nxt.strip())
                j += 1
            out = re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip(" .")
            if out:
                return out
    for i, ln in enumerate(lines):
        if re.match(r"^\*\s+.+?\s+for\.\.\.$", ln, flags=re.IGNORECASE):
            parts = []
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if nxt.startswith("*") or re.match(r"^(At|HAZARD\.\.\.|SOURCE\.\.\.|IMPACT\.\.\.|TORNADO\.|MAX )", nxt, flags=re.IGNORECASE):
                    break
                parts.append(nxt.strip(" ."))
                j += 1
            out = re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip(" .")
            if out:
                return out
    return ""


def _sf_fmt_issued_until(issued_dt, end_dt, issuer: str) -> str:
    bits = []
    if issued_dt is not None:
        try:
            bits.append(issued_dt.astimezone().strftime("issued %B %-d at %-I:%M%p %Z"))
        except Exception:
            bits.append(f"issued {_sf_iso(issued_dt)}")
    if end_dt is not None:
        try:
            bits.append(end_dt.astimezone().strftime("until %B %-d at %-I:%M%p %Z"))
        except Exception:
            bits.append(f"until {_sf_iso(end_dt)}")
    if issuer:
        bits.append(f"by {issuer}")
    return " ".join(bits).strip()


def _sf_nwws_make_headline(event_text: str, *, issued_dt=None, end_dt=None, issuer: str = "") -> str:
    ev = str(event_text or "Alert").strip() or "Alert"
    suffix = _sf_fmt_issued_until(issued_dt, end_dt, issuer)
    return f"{ev} {suffix}".strip() if suffix else ev


### STATION_FEED_EAS_LABELS_FULL_PATCH ###
# Full current FCC EAS event-code labels (47 CFR §11.31 Table 2 to paragraph (e), incl. MEP)
# This block is intentionally additive and tries to patch whatever helper/dict names your prior patch used.

_SF_EAS_EVENT_LABELS_FULL = {
    # National (required)
    "EAN": "Emergency Action Notification",
    "NIC": "National Information Center",
    "NPT": "National Periodic Test",
    "RMT": "Required Monthly Test",
    "RWT": "Required Weekly Test",

    # State / local (optional)
    "ADR": "Administrative Message",
    "AVW": "Avalanche Warning",
    "AVA": "Avalanche Watch",
    "BZW": "Blizzard Warning",
    "BLU": "Blue Alert",
    "CAE": "Child Abduction Emergency",
    "CDW": "Civil Danger Warning",
    "CEM": "Civil Emergency Message",
    "CFW": "Coastal Flood Warning",
    "CFA": "Coastal Flood Watch",
    "DSW": "Dust Storm Warning",
    "EQW": "Earthquake Warning",
    "EVI": "Evacuation Immediate",
    "EWW": "Extreme Wind Warning",
    "FRW": "Fire Warning",
    "FFW": "Flash Flood Warning",
    "FFA": "Flash Flood Watch",
    "FFS": "Flash Flood Statement",
    "FLW": "Flood Warning",
    "FLA": "Flood Watch",
    "FLS": "Flood Statement",
    "HMW": "Hazardous Materials Warning",
    "HWW": "High Wind Warning",
    "HWA": "High Wind Watch",
    "HUW": "Hurricane Warning",
    "HUA": "Hurricane Watch",
    "HLS": "Hurricane Statement",
    "LEW": "Law Enforcement Warning",
    "LAE": "Local Area Emergency",
    "MEP": "Missing and Endangered Persons",
    "NMN": "Network Message Notification",
    "TOE": "911 Telephone Outage Emergency",
    "NUW": "Nuclear Power Plant Warning",
    "DMO": "Practice/Demo Warning",
    "RHW": "Radiological Hazard Warning",
    "SVR": "Severe Thunderstorm Warning",
    "SVA": "Severe Thunderstorm Watch",
    "SVS": "Severe Weather Statement",
    "SPW": "Shelter in Place Warning",
    "SMW": "Special Marine Warning",
    "SPS": "Special Weather Statement",
    "SSA": "Storm Surge Watch",
    "SSW": "Storm Surge Warning",
    "TOR": "Tornado Warning",
    "TOA": "Tornado Watch",
    "TRW": "Tropical Storm Warning",
    "TRA": "Tropical Storm Watch",
    "TSW": "Tsunami Warning",
    "TSA": "Tsunami Watch",
    "VOW": "Volcano Warning",
    "WSW": "Winter Storm Warning",
    "WSA": "Winter Storm Watch",
}

# A few common legacy/enthusiast aliases you may still see in the wild.
# (Harmless if unused; keeps UI from falling back to raw code text.)
_SF_EAS_EVENT_LABELS_FULL.setdefault("EAT", "Emergency Action Termination")
_SF_EAS_EVENT_LABELS_FULL.setdefault("NAT", "National Audible Test")
_SF_EAS_EVENT_LABELS_FULL.setdefault("NEM", "National Emergency Message")
_SF_EAS_EVENT_LABELS_FULL.setdefault("NST", "National Silent Test")

def _sf_eas_event_label_full(code):
    c = (str(code or "").strip().upper())
    if not c:
        return "Alert"
    return _SF_EAS_EVENT_LABELS_FULL.get(c, c)

def _sf_patch_eas_label_helpers():
    # 1) Patch common dict names by updating them in place
    dict_candidates = (
        "_EAS_EVENT_LABELS",
        "EAS_EVENT_LABELS",
        "_EAS_EVENT_NAMES",
        "EAS_EVENT_NAMES",
        "_SF_EAS_EVENT_LABELS",
        "_SF_EAS_EVENT_NAMES",
        "_SAME_EVENT_LABELS",
        "_SAME_EVENT_NAMES",
    )
    for nm in dict_candidates:
        try:
            obj = globals().get(nm)
            if isinstance(obj, dict):
                obj.update(_SF_EAS_EVENT_LABELS_FULL)
        except Exception:
            pass

    # 2) Wrap common helper function names so they fall back to the full map
    fn_candidates = (
        "_sf_eas_event_name",
        "_sf_eas_event_label",
        "_same_event_name",
        "_same_event_label",
        "_eas_event_name",
        "_eas_event_label",
    )
    for nm in fn_candidates:
        try:
            fn = globals().get(nm)
            if not callable(fn):
                continue
            # Avoid double-wrapping
            if getattr(fn, "__name__", "") == "_sf_eas_event_label_full_wrapper":
                continue

            def _make_wrapper(_orig):
                def _sf_eas_event_label_full_wrapper(code, *args, **kwargs):
                    c = (str(code or "").strip().upper())
                    if c in _SF_EAS_EVENT_LABELS_FULL:
                        return _SF_EAS_EVENT_LABELS_FULL[c]
                    try:
                        return _orig(code, *args, **kwargs)
                    except TypeError:
                        return _orig(code)
                return _sf_eas_event_label_full_wrapper

            globals()[nm] = _make_wrapper(fn)
        except Exception:
            pass

_sf_patch_eas_label_helpers()


### STATION_FEED_HOUSEKEEPING_PATCH ###


# Safe station-feed housekeeping helpers (startup-safe JSON prune)
def _sf_hk_interval_s() -> int:
    if _APP_CFG is None:
        return 60
    return max(5, _APP_CFG.station_feed.housekeeping.interval_sec)

def _sf_hk_grace_s() -> int:
    if _APP_CFG is None:
        return 5
    return max(0, _APP_CFG.station_feed.housekeeping.grace_sec)

def _sf_hk_keep_unparseable() -> bool:
    if _APP_CFG is None:
        return True
    return _APP_CFG.station_feed.housekeeping.keep_unparseable

def _sf_hk_alert_expiry_ts(alert_obj):
    """
    Return the best expiry timestamp for a station-feed alert dict using ends/expires.
    Uses the latest parseable time so entries don't get pruned too early.
    """
    if not isinstance(alert_obj, dict):
        return None

    candidates = []
    for k in ("ends", "expires"):
        raw = alert_obj.get(k)
        dt_obj = _sf_parse_dt(raw)
        if dt_obj is None:
            continue
        try:
            candidates.append(float(dt_obj.timestamp()))
        except Exception:
            continue

    return max(candidates) if candidates else None

def _sf_hk_prune_json_file(now_ts: float) -> bool:
    """
    Prune expired entries directly from handled-alerts.json without rewriting from
    the in-memory station-feed cache. This avoids startup nukes when memory is empty.
    Returns True if the file was rewritten.
    """
    if not _sf_enabled():
        return False

    if _APP_CFG is not None and not _APP_CFG.station_feed.housekeeping.enabled:
        return False

    _station_id, path, _source, _max_items, _ttl_s, _min_write_s = _sf_cfg()

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return False
    except Exception:
        log.exception("Station feed housekeeping: could not read %s", path)
        return False

    if not isinstance(payload, dict):
        return False

    alerts = payload.get("alerts")
    if not isinstance(alerts, list):
        return False

    grace_s = float(_sf_hk_grace_s())
    keep_unparseable = _sf_hk_keep_unparseable()

    kept = []
    removed = 0

    for item in alerts:
        if not isinstance(item, dict):
            removed += 1
            continue

        exp_ts = _sf_hk_alert_expiry_ts(item)

        if exp_ts is None:
            if keep_unparseable:
                kept.append(item)
            else:
                removed += 1
            continue

        # Keep if still valid. Only remove if definitely expired.
        if (exp_ts + grace_s) >= now_ts:
            kept.append(item)
        else:
            removed += 1

    if removed <= 0:
        return False

    payload["alerts"] = kept
    payload["generatedAt"] = _sf_now_iso()

    try:
        atomic_write_json(path, payload)
        log.info(
            "Station feed housekeeping: pruned handled-alerts.json (removed=%s kept=%s path=%s)",
            removed,
            len(kept),
            path,
        )
        return True
    except Exception:
        log.exception("Station feed housekeeping: failed to rewrite %s after prune", path)
        return False


# Periodic station-feed housekeeping:
# - one forced startup write (clears stale JSON after restart)
# - ongoing expiry pruning even when no new alerts arrive
# - writes only when a prune/max-items trim actually changed the in-memory set

_STATION_FEED_HK_STARTED = False
_STATION_FEED_HK_FIRST_WRITE_DONE = False

def _sf_station_feed_housekeeping_once():
    """
    Startup-safe station-feed housekeeping:
    - prune in-memory cache (fine)
    - prune handled-alerts.json directly (safe)
    - DO NOT write handled-alerts.json from memory here, because memory may be empty on startup
    """
    if not _sf_enabled():
        return
    if _APP_CFG is not None and not _APP_CFG.station_feed.housekeeping.enabled:
        return

    try:
        now_ts = time.time()
        _station_id, _path, _source, max_items, _ttl_s, _min_write_s = _sf_cfg()

        # Keep the in-memory set tidy, but don't write from it here.
        _sf_prune(now_ts, max_items=max_items)

        # Prune the JSON file itself based on ends/expires so valid alerts survive restarts.
        _sf_hk_prune_json_file(now_ts)
    except Exception:
        log.exception("Station feed housekeeping: tick failed")

def _sf_station_feed_hk_loop():
    while True:
        try:
            _sf_station_feed_housekeeping_once()
        except Exception:
            log.exception("Station feed housekeeping: loop error")
        time.sleep(_sf_hk_interval_s())

def _sf_station_feed_hk_start():
    global _STATION_FEED_HK_STARTED
    if _STATION_FEED_HK_STARTED:
        return
    if not _sf_enabled():
        return

    try:
        import threading

        t = threading.Thread(
            target=_sf_station_feed_hk_loop,
            name="station-feed-housekeeping",
            daemon=True,
        )
        t.start()
        _STATION_FEED_HK_STARTED = True

        try:
            log.info(
                "Station feed housekeeping enabled (interval=%ss)",
                _sf_hk_interval_s(),
            )
        except Exception:
            pass
    except Exception:
        try:
            log.exception("Station feed housekeeping: failed to start")
        except Exception:
            pass

# Housekeeping is started by Orchestrator.__init__ after cfg is loaded.


def _sf_is_non_alert_station_item(*, alert_id=None, source=None, event=None, headline=None, cycle_only=False) -> bool:
    """Return True for internal cycle-only items that should not appear in StationFeed."""
    try:
        aid = str(alert_id or "").strip()
        src = str(source or "").strip().upper()
        ev = str(event or "").strip().lower()
        hd = str(headline or "").strip().lower()
        if aid.startswith("PNS_SAFETY:"):
            return True
        if src == "PNS_CYCLE":
            return True
        if cycle_only and (ev == "severe weather safety rules" or hd == "severe weather safety rules"):
            return True
    except Exception:
        return False
    return False


def _sf_seed_memory_from_payload_file() -> int:
    """
    Restore handled-alerts.json into the in-memory station-feed cache without
    rewriting from an empty cache first. This prevents restart-time wipes when
    the next alert arrives before StationFeed memory has been repopulated.
    """
    if not _sf_enabled() or StationFeedAlert is None:
        return 0

    try:
        _station_id, path, _source, max_items, ttl_s, _min_write_s = _sf_cfg()
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return 0
    except Exception:
        log.exception("Station feed: failed loading existing handled-alerts.json into memory")
        return 0

    alerts = payload.get("alerts") if isinstance(payload, dict) else None
    if not isinstance(alerts, list):
        return 0

    now_ts = time.time()
    restored = 0

    for item in alerts:
        if not isinstance(item, dict):
            continue

        if _sf_is_non_alert_station_item(
            alert_id=item.get("id"),
            source=((item.get("from") or {}).get("name") if isinstance(item.get("from"), dict) else None),
            event=item.get("event"),
            headline=item.get("headline"),
            cycle_only=(str(((item.get("links") or {}).get("mode") or "")).upper() == "VOICE"),
        ):
            continue

        exp_ts = _sf_hk_alert_expiry_ts(item)
        if exp_ts is None:
            exp_ts = now_ts + ttl_s
        if exp_ts <= now_ts:
            continue

        try:
            sender_raw = item.get("from") or {}
            sender = None
            if FeedSender is not None:
                sender = FeedSender(
                    name=str((sender_raw.get("name") if isinstance(sender_raw, dict) else "") or "SeasonalWeather"),
                    kind=str((sender_raw.get("kind") if isinstance(sender_raw, dict) else "") or "unknown"),
                )

            alert = StationFeedAlert(
                id=str(item.get("id") or _sf_sha1_12(json.dumps(item, sort_keys=True))),
                event=str(item.get("event") or "Alert"),
                headline=str(item.get("headline") or item.get("event") or "Alert"),
                severity=str(item.get("severity") or "Unknown"),
                urgency=str(item.get("urgency") or "Unknown"),
                certainty=str(item.get("certainty") or "Unknown"),
                area=str(item.get("area") or ""),
                effective=_sf_iso(item.get("effective")),
                ends=_sf_iso(item.get("ends")),
                expires=_sf_iso(item.get("expires")),
                sent=_sf_iso(item.get("sent")),
                sameCodes=[str(x) for x in (item.get("sameCodes") or [])],
                from_=sender,
                links=dict(item.get("links") or {}),
            )
            _STATION_FEED_STATE[str(alert.id)] = (alert, float(exp_ts))
            restored += 1
        except Exception:
            log.exception("Station feed: failed restoring one handled-alerts.json entry into memory")

    if restored:
        _sf_prune(now_ts, max_items=max_items)
    return restored

def _station_feed_note_cap(ev, *, mode: str, same_locations, out_wav: str, same_code=None, vtec=None) -> None:
    if not _sf_enabled():
        return
    try:
        vtec_list = list(vtec or [])
        vtec_tracks = _sf_vtec_track_ids(vtec_list)
        vtec_actions = {act for (_track, act) in _sf_vtec_tracks(vtec_list)} if vtec_list else set()
        if vtec_actions & {"CAN", "EXP"}:
            _sf_remove_by_vtec_tracks(vtec_tracks)
            _sf_remove_ids(_sf_cap_reference_ids(ev) + [getattr(ev, "alert_id", None)])
            return
        alert_id = (vtec_tracks[0] if vtec_tracks else None) or getattr(ev, "alert_id", None) or getattr(ev, "id", None) or _sf_sha1_12(str(ev))
        event = getattr(ev, "event", None) or "Alert"
        headline = getattr(ev, "headline", None) or event
        severity = getattr(ev, "severity", None) or "Unknown"
        urgency = getattr(ev, "urgency", None) or "Unknown"
        certainty = getattr(ev, "certainty", None) or "Unknown"
        area = getattr(ev, "area_desc", None) or getattr(ev, "area", None) or ""
        # Times: internal CAP event objects don't always carry these fields.
        # Prefer explicit ev.* fields; otherwise derive END from VTEC; optionally backfill from api.weather.gov.
        effective_raw = getattr(ev, "effective", None)
        ends_raw = getattr(ev, "ends", None)
        expires_raw = getattr(ev, "expires", None)
        sent_raw = getattr(ev, "sent", None)

        def _sf_best_end_from_vtec(vtec_list):
            # Pull the END token after '-' if present: ...-YYYYMMDDThhmmZ/
            try:
                ends = []
                for raw in vtec_list or []:
                    ss = "".join(str(raw).split()).strip()
                    if not ss:
                        continue
                    m = re.search(r"-((?:\d{8}|\d{6})T\d{4}Z)", ss)
                    if not m:
                        continue
                    txt = m.group(1).upper()
                    mm = re.fullmatch(r"(\d{8}|\d{6})T(\d{4})Z", txt)
                    if not mm:
                        continue

                    d = mm.group(1)
                    hm = mm.group(2)

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

                    ends.append(dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc))

                return max(ends) if ends else None
            except Exception:
                return None

        # Try to derive end time from VTEC if missing
        vtec_list = vtec or getattr(ev, "vtec", None) or getattr(ev, "vtec_list", None)
        if not isinstance(vtec_list, list):
            vtec_list = [vtec_list] if vtec_list else []
        vtec_end = _sf_best_end_from_vtec(vtec_list)

        if vtec_end:
            ends_raw = ends_raw or vtec_end
            expires_raw = expires_raw or vtec_end
            effective_raw = effective_raw or sent_raw  # best-effort

        # Optional: backfill from NWS alert detail endpoint (handles urn:oid IDs)
        if (_APP_CFG.station_feed.fetch_nws if _APP_CFG else False) and isinstance(alert_id, str) and alert_id.strip():
            try:
                import requests  # type: ignore
                url = f"https://api.weather.gov/alerts/{alert_id}"
                r = requests.get(
                    url,
                    headers={"User-Agent": "(seasonalnet.org, info@seasonalnet.org)"},
                    timeout=8,
                )
                if r.ok:
                    props = (r.json() or {}).get("properties", {}) or {}
                    sent_raw = sent_raw or props.get("sent")
                    effective_raw = effective_raw or props.get("effective")
                    ends_raw = ends_raw or props.get("ends") or props.get("eventEndingTime")
                    expires_raw = expires_raw or props.get("expires")
            except Exception:
                pass


        # Final safety fallback so station-feed entries don't end up immortal/blank
        # (some CAP paths, especially SPS-ish cases, may arrive without ends/expires/VTEC end)
        if not ends_raw and not expires_raw:
            try:
                sent_dt = _sf_parse_dt(sent_raw) if sent_raw else None
            except Exception:
                sent_dt = None
            if sent_dt is not None:
                fallback_end = sent_dt + dt.timedelta(hours=6)
                ends_raw = ends_raw or fallback_end
                expires_raw = expires_raw or fallback_end
                effective_raw = effective_raw or sent_dt

        effective = _sf_iso(effective_raw)
        ends = _sf_iso(ends_raw)
        expires = _sf_iso(expires_raw)
        sent = _sf_iso(sent_raw)

        expires_at = expires_raw or ends_raw

        if (_APP_CFG.station_feed.debug if _APP_CFG else False):
            try:
                keys = sorted(getattr(ev, "__dict__", {}).keys())
            except Exception:
                keys = []
            log.info(
                "Station feed CAP: id=%r sent=%r effective=%r expires=%r ends=%r vtec=%r keys=%s",
                alert_id, sent, effective, expires, ends, vtec_list, keys
            )

        wfo = getattr(ev, "wfo", None) or getattr(ev, "office", None)
        sender_name = f"NWS CAP{f'/{wfo}' if wfo else ''}"
        sender = FeedSender(name=sender_name, kind="origin") if FeedSender else None

        links = {"mode": mode, "wav": out_wav}
        if isinstance(alert_id, str) and alert_id.strip():
            links["nws"] = f"https://api.weather.gov/alerts/{alert_id}"
        if same_code:
            links["same"] = f"same:{same_code}"
        if vtec:
            links["vtec"] = vtec

        alert = StationFeedAlert(
            id=str(alert_id),
            event=str(event),
            headline=str(headline),
            severity=str(severity),
            urgency=str(urgency),
            certainty=str(certainty),
            area=str(area),
            effective=effective,
            ends=ends,
            expires=expires,
            sent=sent,
            sameCodes=[str(x) for x in (same_locations or [])],
            from_=sender,
            links=links,
        )
        _sf_emit(alert, expires_at=expires_at)
    except Exception:
        log.exception("Station feed: failed to note CAP alert")


def _station_feed_note_ern(ev, *, same_locations, out_wav: str) -> None:
    if not _sf_enabled():
        return
    try:
        raw_text = getattr(ev, "text", None) or ""
        parsed = _sf_parse_same_header(raw_text)

        # Sender badge: use header sender if we parsed it; otherwise fall back
        sender_name = None
        if parsed:
            sender_name = parsed.get("sender")
        sender_name = sender_name or getattr(ev, "sender", None) or "ERN"
        # Keep this "unknown" so the UI doesn't auto-append "(relay)" if it uses sender.kind in labels
        sender = FeedSender(name=str(sender_name), kind="unknown") if FeedSender else None

        # Event text: prefer ev.event if already human-readable, else map from SAME event code
        ev_event = (getattr(ev, "event", None) or "").strip()
        event_text = ev_event
        if parsed:
            code = str(parsed.get("event_code") or "").upper()
            if (not event_text) or (event_text.upper() == code):
                event_text = _SF_EAS_EVENT_LABELS.get(code, code or "EAS Alert")
        if not event_text:
            event_text = "EAS Alert"

        # Area text: prefer any precomputed area string, else join whatever same_locations contains
        area_text = str(getattr(ev, "area", None) or "").strip()
        if not area_text:
            area_text = "; ".join([str(x) for x in (same_locations or []) if str(x).strip()])

        # SAME codes + timestamps from header when available
        same_codes = [str(x) for x in (same_locations or [])]
        sent_iso = _sf_iso(getattr(ev, "sent", None))
        effective_iso = None
        ends_iso = None
        expires_iso = None
        expires_at = None

        if parsed:
            same_codes = [str(x) for x in (parsed.get("same_codes") or [])] or same_codes
            start_utc = parsed["start_utc"]
            end_utc = parsed["end_utc"]
            sent_iso = sent_iso or _sf_iso(start_utc)
            effective_iso = _sf_iso(start_utc)
            ends_iso = _sf_iso(end_utc)
            expires_iso = _sf_iso(end_utc)
            expires_at = end_utc

            headline = _sf_make_eas_headline(
                org=parsed.get("org"),
                event_text=event_text,
                area_text=area_text,
                start_utc=start_utc,
                end_utc=end_utc,
                sender=str(sender_name),
            )
        else:
            # Fallback if SAME parse fails: at least don't explode
            headline = getattr(ev, "headline", None) or raw_text or str(event_text)

        alert_id = getattr(ev, "id", None) or _sf_sha1_12(
            f"ern:{event_text}:{headline}:{sender_name}:{out_wav}"
        )

        links = {"mode": "REL", "wav": out_wav, "via": "ERN/GWES"}

        alert = StationFeedAlert(
            id=str(alert_id),
            event=str(event_text),
            headline=str(headline),
            severity="Unknown",
            urgency="Unknown",
            certainty="Unknown",
            area=str(area_text or "Unknown area"),
            effective=effective_iso,
            ends=ends_iso,
            expires=expires_iso,
            sent=sent_iso,
            sameCodes=same_codes,
            from_=sender,
            links=links,
        )
        _sf_emit(alert, expires_at=expires_at)
    except Exception:
        log.exception("Station feed: failed to note ERN relay")


def _station_feed_note_nwws(
    parsed,
    *,
    mode: str,
    same_locations,
    out_wav: str,
    product_id=None,
    expires_at=None,
    vtec=None,
    official_text=None,
    issued_at=None,
    event_text=None,
    headline=None,
    area_text=None,
) -> None:
    if not _sf_enabled():
        return
    try:
        awips = getattr(parsed, "awips_id", None) or getattr(parsed, "awips", None) or ""
        wfo = getattr(parsed, "wfo", None) or ""
        prod_type = getattr(parsed, "product_type", None) or "NWWS"
        base_text = str(official_text or getattr(parsed, "raw_text", "") or "")
        raw_vtec = [str(x) for x in (vtec or []) if str(x).strip()]
        if not raw_vtec and base_text:
            raw_vtec = _VTEC_FIND_RE.findall(base_text)
        vtec_tracks = _sf_vtec_track_ids(raw_vtec)
        vtec_actions = {act for (_track, act) in _sf_vtec_tracks(raw_vtec)} if raw_vtec else set()
        if vtec_actions & {"CAN", "EXP"}:
            _sf_remove_by_vtec_tracks(vtec_tracks)
            return

        key = f"nwws:{prod_type}:{awips}:{wfo}:{issued_at or getattr(parsed, 'issued', None)}"
        alert_id = vtec_tracks[0] if vtec_tracks else _sf_sha1_12(key)
        sender = FeedSender(name="NWWS-OI", kind="relay") if FeedSender else None

        issued_dt = _sf_parse_dt(issued_at) if issued_at is not None else _sf_nwws_best_issued_dt(parsed, base_text)
        if issued_dt is not None and issued_dt.tzinfo is None:
            issued_dt = issued_dt.replace(tzinfo=dt.timezone.utc)

        end_raw = (
            expires_at
            or getattr(parsed, "expires", None)
            or getattr(parsed, "expires_at", None)
            or getattr(parsed, "end", None)
            or getattr(parsed, "end_time", None)
            or getattr(parsed, "valid_until", None)
        )
        if not end_raw and issued_dt is not None:
            end_raw = issued_dt + dt.timedelta(hours=6)
        end_dt = _sf_parse_dt(end_raw)

        event_display = str(event_text or _sf_nwws_event_label(prod_type, vtec_list=raw_vtec, text=base_text)).strip()
        issuer = _sf_nwws_extract_issuer(base_text, fallback_wfo=wfo)
        headline_display = str(headline or _sf_nwws_make_headline(event_display, issued_dt=issued_dt, end_dt=end_dt, issuer=issuer)).strip()
        area_display = str(area_text or "").strip() or _sf_nwws_area_from_text(base_text) or str(wfo)

        links = {"mode": mode, "wav": out_wav}
        if product_id:
            links["nws"] = f"https://api.weather.gov/products/{product_id}"
        if raw_vtec:
            links["vtec"] = raw_vtec

        alert = StationFeedAlert(
            id=str(alert_id),
            event=event_display,
            headline=headline_display,
            severity="Unknown",
            urgency="Unknown",
            certainty="Unknown",
            area=str(area_display),
            effective=_sf_iso(issued_dt),
            ends=_sf_iso(end_dt or end_raw),
            expires=_sf_iso(end_dt or end_raw),
            sent=_sf_iso(issued_dt),
            sameCodes=[str(x) for x in (same_locations or [])],
            from_=sender,
            links=links,
        )
        _sf_emit(alert, expires_at=(end_dt or end_raw))
    except Exception:
        log.exception("Station feed: failed to note NWWS toneout")


# We surgically remove the stale time sentence from the Station ID segment,
# because we will insert a live-updating time WAV right after it.
_TIME_SENTENCE_RE = re.compile(r"\bThe current time is [^.]+\.\s*", re.IGNORECASE)


# NWS header timestamp line (common in SPS/SVS/etc):
#   "310 PM EST Sun Jan 11 2026"
_NWS_HEADER_ISSUED_RE = re.compile(
    r"^(?P<hhmm>\d{3,4})\s*(?P<ampm>AM|PM)\s*(?P<tz>[A-Z]{2,4})\s+"
    r"(?P<dow>[A-Za-z]{3})\s+(?P<mon>[A-Za-z]{3})\s+(?P<day>\d{1,2})\s+(?P<year>\d{4})\s*$"
)

# Generic intro sentences we may have at the top of spoken scripts (we replace these for SPS).
_SPS_INTRO_LEAD_RE = re.compile(
    r"(?is)^(?:This is a statement from the National Weather Service\.|The National Weather Service has issued the following message\.)\s*"
)

# Expiration-ish lines that are usually the only thing we actually want to narrate on EXP/CAN updates.
_EXPIRY_LINE_RE = re.compile(
    r"\b(has expired|has been allowed to expire|has ended|is no longer in effect)\b",
    re.IGNORECASE
)


# VTEC finder (accepts YYYYMMDD or legacy YYMMDD date digits)
_VTEC_FIND_RE = re.compile(
    r"/[A-Z]\.[A-Z]{3}\.[A-Z]{4}\.[A-Z0-9]{2}\.[A-Z]\.\d{4}\.(?:\d{8}|\d{6})T\d{4}Z-(?:\d{8}|\d{6})T\d{4}Z/"
)

# VTEC parser for action/track extraction (office+phen+sig+etn)
_VTEC_PARSE_RE = re.compile(
    r"/(?P<prod>[A-Z])\.(?P<act>[A-Z]{3})\.(?P<office>[A-Z]{4})\.(?P<phen>[A-Z0-9]{2})\.(?P<sig>[A-Z])\.(?P<etn>\d{4})\.(?P<start>(?:\d{8}|\d{6})T\d{4}Z)-(?P<end>(?:\d{8}|\d{6})T\d{4}Z)/"
)

# UGC targeting helpers (NWWS-only)
_UGC_EXPIRES_RE = re.compile(r"\b\d{6}-\s*$")
_UGC_ANY_CODE_RE = re.compile(r"\b[A-Z]{2}[CZ]\d{3}(?:>\d{3})?\b|\b[A-Z]{2}Z\d{3}(?:>\d{3})?\b|\b[A-Z]{3}\d{3}(?:>\d{3})?\b")

# Minimal-but-solid state FIPS mapping for SAME conversion (county codes)
# (Includes all states + DC + major territories for safety.)
_STATE_ABBR_TO_FIPS: dict[str, str] = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
    "PR": "72",
    "VI": "78",
    "GU": "66",
    "AS": "60",
    "MP": "69",
}

# Reverse lookup (FIPS2 -> state abbr) for SAME->county name mapping
_FIPS2_TO_STATE_ABBR: dict[str, str] = {v: k for (k, v) in _STATE_ABBR_TO_FIPS.items()}



def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


# _env_* helpers removed — all configuration now flows through AppConfig.
# Credentials are accessed via cfg.secrets.* (set once in load_config()).


def _safe_event_code(raw: str | None) -> str:
    if not raw:
        return "SPS"
    s = "".join(ch for ch in str(raw).upper() if ch.isalnum())
    return s[:3] if len(s) >= 3 else "SPS"


def _fmt_time(now: dt.datetime) -> str:
    # 12-hour like "6:42 PM"
    return now.strftime("%-I:%M %p")


_TZ_NAME_MAP = {
    "EST": "Eastern Standard Time",
    "EDT": "Eastern Daylight Time",
    "CST": "Central Standard Time",
    "CDT": "Central Daylight Time",
    "MST": "Mountain Standard Time",
    "MDT": "Mountain Daylight Time",
    "PST": "Pacific Standard Time",
    "PDT": "Pacific Daylight Time",
    "AKST": "Alaska Standard Time",
    "AKDT": "Alaska Daylight Time",
    "HST": "Hawaii Standard Time",
    "AST": "Atlantic Standard Time",
    "ADT": "Atlantic Daylight Time",
    "UTC": "Coordinated Universal Time",
    "GMT": "Greenwich Mean Time",
}


def _expand_tz_token(token: str) -> str:
    tok = (token or "").strip()
    if not tok:
        return "local"
    return _TZ_NAME_MAP.get(tok.upper(), tok)


def _short_tz(now: dt.datetime) -> str:
    return _expand_tz_token(now.tzname() or "local")


class Orchestrator:
    def __init__(self, cfg: AppConfig) -> None:
        global _APP_CFG
        _APP_CFG = cfg
        self.cfg = cfg
        self.api = NWSApi()
        self.telnet = LiquidsoapTelnet(
            host=cfg.secrets.liquidsoap_host,
            port=cfg.secrets.liquidsoap_port,
        )

        self._tz = ZoneInfo(cfg.station.timezone)
        self.local_tz = self._tz  # alias for newer code paths (rebroadcast, etc.)

        # NWWS-OI
        self.jid = cfg.secrets.nwws_jid
        self.password = cfg.secrets.nwws_password
        self.nwws_server = cfg.nwws.server
        self.nwws_port = cfg.nwws.port

        # TTS
        self.tts = TTS(
            backend=cfg.tts.backend,
            voice=cfg.tts.voice,
            rate_wpm=cfg.tts.rate_wpm,
            volume=cfg.tts.volume,
            sample_rate=cfg.audio.sample_rate,
            text_overrides=cfg.tts.text_overrides,
            vtp_cfg=cfg.tts.voicetext_paul,
        )

        self.mode = "normal"
        self.heightened_until: dt.datetime | None = None
        self.last_heightened_at: dt.datetime | None = None
        self.last_product_desc: str | None = None

        # RWT/RMT gating timestamps
        self.last_toneout_at: dt.datetime | None = None
        self.cap_last_severe_at: dt.datetime | None = None
        self.ern_last_tone_at: dt.datetime | None = None

        self.cycle_builder = CycleBuilder(
            api=self.api,
            tz_name=cfg.station.timezone,
            obs_stations=cfg.observations.stations,
            reference_points=cfg.cycle.reference_points,
            same_fips_all=cfg.service_area.same_fips_all,
            cycle_cfg=cfg.cycle,
        )

        # Fast membership checks for "in-area" targeting
        self._same_fips_allow_set = {str(x).strip() for x in cfg.service_area.same_fips_all if str(x).strip()}

        # --- NWWS flood-gate controls ---
        self._nwws_logger = logging.getLogger("seasonalweather.nwws")
        self._nwws_raw_seen = 0
        self._nwws_rx_log_first_n = cfg.nwws.resiliency.rx_log_first_n
        self._nwws_decision_log_first_n = cfg.nwws.resiliency.decision_log_first_n
        self._nwws_decision_log_every = cfg.nwws.resiliency.decision_log_every
        self._nwws_allowed_wfos = self._norm_wfo_set(getattr(cfg.nwws, "allowed_wfos", []))

        self.nwws_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=200)

        # CAP queue (only used if CAP enabled and import succeeded)
        self.cap_queue: asyncio.Queue["CapAlertEvent"] = asyncio.Queue(maxsize=200)  # type: ignore[name-defined]
        self._cap_voice_last_by_key: dict[tuple[str, str], dt.datetime] = {}
        self._cap_full_last_by_key: dict[tuple[str, str], dt.datetime] = {}

        # ERN queue (only used if ERN enabled and import succeeded)
        self.ern_queue: asyncio.Queue["ErnSameEvent"] = asyncio.Queue(maxsize=200)  # type: ignore[name-defined]

        # ERN relay cooldown (on-air)
        self._ern_relay_last_any_at: dt.datetime | None = None

        # Prevent concurrent cycle flush/push from overlapping
        self._cycle_lock = asyncio.Lock()

        # Debounced cycle refill task (prevents rebuild spam on bursts of CAP voice cut-ins)
        self._cycle_refill_task: asyncio.Task | None = None

        # Duration (seconds) of the last fully-built cycle audio train.
        # Used by _cycle_loop to schedule the next rebuild based on actual audio length
        # rather than the fixed scheduling interval, preventing mid-broadcast chops.
        self._last_cycle_seq_dur: float = 0.0

        # Live time WAV (drift killer)
        self.live_time_enabled = cfg.live_time.enabled
        self.live_time_interval_seconds = cfg.live_time.interval_seconds

        # --- Cross-source dedupe (NWWS vs CAP) ---
        self._dedupe_ttl_seconds = cfg.dedupe.ttl_seconds
        self._dedupe_lock = asyncio.Lock()
        self._recent_air_keys: dict[str, dt.datetime] = {}


        # --- Periodic rebroadcast rotation (no re-tone) ---
        # Replays voice-only copies of recently-aired products so info isn't heard only once.
        # Disabled by default; enable via SEASONAL_REBROADCAST_ENABLED=1.
        self.rebroadcast_enabled = cfg.rebroadcast.enabled
        self.rebroadcast_interval_seconds = cfg.rebroadcast.interval_seconds
        self.rebroadcast_min_gap_seconds = cfg.rebroadcast.min_gap_seconds
        self.rebroadcast_ttl_seconds = cfg.rebroadcast.ttl_seconds
        self.rebroadcast_max_items = cfg.rebroadcast.max_items
        self.rebroadcast_include_voice = cfg.rebroadcast.include_voice

        self._rebroadcast_lock = asyncio.Lock()
        self._rebroadcast_items: dict[str, SimpleNamespace] = {}
        self._rebroadcast_last_any_at = None

        # --- NWWS decision visibility counters ---
        self._nwws_seen = 0
        self._nwws_acted = 0

        # Start station-feed housekeeping now that cfg is available
        _sf_station_feed_hk_start()

        # --- NWS zone lookup (for NWWS UGC->SAME targeting) ---
        self._zone_client: httpx.AsyncClient | None = None
        self._zone_cache_same: dict[str, list[str]] = {}
        self._zone_cache_fail: dict[str, dt.datetime] = {}  # short-term backoff for bad zones
        self._zone_lock = asyncio.Lock()

        # --- ZoneCounty DBX crosswalk (forecast zone -> county FIPS -> SAME) ---
        self._zonecounty_lock = asyncio.Lock()
        self._zonecounty_loaded = False
        self._zonecounty_map: dict[str, list[str]] = {}


        # --- Marine areas .txt crosswalk (marine zone -> coastal county FIPS -> SAME) ---
        self._mareas_lock = asyncio.Lock()
        self._mareas_loaded = False
        self._mareas_map: dict[str, list[str]] = {}

        # --- SAME county name cache (for station feed ERN area display) ---
        self._same_name_cache: dict[str, str] = {}
        self._same_name_fail: dict[str, dt.datetime] = {}
        self._same_name_lock = asyncio.Lock()

        # --- Persistent active alert tracker ---
        # Survives restarts: active watches/warnings are re-queued as cycle segments.
        _tracker_path = Path(cfg.paths.work_dir) / "alert_state.json"
        self.alert_tracker = AlertTracker(_tracker_path)

        # Discord webhook logger (fire-and-forget; starts its drain task in run())
        self.discord = DiscordLogger.from_config(cfg.logs.discord)

    def _station_feed_seed_from_alert_tracker(self) -> int:
        """
        Restore active alerts from alert_state.json into the in-memory station-feed
        cache on startup. This makes StationFeed survive restarts even when the
        next write comes from a totally different source.
        """
        if not _sf_enabled() or StationFeedAlert is None:
            return 0

        tracker_path = Path(self.cfg.paths.work_dir) / "alert_state.json"
        try:
            payload = json.loads(tracker_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return 0
        except Exception:
            log.exception("Station feed: failed loading AlertTracker state for startup seed")
            return 0

        items = payload.get("active_alerts") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return 0

        now_ts = time.time()
        existing_ids = set(_STATION_FEED_STATE.keys())
        existing_wavs: set[str] = set()
        for _alert_obj, _exp_ts in _STATION_FEED_STATE.values():
            try:
                _links = getattr(_alert_obj, "links", {}) or {}
                _wav = str(_links.get("wav") or "").strip()
                if _wav:
                    existing_wavs.add(_wav)
            except Exception:
                continue

        existing_vtec_tracks: set[str] = set()
        for _alert_obj, _exp_ts in _STATION_FEED_STATE.values():
            try:
                _links = getattr(_alert_obj, "links", {}) or {}
                for _v in (_links.get("vtec") or []):
                    _tid = _vtec_track_id(str(_v))
                    if _tid:
                        existing_vtec_tracks.add(_tid)
            except Exception:
                continue

        seeded = 0
        for item in items:
            if not isinstance(item, dict):
                continue

            expires_raw = item.get("expires")
            exp_dt = _sf_parse_dt(expires_raw)
            if exp_dt is None:
                continue
            try:
                exp_ts = float(exp_dt.timestamp())
            except Exception:
                continue
            if exp_ts <= now_ts:
                continue

            audio_path = str(item.get("audio_path") or "").strip()
            tracker_id = str(item.get("id") or "").strip()
            if tracker_id and tracker_id in existing_ids:
                continue
            if audio_path and audio_path in existing_wavs:
                continue

            source = str(item.get("source") or "").strip().upper()
            if _sf_is_non_alert_station_item(
                alert_id=tracker_id,
                source=source,
                event=item.get("event"),
                headline=item.get("headline"),
                cycle_only=bool(item.get("cycle_only", False)),
            ):
                continue

            if source == "CAP":
                sender_name, sender_kind = "CAP restore", "relay"
            elif source == "NWWS":
                sender_name, sender_kind = "NWWS-OI", "relay"
            elif source in {"PNS_CYCLE", "LOCAL", "SEASONALWEATHER"}:
                sender_name, sender_kind = "SeasonalWeather", "origin"
            else:
                sender_name, sender_kind = source or "SeasonalWeather", "unknown"

            sender = FeedSender(name=sender_name, kind=sender_kind) if FeedSender else None

            vtec_list = item.get("vtec") or []
            if not isinstance(vtec_list, list):
                vtec_list = [vtec_list] if vtec_list else []
            vtec_list = [str(x) for x in vtec_list if str(x).strip()]

            tracker_vtec_tracks = {_vtec_track_id(v) for v in vtec_list if _vtec_track_id(v)}
            if tracker_vtec_tracks and (tracker_vtec_tracks & existing_vtec_tracks):
                continue

            area = ""
            for _raw_vtec in vtec_list:
                _m = _VTEC_PARSE_RE.search(str(_raw_vtec))
                if _m:
                    area = str(_m.group("office") or "").strip()
                    if area:
                        break

            script_text = str(item.get("script_text") or "")
            event = str(item.get("event") or item.get("code") or "Alert")
            headline = str(item.get("headline") or event)
            issued_raw = item.get("issued")
            cycle_only = bool(item.get("cycle_only", False))
            same_locs = [str(x) for x in (item.get("same_locs") or []) if str(x).strip()]
            code = str(item.get("code") or "").strip()

            if source == "NWWS":
                event = _sf_nwws_event_label(code or event, vtec_list=vtec_list, text=script_text or headline)
                if not headline or re.fullmatch(r"[A-Z0-9]{6,16}", headline):
                    headline = _sf_nwws_make_headline(
                        event,
                        issued_dt=_sf_parse_dt(issued_raw),
                        end_dt=_sf_parse_dt(expires_raw),
                        issuer=_sf_nwws_extract_issuer(script_text, fallback_wfo=area),
                    )
                if not area or re.fullmatch(r"[A-Z]{4}", area):
                    area = _sf_nwws_area_from_text(script_text) or area

            links = {"mode": "VOICE" if cycle_only else "FULL"}
            if audio_path:
                links["wav"] = audio_path
            if vtec_list:
                links["vtec"] = vtec_list
            if code:
                links["same"] = f"same:{code}"

            try:
                alert = StationFeedAlert(
                    id=tracker_id or _sf_sha1_12(f"tracker:{source}:{headline}:{audio_path}"),
                    event=event,
                    headline=headline,
                    severity="Unknown",
                    urgency="Unknown",
                    certainty="Unknown",
                    area=area,
                    effective=_sf_iso(issued_raw),
                    ends=_sf_iso(expires_raw),
                    expires=_sf_iso(expires_raw),
                    sent=_sf_iso(issued_raw),
                    sameCodes=same_locs,
                    from_=sender,
                    links=links,
                )
                _STATION_FEED_STATE[str(alert.id)] = (alert, exp_ts)
                existing_ids.add(str(alert.id))
                if audio_path:
                    existing_wavs.add(audio_path)
                existing_vtec_tracks.update({t for t in tracker_vtec_tracks if t})
                seeded += 1
            except Exception:
                log.exception("Station feed: failed seeding one AlertTracker entry into StationFeed")

        if seeded:
            _station_id, _path, _source, max_items, _ttl_s, _min_write_s = _sf_cfg()
            _sf_prune(now_ts, max_items=max_items)
        return seeded

    # --- Now Playing / IP-RDS helpers (edit phrases freely) ---

    _NP_CYCLE_TITLES = {
        "id": "Station identification.",
        "time": "The current time in our service area.",
        "status": "Overall station status and alerts.",
        "hwo": "Hazardous weather outlook for the service area.",
        "hwo-unavailable": "Hazardous weather outlook for the service area.",
        "spc": "Severe weather outlook for the service area.",
        "zfp": "Weather synopsis for the area.",
        "fcst": "The forecast for the service area.",
        "obs": "Current conditions in our area.",
        "outro": "End of the current broadcast cycle.",
        "default": "Weather information for our service area.",
    }

    _NP_ALERT_TEMPLATES = {
        "nwws_full": "{event}.",
        "nwws_update": "Update for a {event}.",
        "nwws_end": "A {event} has ended.",
        "cap_full": "{event}.",
        "cap_update": "Update for a {event}.",
        "ern": "{event} relay.",
        "rwt": "Required weekly test.",
        "rmt": "Required monthly test.",
        "rebroadcast": "Details of a currently active {event}.",
        "default": "A weather alert has been issued.",
    }

    def _np_meta(self, *, title: str, kind: str, extra: dict[str, str] | None = None) -> dict[str, str]:
        # What players display:
        #   - title/artist/album/song
        # Plus internal keying fields prefixed with sw_ (most players ignore them).
        station = self.cfg.station.name
        artist = "SeasonalNet"
        album = "Weather information for Baltimore, Washington DC, and surrounding areas"
        t = (title or "").strip() or "SeasonalWeather"
        song = f"{station} — {t}"

        m: dict[str, str] = {
            "title": t,
            "artist": artist,
            "album": album,
            "song": song,
            "sw_station": station,
            "sw_kind": (kind or "").strip(),
        }
        if extra:
            for k, v in extra.items():
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    m[str(k)] = s
        return m

    def _np_cycle_title(self, key: str) -> str:
        k = (key or "").strip()
        return self._NP_CYCLE_TITLES.get(k, self._NP_CYCLE_TITLES["default"])

    def _np_alert_title(self, template_key: str, *, event: str) -> str:
        tpl = self._NP_ALERT_TEMPLATES.get(template_key, self._NP_ALERT_TEMPLATES["default"])
        return tpl.format(event=(event or "Alert").strip())


    def _norm_wfo_set(self, wfos: list[str] | set[str] | tuple[str, ...]) -> set[str]:
        """
        Normalizes allowed WFOs so YAML can use LWX or KLWX interchangeably.
        Also supports 4-letter Kxxx or 3-letter xxx.
        """
        out: set[str] = set()
        for w in wfos or []:
            s = str(w).strip().upper()
            if not s:
                continue
            out.add(s)
            if len(s) == 3:
                out.add("K" + s)
            if len(s) == 4 and s.startswith("K"):
                out.add(s[1:])
        return out

    def _paths(self) -> tuple[Path, Path, Path, Path]:
        work = Path(self.cfg.paths.work_dir)
        audio = Path(self.cfg.paths.audio_dir)
        cache = Path(self.cfg.paths.cache_dir)
        logs = Path(self.cfg.paths.log_dir)
        return work, audio, cache, logs

    async def _wait_for_liquidsoap(self) -> None:
        for _ in range(60):
            if self.telnet.ping():
                log.info("Liquidsoap telnet is reachable")
                return
            await asyncio.sleep(1)
        raise RuntimeError("Liquidsoap telnet did not become reachable (is seasonalweather-liquidsoap running?)")

    def _update_mode(self) -> None:
        _prev_mode = getattr(self, "mode", "normal")
        now = dt.datetime.now(tz=self._tz)
        if self.heightened_until and now < self.heightened_until:
            self.mode = "heightened"
        else:
            self.mode = "normal"
        if self.mode != _prev_mode:
            try:
                self.discord.mode_changed(old_mode=_prev_mode, new_mode=self.mode)
            except Exception:
                pass

    def _heightened_ago_str(self) -> str | None:
        if not self.last_heightened_at:
            return None
        delta = dt.datetime.now(tz=self._tz) - self.last_heightened_at
        mins = int(delta.total_seconds() // 60)
        if mins < 1:
            return "less than one minute"
        if mins < 60:
            return f"{mins} minutes"
        hrs = mins // 60
        rem = mins % 60
        return f"{hrs} hours" if rem == 0 else f"{hrs} hours and {rem} minutes"

    def _cycle_interval_seconds(self) -> int:
        if self.mode == "heightened" or self.alert_tracker.has_active():
            return self.cfg.cycle.heightened_interval_seconds
        return self.cfg.cycle.normal_interval_seconds

    def _schedule_cycle_refill(self, reason: str) -> None:
        """
        Debounce cycle rebuilds. If multiple events ask for a cycle refill in quick succession
        (CAP burst, back-to-back voice cut-ins, etc.), collapse them into ONE rebuild.
        """
        if self._cycle_refill_task and not self._cycle_refill_task.done():
            return

        async def _runner() -> None:
            await asyncio.sleep(2)
            await self._queue_cycle_once(reason=reason)

        safe_reason = "".join(ch for ch in reason if ch.isalnum() or ch in {"_", "-"}).strip() or "refill"
        self._cycle_refill_task = asyncio.create_task(_runner(), name=f"cycle_refill_{safe_reason}")

    # ---- dedupe helpers ----
    def _sha1_12(self, s: str) -> str:
        h = hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()
        return h[:12]

    def _dedupe_func_full_key(self, event_code: str, same_locs: list[str] | None) -> str | None:
        """
        Cross-source "functional" FULL-alert dedupe key.

        ERN cannot map to VTEC, so we dedupe FULL tone-outs by:
          (event code) + (in-area SAME locations)

        Locations are normalized to:
          - service-area filtered
          - unique
          - order-independent (sorted)

        Returns None if no usable locations exist (avoid deduping on empty targets).
        """
        code_u = _safe_event_code(event_code).strip().upper()
        locs_in = [str(x).strip() for x in (same_locs or []) if str(x).strip()]
        locs = self._filter_same_locations_to_service_area(locs_in)
        if not locs:
            return None
        locs_norm = sorted(set(locs))
        blob = code_u + "|" + ",".join(locs_norm)
        return f"FUNC_FULL:{code_u}:{self._sha1_12(blob)}"


    def _nwws_api_product_matches_raw(self, parsed: ParsedProduct, api_text: str) -> bool:
        """
        Accept an api.weather.gov product override only when it appears to be the
        same issuance as the NWWS payload we already received.

        This protects against the API returning an older "latest" product for the
        same product type/location, which can otherwise cause stale VTEC, stale
        expiries, and incorrect cross-source dedupe decisions.
        """
        if not api_text or not api_text.strip():
            return False

        api_parsed = parse_product_text(api_text)
        if api_parsed:
            raw_awips = (parsed.awips_id or "").strip().upper()
            api_awips = (api_parsed.awips_id or "").strip().upper()
            if raw_awips and api_awips and raw_awips != api_awips:
                return False

            raw_wfo = (parsed.wfo or "").strip().upper()
            api_wfo = (api_parsed.wfo or "").strip().upper()
            if raw_wfo and api_wfo and raw_wfo != api_wfo:
                return False

        raw_vtec = self._extract_vtec(parsed.raw_text or "")
        api_vtec = self._extract_vtec(api_text)
        raw_track_actions = set(self._vtec_tracks(raw_vtec))
        api_track_actions = set(self._vtec_tracks(api_vtec))
        raw_tracks = {track for (track, _act) in raw_track_actions}
        api_tracks = {track for (track, _act) in api_track_actions}

        # If NWWS already gave us a concrete VTEC action for a concrete track,
        # the API override must agree on at least one identical (track, action)
        # pair. Matching only the track is too weak: a stale EXA/CON product can
        # otherwise override a newer EXP/CAN product on the same ETN and cause a
        # full tone-out for what should have been a voice-only expiration.
        if raw_track_actions:
            return bool(api_track_actions) and bool(raw_track_actions & api_track_actions)

        # If the raw payload has track IDs but no usable action pair match, reject.
        if raw_tracks:
            return bool(api_tracks) and bool(raw_tracks & api_tracks)

        # No VTEC in raw payload: fall back to AWIPS/WFO agreement only.
        return True

    async def _resolve_nwws_official_text(self, parsed: ParsedProduct) -> tuple[str, str | None]:
        """
        Prefer the live NWWS payload unless the API product can be validated as the
        same issuance. This avoids stale-product regressions during active events.
        """
        official_text = parsed.raw_text or ""
        pid: str | None = None
        try:
            pid = await self.api.latest_product_id(
                parsed.product_type,
                parsed.wfo[1:] if parsed.wfo.startswith("K") else parsed.wfo,
            )
            if not pid:
                pid = await self.api.latest_product_id(parsed.product_type, parsed.wfo.replace("K", "", 1))
            if pid:
                prod = await self.api.get_product(pid)
                if prod and prod.product_text:
                    if self._nwws_api_product_matches_raw(parsed, prod.product_text):
                        official_text = prod.product_text
                    else:
                        api_vtec = ",".join(self._extract_vtec(prod.product_text)[:2])
                        raw_vtec = ",".join(self._extract_vtec(parsed.raw_text or "")[:2])
                        log.warning(
                            "NWWS API override rejected (stale/mismatched product): type=%s awips=%s wfo=%s pid=%s raw_vtec=%s api_vtec=%s",
                            parsed.product_type,
                            parsed.awips_id or "",
                            parsed.wfo,
                            pid,
                            raw_vtec,
                            api_vtec,
                        )
                        pid = None
        except Exception:
            log.exception('NWWS official-text resolution failed; falling back to raw payload')
            pid = None
        return official_text, pid

    def _extract_vtec(self, text: str) -> list[str]:
        if not text:
            return []
        found = _VTEC_FIND_RE.findall(text)
        out: list[str] = []
        seen: set[str] = set()
        for v in found:
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
            if len(out) >= 6:
                break
        return out

    def _vtec_tracks(self, vtec_list: list[str]) -> list[tuple[str, str]]:
        """
        Return [(track_id, action)] where track_id := OFFICE.PHEN.SIG.ETN, action := NEW/CON/EXT/UPG/etc
        """
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for raw in vtec_list or []:
            s = "".join(str(raw).split()).strip()
            if not s:
                continue
            m = _VTEC_PARSE_RE.search(s)
            if not m:
                continue
            office = m.group("office")
            phen = m.group("phen")
            sig = m.group("sig")
            etn = m.group("etn")
            act = m.group("act")
            track = f"{office}.{phen}.{sig}.{etn}"
            k = f"{track}|{act}"
            if k in seen:
                continue
            seen.add(k)
            out.append((track, act))
        return out[:12]

    async def _dedupe_prune(self) -> None:
        now = dt.datetime.now(tz=self._tz)
        ttl = float(self._dedupe_ttl_seconds)
        dead: list[str] = []
        for k, ts in self._recent_air_keys.items():
            if (now - ts).total_seconds() > ttl:
                dead.append(k)
        for k in dead:
            self._recent_air_keys.pop(k, None)

    async def _dedupe_reserve(self, keys: list[str]) -> tuple[bool, str]:
        """
        Reserve dedupe keys *before* airing to avoid races.
        If we later fail to air, caller should release.
        """
        now = dt.datetime.now(tz=self._tz)
        async with self._dedupe_lock:
            await self._dedupe_prune()
            for k in keys:
                if k in self._recent_air_keys:
                    return (False, k)
            for k in keys:
                self._recent_air_keys[k] = now
            return (True, "")

    async def _dedupe_release(self, keys: list[str]) -> None:
        async with self._dedupe_lock:
            for k in keys:
                self._recent_air_keys.pop(k, None)

    def _cap_vtec_list(self, ev: "CapAlertEvent") -> list[str]:  # type: ignore[name-defined]
        vals: list[str] = []

        # NEW: prefer explicit ev.vtec (cap_nws now populates it)
        v0 = getattr(ev, "vtec", None)
        if isinstance(v0, (list, tuple)):
            vals.extend(str(x).strip() for x in v0 if str(x).strip())
        elif isinstance(v0, str) and v0.strip():
            vals.append(v0.strip())

        # Back-compat: still accept alternative attributes if present
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
                if str(k).strip().upper() == "VTEC":
                    if isinstance(v, str):
                        vals.append(v.strip())
                    elif isinstance(v, (list, tuple)):
                        vals.extend(str(x).strip() for x in v if str(x).strip())

        out: list[str] = []
        seen: set[str] = set()
        for x in vals:
            x2 = "".join(str(x).split()).strip()
            if not x2:
                continue
            if x2 in seen:
                continue
            seen.add(x2)
            out.append(x2)
        return out[:12]

    # ---- SAME location filtering (critical for ERN relay targeting) ----
    def _filter_same_locations_to_service_area(self, locs: list[str] | tuple[str, ...] | None) -> list[str]:
        """
        Keep ONLY SAME FIPS locations that are within our configured service area.
        Preserves order and de-dupes.
        """
        if not locs:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for raw in locs:
            f = str(raw).strip()
            if not f:
                continue
            if f not in self._same_fips_allow_set:
                continue
            if f in seen:
                continue
            seen.add(f)
            out.append(f)
        return out

    # ---- Spoken-script post-processing (NWWS path) ----
    def _nws_header_issued_dt(self, text: str) -> dt.datetime | None:
        """Parse an NWS issued-time header into local time for freshness checks."""
        if not text:
            return None
        for raw in (text or "").splitlines()[:40]:
            s = raw.strip()
            m = _NWS_HEADER_ISSUED_RE.match(s)
            if not m:
                continue
            hhmm = m.group("hhmm")
            ampm = (m.group("ampm") or "").upper()
            mon = m.group("mon")
            day = int(m.group("day"))
            year = int(m.group("year"))
            hhmm_i = int(hhmm)
            hour = hhmm_i // 100
            minute = hhmm_i % 100
            if ampm == "PM" and hour != 12:
                hour += 12
            if ampm == "AM" and hour == 12:
                hour = 0
            try:
                naive = dt.datetime.strptime(f"{year} {mon} {day} {hour:02d}:{minute:02d}", "%Y %b %d %H:%M")
                return naive.replace(tzinfo=self._tz)
            except Exception:
                continue
        return None

    def _pns_safety_is_fresh(self, text: str, parsed_issued: object = None) -> bool:
        """Only air severe-weather safety-rules PNS products while they are still same-day and fresh."""
        now_local = dt.datetime.now(self._tz)
        issued_local = self._nws_header_issued_dt(text)
        if issued_local is None:
            issued_dt = _sf_parse_dt(parsed_issued)
            if issued_dt is not None:
                if issued_dt.tzinfo is None:
                    issued_dt = issued_dt.replace(tzinfo=dt.timezone.utc)
                issued_local = issued_dt.astimezone(self._tz)
        if issued_local is None:
            return False
        age = now_local - issued_local
        if age.total_seconds() < -300:
            return False
        if age > dt.timedelta(hours=18):
            return False
        return issued_local.date() == now_local.date()

    def _nws_header_issued_phrase(self, text: str) -> str | None:
        """
        Extract a nicer spoken timestamp from an NWS header line like:
          "310 PM EST Sun Jan 11 2026"
        Returns e.g. "3:10 PM EST Sunday January 11 2026"
        """
        if not text:
            return None

        dow_map = {
            "SUN": "Sunday",
            "MON": "Monday",
            "TUE": "Tuesday",
            "WED": "Wednesday",
            "THU": "Thursday",
            "FRI": "Friday",
            "SAT": "Saturday",
        }
        mon_map = {
            "JAN": "January",
            "FEB": "February",
            "MAR": "March",
            "APR": "April",
            "MAY": "May",
            "JUN": "June",
            "JUL": "July",
            "AUG": "August",
            "SEP": "September",
            "OCT": "October",
            "NOV": "November",
            "DEC": "December",
        }

        for ln in (text or "").splitlines()[:80]:
            s = (ln or "").strip()
            if not s:
                continue
            m = _NWS_HEADER_ISSUED_RE.match(s)
            if not m:
                continue

            hhmm = m.group("hhmm")
            ampm = m.group("ampm").upper()
            tz = _expand_tz_token(m.group("tz").upper())
            dow = dow_map.get(m.group("dow").strip().upper(), m.group("dow").strip())
            mon = mon_map.get(m.group("mon").strip().upper(), m.group("mon").strip())
            day = str(int(m.group("day")))
            year = m.group("year")

            # hhmm: "310" or "1234"
            if len(hhmm) == 3:
                hour = int(hhmm[0])
                minute = int(hhmm[1:])
            else:
                hour = int(hhmm[:2])
                minute = int(hhmm[2:])

            return f"{hour}:{minute:02d} {ampm} {tz} {dow} {mon} {day} {year}"

        return None

    def _fix_sps_preamble(self, script: str, official_text: str) -> str:
        """
        SPS should sound like LWX/NWR-style: "And now a Special Weather
        Statement from your National Weather Service, issued at ..."
        We also try to include the issued time from the product header.
        """
        s = (script or "").strip()
        if not s:
            return s

        issued = self._nws_header_issued_phrase(official_text)
        lead = "And now a Special Weather Statement from your National Weather Service."
        if issued:
            lead = (
                "And now a Special Weather Statement from your National Weather Service, "
                f"issued at {issued}."
            )

        s2 = _SPS_INTRO_LEAD_RE.sub(lead + "\n", s, count=1)
        if s2 == s:
            s2 = lead + "\n" + s

        # If the next line is literally "Special Weather Statement.", drop it to avoid double-intro.
        s2 = re.sub(r"(?im)^\s*Special Weather Statement\.\s*", "", s2, count=1)
        return s2.strip()

    def _cap_sps_preamble(self, sent_iso: str | None) -> str:
        """
        CAP SPS should use the same NWR-style spoken preamble as NWWS SPS.
        We prefer the CAP sent timestamp and speak it in local station time.
        """
        issued = self._fmt_local_from_utc_iso(sent_iso or "")
        if issued:
            return (
                "And now a Special Weather Statement from your National Weather Service, "
                f"issued at {issued}."
            )
        return "And now a Special Weather Statement from your National Weather Service."

    def _expiry_summary_script(self, official_text: str) -> str | None:
        """
        For VTEC EXP (and often CAN) updates, don't read the whole product.
        Try to narrate the first 1-2 human-friendly 'has expired' lines from the header region.
        """
        if not official_text:
            return None

        hits: list[str] = []
        for ln in (official_text or "").splitlines()[:240]:
            s = re.sub(r"\s+", " ", (ln or "")).strip()
            if not s:
                continue
            if _EXPIRY_LINE_RE.search(s):
                if not s.endswith((".", "!", "?")):
                    s += "."
                hits.append(s)
                if len(hits) >= 2:
                    break

        if not hits:
            # Fallback: search a flattened window for a sentence containing "has expired"
            flat = re.sub(r"\s+", " ", (official_text or "")[:7000]).strip()
            m = re.search(
                r"([^.]{0,220}\bhas (?:expired|been allowed to expire|ended)\b[^.]{0,220}\.)",
                flat,
                flags=re.IGNORECASE,
            )
            if m:
                hits = [m.group(1).strip()]

        if not hits:
            return None

        lines = ["The National Weather Service reports the following update."]
        lines.extend(hits)
        lines.append("End of message.")
        return "\n".join(lines).strip()

    # ---- NWWS UGC -> SAME targeting helpers ----
    def _build_nwws_statement_vtec_action_script(
        self,
        *,
        event_text: str,
        area_text: str,
        official_text: str,
        headline: str,
        vtec_actions: set[str],
    ) -> str:
        """
        Reuse the CAP advisory/statement/message EXP/CAN helper for NWWS voice-only
        updates so CAP and NWWS expiration/cancellation copy stays aligned.
        """
        faux_ev = type("NwwsStatementEvent", (), {})()
        faux_ev.event = str(event_text or "Weather alert").strip()
        faux_ev.area_desc = str(area_text or _sf_nwws_area_from_text(official_text) or "the affected areas").strip()
        faux_ev.description = str(official_text or "").strip()
        faux_ev.headline = str(headline or event_text or "").strip()
        return self._build_statement_vtec_action_script(faux_ev, vtec_actions, [])

    def _state_to_fips2(self, st: str) -> str | None:
        s = (st or "").strip().upper()
        if not s:
            return None
        return _STATE_ABBR_TO_FIPS.get(s)

    def _same_from_state_county(self, state_abbr: str, county3: str) -> str | None:
        f2 = self._state_to_fips2(state_abbr)
        c3 = "".join(ch for ch in (county3 or "") if ch.isdigit())
        if not f2 or len(c3) != 3:
            return None
        return f"0{f2}{c3}"

    # ---- Station-feed helpers: SAME(6) -> County names (ERN relays) ----
    def _same6_to_county_zone_id(self, same6: str) -> tuple[str | None, str | None]:
        """Convert SAME PSSCCC (6 digits) to NWS county-zone ID like 'MDC031'."""
        s = "".join(ch for ch in str(same6 or "").strip() if ch.isdigit())
        if len(s) != 6:
            return (None, None)
        st_fips2 = s[1:3]  # ignore partition
        cty3 = s[3:6]
        st = _FIPS2_TO_STATE_ABBR.get(st_fips2)
        if not st:
            return (None, None)
        return (f"{st}C{cty3}", st)

    async def _same6_area_label(self, same6: str) -> str | None:
        """Best-effort county label for SAME via api.weather.gov/zones/county/<ST>C### (cached)."""
        code = "".join(ch for ch in str(same6 or "").strip() if ch.isdigit())
        if len(code) != 6:
            return None
        hit = self._same_name_cache.get(code)
        if hit:
            return hit
        now = dt.datetime.now(tz=self._tz)
        fail_at = self._same_name_fail.get(code)
        if fail_at and (now - fail_at).total_seconds() < 300:
            return None
        zone_id, st = self._same6_to_county_zone_id(code)
        if not zone_id:
            return None
        async with self._same_name_lock:
            hit2 = self._same_name_cache.get(code)
            if hit2:
                return hit2
            fail_at2 = self._same_name_fail.get(code)
            if fail_at2 and (now - fail_at2).total_seconds() < 300:
                return None
            data = await self._get_zone_json("county", zone_id)
            if not data:
                self._same_name_fail[code] = now
                return None
            props = data.get("properties") if isinstance(data.get("properties"), dict) else {}
            name = str(props.get("name") or "").strip()
            state = str(props.get("state") or st or "").strip().upper()
            if not name:
                self._same_name_fail[code] = now
                return None
            label = f"{name}, {state}" if state else name
            self._same_name_cache[code] = label
            return label

    async def _sf_area_text_from_same_codes(self, same_codes: list[str]) -> str:
        """Resolve SAME codes to a '; '-joined area label string for station feed ERN items."""
        if _APP_CFG is not None and not _APP_CFG.station_feed.ern_area_names:
            return ""
        if _APP_CFG is None:
            return ""
        codes = [str(x).strip() for x in (same_codes or []) if str(x).strip()]
        if not codes:
            return ""
        results = await asyncio.gather(*(self._same6_area_label(c) for c in codes), return_exceptions=True)
        out: list[str] = []
        seen: set[str] = set()
        for r in results:
            if isinstance(r, Exception) or not r:
                continue
            s = str(r).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return "; ".join(out)

    def _extract_ugc_block(self, text: str) -> str:
        """
        Extracts the UGC block (the hyphen-continued zone list ending with the 6-digit expiration)
        from an NWS text product. Returns a *flattened* string with no whitespace.
        """
        if not text:
            return ""
        lines = (text or "").splitlines()
        # Only scan early header region; UGC is always near the top.
        scan = lines[:120]

        end_idx: int | None = None
        for i, ln in enumerate(scan):
            s = (ln or "").strip()
            if not s:
                continue
            if _UGC_EXPIRES_RE.search(s) and _UGC_ANY_CODE_RE.search(s):
                end_idx = i
                break

        if end_idx is None:
            return ""

        start_idx = end_idx
        while start_idx > 0:
            prev = (scan[start_idx - 1] or "").strip()
            if not prev:
                break
            # UGC lines are hyphen-continuations too
            if prev.endswith("-") and _UGC_ANY_CODE_RE.search(prev):
                start_idx -= 1
                continue
            break

        # Flatten (UGC lines are usually already dense; we strip whitespace just in case)
        block = "".join((scan[j] or "").strip() for j in range(start_idx, end_idx + 1))
        block = re.sub(r"\s+", "", block)
        return block.strip()

    def _expand_ugc_tokens(self, ugc_block: str) -> list[str]:
        """
        Takes a flattened UGC block like:
          "MDZ008-011-VAZ...-251200-"
        and returns expanded zone ids:
          ["MDZ008", "MDZ011", ...]
        Supports NNN>NNN ranges and prefix-carry.
        """
        if not ugc_block:
            return []

        # Split on hyphens; last piece is usually the 6-digit expiration.
        parts = [p.strip().strip(".") for p in ugc_block.split("-") if p.strip().strip(".")]

        # Drop the trailing expiration token if present
        if parts and re.fullmatch(r"\d{6}", parts[-1]):
            parts = parts[:-1]

        out: list[str] = []
        seen: set[str] = set()

        prefix: str | None = None  # e.g., "MDZ", "PAC", "ANZ"
        for raw in parts:
            tok = raw.strip().strip(".")
            if not tok:
                continue

            # Normalize weird trailing punctuation
            tok = tok.rstrip(",;")

            # Case A: full token with prefix+number, optionally range
            m_full = re.fullmatch(r"(?P<pfx>[A-Z]{2,3}[CZ]?)?(?P<num>\d{3})(?:>(?P<end>\d{3}))?", tok)
            if m_full:
                pfx = (m_full.group("pfx") or "").upper()
                num = m_full.group("num")
                end = m_full.group("end")

                # If pfx is present and contains letters, lock prefix.
                if pfx and any(ch.isalpha() for ch in pfx):
                    prefix = pfx
                if not prefix:
                    continue

                def emit(n: int) -> None:
                    z = f"{prefix}{n:03d}"
                    if z not in seen:
                        seen.add(z)
                        out.append(z)

                if end:
                    a = int(num)
                    b = int(end)
                    step = 1 if b >= a else -1
                    for n in range(a, b + step, step):
                        emit(n)
                else:
                    emit(int(num))
                continue

            # Case B: explicit prefix+number in other shape (rare)
            m2 = re.fullmatch(r"(?P<pfx>[A-Z]{3})(?P<num>\d{3})(?:>(?P<end>\d{3}))?", tok)
            if m2:
                prefix = m2.group("pfx").upper()
                num = int(m2.group("num"))
                end_s = m2.group("end")
                if end_s:
                    end_n = int(end_s)
                    step = 1 if end_n >= num else -1
                    for n in range(num, end_n + step, step):
                        z = f"{prefix}{n:03d}"
                        if z not in seen:
                            seen.add(z)
                            out.append(z)
                else:
                    z = f"{prefix}{num:03d}"
                    if z not in seen:
                        seen.add(z)
                        out.append(z)
                continue

        return out

    def _extract_ugc_zones(self, text: str) -> list[str]:
        blk = self._extract_ugc_block(text)
        return self._expand_ugc_tokens(blk)

    # ---- ZoneCounty crosswalk (NOAA/NWS recommended: zone -> county FIPS -> SAME) ----
    def _zonecounty_enabled(self) -> bool:
        return self.cfg.zonecounty.enabled

    def _zonecounty_dbx_url(self) -> str:
        return self.cfg.zonecounty.dbx_url.strip()

    def _zonecounty_cache_days(self) -> int:
        return self.cfg.zonecounty.cache_days

    def _zonecounty_dbx_path(self) -> Path:
        _, _audio, cache_dir, _logs = self._paths()
        return cache_dir / "zonecounty.dbx"

    def _parse_zonecounty_dbx(self, path: Path) -> dict[str, list[str]]:
        """
        Parses the NWS ZoneCounty 'bp*.dbx' pipe-delimited file.

        Common schema (NWS):
          STATE | ZONE | ... | FIPS | ...
        We only need STATE (2), ZONE (digits), FIPS (5).

        Returns:
          { "MDZ501": ["024031","024033", ...], ... }  (SAME codes, 6 digits)
        """
        m: dict[str, list[str]] = {}
        seen: dict[str, set[str]] = {}

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if "|" not in ln:
                    continue
                parts = [p.strip() for p in ln.split("|")]
                if len(parts) < 7:
                    continue

                st = (parts[0] or "").strip().upper()
                zn = "".join(ch for ch in (parts[1] or "") if ch.isdigit())
                fips = "".join(ch for ch in (parts[6] or "") if ch.isdigit())

                if len(st) != 2 or not zn or len(fips) != 5:
                    continue

                ugc = f"{st}Z{zn.zfill(3)}"
                same = "0" + fips  # SAME is 6 digits: 0 + (state2+county3)

                if ugc not in seen:
                    seen[ugc] = set()
                    m[ugc] = []
                if same in seen[ugc]:
                    continue
                seen[ugc].add(same)
                m[ugc].append(same)

        return m

    async def _ensure_zonecounty_loaded(self) -> None:
        # ZONECOUNTY_DBX_DISCOVERY_PATCH_v1
        # Hard rule: never delete the last-known-good DBX on a failed refresh.
        if self._zonecounty_loaded:
            return

        async with self._zonecounty_lock:
            if self._zonecounty_loaded:
                return

            if not self._zonecounty_enabled():
                self._zonecounty_loaded = True
                return

            dbx_path = self._zonecounty_dbx_path()
            dbx_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = dbx_path.with_suffix(".tmp")
            lastgood_path = dbx_path.with_suffix(".lastgood")

            url = (self._zonecounty_dbx_url() or "").strip()
            max_age_days = max(1, int(self._zonecounty_cache_days()))
            now = dt.datetime.now(tz=self._tz)

            def _cache_is_fresh() -> bool:
                if not dbx_path.exists():
                    return False
                try:
                    mtime = dt.datetime.fromtimestamp(dbx_path.stat().st_mtime, tz=self._tz)
                    age_s = (now - mtime).total_seconds()
                    return age_s <= (max_age_days * 86400) and dbx_path.stat().st_size > 1024
                except Exception:
                    return False

            async def _try_fetch(candidate_url: str) -> dict[str, list[str]] | None:
                try:
                    client = await self._ensure_zone_client()  # reuse UA/timeouts/headers
                    r = await client.get(candidate_url)
                    if r.status_code != 200 or not r.content or len(r.content) <= 1024:
                        log.warning(
                            "ZoneCounty DBX fetch failed (status=%s url=%s).",
                            r.status_code,
                            candidate_url,
                        )
                        return None

                    # Write to temp then validate by parsing.
                    tmp_path.write_bytes(r.content)
                    parsed = await asyncio.to_thread(self._parse_zonecounty_dbx, tmp_path)

                    if not parsed:
                        log.warning("ZoneCounty DBX candidate parsed 0 zones (url=%s).", candidate_url)
                        try:
                            tmp_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        return None

                    # Backup lastgood and atomically replace.
                    if dbx_path.exists():
                        try:
                            shutil.copy2(dbx_path, lastgood_path)
                        except Exception:
                            log.warning("ZoneCounty lastgood backup failed (file=%s).", dbx_path)

                    os.replace(str(tmp_path), str(dbx_path))
                    log.info("ZoneCounty DBX refreshed: %s (%d bytes) src=%s", dbx_path, len(r.content), candidate_url)
                    return parsed
                except Exception:
                    log.exception("ZoneCounty DBX download/validate failed (url=%s).", candidate_url)
                    return None

            # If cache is fresh, just load it and bail early.
            if _cache_is_fresh():
                try:
                    self._zonecounty_map = await asyncio.to_thread(self._parse_zonecounty_dbx, dbx_path)
                    log.info("ZoneCounty loaded: zones=%d file=%s", len(self._zonecounty_map), dbx_path)
                except Exception:
                    log.exception("ZoneCounty parse failed; disabling ZoneCounty mapping for this run")
                    self._zonecounty_map = {}
                self._zonecounty_loaded = True
                return

            # Refresh path: try explicit URL first (unless 'auto'), then discover from index page.
            updated_map: dict[str, list[str]] | None = None

            if url and url.lower() != "auto":
                updated_map = await _try_fetch(url)

            if updated_map is None:
                # Discovery: scrape https://www.weather.gov/gis/ZoneCounty for bp*.dbx tokens.
                index_url = (self.cfg.zonecounty.index_url or "").strip()
                base_url = (self.cfg.zonecounty.base_url or "").strip()
                if base_url and not base_url.endswith("/"):
                    base_url += "/"

                token_re = re.compile(r"\bbp\d{2}[a-z]{2}\d{2}\.dbx\b", re.IGNORECASE)
                mon_map = {
                    "ja": 1, "fe": 2, "mr": 3, "ap": 4, "my": 5, "jn": 6,
                    "jl": 7, "au": 8, "se": 9, "oc": 10, "no": 11, "de": 12,
                }

                def tok_key(tok: str) -> tuple[int, int, int]:
                    # bp18mr25.dbx -> (2025, 3, 18)
                    try:
                        t = tok.lower()
                        dd = int(t[2:4])
                        mm = mon_map.get(t[4:6], 0)
                        yy = 2000 + int(t[6:8])
                        return (yy, mm, dd)
                    except Exception:
                        return (0, 0, 0)

                try:
                    client = await self._ensure_zone_client()
                    r = await client.get(index_url)
                    if r.status_code == 200 and r.text:
                        toks = sorted({m.group(0).lower() for m in token_re.finditer(r.text)}, key=tok_key, reverse=True)
                        # Try newest-first; cap tries to avoid hammering.
                        for tok in toks[:20]:
                            cand = base_url + tok
                            updated_map = await _try_fetch(cand)
                            if updated_map is not None:
                                break
                    else:
                        log.warning("ZoneCounty discovery fetch failed (status=%s url=%s).", r.status_code, index_url)
                except Exception:
                    log.exception("ZoneCounty discovery failed (index_url=%s).", index_url)

            # Load from updated_map if we refreshed successfully, otherwise fall back to existing cache.
            if updated_map is not None:
                self._zonecounty_map = updated_map
                self._zonecounty_loaded = True
                return

            if not dbx_path.exists():
                log.warning("ZoneCounty DBX not available (no cache file). Zone->SAME mapping will be unavailable.")
                self._zonecounty_loaded = True
                self._zonecounty_map = {}
                return

            try:
                self._zonecounty_map = await asyncio.to_thread(self._parse_zonecounty_dbx, dbx_path)
                log.info("ZoneCounty loaded: zones=%d file=%s", len(self._zonecounty_map), dbx_path)
            except Exception:
                log.exception("ZoneCounty parse failed; disabling ZoneCounty mapping for this run")
                self._zonecounty_map = {}

            self._zonecounty_loaded = True

    def _mareas_enabled(self) -> bool:
        return self.cfg.mareas.enabled

    def _mareas_url(self) -> str:
        """Optional URL for a mareas*.txt style crosswalk."""
        return self.cfg.mareas.url.strip()

    def _mareas_cache_days(self) -> int:
        return self.cfg.mareas.cache_days

    def _mareas_path(self) -> Path:
        _, _audio, cache_dir, _logs = self._paths()
        return cache_dir / "mareas.txt"

    def _parse_mareas_txt(self, path: Path) -> dict[str, list[str]]:
        """
        Parse official NWS mareas*.txt files and a legacy fallback format.

        Supported inputs:

          1) Official NWS format:
               AN|73535|Tidal Potomac from Key Bridge to Indian Head MD|38.7406|-77.0712
             -> zone ANZ535, SAME 073535

          2) Legacy/free-form lines already containing ANZ535 plus 5-digit or 6-digit codes.

        Returns:
          dict like { "ANZ535": ["073535"], ... }
        """
        import re

        pipe_alpha_re = re.compile(r"^[A-Z]{2}$")
        pipe_num_re = re.compile(r"^\d{5}$")

        zone_re = re.compile(r"\b([A-Z]{3}\d{3})\b")
        fips5_re = re.compile(r"\b(\d{5})\b")
        same6_re = re.compile(r"\b(\d{6})\b")

        out: dict[str, list[str]] = {}
        seen: dict[str, set[str]] = {}

        def add(zone: str, same_code: str) -> None:
            z = "".join(ch for ch in str(zone).upper() if ch.isalnum())
            s = "".join(ch for ch in str(same_code) if ch.isdigit()).zfill(6)
            if len(z) != 6 or len(s) != 6:
                return
            if z not in out:
                out[z] = []
                seen[z] = set()
            if s in seen[z]:
                return
            seen[z].add(s)
            out[z].append(s)

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                s0 = (ln or "").strip()
                if not s0 or s0.startswith("#"):
                    continue

                # Official NWS mareas*.txt pipe format:
                #   SSALPHA|SSNUM|ZONENAME|LON|LAT
                parts = [p.strip() for p in s0.split("|")]
                if len(parts) >= 2 and pipe_alpha_re.fullmatch(parts[0].upper()) and pipe_num_re.fullmatch(parts[1]):
                    ssalpha = parts[0].upper()        # e.g. AN
                    ssnum = parts[1]                  # e.g. 73535
                    zone = f"{ssalpha}Z{ssnum[-3:]}"  # -> ANZ535
                    same = f"0{ssnum}"                # -> 073535
                    add(zone, same)
                    continue

                # Legacy/free-form fallback
                s = s0.upper()
                zm = zone_re.search(s)
                if not zm:
                    continue
                zone = zm.group(1).upper()

                codes: list[str] = []

                for x in same6_re.findall(s):
                    d = "".join(ch for ch in x if ch.isdigit())
                    if len(d) == 6:
                        codes.append(d)

                for x in fips5_re.findall(s):
                    d = "".join(ch for ch in x if ch.isdigit())
                    if len(d) == 5:
                        codes.append("0" + d)

                for c in codes:
                    add(zone, c)

        return out

    async def _ensure_mareas_loaded(self) -> None:
        if self._mareas_loaded:
            return

        async with self._mareas_lock:
            if self._mareas_loaded:
                return

            if not self._mareas_enabled():
                self._mareas_loaded = True
                self._mareas_map = {}
                return

            path = self._mareas_path()
            path.parent.mkdir(parents=True, exist_ok=True)

            url = self._mareas_url()
            max_age_days = max(1, int(self._mareas_cache_days()))
            now = dt.datetime.now(tz=self._tz)

            need_download = True
            if path.exists():
                try:
                    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=self._tz)
                    need_download = (now - mtime).total_seconds() > (max_age_days * 86400)
                except Exception:
                    need_download = True

            if url and need_download:
                try:
                    client = await self._ensure_zone_client()
                    r = await client.get(url)
                    if r.status_code == 200 and r.content and len(r.content) > 256:
                        tmp = path.with_suffix(".tmp")
                        tmp.write_bytes(r.content)
                        tmp.replace(path)
                        log.info("Marine areas .txt database refreshed: %s (%d bytes)", path, len(r.content))
                    else:
                        log.warning("Marine areas .txt database fetch failed (status=%s). Using cache if present.", r.status_code)
                except Exception:
                    log.exception("Marine areas .txt database download failed; using cache if present")

            if not path.exists():
                log.info("Marine areas .txt database not available (no cache file). Marine zone->SAME mapping unavailable.")
                self._mareas_loaded = True
                self._mareas_map = {}
                return

            try:
                self._mareas_map = await asyncio.to_thread(self._parse_mareas_txt, path)
                log.info("Marine areas loaded: zones=%d file=%s", len(self._mareas_map), path)
            except Exception:
                log.exception("Marine areas parse failed; disabling marine mapping for this run")
                self._mareas_map = {}

            self._mareas_loaded = True
    async def _ensure_zone_client(self) -> httpx.AsyncClient:
        if self._zone_client is not None:
            return self._zone_client

        # Use an explicit UA for NWS (required by their policy).
        ua = (self.cfg.nws.user_agent or "").strip()
        if not ua:
            ua = (self.cfg.cap.user_agent or "").strip()
        if not ua:
            ua = "SeasonalWeather (NWS zone mapper)"

        self._zone_client = httpx.AsyncClient(
            timeout=httpx.Timeout(15.0, connect=8.0),
            headers={
                "User-Agent": ua,
                "Accept": "application/geo+json, application/json;q=0.9, */*;q=0.8",
            },
        )
        return self._zone_client

    async def _get_zone_json(self, zone_type: str, zone_id: str) -> dict | None:
        """
        Fetch https://api.weather.gov/zones/<zone_type>/<zone_id>
        Returns parsed JSON dict on success, else None.
        """
        zt = (zone_type or "").strip().lower()
        zid = (zone_id or "").strip().upper()
        if not zt or not zid:
            return None

        url = f"https://api.weather.gov/zones/{zt}/{zid}"
        client = await self._ensure_zone_client()

        try:
            r = await client.get(url)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    def _same_list_from_zone_json(self, data: dict) -> list[str]:
        """
        Prefer geocode.SAME if present. Otherwise, fall back to parsing county URLs.
        """
        if not isinstance(data, dict):
            return []
        props = data.get("properties") if isinstance(data.get("properties"), dict) else {}
        geo = props.get("geocode") if isinstance(props.get("geocode"), dict) else {}

        same_vals = geo.get("SAME") if isinstance(geo.get("SAME"), (list, tuple)) else None
        out: list[str] = []
        seen: set[str] = set()

        def add_same(x: str) -> None:
            s = "".join(ch for ch in str(x).strip() if ch.isdigit())
            if not s:
                return
            # SAME codes are 6 digits; some sources omit leading zero. Pad-left.
            if len(s) < 6:
                s = s.zfill(6)
            if len(s) != 6:
                return
            if s in seen:
                return
            seen.add(s)
            out.append(s)

        if same_vals:
            for x in same_vals:
                add_same(str(x))
            return out

        # Fall back: if this forecast zone has "county" URLs, parse their IDs like "MDC031"
        counties = props.get("county")
        if isinstance(counties, str):
            counties = [counties]
        if isinstance(counties, (list, tuple)):
            for u in counties:
                cid = str(u).strip().rstrip("/").split("/")[-1].upper()
                m = re.fullmatch(r"([A-Z]{2})C(\d{3})", cid)
                if not m:
                    continue
                same = self._same_from_state_county(m.group(1), m.group(2))
                if same:
                    add_same(same)

        return out

    async def _ugc_zone_to_same(self, zone_id: str) -> list[str]:
        """
        Map a UGC zone token (e.g., MDZ008, PAC021, ANZ530) to SAME county/marine codes.
        Uses:
          - direct conversion for XXC### when possible
          - NWS zone endpoints for others, preferring geocode.SAME
        """
        zid = (zone_id or "").strip().upper()
        if not zid:
            return []

        # Cache
        if zid in self._zone_cache_same:
            return list(self._zone_cache_same[zid])

        # short-term backoff for repeated failures (prevents hammering)
        fail_ts = self._zone_cache_fail.get(zid)
        if fail_ts and (dt.datetime.now(tz=self._tz) - fail_ts).total_seconds() < 300:
            return []

        # Direct county-zone conversion: "PAC021" etc.
        m_direct = re.fullmatch(r"([A-Z]{2})C(\d{3})", zid)
        if m_direct:
            same = self._same_from_state_county(m_direct.group(1), m_direct.group(2))
            if same:
                self._zone_cache_same[zid] = [same]
                return [same]

        # Forecast/public zones: use ZoneCounty DBX crosswalk first (NOAA/NWS recommended).
        if re.fullmatch(r"[A-Z]{2}Z\d{3}", zid):
            try:
                await self._ensure_zonecounty_loaded()
                lst = self._zonecounty_map.get(zid)
                if lst:
                    self._zone_cache_same[zid] = list(lst)
                    return list(lst)
            except Exception:
                pass

        

        # Marine zones: try mareas crosswalk (ANZ/AMZ/GMZ/LMZ/PZZ/etc)
        if re.fullmatch(r"[A-Z]{3}\d{3}", zid):
            try:
                await self._ensure_mareas_loaded()
                lst2 = self._mareas_map.get(zid)
                if lst2:
                    self._zone_cache_same[zid] = list(lst2)
                    return list(lst2)
            except Exception:
                pass
# Otherwise, ask NWS API. Most UGC tokens are forecast zones; marine ones might be under "marine".
        # We try a small ordered list.
        async with self._zone_lock:
            # Check again after acquiring lock (double-checked caching)
            if zid in self._zone_cache_same:
                return list(self._zone_cache_same[zid])

            types_to_try = ["forecast", "public", "marine", "fire", "offshore"]
            data: dict | None = None
            for zt in types_to_try:
                data = await self._get_zone_json(zt, zid)
                if data:
                    break

            if not data:
                self._zone_cache_fail[zid] = dt.datetime.now(tz=self._tz)
                return []

            same_list = self._same_list_from_zone_json(data)
            if same_list:
                self._zone_cache_same[zid] = list(same_list)
                return list(same_list)

            # If no SAME codes could be derived, treat as failure but don't spam retries.
            self._zone_cache_fail[zid] = dt.datetime.now(tz=self._tz)
            return []

    async def _nwws_same_targets_from_texts(self, primary_text: str, secondary_text: str) -> tuple[list[str], list[str], str, bool]:
        """
        Returns:
          zones_found, in_area_same, source_label, mapping_success
        mapping_success indicates we successfully derived at least one SAME from zones.
        """
        zones = self._extract_ugc_zones(primary_text)
        src = "raw"
        if not zones:
            zones = self._extract_ugc_zones(secondary_text)
            src = "official" if zones else "none"

        if not zones:
            return ([], [], src, False)

        # Map zones -> SAME
        all_same: list[str] = []
        any_mapped = False
        for z in zones[:250]:  # safety cap
            sames = await self._ugc_zone_to_same(z)
            if sames:
                any_mapped = True
                all_same.extend(sames)

        # De-dupe while preserving order
        dedup: list[str] = []
        seen: set[str] = set()
        for s in all_same:
            s2 = str(s).strip()
            if not s2 or s2 in seen:
                continue
            seen.add(s2)
            dedup.append(s2)

        in_area = self._filter_same_locations_to_service_area(dedup)
        return (zones, in_area, src, any_mapped)


    # ---- Rebroadcast rotation (no re-tone) ----
    def _parse_vtec_dt_utc(self, s: str) -> dt.datetime | None:
        '''
        Parse VTEC timestamps like:
          20260111T2300Z  (YYYYMMDD)
          260111T2300Z    (YYMMDD legacy)
        Returns an aware UTC datetime or None.
        '''
        txt = (s or "").strip().upper()
        m = re.fullmatch(r"(\d{8}|\d{6})T(\d{4})Z", txt)
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

    def _best_expiry_from_vtec(self, vtec_list: list[str]) -> dt.datetime | None:
        '''
        Returns the latest END time found across VTEC codes (UTC), or None.
        '''
        ends: list[dt.datetime] = []
        for raw in vtec_list or []:
            s = "".join(str(raw).split()).strip()
            if not s:
                continue

            # Pull the END token after the '-' if present: ...-YYYYMMDDThhmmZ/
            m = re.search(r"-((?:\d{8}|\d{6})T\d{4}Z)", s)
            if not m:
                continue

            t = self._parse_vtec_dt_utc(m.group(1))
            if t:
                ends.append(t)

        if not ends:
            return None
        return max(ends)

    def _rebroadcast_clamp_expiry(self, now_local: dt.datetime, exp_utc: dt.datetime | None) -> dt.datetime:
        '''
        Clamp expiry to a safety TTL so rotation can't grow unbounded if upstream is weird.
        '''
        ttl = max(60, int(self.rebroadcast_ttl_seconds))
        cap = now_local + dt.timedelta(seconds=ttl)

        if not exp_utc:
            return cap

        try:
            if exp_utc.tzinfo is None:
                exp_utc = exp_utc.replace(tzinfo=dt.timezone.utc)
            exp_local = exp_utc.astimezone(self._tz)

            if exp_local <= now_local + dt.timedelta(seconds=20):
                exp_local = now_local + dt.timedelta(seconds=20)

            return min(exp_local, cap)
        except Exception:
            return cap

    def _rebroadcast_make_key(
        self,
        *,
        source: str,
        kind: str,
        event_code: str,
        tracks: list[tuple[str, str]] | None,
        same_locs: list[str] | None,
        script: str,
    ) -> str:
        '''
        Stable-ish identity for the rotation item.
        Prefer track id, else functional FULL key, else script hash.
        '''
        src = "".join(ch for ch in (source or "").upper() if ch.isalnum())[:8] or "SRC"
        knd = "".join(ch for ch in (kind or "").upper() if ch.isalnum())[:8] or "KIND"
        code = _safe_event_code(event_code)

        if tracks:
            t0 = tracks[0][0]
            if t0:
                return f"RB:{src}:{knd}:TRK:{t0}"

        if knd == "FULL":
            fkey = self._dedupe_func_full_key(code, same_locs)
            if fkey:
                return f"RB:{src}:{knd}:{fkey}"

        sh = self._sha1_12((script or "")[:4000])
        return f"RB:{src}:{knd}:{code}:{sh}"

    async def _rebroadcast_prune_locked(self, now: dt.datetime) -> None:
        dead: list[str] = []
        for k, it in self._rebroadcast_items.items():
            try:
                if getattr(it, "expires_at", None) and now >= it.expires_at:
                    dead.append(k)
            except Exception:
                dead.append(k)

        for k in dead:
            it = self._rebroadcast_items.pop(k, None)
            if not it:
                continue
            ap = Path(str(getattr(it, "audio_path", "") or ""))
            if ap and ap.exists():
                try:
                    ap.unlink(missing_ok=True)
                except Exception:
                    pass

        max_items = max(1, int(self.rebroadcast_max_items))
        if len(self._rebroadcast_items) > max_items:
            items = list(self._rebroadcast_items.items())
            items.sort(key=lambda kv: (
                getattr(kv[1], "expires_at", now),
                getattr(kv[1], "created_at", now),
            ))
            for k, it in items[: max(0, len(items) - max_items)]:
                self._rebroadcast_items.pop(k, None)
                ap = Path(str(getattr(it, "audio_path", "") or ""))
                if ap and ap.exists():
                    try:
                        ap.unlink(missing_ok=True)
                    except Exception:
                        pass

    async def _rebroadcast_remove_by_tracks(
        self, *, tracks: list[tuple[str, str]] | None, reason: str
    ) -> int:
        """Remove rebroadcast rotation items that match given VTEC track ids."""
        if not self.rebroadcast_enabled or not tracks:
            return 0

        track_ids: set[str] = {str(t[0]).strip() for t in tracks if t and str(t[0]).strip()}
        if not track_ids:
            return 0

        now = dt.datetime.now(self.local_tz)
        removed = 0

        async with self._rebroadcast_lock:
            await self._rebroadcast_prune_locked(now)
            for k in list(self._rebroadcast_items.keys()):
                parts = k.split(":")
                # Key formats:
                #   RB:src:knd:TRK:<track_id>:<same_sig>
                #   RB:src:knd:ANON:<same_sig>:<hash>
                if len(parts) >= 6 and parts[3] == "TRK" and parts[4] in track_ids:
                    self._rebroadcast_items.pop(k, None)
                    removed += 1

        if removed:
            self.log.info(
                "Rebroadcast: removed %d item(s) tracks=%s reason=%s",
                removed,
                ",".join(sorted(track_ids)),
                reason,
            )
        return removed


    async def _rebroadcast_touch_expiry_by_tracks(
        self, *, tracks: list[tuple[str, str]] | None, expires_at: dt.datetime, reason: str
    ) -> int:
        """Refresh expiry on any matching rotation items without creating new ones."""
        if not self.rebroadcast_enabled or not tracks:
            return 0

        track_ids: set[str] = {str(t[0]).strip() for t in tracks if t and str(t[0]).strip()}
        if not track_ids:
            return 0

        now = dt.datetime.now(self.local_tz)
        touched = 0

        async with self._rebroadcast_lock:
            await self._rebroadcast_prune_locked(now)
            for k, it in self._rebroadcast_items.items():
                parts = k.split(":")
                if len(parts) >= 6 and parts[3] == "TRK" and parts[4] in track_ids:
                    it.expires_at = expires_at
                    touched += 1

        if touched:
            self.log.debug(
                "Rebroadcast: refreshed expiry for %d item(s) tracks=%s expires_at=%s reason=%s",
                touched,
                ",".join(sorted(track_ids)),
                expires_at.isoformat(),
                reason,
            )
        return touched


    def _rebroadcast_pick_due_locked(self, now: dt.datetime) -> SimpleNamespace | None:
        due: list[SimpleNamespace] = []
        for it in self._rebroadcast_items.values():
            try:
                if getattr(it, "expires_at", None) and now >= it.expires_at:
                    continue
                nd = getattr(it, "next_due_at", None)
                if nd and now >= nd:
                    due.append(it)
            except Exception:
                continue

        if not due:
            return None

        due.sort(key=lambda it: (
            getattr(it, "last_aired_at", None) or getattr(it, "created_at", now),
            getattr(it, "created_at", now),
        ))
        return due[0]

    async def _rebroadcast_add(self, *, key: str, desc: str, script: str, expires_at: dt.datetime) -> None:
        if not self.rebroadcast_enabled:
            return
        s = (script or "").strip()
        if not s:
            return

        now = dt.datetime.now(tz=self._tz)
        script_hash = self._sha1_12(s[:4000])

        async with self._rebroadcast_lock:
            await self._rebroadcast_prune_locked(now)
            it = self._rebroadcast_items.get(key)
            if it and getattr(it, "script_hash", "") == script_hash:
                it.desc = desc
                it.expires_at = expires_at
                it.next_due_at = min(getattr(it, "next_due_at", now), now + dt.timedelta(seconds=60))
                return

        # Render outside lock
        out_wav = await self._render_voice_only_audio(s, prefix="rebcast")

        async with self._rebroadcast_lock:
            await self._rebroadcast_prune_locked(now)

            it2 = self._rebroadcast_items.get(key)
            if it2:
                try:
                    Path(out_wav).unlink(missing_ok=True)
                except Exception:
                    pass
                it2.desc = desc
                it2.expires_at = max(getattr(it2, "expires_at", now), expires_at)
                it2.next_due_at = min(getattr(it2, "next_due_at", now), now + dt.timedelta(seconds=60))
                return

            self._rebroadcast_items[key] = SimpleNamespace(
                key=key,
                desc=desc,
                script_hash=script_hash,
                audio_path=str(out_wav),
                created_at=now,
                last_aired_at=None,
                next_due_at=now + dt.timedelta(seconds=max(60, int(self.rebroadcast_interval_seconds))),
                expires_at=expires_at,
            )

            await self._rebroadcast_prune_locked(now)

    async def _rebroadcast_maybe_air(self) -> None:
        if not self.rebroadcast_enabled:
            return

        now = dt.datetime.now(tz=self._tz)

        self._update_mode()
        if self.mode == "heightened":
            return

        if self.last_toneout_at and (now - self.last_toneout_at).total_seconds() < float(self.rebroadcast_min_gap_seconds):
            return

        if self._rebroadcast_last_any_at and (now - self._rebroadcast_last_any_at).total_seconds() < float(self.rebroadcast_min_gap_seconds):
            return

        async with self._rebroadcast_lock:
            await self._rebroadcast_prune_locked(now)
            it = self._rebroadcast_pick_due_locked(now)
            if not it:
                return

            ap = Path(str(getattr(it, "audio_path", "") or ""))
            if not ap.exists():
                it.next_due_at = now + dt.timedelta(seconds=120)
                return

            it.next_due_at = now + dt.timedelta(seconds=max(60, int(self.rebroadcast_interval_seconds)))

        async with self._cycle_lock:
            try:
                self.telnet.flush_cycle()
            except Exception:
                pass
            desc = (getattr(it, "desc", "") or "").strip() or "Earlier message"
            title = self._np_alert_title("rebroadcast", event=desc)
            meta = self._np_meta(title=title, kind="rebroadcast", extra={"sw_alert_source": "rebroadcast", "sw_desc": desc})
            self.telnet.push_alert(str(ap), meta=meta)

        async with self._rebroadcast_lock:
            it2 = self._rebroadcast_items.get(getattr(it, "key", ""))
            if it2:
                it2.last_aired_at = now
                n_items = max(1, len(self._rebroadcast_items))
                it2.next_due_at = now + dt.timedelta(seconds=max(60, int(self.rebroadcast_interval_seconds)) * n_items)

            self._rebroadcast_last_any_at = now

        self._schedule_cycle_refill("post-rebroadcast")
        log.info("REBROADCAST: aired desc=%s audio=%s", getattr(it, "desc", ""), str(ap))

    async def _rebroadcast_loop(self) -> None:
        tick = 20
        try:
            tick = max(10, min(60, int(self.rebroadcast_interval_seconds) // 10))
        except Exception:
            tick = 20

        log.info(
            "Rebroadcast loop starting (enabled=%s interval=%ss ttl=%ss max_items=%d include_voice=%s tick=%ss)",
            self.rebroadcast_enabled,
            self.rebroadcast_interval_seconds,
            self.rebroadcast_ttl_seconds,
            self.rebroadcast_max_items,
            self.rebroadcast_include_voice,
            tick,
        )

        while True:
            await asyncio.sleep(tick)
            try:
                await self._rebroadcast_maybe_air()
            except Exception:
                log.exception("Rebroadcast tick failed")

    # ---- CAP toggles ----
    def _cap_enabled(self) -> bool:
        return self.cfg.cap.enabled

    def _cap_dryrun(self) -> bool:
        return self.cfg.cap.dryrun

    def _cap_poll_seconds(self) -> int:
        return self.cfg.cap.poll_seconds

    def _cap_user_agent(self) -> str:
        return self.cfg.cap.user_agent

    def _cap_url(self) -> str:
        return self.cfg.cap.url

    def _cap_full_enabled(self) -> bool:
        return self.cfg.cap.full.enabled

    def _cap_full_severities(self) -> set[str]:
        return {s.strip().lower() for s in self.cfg.cap.full.severities if s.strip()}

    def _cap_full_events(self) -> set[str]:
        events = [e.strip() for e in self.cfg.cap.full.events if e.strip()]
        if events:
            return set(events)
        # Empty list in yaml means "match all qualifying severities" — use the canonical default set
        return {
            "Tornado Warning",
            "Severe Thunderstorm Warning",
            "Flash Flood Warning",
            "Flood Warning",
            "Hurricane Warning",
            "Tropical Storm Warning",
            "Storm Surge Warning",
            "Extreme Wind Warning",
            "Blizzard Warning",
            "Winter Storm Warning",
            "Ice Storm Warning",
            "High Wind Warning",
            "Wind Chill Warning",
            "Tornado Watch",
            "Severe Thunderstorm Watch",
            "Flash Flood Watch",
            "Flood Watch",
            "Hurricane Watch",
            "Tropical Storm Watch",
            "Storm Surge Watch",
            "Blizzard Watch",
            "Winter Storm Watch",
            "Ice Storm Watch",
            "High Wind Watch",
            "Wind Chill Watch",
            "Winter Weather Advisory",
            "Snow Squall Warning",
        }

    def _cap_full_cooldown_seconds(self) -> int:
        return self.cfg.cap.full.cooldown_seconds

    def _cap_voice_enabled(self) -> bool:
        return self.cfg.cap.voice.enabled

    def _cap_voice_events(self) -> set[str]:
        return {e.strip() for e in self.cfg.cap.voice.events if e.strip()}

    def _cap_voice_cooldown_seconds(self) -> int:
        return self.cfg.cap.voice.cooldown_seconds

    # ---- ERN/GWES SAME monitor toggles ----
    def _ern_enabled(self) -> bool:
        return self.cfg.ern.enabled

    def _ern_dryrun(self) -> bool:
        return self.cfg.ern.dryrun

    def _ern_url(self) -> str:
        return self.cfg.ern.url.strip()

    def _ern_relay_enabled(self) -> bool:
        return self.cfg.ern.relay.enabled

    def _ern_relay_events(self) -> set[str]:
        return {e.strip().upper() for e in self.cfg.ern.relay.events if e.strip()}

    def _ern_relay_min_confidence(self) -> float:
        return self.cfg.ern.relay.min_confidence

    def _ern_relay_cooldown_seconds(self) -> int:
        return self.cfg.ern.relay.cooldown_seconds

    def _ern_relay_senders(self) -> set[str]:
        senders = [s.strip().upper() for s in self.cfg.ern.relay.senders if s.strip()]
        return set(senders)

    # ---- SAME toggles ----
    def _same_enabled(self) -> bool:
        return self.cfg.same.enabled

    def _same_sender(self) -> str:
        return self.cfg.same.sender

    def _same_duration_minutes(self) -> int:
        return self.cfg.same.duration_minutes

    def _same_amplitude(self) -> float:
        return self.cfg.same.amplitude

    # ---- LIVE TIME WAV (drift killer) ----
    def _live_time_wav_path(self) -> Path:
        _, audio_dir, _, _ = self._paths()
        return audio_dir / "cycle_time_now.wav"

    def _live_time_text(self) -> str:
        now = dt.datetime.now(tz=self._tz)
        return f"The current time is, {_fmt_time(now)}, {_short_tz(now)}."

    def _render_live_time_wav_once(self) -> None:
        out = self._live_time_wav_path()
        out.parent.mkdir(parents=True, exist_ok=True)

        tmp_id = uuid.uuid4().hex[:10]
        tts_wav = out.parent / f".cycle_time_tts_{tmp_id}.wav"
        gap = out.parent / f".cycle_time_gap_{tmp_id}.wav"
        tmp_out = out.parent / f".cycle_time_out_{tmp_id}.wav"

        seg_gap = 0.35

        text = self._live_time_text()
        self.tts.synth_to_wav(text, tts_wav)
        write_silence_wav(gap, seg_gap, self.cfg.audio.sample_rate)
        concat_wavs(tmp_out, [gap, tts_wav, gap])

        os.replace(str(tmp_out), str(out))

        for p in (tts_wav, gap):
            try:
                p.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

    async def _live_time_loop(self) -> None:
        try:
            self._render_live_time_wav_once()
            log.info("Live time WAV enabled (interval=%ss path=%s)", self.live_time_interval_seconds, self._live_time_wav_path())
        except Exception:
            log.exception("Live time WAV initial render failed (will retry)")

        while True:
            await asyncio.sleep(max(10, int(self.live_time_interval_seconds)))
            try:
                self._render_live_time_wav_once()
            except Exception:
                log.exception("Live time WAV refresh failed")

    # ---- RWT/RMT scheduler toggles ----
    def _tests_enabled(self) -> bool:
        return self.cfg.tests.enabled

    def _tests_postpone_minutes(self) -> int:
        return self.cfg.tests.postpone_minutes

    def _tests_max_postpone_hours(self) -> int:
        return self.cfg.tests.max_postpone_hours

    def _tests_jitter_seconds(self) -> int:
        return self.cfg.tests.jitter_seconds

    def _tests_toneout_cooldown_seconds(self) -> int:
        return self.cfg.tests.toneout_cooldown_seconds

    def _tests_cap_block_seconds(self) -> int:
        return self.cfg.tests.cap_block_seconds

    def _tests_ern_block_seconds(self) -> int:
        return self.cfg.tests.ern_block_seconds

    def _tests_gate(self) -> tuple[bool, str]:
        now = dt.datetime.now(tz=self._tz)

        if self.heightened_until and now < self.heightened_until:
            return (False, "heightened mode active")
        if self.last_toneout_at:
            if (now - self.last_toneout_at).total_seconds() < self._tests_toneout_cooldown_seconds():
                return (False, "recent tone-out cooldown")

        if self.cap_last_severe_at:
            if (now - self.cap_last_severe_at).total_seconds() < self._tests_cap_block_seconds():
                return (False, "recent severe CAP match")

        if self.ern_last_tone_at:
            if (now - self.ern_last_tone_at).total_seconds() < self._tests_ern_block_seconds():
                return (False, "recent ERN SAME activity")

        return (True, "ok")

    async def _originate_required_test(self, event_code: str) -> None:
        """
        Originates a local RWT/RMT using the existing SAME+audio pipeline.
        Does NOT trigger heightened mode.
        """
        code = (event_code or "").strip().upper()
        if code not in {"RWT", "RMT"}:
            return

        # KEEP YOUR LONG FORM WORDING (unchanged)
        if code == "RWT":
            lines = [
                "This is a required weekly test of the SeasonalWeather alert stream. This is only a test.",
                "SeasonalWeather is an internet delivered weather broadcast stream. It does not transmit over the air.",
                "The digital header bursts and the attention signal you are hearing are included for monitoring and system verification.",
                "Some listeners feed this stream into software or hardware decoders for logging, automation, or research.",
                "During real hazardous weather, this stream may relay official alert messages from upstream sources. This weekly test does not indicate an emergency.",
                "This test verifies that the audio generation pipeline is healthy, including text to speech, tone generation, SAME header formatting, and end of message signaling.",
                "This test also verifies stream continuity, including that the live encoder is running, the output mount is reachable, and the automation can cut in and return to the normal cycle.",
                "If you operate a decoder, check that your audio input is clean and unmodified.",
                "Avoid aggressive noise suppression, echo cancellation, or automatic gain control that can smear the header tones.",
                "Verify that your decoder is receiving the expected sample rate, and that it is not being resampled or converted in a way that changes timing.",
                "Confirm that your input level is strong enough to decode, but not clipping. Clipping can make headers unreadable.",
                "If your decoder did not trigger, verify that it is configured to monitor an internet audio source, and confirm that it is listening during the test window.",
                "If you are simply listening in a browser or app, no action is required.",
                "Weekly tests may be postponed when severe weather is active, or when other alerts are being relayed.",
                "This concludes the required weekly test of the SeasonalWeather alert stream. No action is required.",
                "End of message.",
            ]
        else:
            lines = [
                "This is a required monthly test of the SeasonalWeather alert stream. This is only a test.",
                "SeasonalWeather is an internet delivered weather broadcast stream. It does not transmit over the air.",
                "Monthly tests are longer form checks intended to exercise the full alert path under normal conditions.",
                "The digital header bursts identify the test type and targeted areas for decoders that monitor this stream audio.",
                "The attention signal is included to clearly mark the start of the message in recordings and automated systems.",
                "During real events, this stream may relay official alert messages from upstream sources, and may include protective action guidance when available.",
                "This monthly test does not indicate an emergency, and no action is required.",
                "This test verifies that the system can originate a test message, generate headers, generate tones, synthesize speech, and return cleanly to the normal broadcast cycle.",
                "It also verifies that gating logic behaves correctly, meaning tests can be delayed or skipped during severe weather activity.",
                "If you maintain a decoder or logging setup, verify that timestamps, event codes, and location targeting are being interpreted correctly.",
                "Verify that your monitoring chain is not altering the audio in a way that harms decoding.",
                "Avoid speed changes, time stretching, heavy compression, or filters that distort the header tones.",
                "If you run an automated ingest, verify that your software does not drop the beginning of the message due to buffering or reconnect behavior.",
                "If your system reconnects to the stream, confirm it resumes fast enough to capture the header bursts.",
                "If you record the stream, verify that the recording begins before the headers and does not trim leading audio.",
                "If your decoder did not trigger, check input level, sample rate, and configuration, then verify the stream source URL is correct.",
                "If your decoder triggered but content was unclear, check for clipping, resampling artifacts, or excessive noise processing.",
                "If you are simply listening casually, no action is required.",
                "Monthly tests may be postponed when severe weather is active, or when other alerts are being relayed.",
                "This concludes the required monthly test of the SeasonalWeather alert stream. No action is required.",
                "End of message.",
            ]

        spoken = "\n".join(lines).strip()

        dummy = SimpleNamespace(product_type=code, awips_id=None, wfo="KLWX", raw_text="")
        out_wav = await self._render_alert_audio(dummy, spoken)

        async with self._cycle_lock:
            try:
                self.telnet.flush_cycle()
            except Exception:
                pass
            tkey = "rwt" if code == "RWT" else "rmt"
            title = self._np_alert_title(tkey, event="")
            meta = self._np_meta(title=title, kind="test", extra={"sw_alert_source": "local", "sw_event_code": code})
            self.telnet.push_alert(str(out_wav), meta=meta)

        # --- Station feed note (radio UI: handled-alerts.json) ---
        try:
            if _sf_enabled() and StationFeedAlert is not None:
                now_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
                exp_utc = now_utc + dt.timedelta(minutes=30)

                # Human label if available, else fall back to the raw code
                try:
                    label = _sf_eas_event_label_full(code)
                except Exception:
                    label = code

                sender = FeedSender(name="SeasonalWeather", kind="origin") if FeedSender else None

                same_codes = []
                try:
                    same_codes = sorted(getattr(self, "_same_fips_allow_set", None) or [])
                except Exception:
                    same_codes = []
                same_codes = [str(x) for x in same_codes[:32]]

                alert = StationFeedAlert(
                    id=_sf_sha1_12(f"test:{code}:{now_utc.isoformat()}"),
                    event=str(label),
                    headline=f"{label} (Local origination)",
                    severity="Unknown",
                    urgency="Unknown",
                    certainty="Unknown",
                    area="SeasonalWeather",
                    effective=now_utc.isoformat(),
                    ends=exp_utc.isoformat(),
                    expires=exp_utc.isoformat(),
                    sent=now_utc.isoformat(),
                    sameCodes=same_codes,
                    from_=sender,
                    links={"mode": "TEST", "wav": str(out_wav)},
                )
                _sf_emit(alert, expires_at=exp_utc)
        except Exception:
            log.exception("Station feed: failed to note originated %s test", code)

        self._schedule_cycle_refill("post-test")
        log.info("Originated %s test (audio=%s)", code, out_wav)
        # _TEST_DL_v3_
        self.discord.alert_aired(
            code=code,
            event=f"Required {'Weekly' if code == 'RWT' else 'Monthly'} Test",
            source="SeasonalWeather (local)",
            mode="full",
            is_test=True,
        )

    async def run(self) -> None:
        work, audio, cache, logs = self._paths()
        for p in (work, audio, cache, logs):
            p.mkdir(parents=True, exist_ok=True)

        await self._wait_for_liquidsoap()
        self.discord.service_started(
            cap_enabled=self._cap_enabled(),
            ern_enabled=self._ern_enabled(),
            tests_enabled=self._tests_enabled(),
            mode=self.mode,
        )

        # --- Persistent alert state: restore from disk, drop expired ---
        try:
            _loaded = self.alert_tracker.load()
            _purged = self.alert_tracker.purge_expired()
            log.info(
                "AlertTracker: loaded %d entries, purged %d expired on startup",
                _loaded, _purged,
            )
        except Exception:
            log.exception("AlertTracker: startup load/purge failed")
        # _TRACKER_DL_
        try:
            self.discord.alerttracker_lifecycle(
                loaded=_loaded,
                purged=_purged,
                active=len(self.alert_tracker.get_cycle_alerts()),
            )
        except Exception:
            pass

        try:
            _sf_restored_file = _sf_seed_memory_from_payload_file()
            _sf_restored_tracker = self._station_feed_seed_from_alert_tracker()
            if (_sf_restored_file or _sf_restored_tracker) and _sf_enabled():
                _sf_write(time.time())
                log.info(
                    "Station feed: restored %d alerts from handled-alerts.json and %d from AlertTracker on startup",
                    _sf_restored_file, _sf_restored_tracker,
                )
        except Exception:
            log.exception("Station feed: startup restore from disk/tracker failed")

        tasks: list[asyncio.Task] = []

        if self.live_time_enabled:
            tasks.append(asyncio.create_task(self._live_time_loop(), name="live_time_wav"))

        xmpp = NWWSClient(
            self.jid, self.password, self.nwws_server, self.nwws_port, self.nwws_queue,
            room_jid=self.cfg.nwws.room,
            nick=self.cfg.nwws.nick,
            # TODO: wire stall/reconnect callbacks to self.discord.nwws_stall() / .nwws_reconnected() once NWWSClient exposes them
            stall_seconds=self.cfg.nwws.resiliency.stall_seconds,
            muc_confirm_seconds=self.cfg.nwws.resiliency.muc_confirm_seconds,
            start_wait_seconds=self.cfg.nwws.resiliency.start_wait_seconds,
            join_wait_seconds=self.cfg.nwws.resiliency.join_wait_seconds,
            backoff_max_seconds=self.cfg.nwws.resiliency.backoff_max_seconds,
        )
        tasks.append(asyncio.create_task(xmpp.run_forever(), name="nwws_xmpp"))
        tasks.append(asyncio.create_task(self._consume_nwws(), name="nwws_consumer"))
        tasks.append(asyncio.create_task(self._cycle_loop(), name="cycle_loop"))

        if self.rebroadcast_enabled:
            tasks.append(asyncio.create_task(self._rebroadcast_loop(), name="rebroadcast_loop"))
            log.info(
                "Rebroadcast enabled (interval=%ss ttl=%ss max_items=%d include_voice=%s)",
                self.rebroadcast_interval_seconds,
                self.rebroadcast_ttl_seconds,
                self.rebroadcast_max_items,
                self.rebroadcast_include_voice,
            )

        if self._cap_enabled():
            if NwsCapPoller is None or CapAlertEvent is None:
                log.warning("CAP enabled but cap_nws.py import failed; CAP is disabled.")
            else:
                kwargs = dict(
                    out_queue=self.cap_queue,
                    same_fips_allow=self.cfg.service_area.same_fips_all,
                    poll_seconds=self._cap_poll_seconds(),
                    user_agent=self._cap_user_agent(),
                    ledger_path=self.cfg.cap.ledger_path,
                    ledger_max_age_days=self.cfg.cap.ledger_max_age_days,
                )
                url = self._cap_url().strip()
                if url:
                    kwargs["url"] = url  # type: ignore[assignment]

                cap = NwsCapPoller(**kwargs)  # type: ignore[arg-type]
                tasks.append(asyncio.create_task(cap.run_forever(), name="cap_poller"))
                tasks.append(asyncio.create_task(self._consume_cap(), name="cap_consumer"))
                log.info("CAP ingest enabled (dryrun=%s full=%s voice=%s)", self._cap_dryrun(), self._cap_full_enabled(), self._cap_voice_enabled())
        else:
            log.info("CAP ingest disabled (set cap.enabled: true in config.yaml to enable)")

        if self._ern_enabled():
            if ErnGwesMonitor is None or ErnSameEvent is None:
                log.warning("ERN enabled but ern_gwes.py import failed; ERN is disabled.")
            else:
                url = self._ern_url()
                if not url:
                    log.warning("ERN enabled but SEASONAL_ERN_URL is empty; ERN is disabled.")
                else:
                    ern_cfg = self.cfg.ern
                    mon = ErnGwesMonitor(
                        out_queue=self.ern_queue,
                        same_fips_allow=self.cfg.service_area.same_fips_all,
                        url=url,
                        sample_rate=ern_cfg.sample_rate,
                        dedupe_seconds=ern_cfg.dedupe_seconds,
                        trigger_ratio=ern_cfg.trigger_ratio,
                        tail_seconds=ern_cfg.tail_seconds,
                        confidence_min=ern_cfg.confidence_min,
                        name=ern_cfg.name,
                    )
                    tasks.append(asyncio.create_task(mon.run_forever(), name="ern_monitor"))
                    tasks.append(asyncio.create_task(self._consume_ern(), name="ern_consumer"))
                    log.info(
                        "ERN monitor enabled (dryrun=%s url=%s relay=%s)",
                        self._ern_dryrun(),
                        url,
                        self._ern_relay_enabled(),
                    )
        else:
            log.info("ERN monitor disabled (set ern.enabled: true in config.yaml to enable)")

        if self._tests_enabled():
            try:
                state_path = str(Path(self.cfg.paths.work_dir) / "rwt_rmt_state.json")

                sched = RwtRmtSchedule(
                    enabled=True,
                    tz_name=self.cfg.station.timezone,

                    rwt_enabled=True,
                    rwt_weekday=self.cfg.tests.rwt.weekday,
                    rwt_hour=self.cfg.tests.rwt.hour,
                    rwt_minute=self.cfg.tests.rwt.minute,

                    rmt_enabled=True,
                    rmt_nth=self.cfg.tests.rmt.nth,
                    rmt_weekday=self.cfg.tests.rmt.weekday,
                    rmt_hour=self.cfg.tests.rmt.hour,
                    rmt_minute=self.cfg.tests.rmt.minute,

                    jitter_seconds=self._tests_jitter_seconds(),
                    postpone_minutes=self._tests_postpone_minutes(),
                    max_postpone_hours=self._tests_max_postpone_hours(),
                    state_path=state_path,
                )

                def _rlog(s: str) -> None:
                    log.info("%s", s)

                rsch = RwtRmtScheduler(
                    schedule=sched,
                    gate_fn=self._tests_gate,
                    fire_fn=self._originate_required_test,
                    log_fn=_rlog,
                )
                tasks.append(asyncio.create_task(rsch.run_forever(), name="rwt_rmt_scheduler"))
                log.info("RWT/RMT scheduler enabled (state=%s)", state_path)
            except Exception:
                log.exception("Failed to start RWT/RMT scheduler")
        else:
            log.info("RWT/RMT scheduler disabled (set tests.enabled: true in config.yaml to enable)")

        tasks.append(asyncio.create_task(self.discord.start(), name="discord_log_drain"))
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            exc = t.exception()
            if exc:
                for p in pending:
                    p.cancel()
                raise exc

    async def _consume_nwws(self) -> None:
        while True:
            raw = await self.nwws_queue.get()

            # Flood-gate: allow the client logs for the first N messages, then silence them.
            self._nwws_raw_seen += 1
            if self._nwws_rx_log_first_n > 0 and self._nwws_raw_seen == self._nwws_rx_log_first_n:
                self._nwws_logger.setLevel(logging.WARNING)
                log.info(
                    "NWWS RX logging throttled after %d messages (seasonalweather.nwws -> WARNING)",
                    self._nwws_rx_log_first_n,
                )

            parsed = parse_product_text(raw)
            if not parsed:
                continue

            self._nwws_seen += 1

            allowed_wfo = (not self._nwws_allowed_wfos) or (parsed.wfo in self._nwws_allowed_wfos)
            toneout = parsed.product_type in self.cfg.policy.toneout_product_types

            first_n = max(0, int(self._nwws_decision_log_first_n))
            every = max(0, int(self._nwws_decision_log_every))
            if (first_n and self._nwws_seen <= first_n) or (every and (self._nwws_seen % every) == 0):
                log.info(
                    "NWWS decision: #%d type=%s awips=%s wfo=%s allowed=%s toneout=%s",
                    self._nwws_seen,
                    parsed.product_type,
                    parsed.awips_id or "",
                    parsed.wfo,
                    allowed_wfo,
                    toneout,
                )

            self.last_product_desc = f"{parsed.product_type} ({parsed.awips_id or ''})"

            if not allowed_wfo:
                continue

            if toneout:
                await self._handle_toneout(parsed)

            # PNS cycle injection — SEVERE WEATHER SAFETY RULES bulletins go into
            # the broadcast cycle as a voice-only segment (no cut-in, no tones).
            elif (parsed.product_type or "").strip().upper() == "PNS":
                try:
                    official_pns = parsed.raw_text
                    try:
                        pid_pns = await self.api.latest_product_id("PNS", parsed.wfo[1:] if parsed.wfo.startswith("K") else parsed.wfo)
                        if pid_pns:
                            prod_pns = await self.api.get_product(pid_pns)
                            if prod_pns and prod_pns.product_text:
                                official_pns = prod_pns.product_text
                    except Exception:
                        pass

                    if self._is_safety_rules_pns(official_pns):
                        if not self._pns_safety_is_fresh(official_pns, getattr(parsed, "issued", None)):
                            log.info("PNS safety rules skipped (stale product) wfo=%s awips=%s", parsed.wfo, parsed.awips_id or "")
                            continue
                        pns_script = self._build_pns_safety_script(official_pns)
                        if pns_script.strip():
                            pns_key = f"PNS_SAFETY:{(parsed.wfo or '').strip()}:{self._sha1_12(official_pns[:800])}"
                            ok_pns, _ = await self._dedupe_reserve([pns_key])
                            if ok_pns:
                                try:
                                    pns_exp_utc = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=4)
                                    pns_ae = ActiveAlert(
                                        id=pns_key,
                                        source="PNS_CYCLE",
                                        event="Severe Weather Safety Rules",
                                        code="SPS",
                                        vtec=[],
                                        headline="Severe Weather Safety Rules",
                                        script_text=pns_script,
                                        audio_path=None,
                                        expires=pns_exp_utc.isoformat(),
                                        issued=dt.datetime.now(dt.timezone.utc).isoformat(),
                                        same_locs=list(self.cfg.service_area.same_fips_all or []),
                                        cycle_only=True,
                                    )
                                    self.alert_tracker.add_or_update(pns_ae)
                                    self._schedule_cycle_refill("pns-safety-rules")
                                    log.info(
                                        "PNS SAFETY RULES queued for cycle id=%s wfo=%s awips=%s",
                                        pns_key, parsed.wfo, parsed.awips_id or "",
                                    )
                                except Exception:
                                    log.exception("PNS safety rules cycle inject failed")
                            else:
                                log.info("PNS safety rules skipped (dedupe) wfo=%s", parsed.wfo)
                except Exception:
                    log.exception("PNS safety rules handler error wfo=%s", parsed.wfo)

    def _cap_is_actionable(self, ev: "CapAlertEvent") -> bool:  # type: ignore[name-defined]
        try:
            if str(ev.status or "").strip().lower() != "actual":
                return False
            mt = str(ev.message_type or "").strip().lower()
            if mt and mt not in {"alert", "update", "cancel"}:
                return False
        except Exception:
            return False
        return True

    def _cap_severity_str(self, ev: "CapAlertEvent") -> str:  # type: ignore[name-defined]
        return str(ev.severity or "").strip().lower()

    def _cap_event_to_same_code(self, event: str) -> str:
        e = (event or "").strip()
        m: dict[str, str] = {
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
        if e in m:
            return m[e]

        words = [w for w in re.split(r"\s+", e) if w]
        if len(words) >= 1:
            code = "".join(ch for ch in "".join(w[0] for w in words[:3]) if ch.isalnum()).upper()
            if len(code) >= 3:
                return code[:3]

        return "SPS"


    def _cap_should_full(self, ev: "CapAlertEvent") -> bool:  # type: ignore[name-defined]
        if not self._cap_full_enabled():
            return False
        if not self._cap_is_actionable(ev):
            return False

        # VTEC-aware gate for CAP Update:
        # CAP can send VTEC CON/EXT/COR/ROU as msgType=Update.
        # Do NOT FULL-tone those unless VTEC indicates a FULL-worthy action.
        try:
            mt = str(ev.message_type or "").strip().lower()
        except Exception:
            mt = ""

        if mt == "update":
            try:
                _cap_upd_policy = _vtec_toneout_policy(self._cap_vtec_list(ev))
                if _cap_upd_policy.mode != "FULL":
                    return False
            except Exception:
                # Parsing failed; be conservative on updates.
                return False

        event = (ev.event or "").strip()
        if event and event in self._cap_full_events():
            return True

        sev = self._cap_severity_str(ev)
        if sev and sev in self._cap_full_severities():
            return True

        return False

    def _cap_should_voice(self, ev: "CapAlertEvent") -> bool:  # type: ignore[name-defined]
        if not self._cap_voice_enabled():
            return False
        if not self._cap_is_actionable(ev):
            return False
        allow_events = self._cap_voice_events()
        if allow_events and (ev.event or "").strip() not in allow_events:
            return False
        return True

    def _cap_should_update(self, ev: "CapAlertEvent") -> bool:  # type: ignore[name-defined]
        """
        True for CAP messageType=Update with CON/EXT/CAN/EXP actions on events we
        already watch-and-warn on.  These get voice-only narration (no SAME tones).
        """
        if not self._cap_full_enabled():
            return False
        if not self._cap_is_actionable(ev):
            return False
        mt = str(ev.message_type or "").strip().lower()
        if mt not in {"update", "cancel"}:
            return False
        event = (ev.event or "").strip()
        if event not in self._cap_full_events():
            return False
        vtec = self._cap_vtec_list(ev)
        tracks = self._vtec_tracks(vtec)
        update_actions = {"CON", "EXT", "CAN", "EXP"}
        vtec_actions = {act for (_t, act) in tracks} if tracks else set()
        return bool(vtec_actions & update_actions)

    async def _consume_cap(self) -> None:
        while True:
            ev = await self.cap_queue.get()

            vtec = self._cap_vtec_list(ev)
            tracks = self._vtec_tracks(vtec)
            cap_mt = str(getattr(ev, "message_type", None) or "").strip().lower()
            cap_ref_ids = _sf_cap_reference_ids(ev)

            if cap_mt == "cancel" and not tracks:
                try:
                    same_code = self._cap_event_to_same_code((ev.event or "").strip())
                    self.alert_tracker.remove(self._alert_tracker_id_for_cap(ev, same_code))
                except Exception:
                    log.exception("AlertTracker: failed handling CAP cancel without VTEC id=%s", getattr(ev, "alert_id", None))
                _sf_remove_ids(cap_ref_ids + [getattr(ev, "alert_id", None)])
                log.info("CAP cancel: evicted state without airing id=%s refs=%s", getattr(ev, "alert_id", None), ",".join(cap_ref_ids[:4]))
                continue

            # Keep rebroadcast rotation in sync with VTEC lifecycle.
            # - CAN/EXP => remove immediately so we don't re-air a dead event.
            # - CON/EXT/etc => refresh expiry for any existing rotation items.
            if self.rebroadcast_enabled and tracks:
                vtec_actions = {act for (_t, act) in tracks} if tracks else set()
                now_local = dt.datetime.now(self.local_tz)
                exp_utc = self._best_expiry_from_vtec(vtec)
                expires_at = self._rebroadcast_clamp_expiry(now_local, exp_utc)
                mt = str(getattr(ev, "message_type", None) or "Alert")

                if vtec_actions & {"CAN", "EXP"}:
                    await self._rebroadcast_remove_by_tracks(
                        tracks=tracks,
                        reason=f"cap:{mt}:{','.join(sorted(vtec_actions & {'CAN','EXP'}))}",
                    )
                    _sf_remove_by_vtec_tracks(tracks)
                    _sf_remove_ids(cap_ref_ids + [getattr(ev, "alert_id", None)])
                    # Also remove from AlertTracker (if not already handled by _air_cap_update)
                    try:
                        _can_track_ids = {_vtec_track_id(v) for v in vtec if _vtec_track_id(v)}
                        self.alert_tracker.remove_by_vtec_tracks(
                            _can_track_ids,  # type: ignore[arg-type]
                            reason=f"cap-consume:{mt}:{','.join(sorted(vtec_actions & {'CAN','EXP'}))}",
                        )
                    except Exception:
                        log.exception("AlertTracker: removal failed in _consume_cap CAN/EXP block")
                else:
                    await self._rebroadcast_touch_expiry_by_tracks(
                        tracks=tracks,
                        expires_at=expires_at,
                        reason=f"cap:{mt}:{','.join(sorted(vtec_actions))}",
                    )

            log.info(
                "CAP match: event=%s severity=%s urgency=%s certainty=%s status=%s msgType=%s sent=%s same=%s headline=%s id=%s vtec=%s tracks=%s",
                ev.event,
                ev.severity,
                ev.urgency,
                ev.certainty,
                ev.status,
                ev.message_type,
                ev.sent,
                ",".join(ev.same_fips[:12]) + ("..." if len(ev.same_fips) > 12 else ""),
                ev.headline,
                ev.alert_id,
                ",".join(vtec[:2]) if vtec else "",
                ",".join(t for (t, _a) in tracks[:2]) if tracks else "",
            )

            try:
                sev = str(ev.severity or "").strip().lower()
                if sev in {"severe", "extreme"}:
                    self.cap_last_severe_at = dt.datetime.now(tz=self._tz)
            except Exception:
                pass

            if self._cap_dryrun():
                continue

            if self._cap_should_full(ev):
                await self._air_cap_full(ev)
                continue

            # CON/EXT/CAN/EXP for watched/warned events → voice-only update narration
            if self._cap_should_update(ev):
                await self._air_cap_update(ev)
                continue

            if self._cap_should_voice(ev):
                await self._air_cap_voice(ev)

    async def _air_cap_full(self, ev: "CapAlertEvent") -> None:  # type: ignore[name-defined]
        now = dt.datetime.now(tz=self._tz)

        key = (str(ev.alert_id or "").strip(), str(ev.sent or "").strip())
        last = self._cap_full_last_by_key.get(key)
        if last and (now - last).total_seconds() < self._cap_full_cooldown_seconds():
            log.info("CAP full: cooldown active; skipping id=%s sent=%s event=%s", ev.alert_id, ev.sent, ev.event)
            return

        vtec = self._cap_vtec_list(ev)
        tracks = self._vtec_tracks(vtec)
        vtec_actions = {act for (_t, act) in tracks} if tracks else set()

        ev_event = (ev.event or "").strip()
        _WATCH_EVENTS = {"Tornado Watch", "Severe Thunderstorm Watch"}
        is_watch = ev_event in _WATCH_EVENTS

        # ---- Determine watch number and kind for TOA/SVA ----
        watch_number: int | None = None
        watch_kind = "tornado"
        if is_watch:
            for v in vtec:
                m = _VTEC_PARSE_RE.search(v)
                if not m:
                    continue
                phen = (m.group("phen") or "").upper()
                sig = (m.group("sig") or "").upper()
                if sig != "A":
                    continue
                if phen == "TO":
                    watch_kind = "tornado"
                elif phen == "SV":
                    watch_kind = "severe"
                else:
                    continue
                try:
                    watch_number = int(m.group("etn"))
                except Exception:
                    pass
                break

        # ---- Route to appropriate script builder ----
        if is_watch:
            if vtec_actions & {"EXA", "EXB"}:
                # Watch expansion: full announcement with SAME for added counties
                script = self._build_watch_expansion_script(ev)
            else:
                # NEW or UPG watch
                script = self._build_cap_watch_script(ev, mode="full")
            if not script.strip():
                script = self._build_cap_full_script(ev)
        else:
            script = self._build_cap_full_script(ev)

        if not script.strip():
            return

        same_code = _vtec_toneout_policy(vtec).same_code or self._cap_event_to_same_code(ev_event)
        same_locs_raw = list(ev.same_fips) if getattr(ev, "same_fips", None) else []
        same_locs = self._filter_same_locations_to_service_area(same_locs_raw)

        keys: list[str] = []

        # Track-level dedupe (prevents CAP vs NWWS double-air)
        for track_id, _act in tracks:
            keys.append(f"TRACKFULL:{track_id}")

        # Also keep raw VTEC strings (fine-grain)
        for v in vtec:
            keys.append(f"VTEC:{v}")

        # Functional FULL dedupe is only safe when we do NOT have a concrete
        # VTEC track.  Otherwise, two distinct warnings for the same counties can
        # collide (for example TO.W.0004 vs TO.W.0005).
        if not tracks:
            fkey = self._dedupe_func_full_key(same_code, same_locs)
            if fkey:
                keys.append(fkey)

        fips_part = ",".join(sorted(set(str(x).strip() for x in (same_locs or []) if str(x).strip())))[:800]
        keys.append(f"CAPFULL:{(ev.event or '').strip()}:{(ev.sent or '').strip()}:{self._sha1_12((ev.alert_id or '') + '|' + fips_part)}")

        ok, hit = await self._dedupe_reserve(keys)
        if not ok:
            log.info(
                "CAP full skipped (dedupe hit=%s) id=%s sent=%s event=%s vtec=%s",
                hit,
                ev.alert_id,
                ev.sent,
                ev.event,
                ",".join(vtec[:2]) if vtec else "",
            )
            return

        try:
            dummy = SimpleNamespace(product_type=same_code, awips_id=None, wfo="CAP", raw_text="")
            out_wav = await self._render_alert_audio(dummy, script, same_locations=same_locs if same_locs else None)

            async with self._cycle_lock:
                try:
                    self.telnet.flush_cycle()
                except Exception:
                    pass
                event_label = (ev.event or "").strip() or "Weather alert"
                title = self._np_alert_title("cap_full", event=event_label)
                meta = self._np_meta(
                    title=title,
                    kind="alert",
                    extra={
                        "sw_alert_source": "cap",
                        "sw_alert_mode": "full",
                        "sw_event": event_label,
                        "sw_event_code": (same_code or "").strip().upper(),
                        "sw_alert_id": str(ev.alert_id or "").strip(),
                    },
                )
                self.telnet.push_alert(str(out_wav), meta=meta)

            self._cap_full_last_by_key[key] = now
            self.last_product_desc = f"CAP {ev.event}".strip()

            try:
                code_u = (same_code or "").strip().upper()
                if code_u and code_u in self.cfg.policy.toneout_product_types:
                    self.last_toneout_at = now
                    self.last_heightened_at = now
                    self.heightened_until = now + dt.timedelta(seconds=self.cfg.cycle.min_heightened_seconds)
                    self._update_mode()
            except Exception:
                pass

            self._schedule_cycle_refill("post-cap-full")

            # Rebroadcast rotation (no re-tone)
            try:
                if self.rebroadcast_enabled:
                    exp_utc = self._best_expiry_from_vtec(vtec)
                    expires_at = self._rebroadcast_clamp_expiry(now, exp_utc)
                    key_rb = self._rebroadcast_make_key(
                        source="CAP",
                        kind="FULL",
                        event_code=same_code,
                        tracks=tracks,
                        same_locs=same_locs,
                        script=script,
                    )
                    desc_rb = f"CAP FULL {ev.event}".strip()
                    await self._rebroadcast_add(key=key_rb, desc=desc_rb, script=script, expires_at=expires_at)
            except Exception:
                log.exception("Rebroadcast: failed to add CAP FULL item")
            log.info("CAP ACTION: aired FULL event=%s code=%s id=%s sent=%s vtec=%s audio=%s", ev.event, same_code, ev.alert_id, ev.sent, ",".join(vtec[:2]) if vtec else "", out_wav)
            # _CAP_FULL_DL_
            self.discord.alert_aired(
                code=same_code,
                event=(ev.event or "").strip(),
                source="CAP",
                mode="full",
                area=getattr(ev, "area_desc", "") or "",
                vtec=vtec[:2],
                expires=self._fmt_local_from_utc_iso(
                    str(getattr(ev, "expires", "") or "")
                ),
            )
            _station_feed_note_cap(ev, mode="FULL", same_locations=(same_locs if same_locs else same_locs_raw), out_wav=str(out_wav), same_code=same_code, vtec=vtec)

            # ---- Register to AlertTracker for restart-safe rebroadcast ----
            try:
                tracker_id = self._alert_tracker_id_for_cap(ev, same_code)
                expires_iso = self._alert_expires_from_cap(ev, vtec)
                _is_watch = (ev.event or "").strip() in {"Tornado Watch", "Severe Thunderstorm Watch"}
                _watch_num: int | None = None
                if _is_watch:
                    for _v in vtec:
                        _m = _VTEC_PARSE_RE.search(_v)
                        if _m and (_m.group("sig") or "").upper() == "A":
                            try:
                                _watch_num = int(_m.group("etn"))
                            except Exception:
                                pass
                            break
                alert_entry = ActiveAlert(
                    id=tracker_id,
                    source="CAP",
                    event=str(ev.event or ""),
                    code=same_code,
                    vtec=vtec,
                    headline=str(ev.headline or ""),
                    script_text=script,
                    audio_path=str(out_wav),
                    expires=expires_iso,
                    issued=str(ev.sent or dt.datetime.now(dt.timezone.utc).isoformat()),
                    same_locs=same_locs,
                    cycle_only=False,
                    watch_number=_watch_num,
                )
                self.alert_tracker.add_or_update(alert_entry)
                self.alert_tracker.mark_aired(tracker_id)
                log.info("AlertTracker: registered CAP FULL id=%s event=%s expires=%s", tracker_id, ev.event, expires_iso)
            except Exception:
                log.exception("AlertTracker: failed to register CAP FULL event=%s", ev.event)
        except Exception:
            await self._dedupe_release(keys)
            raise

    async def _air_cap_voice(self, ev: "CapAlertEvent") -> None:  # type: ignore[name-defined]
        now = dt.datetime.now(tz=self._tz)

        key = (str(ev.alert_id or "").strip(), str(ev.sent or "").strip())
        last = self._cap_voice_last_by_key.get(key)
        if last and (now - last).total_seconds() < self._cap_voice_cooldown_seconds():
            log.info("CAP voice: cooldown active; skipping id=%s sent=%s event=%s", ev.alert_id, ev.sent, ev.event)
            return

        script = self._build_cap_voice_script(ev)
        if not script.strip():
            return

        vtec = self._cap_vtec_list(ev)
        tracks = self._vtec_tracks(vtec)

        same_code = _vtec_toneout_policy(vtec).same_code or self._cap_event_to_same_code((ev.event or "").strip())
        same_locs_raw = list(ev.same_fips) if getattr(ev, "same_fips", None) else []
        same_locs = self._filter_same_locations_to_service_area(same_locs_raw)

        keys: list[str] = []
        for track_id, _act in tracks:
            keys.append(f"TRACKVOICE:{track_id}")

        fips_part = ",".join(sorted(set(str(x).strip() for x in (ev.same_fips or []) if str(x).strip())))[:800]
        keys.append(f"CAPVOICE:{(ev.event or '').strip()}:{(ev.sent or '').strip()}:{self._sha1_12((ev.alert_id or '') + '|' + fips_part)}")

        ok, hit = await self._dedupe_reserve(keys)
        if not ok:
            log.info(
                "CAP voice skipped (dedupe hit=%s) id=%s sent=%s event=%s vtec=%s",
                hit,
                ev.alert_id,
                ev.sent,
                ev.event,
                ",".join(vtec[:2]) if vtec else "",
            )
            return

        try:
            out_wav = await self._render_voice_only_audio(script, prefix="capvoice")

            async with self._cycle_lock:
                try:
                    self.telnet.flush_cycle()
                except Exception:
                    pass
                event_label = (ev.event or "").strip() or "Weather alert"
                title = self._np_alert_title("cap_update", event=event_label)
                meta = self._np_meta(
                    title=title,
                    kind="alert",
                    extra={
                        "sw_alert_source": "cap",
                        "sw_alert_mode": "voice",
                        "sw_event": event_label,
                        "sw_event_code": (same_code or "").strip().upper(),
                        "sw_alert_id": str(ev.alert_id or "").strip(),
                    },
                )
                self.telnet.push_alert(str(out_wav), meta=meta)

            self._cap_voice_last_by_key[key] = now
            self.last_product_desc = f"CAP {ev.event}".strip()

            self._schedule_cycle_refill("post-cap-voice")

            # Rebroadcast rotation (no re-tone) - optional for voice-only CAP
            try:
                if self.rebroadcast_enabled and self.rebroadcast_include_voice:
                    exp_utc = self._best_expiry_from_vtec(vtec)
                    expires_at = self._rebroadcast_clamp_expiry(now, exp_utc)
                    same_code2 = _vtec_toneout_policy(vtec).same_code or self._cap_event_to_same_code((ev.event or "").strip())
                    key_rb = self._rebroadcast_make_key(
                        source="CAP",
                        kind="VOICE",
                        event_code=same_code2,
                        tracks=tracks,
                        same_locs=same_locs,
                        script=script,
                    )
                    desc_rb = f"CAP VOICE {ev.event}".strip()
                    await self._rebroadcast_add(key=key_rb, desc=desc_rb, script=script, expires_at=expires_at)
            except Exception:
                log.exception("Rebroadcast: failed to add CAP VOICE item")
            log.info("CAP ACTION: aired voice-only event=%s id=%s sent=%s audio=%s", ev.event, ev.alert_id, ev.sent, out_wav)
            # _CAP_VOICE_DL_
            self.discord.alert_aired(
                code=same_code,
                event=(ev.event or "").strip(),
                source="CAP",
                mode="voice",
                area=getattr(ev, "area_desc", "") or "",
                vtec=vtec[:2],
            )

            # Register to AlertTracker (cycle_only → no SAME retone on cycle replay)
            try:
                vtec_v = self._cap_vtec_list(ev)
                tracker_id_v = self._alert_tracker_id_for_cap(ev, same_code)
                expires_iso_v = self._alert_expires_from_cap(ev, vtec_v)
                _ae = ActiveAlert(
                    id=tracker_id_v,
                    source="CAP",
                    event=str(ev.event or ""),
                    code=same_code,
                    vtec=vtec_v,
                    headline=str(ev.headline or ""),
                    script_text=script,
                    audio_path=str(out_wav),
                    expires=expires_iso_v,
                    issued=str(ev.sent or dt.datetime.now(dt.timezone.utc).isoformat()),
                    same_locs=same_locs,
                    cycle_only=True,
                )
                self.alert_tracker.add_or_update(_ae)
                self.alert_tracker.mark_aired(tracker_id_v)
            except Exception:
                log.exception("AlertTracker: failed to register CAP VOICE event=%s", ev.event)
            _station_feed_note_cap(
                ev,
                mode="VOICE",
                same_locations=(same_locs if same_locs else same_locs_raw),
                out_wav=str(out_wav),
                same_code=same_code,
                vtec=vtec,
            )
        except Exception:
            await self._dedupe_release(keys)
            raise


    async def _air_cap_update(self, ev: "CapAlertEvent") -> None:  # type: ignore[name-defined]
        """
        Voice-only narration for VTEC CON/EXT/CAN/EXP on already-aired events.
        No SAME tones.  Removes entry from AlertTracker on CAN/EXP.
        """
        now = dt.datetime.now(tz=self._tz)
        vtec = self._cap_vtec_list(ev)
        tracks = self._vtec_tracks(vtec)
        vtec_actions = {act for (_t, act) in tracks} if tracks else set()

        ev_event = (ev.event or "").strip()
        _WATCH_EVENTS = {"Tornado Watch", "Severe Thunderstorm Watch"}
        is_watch = ev_event in _WATCH_EVENTS

        # Determine watch number/kind for watches
        watch_number: int | None = None
        watch_kind = "tornado"
        if is_watch:
            for v in vtec:
                m = _VTEC_PARSE_RE.search(v)
                if not m:
                    continue
                phen = (m.group("phen") or "").upper()
                sig = (m.group("sig") or "").upper()
                if sig != "A":
                    continue
                watch_kind = "tornado" if phen == "TO" else "severe"
                try:
                    watch_number = int(m.group("etn"))
                except Exception:
                    pass
                break

        if is_watch:
            script = self._build_watch_vtec_action_script(ev, vtec_actions, tracks, watch_number, watch_kind)
        elif self._cap_prefers_statement_update_script(ev_event, vtec_actions):
            script = self._build_statement_vtec_action_script(ev, vtec_actions, tracks)
        else:
            script = self._build_warning_vtec_action_script(ev, vtec_actions, tracks)

        if not script.strip():
            log.info("CAP update: empty script, skipping event=%s vtec_actions=%s", ev_event, vtec_actions)
            return

        same_code = self._cap_event_to_same_code(ev_event)
        same_locs_raw = list(ev.same_fips) if getattr(ev, "same_fips", None) else []
        same_locs = self._filter_same_locations_to_service_area(same_locs_raw)

        key_str = f"CAPUPDATE:{(ev.alert_id or '').strip()}:{(ev.sent or '').strip()}"
        keys = [key_str]
        for track_id, _ in tracks:
            keys.append(f"TRACKVOICE:{track_id}")

        ok, hit = await self._dedupe_reserve(keys)
        if not ok:
            log.info("CAP update skipped (dedupe hit=%s) event=%s vtec_actions=%s", hit, ev_event, vtec_actions)
            return

        try:
            out_wav = await self._render_voice_only_audio(script, prefix="capupdate")
            async with self._cycle_lock:
                try:
                    self.telnet.flush_cycle()
                except Exception:
                    pass
                event_label = ev_event or "Weather alert"
                title = self._np_alert_title("cap_update", event=event_label)
                meta = self._np_meta(
                    title=title,
                    kind="alert",
                    extra={
                        "sw_alert_source": "cap",
                        "sw_alert_mode": "update",
                        "sw_event": event_label,
                        "sw_event_code": (same_code or "").strip().upper(),
                        "sw_alert_id": str(ev.alert_id or "").strip(),
                    },
                )
                self.telnet.push_alert(str(out_wav), meta=meta)

            self.last_product_desc = f"CAP {ev_event}".strip()
            self._schedule_cycle_refill("post-cap-update")

            # Update or remove from AlertTracker
            try:
                tracker_id = self._alert_tracker_id_for_cap(ev, same_code)
                if vtec_actions & {"CAN", "EXP"}:
                    removed = self.alert_tracker.remove(tracker_id)
                    if not removed:
                        # Try by VTEC track
                        track_ids = {_vtec_track_id(v) for v in vtec if _vtec_track_id(v)}
                        self.alert_tracker.remove_by_vtec_tracks(
                            track_ids,  # type: ignore[arg-type]
                            reason=f"cap-update:{','.join(sorted(vtec_actions))}",
                        )
                    log.info("AlertTracker: removed id=%s event=%s action=%s", tracker_id, ev_event, vtec_actions)
                else:
                    # CON/EXT/EXA: update the stored script to latest narration
                    existing = self.alert_tracker.find_by_vtec_track(tracker_id.replace("CAP:", "")) or self.alert_tracker._alerts.get(tracker_id)
                    if existing:
                        self.alert_tracker.update_script(existing.id, script)
                        self.alert_tracker.mark_aired(existing.id)
                    log.info("AlertTracker: updated id=%s event=%s action=%s", tracker_id, ev_event, vtec_actions)
            except Exception:
                log.exception("AlertTracker: failed to update/remove on CAP update event=%s", ev_event)

            log.info("CAP ACTION: aired UPDATE event=%s code=%s id=%s vtec_actions=%s audio=%s",
                     ev_event, same_code, ev.alert_id, vtec_actions, out_wav)
            # _CAP_UPDATE_DL_
            if vtec_actions & {"CAN", "EXP"}:
                self.discord.alert_expired(
                    code=same_code,
                    event=ev_event,
                    vtec_action=next(iter(vtec_actions & {"CAN", "EXP"})),
                    source="CAP",
                    area=getattr(ev, "area_desc", "") or "",
                    vtec=vtec[:2],
                )
            else:
                self.discord.alert_updated(
                    code=same_code,
                    event=ev_event,
                    vtec_action=next(iter(vtec_actions), "CON"),
                    source="CAP",
                    area=getattr(ev, "area_desc", "") or "",
                    vtec=vtec[:2],
                )
            _station_feed_note_cap(ev, mode="VOICE", same_locations=(same_locs if same_locs else same_locs_raw),
                                   out_wav=str(out_wav), same_code=same_code, vtec=vtec)
        except Exception:
            await self._dedupe_release(keys)
            raise

    def _clean_cap_text(self, s: str, *, limit: int = 900) -> str:
        s2 = (s or "").replace("\r", " ").replace("\n", " ")
        s2 = re.sub(r"\s+", " ", s2).strip()
        s2 = s2.replace("...", ". ").replace("..", ".")
        if len(s2) > limit:
            s2 = s2[:limit].rstrip() + "..."
        return s2


    def _build_cap_watch_script(self, ev: "CapAlertEvent", *, mode: str = "full") -> str:  # type: ignore[name-defined]
        """
        Build a sane, NWR-style script for CAP Tornado Watch / Severe Thunderstorm Watch.
        Returns "" if this CAP event is not a watch.

        Why: CAP watch descriptions are often all-caps blobs with little punctuation,
        which TTS will speed-read. NWR uses a standardized narration instead.
        """
        # ---- Determine watch kind (prefer event label, fall back to VTEC) ----
        kind: str | None = None  # "tornado" or "severe"
        ev_name = (getattr(ev, "event", "") or "").strip().lower()

        if ev_name == "tornado watch":
            kind = "tornado"
        elif ev_name == "severe thunderstorm watch":
            kind = "severe"
        else:
            # Fall back to VTEC phen/sig
            for v in self._cap_vtec_list(ev):
                m = _VTEC_PARSE_RE.search(v)
                if not m:
                    continue
                phen = (m.group("phen") or "").upper()
                sig = (m.group("sig") or "").upper()
                if sig != "A":
                    continue
                if phen == "TO":
                    kind = "tornado"
                    break
                if phen == "SV":
                    kind = "severe"
                    break

        if not kind:
            return ""

        # ---- Helpers ----
        def _parse_vtec_z(tok: str):
            # tok like YYYYMMDDT0000Z or YYMMDDT0000Z
            s = (tok or "").strip().upper()
            mm = re.fullmatch(r"(\d{8}|\d{6})T(\d{4})Z", s)
            if not mm:
                return None
            d = mm.group(1)
            hm = mm.group(2)
            if len(d) == 8:
                year = int(d[0:4]); month = int(d[4:6]); day = int(d[6:8])
            else:
                year = 2000 + int(d[0:2]); month = int(d[2:4]); day = int(d[4:6])
            hour = int(hm[0:2]); minute = int(hm[2:4])
            try:
                return dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc)
            except Exception:
                return None

        def _fmt_time_local(d: dt.datetime) -> str:
            # "8 PM" or "8:30 PM"
            hour12 = d.hour % 12
            if hour12 == 0:
                hour12 = 12
            ampm = "AM" if d.hour < 12 else "PM"
            if d.minute == 0:
                return f"{hour12} {ampm}"
            return f"{hour12}:{d.minute:02d} {ampm}"

        def _daypart(d: dt.datetime) -> str:
            # rough-but-good NWR-ish phrasing
            if d.hour < 12:
                return "morning"
            if d.hour < 17:
                return "afternoon"
            if d.hour < 21:
                return "evening"
            return "tonight"

        def _until_phrase(end_local: dt.datetime) -> str:
            now_local = dt.datetime.now(tz=self._tz)
            t = _fmt_time_local(end_local)
            dp = _daypart(end_local)

            if end_local.date() == now_local.date():
                if dp == "tonight":
                    return f"until {t} tonight"
                return f"until {t} this {dp}"

            if (end_local.date() - now_local.date()).days == 1:
                if dp == "tonight":
                    return f"until {t} tomorrow night"
                return f"until {t} tomorrow {dp}"

            # fallback: weekday
            wd = end_local.strftime("%A")
            return f"until {t} on {wd}"

        def _join_oxford(items: list[str]) -> str:
            xs = [x.strip() for x in items if x and x.strip()]
            if not xs:
                return ""
            if len(xs) == 1:
                return xs[0]
            if len(xs) == 2:
                return f"{xs[0]} and {xs[1]}"
            return ", ".join(xs[:-1]) + f", and {xs[-1]}"

        STATE_NAME = {
            "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut",
            "DE":"Delaware","DC":"the District of Columbia","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois",
            "IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts",
            "MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire",
            "NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon",
            "PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont",
            "VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming",
        }

        # ---- Extract watch number + end time from VTEC ----
        watch_num: int | None = None
        end_utc: dt.datetime | None = None

        for v in self._cap_vtec_list(ev):
            m = _VTEC_PARSE_RE.search(v)
            if not m:
                continue
            phen = (m.group("phen") or "").upper()
            sig = (m.group("sig") or "").upper()
            if sig != "A":
                continue
            if kind == "tornado" and phen != "TO":
                continue
            if kind == "severe" and phen != "SV":
                continue

            try:
                watch_num = int(m.group("etn"))
            except Exception:
                watch_num = None

            end_utc = _parse_vtec_z(m.group("end") or "")
            break

        end_phrase = ""
        if end_utc is not None:
            end_local = end_utc.astimezone(self._tz)
            end_phrase = _until_phrase(end_local)

        # ---- Parse counties/states from CAP areaDesc ----
        area_desc = (getattr(ev, "area_desc", "") or "").strip()
        # CAP areaDesc often: "Cambria, PA; Cameron, PA; ..."
        groups: dict[str, list[str]] = {}
        order: list[str] = []
        misc: list[str] = []

        for raw in re.split(r";\s*", area_desc):
            s = (raw or "").strip().strip(".")
            if not s:
                continue
            if "," in s:
                name, st = s.rsplit(",", 1)
                name = name.strip()
                st = st.strip().upper()
                if st not in groups:
                    groups[st] = []
                    order.append(st)
                groups[st].append(name)
            else:
                misc.append(s)

        # ---- Boilerplate ----
        if kind == "tornado":
            watch_label = "Tornado Watch"
            remember = (
                "Remember, a tornado watch means that conditions are favorable for the development of severe weather, "
                "including tornadoes, large hail, and damaging winds, in and close to the watch area. "
                "While severe weather may not be imminent, persons should remain alert for rapidly changing weather conditions, "
                "and listen for later statements and possible warnings."
            )
        else:
            watch_label = "Severe Thunderstorm Watch"
            remember = (
                "Remember, a severe thunderstorm watch means that conditions are favorable for the development of severe weather, "
                "including large hail and damaging winds, in and close to the watch area. "
                "While severe weather may not be imminent, persons should remain alert for rapidly changing weather conditions, "
                "and listen for later statements and possible warnings."
            )

        stay_tuned = (
            "Stay tuned to NOAA Weather Radio, commercial radio, and television outlets, "
            "or internet sources for the latest severe weather information."
        )

        # ---- Build script ----
        lines: list[str] = []

        if watch_num is not None:
            lines.append(f"The National Weather Service has issued {watch_label} Number {watch_num}.")
        else:
            lines.append(f"The National Weather Service has issued {watch_label}.")

        if end_phrase:
            lines.append(f"Effective {end_phrase}.")

        if groups:
            if len(order) == 1:
                st = order[0]
                st_full = STATE_NAME.get(st, st)
                county_list = _join_oxford(groups.get(st, []))
                if county_list:
                    lines.append(f"This watch includes the following counties, in {st_full}: {county_list}.")
            else:
                segs: list[str] = []
                for st in order:
                    st_full = STATE_NAME.get(st, st)
                    county_list = _join_oxford(groups.get(st, []))
                    if county_list:
                        segs.append(f"in {st_full}: {county_list}")
                if segs:
                    lines.append("This watch includes the following counties: " + "; ".join(segs) + ".")
        elif area_desc:
            # fallback if parsing fails
            lines.append(f"This watch includes the following areas: {area_desc}.")

        # If CAP areaDesc was empty but we have leftovers
        if misc and not groups:
            lines.append("This watch includes: " + _join_oxford(misc) + ".")

        lines.append(remember)
        lines.append(stay_tuned)

        # Keep SeasonalWeather’s usual closer (optional, but consistent)
        if mode == "full":
            lines.append("End of message.")

        # Double-newlines => better pacing
        return "\n\n".join(ln.strip() for ln in lines if ln and ln.strip()).strip()


    # ------------------------------------------------------------------ #
    #  VTEC-action script builders (NWR-style update/cancel narration)    #
    # ------------------------------------------------------------------ #

    _STATE_NAME_FULL: dict[str, str] = {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
        "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
        "DC": "the District of Columbia", "FL": "Florida", "GA": "Georgia",
        "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
        "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
        "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
        "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
        "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
        "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
        "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
        "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
        "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
        "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
        "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    }

    def _parse_cap_area_by_state(self, area_desc: str) -> tuple[dict[str, list[str]], list[str], list[str]]:
        """
        Parse CAP areaDesc (semicolon-separated "County, ST" items).
        Returns (groups_by_state, state_order, misc_items).
        """
        groups: dict[str, list[str]] = {}
        order: list[str] = []
        misc: list[str] = []
        for raw in re.split(r";\s*", area_desc or ""):
            s = (raw or "").strip().strip(".")
            if not s:
                continue
            if "," in s:
                name, st = s.rsplit(",", 1)
                name = name.strip()
                st = st.strip().upper()
                if st not in groups:
                    groups[st] = []
                    order.append(st)
                groups[st].append(name)
            else:
                misc.append(s)
        return groups, order, misc

    def _join_oxford(self, items: list[str]) -> str:
        xs = [x.strip() for x in items if x and x.strip()]
        if not xs:
            return ""
        if len(xs) == 1:
            return xs[0]
        if len(xs) == 2:
            return f"{xs[0]} and {xs[1]}"
        return ", ".join(xs[:-1]) + f", and {xs[-1]}"

    def _fmt_local_from_utc_iso(self, iso_str: str) -> str:
        """
        Parse an ISO-8601 UTC string and return a human-friendly local time phrase.
        Returns "" on failure.
        """
        s = (iso_str or "").strip()
        if not s:
            return ""
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            utc_dt = dt.datetime.fromisoformat(s)
            local_dt = utc_dt.astimezone(self._tz)
            hour12 = local_dt.hour % 12 or 12
            ampm = "AM" if local_dt.hour < 12 else "PM"
            tz_name = _expand_tz_token(local_dt.strftime("%Z"))
            if local_dt.minute == 0:
                return f"{hour12} {ampm} {tz_name}"
            return f"{hour12}:{local_dt.minute:02d} {ampm} {tz_name}"
        except Exception:
            return ""

    def _alert_tracker_id_for_cap(self, ev: "CapAlertEvent", same_code: str) -> str:  # type: ignore[name-defined]
        """
        Return a stable AlertTracker ID for a CAP event.
        Prefers the first VTEC track id so updates slot into the same entry.
        """
        vtec = self._cap_vtec_list(ev)
        for v in vtec:
            tid = _vtec_track_id(v)
            if tid:
                return f"CAP:{tid}"
        return f"CAP:{(ev.alert_id or '').strip()}"

    def _alert_expires_from_cap(self, ev: "CapAlertEvent", vtec: list[str]) -> str:  # type: ignore[name-defined]
        """Best-effort expiry ISO string from VTEC end time or CAP expires field."""
        exp_utc = self._best_expiry_from_vtec(vtec)
        if exp_utc:
            return exp_utc.isoformat()
        raw = getattr(ev, "expires", None) or getattr(ev, "ends", None)
        if raw:
            return str(raw).strip()
        # Fallback: 6 hours from now
        return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=6)).isoformat()

    def _cap_prefers_statement_update_script(self, event: str, vtec_actions: set[str]) -> bool:
        e = (event or '').strip().lower()
        if not e:
            return False
        if not (vtec_actions & {'CAN', 'EXP'}):
            return False
        return e.endswith('advisory') or e.endswith('statement') or e.endswith('message')

    def _cap_expiry_summary_line(self, text: str) -> str:
        src = str(text or '').strip()
        if not src:
            return ''
        flat = re.sub(r'\s+', ' ', src)
        m = re.search(
            r'([^.]{0,220}\b(?:will expire|has expired|has been allowed to expire|has ended|is no longer in effect|the threat has ended)\b[^.]{0,220}[.?!]?)',
            flat,
            flags=re.IGNORECASE,
        )
        if not m:
            return ''
        line = m.group(1).strip()
        if line and not line.endswith(('.', '!', '?')):
            line += '.'
        return line

    def _build_statement_vtec_action_script(
        self,
        ev: "CapAlertEvent",  # type: ignore[name-defined]
        vtec_actions: set[str],
        tracks: list[tuple[str, str]],
    ) -> str:
        """
        Lighter-weight voice cut-in for advisory / statement / message EXP/CAN updates.
        This intentionally sounds like a short NWR-style statement instead of a full
        warning, or watch style update.
        """
        event = self._clean_cap_text(ev.event or '', limit=120)
        area_desc = (getattr(ev, 'area_desc', '') or '').strip()
        desc = str(getattr(ev, 'description', '') or '').strip()
        headline = str(getattr(ev, 'headline', '') or '').strip()

        groups, order, misc = self._parse_cap_area_by_state(area_desc)

        def _county_segs() -> str:
            if not groups:
                return self._clean_cap_text(area_desc or 'the affected areas', limit=400)
            parts: list[str] = []
            for st in order:
                st_full = self._STATE_NAME_FULL.get(st, st)
                county_list = self._join_oxford(groups[st])
                if county_list:
                    parts.append(f'in {st_full}, {county_list}')
            if misc:
                parts.append(self._join_oxford(misc))
            return '; '.join(parts) if parts else self._clean_cap_text(area_desc or 'the affected areas', limit=400)

        summary_line = ''
        if vtec_actions & {'EXP'}:
            summary_line = self._cap_expiry_summary_line(desc) or self._cap_expiry_summary_line(headline)
            if not summary_line and event:
                summary_line = f'The {event} has expired.'
        elif vtec_actions & {'CAN'}:
            summary_line = self._cap_expiry_summary_line(desc) or self._cap_expiry_summary_line(headline)
            if not summary_line and event:
                summary_line = f'The {event} has been cancelled.'

        lines: list[str] = []
        if (event or '').strip().lower() == 'special weather statement':
            lines.append(self._cap_sps_preamble(getattr(ev, 'sent', None)))
        else:
            lines.append('This is a statement from the National Weather Service.')
        area_line = _county_segs()
        if area_line:
            lines.append(f'For the following counties: {area_line}.')
        if summary_line:
            lines.append(summary_line)
        elif event:
            lines.append(f'The {event} has been updated.')
        lines.append('End of message.')
        return "\n".join(ln.strip() for ln in lines if ln and ln.strip()).strip()

    def _build_warning_vtec_action_script(
        self,
        ev: "CapAlertEvent",  # type: ignore[name-defined]
        vtec_actions: set[str],
        tracks: list[tuple[str, str]],
    ) -> str:
        """
        NWR-style voice script for VTEC update actions on warnings (non-watch).

        CON/EXT  →  "…remains in effect until …"
        CAN      →  "…has been cancelled"
        EXP      →  "…has been allowed to expire"
        EXA/EXB  →  "…has been expanded to include …"
        UPG      →  handled by _build_cap_full_script (new warning issued)
        """
        event = self._clean_cap_text(ev.event or "", limit=120)
        area_desc = self._clean_cap_text(getattr(ev, "area_desc", "") or "", limit=400)
        headline = self._clean_cap_text(ev.headline or "", limit=280)
        desc = self._clean_cap_text(getattr(ev, "description", "") or "", limit=800)
        instr = self._clean_cap_text(getattr(ev, "instruction", "") or "", limit=400)

        vtec = self._cap_vtec_list(ev)
        exp_utc = self._best_expiry_from_vtec(vtec)
        exp_phrase = ""
        if exp_utc:
            exp_phrase = self._fmt_local_from_utc_iso(exp_utc.isoformat())
        if not exp_phrase:
            raw_exp = getattr(ev, "expires", None)
            if raw_exp:
                exp_phrase = self._fmt_local_from_utc_iso(str(raw_exp))

        lines: list[str] = []

        if vtec_actions & {"CAN"}:
            lines.append(f"The {event} for the following areas has been cancelled.")
            if area_desc:
                lines.append(f"Areas: {area_desc}.")
            if headline:
                lines.append(headline if headline.endswith((".", "!", "?")) else headline + ".")

        elif vtec_actions & {"EXP"}:
            lines.append(f"The {event} for the following areas has been allowed to expire.")
            if area_desc:
                lines.append(f"Areas: {area_desc}.")

        elif vtec_actions & {"EXA", "EXB"}:
            lines.append(f"The {event} has been expanded.")
            if area_desc:
                lines.append(f"This now includes: {area_desc}.")
            if exp_phrase:
                lines.append(f"This warning remains in effect until {exp_phrase}.")
            if desc:
                lines.append(desc)
            if instr:
                lines.append(instr)

        elif vtec_actions & {"EXT"}:
            lines.append(f"The {event} has been extended.")
            if area_desc:
                lines.append(f"For the following areas: {area_desc}.")
            if exp_phrase:
                lines.append(f"This warning is now in effect until {exp_phrase}.")
            if desc:
                lines.append(desc)
            if instr:
                lines.append(instr)

        else:  # CON (continuation) and anything else
            if headline:
                lines.append(headline if headline.endswith((".", "!", "?")) else headline + ".")
            elif event:
                lead = f"A {event} remains in effect"
                if exp_phrase:
                    lead += f" until {exp_phrase}"
                lead += "."
                lines.append(lead)
            if area_desc:
                lines.append(f"For the following areas: {area_desc}.")
            if desc:
                lines.append(desc)
            if instr:
                lines.append(instr)

        if not lines:
            return self._build_cap_full_script(ev)

        lines.append("End of message.")
        return "\n".join(ln.strip() for ln in lines if ln and ln.strip()).strip()

    def _build_watch_vtec_action_script(
        self,
        ev: "CapAlertEvent",  # type: ignore[name-defined]
        vtec_actions: set[str],
        tracks: list[tuple[str, str]],
        watch_number: int | None,
        kind: str,  # "tornado" or "severe"
    ) -> str:
        """
        NWR-style voice script for VTEC update/cancel actions on watches (TOA/SVA).

        CON      → "Watch Number N remains in effect until …"
        EXA      → "Watch Number N remains in effect until … and now includes …"
        CAN      → "Watch Number N has been cancelled for … in …"
        EXP      → "Watch Number N has been allowed to expire for … in …"
        """
        watch_label = "Tornado Watch" if kind == "tornado" else "Severe Thunderstorm Watch"
        num_phrase = f"Number {watch_number}" if watch_number is not None else ""
        label_with_num = f"{watch_label} {num_phrase}".strip()

        area_desc = (getattr(ev, "area_desc", "") or "").strip()
        groups, order, misc = self._parse_cap_area_by_state(area_desc)

        vtec = self._cap_vtec_list(ev)
        exp_utc = self._best_expiry_from_vtec(vtec)
        exp_phrase = ""
        if exp_utc:
            exp_phrase = self._fmt_local_from_utc_iso(exp_utc.isoformat())
        if not exp_phrase:
            raw_exp = getattr(ev, "expires", None)
            if raw_exp:
                exp_phrase = self._fmt_local_from_utc_iso(str(raw_exp))

        def _county_segs() -> str:
            """Build 'in Maryland: Allegany, Garrett' style phrase."""
            if not groups:
                return area_desc or "the affected areas"
            parts: list[str] = []
            for st in order:
                st_full = self._STATE_NAME_FULL.get(st, st)
                county_list = self._join_oxford(groups[st])
                if county_list:
                    parts.append(f"in {st_full}: {county_list}")
            if parts:
                return "; ".join(parts)
            return area_desc or "the affected areas"

        lines: list[str] = []

        if vtec_actions & {"CAN"}:
            lines.append(f"{label_with_num} has been cancelled for the following areas.")
            lines.append(_county_segs() + ".")

        elif vtec_actions & {"EXP"}:
            lines.append(f"{label_with_num} has been allowed to expire for the following areas.")
            lines.append(_county_segs() + ".")

        elif vtec_actions & {"EXA", "EXB"}:
            # Watch expansion — also used when area grows mid-event
            lines.append(f"{label_with_num} remains in effect" + (f" until {exp_phrase}" if exp_phrase else "") + ".")
            lines.append("This watch now includes the following additional areas.")
            lines.append(_county_segs() + ".")

        else:  # CON / EXT
            lines.append(f"{label_with_num} remains in effect" + (f" until {exp_phrase}" if exp_phrase else "") + ".")
            lines.append(f"This watch includes the following areas: {_county_segs()}.")

        if not lines:
            return self._build_cap_watch_script(ev, mode="full")

        lines.append("Stay tuned to NOAA Weather Radio, commercial radio, and television outlets for the latest severe weather information.")
        lines.append("End of message.")
        return "\n".join(ln.strip() for ln in lines if ln and ln.strip()).strip()

    def _build_watch_expansion_script(self, ev: "CapAlertEvent") -> str:  # type: ignore[name-defined]
        """
        Full NWR-style script for watch EXA/EXB: new SAME tones, full county listing.
        Expansion is treated as a new issuance for the added counties.
        """
        # Determine kind + watch number from VTEC
        kind = "tornado"
        watch_number: int | None = None
        for v in self._cap_vtec_list(ev):
            m = _VTEC_PARSE_RE.search(v)
            if not m:
                continue
            phen = (m.group("phen") or "").upper()
            sig = (m.group("sig") or "").upper()
            if sig != "A":
                continue
            if phen == "TO":
                kind = "tornado"
            elif phen == "SV":
                kind = "severe"
            else:
                continue
            try:
                watch_number = int(m.group("etn"))
            except Exception:
                pass
            break

        tracks = self._vtec_tracks(self._cap_vtec_list(ev))
        return self._build_watch_vtec_action_script(
            ev,
            vtec_actions={"EXA"},
            tracks=tracks,
            watch_number=watch_number,
            kind=kind,
        )

    def _is_safety_rules_pns(self, text: str) -> bool:
        """Return True if this is an NWR-style SEVERE WEATHER SAFETY RULES PNS."""
        t = (text or "").upper()
        return "...SEVERE WEATHER SAFETY RULES..." in t

    def _build_pns_safety_script(self, official_text: str) -> str:
        """
        Extract clean broadcast text from a SEVERE WEATHER SAFETY RULES PNS.
        Uses the existing alert_builder strip-and-parse pipeline.
        """
        from .alert_builder import strip_nws_product_headers, _unwrap_soft_wrap, _collapse_blank_lines, _clean_line
        from .tts import clean_for_tts
        import re as _re

        text = strip_nws_product_headers(official_text or "")
        lines_raw = [ln.rstrip() for ln in text.splitlines()]
        lines = _unwrap_soft_wrap(lines_raw)

        body: list[str] = []
        in_body = False
        for ln in lines:
            s = (ln or "").strip()
            if not in_body:
                # Start reading at the headline marker or "National Weather Service" line
                if s.startswith("...") or "national weather service" in s.lower():
                    in_body = True
                else:
                    continue
            if s.startswith(("&&", "$$")):
                break
            if not s:
                body.append("")
                continue
            cleaned = _clean_line(s)
            if cleaned:
                body.append(cleaned)

        body = _collapse_blank_lines(body)
        script_raw = "\n".join(body)

        intro = "The National Weather Service has issued the following public information statement."
        script = clean_for_tts(script_raw)
        if not script.strip():
            return ""
        return intro + "\n\n" + script

    def _build_cap_full_script(self, ev: "CapAlertEvent") -> str:  # type: ignore[name-defined]
        event = self._clean_cap_text(ev.event or "", limit=120)
        headline = self._clean_cap_text(ev.headline or "", limit=280)
        area = self._clean_cap_text(getattr(ev, "area_desc", "") or "", limit=320)
        desc = self._clean_cap_text(getattr(ev, "description", "") or "", limit=1200)
        instr = self._clean_cap_text(getattr(ev, "instruction", "") or "", limit=700)

        if event.lower() == "special weather statement" and headline.lower().startswith("special weather statement"):
            headline = ""

        lines: list[str] = []
        if event:
            if event.lower() == "special weather statement":
                lines.append(self._cap_sps_preamble(getattr(ev, "sent", None)))
            else:
                lines.append(f"{event}.")

        if headline:
            lines.append(headline if headline.endswith((".", "!", "?")) else headline + ".")

        if desc:
            lines.append(desc)

        if instr:
            lines.append("Instructions.")
            lines.append(instr)

        lines.append("End of message.")
        return "\n".join(ln.strip() for ln in lines if ln and ln.strip()).strip()

    def _build_cap_voice_script(self, ev: "CapAlertEvent") -> str:  # type: ignore[name-defined]
        event = self._clean_cap_text(ev.event or "", limit=120)
        headline = self._clean_cap_text(ev.headline or "", limit=240)
        area = self._clean_cap_text(getattr(ev, "area_desc", "") or "", limit=260)
        desc = self._clean_cap_text(getattr(ev, "description", "") or "", limit=900)
        instr = self._clean_cap_text(getattr(ev, "instruction", "") or "", limit=500)

        if event.lower() == "special weather statement" and headline.lower().startswith("special weather statement"):
            headline = ""

        lines: list[str] = []
        if event:
            if event.lower() == "special weather statement":
                lines.append(self._cap_sps_preamble(getattr(ev, "sent", None)))
            else:
                lines.append(f"{event}.")
        if headline:
            lines.append(headline if headline.endswith((".", "!", "?")) else headline + ".")
        if desc:
            lines.append(desc)
        if instr:
            lines.append("Instructions.")
            lines.append(instr)

        return "\n".join(ln.strip() for ln in lines if ln and ln.strip()).strip()

    async def _render_voice_only_audio(self, script_text: str, *, prefix: str = "capvoice") -> Path:
        _, audio_dir, _, _ = self._paths()
        ts = dt.datetime.now(tz=self._tz).strftime("%Y%m%d-%H%M%S")
        safe_prefix = "".join(ch for ch in prefix if ch.isalnum() or ch in {"_", "-"}).strip() or "voice"

        tts_wav = audio_dir / f"{safe_prefix}_{ts}_tts.wav"
        pre = audio_dir / f"{safe_prefix}_{ts}_pre.wav"
        post = audio_dir / f"{safe_prefix}_{ts}_post.wav"
        out = audio_dir / f"{safe_prefix}_{ts}.wav"

        write_silence_wav(pre, 0.35, self.cfg.audio.sample_rate)
        self.tts.synth_to_wav(script_text, tts_wav)
        write_silence_wav(post, 1.2, self.cfg.audio.sample_rate)
        concat_wavs(out, [pre, tts_wav, post])
        return out

    def _build_ern_relay_script(self, ev: "ErnSameEvent") -> str:  # type: ignore[name-defined]
        code = (ev.event or "").strip().upper()
        sender = (ev.sender or "").strip()

        if code == "RWT":
            main = "This is a relay of a Required Weekly Test received via the Emergency Relay Network. This is only a test."
        elif code == "RMT":
            main = "This is a relay of a Required Monthly Test received via the Emergency Relay Network. This is only a test."
        else:
            main = "This is a relay message received via the Emergency Relay Network."

        lines = [main]
        if sender:
            lines.append(f"Sender: {sender}.")
        lines.append("End of message.")
        return "\n".join(lines)

    async def _consume_ern(self) -> None:
        while True:
            ev = await self.ern_queue.get()

            if ev.kind == "header":
                log.info(
                    "ERN SAME header: org=%s event=%s sender=%s conf=%.3f same=%s text=%s",
                    ev.org,
                    ev.event,
                    (ev.sender or "").strip(),
                    ev.confidence,
                    ",".join(ev.locations[:12]) + ("..." if len(ev.locations) > 12 else ""),
                    ev.text,
                )

                try:
                    if ev.event and ev.event in self.cfg.policy.toneout_product_types and ev.event not in {"RWT", "RMT"}:
                        self.ern_last_tone_at = dt.datetime.now(tz=self._tz)
                except Exception:
                    pass
            else:
                log.info("ERN SAME EOM: conf=%.3f text=%s", ev.confidence, ev.text)

            if self._ern_dryrun():
                continue
            if not self._ern_relay_enabled():
                continue
            if ev.kind != "header":
                continue

            code = (ev.event or "").strip().upper()
            if code not in self._ern_relay_events():
                continue

            conf = float(getattr(ev, "confidence", 0.0) or 0.0)
            if conf < self._ern_relay_min_confidence():
                log.info(
                    "ERN relay: confidence too low (%.3f < %.3f) event=%s sender=%s",
                    conf,
                    self._ern_relay_min_confidence(),
                    code,
                    ev.sender,
                )
                continue

            senders = self._ern_relay_senders()
            sender_u = (ev.sender or "").strip().upper()
            if senders and sender_u not in senders:
                log.info("ERN relay: sender not allowed (sender=%s allowed=%s)", ev.sender, ",".join(sorted(senders)))
                continue

            now = dt.datetime.now(tz=self._tz)
            if self._ern_relay_last_any_at and (now - self._ern_relay_last_any_at).total_seconds() < self._ern_relay_cooldown_seconds():
                log.info("ERN relay: cooldown active; skipping event=%s sender=%s", code, ev.sender)
                continue

            in_area_locs = self._filter_same_locations_to_service_area(getattr(ev, "locations", None))
            if not in_area_locs:
                log.info(
                    "ERN relay: no in-area SAME locations after filtering; skipping event=%s sender=%s decoded=%s",
                    code,
                    ev.sender,
                    ",".join(getattr(ev, "locations", [])[:12]) + ("..." if len(getattr(ev, "locations", [])) > 12 else ""),
                )
                continue

            # Cross-source dedupe: reserve keys BEFORE rendering/airing

            keys: list[str] = []

            fkey3 = self._dedupe_func_full_key(code, in_area_locs)

            if fkey3:

                keys.append(fkey3)


            # ERN-specific fallback (suppresses identical repeats even if functional key is absent)

            sender_u2 = (ev.sender or "").strip().upper()

            loc_sig = ",".join(sorted(set(in_area_locs)))[:1200]

            keys.append(f"ERNRELAY:{code}:{self._sha1_12(sender_u2 + '|' + loc_sig)}")


            ok, hit = await self._dedupe_reserve(keys)

            if not ok:

                log.info("ERN relay skipped (dedupe hit=%s) event=%s sender=%s same=%s", hit, code, ev.sender, loc_sig[:160])

                continue


            script = self._build_ern_relay_script(ev)
            dummy = SimpleNamespace(product_type=code, awips_id=None, wfo="ERN", raw_text="")

            out_wav = await self._render_alert_audio(dummy, script, same_locations=in_area_locs)

            async with self._cycle_lock:
                try:
                    self.telnet.flush_cycle()
                except Exception:
                    pass
                event_label = _same_label_or_code(code)
                title = self._np_alert_title("ern", event=event_label)
                meta = self._np_meta(
                    title=title,
                    kind="alert",
                    extra={
                        "sw_alert_source": "ern",
                        "sw_event_code": code,
                        "sw_event": event_label,
                        "sw_sender": (ev.sender or "").strip(),
                    },
                )
                self.telnet.push_alert(str(out_wav), meta=meta)

            self._ern_relay_last_any_at = now
            self.last_product_desc = f"ERN {code}".strip()

            self._schedule_cycle_refill("post-ern-relay")
            log.info(
                "ERN ACTION: aired relay event=%s sender=%s same_locs=%d audio=%s",
                code,
                ev.sender,
                len(in_area_locs),
                out_wav,
            )
            # _ERN_DL_
            self.discord.alert_aired(
                code=code,
                event=_same_label_or_code(code),
                source=f"ERN/GWES ({(ev.sender or '').strip() or 'unknown'})",
                mode="full",
                is_ern=True,
            )
            sf_ev = ev
            try:
                area_text = await self._sf_area_text_from_same_codes(in_area_locs)
                if area_text:
                    try:
                        setattr(sf_ev, "area", area_text)
                    except Exception:
                        try:
                            sf_ev = SimpleNamespace(**getattr(ev, "__dict__", {}))
                            setattr(sf_ev, "area", area_text)
                        except Exception:
                            sf_ev = ev
            except Exception:
                pass
            _station_feed_note_ern(sf_ev, same_locations=in_area_locs, out_wav=str(out_wav))

    async def _handle_toneout(self, parsed: ParsedProduct) -> None:
        log.info("NWWS toneout candidate: type=%s awips=%s wfo=%s", parsed.product_type, parsed.awips_id or "", parsed.wfo)

        official_text, pid = await self._resolve_nwws_official_text(parsed)

        # --- NEW: derive SAME targeting from UGC zones (NWWS-only) ---
        zones, in_area_same, src, mapped_ok = await self._nwws_same_targets_from_texts(parsed.raw_text or "", official_text or "")

        if zones:
            log.info(
                "NWWS targeting: ugc_zones=%d src=%s mapped_ok=%s in_area_same=%d",
                len(zones),
                src,
                mapped_ok,
                len(in_area_same),
            )

        # If we successfully mapped zones -> SOME SAME codes, and none are in-area => out-of-area, skip entirely.
        if zones and mapped_ok and not in_area_same:
            preview = ",".join(zones[:20]) + ("..." if len(zones) > 20 else "")
            log.info(
                "NWWS out-of-area: type=%s wfo=%s ugc_zones=%s (no intersection with service area); skipping",
                parsed.product_type,
                parsed.wfo,
                preview,
            )
            return

        vtec = self._extract_vtec(official_text)
        tracks = self._vtec_tracks(vtec)
        exp_utc = self._best_expiry_from_vtec(vtec)

        # VTEC toneout policy — vtec.py is authoritative for FULL vs VOICE.
        # This replaces the inline action-only check that ignored significance
        # (e.g. CF.Y.NEW was incorrectly treated as FULL before this fix).
        _nw_policy = _vtec_toneout_policy(vtec)
        vtec_actions = {act for (_t, act) in tracks} if tracks else set()
        should_full = (_nw_policy.mode == "FULL")
        log.debug("NWWS vtec policy: %s", _nw_policy.reason)

        # Keep rebroadcast rotation in sync with VTEC lifecycle.
        if self.rebroadcast_enabled and tracks:
            now_local = dt.datetime.now(self.local_tz)
            expires_at = self._rebroadcast_clamp_expiry(now_local, exp_utc)

            if vtec_actions & {"CAN", "EXP"}:
                await self._rebroadcast_remove_by_tracks(
                    tracks=tracks,
                    reason=f"nwws:{parsed.product_type}:{','.join(sorted(vtec_actions & {'CAN','EXP'}))}",
                )
            else:
                await self._rebroadcast_touch_expiry_by_tracks(
                    tracks=tracks,
                    expires_at=expires_at,
                    reason=f"nwws:{parsed.product_type}:{','.join(sorted(vtec_actions))}",
                )



        # Critical safety gate:
        # If we have UGC zones but could not map ANY of them to SAME, we do NOT air FULL.
        # This prevents blind tone-outs (especially marine zones) when mapping fails.
        if zones and (not mapped_ok) and should_full:
            log.warning(
                "NWWS SAME targeting failed (no zone->SAME mapping). Forcing voice-only type=%s wfo=%s zones=%s",
                parsed.product_type,
                parsed.wfo,
                ",".join(zones[:12]) + ("..." if len(zones) > 12 else ""),
            )
            should_full = False
        keys: list[str] = []

        # Track-level dedupe prevents CAP+NWWS double-air.
        for track_id, _act in tracks:
            keys.append(f"{'TRACKFULL' if should_full else 'TRACKVOICE'}:{track_id}")

        # Keep VTEC strings too (helps when track parse fails on weird edge cases)
        for v in vtec:
            keys.append(f"VTEC:{v}")

        # Functional FULL dedupe is only safe when we do NOT have a concrete
        # VTEC track.  Otherwise, distinct warnings for the same SAME footprint
        # can suppress each other.
        if should_full and not tracks:
            fkey2 = self._dedupe_func_full_key(parsed.product_type, in_area_same)
            if fkey2:
                keys.append(fkey2)

        # Message-level fallback key
        keys.append(f"NWWS:{parsed.product_type}:{parsed.wfo}:{self._sha1_12(official_text[:1200])}:{'FULL' if should_full else 'VOICE'}")

        ok, hit = await self._dedupe_reserve(keys)
        if not ok:
            log.info(
                "NWWS %s skipped (dedupe hit=%s) type=%s awips=%s wfo=%s vtec=%s",
                "FULL" if should_full else "VOICE",
                hit,
                parsed.product_type,
                parsed.awips_id or "",
                parsed.wfo,
                ",".join(vtec[:2]) if vtec else "",
            )
            return

        try:
            spoken = build_spoken_alert(parsed, official_text)

            sf_issued_dt = _sf_nwws_best_issued_dt(parsed, official_text)
            sf_event_label = _sf_nwws_event_label(parsed.product_type, vtec_list=vtec, text=official_text)
            sf_area_text = ""
            if in_area_same:
                try:
                    sf_area_text = await self._sf_area_text_from_same_codes(list(in_area_same))
                except Exception:
                    sf_area_text = ""
            if not sf_area_text:
                sf_area_text = _sf_nwws_area_from_text(official_text)
            sf_headline = _sf_nwws_make_headline(
                sf_event_label,
                issued_dt=sf_issued_dt,
                end_dt=exp_utc,
                issuer=_sf_nwws_extract_issuer(official_text, fallback_wfo=parsed.wfo),
            )

            # --- Less-urgent polish / todos ---
            # SPS preamble: NWR-ish intro, avoid duplicated boilerplate.
            try:
                if (parsed.product_type or '').strip().upper() == 'SPS':
                    old0 = (spoken.script or '').strip()
                    spoken.script = self._fix_sps_preamble(spoken.script, official_text)
                    if (spoken.script or '').strip() != old0:
                        log.info('NWWS SPS preamble normalized (awips=%s wfo=%s)', parsed.awips_id or '', parsed.wfo)
            except Exception:
                log.exception('NWWS SPS preamble normalization failed; continuing with original script')

            # EXP/CAN short narration: if VTEC says expired/cancel and this is voice-only,
            # keep NWWS aligned with the CAP advisory/statement/message helper when the
            # event class is light enough for statement-style narration.
            try:
                if tracks and not should_full:
                    if ('EXP' in vtec_actions) or ('CAN' in vtec_actions):
                        if self._cap_prefers_statement_update_script(sf_event_label, vtec_actions):
                            spoken.script = self._build_nwws_statement_vtec_action_script(
                                event_text=sf_event_label,
                                area_text=sf_area_text,
                                official_text=official_text,
                                headline=sf_headline,
                                vtec_actions=vtec_actions,
                            )
                            log.info(
                                'NWWS statement-style EXP/CAN enabled (act=%s type=%s awips=%s wfo=%s)',
                                ','.join(sorted(vtec_actions))[:64],
                                parsed.product_type,
                                parsed.awips_id or '',
                                parsed.wfo,
                            )
                        else:
                            summ = self._expiry_summary_script(official_text)
                            if summ:
                                spoken.script = summ
                                log.info(
                                    'NWWS EXP/CAN summary enabled (act=%s type=%s awips=%s wfo=%s)',
                                    ','.join(sorted(vtec_actions))[:64],
                                    parsed.product_type,
                                    parsed.awips_id or '',
                                    parsed.wfo,
                                )
            except Exception:
                log.exception('NWWS EXP/CAN summary failed; continuing with original script')

            if should_full:
                # If we have in-area SAME targets, use them. Otherwise, AIR WITHOUT SAME (no 67 fallback).
                same_for_render: list[str] = list(in_area_same) if in_area_same else []
                if zones and not mapped_ok:
                    log.warning(
                        "NWWS SAME targeting unavailable (zone map failed); airing without SAME headers type=%s wfo=%s",
                        parsed.product_type,
                        parsed.wfo,
                    )
                out_wav = await self._render_alert_audio(parsed, spoken.script, same_locations=same_for_render)
            else:
                # Voice-only always has no SAME headers by design.
                out_wav = await self._render_voice_only_audio(spoken.script, prefix="nwwsvoice")

            async with self._cycle_lock:
                try:
                    self.telnet.flush_cycle()
                except Exception:
                    pass
                event_label = sf_event_label
                if (not should_full) and (("EXP" in vtec_actions) or ("CAN" in vtec_actions)):
                    tkey = "nwws_end"
                elif should_full:
                    tkey = "nwws_full"
                else:
                    tkey = "nwws_update"
                title = self._np_alert_title(tkey, event=event_label)
                meta = self._np_meta(
                    title=title,
                    kind="alert",
                    extra={
                        "sw_alert_source": "nwws",
                        "sw_alert_mode": ("full" if should_full else "voice"),
                        "sw_event_code": (_nw_policy.same_code or parsed.product_type or "").strip().upper(),
                        "sw_event": event_label,
                        "sw_wfo": (parsed.wfo or "").strip(),
                        "sw_awips": (parsed.awips_id or "").strip(),
                    },
                )
                self.telnet.push_alert(str(out_wav), meta=meta)

            now = dt.datetime.now(tz=self._tz)

            # Only FULL toneouts should push heightened mode + toneout timestamp.
            if should_full:
                self.last_toneout_at = now
                self.last_heightened_at = now
                self.heightened_until = now + dt.timedelta(seconds=self.cfg.cycle.min_heightened_seconds)
                self._update_mode()

            self._nwws_acted += 1
            log.info(
                "NWWS ACTION: aired %s #%d/%d type=%s awips=%s wfo=%s vtec=%s tracks=%s audio=%s",
                "FULL" if should_full else "VOICE",
                self._nwws_acted,
                self._nwws_seen,
                parsed.product_type,
                parsed.awips_id or "",
                parsed.wfo,
                ",".join(vtec[:2]) if vtec else "",
                ",".join(t for (t, _a) in tracks[:2]) if tracks else "",
                out_wav,
            )
            # _NWWS_DL_
            _dl_vtec_acts = vtec_actions
            _dl_mode = "full" if should_full else "voice"
            if _dl_vtec_acts & {"CAN", "EXP"}:
                self.discord.alert_expired(
                    code=parsed.product_type,
                    event=sf_event_label,
                    vtec_action=next(iter(_dl_vtec_acts & {"CAN", "EXP"})),
                    source="NWWS-OI",
                    area=sf_area_text,
                    vtec=vtec[:2],
                )
            elif not should_full and _dl_vtec_acts & {"CON", "EXT", "EXA", "EXB"}:
                self.discord.alert_updated(
                    code=parsed.product_type,
                    event=sf_event_label,
                    vtec_action=next(iter(_dl_vtec_acts & {"CON", "EXT", "EXA", "EXB"})),
                    source="NWWS-OI",
                    area=sf_area_text,
                    vtec=vtec[:2],
                )
            else:
                self.discord.alert_aired(
                    code=parsed.product_type,
                    event=sf_event_label,
                    source="NWWS-OI",
                    mode=_dl_mode,
                    area=sf_area_text,
                    vtec=vtec[:2],
                    is_test=(parsed.product_type in {"RWT", "RMT"}),
                )
            _sf_mode = getattr(spoken, "mode", ("FULL" if should_full else "VOICE"))
            _station_feed_note_nwws(
                parsed,
                mode=_sf_mode,
                same_locations=list(in_area_same or []),
                out_wav=str(out_wav),
                product_id=pid,
                expires_at=exp_utc,
                vtec=vtec,
                official_text=official_text,
                issued_at=sf_issued_dt,
                event_text=sf_event_label,
                headline=sf_headline,
                area_text=sf_area_text,
            )

            self._schedule_cycle_refill("post-alert")

            # Register / update / remove from AlertTracker
            try:
                _nw_vtec = vtec
                _nw_tracks = tracks
                _nw_vtec_actions = vtec_actions
                _nw_exp_utc = self._best_expiry_from_vtec(_nw_vtec)
                _nw_expires_iso = _nw_exp_utc.isoformat() if _nw_exp_utc else (
                    dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=6)).isoformat()
                _nw_same_code = _nw_policy.same_code or _safe_event_code(parsed.product_type)
                _nw_same_locs = list(in_area_same) if in_area_same else []
                _nw_track_ids = {_vtec_track_id(v) for v in _nw_vtec if _vtec_track_id(v)}
                _nw_event_label = sf_event_label
                _nw_headline = sf_headline
                if _nw_vtec_actions & {"CAN", "EXP"} and not should_full:
                    # Cancellation/expiry voice-only: remove from tracker
                    removed_n = self.alert_tracker.remove_by_vtec_tracks(
                        _nw_track_ids,  # type: ignore[arg-type]
                        reason=f"nwws:{parsed.product_type}:{','.join(sorted(_nw_vtec_actions & {'CAN','EXP'}))}",
                    )
                    log.info("AlertTracker: removed %d entries for NWWS CAN/EXP type=%s awips=%s",
                             removed_n, parsed.product_type, parsed.awips_id or "")
                else:
                    # New issuance, update, or continuation
                    _nw_tid_str = next(iter(_nw_track_ids), None)
                    _nw_tracker_id = f"NWWS:{_nw_tid_str}" if _nw_tid_str else (
                        f"NWWS:{parsed.product_type}:{parsed.wfo}:{(parsed.awips_id or '').strip()}")
                    _nw_is_cycle_only = not should_full
                    _nw_issued = sf_issued_dt or dt.datetime.now(dt.timezone.utc)
                    if _nw_issued.tzinfo is None:
                        _nw_issued = _nw_issued.replace(tzinfo=dt.timezone.utc)
                    _ae_nw = ActiveAlert(
                        id=_nw_tracker_id,
                        source="NWWS",
                        event=_nw_event_label,
                        code=_nw_same_code,
                        vtec=_nw_vtec,
                        headline=_nw_headline,
                        script_text=spoken.script,
                        audio_path=str(out_wav),
                        expires=_nw_expires_iso,
                        issued=_nw_issued.isoformat(),
                        same_locs=_nw_same_locs,
                        cycle_only=_nw_is_cycle_only,
                    )
                    self.alert_tracker.add_or_update(_ae_nw)
                    self.alert_tracker.mark_aired(_nw_tracker_id)
                    log.info("AlertTracker: registered NWWS id=%s type=%s should_full=%s expires=%s",
                             _nw_tracker_id, parsed.product_type, should_full, _nw_expires_iso)
            except Exception:
                log.exception("AlertTracker: failed to register NWWS type=%s", parsed.product_type)

            # Rebroadcast rotation (no re-tone)
            try:
                if self.rebroadcast_enabled and (should_full or self.rebroadcast_include_voice):
                    exp_utc = self._best_expiry_from_vtec(vtec)
                    expires_at = self._rebroadcast_clamp_expiry(now, exp_utc)
                    kind_rb = "FULL" if should_full else "VOICE"
                    key_rb = self._rebroadcast_make_key(
                        source="NWWS",
                        kind=kind_rb,
                        event_code=_nw_policy.same_code or _safe_event_code(parsed.product_type),
                        tracks=tracks,
                        same_locs=in_area_same,
                        script=spoken.script,
                    )
                    desc_rb = f"NWWS {kind_rb} {parsed.product_type} {parsed.wfo} {parsed.awips_id or ''}".strip()
                    await self._rebroadcast_add(key=key_rb, desc=desc_rb, script=spoken.script, expires_at=expires_at)
            except Exception:
                log.exception("Rebroadcast: failed to add NWWS item")
        except Exception:
            await self._dedupe_release(keys)
            raise

    async def _render_alert_audio(
        self,
        parsed: ParsedProduct,
        script_text: str,
        *,
        same_locations: list[str] | None = None,
    ) -> Path:
        _, audio_dir, _, _ = self._paths()
        ts = dt.datetime.now(tz=self._tz).strftime("%Y%m%d-%H%M%S")

        tone = audio_dir / f"alert_{ts}_tone.wav"
        tts_wav = audio_dir / f"alert_{ts}_tts.wav"
        gap = audio_dir / f"alert_{ts}_gap.wav"
        eom = audio_dir / f"alert_{ts}_eom.wav"
        post = audio_dir / f"alert_{ts}_post.wav"
        out = audio_dir / f"alert_{ts}.wav"

        same_hdr_all: Path | None = None
        same_eom_wav: Path | None = None

        if self._same_enabled() and SameHeader is not None:
            try:
                # NEW semantics:
                # - same_locations is None => default to full service area (legacy behavior)
                # - same_locations is []   => explicitly DISABLE SAME for this alert
                if same_locations is not None and len(same_locations) == 0:
                    log.info("SAME targeting disabled for this alert (no locations computed)")
                else:
                    if same_locations is not None:
                        locs = list(same_locations)
                    else:
                        locs = list(self.cfg.service_area.same_fips_all)

                    if not locs:
                        locs = ["000000"]

                    chunks = chunk_locations(locs) if chunk_locations is not None else [[]]

                    event_code = _safe_event_code(parsed.product_type)
                    issued = dt.datetime.now(tz=dt.timezone.utc)

                    hdr_wavs: list[Path] = []
                    for i, loc_chunk in enumerate(chunks):
                        hdr_msg = SameHeader(
                            org="WXR",
                            event=event_code,
                            locations=tuple(loc_chunk) if loc_chunk else tuple(["000000"]),
                            duration_minutes=self._same_duration_minutes(),
                            sender=self._same_sender(),
                            issued_utc=issued,
                        ).as_ascii()

                        hw = audio_dir / f"alert_{ts}_samehdr_{i}.wav"
                        render_same_bursts_wav(  # type: ignore[misc]
                            hw,
                            hdr_msg,
                            sample_rate=self.cfg.audio.sample_rate,
                            amplitude=self._same_amplitude(),
                        )
                        hdr_wavs.append(hw)

                    if len(hdr_wavs) == 1:
                        same_hdr_all = hdr_wavs[0]
                    elif len(hdr_wavs) > 1:
                        msg_gap = audio_dir / f"alert_{ts}_samehdr_msg_gap.wav"
                        write_silence_wav(msg_gap, 1.0, self.cfg.audio.sample_rate)

                        same_hdr_all = audio_dir / f"alert_{ts}_samehdr_all.wav"
                        parts2: list[Path] = []
                        for i, hw in enumerate(hdr_wavs):
                            parts2.append(hw)
                            if i != len(hdr_wavs) - 1:
                                parts2.append(msg_gap)
                        concat_wavs(same_hdr_all, parts2)

                    same_eom_wav = audio_dir / f"alert_{ts}_sameeom.wav"
                    render_same_eom_wav(  # type: ignore[misc]
                        same_eom_wav,
                        sample_rate=self.cfg.audio.sample_rate,
                        amplitude=self._same_amplitude(),
                    )

                    log.info(
                        "SAME enabled: event=%s sender=%s locs=%d chunks=%d",
                        event_code,
                        self._same_sender(),
                        len(locs),
                        len(chunks),
                    )
            except Exception:
                same_hdr_all = None
                same_eom_wav = None
                log.exception("SAME generation failed; continuing without SAME for this alert")
                # _SAME_FAIL_DL_
                self.discord.error(
                    title="SAME generation failed",
                    module="same.py",
                    exception_type="Exception",
                    message="SAME burst rendering raised an exception. Alert aired without SAME headers.",
                    context={"alert_type": parsed.product_type, "wfo": parsed.wfo},
                    fallback="Aired voice-only (no SAME)",
                )

        write_sine_wav(tone, self.cfg.audio.attention_tone_hz, self.cfg.audio.attention_tone_seconds, self.cfg.audio.sample_rate)
        self.tts.synth_to_wav(script_text, tts_wav)
        write_silence_wav(gap, self.cfg.audio.inter_segment_silence_seconds, self.cfg.audio.sample_rate)
        write_silence_wav(post, self.cfg.audio.post_alert_silence_seconds, self.cfg.audio.sample_rate)

        parts: list[Path] = []
        if same_hdr_all:
            parts.extend([same_hdr_all, gap])

        parts.extend([tone, gap, tts_wav])

        if same_eom_wav:
            parts.extend([gap, same_eom_wav])
        else:
            write_sine_wav(eom, self.cfg.audio.eom_beep_hz, self.cfg.audio.eom_beep_seconds, self.cfg.audio.sample_rate, amplitude=0.18)
            parts.extend([gap, eom])

        parts.append(post)

        concat_wavs(out, parts)
        return out

    def _assert_station_wav_format(self, wav_path: Path) -> None:
        try:
            with wave.open(str(wav_path), "rb") as wf:
                channels = int(wf.getnchannels())
                sample_width = int(wf.getsampwidth())
                sample_rate = int(wf.getframerate())
        except wave.Error as exc:
            raise ValueError(f"Input WAV is not readable: {exc}") from exc

        expected_rate = int(self.cfg.audio.sample_rate)
        if channels != 2 or sample_width != 2 or sample_rate != expected_rate:
            raise ValueError(
                f"Input WAV must be stereo 16-bit PCM at {expected_rate} Hz; got channels={channels}, sample_width={sample_width}, sample_rate={sample_rate}"
            )

    async def _render_pre_recorded_alert_audio(
        self,
        *,
        event_code: str,
        source_wav: Path,
        same_locations: list[str] | None = None,
    ) -> Path:
        self._assert_station_wav_format(source_wav)
        _, audio_dir, _, _ = self._paths()
        ts = dt.datetime.now(tz=self._tz).strftime("%Y%m%d-%H%M%S")

        tone = audio_dir / f"api_audio_alert_{ts}_tone.wav"
        gap = audio_dir / f"api_audio_alert_{ts}_gap.wav"
        eom = audio_dir / f"api_audio_alert_{ts}_eom.wav"
        post = audio_dir / f"api_audio_alert_{ts}_post.wav"
        out = audio_dir / f"api_audio_alert_{ts}.wav"

        same_hdr_all: Path | None = None
        same_eom_wav: Path | None = None

        if self._same_enabled() and SameHeader is not None:
            try:
                if same_locations is not None and len(same_locations) == 0:
                    log.info("SAME targeting disabled for this prerecorded alert (no locations computed)")
                else:
                    if same_locations is not None:
                        locs = list(same_locations)
                    else:
                        locs = list(self.cfg.service_area.same_fips_all)

                    if not locs:
                        locs = ["000000"]

                    chunks = chunk_locations(locs) if chunk_locations is not None else [[]]
                    issued = dt.datetime.now(tz=dt.timezone.utc)

                    hdr_wavs: list[Path] = []
                    for i, loc_chunk in enumerate(chunks):
                        hdr_msg = SameHeader(
                            org="WXR",
                            event=_safe_event_code(event_code),
                            locations=tuple(loc_chunk) if loc_chunk else tuple(["000000"]),
                            duration_minutes=self._same_duration_minutes(),
                            sender=self._same_sender(),
                            issued_utc=issued,
                        ).as_ascii()

                        hw = audio_dir / f"api_audio_alert_{ts}_samehdr_{i}.wav"
                        render_same_bursts_wav(
                            hw,
                            hdr_msg,
                            sample_rate=self.cfg.audio.sample_rate,
                            amplitude=self._same_amplitude(),
                        )
                        hdr_wavs.append(hw)

                    if len(hdr_wavs) == 1:
                        same_hdr_all = hdr_wavs[0]
                    elif len(hdr_wavs) > 1:
                        msg_gap = audio_dir / f"api_audio_alert_{ts}_samehdr_msg_gap.wav"
                        write_silence_wav(msg_gap, 1.0, self.cfg.audio.sample_rate)
                        same_hdr_all = audio_dir / f"api_audio_alert_{ts}_samehdr_all.wav"
                        parts2: list[Path] = []
                        for i, hw in enumerate(hdr_wavs):
                            parts2.append(hw)
                            if i != len(hdr_wavs) - 1:
                                parts2.append(msg_gap)
                        concat_wavs(same_hdr_all, parts2)

                    same_eom_wav = audio_dir / f"api_audio_alert_{ts}_sameeom.wav"
                    render_same_eom_wav(
                        same_eom_wav,
                        sample_rate=self.cfg.audio.sample_rate,
                        amplitude=self._same_amplitude(),
                    )
            except Exception:
                same_hdr_all = None
                same_eom_wav = None
                log.exception("SAME generation failed; continuing without SAME for prerecorded manual alert")

        write_sine_wav(tone, self.cfg.audio.attention_tone_hz, self.cfg.audio.attention_tone_seconds, self.cfg.audio.sample_rate)
        write_silence_wav(gap, self.cfg.audio.inter_segment_silence_seconds, self.cfg.audio.sample_rate)
        write_silence_wav(post, self.cfg.audio.post_alert_silence_seconds, self.cfg.audio.sample_rate)

        parts: list[Path] = []
        if same_hdr_all:
            parts.extend([same_hdr_all, gap])
        parts.extend([tone, gap, source_wav])
        if same_eom_wav:
            parts.extend([gap, same_eom_wav])
        else:
            write_sine_wav(eom, self.cfg.audio.eom_beep_hz, self.cfg.audio.eom_beep_seconds, self.cfg.audio.sample_rate, amplitude=0.18)
            parts.extend([gap, eom])
        parts.append(post)

        concat_wavs(out, parts)
        return out

    def _manual_full_eas_should_heighten(self) -> bool:
        return self.cfg.api.manual_full_eas_heightens

    async def _note_manual_station_feed(
        self,
        *,
        event_code: str,
        headline: str,
        voice_mode: str,
        same_locations: list[str] | None,
        out_wav: str,
        sender: str | None = None,
        expires_in_minutes: int | None = None,
        actor: str | None = None,
    ) -> None:
        if not _sf_enabled() or StationFeedAlert is None:
            return

        try:
            same_codes = [str(x).strip() for x in (same_locations or []) if str(x).strip()]
            event_text = _sf_eas_event_label_full(event_code)
            now_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
            expires_at = now_utc + dt.timedelta(minutes=max(1, int(expires_in_minutes or 30)))

            area_text = ""
            if same_codes:
                try:
                    area_text = await self._sf_area_text_from_same_codes(same_codes)
                except Exception:
                    area_text = ""
            if not area_text:
                area_text = str(self.cfg.station.service_area_name or "Unknown area").strip() or "Unknown area"

            sender_name = (sender or self.cfg.station.name or "SeasonalWeather").strip()
            if voice_mode == "full_eas":
                sf_headline = _sf_make_eas_headline(
                    org="WXR",
                    event_text=event_text,
                    area_text=area_text,
                    start_utc=now_utc,
                    end_utc=expires_at,
                    sender=sender_name,
                )
            else:
                sf_headline = (headline or event_text or "Manual message").strip()

            links = {
                "mode": ("FULL" if voice_mode == "full_eas" else "VOICE"),
                "wav": str(out_wav),
                "via": "local-api",
            }
            if actor:
                links["actor"] = str(actor).strip()[:64]

            alert = StationFeedAlert(
                id=_sf_sha1_12(f"api:{event_code}:{headline}:{out_wav}:{now_utc.isoformat()}"),
                event=str(event_text),
                headline=str(sf_headline),
                severity="Unknown",
                urgency="Unknown",
                certainty="Observed",
                area=str(area_text),
                effective=_sf_iso(now_utc),
                ends=_sf_iso(expires_at),
                expires=_sf_iso(expires_at),
                sent=_sf_iso(now_utc),
                sameCodes=same_codes,
                from_=(FeedSender(name=sender_name, kind="origin") if FeedSender else None),
                links=links,
            )
            _sf_emit(alert, expires_at=expires_at)
        except Exception:
            log.exception("Station feed: failed to note manual origination")

    async def _push_manual_originated_audio(
        self,
        *,
        wav_path: Path,
        headline: str,
        event_code: str,
        voice_mode: str,
        sender: str | None = None,
        actor: str | None = None,
        interrupt_policy: str = "interrupt_then_refill",
        same_locations: list[str] | None = None,
        expires_in_minutes: int | None = None,
        heightened_override: bool | None = None,
    ) -> dict[str, object]:
        policy = (interrupt_policy or "interrupt_then_refill").strip().lower()
        if policy != "interrupt_then_refill":
            raise ValueError(f"Unsupported interrupt policy: {interrupt_policy}")

        mode = (voice_mode or "voice_only").strip().lower()
        if mode not in {"voice_only", "full_eas"}:
            raise ValueError(f"Unsupported voice mode: {voice_mode}")

        same_codes = self._filter_same_locations_to_service_area(same_locations)

        async with self._cycle_lock:
            try:
                self.telnet.flush_cycle()
            except Exception:
                pass

            title = (headline or "Manual message").strip() or "Manual message"
            meta = self._np_meta(
                title=title,
                kind="alert",
                extra={
                    "sw_alert_source": "api",
                    "sw_alert_mode": ("full" if mode == "full_eas" else "voice"),
                    "sw_event_code": _safe_event_code(event_code),
                    "sw_event": title,
                    "sw_sender": (sender or "").strip(),
                    "sw_actor": (actor or "").strip(),
                },
            )
            self.telnet.push_alert(str(wav_path), meta=meta)

        now = dt.datetime.now(tz=self._tz)
        self.last_product_desc = title[:200]
        if mode == "full_eas":
            self.last_toneout_at = now
        # Tristate heightened override:
        #   True  → always heighten (works for voice_only too)
        #   False → suppress even if config says to heighten
        #   None  → fall back to station config (manual_full_eas_heightens, full_eas only)
        if heightened_override is not None:
            _should_heighten = heightened_override
        else:
            _should_heighten = mode == "full_eas" and self._manual_full_eas_should_heighten()
        if _should_heighten:
            self.last_heightened_at = now
            self.heightened_until = now + dt.timedelta(seconds=self.cfg.cycle.min_heightened_seconds)
            self._update_mode()

        self._schedule_cycle_refill("post-api-origination")
        await self._note_manual_station_feed(
            event_code=event_code,
            headline=headline,
            voice_mode=mode,
            same_locations=same_codes,
            out_wav=str(wav_path),
            sender=sender,
            expires_in_minutes=expires_in_minutes,
            actor=actor,
        )

        return {
            "ok": True,
            "headline": title,
            "event_code": _safe_event_code(event_code),
            "voice_mode": mode,
            "audio_path": str(wav_path),
            "same_codes": same_codes,
            "actor": (actor or "").strip(),
        }

    async def originate_manual_text(
        self,
        *,
        event_code: str,
        headline: str,
        script_text: str,
        voice_mode: str = "voice_only",
        same_locations: list[str] | None = None,
        sender: str | None = None,
        actor: str | None = None,
        interrupt_policy: str = "interrupt_then_refill",
        expires_in_minutes: int | None = None,
        heightened_override: bool | None = None,
    ) -> dict[str, object]:
        code = _safe_event_code(event_code)
        mode = (voice_mode or "voice_only").strip().lower()
        if mode == "full_eas":
            filtered_same = self._filter_same_locations_to_service_area(same_locations)
            dummy = SimpleNamespace(product_type=code, awips_id=None, wfo="LOCAL", raw_text="")
            wav_path = await self._render_alert_audio(dummy, script_text, same_locations=filtered_same)
        else:
            filtered_same = []
            wav_path = await self._render_voice_only_audio(script_text, prefix="api_text")

        result = await self._push_manual_originated_audio(
            wav_path=wav_path,
            headline=headline,
            event_code=code,
            voice_mode=mode,
            sender=sender,
            actor=actor,
            interrupt_policy=interrupt_policy,
            same_locations=filtered_same,
            expires_in_minutes=expires_in_minutes,
            heightened_override=heightened_override,
        )
        result["script_text"] = script_text
        return result

    async def originate_manual_audio(
        self,
        *,
        event_code: str,
        headline: str,
        wav_path: str | Path,
        voice_mode: str = "voice_only",
        same_locations: list[str] | None = None,
        sender: str | None = None,
        actor: str | None = None,
        interrupt_policy: str = "interrupt_then_refill",
        expires_in_minutes: int | None = None,
        heightened_override: bool | None = None,
    ) -> dict[str, object]:
        code = _safe_event_code(event_code)
        mode = (voice_mode or "voice_only").strip().lower()

        path = Path(str(wav_path))
        if not path.exists():
            raise FileNotFoundError(str(path))
        self._assert_station_wav_format(path)

        if mode == "full_eas":
            filtered_same = self._filter_same_locations_to_service_area(same_locations)
            out_wav = await self._render_pre_recorded_alert_audio(
                event_code=code,
                source_wav=path,
                same_locations=filtered_same,
            )
        elif mode == "voice_only":
            filtered_same = []
            out_wav = path
        else:
            raise ValueError(f"Unsupported voice mode: {voice_mode}")

        return await self._push_manual_originated_audio(
            wav_path=out_wav,
            headline=headline,
            event_code=code,
            voice_mode=mode,
            sender=sender,
            actor=actor,
            interrupt_policy=interrupt_policy,
            same_locations=filtered_same,
            expires_in_minutes=expires_in_minutes,
            heightened_override=heightened_override,
        )

    async def _render_cycle_segment_audio(self, seg: CycleSegment) -> Path:
        _, audio_dir, _, _ = self._paths()
        ts = dt.datetime.now(tz=self._tz).strftime("%Y%m%d-%H%M%S")
        safe_key = "".join(ch for ch in seg.key if ch.isalnum() or ch in {"_", "-"}).strip() or "seg"

        tts_wav = audio_dir / f"cycle_{ts}_{safe_key}_tts.wav"
        gap = audio_dir / f"cycle_{ts}_{safe_key}_gap.wav"
        out = audio_dir / f"cycle_{ts}_{safe_key}.wav"

        seg_gap = 0.45
        self.tts.synth_to_wav(seg.text, tts_wav)
        write_silence_wav(gap, seg_gap, self.cfg.audio.sample_rate)
        concat_wavs(out, [gap, tts_wav, gap])
        return out

    async def _queue_cycle_once(self, reason: str = "scheduled") -> None:
        async with self._cycle_lock:
            self._update_mode()
            interval = self._cycle_interval_seconds()

            ctx = CycleContext(
                mode=self.mode,
                last_heightened_ago=self._heightened_ago_str(),
                last_product_desc=self.last_product_desc,
            )

            try:
                segs = await self.cycle_builder.build_segments(
                    station_name=self.cfg.station.name,
                    service_area_name=self.cfg.station.service_area_name,
                    disclaimer=self.cfg.station.disclaimer,
                    ctx=ctx,
                )

                if self.live_time_enabled:
                    try:
                        if not self._live_time_wav_path().exists():
                            self._render_live_time_wav_once()
                    except Exception:
                        log.exception("Failed to ensure live time WAV exists")

                # Prepend active-alert voice segments (NWR rebroadcast style)
                # These are cycle_only (no SAME retone) and play in alert order.
                try:
                    _active = self.alert_tracker.get_cycle_alerts()
                    if _active:
                        _alert_segs: list[CycleSegment] = []
                        for _ae in _active:
                            if _ae.script_text.strip():
                                _alert_segs.append(CycleSegment(
                                    key=f"alert_{_ae.code}",
                                    title=_ae.event or _ae.code or "Alert",
                                    text=_ae.script_text,
                                ))
                        if _alert_segs:
                            segs = _alert_segs + list(segs)
                            log.debug(
                                "Cycle: prepended %d active alert segment(s) (%s)",
                                len(_alert_segs),
                                ", ".join(a.event for a in _active),
                            )
                except Exception:
                    log.exception("Cycle: alert segment injection failed")

                cycle_items: list[tuple[Path, str]] = []
                durs: list[float] = []

                for seg in segs:
                    if self.live_time_enabled and seg.key == "id":
                        stripped = _TIME_SENTENCE_RE.sub("", seg.text).strip()
                        seg2 = CycleSegment(key=seg.key, title=seg.title, text=stripped)
                        w = await self._render_cycle_segment_audio(seg2)
                        cycle_items.append((w, "id"))
                        cycle_items.append((self._live_time_wav_path(), "time"))
                    else:
                        w = await self._render_cycle_segment_audio(seg)
                        cycle_items.append((w, seg.key))

                for w, _k in cycle_items:
                    try:
                        durs.append(wav_duration_seconds(w))
                    except Exception:
                        durs.append(0.0)

                seq_dur = sum(d for d in durs if d and d > 0.0)
                if seq_dur <= 1.0:
                    seq_dur = float(max(10, interval))

                # Persist so _cycle_loop can base its next-regen sleep on actual audio length.
                self._last_cycle_seq_dur = seq_dur

                try:
                    self.telnet.flush_cycle()
                except Exception:
                    pass

                cover = max(30, interval + 30)
                repeats = int(math.ceil(cover / seq_dur))
                repeats = max(1, min(repeats, 20))

                for _ in range(repeats):
                    for w, k in cycle_items:
                        meta = self._np_meta(
                            title=self._np_cycle_title(k),
                            kind="cycle",
                            extra={"sw_cycle_key": k, "sw_mode": self.mode},
                        )
                        self.telnet.push_cycle(str(w), meta=meta)

                log.info(
                    "Queued segmented %s cycle (%ss, segs=%d, seq_dur=%.1fs, repeats=%d, reason=%s)",
                    self.mode,
                    interval,
                    len(segs),
                    seq_dur,
                    repeats,
                    reason,
                )
                # _CYCLE_DL_
                try:
                    self.discord.cycle_rebuilt(
                        reason=reason,
                        mode=self.mode,
                        interval=interval,
                        seq_dur=seq_dur,
                        segments=len(segs),
                        active_alerts=len(self.alert_tracker.get_cycle_alerts()),
                    )
                except Exception:
                    pass

            except Exception as e:
                log.exception("Segmented cycle build failed (%s): %s", reason, e)

    async def _cycle_loop(self) -> None:
        while True:
            self._update_mode()
            interval = self._cycle_interval_seconds()
            await self._queue_cycle_once(reason="scheduled")
            # Housekeeping: drop expired alerts from tracker once per cycle
            try:
                n = self.alert_tracker.purge_expired()
                if n:
                    log.info("AlertTracker: purged %d expired entry/entries", n)
            except Exception:
                pass

            # Sleep until (seq_dur - lead_time) so the next rebuild fires with enough
            # headroom for fetch + synth before the current audio train runs out.
            # Falls back to the configured interval if seq_dur wasn't captured yet.
            _lead = int(self.cfg.cycle.lead_time_seconds)
            _seq = self._last_cycle_seq_dur
            if _seq > 0:
                _sleep = max(30, _seq - _lead)
            else:
                _sleep = max(30, interval)
            log.debug(
                "Cycle loop sleeping %.1fs (seq_dur=%.1fs lead=%ds interval=%ds mode=%s)",
                _sleep, _seq, _lead, interval, self.mode,
            )
            await asyncio.sleep(_sleep)


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/etc/seasonalweather/config.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    orch = Orchestrator(cfg)
    asyncio.run(orch.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# _DISCORD_LOG_ALL_HOOKS_APPLIED_
