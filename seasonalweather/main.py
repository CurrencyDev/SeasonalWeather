from __future__ import annotations

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
import re
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx

from .config import load_config, AppConfig
from .nws_api import NWSApi
from .nwws_client import NWWSClient
from .product import parse_product_text, ParsedProduct
from .alert_builder import build_spoken_alert
from .tts import TTS
from .audio import write_sine_wav, write_silence_wav, concat_wavs, wav_duration_seconds
from .liquidsoap_telnet import LiquidsoapTelnet
from .cycle import CycleBuilder, CycleContext, CycleSegment

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

# Optional ERN/GWES SAME monitor (Level 3 source)
try:
    from .ern_gwes import ErnGwesMonitor, ErnSameEvent, defaults_from_env
except Exception:  # pragma: no cover
    ErnGwesMonitor = None  # type: ignore
    ErnSameEvent = None  # type: ignore
    defaults_from_env = None  # type: ignore


log = logging.getLogger("seasonalweather")

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


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _env_required(key: str) -> str:
    v = os.environ.get(key)
    if not v:
        raise RuntimeError(f"Missing required env var: {key} (set in /etc/seasonalweather/seasonalweather.env)")
    return v


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v else default


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _safe_event_code(raw: str | None) -> str:
    if not raw:
        return "SPS"
    s = "".join(ch for ch in str(raw).upper() if ch.isalnum())
    return s[:3] if len(s) >= 3 else "SPS"


def _fmt_time(now: dt.datetime) -> str:
    # 12-hour like "6:42 PM"
    return now.strftime("%-I:%M %p")


def _short_tz(now: dt.datetime) -> str:
    return now.tzname() or "local"


class Orchestrator:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.api = NWSApi()
        self.telnet = LiquidsoapTelnet(
            host=_env_str("LIQUIDSOAP_TELNET_HOST", "127.0.0.1"),
            port=_env_int("LIQUIDSOAP_TELNET_PORT", 1234),
        )

        self._tz = ZoneInfo(cfg.station.timezone)

        # NWWS-OI
        self.jid = _env_required("NWWS_JID")
        self.password = _env_required("NWWS_PASSWORD")
        self.nwws_server = _env_str("NWWS_SERVER", cfg.nwws.server)
        self.nwws_port = _env_int("NWWS_PORT", cfg.nwws.port)

        # TTS
        self.tts = TTS(
            backend=cfg.tts.backend,
            voice=cfg.tts.voice,
            rate_wpm=cfg.tts.rate_wpm,
            volume=cfg.tts.volume,
            sample_rate=cfg.audio.sample_rate,
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
        )

        # Fast membership checks for "in-area" targeting
        self._same_fips_allow_set = {str(x).strip() for x in cfg.service_area.same_fips_all if str(x).strip()}

        # --- NWWS flood-gate controls ---
        self._nwws_logger = logging.getLogger("seasonalweather.nwws")
        self._nwws_raw_seen = 0
        self._nwws_rx_log_first_n = _env_int("SEASONAL_NWWS_RX_LOG_FIRST_N", 20)
        self._nwws_decision_log_first_n = _env_int("SEASONAL_NWWS_DECISION_LOG_FIRST_N", 20)
        self._nwws_decision_log_every = _env_int("SEASONAL_NWWS_DECISION_LOG_EVERY", 0)
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

        # Live time WAV (drift killer)
        self.live_time_enabled = _env_bool("SEASONAL_LIVE_TIME_ENABLED", default=True)
        self.live_time_interval_seconds = _env_int("SEASONAL_LIVE_TIME_INTERVAL_SECONDS", 45)

        # --- Cross-source dedupe (NWWS vs CAP) ---
        self._dedupe_ttl_seconds = _env_int("SEASONAL_DEDUPE_TTL_SECONDS", 900)  # 15m default
        self._dedupe_lock = asyncio.Lock()
        self._recent_air_keys: dict[str, dt.datetime] = {}

        # --- NWWS decision visibility counters ---
        self._nwws_seen = 0
        self._nwws_acted = 0

        # --- NWS zone lookup (for NWWS UGC->SAME targeting) ---
        self._zone_client: httpx.AsyncClient | None = None
        self._zone_cache_same: dict[str, list[str]] = {}
        self._zone_cache_fail: dict[str, dt.datetime] = {}  # short-term backoff for bad zones
        self._zone_lock = asyncio.Lock()

        # --- ZoneCounty DBX crosswalk (forecast zone -> county FIPS -> SAME) ---
        self._zonecounty_lock = asyncio.Lock()
        self._zonecounty_loaded = False
        self._zonecounty_map: dict[str, list[str]] = {}


        # --- Marine areas crosswalk (marine zone -> coastal county FIPS -> SAME) ---
        self._mareas_lock = asyncio.Lock()
        self._mareas_loaded = False
        self._mareas_map: dict[str, list[str]] = {}

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
        now = dt.datetime.now(tz=self._tz)
        if self.heightened_until and now < self.heightened_until:
            self.mode = "heightened"
        else:
            self.mode = "normal"

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
        return self.cfg.cycle.heightened_interval_seconds if self.mode == "heightened" else self.cfg.cycle.normal_interval_seconds

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
            tz = m.group("tz").upper()
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
        SPS should sound like NWR-style: "And now, a Special Weather Statement..."
        We also try to include the issued time from the product header.
        """
        s = (script or "").strip()
        if not s:
            return s

        issued = self._nws_header_issued_phrase(official_text)
        lead = "And now, a Special Weather Statement."
        if issued:
            lead = f"And now, a Special Weather Statement, issued at {issued}."

        s2 = _SPS_INTRO_LEAD_RE.sub(lead + "\n", s, count=1)
        if s2 == s:
            s2 = lead + "\n" + s

        # If the next line is literally "Special Weather Statement.", drop it to avoid double-intro.
        s2 = re.sub(r"(?im)^\s*Special Weather Statement\.\s*", "", s2, count=1)
        return s2.strip()

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
        return _env_bool("SEASONAL_ZONECOUNTY_ENABLED", default=True)

    def _zonecounty_dbx_url(self) -> str:
        return _env_str("SEASONAL_ZONECOUNTY_DBX_URL", "").strip()

    def _zonecounty_cache_days(self) -> int:
        return _env_int("SEASONAL_ZONECOUNTY_CACHE_DAYS", 30)

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

            url = self._zonecounty_dbx_url()
            max_age_days = max(1, int(self._zonecounty_cache_days()))
            now = dt.datetime.now(tz=self._tz)

            need_download = True
            if dbx_path.exists():
                try:
                    mtime = dt.datetime.fromtimestamp(dbx_path.stat().st_mtime, tz=self._tz)
                    need_download = (now - mtime).total_seconds() > (max_age_days * 86400)
                except Exception:
                    need_download = True

            if url and need_download:
                try:
                    client = await self._ensure_zone_client()  # reuse UA/timeouts/headers
                    r = await client.get(url)
                    if r.status_code == 200 and r.content and len(r.content) > 1024:
                        tmp = dbx_path.with_suffix(".tmp")
                        tmp.write_bytes(r.content)
                        os.replace(str(tmp), str(dbx_path))
                        log.info("ZoneCounty DBX refreshed: %s (%d bytes)", dbx_path, len(r.content))
                    else:
                        log.warning(
                            "ZoneCounty DBX fetch failed (status=%s). Using existing cache if present.",
                            r.status_code,
                        )
                except Exception:
                    log.exception("ZoneCounty DBX download failed; using existing cache if present")

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



    # ---- Marine areas crosswalk (NWS mareas*.txt: marine zone -> coastal county FIPS -> SAME) ----
    def _mareas_enabled(self) -> bool:
        return _env_bool("SEASONAL_MAREAS_ENABLED", default=True)

    def _mareas_url(self) -> str:
        """Optional URL for a mareas*.txt style crosswalk."""
        return _env_str("SEASONAL_MAREAS_URL", "").strip()

    def _mareas_cache_days(self) -> int:
        return _env_int("SEASONAL_MAREAS_CACHE_DAYS", 30)

    def _mareas_path(self) -> Path:
        _, _audio, cache_dir, _logs = self._paths()
        return cache_dir / "mareas.txt"

    def _parse_mareas_txt(self, path: Path) -> dict[str, list[str]]:
        """
        Forgiving parser: finds a marine zone token like ANZ530 and any 5-digit FIPS or 6-digit SAME codes on the line.
        Outputs SAME codes (6 digits, leading 0 for county FIPS).
        """
        zone_re = re.compile(r"\b([A-Z]{3}\d{3})\b")
        fips5_re = re.compile(r"\b(\d{5})\b")
        same6_re = re.compile(r"\b(\d{6})\b")

        m: dict[str, list[str]] = {}
        seen: dict[str, set[str]] = {}

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                s0 = (ln or "").strip()
                if not s0 or s0.startswith("#"):
                    continue

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

                if not codes:
                    continue

                if zone not in m:
                    m[zone] = []
                    seen[zone] = set()

                for c in codes:
                    c2 = "".join(ch for ch in c if ch.isdigit()).zfill(6)
                    if len(c2) != 6:
                        continue
                    if c2 in seen[zone]:
                        continue
                    seen[zone].add(c2)
                    m[zone].append(c2)

        return m

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
                        log.info("Marine areas crosswalk refreshed: %s (%d bytes)", path, len(r.content))
                    else:
                        log.warning("Marine areas crosswalk fetch failed (status=%s). Using cache if present.", r.status_code)
                except Exception:
                    log.exception("Marine areas crosswalk download failed; using cache if present")

            if not path.exists():
                log.info("Marine areas crosswalk not available (no cache file). Marine zone->SAME mapping unavailable.")
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
        ua = _env_str("SEASONAL_NWS_USER_AGENT", "").strip()
        if not ua:
            ua = _env_str("SEASONAL_CAP_USER_AGENT", "SeasonalWeather (NWS zone mapper)").strip()
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

    # ---- CAP toggles ----
    def _cap_enabled(self) -> bool:
        return _env_bool("SEASONAL_CAP_ENABLED", default=False)

    def _cap_dryrun(self) -> bool:
        return _env_bool("SEASONAL_CAP_DRYRUN", default=True)

    def _cap_poll_seconds(self) -> int:
        return _env_int("SEASONAL_CAP_POLL_SECONDS", 60)

    def _cap_user_agent(self) -> str:
        return _env_str("SEASONAL_CAP_USER_AGENT", "SeasonalWeather (CAP monitor)")

    def _cap_url(self) -> str:
        return _env_str("SEASONAL_CAP_URL", "")

    def _cap_full_enabled(self) -> bool:
        return _env_bool("SEASONAL_CAP_FULL_ENABLED", default=True)

    def _cap_full_severities(self) -> set[str]:
        raw = _env_str("SEASONAL_CAP_FULL_SEVERITIES", "Severe,Extreme")
        return {s.strip().lower() for s in raw.split(",") if s.strip()}

    def _cap_full_events(self) -> set[str]:
        raw = _env_str("SEASONAL_CAP_FULL_EVENTS", "").strip()
        if raw:
            return {s.strip() for s in raw.split(",") if s.strip()}

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
        return _env_int("SEASONAL_CAP_FULL_COOLDOWN_SECONDS", 180)

    def _cap_voice_enabled(self) -> bool:
        return _env_bool("SEASONAL_CAP_VOICE_ENABLED", default=False)

    def _cap_voice_events(self) -> set[str]:
        raw = _env_str("SEASONAL_CAP_VOICE_EVENTS", "Special Weather Statement")
        return {s.strip() for s in raw.split(",") if s.strip()}

    def _cap_voice_cooldown_seconds(self) -> int:
        return _env_int("SEASONAL_CAP_VOICE_COOLDOWN_SECONDS", 600)

    # ---- ERN/GWES SAME monitor toggles ----
    def _ern_enabled(self) -> bool:
        return _env_bool("SEASONAL_ERN_ENABLED", default=False)

    def _ern_dryrun(self) -> bool:
        return _env_bool("SEASONAL_ERN_DRYRUN", default=True)

    def _ern_url(self) -> str:
        return _env_str("SEASONAL_ERN_URL", "").strip()

    def _ern_relay_enabled(self) -> bool:
        return _env_bool("SEASONAL_ERN_RELAY_ENABLED", default=False)

    def _ern_relay_events(self) -> set[str]:
        raw = _env_str("SEASONAL_ERN_RELAY_EVENTS", "RWT,RMT")
        return {s.strip().upper() for s in raw.split(",") if s.strip()}

    def _ern_relay_min_confidence(self) -> float:
        return _env_float("SEASONAL_ERN_RELAY_MIN_CONFIDENCE", 0.80)

    def _ern_relay_cooldown_seconds(self) -> int:
        return _env_int("SEASONAL_ERN_RELAY_COOLDOWN_SECONDS", 300)

    def _ern_relay_senders(self) -> set[str]:
        raw = _env_str("SEASONAL_ERN_RELAY_SENDERS", "").strip()
        if not raw:
            return set()
        return {s.strip().upper() for s in raw.split(",") if s.strip()}

    # ---- SAME toggles ----
    def _same_enabled(self) -> bool:
        return _env_bool("SEASONAL_SAME_ENABLED", default=False)

    def _same_sender(self) -> str:
        return _env_str("SEASONAL_SAME_SENDER", "SEASNWXR")

    def _same_duration_minutes(self) -> int:
        return _env_int("SEASONAL_SAME_DURATION_MINUTES", 60)

    def _same_amplitude(self) -> float:
        return _env_float("SEASONAL_SAME_AMPLITUDE", 0.35)

    # ---- LIVE TIME WAV (drift killer) ----
    def _live_time_wav_path(self) -> Path:
        _, audio_dir, _, _ = self._paths()
        return audio_dir / "cycle_time_now.wav"

    def _live_time_text(self) -> str:
        now = dt.datetime.now(tz=self._tz)
        return f"The current time is {_fmt_time(now)}, {_short_tz(now)}."

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
        return _env_bool("SEASONAL_TESTS_ENABLED", default=False)

    def _tests_postpone_minutes(self) -> int:
        return _env_int("SEASONAL_TESTS_POSTPONE_MINUTES", 15)

    def _tests_max_postpone_hours(self) -> int:
        return _env_int("SEASONAL_TESTS_MAX_POSTPONE_HOURS", 6)

    def _tests_jitter_seconds(self) -> int:
        return _env_int("SEASONAL_TESTS_JITTER_SECONDS", 60)

    def _tests_toneout_cooldown_seconds(self) -> int:
        return _env_int("SEASONAL_TESTS_TONEOUT_COOLDOWN_SECONDS", int(self.cfg.cycle.min_heightened_seconds))

    def _tests_cap_block_seconds(self) -> int:
        return _env_int("SEASONAL_TESTS_CAP_BLOCK_SECONDS", 3600)

    def _tests_ern_block_seconds(self) -> int:
        return _env_int("SEASONAL_TESTS_ERN_BLOCK_SECONDS", 3600)

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
            self.telnet.push_alert(str(out_wav))

        self._schedule_cycle_refill("post-test")
        log.info("Originated %s test (audio=%s)", code, out_wav)

    async def run(self) -> None:
        work, audio, cache, logs = self._paths()
        for p in (work, audio, cache, logs):
            p.mkdir(parents=True, exist_ok=True)

        await self._wait_for_liquidsoap()

        tasks: list[asyncio.Task] = []

        if self.live_time_enabled:
            tasks.append(asyncio.create_task(self._live_time_loop(), name="live_time_wav"))

        xmpp = NWWSClient(self.jid, self.password, self.nwws_server, self.nwws_port, self.nwws_queue)
        tasks.append(asyncio.create_task(xmpp.run_forever(), name="nwws_xmpp"))
        tasks.append(asyncio.create_task(self._consume_nwws(), name="nwws_consumer"))
        tasks.append(asyncio.create_task(self._cycle_loop(), name="cycle_loop"))

        if self._cap_enabled():
            if NwsCapPoller is None or CapAlertEvent is None:
                log.warning("CAP enabled but cap_nws.py import failed; CAP is disabled.")
            else:
                kwargs = dict(
                    out_queue=self.cap_queue,
                    same_fips_allow=self.cfg.service_area.same_fips_all,
                    poll_seconds=self._cap_poll_seconds(),
                    user_agent=self._cap_user_agent(),
                )
                url = self._cap_url().strip()
                if url:
                    kwargs["url"] = url  # type: ignore[assignment]

                cap = NwsCapPoller(**kwargs)  # type: ignore[arg-type]
                tasks.append(asyncio.create_task(cap.run_forever(), name="cap_poller"))
                tasks.append(asyncio.create_task(self._consume_cap(), name="cap_consumer"))
                log.info("CAP ingest enabled (dryrun=%s full=%s voice=%s)", self._cap_dryrun(), self._cap_full_enabled(), self._cap_voice_enabled())
        else:
            log.info("CAP ingest disabled (set SEASONAL_CAP_ENABLED=1 to enable)")

        if self._ern_enabled():
            if ErnGwesMonitor is None or ErnSameEvent is None:
                log.warning("ERN enabled but ern_gwes.py import failed; ERN is disabled.")
            else:
                url = self._ern_url()
                if not url:
                    log.warning("ERN enabled but SEASONAL_ERN_URL is empty; ERN is disabled.")
                else:
                    env_defaults = defaults_from_env() if defaults_from_env is not None else {}
                    mon = ErnGwesMonitor(
                        out_queue=self.ern_queue,
                        same_fips_allow=self.cfg.service_area.same_fips_all,
                        url=url,
                        **env_defaults,
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
            log.info("ERN monitor disabled (set SEASONAL_ERN_ENABLED=1 to enable)")

        if self._tests_enabled():
            try:
                state_path = str(Path(self.cfg.paths.work_dir) / "rwt_rmt_state.json")

                sched = RwtRmtSchedule(
                    enabled=True,
                    tz_name=self.cfg.station.timezone,

                    rwt_enabled=True,
                    rwt_weekday=_env_int("SEASONAL_RWT_WEEKDAY", 2),
                    rwt_hour=_env_int("SEASONAL_RWT_HOUR", 11),
                    rwt_minute=_env_int("SEASONAL_RWT_MINUTE", 0),

                    rmt_enabled=True,
                    rmt_nth=_env_int("SEASONAL_RMT_NTH", 1),
                    rmt_weekday=_env_int("SEASONAL_RMT_WEEKDAY", 2),
                    rmt_hour=_env_int("SEASONAL_RMT_HOUR", 11),
                    rmt_minute=_env_int("SEASONAL_RMT_MINUTE", 0),

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
            log.info("RWT/RMT scheduler disabled (set SEASONAL_TESTS_ENABLED=1 to enable)")

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

    def _cap_is_actionable(self, ev: "CapAlertEvent") -> bool:  # type: ignore[name-defined]
        try:
            if str(ev.status or "").strip().lower() != "actual":
                return False
            mt = str(ev.message_type or "").strip().lower()
            if mt and mt not in {"alert", "update"}:
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
                vtec = self._cap_vtec_list(ev)
                tracks = self._vtec_tracks(vtec)
                full_actions = {"NEW", "UPG", "EXA", "EXB"}
                vtec_actions = {act for (_t, act) in tracks} if tracks else set()

                # If we have VTEC and none are FULL-worthy actions, treat as non-FULL.
                if tracks and not (vtec_actions & full_actions):
                    return False

                # If update has no usable VTEC, be conservative: don't FULL-tone it.
                if not tracks:
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

    async def _consume_cap(self) -> None:
        while True:
            ev = await self.cap_queue.get()

            vtec = self._cap_vtec_list(ev)
            tracks = self._vtec_tracks(vtec)

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

            if self._cap_should_voice(ev):
                await self._air_cap_voice(ev)

    async def _air_cap_full(self, ev: "CapAlertEvent") -> None:  # type: ignore[name-defined]
        now = dt.datetime.now(tz=self._tz)

        key = (str(ev.alert_id or "").strip(), str(ev.sent or "").strip())
        last = self._cap_full_last_by_key.get(key)
        if last and (now - last).total_seconds() < self._cap_full_cooldown_seconds():
            log.info("CAP full: cooldown active; skipping id=%s sent=%s event=%s", ev.alert_id, ev.sent, ev.event)
            return

        script = self._build_cap_full_script(ev)
        if not script.strip():
            return

        vtec = self._cap_vtec_list(ev)
        tracks = self._vtec_tracks(vtec)

        same_code = self._cap_event_to_same_code((ev.event or "").strip())
        same_locs_raw = list(ev.same_fips) if getattr(ev, "same_fips", None) else []
        same_locs = self._filter_same_locations_to_service_area(same_locs_raw)

        keys: list[str] = []

        # Track-level dedupe (prevents CAP vs NWWS double-air)
        for track_id, _act in tracks:
            keys.append(f"TRACKFULL:{track_id}")

        # Also keep raw VTEC strings (fine-grain)
        for v in vtec:
            keys.append(f"VTEC:{v}")

        # Functional FULL dedupe shared with ERN (VTEC-independent)
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
                self.telnet.push_alert(str(out_wav))

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
            log.info("CAP ACTION: aired FULL event=%s code=%s id=%s sent=%s vtec=%s audio=%s", ev.event, same_code, ev.alert_id, ev.sent, ",".join(vtec[:2]) if vtec else "", out_wav)
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
                self.telnet.push_alert(str(out_wav))

            self._cap_voice_last_by_key[key] = now
            self.last_product_desc = f"CAP {ev.event}".strip()

            self._schedule_cycle_refill("post-cap-voice")
            log.info("CAP ACTION: aired voice-only event=%s id=%s sent=%s audio=%s", ev.event, ev.alert_id, ev.sent, out_wav)
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

    def _build_cap_full_script(self, ev: "CapAlertEvent") -> str:  # type: ignore[name-defined]
        event = self._clean_cap_text(ev.event or "", limit=120)
        headline = self._clean_cap_text(ev.headline or "", limit=280)
        area = self._clean_cap_text(getattr(ev, "area_desc", "") or "", limit=320)
        desc = self._clean_cap_text(getattr(ev, "description", "") or "", limit=1200)
        instr = self._clean_cap_text(getattr(ev, "instruction", "") or "", limit=700)

        lines: list[str] = []
        lines.append("The National Weather Service has issued the following message.")
        if event:
            lines.append(f"{event}.")

        if headline:
            lines.append(headline if headline.endswith((".", "!", "?")) else headline + ".")

        if area:
            lines.append(f"For the following areas: {area}.")

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

        lines: list[str] = []
        lines.append("This is a statement from the National Weather Service.")
        if event and event.lower() != "special weather statement":
            lines.append(f"{event}.")
        if headline:
            lines.append(headline if headline.endswith((".", "!", "?")) else headline + ".")
        if area:
            lines.append(f"For the following areas: {area}.")
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
                self.telnet.push_alert(str(out_wav))

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

    async def _handle_toneout(self, parsed: ParsedProduct) -> None:
        log.info("NWWS toneout candidate: type=%s awips=%s wfo=%s", parsed.product_type, parsed.awips_id or "", parsed.wfo)

        official_text = parsed.raw_text
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
                    official_text = prod.product_text
        except Exception:
            pass

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

        # Decide FULL vs voice-only based on VTEC action if present.
        # FULL: NEW, UPG, EXA, EXB
        # Voice-only: CON, EXT, COR, ROU, CAN, EXP (and other non-FULL actions)
        full_actions = {"NEW", "UPG", "EXA", "EXB"}
        vtec_actions = {act for (_t, act) in tracks} if tracks else set()
        should_full = (not tracks) or bool(vtec_actions & full_actions)



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

        # Functional FULL dedupe shared with ERN (VTEC-independent)
        if should_full:
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
            # prefer a small 'has expired' narration instead of reading the whole product.
            try:
                if tracks and not should_full:
                    if ('EXP' in vtec_actions) or ('CAN' in vtec_actions):
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
                self.telnet.push_alert(str(out_wav))

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

            self._schedule_cycle_refill("post-alert")
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
                    station_name=_env_str("STATION_NAME", self.cfg.station.name),
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

                wavs: list[Path] = []
                durs: list[float] = []

                for seg in segs:
                    if self.live_time_enabled and seg.key == "id":
                        stripped = _TIME_SENTENCE_RE.sub("", seg.text).strip()
                        seg2 = CycleSegment(key=seg.key, title=seg.title, text=stripped)
                        w = await self._render_cycle_segment_audio(seg2)
                        wavs.append(w)
                        wavs.append(self._live_time_wav_path())
                    else:
                        w = await self._render_cycle_segment_audio(seg)
                        wavs.append(w)

                for w in wavs:
                    try:
                        durs.append(wav_duration_seconds(w))
                    except Exception:
                        durs.append(0.0)

                seq_dur = sum(d for d in durs if d and d > 0.0)
                if seq_dur <= 1.0:
                    seq_dur = float(max(10, interval))

                try:
                    self.telnet.flush_cycle()
                except Exception:
                    pass

                cover = max(30, interval + 30)
                repeats = int(math.ceil(cover / seq_dur))
                repeats = max(1, min(repeats, 20))

                for _ in range(repeats):
                    for w in wavs:
                        self.telnet.push_cycle(str(w))

                log.info(
                    "Queued segmented %s cycle (%ss, segs=%d, seq_dur=%.1fs, repeats=%d, reason=%s)",
                    self.mode,
                    interval,
                    len(segs),
                    seq_dur,
                    repeats,
                    reason,
                )

            except Exception as e:
                log.exception("Segmented cycle build failed (%s): %s", reason, e)

    async def _cycle_loop(self) -> None:
        while True:
            self._update_mode()
            interval = self._cycle_interval_seconds()
            await self._queue_cycle_once(reason="scheduled")
            await asyncio.sleep(max(30, interval))


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
