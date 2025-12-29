from __future__ import annotations

import datetime as dt
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

from .nws_api import NWSApi
from .tts import clean_for_tts


def _fmt_time(now: dt.datetime) -> str:
    return now.strftime("%-I:%M %p")


def _short_tz(now: dt.datetime) -> str:
    return now.tzname() or "local"


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_SPACE_RE = re.compile(r"\s+")
_ALL_PUNCT_LINE_RE = re.compile(r"^[\W_]+$")

_WMO_HEADER_RE = re.compile(r"^[A-Z]{4}\d{2}\s+[A-Z]{4}\s+\d{6}$")
_ALL_ZERO_RE = re.compile(r"^0{3,}$")
_CODELINE_RE = re.compile(r"^[A-Z0-9/>\-.,\s]{10,}$")

# WFO designators like KLWX/KCTP/KPHI/etc
# NOTE: K[A-Z]{3} also matches airport IDs (KDCA/KBWI/etc), so we whitelist real WFOs we use.
_WFO_ALLOW = {"KLWX", "KCTP", "KPHI"}
_WFO_RE = re.compile(r"\bK[A-Z]{3}\b")


def _last_product_status_line(desc: str) -> str:
    s = (desc or "").strip()
    if not s:
        return ""

    # Keep this line sane for TTS/logs (avoid giant/ugly strings)
    s = clean_for_tts(s)
    s = _scrub_nws_product_text(s)
    s = _trim_chars(s, _env_int("SEASONAL_CYCLE_LAST_PRODUCT_MAX_CHARS", 260))
    if not s:
        return ""

    m = _WFO_RE.search(s)
    if m and m.group(0) in _WFO_ALLOW:
        return f"Most recently received product from {m.group(0)} was: {s}."

    return f"Most recently received product affecting the service area was: {s}."


# FIPS state code -> USPS postal abbreviation (used to derive CAP "area=" states from SAME/FIPS list)
_FIPS_TO_POSTAL = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
    "60": "AS",
    "66": "GU",
    "69": "MP",
    "72": "PR",
    "78": "VI",
}


def _areas_from_same_fips(same_fips_all: List[str]) -> List[str]:
    """
    Derive NWS CAP 'area=' state list from our configured SAME/FIPS allowlist.

    SAME county code format: PSSCCC
      - P = subdivision (0=entire county/zone)
      - SS = state FIPS (2 digits)
      - CCC = county/city FIPS (3 digits)

    Marine SAME codes in our config start with "07" (e.g. 073532) and are NOT state/county zones.
    We skip those for CAP area derivation.
    """
    out: set[str] = set()
    for s in same_fips_all or []:
        s = str(s).strip()
        if len(s) != 6 or not s.isdigit():
            continue
        if s.startswith("07"):  # marine codes
            continue
        st = _FIPS_TO_POSTAL.get(s[1:3])
        if st:
            out.add(st)
    return sorted(out)


def _scrub_nws_product_text(text: str) -> str:
    t = (text or "").replace("\r", "")
    out_lines: list[str] = []

    for raw in t.splitlines():
        line = raw.strip()

        if not line:
            out_lines.append("")
            continue

        if line in {"&&", "$$"}:
            continue

        if _ALL_PUNCT_LINE_RE.match(line):
            continue

        if _WMO_HEADER_RE.match(line):
            continue

        if _ALL_ZERO_RE.match(line):
            continue

        if _URL_RE.search(line):
            line = _URL_RE.sub(" ", line)
        if _EMAIL_RE.search(line):
            line = _EMAIL_RE.sub(" ", line)

        line = _SPACE_RE.sub(" ", line).strip(" -:;()[]<>")

        if not line:
            continue

        if _CODELINE_RE.match(line) and not any(ch.islower() for ch in line):
            continue

        out_lines.append(line)

    cleaned: list[str] = []
    blank = False
    for l in out_lines:
        if l == "":
            if blank:
                continue
            blank = True
            cleaned.append("")
        else:
            blank = False
            cleaned.append(l)

    return "\n".join(cleaned).strip()


def _trim_chars(text: str, max_chars: Optional[int]) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    if max_chars is None or int(max_chars) <= 0:
        return s
    if len(s) <= max_chars:
        return s

    cut = s[:max_chars].rsplit(" ", 1)[0].rstrip()
    if len(cut) < int(max_chars * 0.6):
        cut = s[:max_chars].rstrip()

    return cut + "…"


def _extract_afd_synopsis(raw: str) -> str:
    """
    Extract ONLY the AFD synopsis section, if present.
    AFD headings look like:
      .SYNOPSIS...
      .NEAR TERM /THROUGH TONIGHT/...
      .SHORT TERM /.../...
    """
    txt = (raw or "").replace("\r", "")
    lines = txt.splitlines()

    in_syn = False
    buf: list[str] = []

    for ln in lines:
        s = ln.strip("\n")

        if not in_syn:
            if s.strip().upper().startswith(".SYNOPSIS"):
                in_syn = True
            continue

        # Stop at next AFD section header
        ss = s.strip()
        if ss.startswith(".") and "..." in ss and ss.upper() == ss.upper():
            # Example: ".NEAR TERM /THROUGH TONIGHT/..."
            # This reliably marks the end of synopsis content.
            break

        buf.append(s)

    out = "\n".join(buf).strip()
    return out


@dataclass
class CycleContext:
    mode: str  # "normal" | "heightened"
    last_heightened_ago: Optional[str]
    last_product_desc: Optional[str]


@dataclass(frozen=True)
class CycleSegment:
    key: str
    title: str
    text: str


class CycleBuilder:
    def __init__(
        self,
        api: NWSApi,
        tz_name: str,
        obs_stations: List[str],
        reference_points: List[Tuple[float, float, str]],
        same_fips_all: List[str],
    ) -> None:
        self.api = api
        self.tz = ZoneInfo(tz_name)
        self.obs_stations = obs_stations
        self.points = reference_points
        self.same_fips = set(same_fips_all)

        # Derive CAP "area=" list from SAME/FIPS list (keeps PA/PHI/CTP etc automatically in sync)
        self.alert_areas = _areas_from_same_fips(same_fips_all)
        if not self.alert_areas:
            # fail-safe (should never happen with a real config)
            self.alert_areas = ["MD", "VA", "DC", "WV"]

    def _product_max_chars(self, kind: str, mode: str) -> Optional[int]:
        """
        Existing knobs stay the same:
          SEASONAL_CYCLE_HWO_MAX_CHARS_NORMAL / _HEIGHTENED
          SEASONAL_CYCLE_AFD_MAX_CHARS_NORMAL / _HEIGHTENED

        New (optional) knob for synopsis:
          SEASONAL_CYCLE_SYN_MAX_CHARS_NORMAL / _HEIGHTENED

        Defaults for synopsis are intentionally NOT unlimited so it can’t DOS your disk.
        """
        k = (kind or "").strip().upper()
        m = (mode or "normal").strip().lower()

        if k == "HWO":
            if m == "heightened":
                return _env_int("SEASONAL_CYCLE_HWO_MAX_CHARS_HEIGHTENED", 1200)
            return _env_int("SEASONAL_CYCLE_HWO_MAX_CHARS_NORMAL", 0)

        if k == "AFD":
            if m == "heightened":
                return _env_int("SEASONAL_CYCLE_AFD_MAX_CHARS_HEIGHTENED", 1000)
            return _env_int("SEASONAL_CYCLE_AFD_MAX_CHARS_NORMAL", 0)

        if k in {"SYN", "SYNOPSIS"}:
            if m == "heightened":
                return _env_int("SEASONAL_CYCLE_SYN_MAX_CHARS_HEIGHTENED", 900)
            return _env_int("SEASONAL_CYCLE_SYN_MAX_CHARS_NORMAL", 1500)

        return None

    async def _fetch_product_text(self, kind: str, office: str) -> Optional[str]:
        try:
            pid = await self.api.latest_product_id(kind, office)
            if not pid:
                return None
            prod = await self.api.get_product(pid)
            if not prod or not prod.product_text:
                return None
            return prod.product_text
        except Exception:
            return None

    async def _build_synopsis_text(self, ctx: CycleContext) -> Optional[str]:
        """
        “Synopsis” segment source order:
          1) SYN product if available
          2) AFD .SYNOPSIS only (NOT the full AFD)
          3) None (fail closed; never read full ZFP by accident)
        """
        # 1) Dedicated synopsis product if LWX publishes it
        syn_raw = await self._fetch_product_text("SYN", "LWX")
        if syn_raw:
            syn_clean = clean_for_tts(syn_raw)
            syn_clean = _trim_chars(syn_clean, self._product_max_chars("SYN", ctx.mode))
            syn_clean = _scrub_nws_product_text(syn_clean)
            if syn_clean:
                return syn_clean

        # 2) AFD synopsis extraction fallback
        afd_raw = await self._fetch_product_text("AFD", "LWX")
        if afd_raw:
            syn = _extract_afd_synopsis(afd_raw)
            if syn:
                syn_clean = clean_for_tts(syn)
                syn_clean = _trim_chars(syn_clean, self._product_max_chars("SYNOPSIS", ctx.mode))
                syn_clean = _scrub_nws_product_text(syn_clean)
                if syn_clean:
                    return syn_clean

        return None

    async def build_segments(
        self,
        station_name: str,
        service_area_name: str,
        disclaimer: str,
        ctx: CycleContext,
    ) -> List[CycleSegment]:
        now = dt.datetime.now(tz=self.tz)
        time_str = _fmt_time(now)
        tz_short = _short_tz(now)

        # --- Active alerts (filter to our SAME/FIPS list) ---
        alerts = await self.api.active_alerts(self.alert_areas)
        active_titles: List[str] = []
        for feat in alerts:
            props = (feat or {}).get("properties") or {}
            geocode = props.get("geocode") or {}
            sames = geocode.get("SAME") or geocode.get("same") or []
            if isinstance(sames, list) and any(str(x).zfill(6) in self.same_fips for x in sames):
                title = props.get("event") or props.get("headline")
                if isinstance(title, str) and title.strip():
                    active_titles.append(title.strip())
        active_titles = sorted(set(active_titles))[:8]

        # --- Latest HWO (best-effort) ---
        hwo_text: Optional[str] = None
        try:
            pid = await self.api.latest_product_id("HWO", "LWX")
            if pid:
                prod = await self.api.get_product(pid)
                if prod and prod.product_text:
                    hwo_clean = clean_for_tts(prod.product_text)
                    hwo_text = _trim_chars(hwo_clean, self._product_max_chars("HWO", ctx.mode))
        except Exception:
            hwo_text = None

        # --- Synopsis (NOT full ZFP) ---
        synopsis_text = await self._build_synopsis_text(ctx)

        # --- Forecast snippets (gridpoint) ---
        fc_lines: List[str] = []
        max_points = 1 if ctx.mode == "heightened" else 3
        for lat, lon, label in self.points[:max_points]:
            try:
                periods = await self.api.point_forecast_periods(lat, lon)
                if periods:
                    p1 = periods[0]
                    p2 = periods[1] if len(periods) > 1 else None

                    line = f"{label}: {p1.get('name','')} — {p1.get('detailedForecast','')}"
                    if ctx.mode != "heightened" and p2 and p2.get("detailedForecast"):
                        line += f" {p2.get('name','')} — {p2.get('detailedForecast','')}"
                    line = _scrub_nws_product_text(line)
                    if line:
                        fc_lines.append(line)
            except Exception:
                continue

        # --- Observations ---
        obs_lines: List[str] = []
        max_obs = 1 if ctx.mode == "heightened" else len(self.obs_stations)
        for st in self.obs_stations[:max_obs]:
            try:
                props = await self.api.latest_observation(st)
                if not props:
                    continue
                desc = props.get("textDescription") or ""
                temp_c = (props.get("temperature") or {}).get("value")
                temp_f = None
                if isinstance(temp_c, (int, float)):
                    temp_f = temp_c * 9 / 5 + 32
                wind_mps = (props.get("windSpeed") or {}).get("value")
                wind_mph = None
                if isinstance(wind_mps, (int, float)):
                    wind_mph = wind_mps * 2.23694

                seg = f"{st}: "
                if temp_f is not None:
                    seg += f"{round(temp_f)} degrees. "
                if isinstance(desc, str) and desc:
                    seg += f"{desc}. "
                if wind_mph is not None and wind_mph >= 3:
                    seg += f"Wind {round(wind_mph)} miles per hour."

                seg = _scrub_nws_product_text(seg).strip()
                if seg:
                    obs_lines.append(seg)
            except Exception:
                continue

        # --- Station ID ---
        if ctx.mode == "heightened":
            station_id = (
                f"This is {station_name}, an automated I P radio station weather broadcast stream for {service_area_name}. "
                f"This station is currently in a heightened, shortened broadcast cycle state due to severe weather affecting the service area. "
                f"The current time is {time_str}, {tz_short}. "
                f"{disclaimer}"
            )
        else:
            station_id = (
                f"This is {station_name}, an automated I P radio station, powered by Icecast, providing weather broadcasts for {service_area_name}. "
                f"This station provides National Weather Service forecasts, hazardous weather outlooks, and severe weather information for the service area. "
                f"The current time is {time_str}, {tz_short}. "
                f"{disclaimer}"
            )

        # --- Status ---
        status_bits: List[str] = []
        status_bits.append(f"Broadcast mode: {ctx.mode}.")
        if ctx.last_heightened_ago:
            status_bits.append(f"Most recent heightened mode activation was {ctx.last_heightened_ago} ago.")
        if ctx.last_product_desc:
            line = _last_product_status_line(ctx.last_product_desc)
            if line:
                status_bits.append(line)
        if active_titles:
            status_bits.append("Active alerts in the service area include: " + ", ".join(active_titles) + ".")
        else:
            status_bits.append("There are no active warnings in the service area at this time.")

        status_text = "Station status. " + " ".join(status_bits)

        segments: List[CycleSegment] = [
            CycleSegment(key="id", title="Station ID", text=station_id),
            CycleSegment(key="status", title="Status", text=status_text),
        ]

        # --- HWO ---
        if hwo_text:
            segments.append(
                CycleSegment(
                    key="hwo",
                    title="Hazardous Weather Outlook",
                    text="Hazardous Weather Outlook. " + hwo_text,
                )
            )

        # --- “ZFP” key retained, but it’s now SYNOPSIS only (to avoid 1GB WAVs) ---
        if synopsis_text:
            segments.append(
                CycleSegment(
                    key="zfp",
                    title="Synopsis",
                    text=(
                        "Taking a look at the weather features affecting our region for the next several days. "
                        + synopsis_text
                    ),
                )
            )

        # --- Forecast ---
        if fc_lines:
            forecast_text = "Forecast highlights. " + " ".join(fc_lines)
            segments.append(CycleSegment(key="fcst", title="Forecast", text=forecast_text))

        # --- Observations ---
        if obs_lines:
            obs_text = "Latest observations. " + " ".join(obs_lines)
            segments.append(CycleSegment(key="obs", title="Observations", text=obs_text))

        segments.append(
            CycleSegment(
                key="outro",
                title="Outro",
                text="This station will return with updated information on the next cycle.",
            )
        )

        return segments

    async def build_text(
        self,
        station_name: str,
        service_area_name: str,
        disclaimer: str,
        ctx: CycleContext,
    ) -> str:
        segs = await self.build_segments(station_name, service_area_name, disclaimer, ctx)
        return "\n\n".join(s.text for s in segs if s.text and s.text.strip())
