from __future__ import annotations

import datetime as dt
import json
import os
import re
import httpx
from dataclasses import dataclass
from typing import List, Optional, Tuple
from typing import Any, Dict, Iterable, Mapping
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



def _spc_threats_phrase(torn_prob: int, hail_prob: int, wind_prob: int, *, severity_dn: int) -> str:
    """
    Build a short, broadcast-friendly threat phrase from SPC probabilistic layers.

    Probabilities are typically percent (e.g., 2, 5, 15, 30).
    """
    threats: List[str] = []

    if torn_prob >= 2:
        threats.append("a few tornadoes")
    if hail_prob >= 5:
        threats.append("large hail")
    if wind_prob >= 5:
        threats.append("damaging winds")

    if threats:
        if len(threats) == 1:
            return threats[0].capitalize()
        if len(threats) == 2:
            return f"{threats[0].capitalize()} and {threats[1]}"
        return f"{threats[0].capitalize()}, {threats[1]}, and {threats[2]}"

    if severity_dn >= 6:
        return "Large hail, damaging winds, and a few tornadoes"
    if severity_dn >= 4:
        return "Large hail and damaging winds"
    return "Isolated severe storms"


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

        # caches for SPC/CWA lookups (best-effort)
        self._wfo_geom_cache: Dict[str, Dict[str, Any]] = {}
        self._arcgis_layer_cache: Dict[str, int] = {}

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

    async def _arcgis_get_json(self, url: str, params: Mapping[str, Any], timeout_s: float) -> Optional[Dict[str, Any]]:
        """
        ArcGIS REST helper.

        IMPORTANT:
          - If we pass a polygon geometry, the querystring can be huge; GET requests may be truncated.
          - Use POST (form-encoded) whenever the request includes a 'geometry' parameter.
        """
        try:
            async with httpx.AsyncClient(
                timeout=timeout_s,
                headers={"User-Agent": "SeasonalWeather/SeasonalNet"},
            ) as client:
                p = dict(params)
                if "geometry" in p:
                    r = await client.post(url, data=p)
                else:
                    r = await client.get(url, params=p)
                r.raise_for_status()
                return r.json()
        except Exception:
            return None


    async def _arcgis_find_layer_id(self, base_url: str, want_keywords: Iterable[str], timeout_s: float) -> Optional[int]:
        """
        Fetch MapServer metadata and find the first layer whose name contains all keywords.
        Caches results per (base_url, keywords).
        """
        key = f"{base_url}|" + "|".join(k.lower().strip() for k in want_keywords if k)
        if key in self._arcgis_layer_cache:
            return self._arcgis_layer_cache[key]

        svc = await self._arcgis_get_json(base_url, {"f": "pjson"}, timeout_s)
        layers = (svc or {}).get("layers") or []
        want = [k.lower().strip() for k in want_keywords if k and k.strip()]
        for layer in layers:
            name = str((layer or {}).get("name") or "").lower()
            if name and all(k in name for k in want):
                lid = (layer or {}).get("id")
                if isinstance(lid, int):
                    self._arcgis_layer_cache[key] = lid
                    return lid
        return None

    async def _arcgis_query(
        self,
        base_url: str,
        layer_id: int,
        where: str,
        *,
        geometry: Optional[Dict[str, Any]] = None,
        out_fields: str = "*",
        return_geometry: bool = False,
        timeout_s: float = 6.0,
    ) -> List[Dict[str, Any]]:
        url = f"{base_url.rstrip('/')}/{layer_id}/query"
        params: Dict[str, Any] = {
            "f": "pjson",
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "true" if return_geometry else "false",
            "outSR": 4326,
        }
        if geometry:
            params.update(
                {
                    "geometry": json.dumps(geometry),
                    "geometryType": "esriGeometryPolygon",
                    "spatialRel": "esriSpatialRelIntersects",
                    "inSR": 4326,
                }
            )

        data = await self._arcgis_get_json(url, params, timeout_s)
        feats = (data or {}).get("features") or []
        out: List[Dict[str, Any]] = []
        for f in feats:
            if isinstance(f, dict):
                out.append(f)
        return out

    async def _wfo_cwa_geometry(self, wfo: str, timeout_s: float) -> Optional[Dict[str, Any]]:
        """
        Returns an ESRI polygon geometry (wkid 4326) for a WFO CWA, best-effort.
        """
        wfo = (wfo or "").strip().upper()
        if not wfo:
            return None
        if wfo in self._wfo_geom_cache:
            return self._wfo_geom_cache[wfo]

        ref_base = "https://mapservices.weather.noaa.gov/static/rest/services/nws_reference_maps/nws_reference_map/MapServer"
        layer_id = (
            await self._arcgis_find_layer_id(ref_base, ["county warning area"], timeout_s)
            or await self._arcgis_find_layer_id(ref_base, ["cwa"], timeout_s)
            or await self._arcgis_find_layer_id(ref_base, ["weather forecast office"], timeout_s)
            or await self._arcgis_find_layer_id(ref_base, ["wfo"], timeout_s)
        )
        if layer_id is None:
            return None

        field_candidates = ["WFO", "WFO_ID", "WFOID", "CWA", "OFFICE", "SITE", "SITEID", "ID"]
        for fld in field_candidates:
            feats = await self._arcgis_query(ref_base, layer_id, f"{fld}='{wfo}'", return_geometry=True, timeout_s=timeout_s)
            if feats:
                geom = (feats[0] or {}).get("geometry")
                if isinstance(geom, dict) and ("rings" in geom):
                    if "spatialReference" not in geom:
                        geom["spatialReference"] = {"wkid": 4326}
                    self._wfo_geom_cache[wfo] = geom
                    return geom
        return None

    def _spc_dn_to_code(self, dn: int) -> str:
        if dn >= 8:
            return "HIGH"
        if dn >= 6:
            return "MDT"
        if dn >= 5:
            return "ENH"
        if dn >= 4:
            return "SLGT"
        if dn >= 3:
            return "MRGL"
        if dn >= 2:
            return "TSTM"
        return ""

    def _spc_code_to_spoken(self, code: str) -> str:
        c = (code or "").strip().upper()
        return {
            "MRGL": "marginal",
            "SLGT": "slight",
            "ENH": "enhanced",
            "MDT": "moderate",
            "HIGH": "high",
            "TSTM": "general thunderstorm",
        }.get(c, c.lower())

    async def _spc_max_risk_dn(self, day: int, wfos: List[str], timeout_s: float) -> int:
        """
        Return the maximum categorical risk DN intersecting any WFO CWA.
        Best-effort: returns 0 on failure.

        ArcGIS attribute keys vary in case; NOAA commonly returns:
          - dn, label, label2
        """
        spc_base = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/SPC_wx_outlks/MapServer"
        layer_id = await self._arcgis_find_layer_id(spc_base, [f"day {day}", "categorical"], timeout_s)
        if layer_id is None:
            return 0

        # Map both codes and words to DN
        code_to_dn = {"TSTM": 2, "MRGL": 3, "SLGT": 4, "ENH": 5, "MDT": 6, "HIGH": 8}
        word_to_dn = {
            "GENERAL": 2,
            "THUNDER": 2,
            "MARGINAL": 3,
            "SLIGHT": 4,
            "ENHANCED": 5,
            "MODERATE": 6,
            "HIGH": 8,
        }

        def get_attr(attrs: dict, *names: str):
            for n in names:
                if n in attrs:
                    return attrs.get(n)
            for n in names:
                ln = n.lower()
                if ln in attrs:
                    return attrs.get(ln)
            return None

        max_dn = 0
        for wfo in wfos:
            geom = await self._wfo_cwa_geometry(wfo, timeout_s)
            if not geom:
                continue

            feats = await self._arcgis_query(
                spc_base, layer_id, "1=1",
                geometry=geom, return_geometry=False, timeout_s=timeout_s
            )

            for f in feats:
                attrs = (f or {}).get("attributes") or {}

                # Prefer explicit dn if present
                dn = get_attr(attrs, "DN", "dn")
                if isinstance(dn, (int, float)):
                    max_dn = max(max_dn, int(dn))
                    continue

                # Otherwise parse label/label2
                lab = get_attr(attrs, "LABEL", "label", "LABEL2", "label2", "CAT", "cat", "CATEGORY", "category", "RISK", "risk")
                if isinstance(lab, str) and lab.strip():
                    u = lab.strip().upper()

                    # Try codes first
                    for code, dnv in code_to_dn.items():
                        if code in u:
                            max_dn = max(max_dn, dnv)
                            break
                    else:
                        # Then words (e.g., "Marginal", "Slight", etc.)
                        for word, dnv in word_to_dn.items():
                            if word in u:
                                max_dn = max(max_dn, dnv)
                                break

        return max_dn


    async def _spc_max_prob(self, day: int, hazard: str, wfos: List[str], timeout_s: float) -> int:
        """
        Max probability (percent) for a hazard (tornado/hail/wind) intersecting any WFO CWA.
        Returns 0 on failure.

        NOAA ArcGIS layers often store the probability in 'dn' (lowercase).
        """
        spc_base = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/SPC_wx_outlks/MapServer"
        haz = (hazard or "").strip().lower()
        if haz not in {"tornado", "hail", "wind"}:
            return 0

        layer_id = await self._arcgis_find_layer_id(spc_base, [f"day {day}", haz], timeout_s)
        if layer_id is None and haz == "tornado":
            layer_id = await self._arcgis_find_layer_id(spc_base, [f"day {day}", "torn"], timeout_s)
        if layer_id is None:
            return 0

        def get_attr(attrs: dict, *names: str):
            for n in names:
                if n in attrs:
                    return attrs.get(n)
            for n in names:
                ln = n.lower()
                if ln in attrs:
                    return attrs.get(ln)
            return None

        max_p = 0
        for wfo in wfos:
            geom = await self._wfo_cwa_geometry(wfo, timeout_s)
            if not geom:
                continue
            feats = await self._arcgis_query(spc_base, layer_id, "1=1", geometry=geom, return_geometry=False, timeout_s=timeout_s)
            for f in feats:
                attrs = (f or {}).get("attributes") or {}
                v = get_attr(attrs, "PROB", "prob", "PROBABILITY", "probability", "PERCENT", "percent", "VALUE", "value", "VAL", "val", "DN", "dn")
                if isinstance(v, (int, float)):
                    max_p = max(max_p, int(v))
        return max_p


    async def _build_spc_outlook_text(self, ctx: CycleContext, now: dt.datetime) -> Optional[str]:
        """
        Optional SPC convective outlook readout (Day 1-3) scoped to configured WFO CWAs.

        Enabled with:
          SEASONAL_CYCLE_SPC_ENABLE=1
          SEASONAL_CYCLE_SPC_WFOS=LWX,CTP,PHI
          SEASONAL_CYCLE_SPC_MIN_DN=3  (3=MRGL)
          SEASONAL_CYCLE_SPC_DAYS=3    (1..3)
        """
        if _env_int("SEASONAL_CYCLE_SPC_ENABLE", 0) != 1:
            return None

        wfos = [x.strip().upper() for x in os.environ.get("SEASONAL_CYCLE_SPC_WFOS", "LWX").split(",") if x.strip()]
        if not wfos:
            wfos = ["LWX"]

        days = 1 if ctx.mode == "heightened" else max(1, min(3, _env_int("SEASONAL_CYCLE_SPC_DAYS", 3)))
        min_dn = _env_int("SEASONAL_CYCLE_SPC_MIN_DN", 3)
        try:
            timeout_s = float(os.environ.get("SEASONAL_CYCLE_SPC_TIMEOUT_S", "6.0") or "6.0")
        except Exception:
            timeout_s = 6.0

        lines: List[str] = []

        for day in range(1, days + 1):
            dn = await self._spc_max_risk_dn(day, wfos, timeout_s)
            if dn < min_dn:
                continue

            code = self._spc_dn_to_code(dn)
            spoken = self._spc_code_to_spoken(code)

            if day == 1:
                torn = await self._spc_max_prob(1, "tornado", wfos, timeout_s)
                hail = await self._spc_max_prob(1, "hail", wfos, timeout_s)
                wind = await self._spc_max_prob(1, "wind", wfos, timeout_s)
                threats = _spc_threats_phrase(torn, hail, wind, severity_dn=dn)
                lines.append(
                    f"For today's convective outlook in our service area, there is a {spoken} risk of severe thunderstorms. {threats} will be possible."
                )
            elif day == 2:
                torn = await self._spc_max_prob(2, "tornado", wfos, timeout_s)
                hail = await self._spc_max_prob(2, "hail", wfos, timeout_s)
                wind = await self._spc_max_prob(2, "wind", wfos, timeout_s)
                threats = _spc_threats_phrase(torn, hail, wind, severity_dn=dn)
                lines.append(
                    f"For tomorrow's convective outlook in our service area, there is a {spoken} risk of severe thunderstorms. {threats} will be possible."
                )
            elif day == 3:
                d3 = now + dt.timedelta(days=2)
                lines.append(f"For {d3.strftime('%A')}, a {spoken} risk of severe thunderstorms is possible.")

        if not lines:
            return None

        return "And now, for the Storm Prediction Center's convective outlook for severe thunderstorms in our area. " + " ".join(lines)

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
        max_points = 1 if ctx.mode == "heightened" else _env_int("SEASONAL_CYCLE_FC_MAX_POINTS_NORMAL", 6)
        field = "shortForecast" if _env_int("SEASONAL_CYCLE_FC_USE_SHORT", 1) else "detailedForecast"
        max_periods = 1 if ctx.mode == "heightened" else _env_int("SEASONAL_CYCLE_FC_PERIODS_NORMAL", 2)
        line_max = _env_int("SEASONAL_CYCLE_FC_LINE_MAX_CHARS", 260)

        pts = list(self.points)
        if pts:
            rot_period = _env_int("SEASONAL_CYCLE_FC_ROTATE_PERIOD_S", 300)
            rot_step = _env_int("SEASONAL_CYCLE_FC_ROTATE_STEP", max_points)
            slot = int(now.timestamp() // max(rot_period, 1))
            offset = (slot * max(rot_step, 1)) % len(pts)
            pts = pts[offset:] + pts[:offset]

        for lat, lon, label in pts[:max_points]:
            try:
                periods = await self.api.point_forecast_periods(lat, lon)
                if periods:
                    p1 = periods[0]
                    p2 = periods[1] if len(periods) > 1 else None

                    line = f"{label}: {p1.get('name','')} — {p1.get(field,'')}"
                    if max_periods >= 2 and p2 and p2.get(field):
                        line += f" {p2.get('name','')} — {p2.get(field,'')}"
                    line = _scrub_nws_product_text(line)
                    line = _trim_chars(line, line_max)
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
                f"This is the SeasonalNet I P Weather Radio Station, {station_name}, "
                f"with station programming and streaming facilities originating from SeasonalNet, "
                f"providing weather information for {service_area_name}. "
                f"Due to severe weather affecting the service area, normal broadcasts have been curtailed to bring you the latest severe weather information. "
                f"The current time is {time_str}, {tz_short}. "
                f"{disclaimer}"
            )
        else:
            station_id = (
                f"This is the SeasonalNet I P Weather Radio Station, {station_name}, "
                f"with station programming and streaming facilities originating from SeasonalNet, "
                f"providing weather information for {service_area_name}. "
                f"The current time is {time_str}, {tz_short}. "
                f"{disclaimer}"
            )
        # --- Status ---
        status_bits: List[str] = []
        status_bits.append(f"The station's broadcast mode is currently in {ctx.mode} broadcast mode at this time.")
        if ctx.last_heightened_ago:
            status_bits.append(f"The most recent, heightened broadcast cycle mode activation was {ctx.last_heightened_ago} ago.")
        if ctx.last_product_desc:
            line = _last_product_status_line(ctx.last_product_desc)
            if line:
                status_bits.append(line)
        if active_titles:
            status_bits.append("These are the active watches, warnings, and advisories in effect: " + ", ".join(active_titles) + ".")
        else:
            status_bits.append("There are no active watches, warnings, or advisories in the service area at this time.")

        status_text = "And now for the overall station status, and alerts in the service area. " + " ".join(status_bits)

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
                    text="And now for the hazardous weather outlook for the service area. " + hwo_text,
                )
            )
        else:
            if _env_int("SEASONAL_CYCLE_HWO_SPEAK_UNAVAILABLE", 1) == 1:
                segments.append(
                    CycleSegment(
                        key="hwo-unavailable",
                        title="Hazardous Weather Outlook",
                        text="The hazardous weather outlook from LWX was unavailable.",
                    )
                )

        # --- SPC Convective Outlook (optional) ---
        spc_text = await self._build_spc_outlook_text(ctx, now)
        if spc_text:
            segments.append(
                CycleSegment(
                    key="spc",
                    title="SPC Convective Outlook",
                    text=spc_text,
                )
            )



        # --- “ZFP” key retained, but it’s now SYNOPSIS only (to avoid 1GB WAVs) ---
        if synopsis_text:
            segments.append(
                CycleSegment(
                    key="zfp",
                    title="Synopsis",
                    text=("This is the weather synopsis for our area. And now for the weather features affecting our area over the next several days. " + synopsis_text),
                )
            )

        # --- Forecast ---
        if fc_lines:
            if ctx.mode == "heightened":
                forecast_text = "This is the summarized forecast section for our area. " + " ".join(fc_lines)
            else:
                forecast_text = "This is the overall forecast section for our area from the National Weather Service. " + " ".join(fc_lines)
            segments.append(CycleSegment(key="fcst", title="Forecast", text=forecast_text))

        # --- Observations ---
        if obs_lines:
            obs_text = "And now for the current observed weather conditions in our area. " + " ".join(obs_lines)
            segments.append(CycleSegment(key="obs", title="Observations", text=obs_text))

        segments.append(
            CycleSegment(
                key="outro",
                title="Outro",
                text="This is the end of the current broadcast cycle. Updated information will follow on the next rotation.",
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
