"""
SeasonalWeather -- rwr.py
Regional Weather Roundup (RWR) parser and NWR-style observation formatter.

Primary path : parse LWX RWR product text -> structured data -> NWR spoken prose
Fallback path: api.weather.gov ASOS observations -> same spoken format
Pressure cache: persistent JSON, derives R/F/S trend across service restarts

Public API
----------
parse_rwr(text)              Parse raw RWR product text -> RwrProduct
ObsPressureCache             Persistent per-station pressure history
asos_to_rwr_station(...)     Adapt NWS API obs dict to RwrStation
build_rwr_obs_text(...)      Main entry point: returns spoken obs segment text
"""
from __future__ import annotations

import json
import re
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Sky condition code -> spoken
# ---------------------------------------------------------------------------

_SKY_SPOKEN: Dict[str, str] = {
    # ------------------------------------------------------------------
    # Sky condition only
    # ------------------------------------------------------------------
    "CLOUDY":   "cloudy",
    "MOCLDY":   "mostly cloudy",
    "PTCLDY":   "partly cloudy",
    "FAIR":     "fair",
    "CLEAR":    "clear",
    "SUNNY":    "sunny",
    "MOSUNNY":  "mostly sunny",
    "PTSUNNY":  "partly sunny",
    "OVERCAST": "overcast",

    # ------------------------------------------------------------------
    # Obscurations / visibility phenomena
    # ------------------------------------------------------------------
    "FOG":          "fog",
    "FZFOG":        "freezing fog",
    "FOG/MIST":     "fog",                    # api.weather.gov composite
    "FREEZINGFOG":  "freezing fog",           # ASOS space-stripped form
    "MIST":         "mist",
    "HAZE":         "haze",
    "SMOKE":        "smoke",
    "DUST":         "dust",
    "BLDU":         "blowing dust",
    "BLDS":         "blowing dust",
    "BLGSNO":       "blowing snow",
    "BLGSNO+":      "heavy blowing snow",

    # ------------------------------------------------------------------
    # Rain
    # RWR spaced form | concatenated ASOS-path form | api.weather.gov form
    # ------------------------------------------------------------------
    "RAIN":             "rain",
    "LGT RAIN":         "light rain",
    "HVY RAIN":         "heavy rain",
    "LGTRAIN":          "light rain",          # space-stripped RWR
    "HVYRAIN":          "heavy rain",          # space-stripped RWR
    "LIGHTRAIN":        "light rain",          # api.weather.gov
    "HEAVYRAIN":        "heavy rain",          # api.weather.gov
    "RAIN SHWRS":       "rain showers",
    "RAINSHWRS":        "rain showers",
    "RAINSHOWERS":      "rain showers",
    "LGT RAIN SHWRS":   "light rain showers",
    "HVY RAIN SHWRS":   "heavy rain showers",
    "LIGHTRAINSHOWERS": "light rain showers",  # api.weather.gov
    "HEAVYRAINSHOWERS": "heavy rain showers",  # api.weather.gov

    # ------------------------------------------------------------------
    # Drizzle
    # ------------------------------------------------------------------
    "DRZL":                 "drizzle",
    "DRIZZLE":              "drizzle",
    "LGT DRZL":             "light drizzle",
    "HVY DRZL":             "heavy drizzle",
    "LGTDRZL":              "light drizzle",
    "HVYDRZL":              "heavy drizzle",
    "LIGHTDRIZZLE":         "light drizzle",
    "HEAVYDRIZZLE":         "heavy drizzle",

    # ------------------------------------------------------------------
    # Freezing rain
    # ------------------------------------------------------------------
    "FZRAIN":               "freezing rain",
    "LGT FZRN":             "light freezing rain",
    "HVY FZRN":             "heavy freezing rain",
    "LGTFZRN":              "light freezing rain",
    "HVYFZRN":              "heavy freezing rain",
    "LIGHTFREEZINGRAIN":    "light freezing rain",
    "HEAVYFREEZINGRAIN":    "heavy freezing rain",

    # ------------------------------------------------------------------
    # Freezing drizzle
    # ------------------------------------------------------------------
    "FZDRZL":               "freezing drizzle",
    "LGT FZDRZL":           "light freezing drizzle",
    "HVY FZDRZL":           "heavy freezing drizzle",
    "LGTFZDRZL":            "light freezing drizzle",
    "HVYFZDRZL":            "heavy freezing drizzle",
    "LIGHTFREEZINGDRIZZLE": "light freezing drizzle",
    "HEAVYFREEZINGDRIZZLE": "heavy freezing drizzle",

    # ------------------------------------------------------------------
    # Snow
    # ------------------------------------------------------------------
    "SNOW":             "snow",
    "LGT SNOW":         "light snow",
    "HVY SNOW":         "heavy snow",
    "LGTSNOW":          "light snow",
    "HVYSNOW":          "heavy snow",
    "LIGHTSNOW":        "light snow",
    "HEAVYSNOW":        "heavy snow",
    "FLRIES":           "snow flurries",
    "FLURRIES":         "snow flurries",
    "SNOW SHWRS":       "snow showers",
    "SNOWSHWRS":        "snow showers",
    "SNOWSHOWERS":      "snow showers",
    "LGT SNOW SHWRS":   "light snow showers",
    "HVY SNOW SHWRS":   "heavy snow showers",
    "LIGHTSNOWSHOWERS": "light snow showers",  # api.weather.gov
    "LIGHTSNOWSHOWER":  "light snow showers",  # api.weather.gov variant
    "HEAVYSNOWSHOWERS": "heavy snow showers",
    "BLIZZD":           "blizzard conditions",
    "BLGSNO":           "blowing snow",        # duplicate handled; last wins

    # ------------------------------------------------------------------
    # Sleet / ice pellets
    # ------------------------------------------------------------------
    "SLEET":            "sleet",
    "LGT SLEET":        "light sleet",
    "HVY SLEET":        "heavy sleet",
    "LGTSLEET":         "light sleet",
    "HVYSLEET":         "heavy sleet",
    "LIGHTSLEET":       "light sleet",
    "HEAVYSLEET":       "heavy sleet",
    "ICEGRPL":          "ice and snow pellets",
    "ICEPELLETS":       "ice pellets",
    "LIGHTICEANDSNOWPELLETS": "light ice and snow pellets",
    "LIGHTICE":         "light ice pellets",
    "HEAVYICE":         "heavy ice pellets",

    # ------------------------------------------------------------------
    # Mixed precipitation
    # ------------------------------------------------------------------
    "RAINANDSNOW":      "rain and snow",
    "SNOWANDRAIN":      "snow and rain",
    "WINTRY MIX":       "wintry mix",
    "WINTRYMIX":        "wintry mix",

    # ------------------------------------------------------------------
    # Thunderstorms
    # ------------------------------------------------------------------
    "TSTRM":            "thunderstorms",
    "TSTMS":            "thunderstorms",
    "TSRAIN":           "thunderstorms and rain",
    "THUNDERSTORM":     "thunderstorm",
    "THUNDERSTORMS":    "thunderstorms",
    "LIGHTTHUNDERSTORMANDHEAVYRAIN": "thunderstorm with heavy rain",
    "LIGHTTHUNDERSTORMANDRAIN":      "thunderstorm with rain",
    "HEAVYTHUNDERSTORMANDRAIN":      "severe thunderstorm with rain",
    "THUNDERSTORMWITHRAIN":          "thunderstorm with rain",
    "THUNDERSTORMWITHHEAVYRAIN":     "thunderstorm with heavy rain",
    "THUNDERSTORMWITHLIGHTRAIN":     "thunderstorm with light rain",
}


# Sky codes from ASOS textDescription -> normalise to lower-case spoken form
# (ASOS already returns English text; we just lower-case and clean it up)

# ---------------------------------------------------------------------------
# Compass directions (used for both RWR wind parsing and ASOS deg->compass)
# ---------------------------------------------------------------------------

_COMPASS_DIRS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]

_COMPASS_SPOKEN: Dict[str, str] = {
    "N":   "north",          "NNE": "north-northeast",
    "NE":  "northeast",      "ENE": "east-northeast",
    "E":   "east",           "ESE": "east-southeast",
    "SE":  "southeast",      "SSE": "south-southeast",
    "S":   "south",          "SSW": "south-southwest",
    "SW":  "southwest",      "WSW": "west-southwest",
    "W":   "west",           "WNW": "west-northwest",
    "NW":  "northwest",      "NNW": "north-northwest",
}

def _degrees_to_compass(degrees: float) -> str:
    """Convert wind direction in degrees to 16-point compass abbreviation."""
    idx = round(float(degrees) / 22.5) % 16
    return _COMPASS_DIRS[idx]


# ---------------------------------------------------------------------------
# Pressure trend
# ---------------------------------------------------------------------------

_TREND_SPOKEN: Dict[str, str] = {
    "R": "rising",
    "F": "falling",
    "S": "steady",
}

# ---------------------------------------------------------------------------
# RWR section title -> spoken intro phrase
# None means use "elsewhere in the region"
# ---------------------------------------------------------------------------

_SECTION_SPOKEN: Dict[str, Optional[str]] = {
    "WASHINGTON METRO":                "the Washington metro area",
    "BALTIMORE METRO":                 "the Baltimore metro area",
    "MARYLAND EASTERN SHORE":          "the Maryland Eastern Shore",
    "SOUTHERN MARYLAND":               "southern Maryland",
    "NORTH CENTRAL MARYLAND":          "north central Maryland",
    "WESTERN MARYLAND":                "western Maryland",
    "SHENANDOAH VALLEY":               "the Shenandoah Valley",
    "EASTERN WEST VIRGINIA PANHANDLE": "eastern West Virginia",
    "CENTRAL FOOTHILLS":               "the central foothills",
    "NORTH AND CENTRAL PIEDMONT":      "the north and central Piedmont",
    "OTHER REGIONAL OBSERVATIONS":     None,
}

# Default LWX station name expansions for Paul
# (abbreviated RWR names -> natural spoken names)
# Overridable via rwr.station_names in config.yaml
_DEFAULT_STATION_NAMES: Dict[str, str] = {
    "WASHINGTON NAT":   "Reagan National Airport",
    "DULLES":           "Dulles International Airport",
    "ANDREWS AFB":      "Joint Base Andrews",
    "FT BELVOIR":       "Fort Belvoir",
    "QUANTICO":         "Quantico",
    "COLLEGE PARK":     "College Park Airport",
    "LEESBURG":         "Leesburg",
    "MANASSAS":         "Manassas Airport",
    "GAITHERSBURG":     "Gaithersburg",
    "BWI AIRPORT":      "Baltimore-Washington International Airport",
    "BALT INNER HAR":   "Baltimore Inner Harbor",
    "MARTIN STATE":     "Martin State Airport",
    "ANNAPOLIS":        "Annapolis",
    "FORT MEADE":       "Fort Meade",
    "OCEAN CITY":       "Ocean City",
    "SALISBURY":        "Salisbury",
    "CAMBRIDGE":        "Cambridge",
    "EASTON":           "Easton",
    "PATUXENT RIVER":   "Patuxent River Naval Air Station",
    "ST INIGOES":       "Saint Inigoes",
    "FREDERICK":        "Frederick",
    "HAGERSTOWN APT":   "Hagerstown Regional Airport",
    "WESTMINSTER":      "Westminster",
    "OAKLAND":          "Oakland",
    "CUMBERLAND":       "Cumberland",
    "WINCHESTER":       "Winchester",
    "NEW MARKET":       "New Market",
    "STAUNTON":         "Staunton",
    "WAYNESBORO":       "Waynesboro",
    "MARTINSBURG":      "Martinsburg",
    "PETERSBURG":       "Petersburg West Virginia",
    "CHARLOTTESVILLE":  "Charlottesville",
    "CULPEPER":         "Culpeper",
    "ORANGE":           "Orange",
    "GORDONSVILLE":     "Gordonsville",
    "WARRENTON":        "Warrenton",
    "FREDERICKSBURG":   "Fredericksburg",
    "FREDERICKSBG":     "Fredericksburg",
    "NEW YORK CITY":    "New York City",
    "PHILADELPHIA":     "Philadelphia",
    "PITTSBURGH":       "Pittsburgh",
    "ROANOKE":          "Roanoke",
    "RICHMOND":         "Richmond",
    "RALEIGH":          "Raleigh",
    "CHARLOTTSVILL":    "Charlottesville",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RwrStation:
    name: str                         # cleaned, spoken-ready name
    name_raw: str                     # original from product (for anchor matching)
    sky_raw: str                      # raw sky code (e.g. "MOCLDY")
    temp_f: Optional[int]
    dewpoint_f: Optional[int]
    rh: Optional[int]
    wind_raw: str                     # raw wind field (e.g. "S7", "NW15G26", "CALM")
    pres_raw: str                     # raw pressure+trend (e.g. "29.95R")
    remarks: str
    is_nws: bool = True               # False if * prefix (non-NWS station)


@dataclass
class RwrSection:
    title: str                        # e.g. "WASHINGTON METRO"
    zone_codes: List[str]             # parsed from routing line
    stations: List[RwrStation]
    is_marine: bool = False


@dataclass
class RwrMarineStation:
    """One station row from the RWR MARINE OBSERVATIONS section."""
    name: str               # spoken-ready name (from name_map or title-cased raw)
    name_raw: str           # upper-case original from product, e.g. "THOMAS PT LIGHT"
    obs_time_utc: Optional[str]   # HHMM UTC string, e.g. "1800"
    air_temp_f: Optional[int]
    sea_temp_f: Optional[int]
    wind_dir_deg: Optional[int]   # degrees true (0-359)
    wind_spd_kt: Optional[int]
    wind_gust_kt: Optional[int]
    pres_mb: Optional[float]
    pres_trend: Optional[str]     # 'R'=rising / 'F'=falling / 'S'=steady / None


@dataclass
class RwrProduct:
    issuance_time_str: Optional[str]  # e.g. "1:00 AM Eastern Daylight Time"
    issuance_dt: Optional[dt.datetime]
    office: str
    sections: List[RwrSection]
    marine_stations: List[RwrMarineStation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_RWR_TIME_RE = re.compile(
    r'\b(\d{3,4})\s+(AM|PM)\s+([A-Z]{2,4})\s+'
    r'(?:MON|TUE|WED|THU|FRI|SAT|SUN)\s+'
    r'([A-Z]{3})\s+(\d{1,2})\s+(\d{4})'
)

_TZ_SPOKEN: Dict[str, str] = {
    "EST": "Eastern Standard Time",
    "EDT": "Eastern Daylight Time",
    "CST": "Central Standard Time",
    "CDT": "Central Daylight Time",
    "MST": "Mountain Standard Time",
    "MDT": "Mountain Daylight Time",
    "PST": "Pacific Standard Time",
    "PDT": "Pacific Daylight Time",
}

_MONTH_MAP: Dict[str, int] = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

def _parse_rwr_time(text: str) -> Tuple[Optional[str], Optional[dt.datetime]]:
    """
    Parse the issuance time line from RWR product text.
    Returns (spoken_time_str, datetime_utc_approx).
    e.g. "100 AM EDT SUN MAR 22 2026" -> ("1:00 AM Eastern Daylight Time", datetime(...))
    """
    m = _RWR_TIME_RE.search(text)
    if not m:
        return None, None
    hm_raw, ampm, tz_abbr, mon_str, day_str, year_str = (
        m.group(1), m.group(2), m.group(3),
        m.group(4), m.group(5), m.group(6),
    )
    # Parse hour/minute
    if len(hm_raw) == 3:
        h = int(hm_raw[0])
        mins = hm_raw[1:]
    else:
        h = int(hm_raw[:2])
        mins = hm_raw[2:]
    spoken = f"{h}:{mins} {ampm} {_TZ_SPOKEN.get(tz_abbr, tz_abbr)}"

    # Best-effort UTC datetime (for staleness check)
    try:
        month = _MONTH_MAP.get(mon_str.upper(), 1)
        day = int(day_str)
        year = int(year_str)
        h24 = h if ampm == "AM" else (h + 12 if h < 12 else 12)
        if ampm == "AM" and h == 12:
            h24 = 0
        # Approximate UTC (EDT = UTC-4, EST = UTC-5)
        tz_offset_h = {"EDT": -4, "EST": -5, "CDT": -5, "CST": -6,
                       "MDT": -6, "MST": -7, "PDT": -7, "PST": -8}.get(tz_abbr, 0)
        naive = dt.datetime(year, month, day, h24, int(mins))
        utc_approx = naive - dt.timedelta(hours=tz_offset_h)
        aware = utc_approx.replace(tzinfo=dt.timezone.utc)
        return spoken, aware
    except Exception:
        return spoken, None


_RWR_ROUTING_RE = re.compile(r'^[A-Z]{2}[Z0-9]\d{2,3}[>A-Z0-9-]+-\d{6}-\s*$')
_RWR_CITY_HEADER_RE = re.compile(r'^CITY\s+SKY/WX')
_RWR_DATA_SKIP_RE = re.compile(
    r'^(?:\$\$|={3,}|Note:|TC=|\*\s*=|STATION/POSITION|AIR SEA)'
)

# Marine observations section: "DDD/ SS/ GG" wind anchor and pressure
_MARINE_WIND_RE = re.compile(r'\b(\d{3})/\s*(\d{1,3})/\s*(\d{1,3})\b')
_MARINE_PRES_RE = re.compile(r'(\d{4}\.\d)([RFS]?)')
_MARINE_TIME_RE = re.compile(r'\b(\d{4})\b')


def _find_cols(header: str) -> Dict[str, int]:
    """
    Find column start positions from the CITY/SKY header line.
    Handles any WFO's column layout by searching for column names.
    """
    cols: Dict[str, int] = {"CITY": 0}
    for name in ("SKY/WX", "TMP", "DP", "RH", "WIND", "PRES", "REMARKS"):
        idx = header.find(name)
        if idx >= 0:
            cols[name] = idx
    return cols


def _slice_col(row: str, cols: Dict[str, int], key: str, next_key: str) -> str:
    start = cols.get(key, -1)
    if start < 0 or start >= len(row):
        return ""
    # End is start of next column (or end of string if next not found)
    end = cols.get(next_key)
    if end is None:
        end = len(row)
    return row[start:end].strip()


def _parse_rwr_wind(s: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Returns (direction_abbr, speed_mph, gust_mph).
    direction_abbr: compass abbr like 'S', 'NW', 'VRB', or None for calm/missing.
    """
    s = (s or "").strip().upper()
    if not s or s in ("MISG", "N/A", "M", ""):
        return None, None, None
    if s == "CALM":
        return None, 0, None
    m = re.match(r'^VRB(\d+)$', s)
    if m:
        return "VRB", int(m.group(1)), None
    m = re.match(r'^([NSEW]{1,3})(\d+)(?:G(\d+))?$', s)
    if m:
        gust = int(m.group(3)) if m.group(3) else None
        return m.group(1), int(m.group(2)), gust
    return None, None, None


def _parse_rwr_pressure(s: str) -> Tuple[Optional[float], Optional[str]]:
    """Returns (pressure_inhg, trend_char 'R'/'F'/'S'/None)."""
    s = (s or "").strip().upper()
    m = re.match(r'^(\d{2}\.\d{2})([RFS]?)', s)
    if m:
        return float(m.group(1)), (m.group(2) or None)
    return None, None


def _parse_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if s in ("N/A", "M", "MISG", "NOT AVBL", ""):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _clean_station_name(raw: str, name_map: Dict[str, str]) -> Tuple[str, str, bool]:
    """
    Returns (spoken_name, raw_name_upper, is_nws).
    NWS RWR marks non-NWS stations with a trailing * in the name field
    (e.g. "COLLEGE PARK*", "LEESBURG*"). Strip it and record is_nws=False.
    """
    s = raw.strip()
    is_nws = "*" not in s
    s = s.replace("*", "").strip()
    raw_upper = s.upper()
    spoken = name_map.get(raw_upper) or _DEFAULT_STATION_NAMES.get(raw_upper)
    if not spoken:
        # Title-case the abbreviation as-is (e.g. "FORT MEADE" -> "Fort Meade")
        spoken = raw_upper.title()
    return spoken, raw_upper, is_nws


def _parse_data_row(
    row: str,
    cols: Dict[str, int],
    name_map: Dict[str, str],
) -> Optional[RwrStation]:
    """Parse a single RWR data row using column positions from the header."""
    if not row.strip() or _RWR_DATA_SKIP_RE.match(row.strip()):
        return None
    # Need at least the station name region
    if len(row) < cols.get("TMP", 25):
        return None

    raw_name = _slice_col(row, cols, "CITY", "SKY/WX")
    if not raw_name or raw_name.upper() in ("CITY",):
        return None

    # Skip column-header lines
    if raw_name.upper().startswith("CITY"):
        return None

    sky = _slice_col(row, cols, "SKY/WX", "TMP")
    temp_str = _slice_col(row, cols, "TMP", "DP")
    dp_str = _slice_col(row, cols, "DP", "RH")
    rh_str = _slice_col(row, cols, "RH", "WIND")

    # Wind and pressure: NWS right-justifies the pressure value, so it can start
    # 1 char before the "PRES" header label position. Use a regex on the tail
    # of the row (from WIND column onward) to find the pressure value robustly.
    wind_start = cols.get("WIND", 35)
    tail = row[wind_start:].rstrip()

    pres_raw = ""
    remarks = ""
    wind_end_in_tail = len(tail)

    pres_m = re.search(r'(\d{2}\.\d{2}[RFS]?)', tail)
    if pres_m:
        pres_raw = pres_m.group(0)
        wind_end_in_tail = pres_m.start()
        remarks = tail[pres_m.end():].strip()

    wind_raw = tail[:wind_end_in_tail].strip()

    # Skip if name looks like a section header (no numeric data present)
    if re.match(r'^[A-Z\s]{15,}$', raw_name) and not pres_raw and not temp_str:
        return None

    spoken_name, raw_upper, is_nws = _clean_station_name(raw_name, name_map)
    temp_f = _parse_int(temp_str)
    dp_f = _parse_int(dp_str)
    rh = _parse_int(rh_str)

    return RwrStation(
        name=spoken_name,
        name_raw=raw_upper,
        sky_raw=(sky.upper() if sky else ""),
        temp_f=temp_f,
        dewpoint_f=dp_f,
        rh=rh,
        wind_raw=wind_raw,
        pres_raw=pres_raw,
        remarks=remarks,
        is_nws=is_nws,
    )


# ---------------------------------------------------------------------------
# Marine observations section parser (phase 2)
# ---------------------------------------------------------------------------

def _parse_marine_data_row(
    line: str,
    name_map: Dict[str, str],
) -> Optional[RwrMarineStation]:
    """
    Parse one RWR MARINE OBSERVATIONS data row.

    Format (station name fixed 16-char field, then regex-anchored fields):
      THOMAS PT LIGHT  1800   75     150/ 13/ 13 1020.7F
      TOLCHESTER       1730   73 55  220/  9/ 10 1019.8F
      PINEY POINT      1730          130/  7/  8   N/A

    Wind "DDD/ SS/ GG" is the reliable column anchor; everything before it
    contains the time and optional air/sea temp values.
    """
    line = (line or "").rstrip()
    if len(line) < 20:
        return None

    stripped = line.strip().upper()
    if not stripped or stripped == "$$":
        return None
    # Skip the 3-line inner header block
    if (stripped.startswith("STATION") or stripped.startswith("AIR ")
            or stripped.startswith("(UTC")):
        return None

    # Wind field is the reliable anchor: "DDD/ SS/ GG"
    wind_m = _MARINE_WIND_RE.search(line)
    if not wind_m:
        return None

    wind_dir_deg = int(wind_m.group(1))
    wind_spd_kt  = int(wind_m.group(2))
    wind_gust_kt = int(wind_m.group(3))

    # Station name: fixed 16-char field at start of line
    name_raw = line[:16].strip().upper()
    if not name_raw:
        return None

    # Time: first 4-digit HHMM in cols 16-26
    time_region = line[16:26] if len(line) > 16 else ""
    time_m = _MARINE_TIME_RE.search(time_region)
    obs_time_utc = time_m.group(1) if time_m else None

    # Temperatures: any 2-3 digit integers between end-of-time and start-of-wind
    time_end_in_line = (16 + time_m.end()) if time_m else 21
    temp_region = line[time_end_in_line:wind_m.start()]
    temp_nums = re.findall(r'\d{2,3}', temp_region)
    air_temp_f: Optional[int] = int(temp_nums[0]) if len(temp_nums) >= 1 else None
    sea_temp_f: Optional[int] = int(temp_nums[1]) if len(temp_nums) >= 2 else None

    # Pressure: after wind field; N/A is silently absent
    after_wind = line[wind_m.end():]
    pres_m = _MARINE_PRES_RE.search(after_wind)
    pres_mb:    Optional[float] = float(pres_m.group(1)) if pres_m else None
    pres_trend: Optional[str]  = (pres_m.group(2) or None) if pres_m else None

    spoken = name_map.get(name_raw) or name_raw.title()

    return RwrMarineStation(
        name=spoken,
        name_raw=name_raw,
        obs_time_utc=obs_time_utc,
        air_temp_f=air_temp_f,
        sea_temp_f=sea_temp_f,
        wind_dir_deg=wind_dir_deg,
        wind_spd_kt=wind_spd_kt,
        wind_gust_kt=wind_gust_kt,
        pres_mb=pres_mb,
        pres_trend=pres_trend,
    )


def _parse_marine_section(
    lines: List[str],
    i: int,
    n: int,
    name_map: Dict[str, str],
) -> Tuple[List[RwrMarineStation], int]:
    """
    Consume a marine-obs section from lines[i:] up to and including $$.
    Returns (parsed_stations, new_i).
    """
    stations: List[RwrMarineStation] = []
    while i < n:
        ln = lines[i]
        if ln.strip() == "$$":
            i += 1
            break
        st = _parse_marine_data_row(ln, name_map)
        if st:
            stations.append(st)
        i += 1
    return stations, i


# ---------------------------------------------------------------------------
# Top-level RWR product parser
# ---------------------------------------------------------------------------

def parse_rwr(text: str, name_map: Optional[Dict[str, str]] = None) -> Optional[RwrProduct]:
    """
    Parse a raw NWS RWR product text string into a structured RwrProduct.
    Returns None if the text doesn't look like an RWR product.
    name_map: optional {RAW_UPPER: spoken_name} overrides for station names.
    """
    nm = {k.upper(): v for k, v in (name_map or {}).items()}
    lines = (text or "").replace("\r", "").splitlines()

    # Extract issuance time from product header
    issuance_spoken, issuance_dt = _parse_rwr_time(text)

    # Find issuing office (line after WMO header: "RWRLWX" -> "LWX")
    office = ""
    for ln in lines[:10]:
        m = re.match(r'^RWR([A-Z]{2,4})\s*$', ln.strip())
        if m:
            office = m.group(1)
            break

    sections: List[RwrSection] = []
    all_marine: List[RwrMarineStation] = []
    i = 0
    n = len(lines)

    while i < n:
        ln = lines[i].rstrip()

        # Detect section routing line
        if _RWR_ROUTING_RE.match(ln.strip()):
            zone_codes = re.findall(r'[A-Z]{2}[Z0-9]\d{2,3}', ln)
            i += 1

            # Next non-empty line = section title
            section_title = ""
            while i < n and not lines[i].strip():
                i += 1
            if i < n:
                section_title = lines[i].strip()
                i += 1

            # Is this a marine observations section?
            is_marine = bool(re.search(r'MARINE|BUOY|OFFSHORE', section_title, re.IGNORECASE))

            if is_marine:
                # Marine sections use a completely different fixed-width format.
                # Hand off to the dedicated parser (phase 2).
                parsed, i = _parse_marine_section(lines, i, n, nm)
                all_marine.extend(parsed)
                continue

            # Find the CITY/SKY header line (land sections only)
            cols: Dict[str, int] = {}
            while i < n:
                ln2 = lines[i]
                if _RWR_CITY_HEADER_RE.match(ln2):
                    cols = _find_cols(ln2)
                    i += 1
                    break
                if ln2.strip() == "$$":
                    break
                i += 1

            if not cols:
                # No header found, skip section
                while i < n and lines[i].strip() != "$$":
                    i += 1
                i += 1  # skip $$
                continue

            # Parse data rows until $$
            stations: List[RwrStation] = []
            while i < n:
                ln2 = lines[i]
                if ln2.strip() == "$$":
                    i += 1
                    break
                station = _parse_data_row(ln2, cols, nm)
                if station:
                    stations.append(station)
                i += 1

            if stations or is_marine:
                sections.append(RwrSection(
                    title=section_title,
                    zone_codes=zone_codes,
                    stations=stations,
                    is_marine=is_marine,
                ))
            continue

        i += 1

    if not sections and not all_marine:
        return None

    return RwrProduct(
        issuance_time_str=issuance_spoken,
        issuance_dt=issuance_dt,
        office=office,
        sections=sections,
        marine_stations=all_marine,
    )


# ---------------------------------------------------------------------------
# ObsPressureCache
# ---------------------------------------------------------------------------

class ObsPressureCache:
    """
    Persistent per-station pressure history for trend derivation.
    Survives service restarts (JSON file in work_dir).
    """

    def __init__(
        self,
        path: str,
        max_hours: float = 3.0,
        trend_threshold_inhg: float = 0.02,
    ) -> None:
        self._path = Path(path)
        self._max_secs = max_hours * 3600
        self._threshold = trend_threshold_inhg
        self._data: Dict[str, List[Dict]] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self._path.exists():
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            self._data = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            pass

    def _now_iso(self) -> str:
        return dt.datetime.now(tz=dt.timezone.utc).isoformat()

    def _prune(self, station_id: str) -> None:
        cutoff = (
            dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(seconds=self._max_secs)
        ).isoformat()
        if station_id in self._data:
            self._data[station_id] = [
                e for e in self._data[station_id] if e.get("ts", "") >= cutoff
            ]

    def update(self, station_id: str, pressure_inhg: float) -> None:
        """Record a new pressure reading for the given station."""
        sid = station_id.strip().upper()
        if sid not in self._data:
            self._data[sid] = []
        self._data[sid].append({"ts": self._now_iso(), "p": round(pressure_inhg, 4)})
        self._prune(sid)
        self._save()

    def get_trend(self, station_id: str, current_inhg: float) -> Optional[str]:
        """
        Derive 'rising', 'falling', or 'steady' by comparing current pressure
        to the oldest cached reading within the window.
        Returns None if insufficient history (< 2 readings).
        """
        sid = station_id.strip().upper()
        self._prune(sid)
        entries = self._data.get(sid, [])
        if len(entries) < 2:
            return None
        # Compare to the oldest in-window reading
        delta = current_inhg - entries[0]["p"]
        if delta > self._threshold:
            return "rising"
        if delta < -self._threshold:
            return "falling"
        return "steady"


# ---------------------------------------------------------------------------
# ASOS -> RwrStation adapter
# ---------------------------------------------------------------------------

def asos_to_rwr_station(
    station_id: str,
    props: Dict[str, Any],
    name_map: Optional[Dict[str, str]] = None,
    station_name_override: Optional[str] = None,
    cache: Optional[ObsPressureCache] = None,
) -> Optional[RwrStation]:
    """
    Convert an api.weather.gov observations/latest response dict to RwrStation.

    Temperature, dew point are in Celsius (converted to F).
    Wind speed is in m/s (converted to mph).
    Pressure is in Pascals (converted to inHg).
    """
    if not props:
        return None

    sid = station_id.strip().upper()

    # Determine spoken station name
    nm = {k.upper(): v for k, v in (name_map or {}).items()}
    if station_name_override:
        spoken_name = station_name_override
    else:
        spoken_name = nm.get(sid) or _DEFAULT_STATION_NAMES.get(sid, sid.title())

    # Sky / text description
    sky_text = (props.get("textDescription") or "").strip().lower()

    # Temperature (Celsius -> F)
    temp_f: Optional[int] = None
    temp_c = (props.get("temperature") or {}).get("value")
    if isinstance(temp_c, (int, float)):
        temp_f = round(temp_c * 9 / 5 + 32)

    # Dew point (Celsius -> F)
    dp_f: Optional[int] = None
    dp_c = (props.get("dewpoint") or {}).get("value")
    if isinstance(dp_c, (int, float)):
        dp_f = round(dp_c * 9 / 5 + 32)

    # Relative humidity
    rh: Optional[int] = None
    rh_v = (props.get("relativeHumidity") or {}).get("value")
    if isinstance(rh_v, (int, float)):
        rh = round(rh_v)

    # Wind direction (degrees -> compass abbreviation)
    wind_raw = ""
    wind_dir_deg = (props.get("windDirection") or {}).get("value")
    wind_speed_mps = (props.get("windSpeed") or {}).get("value")
    wind_gust_mps = (props.get("windGust") or {}).get("value")

    if isinstance(wind_speed_mps, (int, float)):
        speed_mph = round(wind_speed_mps * 2.23694)
        if speed_mph == 0:
            wind_raw = "CALM"
        elif isinstance(wind_dir_deg, (int, float)):
            compass = _degrees_to_compass(wind_dir_deg)
            wind_raw = f"{compass}{speed_mph}"
            if isinstance(wind_gust_mps, (int, float)):
                gust_mph = round(wind_gust_mps * 2.23694)
                if gust_mph > speed_mph:
                    wind_raw += f"G{gust_mph}"
        else:
            wind_raw = f"VRB{speed_mph}"

    # Pressure (Pascals -> inHg)
    pres_raw = ""
    pres_pa = (props.get("seaLevelPressure") or {}).get("value")
    if isinstance(pres_pa, (int, float)):
        pres_inhg = pres_pa / 3386.39
        # Derive trend from cache if available
        trend_char = ""
        if cache is not None:
            trend = cache.get_trend(sid, pres_inhg)
            trend_char = {"rising": "R", "falling": "F", "steady": "S"}.get(trend or "", "")
            cache.update(sid, pres_inhg)
        pres_raw = f"{pres_inhg:.2f}{trend_char}"

    return RwrStation(
        name=spoken_name,
        name_raw=sid,
        sky_raw=sky_text.upper().replace(" ", ""),  # store normalised for lookup
        temp_f=temp_f,
        dewpoint_f=dp_f,
        rh=rh,
        wind_raw=wind_raw,
        pres_raw=pres_raw,
        remarks="",
        is_nws=True,
    )


# ---------------------------------------------------------------------------
# Spoken formatters
# ---------------------------------------------------------------------------

def _sky_spoken(sky_raw: str, sky_text_fallback: str = "") -> str:
    """
    Return spoken sky condition string.
    sky_raw: RWR sky code (e.g. 'MOCLDY') or normalised ASOS code.
    sky_text_fallback: ASOS textDescription lower-case (used if code lookup fails).
    """
    code = (sky_raw or "").strip().upper()
    spoken = _SKY_SPOKEN.get(code)
    if spoken:
        return spoken
    # Try the ASOS text description directly (already English)
    fb = (sky_text_fallback or "").strip().lower()
    if fb and fb not in ("n/a", "not available", ""):
        return fb
    # Last resort: title-case the raw code
    if code and code not in ("N/A", ""):
        return code.title()
    return ""


def _format_wind_spoken(wind_raw: str) -> Optional[str]:
    """
    Returns spoken wind phrase like 'Winds were south at 7 miles an hour'
    or 'Winds were northwest at 15 miles an hour, with gusts to 26',
    or 'Winds were calm', or None if data missing.
    """
    dir_abbr, speed, gust = _parse_rwr_wind(wind_raw)
    if speed is None and dir_abbr is None:
        return None
    if speed == 0:
        return "Winds were calm"
    if dir_abbr == "VRB":
        base = f"Winds were variable at {speed} miles an hour"
    elif dir_abbr:
        compass = _COMPASS_SPOKEN.get(dir_abbr, dir_abbr.lower())
        base = f"Winds were {compass} at {speed} miles an hour"
    else:
        return None
    if gust and gust > (speed or 0):
        base += f", with gusts to {gust}"
    return base


def format_station_full(
    station: RwrStation,
    trend_override: Optional[str] = None,
) -> str:
    """
    Full NWR-style spoken observation for an anchor station.
    Matches LWX BMH format: sky / temp+dp / humidity / wind+pressure sentence.
    trend_override: 'rising'/'falling'/'steady' to override product trend char.
    """
    parts: List[str] = []

    # Sky condition
    sky = _sky_spoken(station.sky_raw)
    if sky:
        parts.append(f"At {station.name}, {sky}.")
    else:
        parts.append(f"At {station.name}.")

    # Temperature + dew point
    if station.temp_f is not None:
        temp_line = f"The temperature was {station.temp_f} degrees"
        if station.dewpoint_f is not None:
            temp_line += f", dew point {station.dewpoint_f}"
        parts.append(temp_line + ".")

    # Relative humidity
    if station.rh is not None:
        parts.append(f"Humidity was {station.rh} percent.")

    # Wind + pressure (joined in one sentence like LWX BMH)
    wind_phrase = _format_wind_spoken(station.wind_raw)
    pres_inhg, trend_char = _parse_rwr_pressure(station.pres_raw)
    trend = trend_override or _TREND_SPOKEN.get(trend_char or "", None)

    pres_phrase: Optional[str] = None
    if pres_inhg is not None:
        pres_phrase = f"the barometric pressure was {pres_inhg:.2f} inches"
        if trend:
            pres_phrase += f" and {trend}"

    if wind_phrase and pres_phrase:
        parts.append(f"{wind_phrase} and {pres_phrase}.")
    elif wind_phrase:
        parts.append(f"{wind_phrase}.")
    elif pres_phrase:
        cap = pres_phrase[0].upper() + pres_phrase[1:]
        parts.append(f"{cap}.")

    return " ".join(parts)


def format_station_compact(station: RwrStation) -> str:
    """
    Compact spoken observation: 'At [name], [sky], [temp] degrees.'
    Used for surrounding-area stations after the anchor.
    """
    sky = _sky_spoken(station.sky_raw)
    if station.temp_f is not None and sky:
        return f"At {station.name}, {sky}, {station.temp_f} degrees."
    elif station.temp_f is not None:
        return f"At {station.name}, {station.temp_f} degrees."
    elif sky:
        return f"At {station.name}, {sky}."
    else:
        return f"At {station.name}."


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_rwr_obs_text(
    product: RwrProduct,
    anchor_names: List[str],
    max_compact_per_section: int = 8,
    intro_prefix: str = "And now for the current observed weather conditions in our area",
    cache: Optional[ObsPressureCache] = None,
    skip_marine: bool = True,
) -> str:
    """
    Assemble NWR-style spoken obs segment from a parsed RwrProduct.

    anchor_names: list of raw station name_raw values (upper-case) that get
                  full-detail treatment. Empty = auto-pick first station in
                  first non-marine section that has temp + pressure.
    max_compact_per_section: max compact stations to read per section.
    intro_prefix: spoken intro before the time/anchor. The time and first
                  station flow naturally from this.
    cache: ObsPressureCache for trend override when product trend char is missing.
    skip_marine: skip marine observation sections (phase 2).
    """
    if not product or not product.sections:
        return ""

    anchors = {n.strip().upper() for n in anchor_names if n.strip()}

    # Auto-derive anchor if none configured
    auto_anchor: Optional[str] = None
    if not anchors:
        for sec in product.sections:
            if sec.is_marine and skip_marine:
                continue
            for st in sec.stations:
                if st.temp_f is not None and st.pres_raw:
                    auto_anchor = st.name_raw
                    anchors = {auto_anchor}
                    break
            if auto_anchor:
                break

    # Build spoken output
    spoken_parts: List[str] = []

    # Intro with time
    time_str = product.issuance_time_str or ""
    if time_str:
        spoken_parts.append(f"{intro_prefix} as of {time_str}.")
    else:
        spoken_parts.append(f"{intro_prefix}.")

    # First pass: anchor station(s) — full detail
    anchor_done: set = set()
    for sec in product.sections:
        if sec.is_marine and skip_marine:
            continue
        for st in sec.stations:
            if st.name_raw in anchors:
                trend_override = None
                if cache is not None:
                    pres_inhg, trend_char = _parse_rwr_pressure(st.pres_raw)
                    if pres_inhg is not None and not trend_char:
                        trend_override = cache.get_trend(st.name_raw, pres_inhg)
                spoken_parts.append(format_station_full(st, trend_override=trend_override))
                anchor_done.add(st.name_raw)
        if anchor_done:
            break  # Anchor section done; compact starts

    if not anchor_done:
        # No anchor found at all — nothing to read
        return ""

    # Second pass: compact surrounding stations, grouped by section
    surroundings_intro_done = False

    for sec in product.sections:
        if sec.is_marine and skip_marine:
            continue

        # Collect compact stations for this section (skip anchors already read)
        compact = [
            st for st in sec.stations
            if st.name_raw not in anchor_done
            and st.temp_f is not None  # skip stations with no usable data
        ][:max_compact_per_section]

        if not compact:
            continue

        # Section intro
        if not surroundings_intro_done:
            spoken_parts.append("Now for some observations from the surrounding area.")
            surroundings_intro_done = True

        # Named section header (if not the first/anchor section)
        section_spoken = _SECTION_SPOKEN.get(sec.title.upper())
        if section_spoken is not None:
            spoken_parts.append(f"In {section_spoken}.")
        else:
            spoken_parts.append("Elsewhere in the region.")

        # Compact station list
        for st in compact:
            spoken_parts.append(format_station_compact(st))

    return " ".join(spoken_parts)


def build_asos_obs_text(
    stations: List[Tuple[str, Dict[str, Any]]],
    anchor_id: str,
    max_compact: int = 8,
    intro_prefix: str = "And now for the current observed weather conditions in our area",
    cache: Optional[ObsPressureCache] = None,
    name_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build NWR-style spoken obs from raw ASOS observation dicts.
    Used as fallback when RWR is stale or unavailable.

    stations: list of (station_id, props_dict) pairs, in priority order.
    anchor_id: station ID for the full-detail anchor (first in list if empty).
    """
    if not stations:
        return ""

    anchor = (anchor_id or "").strip().upper() or stations[0][0].upper()

    rwr_stations: List[Tuple[str, RwrStation]] = []
    for sid, props in stations:
        st = asos_to_rwr_station(sid, props, name_map=name_map, cache=cache)
        if st:
            rwr_stations.append((sid.upper(), st))

    if not rwr_stations:
        return ""

    parts: List[str] = [f"{intro_prefix}."]

    # Anchor: full detail
    anchor_done = False
    compact_stns: List[RwrStation] = []
    for sid, st in rwr_stations:
        if sid == anchor and not anchor_done:
            trend_override: Optional[str] = None
            if cache:
                pres_inhg, trend_char = _parse_rwr_pressure(st.pres_raw)
                if pres_inhg is not None and not trend_char:
                    trend_override = cache.get_trend(sid, pres_inhg)
            parts.append(format_station_full(st, trend_override=trend_override))
            anchor_done = True
        elif st.temp_f is not None:
            compact_stns.append(st)

    if not anchor_done:
        # Anchor not found; use first available
        _, first = rwr_stations[0]
        parts.append(format_station_full(first))
        compact_stns = [st for _, st in rwr_stations[1:] if st.temp_f is not None]

    # Compact surrounding stations
    if compact_stns:
        parts.append("Now for some observations from the surrounding area.")
        for st in compact_stns[:max_compact]:
            parts.append(format_station_compact(st))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Marine observations text builder
# ---------------------------------------------------------------------------

_MARINE_PRES_TREND_SPOKEN: Dict[str, str] = {
    "R": "rising",
    "F": "falling",
    "S": "steady",
}


def _format_marine_wind(st: RwrMarineStation) -> Optional[str]:
    """Return NWR-style wind phrase, or None if no wind data."""
    if st.wind_dir_deg is None or st.wind_spd_kt is None:
        return None
    compass = _degrees_to_compass(float(st.wind_dir_deg))
    compass_spoken = _COMPASS_SPOKEN.get(compass, compass.lower())
    if st.wind_spd_kt == 0:
        return "the wind was calm"
    if st.wind_gust_kt is not None and st.wind_gust_kt > st.wind_spd_kt:
        return (
            f"the wind was {compass_spoken} at {st.wind_spd_kt} knots,"
            f" gusting to {st.wind_gust_kt}"
        )
    return f"the wind was {compass_spoken} at {st.wind_spd_kt} knots"


def _format_marine_station_full(st: RwrMarineStation) -> str:
    """
    Full NWR-style marine obs for an anchor station.
    Includes wind, both temperatures where available, and pressure.
    """
    bits: List[str] = []

    wind = _format_marine_wind(st)
    if wind:
        bits.append(f"At {st.name}, {wind}.")
    else:
        bits.append(f"At {st.name}.")

    # Temperatures — speak both when available, single "temperature" when only one
    if st.air_temp_f is not None and st.sea_temp_f is not None:
        bits.append(
            f"The air temperature was {st.air_temp_f}"
            f" and the water temperature was {st.sea_temp_f}."
        )
    elif st.air_temp_f is not None:
        bits.append(f"The temperature was {st.air_temp_f}.")
    elif st.sea_temp_f is not None:
        bits.append(f"The water temperature was {st.sea_temp_f}.")

    # Pressure with trend — anchor stations only
    if st.pres_mb is not None:
        trend_word = _MARINE_PRES_TREND_SPOKEN.get((st.pres_trend or "").upper(), "")
        pres_bit = f"Barometric pressure {st.pres_mb:.1f} millibars"
        if trend_word:
            pres_bit += f" and {trend_word}"
        bits.append(pres_bit + ".")

    return " ".join(bits)


def _format_marine_station_compact(st: RwrMarineStation) -> str:
    """
    Compact NWR-style marine obs for surrounding stations.
    Wind and temperatures only — no pressure.
    """
    wind = _format_marine_wind(st)

    if wind:
        intro = f"At {st.name}, {wind}."
    else:
        intro = f"At {st.name}."

    if st.air_temp_f is not None and st.sea_temp_f is not None:
        temp = (
            f"The air temperature was {st.air_temp_f}"
            f" and the water temperature was {st.sea_temp_f}."
        )
    elif st.air_temp_f is not None:
        temp = f"The temperature was {st.air_temp_f}."
    elif st.sea_temp_f is not None:
        temp = f"The water temperature was {st.sea_temp_f}."
    else:
        temp = ""

    return f"{intro} {temp}".strip() if temp else intro


def build_marine_obs_text(
    product: RwrProduct,
    max_stations: int = 0,
    anchor_names: Optional[List[str]] = None,
    intro_prefix: str = "Marine observations for the service area",
    name_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Assemble NWR-style spoken marine observations segment from
    RwrProduct.marine_stations (populated by parse_rwr's phase-2 marine parser).

    Anchor stations (named in anchor_names) get full detail: wind, both
    temperatures, and barometric pressure.  All other stations get compact
    treatment: wind and temperatures only, matching real NWR marine obs style.

    anchor_names: raw upper-case station names to speak first with full detail,
                  e.g. ["THOMAS PT LIGHT"].  Others follow in product order.
    max_stations: cap on total stations spoken (0 = all).
    name_map:     optional spoken-name overrides applied on top of any names
                  already baked in by parse_rwr.
    """
    if not product or not product.marine_stations:
        return None

    stations = list(product.marine_stations)
    nm = {k.upper(): v for k, v in (name_map or {}).items()}
    anchors = {a.strip().upper() for a in (anchor_names or []) if a.strip()}

    # Apply any name-map overrides from this call
    resolved: List[RwrMarineStation] = []
    for st in stations:
        override = nm.get(st.name_raw)
        if override and override != st.name:
            resolved.append(RwrMarineStation(
                name=override,
                name_raw=st.name_raw,
                obs_time_utc=st.obs_time_utc,
                air_temp_f=st.air_temp_f,
                sea_temp_f=st.sea_temp_f,
                wind_dir_deg=st.wind_dir_deg,
                wind_spd_kt=st.wind_spd_kt,
                wind_gust_kt=st.wind_gust_kt,
                pres_mb=st.pres_mb,
                pres_trend=st.pres_trend,
            ))
        else:
            resolved.append(st)

    # Anchors first (full detail), then the rest in product order (compact)
    anchor_list = [s for s in resolved if s.name_raw in anchors]
    compact_list = [s for s in resolved if s.name_raw not in anchors]

    ordered = anchor_list + compact_list
    cap = max_stations if max_stations > 0 else len(ordered)
    ordered = ordered[:cap]

    if not ordered:
        return None

    parts: List[str] = []

    time_str = product.issuance_time_str or ""
    if time_str:
        parts.append(f"{intro_prefix} as of {time_str}.")
    else:
        parts.append(f"{intro_prefix}.")

    anchor_done: set = set()
    for st in ordered:
        if st.name_raw in anchors and st.name_raw not in anchor_done:
            parts.append(_format_marine_station_full(st))
            anchor_done.add(st.name_raw)
        else:
            parts.append(_format_marine_station_compact(st))

    return " ".join(parts)
