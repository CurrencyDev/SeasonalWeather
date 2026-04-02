"""
ugc.py — Universal Geographic Code (UGC) parsing library for SeasonalWeather.

Single source of truth for:
  - Locating and extracting UGC blocks from NWS text products
  - Expanding UGC zone tokens with prefix carry and NNN>NNN range support
  - Resolving UGC expiration timestamps (DDHHMM UTC) to aware datetimes
  - State abbreviation ↔ FIPS mappings (eliminates duplication across modules)
  - Direct county-zone → SAME conversion without network calls
  - Parsing NWS ZoneCounty .dbx crosswalk files (forecast zone → county SAME)
  - Parsing NWS mareas*.txt crosswalk files (marine zone → coastal SAME)

What a UGC block carries:
  - Zone / county target list (the product's geographic footprint)
  - Expiration time (DDHHMM UTC, month-ambiguous → resolved by this module)

Products that carry UGC but may lack VTEC:
  - Small Craft Advisories (SC.Y in modern products)
  - Gale Watches / Gale Warnings (GL.A, GL.W)
  - Hazardous Seas Warnings (HF.W, HF.A)
  - Marine Weather Statements (MWS)
  - Special Marine Warnings (MA.W)
  - Fire Weather products (FW.W, FW.A)
  All of these carry a valid UGC expiry that NWR stations use to gate
  repeat broadcasts — capturing it here enables SeasonalWeather to do the same.

Design rules (do not break these):
  - Zero imports from the SeasonalWeather package. Pure stdlib only.
  - No side effects. All functions are pure / stateless.
  - parse_ugc_block() is the primary entry point; others are composable primitives.
  - Zone → SAME resolution requiring network calls stays in Orchestrator.
  - same_from_county_zone() handles direct (no-network) XXC### conversions only.

Analogous role in real NWR infrastructure:
  - UGC targeting layer ~ NWRWAVES zone filter
  - The Orchestrator (main.py) ~ BMH: receives the zone list, decides on-air action

Usage (library):
    from seasonalweather.same.ugc import parse_ugc_block, extract_ugc_zones

    block = parse_ugc_block(product_text)
    if block:
        print(block.zones)        # ['VAC059', 'VAC153', 'VAC683', 'VAC685']
        print(block.expires_utc)  # datetime(2026, 4, 1, 21, 30, tzinfo=UTC)

Usage (CLI):
    python -m seasonalweather.same.ugc                  # run built-in test suite
    python -m seasonalweather.same.ugc "VAC059-153-012130-"  # parse a block
    python -m seasonalweather.same.ugc MDC031           # zone → SAME lookup
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Module-level compiled regexes
# ---------------------------------------------------------------------------

# Matches the terminal DDHHMM expiry token at the end of a UGC line.
_UGC_EXPIRES_RE = re.compile(r"\b\d{6}-\s*$")

# Matches any UGC zone token anywhere in a string.
# Covers: XXC### (county), XXZ### (forecast zone), XXX### (marine/fire, 3-letter)
# Also matches NNN>NNN range notation.
_UGC_ANY_CODE_RE = re.compile(
    r"\b[A-Z]{2}[CZ]\d{3}(?:>\d{3})?\b"
    r"|\b[A-Z]{2}Z\d{3}(?:>\d{3})?\b"
    r"|\b[A-Z]{3}\d{3}(?:>\d{3})?\b"
)

# Parses a single UGC zone token: optional prefix + 3-digit num + optional range.
# Handles county (XXC###), forecast zone (XXZ###), and 3-letter marine (ANZ###, GMZ###).
_UGC_TOKEN_RE = re.compile(
    r"^(?P<pfx>[A-Z]{2,3}[CZ]?)?"
    r"(?P<num>\d{3})"
    r"(?:>(?P<end>\d{3}))?$"
)

# County zone ID: exactly two alpha state + "C" + three digits.
_COUNTY_ZONE_RE = re.compile(r"^([A-Z]{2})C(\d{3})$")


# ---------------------------------------------------------------------------
# FIPS / state maps  — canonical home, replaces duplicates in main.py + cap_nws.py
# ---------------------------------------------------------------------------

STATE_ABBR_TO_FIPS: dict[str, str] = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
    "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56",
    # Territories
    "AS": "60", "GU": "66", "MP": "69", "PR": "72", "UM": "74", "VI": "78",
}

FIPS_TO_STATE_ABBR: dict[str, str] = {v: k for k, v in STATE_ABBR_TO_FIPS.items()}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class UGCBlock:
    """
    Parsed result of a UGC targeting block from an NWS text product.

    Attributes:
        zones:           Expanded zone ID list, e.g. ['VAC059', 'VAC153', 'VAC683'].
        expires_ddhhmm:  Raw expiry token from the product, e.g. '012130'.
                         Empty string if no expiry token was found.
        expires_utc:     Resolved aware UTC datetime, or None if resolution
                         failed (month ambiguity could not be resolved, token
                         absent, or date is impossible like Feb 30).
        raw:             Flattened (no-whitespace) source UGC block string.
    """
    zones: list[str]
    expires_ddhhmm: str
    expires_utc: "datetime | None"
    raw: str


# ---------------------------------------------------------------------------
# Public API — parsing
# ---------------------------------------------------------------------------

def extract_ugc_block(text: str) -> str:
    """
    Locate and extract the UGC targeting block from an NWS text product.

    The UGC block is the hyphen-delimited zone list near the top of the
    product, ending with a 6-digit DDHHMM expiration token. It may span
    multiple continuation lines, each ending with a hyphen.

    Returns a *flattened* (no whitespace) string, or "" if not found.

    Example input:
        WUUS51 KLWX 012043
        SVRLWX
        VAC059-153-683-685-012130-

    Returns: "VAC059-153-683-685-012130-"
    """
    if not text:
        return ""
    lines = text.splitlines()
    # UGC is always in the header region; no need to scan the full product.
    scan = lines[:120]

    end_idx: int | None = None
    for i, ln in enumerate(scan):
        s = (ln or "").strip()
        if not s:
            continue
        # The terminal UGC line has both a zone code and a DDHHMM expiry token.
        if _UGC_EXPIRES_RE.search(s) and _UGC_ANY_CODE_RE.search(s):
            end_idx = i
            break

    if end_idx is None:
        return ""

    # Walk backwards to collect continuation lines (they end with "-" and
    # contain zone codes — but NOT just an expiry token).
    start_idx = end_idx
    while start_idx > 0:
        prev = (scan[start_idx - 1] or "").strip()
        if not prev:
            break
        if prev.endswith("-") and _UGC_ANY_CODE_RE.search(prev):
            start_idx -= 1
            continue
        break

    block = "".join((scan[j] or "").strip() for j in range(start_idx, end_idx + 1))
    block = re.sub(r"\s+", "", block)
    return block.strip()


def expand_ugc_tokens(ugc_block: str) -> list[str]:
    """
    Expand a flattened UGC block string into a deduplicated list of zone IDs.

    Handles:
      - State/type prefix carry across tokens (VAC059-153 → ['VAC059', 'VAC153'])
      - NNN>NNN range notation (MDZ001>005 → ['MDZ001', ..., 'MDZ005'])
      - County zones (MDC031), forecast zones (MDZ008), marine zones (ANZ530)
      - Multi-state products with prefix resets (MDC031-VAC059-...)

    The trailing DDHHMM expiry token is stripped before expansion.
    Returns [] if the block contains no zone codes.
    """
    if not ugc_block:
        return []

    parts = [p.strip().strip(".") for p in ugc_block.split("-") if p.strip().strip(".")]

    # Drop the trailing 6-digit expiry token.
    if parts and re.fullmatch(r"\d{6}", parts[-1]):
        parts = parts[:-1]

    out: list[str] = []
    seen: set[str] = set()
    prefix: str | None = None  # carried state+type prefix, e.g. "VAC", "MDZ", "ANZ"

    def _emit(n: int) -> None:
        if prefix is None:
            return
        z = f"{prefix}{n:03d}"
        if z not in seen:
            seen.add(z)
            out.append(z)

    for raw in parts:
        tok = raw.strip().rstrip(",;").strip(".")
        if not tok:
            continue

        m = _UGC_TOKEN_RE.fullmatch(tok)
        if not m:
            continue  # WMO headers, stray punctuation, etc. — skip silently

        pfx = (m.group("pfx") or "").upper()
        num_s = m.group("num")
        end_s = m.group("end")

        # A new prefix resets the carry (handles multi-state products).
        if pfx and any(ch.isalpha() for ch in pfx):
            prefix = pfx

        if prefix is None:
            continue  # number-only token with no prior prefix — malformed, skip

        num = int(num_s)
        if end_s:
            end_n = int(end_s)
            step = 1 if end_n >= num else -1
            for n in range(num, end_n + step, step):
                _emit(n)
        else:
            _emit(num)

    return out


def extract_ugc_zones(text: str) -> list[str]:
    """
    Convenience wrapper: extract_ugc_block() → expand_ugc_tokens().

    Returns the expanded zone list, or [] if no UGC block was found.
    This is a pure text operation — no network calls.
    """
    return expand_ugc_tokens(extract_ugc_block(text))


def resolve_ugc_expires(ddhhmm: str, *, reference_utc: "datetime | None" = None) -> "datetime | None":
    """
    Convert a UGC DDHHMM expiration token to an aware UTC datetime.

    The month is implicit in UGC. NWS products expire within a bounded window
    (typically ≤7 days), so we resolve by finding the candidate month that
    produces a datetime closest to reference_utc, preferring a future time.

    Args:
        ddhhmm:         6-digit string, e.g. "012130" → day 01, 21:30 UTC.
        reference_utc:  Anchor time; defaults to datetime.now(UTC) if not supplied.

    Returns:
        An aware UTC datetime, or None if the token is malformed, empty,
        or produces only impossible dates (e.g. "300000" in February).
    """
    s = "".join(ch for ch in (ddhhmm or "").strip() if ch.isdigit())
    if len(s) != 6:
        return None
    try:
        dd = int(s[0:2])
        hh = int(s[2:4])
        mm_min = int(s[4:6])
    except ValueError:
        return None

    if not (1 <= dd <= 31 and 0 <= hh <= 23 and 0 <= mm_min <= 59):
        return None

    ref = reference_utc if reference_utc is not None else datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)

    # Try this month, next month, and previous month to handle boundary conditions.
    candidates: list[datetime] = []
    for delta in (0, 1, -1):
        year = ref.year
        month = ref.month + delta
        if month > 12:
            month -= 12
            year += 1
        elif month < 1:
            month += 12
            year -= 1
        try:
            candidates.append(datetime(year, month, dd, hh, mm_min, tzinfo=timezone.utc))
        except ValueError:
            pass  # e.g. Feb 30 — impossible date, skip

    if not candidates:
        return None

    # Pick the candidate closest to reference in absolute time.
    # This correctly handles:
    #   - same-day future (product just issued, expires in hours)
    #   - same-day past  (product just expired minutes ago)
    #   - month boundary (day 01 token at end of month → next month)
    # "Prefer any future" is wrong: May 1 is not a better answer than April 1
    # when ref is April 1 22:00 and the product expired at April 1 21:30.
    return min(candidates, key=lambda c: abs((c - ref).total_seconds()))


def parse_ugc_block(text: str, *, reference_utc: "datetime | None" = None) -> "UGCBlock | None":
    """
    Full pipeline: locate UGC block → expand zones → resolve expiry.

    This is the primary entry point for callers that want structured output.
    Returns None if no valid UGC block is found in the text.

    Args:
        text:           Raw NWS text product.
        reference_utc:  Expiry resolution anchor (defaults to datetime.now(UTC)).
    """
    raw = extract_ugc_block(text)
    if not raw:
        return None

    # Pull the expiry token from the end of the flattened block.
    parts = [p.strip().strip(".") for p in raw.split("-") if p.strip().strip(".")]
    ddhhmm = ""
    if parts and re.fullmatch(r"\d{6}", parts[-1]):
        ddhhmm = parts[-1]

    zones = expand_ugc_tokens(raw)
    if not zones:
        return None

    expires_utc = resolve_ugc_expires(ddhhmm, reference_utc=reference_utc) if ddhhmm else None

    return UGCBlock(
        zones=zones,
        expires_ddhhmm=ddhhmm,
        expires_utc=expires_utc,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Public API — FIPS / SAME helpers
# ---------------------------------------------------------------------------

def same_from_county_zone(zone_id: str) -> "str | None":
    """
    Convert a county zone ID directly to a 6-digit SAME FIPS code.

    Only works for county-type zones (format XXC###, e.g. MDC031).
    Forecast zones (MDZ008), marine zones (ANZ530), and fire-weather zones
    all return None — those require a network lookup to resolve to SAME.

    The SAME code format is 0SSCC where SS is state FIPS and CC is county.
    Leading zero is always prepended.

    Examples:
        same_from_county_zone("MDC031")  →  "024031"  (Montgomery Co, MD)
        same_from_county_zone("VAC059")  →  "051059"  (Fairfax Co, VA)
        same_from_county_zone("MDC510")  →  "024510"  (Baltimore City, MD)
        same_from_county_zone("MDZ008")  →  None      (forecast zone)
        same_from_county_zone("ANZ530")  →  None      (marine zone)
    """
    zid = (zone_id or "").strip().upper()
    m = _COUNTY_ZONE_RE.fullmatch(zid)
    if not m:
        return None
    fips2 = STATE_ABBR_TO_FIPS.get(m.group(1))
    if not fips2:
        return None
    return f"0{fips2}{m.group(2)}"


def state_abbr_from_same(same6: str) -> "str | None":
    """
    Return the two-letter state abbreviation for a 6-digit SAME FIPS code.

    SAME format is PSSCCC: P=partition digit, SS=state FIPS, CCC=county FIPS.
    State FIPS occupies positions [1:3], NOT [0:2].

    Examples:
        state_abbr_from_same("024031")  →  "MD"
        state_abbr_from_same("051059")  →  "VA"
        state_abbr_from_same("011001")  →  "DC"
    """
    s = "".join(ch for ch in (same6 or "").strip() if ch.isdigit())
    if len(s) != 6:
        return None
    return FIPS_TO_STATE_ABBR.get(s[1:3])


def fips2_from_same(same6: str) -> "str | None":
    """
    Return the 2-digit state FIPS string for a 6-digit SAME FIPS code.

    Returns None if the SAME code is malformed or the state is not in the table.

    Examples:
        fips2_from_same("024031")  →  "24"
        fips2_from_same("051059")  →  "51"
    """
    s = "".join(ch for ch in (same6 or "").strip() if ch.isdigit())
    if len(s) != 6:
        return None
    fips2 = s[1:3]
    return fips2 if fips2 in FIPS_TO_STATE_ABBR else None


# ---------------------------------------------------------------------------
# NWS crosswalk file parsers
#
# These are pure file→dict converters.  Lifecycle management (downloading,
# caching, locking, async coordination) stays in Orchestrator — only the
# parsing logic lives here.
# ---------------------------------------------------------------------------

def parse_zonecounty_dbx(path: Path) -> "dict[str, list[str]]":
    """
    Parse an NWS ZoneCounty ``bp*.dbx`` pipe-delimited crosswalk file.

    Maps forecast zone IDs to the county SAME codes they contain.  NWS
    publishes this file at https://www.weather.gov/gis/ZoneCounty and it is
    the NOAA-recommended authoritative source for zone → county FIPS mapping.

    Schema (pipe-delimited, relevant columns):
        col 0  STATE  — 2-letter state abbreviation (e.g. MD)
        col 1  ZONE   — zone number digits (e.g. 501)
        col 6  FIPS   — 5-digit county FIPS (e.g. 24031)

    Returns:
        ``{"MDZ501": ["024031", "024033", ...], ...}``
        SAME codes are always 6 digits (leading-zero padded).

    Unknown / malformed lines are silently skipped.  An empty dict is
    returned if the file is unreadable or contains no valid records.
    """
    out: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if "|" not in ln:
                    continue
                parts = [p.strip() for p in ln.split("|")]
                if len(parts) < 7:
                    continue

                st   = (parts[0] or "").strip().upper()
                zn   = "".join(ch for ch in (parts[1] or "") if ch.isdigit())
                fips = "".join(ch for ch in (parts[6] or "") if ch.isdigit())

                if len(st) != 2 or not zn or len(fips) != 5:
                    continue

                ugc_id = f"{st}Z{zn.zfill(3)}"
                same   = "0" + fips  # SAME = 0 + state2 + county3 = 6 digits

                if ugc_id not in seen:
                    seen[ugc_id] = set()
                    out[ugc_id]  = []
                if same not in seen[ugc_id]:
                    seen[ugc_id].add(same)
                    out[ugc_id].append(same)
    except OSError:
        pass  # caller handles missing/unreadable file

    return out


def parse_mareas_txt(path: Path) -> "dict[str, list[str]]":
    """
    Parse an NWS ``mareas*.txt`` marine-area crosswalk file.

    Maps marine zone IDs to the coastal county SAME codes they cover.
    Supports two input formats:

    1. **Official NWS pipe format** (preferred)::

           AN|73535|Tidal Potomac from Key Bridge to Indian Head MD|38.7406|-77.0712

       Produces: ``ANZ535 → ["073535"]``

    2. **Legacy / free-form** — lines that already contain a 3-letter+3-digit
       zone ID (e.g. ``ANZ535``) alongside 5- or 6-digit FIPS codes.
       Used for hand-curated crosswalk files and older NWS formats.

    Returns:
        ``{"ANZ535": ["073535"], "ANZ536": ["024009", ...], ...}``
        SAME codes are always 6 digits.

    Lines starting with ``#`` and blank lines are ignored.
    """
    _pipe_alpha = re.compile(r"^[A-Z]{2}$")
    _pipe_num   = re.compile(r"^\d{5}$")
    _zone_re    = re.compile(r"\b([A-Z]{3}\d{3})\b")
    _fips5_re   = re.compile(r"\b(\d{5})\b")
    _same6_re   = re.compile(r"\b(\d{6})\b")

    out:  dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}

    def _add(zone: str, same_code: str) -> None:
        z = "".join(ch for ch in str(zone).upper() if ch.isalnum())
        s = "".join(ch for ch in str(same_code) if ch.isdigit()).zfill(6)
        if len(z) != 6 or len(s) != 6:
            return
        if z not in out:
            out[z]  = []
            seen[z] = set()
        if s not in seen[z]:
            seen[z].add(s)
            out[z].append(s)

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                s0 = (ln or "").strip()
                if not s0 or s0.startswith("#"):
                    continue

                parts = [p.strip() for p in s0.split("|")]
                if (len(parts) >= 2
                        and _pipe_alpha.fullmatch(parts[0].upper())
                        and _pipe_num.fullmatch(parts[1])):
                    # Official NWS pipe format: SSALPHA|SSNUM|NAME|LON|LAT
                    ssalpha = parts[0].upper()
                    ssnum   = parts[1]
                    _add(f"{ssalpha}Z{ssnum[-3:]}", f"0{ssnum}")
                    continue

                # Legacy / free-form fallback
                s = s0.upper()
                zm = _zone_re.search(s)
                if not zm:
                    continue
                zone = zm.group(1).upper()
                for x in _same6_re.findall(s):
                    d = "".join(ch for ch in x if ch.isdigit())
                    if len(d) == 6:
                        _add(zone, d)
                for x in _fips5_re.findall(s):
                    d = "".join(ch for ch in x if ch.isdigit())
                    if len(d) == 5:
                        _add(zone, "0" + d)
    except OSError:
        pass  # caller handles missing/unreadable file

    return out


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# Fixed reference time so expiry tests are deterministic regardless of
# when they're run.  Represents a product issued 2026-04-01 at 20:43Z.
_TEST_REF_UTC = datetime(2026, 4, 1, 20, 43, 0, tzinfo=timezone.utc)


def _run_tests() -> int:
    """
    Run built-in test suite.  Returns number of failures.

    Tests exercise every public function, including edge cases:
      - Multi-state prefix carry
      - NNN>NNN range expansion
      - Marine zone 3-letter prefixes
      - Expiry month-boundary resolution (April → May)
      - Impossible dates (Feb 30)
      - Already-expired products
      - Empty / invalid inputs throughout
    """
    failures = 0
    results: list[tuple[str, bool, str]] = []

    def check(label: str, got: object, expected: object) -> None:
        nonlocal failures
        ok = (got == expected)
        if not ok:
            failures += 1
        results.append((label, ok, f"got={got!r}  expected={expected!r}"))

    # ------------------------------------------------------------------
    # extract_ugc_block
    # ------------------------------------------------------------------
    svr_product = (
        "WUUS51 KLWX 012043\n"
        "SVRLWX\n"
        "VAC059-153-683-685-012130-\n"
        "/O.NEW.KLWX.SV.W.0033.260401T2043Z-260401T2130Z/"
    )
    check(
        "extract_ugc_block: SVR county list",
        extract_ugc_block(svr_product),
        "VAC059-153-683-685-012130-",
    )

    marine_product = (
        "FZUS51 KLWX 011800\n"
        "CWFLWX\n"
        "ANZ530-531-532-533-534-535-011200-\n"
        "/O.NEW.KLWX.SC.Y.0001.260401T1800Z-260402T1200Z/"
    )
    check(
        "extract_ugc_block: marine CWF / Small Craft Advisory",
        extract_ugc_block(marine_product),
        "ANZ530-531-532-533-534-535-011200-",
    )

    multiline_product = (
        "SOME HEADER LINE\n"
        "MDZ001-002-003-\n"
        "VAZ505-012130-\n"
        "product body text"
    )
    check(
        "extract_ugc_block: multiline continuation",
        extract_ugc_block(multiline_product),
        "MDZ001-002-003-VAZ505-012130-",
    )

    check("extract_ugc_block: empty text", extract_ugc_block(""), "")
    check("extract_ugc_block: no UGC in text", extract_ugc_block("No UGC here."), "")

    # ------------------------------------------------------------------
    # expand_ugc_tokens
    # ------------------------------------------------------------------
    check(
        "expand_ugc_tokens: SVR with prefix carry",
        expand_ugc_tokens("VAC059-153-683-685-012130-"),
        ["VAC059", "VAC153", "VAC683", "VAC685"],
    )

    check(
        "expand_ugc_tokens: range expansion MDZ001>005",
        expand_ugc_tokens("MDZ001>005-012130-"),
        ["MDZ001", "MDZ002", "MDZ003", "MDZ004", "MDZ005"],
    )

    check(
        "expand_ugc_tokens: multi-state prefix switch",
        expand_ugc_tokens("MDC031-VAC059-012130-"),
        ["MDC031", "VAC059"],
    )

    check(
        "expand_ugc_tokens: marine 3-letter prefix with carry",
        expand_ugc_tokens("ANZ530-531-532-012130-"),
        ["ANZ530", "ANZ531", "ANZ532"],
    )

    check(
        "expand_ugc_tokens: county zones same state",
        expand_ugc_tokens("MDC003-031-510-012130-"),
        ["MDC003", "MDC031", "MDC510"],
    )

    check(
        "expand_ugc_tokens: multi-state with ranges",
        expand_ugc_tokens("MDZ001>003-VAZ501>503-012130-"),
        ["MDZ001", "MDZ002", "MDZ003", "VAZ501", "VAZ502", "VAZ503"],
    )

    check("expand_ugc_tokens: empty string", expand_ugc_tokens(""), [])

    check(
        "expand_ugc_tokens: expiry token only (no zones)",
        expand_ugc_tokens("012130-"),
        [],
    )

    # ------------------------------------------------------------------
    # resolve_ugc_expires
    # ------------------------------------------------------------------
    # Same-day future: ref=20:43Z, token "012130" → same day 21:30Z
    check(
        "resolve_ugc_expires: same-day future",
        resolve_ugc_expires("012130", reference_utc=_TEST_REF_UTC),
        datetime(2026, 4, 1, 21, 30, tzinfo=timezone.utc),
    )

    # Already expired: ref is after the nominal expiry → return most recent past
    check(
        "resolve_ugc_expires: product already expired",
        resolve_ugc_expires("012130", reference_utc=datetime(2026, 4, 1, 22, 0, tzinfo=timezone.utc)),
        datetime(2026, 4, 1, 21, 30, tzinfo=timezone.utc),
    )

    # Month boundary: April 30 reference, token for day 01 → resolves to May 1
    check(
        "resolve_ugc_expires: month boundary (Apr → May)",
        resolve_ugc_expires(
            "011200",
            reference_utc=datetime(2026, 4, 30, 18, 0, tzinfo=timezone.utc),
        ),
        datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
    )

    # Year boundary: Dec 31 reference, token for day 01 → resolves to Jan 1 next year
    check(
        "resolve_ugc_expires: year boundary (Dec → Jan)",
        resolve_ugc_expires(
            "010600",
            reference_utc=datetime(2026, 12, 31, 23, 0, tzinfo=timezone.utc),
        ),
        datetime(2027, 1, 1, 6, 0, tzinfo=timezone.utc),
    )

    # Impossible date: Feb 30 doesn't exist; candidates are Jan 30 (16 days past)
    # and Mar 30 (43 days future).  Jan 30 is closer to Feb 15 by the closest-wins rule.
    check(
        "resolve_ugc_expires: impossible date Feb 30 → Jan 30 (closest)",
        resolve_ugc_expires(
            "300000",
            reference_utc=datetime(2026, 2, 15, 0, 0, tzinfo=timezone.utc),
        ),
        datetime(2026, 1, 30, 0, 0, tzinfo=timezone.utc),
    )

    check("resolve_ugc_expires: invalid token (day=99)", resolve_ugc_expires("990000"), None)
    check("resolve_ugc_expires: empty string", resolve_ugc_expires(""), None)
    check("resolve_ugc_expires: too short", resolve_ugc_expires("0121"), None)
    # Ref is Apr 1 20:43Z; token "010000" = Apr 1 00:00Z (20h past) vs May 1 00:00Z
    # (~9 days future).  Apr 1 is closer — consistent with a morning product that has since expired.
    check("resolve_ugc_expires: midnight — closest wins (Apr 1, not May 1)",
          resolve_ugc_expires("010000", reference_utc=_TEST_REF_UTC),
          datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc))

    # ------------------------------------------------------------------
    # parse_ugc_block
    # ------------------------------------------------------------------
    block = parse_ugc_block(svr_product, reference_utc=_TEST_REF_UTC)
    check("parse_ugc_block: returns UGCBlock", block is not None, True)
    if block is not None:
        check("parse_ugc_block: zones",         block.zones,         ["VAC059", "VAC153", "VAC683", "VAC685"])
        check("parse_ugc_block: expires_ddhhmm", block.expires_ddhhmm, "012130")
        check("parse_ugc_block: expires_utc",   block.expires_utc,
              datetime(2026, 4, 1, 21, 30, tzinfo=timezone.utc))
        check("parse_ugc_block: raw",            block.raw,           "VAC059-153-683-685-012130-")

    check("parse_ugc_block: no UGC → None", parse_ugc_block("No UGC here."), None)
    check("parse_ugc_block: empty text → None", parse_ugc_block(""), None)

    # ------------------------------------------------------------------
    # same_from_county_zone
    # ------------------------------------------------------------------
    check("same_from_county_zone: MDC031 → 024031",  same_from_county_zone("MDC031"),  "024031")
    check("same_from_county_zone: VAC059 → 051059",  same_from_county_zone("VAC059"),  "051059")
    check("same_from_county_zone: MDC510 (Balt. City)", same_from_county_zone("MDC510"), "024510")
    check("same_from_county_zone: DCC001 (DC)",      same_from_county_zone("DCC001"),  "011001")
    check("same_from_county_zone: forecast zone → None", same_from_county_zone("MDZ008"), None)
    check("same_from_county_zone: marine zone → None",   same_from_county_zone("ANZ530"), None)
    check("same_from_county_zone: empty → None",         same_from_county_zone(""),       None)
    check("same_from_county_zone: lowercase normalised",  same_from_county_zone("mdc031"), "024031")

    # ------------------------------------------------------------------
    # state_abbr_from_same
    # ------------------------------------------------------------------
    check("state_abbr_from_same: MD", state_abbr_from_same("024031"), "MD")
    check("state_abbr_from_same: VA", state_abbr_from_same("051059"), "VA")
    check("state_abbr_from_same: DC", state_abbr_from_same("011001"), "DC")
    check("state_abbr_from_same: WV", state_abbr_from_same("054003"), "WV")
    check("state_abbr_from_same: empty → None",    state_abbr_from_same(""),      None)
    check("state_abbr_from_same: too short → None", state_abbr_from_same("12345"), None)
    check("state_abbr_from_same: unknown FIPS → None", state_abbr_from_same("099999"), None)

    # ------------------------------------------------------------------
    # fips2_from_same
    # ------------------------------------------------------------------
    check("fips2_from_same: MD → 24", fips2_from_same("024031"), "24")
    check("fips2_from_same: VA → 51", fips2_from_same("051059"), "51")
    check("fips2_from_same: DC → 11", fips2_from_same("011001"), "11")
    check("fips2_from_same: empty → None",         fips2_from_same(""),      None)
    check("fips2_from_same: unknown FIPS → None",  fips2_from_same("099999"), None)

    # ------------------------------------------------------------------
    # parse_zonecounty_dbx
    # ------------------------------------------------------------------
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dbx", delete=False, encoding="utf-8") as tf:
        # Real NWS .dbx: STATE|ZONE|ZONENAME|STATEZONE|COUNTY|TIMEZONE|FIPS|COUNTYNAME
        tf.write("MD|501|Baltimore Metro|MD501|Baltimore County|EST|24005|Baltimore County|\n")
        tf.write("MD|501|Baltimore Metro|MD501|Howard County|EST|24027|Howard County|\n")
        tf.write("VA|001|Northern Virginia|VA001|Fairfax County|EST|51059|Fairfax County|\n")
        tf.write("# comment line\n")
        tf.write("BADLINE no pipe\n")
        tf.write("SHORT|LINE\n")
        dbx_path = tf.name
    try:
        dbx = parse_zonecounty_dbx(Path(dbx_path))
        check("parse_zonecounty_dbx: MDZ501 SAME codes",
              sorted(dbx.get("MDZ501", [])), ["024005", "024027"])
        check("parse_zonecounty_dbx: VAZ001 SAME codes",
              sorted(dbx.get("VAZ001", [])), ["051059"])
        check("parse_zonecounty_dbx: unknown zone → empty",
              dbx.get("MDZ999"), None)
        check("parse_zonecounty_dbx: comment/bad lines skipped",
              len(dbx), 2)
    finally:
        os.unlink(dbx_path)

    check("parse_zonecounty_dbx: missing file → empty dict",
          parse_zonecounty_dbx(Path("/nonexistent/path/to.dbx")), {})

    # ------------------------------------------------------------------
    # parse_mareas_txt
    # ------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        # Official NWS pipe format rows
        tf.write("AN|73535|Tidal Potomac Key Bridge to Indian Head MD|38.74|-77.07\n")
        tf.write("AN|73536|Tidal Potomac Indian Head to Cobb Island|38.39|-77.17\n")
        # Legacy free-form row (zone + 6-digit SAME)
        tf.write("ANZ537 area code 073537\n")
        # Legacy with 5-digit FIPS
        tf.write("ANZ538 region 73538\n")
        # Comment and blank
        tf.write("# ignore me\n")
        tf.write("\n")
        mareas_path = tf.name
    try:
        mareas = parse_mareas_txt(Path(mareas_path))
        check("parse_mareas_txt: ANZ535 official pipe format",
              mareas.get("ANZ535"), ["073535"])
        check("parse_mareas_txt: ANZ536 official pipe format",
              mareas.get("ANZ536"), ["073536"])
        check("parse_mareas_txt: ANZ537 legacy 6-digit SAME",
              mareas.get("ANZ537"), ["073537"])
        check("parse_mareas_txt: ANZ538 legacy 5-digit FIPS",
              mareas.get("ANZ538"), ["073538"])
        check("parse_mareas_txt: comment/blank skipped",
              len(mareas), 4)
    finally:
        os.unlink(mareas_path)

    check("parse_mareas_txt: missing file → empty dict",
          parse_mareas_txt(Path("/nonexistent/path/to.txt")), {})

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print(f"Running {len(results)} test assertions...\n")
    for label, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
        if not ok:
            print(f"         {detail}")
    print()
    if failures:
        print(f"FAILED: {failures}/{len(results)} assertions failed.")
    else:
        print(f"All {len(results)} assertions passed.")
    return failures


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    """
    Inspect UGC parsing interactively, or run the built-in test suite.

    Usage:
        python -m seasonalweather.same.ugc
            → run built-in test suite

        python -m seasonalweather.same.ugc "VAC059-153-683-685-012130-"
            → parse a raw UGC block string

        python -m seasonalweather.same.ugc MDC031
            → zone-ID → SAME lookup
    """
    args = sys.argv[1:]

    if not args:
        sys.exit(_run_tests())

    for arg in args:
        arg = arg.strip()
        print(f"Input: {arg!r}")

        # Try as a county zone ID first (short, no hyphens)
        if re.fullmatch(r"[A-Za-z]{3}\d{3}", arg):
            same = same_from_county_zone(arg)
            st = state_abbr_from_same(same) if same else None
            print(f"  same_from_county_zone → {same!r}")
            print(f"  state_abbr_from_same  → {st!r}")
            print()
            continue

        # Try as a raw UGC block or product text
        block = parse_ugc_block(arg)
        if block:
            print(f"  parse_ugc_block:")
            print(f"    zones          = {block.zones}")
            print(f"    expires_ddhhmm = {block.expires_ddhhmm!r}")
            print(f"    expires_utc    = {block.expires_utc}")
            print(f"    raw            = {block.raw!r}")
        else:
            zones = expand_ugc_tokens(arg)
            if zones:
                print(f"  expand_ugc_tokens: {zones}")
            else:
                print("  (no UGC block found; try passing a full product header)")
        print()


if __name__ == "__main__":
    _cli()
