"""
vtec.py — VTEC parsing and toneout policy library for SeasonalWeather.

This module is the single source of truth for:
  - Parsing raw VTEC strings into structured objects
  - Significance → SAME eligibility rules
  - Action → FULL-worthiness rules
  - phen.sig → SAME event code mapping
  - toneout_policy(): the one call the Orchestrator uses to decide FULL vs VOICE

Design rules (do not break these):
  - Zero imports from the SeasonalWeather package. Pure stdlib only.
  - No side effects. Every function is pure / stateless.
  - toneout_policy() is the authoritative output. The Orchestrator does not
    second-guess it. It returns a ToneoutPolicy and the caller obeys.
  - track_id format is OFFICE.PHEN.SIG.ETN — matches active_alerts._vtec_track_id.

Analogous role in real NWR infrastructure:
  - This module ~ NWRWAVES VTEC awareness layer
  - The Orchestrator (main.py) ~ BMH: receives a decision, executes it

Usage (library):
    from seasonalweather.vtec import toneout_policy
    policy = toneout_policy(["/O.NEW.KLWX.CF.Y.0004.260325T1400Z-260325T1800Z/"])
    # ToneoutPolicy(mode='VOICE', same_code=None, reason='sig=Y:advisory:voice-only', ...)

Usage (CLI — ask it what it thinks a string is):
    python -m seasonalweather.alerts.vtec "/O.NEW.KLWX.CF.Y.0004.260325T1400Z-260325T1800Z/"
    python -m seasonalweather.alerts.vtec "/O.NEW.KLWX.TO.W.0012.260325T1400Z-260325T1800Z/"
    python -m seasonalweather.alerts.vtec  # runs built-in test vectors
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# VTEC regex
# Handles both 6-digit (YYMMDD) and 8-digit (YYYYMMDD) date prefixes.
# ---------------------------------------------------------------------------
_VTEC_RE = re.compile(
    r"/(?P<kind>[A-Z])"
    r"\.(?P<action>[A-Z]{3})"
    r"\.(?P<office>[A-Z]{4})"
    r"\.(?P<phen>[A-Z0-9]{2})"
    r"\.(?P<sig>[A-Z])"
    r"\.(?P<etn>\d{4})"
    r"\.(?P<start>(?:\d{8}|\d{6})T\d{4}Z)"
    r"-(?P<end>(?:\d{8}|\d{6})T\d{4}Z)/"
)


# ---------------------------------------------------------------------------
# Significance → SAME/toneout eligibility
#
# Source: VTEC Technical Bulletin (NWS Instruction 10-1703),
#         real NOAA Weather Radio / NWRWAVES behavior.
#
#   W = Warning  → SAME-eligible, FULL on qualifying action
#   A = Watch    → SAME-eligible, FULL on qualifying action
#   Y = Advisory → Voice-only. NEVER sends SAME tones on real NWR.
#   S = Statement→ Voice-only.
#   F = Forecast → Voice-only.
#   O = Outlook  → Voice-only.
#   N = Synopsis → Voice-only.
# ---------------------------------------------------------------------------
_SIG_SAME_ELIGIBLE: frozenset[str] = frozenset({"W", "A"})

_SIG_LABELS: dict[str, str] = {
    "W": "Warning",
    "A": "Watch",
    "Y": "Advisory",
    "S": "Statement",
    "F": "Forecast",
    "O": "Outlook",
    "N": "Synopsis",
}

# ---------------------------------------------------------------------------
# Actions → FULL-worthiness
#
# FULL (sends SAME tones on air, triggers heightened mode):
#   NEW — New issuance
#   UPG — Upgraded (e.g. Watch → Warning)
#   EXA — Extended in area
#   EXB — Extended in area and time
#
# Voice-only (no SAME tones):
#   CON — Continued
#   EXT — Extended in time only
#   COR — Correction
#   ROU — Routine
#   CAN — Cancelled
#   EXP — Expired
# ---------------------------------------------------------------------------
_FULL_ACTIONS: frozenset[str] = frozenset({"NEW", "UPG", "EXA", "EXB"})
_VOICE_ACTIONS: frozenset[str] = frozenset({"CON", "EXT", "COR", "ROU", "CAN", "EXP"})

_ACTION_LABELS: dict[str, str] = {
    "NEW": "New",
    "CON": "Continued",
    "EXT": "Extended (time)",
    "EXA": "Extended (area)",
    "EXB": "Extended (area+time)",
    "UPG": "Upgraded",
    "CAN": "Cancelled",
    "EXP": "Expired",
    "COR": "Correction",
    "ROU": "Routine",
}

# ---------------------------------------------------------------------------
# phen.sig → SAME event code
#
# This is the authoritative table. Advisory (Y), Statement (S), etc. are
# deliberately absent — they have no SAME code and are always voice-only.
# "If it's not in this table, there is no SAME code for it."
# ---------------------------------------------------------------------------
_PHEN_SIG_TO_SAME: dict[str, str] = {
    # Coastal Flood
    "CF.W": "CFW",  "CF.A": "CFA",
    # Tornado
    "TO.W": "TOR",  "TO.A": "TOA",
    # Severe Thunderstorm
    "SV.W": "SVR",  "SV.A": "SVA",
    # Flash Flood
    "FF.W": "FFW",  "FF.A": "FFA",
    # Flood (riverine)
    "FL.W": "FLW",  "FL.A": "FLA",
    # Winter Storm
    "WS.W": "WSW",  "WS.A": "WSA",
    # Blizzard
    "BZ.W": "BZW",
    # Ice Storm
    "IS.W": "ISW",
    # High Wind
    "HW.W": "HWW",  "HW.A": "HWA",
    # Wind Chill
    "WC.W": "WCW",  "WC.A": "WCA",
    # Extreme Wind
    "EW.W": "EWW",
    # Hurricane
    "HU.W": "HUW",  "HU.A": "HUA",
    # Tropical Storm
    "TR.W": "TRW",  "TR.A": "TRA",
    # Storm Surge
    "SS.W": "SSW",  "SS.A": "SSA",
    # Tsunami
    "TS.W": "TSW",  "TS.A": "TSA",
    # Avalanche
    "AV.W": "AVW",  "AV.A": "AVA",
    # Dust Storm
    "DS.W": "DSW",
    # Earthquake
    "EQ.W": "EQW",
    # Volcano
    "VO.W": "VOW",
    # Fire Warning
    "FW.W": "FRW",
    # Snow Squall
    "SQ.W": "SQW",
    # Hazardous Materials
    "HZ.W": "HMW",
    # Civil Danger
    "CD.W": "CDW",
    # Nuclear Power Plant
    "NU.W": "NUW",
    # Law Enforcement
    "LE.W": "LEW",
    # Shelter in Place
    "SP.W": "SPW",
    # Radiological Hazard
    "RH.W": "RHW",
    # Special Marine Warning (action-based, marine-only)
    "MA.W": "SMW",
}


# ---------------------------------------------------------------------------
# phen.sig → human-readable event label
#
# The authoritative label table for VTEC phen.sig pairs.  Callers needing a
# display name (NWWS ingestion, AlertTracker, station feed, etc.) should call
# phen_sig_label() rather than maintaining separate mini-maps in main.py or
# elsewhere.
#
# Significance codes used:
#   W = Warning   A = Watch   Y = Advisory   S = Statement
#   F = Forecast  O = Outlook  N = Synopsis
# ---------------------------------------------------------------------------
_PHEN_SIG_TO_LABEL: dict[str, str] = {
    # Convective
    "TO.W": "Tornado Warning",             "TO.A": "Tornado Watch",
    "SV.W": "Severe Thunderstorm Warning", "SV.A": "Severe Thunderstorm Watch",
    "EW.W": "Extreme Wind Warning",
    # Flash / Areal Flood
    "FF.W": "Flash Flood Warning",         "FF.A": "Flash Flood Watch",
    "FF.Y": "Flash Flood Advisory",
    "FA.W": "Flood Warning",               "FA.A": "Flood Watch",             "FA.Y": "Flood Advisory",
    "FL.W": "Flood Warning",               "FL.A": "Flood Watch",             "FL.Y": "Flood Advisory",
    # Coastal / Storm Surge
    "CF.W": "Coastal Flood Warning",       "CF.A": "Coastal Flood Watch",     "CF.Y": "Coastal Flood Advisory",
    "SS.W": "Storm Surge Warning",         "SS.A": "Storm Surge Watch",
    "SU.W": "High Surf Warning",           "SU.Y": "High Surf Advisory",
    "BH.S": "Beach Hazards Statement",
    "RP.S": "Rip Current Statement",
    # Tropical
    "HU.W": "Hurricane Warning",           "HU.A": "Hurricane Watch",
    "HU.S": "Hurricane Local Statement",
    "TR.W": "Tropical Storm Warning",      "TR.A": "Tropical Storm Watch",
    "TY.W": "Typhoon Warning",             "TY.A": "Typhoon Watch",
    # Marine
    "MA.W": "Special Marine Warning",
    "SC.Y": "Small Craft Advisory",
    "SE.W": "Hazardous Seas Warning",      "SE.A": "Hazardous Seas Watch",
    "GL.W": "Gale Warning",                "GL.A": "Gale Watch",
    "SR.W": "Storm Warning",               "SR.A": "Storm Watch",
    "HF.W": "Hurricane Force Wind Warning","HF.A": "Hurricane Force Wind Watch",
    "MH.W": "Marine Hurricane Force Wind Warning",
    "MH.Y": "Marine Hurricane Force Wind Advisory",
    "UP.W": "Ice Accretion Warning",       "UP.Y": "Freezing Spray Advisory",
    "RB.Y": "Small Craft Advisory for Rough Bar",
    "SI.Y": "Small Craft Advisory for Rough Bar",
    "SW.Y": "Small Craft Advisory for Winds",
    "LO.Y": "Lake Wind Advisory",
    "MS.Y": "Dense Fog Advisory for Mariners",
    # Tsunami
    "TS.W": "Tsunami Warning",             "TS.A": "Tsunami Watch",           "TS.Y": "Tsunami Advisory",
    # Winter
    "BZ.W": "Blizzard Warning",            "BZ.A": "Blizzard Watch",
    "IS.W": "Ice Storm Warning",
    "WS.W": "Winter Storm Warning",        "WS.A": "Winter Storm Watch",
    "WW.Y": "Winter Weather Advisory",
    "LE.W": "Lake Effect Snow Warning",    "LE.A": "Lake Effect Snow Watch",  "LE.Y": "Lake Effect Snow Advisory",
    "SQ.W": "Snow Squall Warning",
    "IP.Y": "Sleet Advisory",
    # Wind
    "HW.W": "High Wind Warning",           "HW.A": "High Wind Watch",
    "WI.Y": "Wind Advisory",
    "LW.Y": "Lake Wind Advisory",
    # Heat / Cold
    "EH.W": "Excessive Heat Warning",      "EH.A": "Excessive Heat Watch",
    "HT.Y": "Heat Advisory",
    "HZ.W": "Hard Freeze Warning",         "HZ.A": "Hard Freeze Watch",
    "FZ.W": "Freeze Warning",              "FZ.A": "Freeze Watch",
    "FR.Y": "Frost Advisory",
    "WC.W": "Wind Chill Warning",          "WC.A": "Wind Chill Watch",        "WC.Y": "Wind Chill Advisory",
    "EC.W": "Extreme Cold Warning",        "EC.A": "Extreme Cold Watch",
    # Fire
    "FW.W": "Red Flag Warning",            "FW.A": "Fire Weather Watch",
    # Fog / Smoke / Dust
    "FG.Y": "Dense Fog Advisory",
    "ZF.Y": "Freezing Fog Advisory",
    "SM.Y": "Dense Smoke Advisory",
    "DU.W": "Blowing Dust Warning",        "DU.Y": "Blowing Dust Advisory",
    "DS.W": "Dust Storm Warning",
    "BS.Y": "Blowing Snow Advisory",
    # Geophysical
    "AV.W": "Avalanche Warning",           "AV.A": "Avalanche Watch",         "AV.Y": "Avalanche Advisory",
    "EQ.W": "Earthquake Warning",
    "VO.W": "Volcano Warning",
    # Non-met (VTEC-issued)
    "CD.W": "Civil Danger Warning",
    "NU.W": "Nuclear Power Plant Warning",
    "SP.W": "Shelter in Place Warning",
    "RH.W": "Radiological Hazard Warning",
}


def phen_sig_label(phen_sig: str) -> str | None:
    """
    Return the human-readable event label for a VTEC phen.sig pair.

    Returns None when the pair is not in the table — callers should fall back
    to text extraction or the raw AWIPS product type.

        phen_sig_label("TO.W")   # → "Tornado Warning"
        phen_sig_label("MA.W")   # → "Special Marine Warning"
        phen_sig_label("CF.Y")   # → "Coastal Flood Advisory"
        phen_sig_label("ZZ.Z")   # → None
    """
    return _PHEN_SIG_TO_LABEL.get((phen_sig or "").strip().upper())


# ---------------------------------------------------------------------------
# Exported regex aliases
# These are the canonical VTEC regexes for the whole codebase.  Callers
# should import from here rather than defining their own copies.
# ---------------------------------------------------------------------------
VTEC_FIND_RE = re.compile(
    r"/[A-Z]\.[A-Z]{3}\.[A-Z]{4}\.[A-Z0-9]{2}\.[A-Z]\.\d{4}\.(?:\d{8}|\d{6})T\d{4}Z-(?:\d{8}|\d{6})T\d{4}Z/"
)
VTEC_PARSE_RE = re.compile(
    r"/(?P<prod>[A-Z])\.(?P<act>[A-Z]{3})\.(?P<office>[A-Z]{4})\.(?P<phen>[A-Z0-9]{2})\.(?P<sig>[A-Z])\.(?P<etn>\d{4})\.(?P<start>(?:\d{8}|\d{6})T\d{4}Z)-(?P<end>(?:\d{8}|\d{6})T\d{4}Z)/"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedVTEC:
    """
    Structured representation of a single parsed VTEC string.

    Attributes match the raw VTEC field order. All strings are uppercased
    and stripped exactly as they appear in the VTEC.
    """
    kind: str        # O=Operational, T=Test, E=Experimental
    action: str      # NEW, CON, EXT, EXA, EXB, UPG, CAN, EXP, COR, ROU
    office: str      # WFO 4-letter, e.g. KLWX
    phenomena: str   # 2-letter phenomena code, e.g. CF, TO, SV
    significance: str  # W, A, Y, S, F, O, N
    etn: str         # event tracking number, zero-padded 4 digits
    start: str       # as-parsed timestamp, e.g. 260325T1400Z
    end: str         # as-parsed timestamp, e.g. 260325T1800Z
    raw: str         # the original raw string this was parsed from

    @property
    def phen_sig(self) -> str:
        """e.g. 'CF.Y', 'TO.W', 'SV.A'"""
        return f"{self.phenomena}.{self.significance}"

    @property
    def track_id(self) -> str:
        """
        Canonical deduplication / tracking key.
        Stable across action updates (NEW → CON → CAN all share the same track_id).
        Format matches active_alerts._vtec_track_id: OFFICE.PHEN.SIG.ETN
        """
        return f"{self.office}.{self.phenomena}.{self.significance}.{self.etn}"

    @property
    def is_full_action(self) -> bool:
        """True if action qualifies for a FULL toneout (NEW/UPG/EXA/EXB)."""
        return self.action in _FULL_ACTIONS

    @property
    def is_same_eligible(self) -> bool:
        """True if significance permits SAME tones (W or A only)."""
        return self.significance in _SIG_SAME_ELIGIBLE

    @property
    def same_code(self) -> str | None:
        """SAME event code for this phen.sig, or None if voice-only."""
        return _PHEN_SIG_TO_SAME.get(self.phen_sig)

    @property
    def sig_label(self) -> str:
        return _SIG_LABELS.get(self.significance, self.significance)

    @property
    def action_label(self) -> str:
        return _ACTION_LABELS.get(self.action, self.action)

    def describe(self) -> str:
        """Human-readable one-liner for logs / CLI output."""
        same = self.same_code or "(no SAME code)"
        eligible = "SAME-eligible" if self.is_same_eligible else "voice-only"
        full = "FULL-action" if self.is_full_action else "non-FULL-action"
        return (
            f"{self.office} {self.phenomena}.{self.significance} "
            f"({self.sig_label}) ETN={self.etn} action={self.action} "
            f"({self.action_label}) → {eligible}, {full}, SAME={same}"
        )


@dataclass(frozen=True)
class ToneoutPolicy:
    """
    The authoritative toneout decision for an event.

    The Orchestrator receives this and executes it. It does not override it.

    mode:
        "FULL"  — Air with SAME attention tones. same_code is guaranteed non-None.
        "VOICE" — Air voice-only. No SAME tones. same_code may be None.

    same_code:
        The EAS/SAME event code to use in the SAME header, or None for voice-only.
        Always None when mode is "VOICE".

    reason:
        Structured log string explaining the decision. Always log this.

    primary_vtec:
        The ParsedVTEC that drove the decision, or None if no VTEC was parseable.

    cancel_tracks:
        Set of track_ids that carry CAN or EXP actions in this VTEC list.
        The Orchestrator should remove these from AlertTracker.

    continuation_tracks:
        Set of track_ids with CON/EXT/COR/ROU that are voice-only updates
        to active events. Useful for cycle rebroadcast expiry refresh.
    """
    mode: Literal["FULL", "VOICE"]
    same_code: str | None
    reason: str
    primary_vtec: ParsedVTEC | None
    cancel_tracks: frozenset[str]
    continuation_tracks: frozenset[str]

    def __post_init__(self) -> None:
        # Enforce the contract: FULL must have a same_code.
        if self.mode == "FULL" and not self.same_code:
            raise ValueError("ToneoutPolicy FULL mode requires a non-empty same_code")
        # And VOICE must not carry a same_code.
        if self.mode == "VOICE" and self.same_code is not None:
            raise ValueError("ToneoutPolicy VOICE mode must have same_code=None")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_vtec(raw: str) -> ParsedVTEC | None:
    """
    Parse a single raw VTEC string. Returns None if the string does not
    contain a valid VTEC.

    Accepts the full VTEC string with or without surrounding whitespace.
    """
    s = "".join(raw.split())
    m = _VTEC_RE.search(s)
    if not m:
        return None
    return ParsedVTEC(
        kind=m.group("kind"),
        action=m.group("action"),
        office=m.group("office"),
        phenomena=m.group("phen"),
        significance=m.group("sig"),
        etn=m.group("etn"),
        start=m.group("start"),
        end=m.group("end"),
        raw=raw,
    )


def parse_vtec_list(vtec_strings: list[str]) -> list[ParsedVTEC]:
    """
    Parse a list of raw VTEC strings. Unparseable entries are silently dropped.
    Deduplicates by track_id + action so the same event never appears twice.
    """
    seen: set[str] = set()
    out: list[ParsedVTEC] = []
    for raw in (vtec_strings or []):
        v = parse_vtec(raw)
        if v is None:
            continue
        key = f"{v.track_id}|{v.action}"
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def toneout_policy(vtec_strings: list[str]) -> ToneoutPolicy:
    """
    Given a list of raw VTEC strings from any source (NWWS-OI, CAP, ERN),
    return the authoritative toneout policy for this event.

    The Orchestrator calls this and obeys the result. No overrides.

    Decision logic (mirrors real NWRWAVES / NWR BMH behavior):

      1. No parseable VTEC at all
             → VOICE, same_code=None, reason="no-vtec"

      2. All parsed VTECs have voice-only significance (Y/S/F/O/N)
             → VOICE, same_code=None, reason="sig=<codes>:voice-only"
             (This is what broke today: CF.Y.NEW was given SAME tones.)

      3. SAME-eligible significance (W/A) but action is not FULL-worthy (CON/EXT/etc.)
             → VOICE, same_code=None, reason="sig=W:action=CON:non-full"
             (Continuation updates — still voice-only on air, but track is known.)

      4. SAME-eligible significance AND FULL-worthy action (NEW/UPG/EXA/EXB)
             → FULL, same_code from _PHEN_SIG_TO_SAME, reason="sig=W:action=NEW:full"

    cancel_tracks and continuation_tracks are populated from the full VTEC list
    regardless of what mode is returned, so the Orchestrator can keep AlertTracker
    and the rebroadcast rotation up to date.
    """
    parsed = parse_vtec_list(vtec_strings)

    cancel_tracks: set[str] = set()
    continuation_tracks: set[str] = set()

    for v in parsed:
        if v.action in {"CAN", "EXP"}:
            cancel_tracks.add(v.track_id)
        elif v.action in {"CON", "EXT", "COR", "ROU"}:
            continuation_tracks.add(v.track_id)

    _no_policy = lambda reason, primary=None: ToneoutPolicy(
        mode="VOICE",
        same_code=None,
        reason=reason,
        primary_vtec=primary,
        cancel_tracks=frozenset(cancel_tracks),
        continuation_tracks=frozenset(continuation_tracks),
    )

    # Case 1: nothing parseable
    if not parsed:
        return _no_policy("no-vtec")

    # Partition into SAME-eligible and voice-only
    eligible = [v for v in parsed if v.is_same_eligible]
    voice_only_sigs = {v.significance for v in parsed if not v.is_same_eligible}

    # Case 2: all VTECs are voice-only significance
    if not eligible:
        sig_str = ",".join(sorted(voice_only_sigs))
        labels = "/".join(
            _SIG_LABELS.get(s, s) for s in sorted(voice_only_sigs)
        )
        return _no_policy(
            f"sig={sig_str}:{labels}:voice-only",
            primary=parsed[0],
        )

    # From here, we have at least one SAME-eligible VTEC.
    # Find the most action-worthy one: prefer FULL actions first.
    full_candidates = [v for v in eligible if v.is_full_action]
    primary = full_candidates[0] if full_candidates else eligible[0]

    # Case 3: eligible significance but non-FULL action
    if not full_candidates:
        acts = ",".join(sorted({v.action for v in eligible}))
        return _no_policy(
            f"sig={primary.significance}:action={acts}:non-full-action",
            primary=primary,
        )

    # Case 4: FULL toneout
    same_code = primary.same_code
    if same_code is None:
        # Eligible significance but no table entry — shouldn't happen for standard
        # phen codes, but handle gracefully rather than crash.
        return _no_policy(
            f"sig={primary.significance}:action={primary.action}:phen={primary.phenomena}:no-same-table-entry",
            primary=primary,
        )

    return ToneoutPolicy(
        mode="FULL",
        same_code=same_code,
        reason=f"sig={primary.significance}:{primary.sig_label}:action={primary.action}:phen_sig={primary.phen_sig}",
        primary_vtec=primary,
        cancel_tracks=frozenset(cancel_tracks),
        continuation_tracks=frozenset(continuation_tracks),
    )


def lookup_same_code(phen: str, sig: str) -> str | None:
    """
    Direct phen+sig lookup for callers that already have parsed fields.
    Returns None if there is no SAME code for this combination.
    """
    return _PHEN_SIG_TO_SAME.get(f"{phen.upper()}.{sig.upper()}")


# ---------------------------------------------------------------------------
# Built-in test vectors
# ---------------------------------------------------------------------------

_TEST_VECTORS: list[tuple[str, str, str | None, str]] = [
    # (vtec_string, expected_mode, expected_same_code, label)
    (
        "/O.NEW.KLWX.CF.Y.0004.260325T1400Z-260325T1800Z/",
        "VOICE", None,
        "CF.Y NEW — today's bug: advisory must never get SAME tones",
    ),
    (
        "/O.NEW.KLWX.CF.W.0003.260325T1400Z-260325T1800Z/",
        "FULL", "CFW",
        "CF.W NEW — Coastal Flood Warning, should FULL",
    ),
    (
        "/O.NEW.KLWX.CF.A.0001.260325T1400Z-260325T1800Z/",
        "FULL", "CFA",
        "CF.A NEW — Coastal Flood Watch, should FULL",
    ),
    (
        "/O.CON.KLWX.CF.W.0003.260325T1400Z-260325T1800Z/",
        "VOICE", None,
        "CF.W CON — continuation of warning, non-FULL action",
    ),
    (
        "/O.NEW.KLWX.TO.W.0012.260325T1800Z-260325T2000Z/",
        "FULL", "TOR",
        "TO.W NEW — Tornado Warning, should FULL",
    ),
    (
        "/O.NEW.KLWX.SV.W.0045.260325T1800Z-260325T2100Z/",
        "FULL", "SVR",
        "SV.W NEW — Severe Thunderstorm Warning, should FULL",
    ),
    (
        "/O.NEW.KLWX.FF.Y.0001.260325T1400Z-260325T1800Z/",
        "VOICE", None,
        "FF.Y NEW — Flash Flood Advisory, voice-only",
    ),
    (
        "/O.EXA.KLWX.WS.W.0002.260325T0000Z-260326T1200Z/",
        "FULL", "WSW",
        "WS.W EXA — area extension, should FULL",
    ),
    (
        "/O.CAN.KLWX.TO.W.0012.260325T1800Z-260325T2000Z/",
        "VOICE", None,
        "TO.W CAN — cancellation is voice-only, track goes to cancel_tracks",
    ),
    (
        "/O.NEW.KLWX.SP.S.0001.260325T1400Z-260325T1800Z/",
        "VOICE", None,
        "SP.S NEW — Special Weather Statement, S=Statement is voice-only",
    ),
    (
        "not a vtec string at all",
        "VOICE", None,
        "No VTEC — should return no-vtec",
    ),
]


def _run_tests() -> int:
    """Run built-in test vectors. Returns number of failures."""
    failures = 0
    print(f"Running {len(_TEST_VECTORS)} test vectors...\n")
    for vtec_str, exp_mode, exp_same, label in _TEST_VECTORS:
        policy = toneout_policy([vtec_str])
        ok = (policy.mode == exp_mode) and (policy.same_code == exp_same)
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  [{status}] {label}")
        print(f"         input : {vtec_str}")
        print(f"         got   : mode={policy.mode} same_code={policy.same_code} reason={policy.reason}")
        if not ok:
            print(f"         expect: mode={exp_mode} same_code={exp_same}")
        print()
    if failures:
        print(f"FAILED: {failures}/{len(_TEST_VECTORS)} tests failed.")
    else:
        print(f"All {len(_TEST_VECTORS)} tests passed.")
    return failures


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    """
    Ask vtec.py what it thinks a VTEC string (or list of strings) is.

    Usage:
        python -m seasonalweather.alerts.vtec
            → runs built-in test vectors

        python -m seasonalweather.alerts.vtec "/O.NEW.KLWX.CF.Y.0004.260325T1400Z-260325T1800Z/"
            → prints full parse + policy for that string

        python -m seasonalweather.alerts.vtec "STRING1" "STRING2"
            → treats all args as a multi-VTEC event (as CAP sometimes sends)
    """
    args = sys.argv[1:]

    if not args:
        sys.exit(_run_tests())

    print(f"Input ({len(args)} string(s)):")
    for s in args:
        v = parse_vtec(s)
        if v:
            print(f"  Parsed : {v.describe()}")
            print(f"           track_id={v.track_id}")
        else:
            print(f"  Parsed : (no VTEC found in {s!r})")
    print()

    policy = toneout_policy(args)
    print("toneout_policy() result:")
    print(f"  mode             : {policy.mode}")
    print(f"  same_code        : {policy.same_code}")
    print(f"  reason           : {policy.reason}")
    print(f"  primary_vtec     : {policy.primary_vtec.describe() if policy.primary_vtec else None}")
    print(f"  cancel_tracks    : {set(policy.cancel_tracks) or '(none)'}")
    print(f"  continuation_trk : {set(policy.continuation_tracks) or '(none)'}")


if __name__ == "__main__":
    _cli()
