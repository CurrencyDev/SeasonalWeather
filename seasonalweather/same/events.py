"""
same_events.py — SAME/EAS event code library for SeasonalWeather.

This module is the single source of truth for human-readable labels and
metadata associated with EAS/SAME event codes.

Who uses this:
  - ERN relay path (pure SAME, no VTEC — needs code → label, urgency)
  - Station feed / AlertTracker (event display names for cycle segments)
  - Discord embed / webhook log (human-readable event names)
  - AWIPS product-type fallback in NWWS when there is genuinely no VTEC

Who does NOT use this for policy decisions:
  - vtec.py owns phen.sig → SAME code mapping and FULL/VOICE policy.
    Do not duplicate that table here. This module labels codes; vtec.py
    decides which code applies and whether tones go out.

Custom event codes:
  - Config-defined codes (config.yaml vtec_event_labels or similar) are
    additive overrides. They do not replace this table; callers merge them
    on top. This module ships the NWS standard set.

Design rules:
  - Zero imports from the SeasonalWeather package. Pure stdlib only.
  - No side effects. All functions are pure / stateless.

Usage (library):
    from seasonalweather.same.events import label_for, urgency_for, is_known

    label_for("TOR")    # → "Tornado Warning"
    label_for("CFW")    # → "Coastal Flood Warning"
    label_for("CFA")    # → "Coastal Flood Watch"
    label_for("CFY")    # → None  (advisory — no SAME code in standard set)
    urgency_for("TOR")  # → "Immediate"
    is_known("RWT")     # → True

Usage (CLI):
    python -m seasonalweather.same.events TOR CFW CFA
    python -m seasonalweather.same.events          # prints full table
"""
from __future__ import annotations

import sys
from typing import Literal


# ---------------------------------------------------------------------------
# Urgency categories
# Used by ERN relay and Discord embed to colour/prioritise display.
# These mirror the EAS urgency model loosely adapted for broadcast priority.
#
#   Immediate — life-safety warnings; highest broadcast priority
#   Expected  — watches and most warnings; standard broadcast priority
#   Future    — advisories, statements, outlooks; voice-only / cycle only
#   Past      — test messages, administrative
# ---------------------------------------------------------------------------
Urgency = Literal["Immediate", "Expected", "Future", "Past"]


# ---------------------------------------------------------------------------
# Master event code table
#
# Format: code → (label, urgency)
#
# Source: FCC 47 CFR §11.31 Table 2 to paragraph (e), NWS EAS Implementation
#         Guide, IPAWS OPEN alerting guidelines.
#
# Advisory-level and statement-level codes are included for completeness
# (AlertTracker display, ERN relay labelling) but carry urgency "Future"
# to signal they are voice-only / non-toneout events.
# ---------------------------------------------------------------------------
_TABLE: dict[str, tuple[str, Urgency]] = {

    # ---- National (required) ----
    "EAN": ("Emergency Action Notification",    "Immediate"),
    "EAT": ("Emergency Action Termination",     "Past"),
    "NIC": ("National Information Center",      "Expected"),
    "NPT": ("National Periodic Test",           "Past"),
    "RMT": ("Required Monthly Test",            "Past"),
    "RWT": ("Required Weekly Test",             "Past"),
    "NEM": ("National Emergency Message",       "Immediate"),
    "NAT": ("National Audible Test",            "Past"),
    "NST": ("National Silent Test",             "Past"),

    # ---- Tornado / Severe Thunderstorm ----
    "TOR": ("Tornado Warning",                  "Immediate"),
    "TOA": ("Tornado Watch",                    "Expected"),
    "SVR": ("Severe Thunderstorm Warning",      "Immediate"),
    "SVA": ("Severe Thunderstorm Watch",        "Expected"),
    "SVS": ("Severe Weather Statement",         "Future"),
    "SQW": ("Snow Squall Warning",              "Immediate"),
    "EWW": ("Extreme Wind Warning",             "Immediate"),

    # ---- Flood ----
    "FFW": ("Flash Flood Warning",              "Immediate"),
    "FFA": ("Flash Flood Watch",                "Expected"),
    "FFS": ("Flash Flood Statement",            "Future"),
    "FLW": ("Flood Warning",                    "Expected"),
    "FLA": ("Flood Watch",                      "Expected"),
    "FLS": ("Flood Statement",                  "Future"),

    # ---- Coastal / Marine ----
    "CFW": ("Coastal Flood Warning",            "Expected"),
    "CFA": ("Coastal Flood Watch",              "Expected"),
    "SMW": ("Special Marine Warning",           "Immediate"),
    "MWS": ("Marine Weather Statement",         "Future"),

    # ---- Winter ----
    "WSW": ("Winter Storm Warning",             "Expected"),
    "WSA": ("Winter Storm Watch",               "Expected"),
    "BZW": ("Blizzard Warning",                 "Expected"),
    "ISW": ("Ice Storm Warning",                "Expected"),
    "WCW": ("Wind Chill Warning",               "Expected"),
    "WCA": ("Wind Chill Watch",                 "Expected"),
    "LEW": ("Lake Effect Snow Warning",         "Expected"),   # some WFOs use LEW
    "LAW": ("Lake Effect Snow Watch",           "Expected"),

    # ---- Wind ----
    "HWW": ("High Wind Warning",                "Expected"),
    "HWA": ("High Wind Watch",                  "Expected"),

    # ---- Tropical ----
    "HUW": ("Hurricane Warning",                "Immediate"),
    "HUA": ("Hurricane Watch",                  "Expected"),
    "HLS": ("Hurricane Statement",              "Future"),
    "TRW": ("Tropical Storm Warning",           "Expected"),
    "TRA": ("Tropical Storm Watch",             "Expected"),
    "SSW": ("Storm Surge Warning",              "Immediate"),
    "SSA": ("Storm Surge Watch",                "Expected"),

    # ---- Tsunami ----
    "TSW": ("Tsunami Warning",                  "Immediate"),
    "TSA": ("Tsunami Watch",                    "Expected"),

    # ---- Fire ----
    "FRW": ("Fire Warning",                     "Immediate"),
    "FEW": ("Fire Emergency Warning",           "Immediate"),

    # ---- Avalanche ----
    "AVW": ("Avalanche Warning",                "Immediate"),
    "AVA": ("Avalanche Watch",                  "Expected"),

    # ---- Geological / Weather hazards ----
    "EQW": ("Earthquake Warning",               "Immediate"),
    "VOW": ("Volcano Warning",                  "Immediate"),
    "DSW": ("Dust Storm Warning",               "Expected"),

    # ---- Civil / Safety ----
    "CDW": ("Civil Danger Warning",             "Immediate"),
    "CEM": ("Civil Emergency Message",          "Immediate"),
    "CAE": ("Child Abduction Emergency",        "Immediate"),
    "EVI": ("Evacuation Immediate",             "Immediate"),
    "HMW": ("Hazardous Materials Warning",      "Immediate"),
    "NUW": ("Nuclear Power Plant Warning",      "Immediate"),
    "RHW": ("Radiological Hazard Warning",      "Immediate"),
    "SPW": ("Shelter in Place Warning",         "Immediate"),
    "LEW": ("Law Enforcement Warning",          "Immediate"),   # overrides snow LEW above
    "TOE": ("911 Telephone Outage Emergency",   "Immediate"),
    "BLU": ("Blue Alert",                       "Immediate"),
    "LAE": ("Local Area Emergency",             "Expected"),
    "MEP": ("Missing and Endangered Persons",   "Immediate"),

    # ---- Administrative / Other ----
    "ADR": ("Administrative Message",           "Future"),
    "NMN": ("Network Message Notification",     "Future"),
    "DMO": ("Practice/Demo Warning",            "Past"),
    "SPS": ("Special Weather Statement",        "Future"),
}

# Ensure LEW resolves to the civil meaning (it was assigned twice above).
# Law Enforcement Warning takes precedence in EAS.
_TABLE["LEW"] = ("Law Enforcement Warning", "Immediate")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def label_for(code: str, *, default: str | None = None) -> str | None:
    """
    Return the human-readable label for a SAME/EAS event code.

    Returns `default` (None by default) if the code is not in the standard
    table. Callers that have config-defined custom codes should check their
    own map first, then fall back to this.

        label_for("TOR")          # → "Tornado Warning"
        label_for("CFW")          # → "Coastal Flood Warning"
        label_for("ZZZ")          # → None
        label_for("ZZZ", default="Unknown") # → "Unknown"
    """
    entry = _TABLE.get((code or "").strip().upper())
    return entry[0] if entry is not None else default


def urgency_for(code: str) -> Urgency:
    """
    Return the urgency category for a SAME/EAS event code.
    Unknown codes default to "Future" (voice-only / lowest priority).

        urgency_for("TOR")   # → "Immediate"
        urgency_for("CFW")   # → "Expected"
        urgency_for("RWT")   # → "Past"
        urgency_for("???")   # → "Future"
    """
    entry = _TABLE.get((code or "").strip().upper())
    return entry[1] if entry is not None else "Future"


def is_known(code: str) -> bool:
    """True if the code is in the standard SAME/EAS event code table."""
    return (code or "").strip().upper() in _TABLE


def all_codes() -> list[str]:
    """Return all standard SAME/EAS event codes, sorted alphabetically."""
    return sorted(_TABLE.keys())


def label_or_code(code: str) -> str:
    """
    Return the label for a code, or the code itself if unknown.
    Convenient for display contexts where falling back to the raw code
    is acceptable.

        label_or_code("TOR")   # → "Tornado Warning"
        label_or_code("XYZ")   # → "XYZ"
    """
    c = (code or "").strip().upper()
    entry = _TABLE.get(c)
    return entry[0] if entry is not None else (c or "Unknown")


def resolve(code: str, *, custom: dict[str, str] | None = None) -> str:
    """
    Resolve a SAME/EAS event code to a human label, with optional custom
    overrides applied first.

    This is the preferred call for AlertTracker display names and Discord
    embed event labels. Custom codes from config are checked first, then
    the standard table, then the raw code is returned as a last resort.

        resolve("TOR")                          # → "Tornado Warning"
        resolve("XYZ", custom={"XYZ": "My Local Alert"})  # → "My Local Alert"
        resolve("XYZ")                          # → "XYZ"
    """
    c = (code or "").strip().upper()
    if custom:
        hit = custom.get(c)
        if hit:
            return str(hit)
    entry = _TABLE.get(c)
    if entry is not None:
        return entry[0]
    return c or "Unknown"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    """
    Look up one or more SAME/EAS event codes, or print the full table.

    Usage:
        python -m seasonalweather.same.events
            → prints the full table (code, label, urgency)

        python -m seasonalweather.same.events TOR CFW CFA RWT
            → looks up each code
    """
    args = [a.strip().upper() for a in sys.argv[1:] if a.strip()]

    if not args:
        # Full table
        print(f"{'CODE':<6}  {'URGENCY':<12}  LABEL")
        print("-" * 60)
        for code in all_codes():
            lbl, urg = _TABLE[code]
            print(f"{code:<6}  {urg:<12}  {lbl}")
        print(f"\n{len(_TABLE)} codes total.")
        return

    for code in args:
        c = code.upper()
        entry = _TABLE.get(c)
        if entry:
            lbl, urg = entry
            print(f"{c}  →  {lbl}  [{urg}]")
        else:
            print(f"{c}  →  (not in standard table)")


if __name__ == "__main__":
    _cli()
