"""
Alert-focus hold policy.

The active-alert registry answers "what should remain in cycle rotation".
This module answers the narrower question "what active alert is important
and short-fused enough to keep the conductor in alert-focus mode".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol


DEFAULT_EXCLUDED_SOURCES: tuple[str, ...] = ("PNS_CYCLE",)
DEFAULT_TEST_EVENT_CODES: tuple[str, ...] = ("DMO", "NAT", "NPT", "NST", "RMT", "RWT")
DEFAULT_HOLD_VTEC_SIGNIFICANCE: tuple[str, ...] = ("W", "A")
DEFAULT_HOLD_EVENT_CODES: tuple[str, ...] = (
    "TOA", "TOR", "SVA", "SVR", "SMW", "FFA", "FFW", "FLW",
    "TRA", "TRW", "HUA", "HUW", "WSW", "BZW", "SQW", "HWW",
    "CFW", "CDW", "CEM", "EAN", "EWW", "EQW", "EVI", "FRW",
    "HMW", "LAE", "LEW", "NUW", "RHW", "SPW", "TOE",
)

# VTEC/SAME marine hazards that may be long-lived and should not pin the
# station in alert-focus unless explicitly allowed below.  SMW is included so
# the marine guard stays explicit: it is marine, but short-fused/life-safety.
DEFAULT_MARINE_EVENT_CODES: tuple[str, ...] = (
    "SMW",  # Special Marine Warning — allowed by DEFAULT_MARINE_HOLD_EVENT_CODES
    "MWS",  # Marine Weather Statement
    "SCY",  # Small Craft Advisory
    "GLW",  # Gale Warning
    "GLA",  # Gale Watch
    "SRW",  # Storm Warning
    "SRA",  # Storm Watch
    "HSW",  # Hazardous Seas Warning
    "HSA",  # Hazardous Seas Watch
    "HFW",  # Hurricane Force Wind Warning
    "HFA",  # Hurricane Force Wind Watch
    "MHY",  # Marine Hurricane Force Wind Advisory
)
DEFAULT_MARINE_HOLD_EVENT_CODES: tuple[str, ...] = ("SMW",)


class FocusAlertLike(Protocol):
    source: str
    event: str
    code: str
    headline: str

    def vtec_track_ids(self) -> list[str]: ...


@dataclass(frozen=True)
class AlertFocusPolicy:
    """
    Configurable allowlist for active alerts that may hold alert-focus mode.

    ``hold_event_codes`` should normally inherit the operational tone-out SAME
    code policy, minus test codes.  VTEC-bearing alerts must also have at least
    one allowed significance, so advisory/statement VTEC products cannot hold
    focus just because they remain active in AlertTracker.
    """

    hold_event_codes: tuple[str, ...] = DEFAULT_HOLD_EVENT_CODES
    excluded_sources: tuple[str, ...] = DEFAULT_EXCLUDED_SOURCES
    test_event_codes: tuple[str, ...] = DEFAULT_TEST_EVENT_CODES
    hold_vtec_significance: tuple[str, ...] = DEFAULT_HOLD_VTEC_SIGNIFICANCE
    marine_event_codes: tuple[str, ...] = DEFAULT_MARINE_EVENT_CODES
    marine_hold_event_codes: tuple[str, ...] = DEFAULT_MARINE_HOLD_EVENT_CODES


def _upper_set(values: Iterable[str] | None) -> set[str]:
    return {str(v).strip().upper() for v in (values or ()) if str(v).strip()}


def _vtec_significances(alert: FocusAlertLike) -> set[str]:
    out: set[str] = set()
    try:
        track_ids = alert.vtec_track_ids()
    except Exception:
        track_ids = []
    for tid in track_ids or []:
        parts = str(tid or "").strip().upper().split(".")
        if len(parts) == 4 and parts[2]:
            out.add(parts[2])
    return out


def alert_holds_focus(alert: FocusAlertLike, policy: AlertFocusPolicy) -> bool:
    """Return True if this active alert should keep alert-focus mode active."""
    source = str(getattr(alert, "source", "") or "").strip().upper()
    code = str(getattr(alert, "code", "") or "").strip().upper()
    headline = str(getattr(alert, "headline", "") or "")
    event = str(getattr(alert, "event", "") or "")
    text = f"{event} {headline}".strip().lower()

    if source in _upper_set(policy.excluded_sources):
        return False
    if not code:
        return False
    if code in _upper_set(policy.test_event_codes):
        return False
    if source == "ERN" and "test" in text:
        return False

    hold_codes = _upper_set(policy.hold_event_codes)
    if code not in hold_codes:
        return False

    # For VTEC-bearing products, warning/watch significance is required.  This
    # keeps advisories/statements/lower-level products in the cycle without
    # letting them pin alert-focus mode for hours or days.
    sigs = _vtec_significances(alert)
    if sigs and sigs.isdisjoint(_upper_set(policy.hold_vtec_significance)):
        return False

    marine_codes = _upper_set(policy.marine_event_codes)
    marine_hold_codes = _upper_set(policy.marine_hold_event_codes)
    if code in marine_codes and code not in marine_hold_codes:
        return False

    return True
