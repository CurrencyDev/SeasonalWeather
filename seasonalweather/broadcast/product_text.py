"""
product_text.py — NWS product text helpers and alert script builders.

This module owns:
  - Pure text utilities shared across the alert pipeline (clean_cap_text,
    join_oxford, parse_cap_area_by_state, etc.)
  - CAP-path script builders (build_statement_vtec_action_script,
    build_warning_vtec_action_script)
  - NWWS-path helpers (expiry_summary_script, cap_prefers_statement_update_script,
    build_nwws_partial_cancel_script, parse_nwws_product_segments)
  - Existing helpers (cap_statement_intro, cap_statement_area_noun, etc.)

Design rules:
  - No imports from seasonalweather.main.  Pure stdlib + intra-package only.
  - No side effects.  All functions are pure / stateless unless noted.
  - Orchestrator methods that previously did this work become thin shims that
    call these free functions, forwarding self._tz, self._cap_vtec_list(ev)
    etc. as explicit arguments.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Marine detection constants
# ---------------------------------------------------------------------------
_MARINE_UGC_RE = re.compile(r"\b(?:ANZ|AMZ|GMZ|LMZ|PHZ|PKZ|PZZ|SLZ)\d{3}\b", re.IGNORECASE)
_MARINE_AREA_HINTS = (
    "tidal potomac", "chesapeake bay", "atlantic coastal waters", "coastal waters",
    "patapsco river", "patuxent river", "harbor", "sound", "sounds", "inlet",
    "strait", "straits", "gulf", "ocean", "offshore", "nearshore", "open lake",
    "lake huron", "lake michigan", "lake superior", "lake erie", "marine",
)
_MARINE_PHEN = {"SC", "GL", "SR", "HF", "SE", "UP", "RB", "SI", "BW", "MF", "MH", "MS", "LO", "SU", "MA"}


# ---------------------------------------------------------------------------
# US state abbreviation -> full name
# ---------------------------------------------------------------------------
STATE_NAME_FULL: dict[str, str] = {
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


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_cap_text(s: str, *, limit: int = 900) -> str:
    """Normalise whitespace, collapse ellipses, strip stray AWIPS IDs."""
    s2 = (s or "").replace("\r", " ").replace("\n", " ")
    s2 = re.sub(r"\s+", " ", s2).strip()
    s2 = s2.replace("...", ". ").replace("..", ".")
    s2 = re.sub(r"^[A-Z][A-Z0-9]{4,7}\s+", "", s2)
    if len(s2) > limit:
        s2 = s2[:limit].rstrip() + "..."
    return s2


def join_oxford(items: list[str]) -> str:
    """Oxford-comma join: "a", "a and b", "a, b, and c"."""
    xs = [x.strip() for x in items if x and x.strip()]
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    if len(xs) == 2:
        return f"{xs[0]} and {xs[1]}"
    return ", ".join(xs[:-1]) + f", and {xs[-1]}"


def parse_cap_area_by_state(
    area_desc: str,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
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


# ---------------------------------------------------------------------------
# CAP statement / area helpers (pre-existing, preserved)
# ---------------------------------------------------------------------------

def cap_is_special_weather_statement(event: str | None) -> bool:
    return str(event or "").strip().lower() == "special weather statement"


def cap_normalize_nws_headline(parameters: dict[str, list[str]] | None) -> str:
    params = parameters or {}
    nws_hl_list = params.get("NWSheadline") or []
    nws_hl = str(nws_hl_list[0]).strip() if nws_hl_list else ""
    if nws_hl and nws_hl.isupper():
        nws_hl = nws_hl.capitalize()
    return nws_hl


cap_nwsheadline = cap_normalize_nws_headline  # back-compat alias


def _iter_param_values(parameters: dict[str, list[str]] | None, key: str) -> list[str]:
    params = parameters or {}
    vals = params.get(key) or []
    return [str(v).strip() for v in vals if str(v).strip()]


def _same_codes_from_event(ev: Any) -> list[str]:
    vals = getattr(ev, "same_fips", None) or []
    out: list[str] = []
    for v in vals:
        s = re.sub(r"\D+", "", str(v))
        if s:
            out.append(s.zfill(6))
    return out


def _same_codes_from_parameters(parameters: dict[str, list[str]] | None) -> list[str]:
    out: list[str] = []
    for raw in _iter_param_values(parameters, "SAME") + _iter_param_values(parameters, "FIPS6"):
        for part in re.split(r"[\s,;]+", raw):
            s = re.sub(r"\D+", "", part)
            if s:
                out.append(s.zfill(6))
    return out


def _has_marine_ugc(parameters: dict[str, list[str]] | None) -> bool:
    for raw in _iter_param_values(parameters, "UGC"):
        if _MARINE_UGC_RE.search(raw):
            return True
    return False


def _has_marine_vtec(vtec: list[str] | None) -> bool:
    for raw in vtec or []:
        m = re.search(r"/O\.[A-Z]+\.[A-Z]{4}\.([A-Z]{2})\.[A-Z]\.", str(raw).upper())
        if m and m.group(1) in _MARINE_PHEN:
            return True
    return False


def _looks_marine_text(area_desc: str | None) -> bool:
    text = str(area_desc or "").strip().lower()
    return bool(text) and any(hint in text for hint in _MARINE_AREA_HINTS)


def cap_statement_area_noun(
    *,
    event: str | None,
    area_desc: str | None,
    parameters: dict[str, list[str]] | None,
    vtec: list[str] | None,
    ev: Any | None = None,
) -> str:
    same_codes = _same_codes_from_parameters(parameters)
    if ev is not None:
        same_codes.extend(_same_codes_from_event(ev))
    if any(code.startswith("07") for code in same_codes):
        return "areas"
    if _has_marine_ugc(parameters):
        return "areas"
    if _has_marine_vtec(vtec):
        return "areas"
    if _looks_marine_text(area_desc):
        return "areas"
    return "counties"


def cap_area_label(ev: Any) -> str:
    """Back-compat alias."""
    return cap_statement_area_noun(
        event=getattr(ev, "event", None),
        area_desc=getattr(ev, "area_desc", None),
        parameters=getattr(ev, "parameters", {}) or {},
        vtec=getattr(ev, "vtec", None) or [],
        ev=ev,
    )


def cap_statement_intro(
    *,
    event: str | None,
    sent_iso: str | None,
    sps_preamble: Callable[[str | None], str],
) -> str:
    if cap_is_special_weather_statement(event):
        return sps_preamble(sent_iso)
    return "This is a statement from the National Weather Service."


def cap_uses_sps_preamble(ev: Any, event: str | None) -> bool:
    """Back-compat alias."""
    return cap_is_special_weather_statement(event)


def cap_full_opening_line(
    *,
    event: str | None,
    sent_iso: str | None,
    parameters: dict[str, list[str]] | None,
    sps_preamble: Callable[[str | None], str],
) -> str:
    if not str(event or "").strip():
        return ""
    if cap_is_special_weather_statement(event):
        return sps_preamble(sent_iso)
    nws_hl = cap_normalize_nws_headline(parameters)
    if nws_hl:
        return nws_hl if nws_hl.endswith((".", "!", "?")) else nws_hl + "."
    return f"{str(event).strip()}."


# ---------------------------------------------------------------------------
# Expiry / cancellation helpers
# ---------------------------------------------------------------------------

def cap_expiry_summary_line(text: str) -> str:
    """
    Extract a single-sentence expiry summary from product or headline text.
    Returns "" if no NWS expiry phrase is found.
    """
    src = str(text or "").strip()
    if not src:
        return ""
    flat = re.sub(r"\s+", " ", src)
    m = re.search(
        r"([^.]{0,220}\b(?:will expire|has expired|has been allowed to expire"
        r"|has ended|is no longer in effect|the threat has ended)\b[^.]{0,220}[.?!]?)",
        flat,
        flags=re.IGNORECASE,
    )
    if not m:
        return ""
    line = m.group(1).strip()
    if line and not line.endswith((".", "!", "?")):
        line += "."
    return line


def cap_prefers_statement_update_script(event: str, vtec_actions: set[str]) -> bool:
    """
    True when a CAP/NWWS event should use the lighter statement-style EXP/CAN
    narration rather than the warning-style builder.

    Applies to advisory, statement, and message class events only.
    Warnings (including Special Marine Warning / MA.W) return False.
    """
    e = (event or "").strip().lower()
    if not e:
        return False
    if not (vtec_actions & {"CAN", "EXP"}):
        return False
    return e.endswith("advisory") or e.endswith("statement") or e.endswith("message")


def expiry_summary_script(official_text: str) -> str | None:
    """
    Build a minimal voice script from an NWWS product that carries only an
    expiry/cancellation paragraph.  Returns None if no suitable sentence found.
    """
    flat = re.sub(r"\s+", " ", str(official_text or "")).strip()
    line = cap_expiry_summary_line(flat)
    if not line:
        return None
    return f"{line}\nEnd of message."


# ---------------------------------------------------------------------------
# CAP script builders (free functions — Orchestrator shims call these)
# ---------------------------------------------------------------------------

def build_statement_vtec_action_script(
    *,
    event: str,
    area_desc: str,
    description: str,
    headline: str,
    vtec: list[str],
    vtec_actions: set[str],
    parameters: dict | None,
    sps_preamble: Callable[[str | None], str],
    sent_iso: str | None = None,
) -> str:
    """
    Lighter-weight voice cut-in for advisory / statement / message EXP/CAN.
    Sounds like a short NWR-style statement rather than a full warning read.
    """
    event = clean_cap_text(event or "", limit=120)
    groups, order, misc = parse_cap_area_by_state(area_desc)

    def _county_segs() -> str:
        if not groups:
            return clean_cap_text(area_desc or "the affected areas", limit=400)
        parts: list[str] = []
        for st in order:
            st_full = STATE_NAME_FULL.get(st, st)
            county_list = join_oxford(groups[st])
            if county_list:
                parts.append(f"in {st_full}, {county_list}")
        if misc:
            parts.append(join_oxford(misc))
        return "; ".join(parts) if parts else clean_cap_text(area_desc or "the affected areas", limit=400)

    summary_line = ""
    if vtec_actions & {"EXP"}:
        summary_line = cap_expiry_summary_line(description) or cap_expiry_summary_line(headline)
        if not summary_line and event:
            summary_line = f"The {event} has expired."
    elif vtec_actions & {"CAN"}:
        summary_line = cap_expiry_summary_line(description) or cap_expiry_summary_line(headline)
        if not summary_line and event:
            summary_line = f"The {event} has been cancelled."

    lines: list[str] = []
    lines.append(cap_statement_intro(event=event, sent_iso=sent_iso, sps_preamble=sps_preamble))
    area_line = _county_segs()
    if area_line:
        area_noun = cap_statement_area_noun(
            event=event, area_desc=area_desc, parameters=parameters, vtec=vtec,
        )
        lines.append(f"For the following {area_noun}: {area_line}.")
    if summary_line:
        lines.append(summary_line)
    elif event:
        lines.append(f"The {event} has been updated.")
    lines.append("End of message.")
    return "\n".join(ln.strip() for ln in lines if ln and ln.strip()).strip()


def build_warning_vtec_action_script(
    *,
    event: str,
    headline: str,
    description: str,
    instruction: str,
    area_desc: str,
    vtec_actions: set[str],
    exp_phrase: str,
) -> str:
    """
    NWR-style voice script for VTEC update actions on warnings (non-watch).

    CON/EXT -> "remains in effect until"
    CAN     -> "has been cancelled"
    EXP     -> "has been allowed to expire"
    EXA/EXB -> "has been expanded"
    """
    event = clean_cap_text(event or "", limit=120)
    headline = clean_cap_text(headline or "", limit=280)
    description = clean_cap_text(description or "", limit=800)
    instruction = clean_cap_text(instruction or "", limit=400)

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
        if description:
            lines.append(description)
        if instruction:
            lines.append(instruction)

    elif vtec_actions & {"EXT"}:
        lines.append(f"The {event} has been extended.")
        if area_desc:
            lines.append(f"For the following areas: {area_desc}.")
        if exp_phrase:
            lines.append(f"This warning is now in effect until {exp_phrase}.")
        if description:
            lines.append(description)
        if instruction:
            lines.append(instruction)

    else:  # CON and anything else
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
        if description:
            lines.append(description)
        if instruction:
            lines.append(instruction)

    lines.append("End of message.")
    return "\n".join(ln.strip() for ln in lines if ln and ln.strip()).strip()


# ---------------------------------------------------------------------------
# NWWS multi-segment partial cancel machinery
# ---------------------------------------------------------------------------

_SEG_VTEC_RE = re.compile(
    r"/[A-Z]\.(?P<act>[A-Z]{3})\.[A-Z]{4}\.[A-Z0-9]{2}\.[A-Z]\.\d{4}\.",
    re.IGNORECASE,
)
_SEG_HEADLINE_RE = re.compile(r"^\.\.\.(.+?)\.\.\.$")
_SEG_UNTIL_RE = re.compile(r"\bUNTIL\s+([\d:]+\s*(?:AM|PM)\s+[A-Z]{2,4})", re.IGNORECASE)
_SEG_AREA_INTRO_RE = re.compile(
    r"^(?:The\s+affected\s+areas?\s*(?:were|are|include[sd]?)?|"
    r"For\s+the\s+following\s+areas?)[.\s]*",
    re.IGNORECASE,
)
_SEG_META_RE = re.compile(
    r"^(?:LAT\.\.\.LON|TIME\.\.\.MOT\.\.\.LOC|HAIL\.\.\.|WIND\.\.\.|NNNN)",
    re.IGNORECASE,
)
_SEG_PPA_RE = re.compile(r"^PRECAUTIONARY/PREPAREDNESS ACTIONS", re.IGNORECASE)
_SEG_LOC_RE = re.compile(r"^Locations?\s+(?:impacted|affected)\s+include", re.IGNORECASE)
_SEG_UGC_RE = re.compile(r"^[A-Z]{2}[CZ]\d{3}(?:-\d{3})*-\d{6}-?$")
_SEG_TIMESTAMP_RE = re.compile(r"^\d{3,4}\s+(?:AM|PM)\s+[A-Z]{2,4}")
_TZ_FIX_RE = re.compile(r"\b(Edt|Est|Cdt|Cst|Mdt|Mst|Pdt|Pst|Akdt|Akst|Hst)\b")
_AMPM_FIX_RE = re.compile(r"\b(Am|Pm)\b")


def _fix_headline_case(h: str) -> str:
    """ALL-CAPS NWS headline → sentence case, preserving TZ abbreviations."""
    if not h.isupper():
        return h
    h = h.capitalize()
    h = _TZ_FIX_RE.sub(lambda m: m.group(1).upper(), h)
    h = _AMPM_FIX_RE.sub(lambda m: m.group(1).upper(), h)
    return h


@dataclass
class NwwsProductSegment:
    """One $$-delimited section of a multi-segment NWWS product."""
    actions: set[str]       # VTEC action codes present (e.g. {"CAN"}, {"CON"})
    headline: str           # Cleaned headline from ...X... line
    area_text: str          # Pipe-joined geographic area names
    reason_text: str        # Why the event is occurring/ending (narrative prose)
    precautions: str        # PRECAUTIONARY/PREPAREDNESS content
    expiry_phrase: str      # e.g. "315 PM EDT" extracted from headline


def parse_nwws_product_segments(product_text: str) -> list[NwwsProductSegment]:
    """
    Split a multi-section NWWS product on && delimiters and parse each section.

    NWS products use && to separate UGC/VTEC sections within a product and $$
    to close the product.  We split on && so each UGC+VTEC block becomes its
    own segment.  Sections without any VTEC lines are silently skipped.

    Used to generate per-action narration for products that carry mixed VTEC
    actions (e.g. a partial CAN+CON in an MWS where some zones are cancelled
    while others continue).
    """
    segments: list[NwwsProductSegment] = []

    # Strip everything after the first $$ (LAT/LON block, signature) then
    # split on && to get per-UGC-zone sections.
    body = re.split(r"\$\$", product_text, maxsplit=1)[0]
    raw_sections = re.split(r"&&", body)

    for raw_sec in raw_sections:
        actions: set[str] = set()
        for m in _SEG_VTEC_RE.finditer(raw_sec):
            actions.add(m.group("act").upper())
        if not actions:
            continue

        lines = [ln.rstrip() for ln in raw_sec.splitlines()]

        # Headline
        headline = ""
        headline_idx = -1
        for i, ln in enumerate(lines):
            hm = _SEG_HEADLINE_RE.match(ln.strip())
            if hm:
                headline = _fix_headline_case(hm.group(1).strip()).rstrip(".")
                headline_idx = i
                break

        # Expiry phrase from headline
        expiry_phrase = ""
        if headline:
            um = _SEG_UNTIL_RE.search(headline.upper())
            if um:
                expiry_phrase = um.group(1).strip()

        # Area text: lines after the area intro phrase, until blank line
        area_parts: list[str] = []
        in_area = False
        area_done = False
        for ln in lines:
            s = ln.strip()
            if area_done:
                break
            if _SEG_AREA_INTRO_RE.match(s):
                in_area = True
                rest = _SEG_AREA_INTRO_RE.sub("", s).strip().rstrip(".")
                if rest:
                    area_parts.append(rest)
                continue
            if in_area:
                if not s:
                    area_done = True
                    continue
                if s.endswith("...") or (re.match(r"^[A-Z]", s) and not re.match(r"^At \d", s, re.IGNORECASE)):
                    area_parts.append(s.rstrip(".").strip())
                else:
                    area_done = True

        area_text = "; ".join(p for p in area_parts if p)

        # Body / reason lines and precautionary text
        body_parts: list[str] = []
        precaution_parts: list[str] = []
        in_precautions = False
        in_area_skip = False

        start = headline_idx + 1 if headline_idx >= 0 else 0
        for ln in lines[start:]:
            s = ln.strip()
            if s.startswith("&&") or s.startswith("$$"):
                break
            if not s:
                in_area_skip = False
                continue
            if _SEG_UGC_RE.match(s) or _SEG_TIMESTAMP_RE.match(s):
                continue
            if s.startswith("/") and s.endswith("/") and "." in s:
                continue
            if _SEG_HEADLINE_RE.match(s) or _SEG_META_RE.match(s):
                continue
            if _SEG_PPA_RE.match(s):
                in_precautions = True
                continue
            if _SEG_AREA_INTRO_RE.match(s):
                in_area_skip = True
                continue
            if _SEG_LOC_RE.match(s):
                continue
            if in_area_skip:
                if s.endswith("...") or re.match(r"^[A-Z][a-z]+,", s):
                    continue
                in_area_skip = False
            tag_m = re.match(r"^([A-Z]+)\.\.\.(.*?)\.?$", s)
            if tag_m and tag_m.group(1) in {"HAZARD", "SOURCE", "IMPACT"}:
                body_parts.append(
                    f"{tag_m.group(1).capitalize()}: {tag_m.group(2).strip().rstrip('.')}"
                )
                continue
            if in_precautions:
                precaution_parts.append(s.rstrip("."))
            else:
                body_parts.append(s.rstrip("."))

        segments.append(NwwsProductSegment(
            actions=actions,
            headline=headline,
            area_text=area_text,
            reason_text=" ".join(body_parts).strip(),
            precautions=" ".join(precaution_parts).strip(),
            expiry_phrase=expiry_phrase,
        ))

    return [s for s in segments if s.actions]


def build_nwws_partial_cancel_script(
    event_label: str,
    segments: list[NwwsProductSegment],
) -> str:
    """
    Build a voice script for a partial CAN+CON NWWS product.

    For example, an MWS where one zone is cancelled while other zones
    continue:

        "The Special Marine Warning has been cancelled for the Tidal Potomac
         from Cobb Island MD to Smith Point VA. The thunderstorms have moved
         out of the warned area and no longer pose a significant threat.
         The Special Marine Warning remains in effect until 3:15 PM Eastern
         Daylight Time for the Chesapeake Bay from Drum Point MD to Smith
         Point VA, and Tangier Sound. Move to safe harbor."
    """
    event = (event_label or "Weather Alert").strip()
    lines: list[str] = []

    can_segs = [s for s in segments if "CAN" in s.actions]
    exp_segs = [s for s in segments if "EXP" in s.actions]
    con_segs = [s for s in segments if "CON" in s.actions or "EXT" in s.actions]

    for seg in can_segs:
        area = seg.area_text or "some areas"
        lines.append(f"The {event} has been cancelled for the following areas: {area}.")
        if seg.reason_text:
            lines.append(f"{seg.reason_text.rstrip('.')}.")

    for seg in exp_segs:
        area = seg.area_text or "some areas"
        lines.append(f"The {event} has been allowed to expire for the following areas: {area}.")
        if seg.reason_text:
            lines.append(f"{seg.reason_text.rstrip('.')}.")

    for seg in con_segs:
        area = seg.area_text or "other areas"
        exp_part = f" until {seg.expiry_phrase}" if seg.expiry_phrase else ""
        lines.append(
            f"The {event} remains in effect{exp_part} "
            f"for the following areas: {area}."
        )
        if seg.reason_text:
            lines.append(f"{seg.reason_text.rstrip('.')}.")
        if seg.precautions:
            lines.append(f"{seg.precautions.rstrip('.')}.")

    if not lines:
        return ""
    lines.append("End of message.")
    return "\n".join(ln.strip() for ln in lines if ln and ln.strip()).strip()


__all__ = [
    # Constants
    "STATE_NAME_FULL",
    # Text utilities
    "clean_cap_text",
    "join_oxford",
    "parse_cap_area_by_state",
    # CAP helpers (pre-existing)
    "cap_area_label",
    "cap_expiry_summary_line",
    "cap_full_opening_line",
    "cap_is_special_weather_statement",
    "cap_normalize_nws_headline",
    "cap_nwsheadline",
    "cap_prefers_statement_update_script",
    "cap_statement_area_noun",
    "cap_statement_intro",
    "cap_uses_sps_preamble",
    # Script builders
    "build_statement_vtec_action_script",
    "build_warning_vtec_action_script",
    # NWWS helpers
    "expiry_summary_script",
    "NwwsProductSegment",
    "parse_nwws_product_segments",
    "build_nwws_partial_cancel_script",
]
