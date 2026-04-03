from __future__ import annotations

import re
from typing import Any, Callable

_MARINE_UGC_RE = re.compile(r"\b(?:ANZ|AMZ|GMZ|LMZ|PHZ|PKZ|PZZ|SLZ)\d{3}\b", re.IGNORECASE)
_MARINE_AREA_HINTS = (
    "tidal potomac",
    "chesapeake bay",
    "atlantic coastal waters",
    "coastal waters",
    "patapsco river",
    "patuxent river",
    "harbor",
    "sound",
    "sounds",
    "inlet",
    "strait",
    "straits",
    "gulf",
    "ocean",
    "offshore",
    "nearshore",
    "open lake",
    "lake huron",
    "lake michigan",
    "lake superior",
    "lake erie",
    "marine",
)
# Common VTEC phenomena used for marine/coastal marine products. This is only
# a supplement to the stronger SAME/UGC checks above.
_MARINE_PHEN = {"SC", "GL", "SR", "HF", "SE", "UP", "RB", "SI", "BW", "MF", "MH", "MS", "LO", "SU"}


def cap_is_special_weather_statement(event: str | None) -> bool:
    return str(event or "").strip().lower() == "special weather statement"


def cap_normalize_nws_headline(parameters: dict[str, list[str]] | None) -> str:
    params = parameters or {}
    nws_hl_list = params.get("NWSheadline") or []
    nws_hl = str(nws_hl_list[0]).strip() if nws_hl_list else ""
    if nws_hl and nws_hl.isupper():
        nws_hl = nws_hl.capitalize()
    return nws_hl


# Back-compat alias for an earlier helper name used by one patch generation.
cap_nwsheadline = cap_normalize_nws_headline


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


# Back-compat alias for an earlier helper name used by one patch generation.
def cap_area_label(ev: Any) -> str:
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


# Back-compat alias for an earlier helper name used by one patch generation.
def cap_uses_sps_preamble(ev: Any, event: str | None) -> bool:
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


__all__ = [
    "cap_area_label",
    "cap_full_opening_line",
    "cap_is_special_weather_statement",
    "cap_normalize_nws_headline",
    "cap_nwsheadline",
    "cap_statement_area_noun",
    "cap_statement_intro",
    "cap_uses_sps_preamble",
]
