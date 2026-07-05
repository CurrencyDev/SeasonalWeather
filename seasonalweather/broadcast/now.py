"""Short-Term Forecast (NOW) product text handling.

NOW products are routine, geographically targeted statements.  They are not
alert tone-out products and do not belong in AlertTracker.  This module keeps
text extraction pure so the runtime can safely turn an accepted NOW product
into an expiring cycle insert.
"""
from __future__ import annotations

import re

from ..tts.tts import clean_for_tts

_NOW_MARKER_RE = re.compile(r"^\s*\.NOW\.\.\.\s*$", re.IGNORECASE)
_NOW_STOP_RE = re.compile(r"^\s*(?:&&|\$\$|NNNN)\s*$", re.IGNORECASE)
_NOW_MACHINE_BLOCK_RE = re.compile(
    r"^(?:"
    r"LAT\.\.\.LON|"
    r"TIME\.\.\.MOT\.\.\.LOC|"
    r"TORNADO\.\.\.|"
    r"TORNADO DAMAGE THREAT\.\.\.|"
    r"THUNDERSTORM DAMAGE THREAT\.\.\.|"
    r"FLASH FLOOD DAMAGE THREAT\.\.\.|"
    r"DAMAGE THREAT\.\.\.|"
    r"HAIL THREAT\.\.\.|"
    r"MAX HAIL SIZE\.\.\.|"
    r"WIND THREAT\.\.\.|"
    r"MAX WIND GUST\.\.\.|"
    r"EXPECTED RAINFALL RATE\.\.\.|"
    r"RAINFALL AMOUNT\.\.\."
    r")",
    re.IGNORECASE,
)
_LOCATIONS_INCLUDE_RE = re.compile(
    r"^(Locations?\s+(?:impacted|affected)\s+include)\.{3}\s*$",
    re.IGNORECASE,
)


def extract_now_narrative(product_text: str) -> str:
    """Return only the human-readable body after the standard ``.NOW...`` marker.

    The extractor deliberately fails closed when the marker is absent.  Reading
    from an inferred offset risks sending routing headers, UGC codes, or other
    machine fields to TTS.  Terminal machine-readable blocks are discarded even
    when the office omits the usual ``&&`` delimiter.
    """
    lines = (product_text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()

    marker_index = next(
        (idx for idx, raw in enumerate(lines) if _NOW_MARKER_RE.match(raw)),
        None,
    )
    if marker_index is None:
        return ""

    paragraphs: list[str] = []
    current: list[str] = []

    def _flush() -> None:
        if not current:
            return
        paragraphs.append(" ".join(current).strip())
        current.clear()

    for raw in lines[marker_index + 1 :]:
        line = raw.strip()
        if _NOW_STOP_RE.match(line) or _NOW_MACHINE_BLOCK_RE.match(line):
            break
        if not line:
            _flush()
            continue

        loc_match = _LOCATIONS_INCLUDE_RE.match(line)
        if loc_match:
            line = f"{loc_match.group(1)}:"
        current.append(line)

    _flush()
    return "\n".join(p for p in paragraphs if p).strip()


def build_now_script(product_text: str, *, intro: str) -> str:
    """Build TTS-ready routine-cycle narration for a NOW product."""
    body = extract_now_narrative(product_text)
    if not body:
        return ""

    lead = (intro or "A statement from the National Weather Service.").strip()
    if lead and not lead.endswith((".", "!", "?")):
        lead += "."

    return clean_for_tts(f"{lead}\n{body}").strip()
