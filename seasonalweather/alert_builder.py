from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .product import ParsedProduct
from .tts import clean_for_tts

_STAR_RE = re.compile(r"^\s*\*\s+")
_SPACE_RE = re.compile(r"\s+")
_END_PUNCT_RE = re.compile(r"(\.\.\.|[.!?:;])$")

# UGC example: PAC009-013-112115-
_UGC_RE = re.compile(r"^[A-Z]{2}[CZ]\d{3}(?:-\d{3})*-\d{6}-?$")
# WMO example: WWUS51 KCTP 112100
_WMO_RE = re.compile(r"^[A-Z]{4}\d{2}\s+[A-Z]{4}\s+\d{6}$")
# "PRECAUTIONARY/PREPAREDNESS ACTIONS..." header (we don't speak the header)
_PPA_RE = re.compile(r"^PRECAUTIONARY/PREPAREDNESS ACTIONS\b", re.IGNORECASE)


_META_SKIP_PREFIXES = (
    "LAT...LON",
    "TIME...MOT...LOC",
    "MAX HAIL",
    "MAX WIND",
    "&&",
    "$$",
)

_TAGS = ("HAZARD", "SOURCE", "IMPACT")


@dataclass
class SpokenAlert:
    title: str
    script: str


def _unwrap_soft_wrap(lines: List[str]) -> List[str]:
    """
    Joins NWS soft-wrapped lines (usually indented continuations).
    Keeps true paragraph breaks.
    """
    out: List[str] = []
    for raw in lines:
        ln = (raw or "").rstrip("\n")
        if not ln.strip():
            out.append("")
            continue

        indent = len(ln) - len(ln.lstrip(" \t"))
        if indent >= 2 and out and out[-1].strip():
            prev = out[-1].rstrip()
            # Join if previous line doesn't look complete.
            if not _END_PUNCT_RE.search(prev):
                out[-1] = prev + " " + ln.strip()
                continue

        out.append(ln.rstrip())
    return out


def _collapse_blank_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        if ln == "":
            if out and out[-1] == "":
                continue
        out.append(ln)
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return out


def _find_body_start(lines: List[str]) -> int:
    # Prefer the normal NWS narrative intro (what NWR effectively reads)
    for i, ln in enumerate(lines):
        s = (ln or "").strip().lower()
        if s.startswith("the national weather service"):
            return i
    # Fallback: first “has issued”
    for i, ln in enumerate(lines):
        s = (ln or "").strip().lower()
        if "has issued" in s and "national weather service" in s:
            return i
    return 0


def _clean_line(s: str) -> str:
    s2 = _STAR_RE.sub("", (s or "").strip())

    # Never speak the PRECAUTIONARY header (but we DO speak the content after it)
    if _PPA_RE.match(s2):
        return ""

    # Convert marker ellipses to CAP-ish punctuation
    for tag in _TAGS:
        if s2.startswith(tag + "..."):
            rest = s2[len(tag + "...") :].lstrip()
            s2 = f"{tag}. {rest}" if rest else f"{tag}."
            break

    # If a line ends with "...", turn that into a sentence-ish period
    if s2.endswith("..."):
        s2 = s2[:-3].rstrip() + "."

    # Squash weird spacing
    s2 = _SPACE_RE.sub(" ", s2).strip()
    return s2


def build_spoken_alert_full(parsed: ParsedProduct, official_text: str) -> SpokenAlert:
    lines = _unwrap_soft_wrap([ln.rstrip() for ln in official_text.splitlines()])
    start = _find_body_start(lines)

    body: List[str] = []
    for ln in lines[start:]:
        s = (ln or "").strip()
        if not s:
            body.append("")
            continue

        if s.startswith(("&&", "$$")):
            break
        if any(s.startswith(pfx) for pfx in _META_SKIP_PREFIXES):
            continue
        if re.fullmatch(r"[0-9]{3,}", s):  # "000"
            continue
        if _WMO_RE.match(s):
            continue
        if re.fullmatch(r"[A-Z]{6}", s):  # "SQWCTP" etc
            continue
        if _UGC_RE.match(s):
            continue
        # VTEC line (starts with / and ends with /)
        if s.startswith("/") and s.endswith("/") and "." in s:
            continue

        cleaned = _clean_line(s)
        if cleaned:
            body.append(cleaned)


    body = _collapse_blank_lines(body)

    title = f"{parsed.product_type} from {parsed.wfo}"
    script_raw = "\n".join(body)

    # Make punctuation/spacing match what your CAP path tends to sound like.
    script = clean_for_tts(script_raw)
    return SpokenAlert(title=title, script=script)


# Keep the old name so main.py doesn’t need changes.
# NWWS should be full-length like CAP/NWR now.
def build_spoken_alert(parsed: ParsedProduct, official_text: str) -> SpokenAlert:
    return build_spoken_alert_full(parsed, official_text)
