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



def strip_nws_product_headers(raw: str) -> str:
    """
    Remove NWS/WMO/AWIPS + UGC/VTEC boilerplate that often appears at the top of
    api.weather.gov/products/* productText (especially NWWS paths).

    Goal: prevent zone codes (e.g., VAZ025-027-...) and VTEC lines (/O.CON.../)
    from leaking into spoken/TTS output.

    Safe-by-default: if patterns aren't found, returns input mostly unchanged.
    """
    if not raw:
        return raw or ""

    s = raw.replace("\r\n", "\n").replace("\r", "\n")

    lines = s.split("\n")
    i = 0

    def is_wmo_header(line: str) -> bool:
        # e.g. "WWUS41 KLWX 180551"
        return bool(re.match(r"^[A-Z]{4}\d{2}\s+[A-Z]{4}\s+\d{6}$", line.strip()))

    def is_awips_id(line: str) -> bool:
        # e.g. "WSWLWX" (varies; keep broad but not too broad)
        t = line.strip()
        return bool(re.match(r"^[A-Z0-9]{3,16}$", t))

    def has_ugc_codes(line: str) -> bool:
        # UGC contains patterns like VAZ025, MDZ011, DCZ001 etc (Z or C)
        return bool(re.search(r"\b[A-Z]{2}[CZ]\d{3}\b", line))

    def is_vtec_line(line: str) -> bool:
        t = line.strip()
        # VTEC lines are typically wrapped in slashes and start with /O. or /T.
        return (t.startswith("/O.") or t.startswith("/T.") or t.startswith("/E.")) and t.endswith("/")

    def is_noise_line(line: str) -> bool:
        t = line.strip()
        return t in {"", "NNNN", "$$", "&&"} or t.isdigit()

    # Drop leading empties/noise
    while i < len(lines) and is_noise_line(lines[i]):
        i += 1

    # Drop WMO header + AWIPS if present
    if i < len(lines) and is_wmo_header(lines[i]):
        i += 1
        # sometimes there is a blank line after
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        if i < len(lines) and is_awips_id(lines[i]):
            # Only drop if next looks like header-ish too (blank/UGC/VTEC)
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt == "" or has_ugc_codes(nxt) or nxt.startswith("/O."):
                i += 1

    # Drop subsequent blanks
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    # Drop UGC lines at top (may wrap across multiple lines)
    # Typical: "VAZ025-027-...-180700-"
    while i < len(lines):
        t = lines[i].strip()
        if not t:
            i += 1
            continue
        if has_ugc_codes(t):
            i += 1
            # Some UGC blocks wrap; keep skipping while UGC-looking continues
            while i < len(lines) and (has_ugc_codes(lines[i]) or lines[i].strip().endswith("-")):
                i += 1
            continue
        break

    # Drop VTEC lines immediately after UGC (often multiple)
    while i < len(lines) and is_vtec_line(lines[i]):
        i += 1

    # Drop a blank line after headers if present
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    # Now keep the rest, but scrub stray header artifacts anywhere (belt+suspenders)
    out: list[str] = []
    for ln in lines[i:]:
        t = ln.strip()
        if t in {"NNNN"}:
            continue
        if is_vtec_line(ln):
            continue
        # Occasionally a UGC line can reappear in relayed products; drop it.
        if has_ugc_codes(ln) and re.search(r"-\d{6}\b", ln):
            continue
        # Drop SAME ZCZC framing if it shows up in productText (rare but possible)
        if t.startswith("ZCZC-") and len(t) < 120:
            continue
        out.append(ln)

    # Trim excessive leading/trailing blank lines
    cleaned = "\n".join(out).strip("\n")
    return cleaned
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
    # Prefer the NWS headline marker (often where meaningful narration begins)
    for i, ln in enumerate(lines):
        s = (ln or "").strip()
        if s.startswith("...") and s.endswith("...") and len(s) >= 12:
            return i

    # Otherwise prefer the normal NWS narrative intro
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
    official_text = strip_nws_product_headers(official_text)
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
    script = clean_for_tts(strip_nws_product_headers(script_raw))
    return SpokenAlert(title=title, script=script)


# Keep the old name so main.py doesn’t need changes.
# NWWS should be full-length like CAP/NWR now.
def build_spoken_alert(parsed: ParsedProduct, official_text: str) -> SpokenAlert:
    return build_spoken_alert_full(parsed, official_text)
