from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .product import ParsedProduct


_STAR_RE = re.compile(r"^\s*\*\s+")
_META_SKIP_PREFIXES = (
    "LAT...LON", "TIME...MOT...LOC", "MAX HAIL", "MAX WIND", "&&", "$$"
)


def _is_meta_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith(_META_SKIP_PREFIXES):
        return True
    # raw cap headers often include "000" and UGC blocks; skip those
    if re.fullmatch(r"[0-9]{3,}", s):
        return True
    if re.fullmatch(r"[A-Z]{2,}-[0-9]{3,}.*", s):
        return False
    return False


@dataclass
class SpokenAlert:
    title: str
    script: str


def build_spoken_alert(parsed: ParsedProduct, official_text: str) -> SpokenAlert:
    lines = [ln.rstrip() for ln in official_text.splitlines()]

    # Find the first "has issued" line (keeps NWS wording)
    issued_idx = None
    for i, ln in enumerate(lines[:120]):
        if "has issued" in ln.lower() and "national weather service" in " ".join(lines[max(0, i-2):i+1]).lower():
            issued_idx = i
            break
        if "the national weather service" in ln.lower() and "has issued" in ln.lower():
            issued_idx = i
            break

    picked: List[str] = []

    # Title
    product_name = parsed.product_type
    title = f"{product_name} from {parsed.wfo}"

    # Grab a compact top block around "issued"
    if issued_idx is not None:
        for ln in lines[max(0, issued_idx - 2):issued_idx + 12]:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("$$") or s.startswith("&&"):
                break
            # skip routing headers
            if re.match(r"^[A-Z]{4}\d{2}\s+[A-Z]{4}\s+\d{6}", s):
                continue
            if re.fullmatch(r"[A-Z]{6}", s):
                continue
            picked.append(s)

    # Bullet lines starting with "*"
    bullet_lines: List[str] = []
    for ln in lines:
        if _STAR_RE.match(ln):
            bullet_lines.append(_STAR_RE.sub("", ln.strip()))
    # Keep bullets but de-dupe
    for b in bullet_lines:
        if b and b not in picked:
            picked.append(b)

    # Hazard/source/impact lines
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith(("HAZARD", "SOURCE", "IMPACT")):
            picked.append(s)

    # Key “at time” line(s)
    for ln in lines:
        s = ln.strip()
        if re.match(r"^At\s+\d{1,4}\s*(AM|PM)\s+", s):
            picked.append(s)

    # “TAKE COVER NOW” etc
    for ln in lines:
        s = ln.strip()
        if "TAKE COVER" in s or "TAKE SHELTER" in s or "SEEK SHELTER" in s:
            picked.append(s)

    # Precautionary section
    prec = []
    in_prec = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("PRECAUTIONARY/PREPAREDNESS ACTIONS"):
            in_prec = True
            continue
        if in_prec:
            if s.startswith(("&&", "$$")):
                break
            if s and not any(s.startswith(p) for p in _META_SKIP_PREFIXES):
                prec.append(s)
    if prec:
        picked.append("Precautionary and preparedness actions.")
        picked.extend(prec[:12])

    # Cleanup and cap length
    cleaned: List[str] = []
    for ln in picked:
        s = ln.strip()
        if not s:
            continue
        if any(s.startswith(pfx) for pfx in _META_SKIP_PREFIXES):
            continue
        # skip UGC/time codes lines
        if re.fullmatch(r"[A-Z]{3}\d{3}-\d{6}-", s):
            continue
        if s not in cleaned:
            cleaned.append(s)

    # If we somehow got nothing useful, fallback to a trimmed version of the raw text.
    if len(cleaned) < 3:
        raw = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if s.startswith(("$$", "&&")):
                break
            if re.match(r"^[A-Z]{4}\d{2}\s+[A-Z]{4}\s+\d{6}", s):
                continue
            if re.fullmatch(r"[A-Z]{6}", s):
                continue
            raw.append(s)
        cleaned = raw[:60]

    script = "\n".join(cleaned[:80])
    return SpokenAlert(title=title, script=script)
