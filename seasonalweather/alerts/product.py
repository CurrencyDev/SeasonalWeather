from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional


_WMO_RE = re.compile(r"^[A-Z]{4}\d{2}\s+[A-Z]{4}\s+\d{6}")
_AWIPS_RE = re.compile(r"^[A-Z]{3}[A-Z]{3}$")
_VTEC_RE = re.compile(r"/([OE])\.([A-Z]{3})\.([A-Z]{4})\.([A-Z]{2})\.([A-Z])\.(\d{4})\.(\d{6}T\d{4}Z)-(\d{6}T\d{4}Z)/")


@dataclass
class VTEC:
    kind: str  # O or E
    action: str
    office: str
    phenomena: str
    significance: str
    event_number: str
    start: str
    end: str

    @property
    def event_id(self) -> str:
        return f"{self.office}.{self.phenomena}{self.significance}.{self.event_number}"


@dataclass
class ParsedProduct:
    product_type: str
    wfo: str
    awips_id: Optional[str]
    vtec: Optional[VTEC]
    raw_text: str


def parse_product_text(text: str) -> Optional[ParsedProduct]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    # Find WMO line + WFO
    wfo = None
    for ln in lines[:10]:
        m = _WMO_RE.match(ln.strip())
        if m:
            parts = ln.split()
            if len(parts) >= 2:
                wfo = parts[1].strip()
            break

    # Find AWIPS ID
    awips = None
    for ln in lines[:25]:
        s = ln.strip()
        if _AWIPS_RE.match(s):
            awips = s
            break

    if awips:
        product_type = awips[:3]
        if not wfo:
            wfo = awips[3:]
    elif wfo:
        # Fallback: try infer product type from WMO prefix (e.g., WFUS51, WWUS51)
        product_type = None
        for ln in lines[:5]:
            if _WMO_RE.match(ln.strip()):
                product_type = None
                break
        # No clean fallback -> bail
        return None
    else:
        return None

    # Parse VTEC
    vtec = None
    for ln in lines[:80]:
        if "/" in ln and ".K" in ln:
            m = _VTEC_RE.search(ln)
            if m:
                vtec = VTEC(
                    kind=m.group(1),
                    action=m.group(2),
                    office=m.group(3),
                    phenomena=m.group(4),
                    significance=m.group(5),
                    event_number=m.group(6),
                    start=m.group(7),
                    end=m.group(8),
                )
                break

    return ParsedProduct(
        product_type=product_type,
        wfo=wfo or "UNK",
        awips_id=awips,
        vtec=vtec,
        raw_text=text,
    )
