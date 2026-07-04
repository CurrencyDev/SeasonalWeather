from __future__ import annotations

import re
from typing import Any


def build_ipaws_script(ev: Any) -> str:
    """
    Build a NWR-style TTS script for an IPAWS civil alert.

    Format mirrors how real NWR handles NWEMs:
      The following message is transmitted at the request of [authority].
      [headline if useful]
      [description]
      [instruction, if distinct]

    The authority line is omitted only when the cleaned senderName is
    unusable AND no area description is available to anchor it.
    """
    authority = (ev.sender_name_clean or "").strip()
    event = (ev.event or "").strip()
    headline = (ev.headline or "").strip()
    description = (ev.description or "").strip()
    instruction = (ev.instruction or "").strip()

    # Normalize whitespace/newlines that appear in IPAWS description fields.
    def _norm(s: str, limit: int = 900) -> str:
        s2 = re.sub(r"[\r\n]+", " ", s)
        s2 = re.sub(r"\s{2,}", " ", s2).strip()
        if len(s2) > limit:
            s2 = s2[:limit].rstrip() + "..."
        return s2

    description = _norm(description, 900)
    instruction = _norm(instruction, 600)
    headline = _norm(headline, 280)

    lines: list[str] = []

    # Preamble line.
    if authority:
        lines.append(
            f"The following message is transmitted at the request of {authority}."
        )
    else:
        # Fallback when senderName is generic or absent.
        lines.append("The following message is transmitted at the request of local authorities.")

    # Headline — only include when it adds information beyond the event name.
    # NWS-style "Civil Emergency Message" headlines are redundant; the real
    # content is in description.  But some senders write a useful summary
    # (e.g. "Tornado Watch in effect until 10pm for Worth County").
    hl_lower = headline.lower()
    ev_lower = event.lower()
    if headline and hl_lower != ev_lower and not hl_lower.startswith(ev_lower):
        lines.append(headline if headline.endswith((".", "!", "?")) else headline + ".")

    # Body.
    if description:
        lines.append(description)

    # Instruction — skip if it's a verbatim repeat of the description.
    if instruction and instruction.lower() != description.lower():
        lines.append("Instructions.")
        lines.append(instruction)

    return "\n".join(ln.strip() for ln in lines if ln.strip()).strip()
