from __future__ import annotations

import datetime as dt
from typing import Sequence

from ..same.events import label_or_code, org_broadcast_prefix


_TEST_EVENT_CODES = {"DMO", "NAT", "NPT", "NST", "RMT", "RWT"}


def _article(word: str) -> str:
    w = str(word or "").strip()
    return "an" if w[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def _sentence(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    return s if s.endswith((".", "!", "?")) else s + "."


def _join_human(items: Sequence[str]) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for raw in items or []:
        s = str(raw or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        values.append(s)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _parse_duration_minutes(tttt: str | None) -> int | None:
    raw = str(tttt or "").strip()
    if len(raw) != 4 or not raw.isdigit():
        return None
    hours = int(raw[:2])
    minutes = int(raw[2:])
    if minutes > 59:
        return None
    return hours * 60 + minutes


def _same_jday_to_utc(jjjhhmm: str | None, *, now_utc: dt.datetime | None = None) -> dt.datetime | None:
    raw = str(jjjhhmm or "").strip()
    if len(raw) != 7 or not raw.isdigit():
        return None

    now = now_utc or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    now = now.astimezone(dt.timezone.utc)

    jday = int(raw[:3])
    hour = int(raw[3:5])
    minute = int(raw[5:7])
    if jday < 1 or jday > 366 or hour > 23 or minute > 59:
        return None

    candidates: list[dt.datetime] = []
    for year in (now.year - 1, now.year, now.year + 1):
        try:
            base = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
            candidates.append(base + dt.timedelta(days=jday - 1, hours=hour, minutes=minute))
        except Exception:
            continue
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: abs((candidate - now).total_seconds()))


def _fmt_when(value: dt.datetime, *, tz: dt.tzinfo | None = None) -> str:
    target = value
    if target.tzinfo is None:
        target = target.replace(tzinfo=dt.timezone.utc)
    if tz is not None:
        target = target.astimezone(tz)
    else:
        target = target.astimezone()
    try:
        return target.strftime("%-I:%M %p %Z on %A, %B %-d")
    except Exception:
        return target.isoformat()


def build_ern_relay_script(
    ev,
    *,
    same_locations: Sequence[str] | None = None,
    area_text: str = "",
    tz: dt.tzinfo | None = None,
    now_utc: dt.datetime | None = None,
) -> str:
    """
    Build the spoken script for an ERN/GWES SAME relay.

    ERN/GWES gives SeasonalWeather decoded SAME metadata, not the full official
    alert body. This intentionally speaks the trustworthy header fields only:
    event, originator, area, effective/purge timing, and sender.
    """
    code = str(getattr(ev, "event", None) or "").strip().upper()
    event_label = label_or_code(code) if code else "EAS Alert"
    article = _article(event_label)
    sender = str(getattr(ev, "sender", None) or "").strip()
    org = str(getattr(ev, "org", None) or "").strip().upper()

    if code in _TEST_EVENT_CODES:
        intro = (
            f"This is a relay of {article} {event_label} received via the Emergency Relay Network. "
            "This is only a test."
        )
    else:
        intro = f"This is a relay of {article} {event_label} received via the Emergency Relay Network."

    lines: list[str] = [intro]

    if org:
        lines.append(f"{org_broadcast_prefix(org)} {article} {event_label}.")

    area = str(area_text or "").strip()
    if area:
        lines.append(_sentence(f"The message applies to the following locations: {area}"))
    else:
        same_text = _join_human([str(x).strip() for x in (same_locations or []) if str(x).strip()])
        if same_text:
            lines.append(_sentence(f"The message applies to the following SAME locations: {same_text}"))

    start_utc = _same_jday_to_utc(getattr(ev, "jjjhhmm", None), now_utc=now_utc)
    duration_min = _parse_duration_minutes(getattr(ev, "tttt", None))
    if start_utc is not None and duration_min is not None:
        end_utc = start_utc + dt.timedelta(minutes=duration_min)
        lines.append(
            f"The message is valid from: {_fmt_when(start_utc, tz=tz)}. "
            f"And the message is valid until: {_fmt_when(end_utc, tz=tz)}."
        )
    elif start_utc is not None:
        lines.append(f"The message is valid from: {_fmt_when(start_utc, tz=tz)}.")

    if sender:
        lines.append(f"The message was received from: {sender}.")

    lines.append("End of message.")
    return "\n".join(line.strip() for line in lines if line and line.strip())
