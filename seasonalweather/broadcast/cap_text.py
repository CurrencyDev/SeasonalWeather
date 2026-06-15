from __future__ import annotations

import datetime as dt
import re
from typing import Any, Callable
from zoneinfo import ZoneInfo

from .product_text import (
    STATE_NAME_FULL as _STATE_NAME_FULL_MAP,
    build_statement_vtec_action_script as _build_statement_vtec_action_script_fn,
    build_warning_vtec_action_script as _build_warning_vtec_action_script_fn,
    cap_expiry_summary_line as _cap_expiry_summary_line,
    fmt_local_from_utc_iso as _fmt_local_from_utc_iso,
    cap_prefers_statement_update_script as _cap_prefers_statement_update_script_fn,
    build_nws_full_alert_script as _build_nws_full_alert_script,
    build_nws_voice_alert_script as _build_nws_voice_alert_script,
    NwsAlertTextInput,
    clean_cap_text as _pt_clean_cap_text,
    join_oxford as _pt_join_oxford,
    parse_cap_area_by_state as _pt_parse_cap_area_by_state,
    nws_header_issued_phrase as _pt_nws_header_issued_phrase,
    sps_preamble as _pt_sps_preamble,
)
from ..alerts.vtec import VTEC_PARSE_RE as _VTEC_PARSE_RE


class CapTextRenderer:
    _STATE_NAME_FULL: dict[str, str] = _STATE_NAME_FULL_MAP

    def __init__(
        self,
        *,
        local_tz: ZoneInfo,
        cap_vtec_list: Callable[[Any], list[str]],
        vtec_tracks: Callable[[list[str]], list[tuple[str, str]]],
        best_expiry_from_vtec: Callable[[list[str]], dt.datetime | None],
    ) -> None:
        self._tz = local_tz
        self._cap_vtec_list = cap_vtec_list
        self._vtec_tracks = vtec_tracks
        self._best_expiry_from_vtec = best_expiry_from_vtec

    def _nws_header_issued_phrase(self, text: str) -> str | None:
        return _pt_nws_header_issued_phrase(text)

    def _cap_sps_preamble(self, sent_iso: str | None) -> str:
        return _pt_sps_preamble(sent_iso, local_tz=self._tz)

    def _clean_cap_text(self, s: str, *, limit: int = 900) -> str:
        """Shim → product_text.clean_cap_text()."""
        return _pt_clean_cap_text(s, limit=limit)

    def _build_cap_watch_script(self, ev: "CapAlertEvent", *, mode: str = "full") -> str:  # type: ignore[name-defined]
        """
        Build a sane, NWR-style script for CAP Tornado Watch / Severe Thunderstorm Watch.
        Returns "" if this CAP event is not a watch.

        Why: CAP watch descriptions are often all-caps blobs with little punctuation,
        which TTS will speed-read. NWR uses a standardized narration instead.
        """
        # ---- Determine watch kind (prefer event label, fall back to VTEC) ----
        kind: str | None = None  # "tornado" or "severe"
        ev_name = (getattr(ev, "event", "") or "").strip().lower()

        if ev_name == "tornado watch":
            kind = "tornado"
        elif ev_name == "severe thunderstorm watch":
            kind = "severe"
        else:
            # Fall back to VTEC phen/sig
            for v in self._cap_vtec_list(ev):
                m = _VTEC_PARSE_RE.search(v)
                if not m:
                    continue
                phen = (m.group("phen") or "").upper()
                sig = (m.group("sig") or "").upper()
                if sig != "A":
                    continue
                if phen == "TO":
                    kind = "tornado"
                    break
                if phen == "SV":
                    kind = "severe"
                    break

        if not kind:
            return ""

        # ---- Helpers ----
        def _parse_vtec_z(tok: str):
            # tok like YYYYMMDDT0000Z or YYMMDDT0000Z
            s = (tok or "").strip().upper()
            mm = re.fullmatch(r"(\d{8}|\d{6})T(\d{4})Z", s)
            if not mm:
                return None
            d = mm.group(1)
            hm = mm.group(2)
            if len(d) == 8:
                year = int(d[0:4]); month = int(d[4:6]); day = int(d[6:8])
            else:
                year = 2000 + int(d[0:2]); month = int(d[2:4]); day = int(d[4:6])
            hour = int(hm[0:2]); minute = int(hm[2:4])
            try:
                return dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc)
            except Exception:
                return None

        def _fmt_time_local(d: dt.datetime) -> str:
            # "8 PM" or "8:30 PM"
            hour12 = d.hour % 12
            if hour12 == 0:
                hour12 = 12
            ampm = "AM" if d.hour < 12 else "PM"
            if d.minute == 0:
                return f"{hour12} {ampm}"
            return f"{hour12}:{d.minute:02d} {ampm}"

        def _daypart(d: dt.datetime) -> str:
            # rough-but-good NWR-ish phrasing
            if d.hour < 12:
                return "morning"
            if d.hour < 17:
                return "afternoon"
            if d.hour < 21:
                return "evening"
            return "tonight"

        def _until_phrase(end_local: dt.datetime) -> str:
            now_local = dt.datetime.now(tz=self._tz)
            t = _fmt_time_local(end_local)
            dp = _daypart(end_local)

            if end_local.date() == now_local.date():
                if dp == "tonight":
                    return f"until {t} tonight"
                return f"until {t} this {dp}"

            if (end_local.date() - now_local.date()).days == 1:
                if dp == "tonight":
                    return f"until {t} tomorrow night"
                return f"until {t} tomorrow {dp}"

            # fallback: weekday
            wd = end_local.strftime("%A")
            return f"until {t} on {wd}"

        def _join_oxford(items: list[str]) -> str:
            xs = [x.strip() for x in items if x and x.strip()]
            if not xs:
                return ""
            if len(xs) == 1:
                return xs[0]
            if len(xs) == 2:
                return f"{xs[0]} and {xs[1]}"
            return ", ".join(xs[:-1]) + f", and {xs[-1]}"

        STATE_NAME = {
            "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut",
            "DE":"Delaware","DC":"the District of Columbia","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois",
            "IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts",
            "MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire",
            "NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon",
            "PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont",
            "VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming",
        }

        # ---- Extract watch number + end time from VTEC ----
        watch_num: int | None = None
        end_utc: dt.datetime | None = None

        for v in self._cap_vtec_list(ev):
            m = _VTEC_PARSE_RE.search(v)
            if not m:
                continue
            phen = (m.group("phen") or "").upper()
            sig = (m.group("sig") or "").upper()
            if sig != "A":
                continue
            if kind == "tornado" and phen != "TO":
                continue
            if kind == "severe" and phen != "SV":
                continue

            try:
                watch_num = int(m.group("etn"))
            except Exception:
                watch_num = None

            end_utc = _parse_vtec_z(m.group("end") or "")
            break

        end_phrase = ""
        if end_utc is not None:
            end_local = end_utc.astimezone(self._tz)
            end_phrase = _until_phrase(end_local)

        # ---- Parse counties/states from CAP areaDesc ----
        area_desc = (getattr(ev, "area_desc", "") or "").strip()
        # CAP areaDesc often: "Cambria, PA; Cameron, PA; ..."
        groups: dict[str, list[str]] = {}
        order: list[str] = []
        misc: list[str] = []

        for raw in re.split(r";\s*", area_desc):
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

        # ---- Boilerplate ----
        if kind == "tornado":
            watch_label = "Tornado Watch"
            remember = (
                "Remember, a tornado watch means that conditions are favorable for the development of severe weather, "
                "including tornadoes, large hail, and damaging winds, in and close to the watch area. "
                "While severe weather may not be imminent, persons should remain alert for rapidly changing weather conditions, "
                "and listen for later statements and possible warnings."
            )
        else:
            watch_label = "Severe Thunderstorm Watch"
            remember = (
                "Remember, a severe thunderstorm watch means that conditions are favorable for the development of severe weather, "
                "including large hail and damaging winds, in and close to the watch area. "
                "While severe weather may not be imminent, persons should remain alert for rapidly changing weather conditions, "
                "and listen for later statements and possible warnings."
            )

        stay_tuned = (
            "Stay tuned to NOAA Weather Radio, commercial radio, and television outlets, "
            "or internet sources for the latest severe weather information."
        )

        # ---- Build script ----
        lines: list[str] = []

        if watch_num is not None:
            lines.append(f"The National Weather Service has issued {watch_label} Number {watch_num}.")
        else:
            lines.append(f"The National Weather Service has issued {watch_label}.")

        if end_phrase:
            lines.append(f"Effective {end_phrase}.")

        if groups:
            if len(order) == 1:
                st = order[0]
                st_full = STATE_NAME.get(st, st)
                county_list = _join_oxford(groups.get(st, []))
                if county_list:
                    lines.append(f"This watch includes the following counties, in {st_full}: {county_list}.")
            else:
                segs: list[str] = []
                for st in order:
                    st_full = STATE_NAME.get(st, st)
                    county_list = _join_oxford(groups.get(st, []))
                    if county_list:
                        segs.append(f"in {st_full}: {county_list}")
                if segs:
                    lines.append("This watch includes the following counties: " + "; ".join(segs) + ".")
        elif area_desc:
            # fallback if parsing fails
            lines.append(f"This watch includes the following areas: {area_desc}.")

        # If CAP areaDesc was empty but we have leftovers
        if misc and not groups:
            lines.append("This watch includes: " + _join_oxford(misc) + ".")

        lines.append(remember)
        lines.append(stay_tuned)

        # Keep SeasonalWeather’s usual closer (optional, but consistent)
        if mode == "full":
            lines.append("End of message.")

        # Double-newlines => better pacing
        return "\n\n".join(ln.strip() for ln in lines if ln and ln.strip()).strip()

    def _parse_cap_area_by_state(self, area_desc: str) -> tuple[dict[str, list[str]], list[str], list[str]]:
        """Shim → product_text.parse_cap_area_by_state()."""
        return _pt_parse_cap_area_by_state(area_desc)

    def _join_oxford(self, items: list[str]) -> str:
        """Shim → product_text.join_oxford()."""
        return _pt_join_oxford(items)

    def _fmt_local_from_utc_iso(self, iso_str: str) -> str:
        return _fmt_local_from_utc_iso(iso_str, local_tz=self._tz)

    def _cap_prefers_statement_update_script(self, event: str, vtec_actions: set[str]) -> bool:
        """Shim → product_text.cap_prefers_statement_update_script()."""
        return _cap_prefers_statement_update_script_fn(event, vtec_actions)

    def _cap_expiry_summary_line(self, text: str) -> str:
        """Shim → product_text.cap_expiry_summary_line()."""
        return _cap_expiry_summary_line(text)

    def _build_statement_vtec_action_script(
        self,
        ev: "CapAlertEvent",  # type: ignore[name-defined]
        vtec_actions: set[str],
        tracks: list[tuple[str, str]],
    ) -> str:
        """Shim → product_text.build_statement_vtec_action_script()."""
        return _build_statement_vtec_action_script_fn(
            event=getattr(ev, "event", "") or "",
            area_desc=(getattr(ev, "area_desc", "") or "").strip(),
            description=str(getattr(ev, "description", "") or "").strip(),
            headline=str(getattr(ev, "headline", "") or "").strip(),
            vtec=self._cap_vtec_list(ev),
            vtec_actions=vtec_actions,
            parameters=getattr(ev, "parameters", {}) or {},
            sps_preamble=self._cap_sps_preamble,
            sent_iso=getattr(ev, "sent", None),
        )

    def _build_warning_vtec_action_script(
        self,
        ev: "CapAlertEvent",  # type: ignore[name-defined]
        vtec_actions: set[str],
        tracks: list[tuple[str, str]],
    ) -> str:
        """Shim → product_text.build_warning_vtec_action_script()."""
        vtec = self._cap_vtec_list(ev)
        exp_utc = self._best_expiry_from_vtec(vtec)
        exp_phrase = ""
        if exp_utc:
            exp_phrase = self._fmt_local_from_utc_iso(exp_utc.isoformat())
        if not exp_phrase:
            raw_exp = getattr(ev, "expires", None)
            if raw_exp:
                exp_phrase = self._fmt_local_from_utc_iso(str(raw_exp))

        result = _build_warning_vtec_action_script_fn(
            event=getattr(ev, "event", "") or "",
            headline=getattr(ev, "headline", "") or "",
            description=str(getattr(ev, "description", "") or ""),
            instruction=str(getattr(ev, "instruction", "") or ""),
            area_desc=(getattr(ev, "area_desc", "") or "").strip(),
            vtec_actions=vtec_actions,
            exp_phrase=exp_phrase,
        )
        # If the free function produced nothing, fall through to the full script.
        if not result or result.strip() == "End of message.":
            return self._build_cap_full_script(ev)
        return result

    def _build_watch_vtec_action_script(
        self,
        ev: "CapAlertEvent",  # type: ignore[name-defined]
        vtec_actions: set[str],
        tracks: list[tuple[str, str]],
        watch_number: int | None,
        kind: str,  # "tornado" or "severe"
    ) -> str:
        """
        NWR-style voice script for VTEC update/cancel actions on watches (TOA/SVA).

        CON      → "Watch Number N remains in effect until …"
        EXA      → "Watch Number N remains in effect until … and now includes …"
        CAN      → "Watch Number N has been cancelled for … in …"
        EXP      → "Watch Number N has been allowed to expire for … in …"
        """
        watch_label = "Tornado Watch" if kind == "tornado" else "Severe Thunderstorm Watch"
        num_phrase = f"Number {watch_number}" if watch_number is not None else ""
        label_with_num = f"{watch_label} {num_phrase}".strip()

        area_desc = (getattr(ev, "area_desc", "") or "").strip()
        groups, order, misc = self._parse_cap_area_by_state(area_desc)

        vtec = self._cap_vtec_list(ev)
        exp_utc = self._best_expiry_from_vtec(vtec)
        exp_phrase = ""
        if exp_utc:
            exp_phrase = self._fmt_local_from_utc_iso(exp_utc.isoformat())
        if not exp_phrase:
            raw_exp = getattr(ev, "expires", None)
            if raw_exp:
                exp_phrase = self._fmt_local_from_utc_iso(str(raw_exp))

        def _county_segs() -> str:
            """Build 'in Maryland: Allegany, Garrett' style phrase."""
            if not groups:
                return area_desc or "the affected areas"
            parts: list[str] = []
            for st in order:
                st_full = self._STATE_NAME_FULL.get(st, st)
                county_list = self._join_oxford(groups[st])
                if county_list:
                    parts.append(f"in {st_full}: {county_list}")
            if parts:
                return "; ".join(parts)
            return area_desc or "the affected areas"

        lines: list[str] = []

        if vtec_actions & {"CAN"}:
            lines.append(f"{label_with_num} has been cancelled for the following areas.")
            lines.append(_county_segs() + ".")

        elif vtec_actions & {"EXP"}:
            lines.append(f"{label_with_num} has been allowed to expire for the following areas.")
            lines.append(_county_segs() + ".")

        elif vtec_actions & {"EXA", "EXB"}:
            # Watch expansion — also used when area grows mid-event
            lines.append(f"{label_with_num} remains in effect" + (f" until {exp_phrase}" if exp_phrase else "") + ".")
            lines.append("This watch now includes the following additional areas.")
            lines.append(_county_segs() + ".")

        else:  # CON / EXT
            lines.append(f"{label_with_num} remains in effect" + (f" until {exp_phrase}" if exp_phrase else "") + ".")
            lines.append(f"This watch includes the following areas: {_county_segs()}.")

        if not lines:
            return self._build_cap_watch_script(ev, mode="full")

        lines.append("Stay tuned to NOAA Weather Radio, commercial radio, and television outlets for the latest severe weather information.")
        lines.append("End of message.")
        return "\n".join(ln.strip() for ln in lines if ln and ln.strip()).strip()

    def _build_watch_expansion_script(self, ev: "CapAlertEvent") -> str:  # type: ignore[name-defined]
        """
        Full NWR-style script for watch EXA/EXB: new SAME tones, full county listing.
        Expansion is treated as a new issuance for the added counties.
        """
        # Determine kind + watch number from VTEC
        kind = "tornado"
        watch_number: int | None = None
        for v in self._cap_vtec_list(ev):
            m = _VTEC_PARSE_RE.search(v)
            if not m:
                continue
            phen = (m.group("phen") or "").upper()
            sig = (m.group("sig") or "").upper()
            if sig != "A":
                continue
            if phen == "TO":
                kind = "tornado"
            elif phen == "SV":
                kind = "severe"
            else:
                continue
            try:
                watch_number = int(m.group("etn"))
            except Exception:
                pass
            break

        tracks = self._vtec_tracks(self._cap_vtec_list(ev))
        return self._build_watch_vtec_action_script(
            ev,
            vtec_actions={"EXA"},
            tracks=tracks,
            watch_number=watch_number,
            kind=kind,
        )

    def _build_cap_full_script(self, ev: "CapAlertEvent") -> str:  # type: ignore[name-defined]
        """CAP adapter → central NWS full-alert formatter."""
        return _build_nws_full_alert_script(
            NwsAlertTextInput(
                event=str(getattr(ev, "event", "") or ""),
                headline=str(getattr(ev, "headline", "") or ""),
                description=str(getattr(ev, "description", "") or ""),
                instruction=str(getattr(ev, "instruction", "") or ""),
                area_desc=str(getattr(ev, "area_desc", "") or ""),
                sent_iso=getattr(ev, "sent", None),
                expires_iso=getattr(ev, "expires", None),
                parameters=getattr(ev, "parameters", {}) or {},
                vtec=self._cap_vtec_list(ev),
            ),
            sps_preamble=self._cap_sps_preamble,
        )

    def _build_cap_voice_script(self, ev: "CapAlertEvent") -> str:  # type: ignore[name-defined]
        """CAP adapter → central NWS voice/update formatter."""
        return _build_nws_voice_alert_script(
            NwsAlertTextInput(
                event=str(getattr(ev, "event", "") or ""),
                headline=str(getattr(ev, "headline", "") or ""),
                description=str(getattr(ev, "description", "") or ""),
                instruction=str(getattr(ev, "instruction", "") or ""),
                area_desc=str(getattr(ev, "area_desc", "") or ""),
                sent_iso=getattr(ev, "sent", None),
                expires_iso=getattr(ev, "expires", None),
                parameters=getattr(ev, "parameters", {}) or {},
                vtec=self._cap_vtec_list(ev),
            ),
            sps_preamble=self._cap_sps_preamble,
        )
