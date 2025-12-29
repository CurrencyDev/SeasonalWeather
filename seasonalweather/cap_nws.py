from __future__ import annotations

# =========================================================================================
#      MP"""""`MM                                                       dP              MM'"""'YMM
#      M  mmmmm..M                                                       88              M' .mmm. `M
#      M.      `YM .d8888b. .d8888b. .d8888b. .d8888b. 88d888b. .d8888b. 88              M  MMMMMooM dP    dP 88d888b. 88d888b. .d8888b. 88d888b. .d8888b. dP    dP
#      MMMMMMM.  M 88ooood8 88'  `88 Y8ooooo. 88'  `88 88'  `88 88'  `88 88              M  MMMMMMMM 88    88 88'  `88 88'  `88 88ooood8 88'  `88 88'  `"" 88    88
#      M. .MMM'  M 88.  ... 88.  .88       88 88.  .88 88    88 88.  .88 88              M. `MMM' .M 88.  .88 88       88       88.  ... 88    88 88.  ... 88.  .88
#      Mb.     .dM `88888P' `88888P8 `88888P' `88888P' dP    dP `88888P8 dP              MM.     .dM `88888P' dP       dP       `88888P' dP    dP `88888P' `8888P88
#      MMMMMMMMMMM                                                Seasonal_Currency      MMMMMMMMMMM                                                            .88
#                                                                                                                                                           d8888P.
# =========================================================================================

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

from .cap_ledger import CapLedger

log = logging.getLogger("seasonalweather.cap")


_STATE_FIPS_TO_ABBR: dict[str, str] = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
    # Territories (rare for you, but harmless to support)
    "60": "AS",
    "66": "GU",
    "69": "MP",
    "72": "PR",
    "74": "UM",
    "78": "VI",
}


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v.strip() if v else default


def _norm_same(s: str) -> str | None:
    """
    Normalize SAME FIPS strings.

    Expected: PSSCCC (6 digits). We keep leading zeros.
    """
    if s is None:
        return None
    x = "".join(ch for ch in str(s).strip() if ch.isdigit())
    if len(x) != 6:
        return None
    return x


def _same_state_fips(same6: str) -> str | None:
    """
    SAME is PSSCCC; state is SS => same6[1:3]
    """
    if not same6 or len(same6) != 6:
        return None
    return same6[1:3]


def _derive_area_states(same_fips_allow: Iterable[Any]) -> list[str]:
    """
    Derive NWS CAP ?area= state list from SAME allow list.

    Important: state code is SAME[1:3], NOT [:2].
    """
    states: list[str] = []
    seen: set[str] = set()

    for raw in same_fips_allow:
        s6 = _norm_same(str(raw))
        if not s6:
            continue
        ss = _same_state_fips(s6)
        if not ss:
            continue
        abbr = _STATE_FIPS_TO_ABBR.get(ss)
        if not abbr:
            continue
        if abbr in seen:
            continue
        seen.add(abbr)
        states.append(abbr)

    # Stable ordering helps debugging/log comparisons
    states.sort()

    # If derivation fails, fall back to your historical default region.
    if not states:
        states = ["DC", "MD", "VA", "WV"]

    return states


@dataclass(frozen=True, slots=True)
class CapAlertEvent:
    alert_id: str
    sent: str | None

    status: str | None
    message_type: str | None

    event: str | None
    severity: str | None
    urgency: str | None
    certainty: str | None

    headline: str | None
    area_desc: str | None
    description: str | None
    instruction: str | None

    same_fips: list[str]

    # NEW: NWS CAP "parameters" passthrough + VTEC list extraction
    parameters: dict[str, list[str]]
    vtec: list[str]


class NwsCapPoller:
    """
    Polls api.weather.gov CAP alerts and enqueues *service-area-matching* alerts.

    Constructor matches main.py usage:
      NwsCapPoller(out_queue=..., same_fips_allow=..., poll_seconds=..., user_agent=..., url=?)

    Notes:
      - No import-time network, ever.
      - Filters to alerts whose geocode.SAME intersects same_fips_allow.
      - Dedupe on (alert_id, sent) so updates can flow but repeats don't spam.
      - Persistent dedupe across restarts via CapLedger.
      - NEW: extracts props.parameters and props.parameters.VTEC when present.
    """

    def __init__(
        self,
        *,
        out_queue: asyncio.Queue[CapAlertEvent],
        same_fips_allow: list[Any],
        poll_seconds: int = 60,
        user_agent: str = "SeasonalWeather (CAP monitor)",
        url: str | None = None,
    ) -> None:
        self.out_queue = out_queue
        self.poll_seconds = max(15, int(poll_seconds))
        self.user_agent = user_agent.strip() or "SeasonalWeather (CAP monitor)"

        allow_set: set[str] = set()
        for raw in same_fips_allow:
            s6 = _norm_same(str(raw))
            if s6:
                allow_set.add(s6)
        self.allow_set = allow_set

        self.area_states = _derive_area_states(same_fips_allow)

        # If caller provides a full URL, use it as-is. Otherwise build params.
        self.url = (url or "").strip() or "https://api.weather.gov/alerts/active"
        self._use_params = (url is None) or (not url.strip())
        self._params = {"area": ",".join(self.area_states), "status": "actual"} if self._use_params else None

        # In-memory dedupe (fast path within one process lifetime)
        self._seen_keys: set[str] = set()

        # Persistent dedupe (prevents restart spam)
        ledger_path = _env_str("SEASONAL_CAP_LEDGER_PATH", "/var/lib/seasonalweather/cap_ledger.json")
        ledger_max_age_days = _env_int("SEASONAL_CAP_LEDGER_MAX_AGE_DAYS", 14)
        self._ledger = CapLedger(path=Path(ledger_path), max_age_days=ledger_max_age_days)

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(15.0, connect=10.0),
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/geo+json, application/json;q=0.9, */*;q=0.1",
            },
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception:
            pass

    async def _fetch_json(self) -> dict[str, Any]:
        """
        Fetch JSON with mild retry/backoff for transient NWS hiccups.
        """
        tries = 0
        backoff = 1.0

        while True:
            tries += 1
            try:
                r = await self._client.get(self.url, params=self._params)
                # Respect rate limiting a bit
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, dict):
                    raise ValueError("CAP response JSON was not an object")
                return data
            except Exception as e:
                if tries >= 5:
                    raise
                # jittery exponential backoff
                sleep_s = min(20.0, backoff) + random.random() * 0.25
                log.warning("CAP fetch failed (try %d/5): %s; sleeping %.2fs", tries, e, sleep_s)
                await asyncio.sleep(sleep_s)
                backoff *= 2.0

    def _extract_same_list(self, props: dict[str, Any]) -> list[str]:
        geocode = props.get("geocode")
        if not isinstance(geocode, dict):
            return []
        same = geocode.get("SAME")
        if not isinstance(same, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for raw in same:
            s6 = _norm_same(str(raw))
            if not s6:
                continue
            if s6 in seen:
                continue
            seen.add(s6)
            out.append(s6)
        return out

    def _extract_parameters(self, props: dict[str, Any]) -> dict[str, list[str]]:
        """
        NWS CAP includes properties.parameters where values are typically lists of strings.
        We normalize into dict[str, list[str]].
        """
        p = props.get("parameters")
        if not isinstance(p, dict):
            return {}
        out: dict[str, list[str]] = {}
        for k, v in p.items():
            kk = str(k).strip()
            if not kk:
                continue
            if isinstance(v, list):
                out[kk] = [str(x).strip() for x in v if str(x).strip()]
            elif v is not None:
                s = str(v).strip()
                if s:
                    out[kk] = [s]
        return out

    def _extract_vtec(self, params: dict[str, list[str]]) -> list[str]:
        """
        VTEC (PVTEC) is commonly delivered as properties.parameters.VTEC (list[str]).
        We normalize by stripping whitespace and removing embedded spaces.
        """
        vals = params.get("VTEC", []) if isinstance(params, dict) else []
        out: list[str] = []
        seen: set[str] = set()
        for raw in vals:
            x = "".join(str(raw).split()).strip()
            if not x:
                continue
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    def _matches_service_area(self, same_list: list[str]) -> bool:
        if not same_list:
            return False
        if not self.allow_set:
            # If allow list is empty, treat as "match nothing" to avoid accidental whole-US spam.
            return False
        return any(s in self.allow_set for s in same_list)

    def _event_from_feature(self, feat: dict[str, Any]) -> CapAlertEvent | None:
        props = feat.get("properties")
        if not isinstance(props, dict):
            return None

        # IDs: NWS sometimes uses "id" at top-level and "@id"/"id" in properties.
        alert_id = (
            str(props.get("id") or props.get("@id") or feat.get("id") or "").strip()
        )
        if not alert_id:
            return None

        sent = props.get("sent")
        sent_s = str(sent).strip() if sent is not None else None

        same_list = self._extract_same_list(props)
        if not self._matches_service_area(same_list):
            return None

        params = self._extract_parameters(props)
        vtec = self._extract_vtec(params)

        ev = CapAlertEvent(
            alert_id=alert_id,
            sent=sent_s,
            status=str(props.get("status")).strip() if props.get("status") is not None else None,
            message_type=str(props.get("messageType")).strip() if props.get("messageType") is not None else None,
            event=str(props.get("event")).strip() if props.get("event") is not None else None,
            severity=str(props.get("severity")).strip() if props.get("severity") is not None else None,
            urgency=str(props.get("urgency")).strip() if props.get("urgency") is not None else None,
            certainty=str(props.get("certainty")).strip() if props.get("certainty") is not None else None,
            headline=str(props.get("headline")).strip() if props.get("headline") is not None else None,
            area_desc=str(props.get("areaDesc")).strip() if props.get("areaDesc") is not None else None,
            description=str(props.get("description")).strip() if props.get("description") is not None else None,
            instruction=str(props.get("instruction")).strip() if props.get("instruction") is not None else None,
            same_fips=same_list,
            parameters=params,
            vtec=vtec,
        )
        return ev

    def _dedupe_key(self, ev: CapAlertEvent) -> str:
        # sent may be None; still stable enough for dedupe
        return f"{ev.alert_id}|{ev.sent or ''}"

    async def run_forever(self) -> None:
        """
        Main loop. Never returns unless cancelled or fatal exceptions bubble out.
        """
        try:
            if self._use_params:
                log.info(
                    "NWS CAP poller starting (poll=%ss url=%s area=%s)",
                    self.poll_seconds,
                    self.url,
                    ",".join(self.area_states),
                )
            else:
                log.info(
                    "NWS CAP poller starting (poll=%ss url=%s)",
                    self.poll_seconds,
                    self.url,
                )

            while True:
                t0 = time.time()
                marked_any = False
                emitted = 0

                try:
                    data = await self._fetch_json()
                    feats = data.get("features")
                    if not isinstance(feats, list):
                        feats = []

                    for feat in feats:
                        if not isinstance(feat, dict):
                            continue
                        ev = self._event_from_feature(feat)
                        if ev is None:
                            continue

                        k = self._dedupe_key(ev)

                        # Fast path: already seen in-memory
                        if k in self._seen_keys:
                            continue

                        # Persistent path: already seen on-disk (prevents restart spam)
                        try:
                            if self._ledger.has(k):
                                self._seen_keys.add(k)
                                continue
                        except Exception:
                            # Ledger should never take the service down
                            log.exception("CAP ledger has() failed (continuing without persistent dedupe for this check)")

                        # Mark seen (persist) *before* enqueue so a crash doesn't re-air on restart
                        try:
                            self._ledger.mark(k)
                            marked_any = True
                        except Exception:
                            log.exception("CAP ledger mark() failed (continuing)")

                        self._seen_keys.add(k)

                        try:
                            self.out_queue.put_nowait(ev)
                            emitted += 1
                        except asyncio.QueueFull:
                            log.warning("CAP queue full; dropping alert id=%s event=%s", ev.alert_id, ev.event)

                    # Persist ledger updates to disk
                    if marked_any:
                        try:
                            self._ledger.flush()
                        except Exception:
                            log.exception("CAP ledger flush() failed (non-fatal)")

                    log.info("CAP poll: emitted %d matching alerts (features=%d)", emitted, len(feats))

                    # guard: prevent unbounded growth if the process runs forever
                    if len(self._seen_keys) > 50000:
                        # This is safe: persistent ledger still prevents restart spam.
                        log.warning("CAP in-memory dedupe set is large (%d); clearing", len(self._seen_keys))
                        self._seen_keys.clear()

                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("CAP poll loop error")

                # sleep to next poll, accounting for runtime
                elapsed = time.time() - t0
                sleep_s = max(1.0, float(self.poll_seconds) - elapsed)
                await asyncio.sleep(sleep_s)

        finally:
            await self.aclose()
