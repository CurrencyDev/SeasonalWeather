"""
active_alerts.py — Persistent active-alert registry for the broadcast cycle.

All alert types that produce audio (CAP FULL, CAP UPDATE, CAP VOICE, NWWS FULL/VOICE,
ERN relay, PNS cycle-only) are registered here while they are active.

On restart, active entries are re-queued as voice-only cycle segments replicating
NWR rebroadcast behaviour — you never lose a watch or warning from the air because
the process restarted.

Design principles:
  - Single asyncio event loop; no extra locking needed (Python GIL + single-threaded).
  - Atomic write: write to .tmp, then os.replace → crash-safe.
  - Schema-versioned JSON so we can add fields without breaking existing state files.
  - All callers should treat IDs as strings and prefer VTEC-derived keys for
    watch/warning events so updates find the same slot as the original.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("seasonalweather.active_alerts")

_SCHEMA_VERSION = 2

# VTEC track-id parser: OFFICE.PHEN.SIG.ETN
_VTEC_TRACK_RE = re.compile(
    r"/[A-Z]\.(?:[A-Z]{3})\."
    r"(?P<office>[A-Z]{4})\."
    r"(?P<phen>[A-Z0-9]{2})\."
    r"(?P<sig>[A-Z])\."
    r"(?P<etn>\d{4})\.",
)


def _vtec_track_id(vtec_str: str) -> str | None:
    """Extract 'OFFICE.PHEN.SIG.ETN' from a raw VTEC string, or None."""
    s = "".join(vtec_str.split()).strip()
    m = _VTEC_TRACK_RE.search(s)
    if not m:
        return None
    return f"{m.group('office')}.{m.group('phen')}.{m.group('sig')}.{m.group('etn')}"


@dataclass
class ActiveAlert:
    # ---- Identity ----
    id: str
    """
    Stable key for this alert slot.
    Prefer VTEC-derived: "CAP:KLWX.TO.A.0045" or "NWWS:KLWX.TO.A.0045".
    Falls back to CAP alert urn or sha1.
    """
    source: str           # "CAP" | "NWWS" | "ERN" | "PNS_CYCLE"
    event: str            # "Tornado Watch", "Severe Thunderstorm Warning", ...
    code: str             # SAME code: "TOA", "SVR", "TOR", ...

    # ---- VTEC ----
    vtec: list[str]       # raw VTEC strings  (may be empty for ERN/PNS)

    # ---- Content ----
    headline: str
    script_text: str      # TTS script that was (or will be) broadcast
    audio_path: Optional[str]   # rendered WAV path on disk (may be stale after restart)

    # ---- Timing ----
    expires: str          # ISO-8601 UTC  (e.g. "2026-03-12T00:00:00+00:00")
    issued: str           # ISO-8601 issued time

    # ---- Targeting ----
    same_locs: list[str]  # in-area FIPS codes used at time of origination

    # ---- Flags ----
    cycle_only: bool = False
    """
    True → voice segment in cycle only; no SAME tones, no 1050 Hz.
    Used for: PNS SEVERE WEATHER SAFETY RULES, advisory-level CAP voice, etc.
    """

    # ---- Watch metadata ----
    watch_number: Optional[int] = None
    """ETN (watch number) from VTEC, populated for TOA/SVA events."""

    # ---- Bookkeeping ----
    first_aired: Optional[str] = None
    last_aired: Optional[str] = None
    airing_count: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def expires_dt(self) -> dt.datetime:
        s = (self.expires or "").strip()
        if not s:
            return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return dt.datetime.fromisoformat(s)
        except Exception:
            return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)

    def is_expired(self, now: Optional[dt.datetime] = None) -> bool:
        now = now or dt.datetime.now(dt.timezone.utc)
        try:
            return now >= self.expires_dt()
        except Exception:
            return False

    def audio_exists(self) -> bool:
        if not self.audio_path:
            return False
        try:
            return Path(self.audio_path).exists()
        except Exception:
            return False

    def vtec_track_ids(self) -> list[str]:
        """Return all track IDs present in this alert's VTEC list."""
        out: list[str] = []
        for v in (self.vtec or []):
            t = _vtec_track_id(v)
            if t:
                out.append(t)
        return out


class AlertTracker:
    """
    Registry of currently active alerts for the broadcast cycle.

    Persists atomically to *state_path* — service restarts do not lose
    active watches or warnings from the on-air rotation.

    All callers run in the same asyncio event loop so no extra asyncio.Lock
    is required (Python GIL + single-threaded asyncio protect the dict).
    """

    def __init__(self, state_path: str | Path) -> None:
        self._path = Path(state_path)
        self._alerts: dict[str, ActiveAlert] = {}

    # ------------------------------------------------------------------ #
    #  Mutation                                                            #
    # ------------------------------------------------------------------ #

    def add_or_update(self, alert: ActiveAlert) -> None:
        """Register or refresh an alert slot. Persists immediately."""
        self._alerts[alert.id] = alert
        self._persist()

    def remove(self, alert_id: str) -> bool:
        """Remove by ID. Returns True if it existed."""
        existed = alert_id in self._alerts
        if existed:
            self._alerts.pop(alert_id)
            self._persist()
        return existed

    def mark_aired(self, alert_id: str) -> None:
        a = self._alerts.get(alert_id)
        if not a:
            return
        now_iso = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
        if a.first_aired is None:
            a.first_aired = now_iso
        a.last_aired = now_iso
        a.airing_count += 1
        self._persist()

    def update_audio_path(self, alert_id: str, audio_path: str) -> None:
        a = self._alerts.get(alert_id)
        if a:
            a.audio_path = audio_path
            self._persist()

    def update_script(self, alert_id: str, script_text: str) -> None:
        a = self._alerts.get(alert_id)
        if a:
            a.script_text = script_text
            self._persist()

    # ------------------------------------------------------------------ #
    #  Queries                                                             #
    # ------------------------------------------------------------------ #

    def get_active(self, now: Optional[dt.datetime] = None) -> list[ActiveAlert]:
        """Return non-expired alerts sorted by issued time (oldest first)."""
        now = now or dt.datetime.now(dt.timezone.utc)
        active = [a for a in self._alerts.values() if not a.is_expired(now)]
        active.sort(key=lambda a: (a.issued or "", a.id))
        return active

    def get_cycle_alerts(self, now: Optional[dt.datetime] = None) -> list[ActiveAlert]:
        """All active alerts (for cycle segment injection)."""
        return self.get_active(now)

    def has_active(self, now: Optional[dt.datetime] = None) -> bool:
        return bool(self.get_active(now))

    def is_known(self, alert_id: str) -> bool:
        return alert_id in self._alerts

    def find_by_vtec_track(self, track_id: str) -> ActiveAlert | None:
        """Find first alert whose VTEC list contains the given track id."""
        for a in self._alerts.values():
            if track_id in a.vtec_track_ids():
                return a
        return None

    # ------------------------------------------------------------------ #
    #  Housekeeping                                                        #
    # ------------------------------------------------------------------ #

    def purge_expired(self, now: Optional[dt.datetime] = None) -> int:
        now = now or dt.datetime.now(dt.timezone.utc)
        dead = [aid for aid, a in self._alerts.items() if a.is_expired(now)]
        for aid in dead:
            self._alerts.pop(aid)
        if dead:
            self._persist()
        return len(dead)

    def remove_by_vtec_tracks(self, track_ids: set[str], reason: str = "") -> int:
        """
        Remove alerts whose VTEC contains any of the given track IDs.
        Returns the number of entries removed.
        """
        removed = 0
        for aid in list(self._alerts.keys()):
            a = self._alerts.get(aid)
            if not a:
                continue
            for tid in a.vtec_track_ids():
                if tid in track_ids:
                    self._alerts.pop(aid, None)
                    removed += 1
                    log.info(
                        "AlertTracker: removed id=%s event=%s track=%s reason=%s",
                        aid, a.event, tid, reason,
                    )
                    break
        if removed:
            self._persist()
        return removed

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def load(self) -> int:
        """Load state from disk. Returns number of entries loaded (0 = nothing)."""
        if not self._path.exists():
            return 0
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return 0
            alerts_raw = payload.get("active_alerts", [])
            loaded = 0
            _known_fields = set(ActiveAlert.__dataclass_fields__.keys())
            for raw in alerts_raw:
                if not isinstance(raw, dict):
                    continue
                try:
                    kwargs = {k: raw[k] for k in _known_fields if k in raw}
                    a = ActiveAlert(**kwargs)  # type: ignore[arg-type]
                    self._alerts[a.id] = a
                    loaded += 1
                except Exception:
                    log.debug("AlertTracker: skipped malformed entry: %s", raw)
            log.info(
                "AlertTracker: loaded %d entries from %s", loaded, self._path
            )
            return loaded
        except FileNotFoundError:
            return 0
        except Exception:
            log.exception(
                "AlertTracker: load failed (starting empty) path=%s", self._path
            )
            return 0

    def _persist(self) -> None:
        """Atomically write current state. Crash-safe via tmp + os.replace."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "written_at": dt.datetime.now(dt.timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "active_alerts": [asdict(a) for a in self._alerts.values()],
            }
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp, self._path)
        except Exception:
            log.exception("AlertTracker: persist failed path=%s", self._path)
