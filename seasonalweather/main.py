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

import argparse
import asyncio
import datetime as dt
import hashlib
import logging

import re
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from .config import load_config, AppConfig
from .logging_config import setup_logging

# Module-level config reference — set once at startup before Orchestrator is created.
_APP_CFG: "AppConfig | None" = None
from .alerts.nws_api import NWSApi
from .nwws.client import NWWSClient
from .alerts.product import parse_product_text, ParsedProduct
from .alerts.builder import build_spoken_alert
from .tts.tts import TTS
from .liquidsoap_telnet import LiquidsoapTelnet
from .broadcast.cycle import CycleBuilder, CycleContext
from .broadcast.segment_store import SegmentStore
from .broadcast.conductor import CycleConductor
from .broadcast.segment_refresher import SegmentRefresher
from .broadcast.cap_text import CapTextRenderer
from .broadcast.product_text import render_nwws_product_script
from .broadcast.audio_origination import AudioOriginator, safe_event_code as _safe_event_code
from .broadcast.ern_relay_runtime import ErnRelayRuntime
from .broadcast.ipaws_runtime import IpawsRuntime

# Active alert tracker (persistent cycle state across restarts)
from .alerts.active import ActiveAlert, AlertTracker, _vtec_track_id
from .alerts.focus import alert_holds_focus
from .database.bootstrap import bootstrap_database_from_config
from .database.housekeeping import DatabaseHousekeeper
from .database.inserts import CycleInsertRepository
from .database.station_feed import StationFeedRepository
from .discord_log import DiscordLogger
from .health_state import HealthStateMachine
from .broadcast.pns import PnsStateMachine, parse_nws_header_issued_dt, pns_text_same_issuance
from .broadcast.ern_script import (
    _parse_duration_minutes as _ern_parse_duration_minutes,
    _same_jday_to_utc as _ern_same_jday_to_utc,
)

# VTEC policy + SAME event code libraries — Orchestrator defers to these.
from .alerts.vtec import (
    toneout_policy as _vtec_toneout_policy,
    same_codes_for_vtec as _vtec_same_codes_for_vtec,
    VTEC_FIND_RE as _VTEC_FIND_RE,
    VTEC_PARSE_RE as _VTEC_PARSE_RE,
)
from .same.events import label_or_code as _same_label_or_code
from .same.targeting import SameTargetResolver
from .same.locations import normalize_same_allow_set as _normalize_same_allow_set

# RWT/RMT scheduler
from .broadcast.rwt_rmt import RwtRmtSchedule, RwtRmtScheduler
from .broadcast.tests import default_test_script_lines, format_test_presentation_template

# Optional CAP (api.weather.gov/alerts/active)
try:
    from .alerts.cap_nws import NwsCapPoller, CapAlertEvent
except Exception:  # pragma: no cover
    NwsCapPoller = None  # type: ignore
    CapAlertEvent = None  # type: ignore

# Optional IPAWS CAP (apps.fema.gov IPAWS Open feed)
try:
    from .alerts.ipaws_cap import IpawsCapPoller, IpawsCapEvent
except Exception:  # pragma: no cover
    IpawsCapPoller = None  # type: ignore
    IpawsCapEvent = None  # type: ignore


# Optional ERN/GWES SAME monitor (Level 3 source)
try:
    from .broadcast.ern_gwes import ErnGwesMonitor, ErnSameEvent
except Exception:  # pragma: no cover
    ErnGwesMonitor = None  # type: ignore
    ErnSameEvent = None  # type: ignore


log = logging.getLogger("seasonalweather")

from .broadcast.station_feed_runtime import (
    set_app_config as _sf_set_app_config,
    set_repository as _sf_set_repository,
    enabled as _sf_enabled,
    station_feed_housekeeping_start as _sf_station_feed_hk_start,
    seed_from_alert_tracker as _station_feed_seed_from_alert_tracker,
    remove_ids as _sf_remove_ids,
    remove_ern_relays_matching as _sf_remove_ern_relays_matching,
    cap_reference_ids as _sf_cap_reference_ids,
    nwws_best_issued_dt as _sf_nwws_best_issued_dt,
    nwws_event_label as _sf_nwws_event_label,
    nwws_area_from_text as _sf_nwws_area_from_text,
    nwws_make_headline as _sf_nwws_make_headline,
    nwws_extract_issuer as _sf_nwws_extract_issuer,
    note_cap as _station_feed_note_cap,
    note_ern as _station_feed_note_ern,
    note_nwws as _station_feed_note_nwws,
    note_manual as _station_feed_note_manual,
    note_required_test as _station_feed_note_required_test,
)


# _VTEC_FIND_RE and _VTEC_PARSE_RE are now imported from .alerts.vtec above.


def _setup_logging(cfg: AppConfig | None = None) -> None:
    setup_logging(cfg)


# _env_* helpers removed — all configuration now flows through AppConfig.
# Credentials are accessed via cfg.secrets.* (set once in load_config()).


class Orchestrator:
    def __init__(self, cfg: AppConfig) -> None:
        global _APP_CFG
        _APP_CFG = cfg
        _sf_set_app_config(cfg)
        self.cfg = cfg
        self.api = NWSApi()
        self.telnet = LiquidsoapTelnet(
            host=cfg.secrets.liquidsoap_host,
            port=cfg.secrets.liquidsoap_port,
        )

        self._tz = ZoneInfo(cfg.station.timezone)
        self.local_tz = self._tz
        self.cap_text = CapTextRenderer(
            local_tz=self._tz,
            cap_vtec_list=self._cap_vtec_list,
            vtec_tracks=self._vtec_tracks,
            best_expiry_from_vtec=self._best_expiry_from_vtec,
        )

        # Isolated policy/state machines. main.py only wires inputs/outputs;
        # classification and health decisions live in their own modules.
        self.pns_state = PnsStateMachine(cfg.pns, tz=self._tz)
        self.health_state = HealthStateMachine(cfg.health)

        # NWWS-OI
        self.jid = cfg.secrets.nwws_jid
        self.password = cfg.secrets.nwws_password
        self.nwws_server = cfg.nwws.server
        self.nwws_port = cfg.nwws.port

        # TTS
        self.tts = TTS(
            backend=cfg.tts.backend,
            voice=cfg.tts.voice,
            rate_wpm=cfg.tts.rate_wpm,
            volume=cfg.tts.volume,
            sample_rate=cfg.audio.sample_rate,
            text_overrides=cfg.tts.text_overrides,
            vtp_cfg=cfg.tts.voicetext_paul,
        )

        self.mode = "normal"
        self.heightened_until: dt.datetime | None = None
        self.last_heightened_at: dt.datetime | None = None
        self.last_product_desc: str | None = None

        # RWT/RMT gating timestamps
        self.last_toneout_at: dt.datetime | None = None
        self.cap_last_severe_at: dt.datetime | None = None
        self.ern_last_tone_at: dt.datetime | None = None

        self.cycle_builder = CycleBuilder(
            api=self.api,
            tz_name=cfg.station.timezone,
            obs_stations=cfg.observations.stations,
            reference_points=cfg.cycle.reference_points,
            same_fips_all=cfg.service_area.same_fips_all,
            cycle_cfg=cfg.cycle,
        )

        # Fast membership checks for "in-area" targeting
        self._same_fips_allow_set = _normalize_same_allow_set(cfg.service_area.same_fips_all)
        self.targeting = SameTargetResolver(
            cfg=cfg,
            local_tz=self._tz,
            same_fips_allow_set=self._same_fips_allow_set,
        )

        # --- NWWS flood-gate controls ---
        self._nwws_logger = logging.getLogger("seasonalweather.nwws")
        self._nwws_raw_seen = 0
        self._nwws_rx_log_first_n = cfg.nwws.resiliency.rx_log_first_n
        self._nwws_decision_log_first_n = cfg.nwws.resiliency.decision_log_first_n
        self._nwws_decision_log_every = cfg.nwws.resiliency.decision_log_every
        self._nwws_allowed_wfos = self._norm_wfo_set(getattr(cfg.nwws, "allowed_wfos", []))

        self.nwws_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=200)

        # CAP queue (only used if CAP enabled and import succeeded)
        self.cap_queue: asyncio.Queue["CapAlertEvent"] = asyncio.Queue(maxsize=200)  # type: ignore[name-defined]
        self._cap_voice_last_by_key: dict[tuple[str, str], dt.datetime] = {}
        self._cap_full_last_by_key: dict[tuple[str, str], dt.datetime] = {}

        # IPAWS queue (only used if IPAWS enabled and import succeeded)
        self.ipaws_queue: asyncio.Queue["IpawsCapEvent"] = asyncio.Queue(maxsize=200)  # type: ignore[name-defined]

        # ERN queue (only used if ERN enabled and import succeeded)
        self.ern_queue: asyncio.Queue["ErnSameEvent"] = asyncio.Queue(maxsize=200)  # type: ignore[name-defined]

        # ERN relay cooldown (on-air)
        self._ern_relay_last_any_at: dt.datetime | None = None

        # Prevent concurrent cycle flush/push from overlapping
        self._cycle_lock = asyncio.Lock()

        # CycleConductor handles continuous buffering and live time synthesis.

        # --- Cross-source dedupe (NWWS vs CAP) ---
        self._dedupe_ttl_seconds = cfg.dedupe.ttl_seconds
        self._dedupe_lock = asyncio.Lock()
        self._recent_air_keys: dict[str, dt.datetime] = {}


        # --- NWWS decision visibility counters ---
        self._nwws_seen = 0
        self._nwws_acted = 0


        # --- Embedded SQLite runtime state ---
        self.database = bootstrap_database_from_config(cfg) if getattr(cfg.database, "enabled", True) else None
        self.cycle_insert_repo = CycleInsertRepository(self.database) if self.database is not None else None
        self.db_housekeeper = DatabaseHousekeeper(cfg, self.database) if self.database is not None else None
        self.station_feed_repo = StationFeedRepository(self.database) if self.database is not None else None
        _sf_set_repository(self.station_feed_repo)

        # Start StationFeed housekeeping after SQLite is ready so the public
        # /v1/handled-alerts read model stays tidy.
        _sf_station_feed_hk_start()

        # --- Persistent active alert tracker ---
        # Survives restarts: active watches/warnings are re-queued as cycle segments.
        _tracker_path = Path(cfg.paths.work_dir) / "alert_state.json"
        self.alert_tracker = AlertTracker(_tracker_path, database=self.database)

        # Discord webhook logger (fire-and-forget; starts its drain task in run())
        self.discord = DiscordLogger.from_config(cfg.logs.discord)

        # AudioOriginator owns TTS/WAV/SAME assembly. The orchestrator only
        # decides what alert/message should be aired.
        self.audio_originator = AudioOriginator(
            cfg=cfg,
            tts=self.tts,
            local_tz=self._tz,
            paths=self._paths,
            discord=self.discord,
        )
        self.ern_relay_runtime = ErnRelayRuntime(self)
        self.ipaws_runtime = IpawsRuntime(self)


        # SegmentStore: persistent per-segment audio cache.
        # Placed last so alert_tracker and all other attributes are available.
        self._seg_store = SegmentStore(
            work_dir=Path(cfg.paths.work_dir),
            audio_dir=Path(cfg.paths.audio_dir),
            database=self.database,
        )
        self._seg_store.load()

        # SegmentRefresher: keeps each segment's text + audio up to date
        # independently on its own cadence.
        self.refresher = SegmentRefresher(
            store=self._seg_store,
            cycle_builder=self.cycle_builder,
            tts=self.tts,
            alert_tracker=self.alert_tracker,
            ctx_fn=self._make_cycle_ctx,
            station_name=cfg.station.name,
            service_area_name=cfg.station.service_area_name,
            disclaimer=cfg.station.disclaimer,
            tz=self._tz,
            sample_rate=cfg.audio.sample_rate,
            on_alert_segments_changed=lambda reason: self.conductor.notify_flush(
                reset_rotation=True, reason=reason
            ),
        )

        # CycleConductor: continuous cycle driver.  Flush notifications restart
        # the active-alert priority rotation after alert-state changes.
        self.conductor = CycleConductor(
            store=self._seg_store,
            telnet=self.telnet,
            tts=self.tts,
            alert_tracker=self.alert_tracker,
            tz=self._tz,
            audio_dir=Path(cfg.paths.audio_dir),
            sample_rate=cfg.audio.sample_rate,
            np_meta_fn=self._np_meta,
            discord_fn=self.discord.cycle_rebuilt,
            active_alerts_fn=lambda: len(self.alert_tracker.get_cycle_alerts()),
            mode_fn=lambda: self.mode,
            alert_focus_policy=cfg.cycle.alert_focus,
            scheduled_inserts_fn=self._cycle_due_inserts,
            mark_insert_aired_fn=self._mark_cycle_insert_aired,
        )
        self.alert_tracker.set_change_callback(self._on_alert_tracker_changed)

    def _utc_iso(self, value: dt.datetime | None = None) -> str:
        when = value or dt.datetime.now(dt.timezone.utc)
        return when.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()

    def _cycle_due_inserts(self, placement: str, rotation_count: int, focus: bool) -> list[dict]:
        repo = getattr(self, "cycle_insert_repo", None)
        if repo is None:
            return []
        return repo.list_due(
            placement=placement,
            rotation_count=rotation_count,
            now_iso=self._utc_iso(),
            active_alert_focus=focus,
        )

    def _mark_cycle_insert_aired(self, insert_id: str, rotation_count: int) -> None:
        repo = getattr(self, "cycle_insert_repo", None)
        if repo is None:
            return
        repo.mark_aired(
            insert_id=insert_id,
            aired_at=self._utc_iso(),
            rotation_count=rotation_count,
        )

    def _on_alert_tracker_changed(self, reason: str) -> None:
        """Wake audio synthesis and reset rotation after active-alert state changes."""
        try:
            self.refresher.notify_alerts_changed()
            self.refresher.trigger_immediate("status")
        except Exception:
            log.debug("AlertTracker change: refresher notify failed", exc_info=True)
        try:
            self.conductor.notify_flush(reset_rotation=True, reason=f"alert-state:{reason}")
        except Exception:
            log.debug("AlertTracker change: conductor notify failed", exc_info=True)


    # --- Now Playing / IP-RDS helpers (edit phrases freely) ---

    _NP_CYCLE_TITLES = {
        "id": "Station identification.",
        "time": "The current time in our service area.",
        "status": "Overall station status and alerts.",
        "hwo": "Hazardous weather outlook for the service area.",
        "hwo-unavailable": "Hazardous weather outlook for the service area.",
        "spc": "Severe weather outlook for the service area.",
        "zfp": "Weather synopsis for the area.",
        "fcst": "The forecast for the service area.",
        "obs": "Current conditions in our area.",
        "outro": "End of the current broadcast cycle.",
        "default": "Weather information for our service area.",
    }

    _NP_ALERT_TEMPLATES = {
        "nwws_full": "{event}.",
        "nwws_update": "Update for a {event}.",
        "nwws_end": "A {event} has ended.",
        "cap_full": "{event}.",
        "cap_update": "Update for a {event}.",
        "ern": "{event} relay.",
        "rwt": "Required weekly test.",
        "rmt": "Required monthly test.",
        "default": "A weather alert has been issued.",
    }

    def _np_meta(self, *, title: str, kind: str, extra: dict[str, str] | None = None) -> dict[str, str]:
        # What players display:
        #   - title/artist/album/song
        # Plus internal keying fields prefixed with sw_ (most players ignore them).
        station = self.cfg.station.name
        artist = "SeasonalNet"
        album = "Weather information for Baltimore, Washington DC, and surrounding areas"
        t = (title or "").strip() or "SeasonalWeather"
        song = f"{station} — {t}"

        m: dict[str, str] = {
            "title": t,
            "artist": artist,
            "album": album,
            "song": song,
            "sw_station": station,
            "sw_kind": (kind or "").strip(),
        }
        if extra:
            for k, v in extra.items():
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    m[str(k)] = s
        return m

    def _np_cycle_title(self, key: str) -> str:
        k = (key or "").strip()
        return self._NP_CYCLE_TITLES.get(k, self._NP_CYCLE_TITLES["default"])

    def _np_alert_title(self, template_key: str, *, event: str) -> str:
        tpl = self._NP_ALERT_TEMPLATES.get(template_key, self._NP_ALERT_TEMPLATES["default"])
        return tpl.format(event=(event or "Alert").strip())


    def _norm_wfo_set(self, wfos: list[str] | set[str] | tuple[str, ...]) -> set[str]:
        """
        Normalizes allowed WFOs so YAML can use LWX or KLWX interchangeably.
        Also supports 4-letter Kxxx or 3-letter xxx.
        """
        out: set[str] = set()
        for w in wfos or []:
            s = str(w).strip().upper()
            if not s:
                continue
            out.add(s)
            if len(s) == 3:
                out.add("K" + s)
            if len(s) == 4 and s.startswith("K"):
                out.add(s[1:])
        return out

    def _paths(self) -> tuple[Path, Path, Path, Path]:
        work = Path(self.cfg.paths.work_dir)
        audio = Path(self.cfg.paths.audio_dir)
        cache = Path(self.cfg.paths.cache_dir)
        logs = Path(self.cfg.paths.log_dir)
        return work, audio, cache, logs

    async def _wait_for_liquidsoap(self) -> None:
        for _ in range(60):
            if self.telnet.ping():
                log.info("Liquidsoap telnet is reachable")
                return
            await asyncio.sleep(1)
        raise RuntimeError("Liquidsoap telnet did not become reachable (is seasonalweather-liquidsoap running?)")

    def _make_cycle_ctx(self) -> "CycleContext":
        """Return a CycleContext reflecting current station state.
        Used by SegmentRefresher so it can build segments without holding
        a reference to the full Orchestrator.
        """
        health = self.health_state.context()
        return CycleContext(
            mode=self.mode,
            last_heightened_ago=self._heightened_ago_str(),
            last_product_desc=self.last_product_desc,
            health_mode=health.mode,
            health_notice=health.notice,
            health_status_line=health.status_line,
            health_detached_loop_only=health.detached_loop_only,
        )

    def _update_mode(self) -> None:
        _prev_mode = getattr(self, "mode", "normal")
        now = dt.datetime.now(tz=self._tz)
        if self.heightened_until and now < self.heightened_until:
            self.mode = "heightened"
        else:
            self.mode = "normal"
        if self.mode != _prev_mode:
            try:
                self.discord.mode_changed(old_mode=_prev_mode, new_mode=self.mode)
            except Exception:
                pass
            # Trigger immediate id segment refresh so heightened/normal
            # station ID goes on air on the next cycle rotation.
            try:
                self.refresher.trigger_immediate("id")
            except AttributeError:
                pass  # refresher not yet initialised

    def _heightened_ago_str(self) -> str | None:
        if not self.last_heightened_at:
            return None
        delta = dt.datetime.now(tz=self._tz) - self.last_heightened_at
        mins = int(delta.total_seconds() // 60)
        if mins < 1:
            return "less than one minute"
        if mins < 60:
            return f"{mins} minutes"
        hrs = mins // 60
        rem = mins % 60
        return f"{hrs} hours" if rem == 0 else f"{hrs} hours and {rem} minutes"

    def _has_focus_holding_alerts(self) -> bool:
        try:
            return any(
                alert_holds_focus(a, self.cfg.cycle.alert_focus)
                for a in self.alert_tracker.get_cycle_alerts()
            )
        except Exception:
            return False

    def _cycle_interval_seconds(self) -> int:
        if self.mode == "heightened" or self._has_focus_holding_alerts():
            return self.cfg.cycle.heightened_interval_seconds
        return self.cfg.cycle.normal_interval_seconds

    def _schedule_cycle_refill(self, reason: str) -> None:
        """Reset continuous-cycle buffer/order after an external state change."""
        try:
            self.refresher.trigger_immediate("id", "status")
            self.refresher.notify_alerts_changed()
        except AttributeError:
            pass  # refresher not yet initialised (startup edge case)
        except Exception:
            log.debug("Cycle refill: refresher notify failed", exc_info=True)
        try:
            self.conductor.notify_flush(reset_rotation=True, reason=reason)
        except AttributeError:
            pass  # conductor not yet initialised (startup edge case)

    # ---- dedupe helpers ----
    def _sha1_12(self, s: str) -> str:
        h = hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()
        return h[:12]

    def _dedupe_func_full_key(self, event_code: str, same_locs: list[str] | None) -> str | None:
        """
        Cross-source "functional" FULL-alert dedupe key.

        ERN cannot map to VTEC, so we dedupe FULL tone-outs by:
          (event code) + (in-area SAME locations)

        Locations are normalized to:
          - service-area filtered
          - unique
          - order-independent (sorted)

        Returns None if no usable locations exist (avoid deduping on empty targets).
        """
        code_u = _safe_event_code(event_code).strip().upper()
        locs_in = [str(x).strip() for x in (same_locs or []) if str(x).strip()]
        locs = self._filter_same_locations_to_service_area(locs_in)
        if not locs:
            return None
        locs_norm = sorted(set(locs))
        blob = code_u + "|" + ",".join(locs_norm)
        return f"FUNC_FULL:{code_u}:{self._sha1_12(blob)}"


    def _nwws_api_product_matches_raw(self, parsed: ParsedProduct, api_text: str) -> bool:
        """
        Accept an api.weather.gov product override only when it appears to be the
        same issuance as the NWWS payload we already received.

        This protects against the API returning an older "latest" product for the
        same product type/location, which can otherwise cause stale VTEC, stale
        expiries, and incorrect cross-source dedupe decisions.
        """
        if not api_text or not api_text.strip():
            return False

        api_parsed = parse_product_text(api_text)
        if api_parsed:
            raw_awips = (parsed.awips_id or "").strip().upper()
            api_awips = (api_parsed.awips_id or "").strip().upper()
            if raw_awips and api_awips and raw_awips != api_awips:
                return False

            raw_wfo = (parsed.wfo or "").strip().upper()
            api_wfo = (api_parsed.wfo or "").strip().upper()
            if raw_wfo and api_wfo and raw_wfo != api_wfo:
                return False

        raw_vtec = self._extract_vtec(parsed.raw_text or "")
        api_vtec = self._extract_vtec(api_text)
        raw_track_actions = set(self._vtec_tracks(raw_vtec))
        api_track_actions = set(self._vtec_tracks(api_vtec))
        raw_tracks = {track for (track, _act) in raw_track_actions}
        api_tracks = {track for (track, _act) in api_track_actions}

        # If NWWS already gave us a concrete VTEC action for a concrete track,
        # the API override must agree on at least one identical (track, action)
        # pair. Matching only the track is too weak: a stale EXA/CON product can
        # otherwise override a newer EXP/CAN product on the same ETN and cause a
        # full tone-out for what should have been a voice-only expiration.
        if raw_track_actions:
            return bool(api_track_actions) and bool(raw_track_actions & api_track_actions)

        # If the raw payload has track IDs but no usable action pair match, reject.
        if raw_tracks:
            return bool(api_tracks) and bool(raw_tracks & api_tracks)

        # No VTEC in raw payload: fall back to AWIPS/WFO agreement only.
        return True

    async def _resolve_nwws_official_text(self, parsed: ParsedProduct) -> tuple[str, str | None]:
        """
        Prefer the live NWWS payload unless the API product can be validated as the
        same issuance. This avoids stale-product regressions during active events.
        """
        official_text = parsed.raw_text or ""
        pid: str | None = None
        try:
            pid = await self.api.latest_product_id(
                parsed.product_type,
                parsed.wfo[1:] if parsed.wfo.startswith("K") else parsed.wfo,
            )
            if not pid:
                pid = await self.api.latest_product_id(parsed.product_type, parsed.wfo.replace("K", "", 1))
            if pid:
                prod = await self.api.get_product(pid)
                if prod and prod.product_text:
                    if self._nwws_api_product_matches_raw(parsed, prod.product_text):
                        official_text = prod.product_text
                    else:
                        api_vtec = ",".join(self._extract_vtec(prod.product_text)[:2])
                        raw_vtec = ",".join(self._extract_vtec(parsed.raw_text or "")[:2])
                        log.warning(
                            "NWWS API override rejected (stale/mismatched product): type=%s awips=%s wfo=%s pid=%s raw_vtec=%s api_vtec=%s",
                            parsed.product_type,
                            parsed.awips_id or "",
                            parsed.wfo,
                            pid,
                            raw_vtec,
                            api_vtec,
                        )
                        pid = None
        except Exception:
            log.exception('NWWS official-text resolution failed; falling back to raw payload')
            pid = None
        return official_text, pid

    def _extract_vtec(self, text: str) -> list[str]:
        if not text:
            return []
        found = _VTEC_FIND_RE.findall(text)
        out: list[str] = []
        seen: set[str] = set()
        for v in found:
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
            if len(out) >= 6:
                break
        return out

    def _vtec_tracks(self, vtec_list: list[str]) -> list[tuple[str, str]]:
        """
        Return [(track_id, action)] where track_id := OFFICE.PHEN.SIG.ETN, action := NEW/CON/EXT/UPG/etc
        """
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for raw in vtec_list or []:
            s = "".join(str(raw).split()).strip()
            if not s:
                continue
            m = _VTEC_PARSE_RE.search(s)
            if not m:
                continue
            office = m.group("office")
            phen = m.group("phen")
            sig = m.group("sig")
            etn = m.group("etn")
            act = m.group("act")
            track = f"{office}.{phen}.{sig}.{etn}"
            k = f"{track}|{act}"
            if k in seen:
                continue
            seen.add(k)
            out.append((track, act))
        return out[:12]

    async def _dedupe_prune(self) -> None:
        now = dt.datetime.now(tz=self._tz)
        ttl = float(self._dedupe_ttl_seconds)
        dead: list[str] = []
        for k, ts in self._recent_air_keys.items():
            if (now - ts).total_seconds() > ttl:
                dead.append(k)
        for k in dead:
            self._recent_air_keys.pop(k, None)

    async def _dedupe_reserve(self, keys: list[str]) -> tuple[bool, str]:
        """
        Reserve dedupe keys *before* airing to avoid races.
        If we later fail to air, caller should release.
        """
        now = dt.datetime.now(tz=self._tz)
        async with self._dedupe_lock:
            await self._dedupe_prune()
            for k in keys:
                if k in self._recent_air_keys:
                    return (False, k)
            for k in keys:
                self._recent_air_keys[k] = now
            return (True, "")

    async def _dedupe_release(self, keys: list[str]) -> None:
        async with self._dedupe_lock:
            for k in keys:
                self._recent_air_keys.pop(k, None)

    def _cap_vtec_list(self, ev: "CapAlertEvent") -> list[str]:  # type: ignore[name-defined]
        vals: list[str] = []

        # NEW: prefer explicit ev.vtec (cap_nws now populates it)
        v0 = getattr(ev, "vtec", None)
        if isinstance(v0, (list, tuple)):
            vals.extend(str(x).strip() for x in v0 if str(x).strip())
        elif isinstance(v0, str) and v0.strip():
            vals.append(v0.strip())

        # Back-compat: still accept alternative attributes if present
        for attr in ("vtec_codes", "vtec_code", "vtecList"):
            v = getattr(ev, attr, None)
            if not v:
                continue
            if isinstance(v, str):
                vals.append(v.strip())
            elif isinstance(v, (list, tuple)):
                vals.extend(str(x).strip() for x in v if str(x).strip())

        params = getattr(ev, "parameters", None)
        if isinstance(params, dict):
            for k, v in params.items():
                if str(k).strip().upper() == "VTEC":
                    if isinstance(v, str):
                        vals.append(v.strip())
                    elif isinstance(v, (list, tuple)):
                        vals.extend(str(x).strip() for x in v if str(x).strip())

        out: list[str] = []
        seen: set[str] = set()
        for x in vals:
            x2 = "".join(str(x).split()).strip()
            if not x2:
                continue
            if x2 in seen:
                continue
            seen.add(x2)
            out.append(x2)
        return out[:12]

    # ---- SAME location filtering / area display ----
    def _filter_same_locations_to_service_area(
        self,
        locs: list[str] | tuple[str, ...] | None,
        *,
        allow_statewide_input: bool = True,
    ) -> list[str]:
        return self.targeting._filter_same_locations_to_service_area(
            locs,
            allow_statewide_input=allow_statewide_input,
        )

    async def _sf_area_text_from_same_codes(self, same_codes: list[str]) -> str:
        return await self.targeting._sf_area_text_from_same_codes(same_codes)

    async def _nwws_same_targets_from_texts(
        self,
        primary_text: str,
        secondary_text: str,
    ) -> tuple[list[str], list[str], str, bool, "dt.datetime | None"]:
        return await self.targeting._nwws_same_targets_from_texts(primary_text, secondary_text)

    async def _nwws_wcn_watch_same_targets_from_area_desc(self, official_text: str) -> list[str]:
        return await self.targeting._nwws_wcn_watch_same_targets_from_area_desc(official_text)


    # ---- Rebroadcast rotation (no re-tone) ----
    def _parse_vtec_dt_utc(self, s: str) -> dt.datetime | None:
        '''
        Parse VTEC timestamps like:
          20260111T2300Z  (YYYYMMDD)
          260111T2300Z    (YYMMDD legacy)
        Returns an aware UTC datetime or None.
        '''
        txt = (s or "").strip().upper()
        m = re.fullmatch(r"(\d{8}|\d{6})T(\d{4})Z", txt)
        if not m:
            return None

        d = m.group(1)
        hm = m.group(2)

        try:
            if len(d) == 8:
                year = int(d[0:4])
                month = int(d[4:6])
                day = int(d[6:8])
            else:
                year = 2000 + int(d[0:2])
                month = int(d[2:4])
                day = int(d[4:6])

            hour = int(hm[0:2])
            minute = int(hm[2:4])

            return dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc)
        except Exception:
            return None

    def _best_expiry_from_vtec(self, vtec_list: list[str]) -> dt.datetime | None:
        '''
        Returns the latest END time found across VTEC codes (UTC), or None.
        '''
        ends: list[dt.datetime] = []
        for raw in vtec_list or []:
            s = "".join(str(raw).split()).strip()
            if not s:
                continue

            # Pull the END token after the '-' if present: ...-YYYYMMDDThhmmZ/
            m = re.search(r"-((?:\d{8}|\d{6})T\d{4}Z)", s)
            if not m:
                continue

            t = self._parse_vtec_dt_utc(m.group(1))
            if t:
                ends.append(t)

        if not ends:
            return None
        return max(ends)

    # ---- CAP toggles ----
    def _cap_enabled(self) -> bool:
        return self.cfg.cap.enabled

    def _cap_dryrun(self) -> bool:
        return self.cfg.cap.dryrun

    def _cap_poll_seconds(self) -> int:
        return self.cfg.cap.poll_seconds

    def _cap_user_agent(self) -> str:
        return self.cfg.cap.user_agent

    def _cap_url(self) -> str:
        return self.cfg.cap.url

    def _cap_full_enabled(self) -> bool:
        return self.cfg.cap.full.enabled

    def _cap_full_severities(self) -> set[str]:
        return {s.strip().lower() for s in self.cfg.cap.full.severities if s.strip()}

    def _cap_full_events(self) -> set[str]:
        events = [e.strip() for e in self.cfg.cap.full.events if e.strip()]
        if events:
            return set(events)
        # Empty list in yaml means "match all qualifying severities" — use the canonical default set
        return {
            "Tornado Warning",
            "Severe Thunderstorm Warning",
            "Flash Flood Warning",
            "Flood Warning",
            "Hurricane Warning",
            "Tropical Storm Warning",
            "Storm Surge Warning",
            "Extreme Wind Warning",
            "Blizzard Warning",
            "Winter Storm Warning",
            "Ice Storm Warning",
            "High Wind Warning",
            "Wind Chill Warning",
            "Tornado Watch",
            "Severe Thunderstorm Watch",
            "Flash Flood Watch",
            "Flood Watch",
            "Hurricane Watch",
            "Tropical Storm Watch",
            "Storm Surge Watch",
            "Blizzard Watch",
            "Winter Storm Watch",
            "Ice Storm Watch",
            "High Wind Watch",
            "Wind Chill Watch",
            "Winter Weather Advisory",
            "Snow Squall Warning",
            # Marine
            "Special Marine Warning",
        }

    def _cap_full_cooldown_seconds(self) -> int:
        return self.cfg.cap.full.cooldown_seconds

    def _cap_voice_enabled(self) -> bool:
        return self.cfg.cap.voice.enabled

    def _cap_voice_events(self) -> set[str]:
        return {e.strip() for e in self.cfg.cap.voice.events if e.strip()}

    def _cap_voice_cooldown_seconds(self) -> int:
        return self.cfg.cap.voice.cooldown_seconds

    # ---- IPAWS CAP feed toggles ----
    def _ipaws_enabled(self) -> bool:
        return self.cfg.ipaws.enabled

    def _ipaws_dryrun(self) -> bool:
        return self.cfg.ipaws.dryrun

    def _ipaws_poll_seconds(self) -> int:
        return self.cfg.ipaws.poll_seconds

    def _ipaws_user_agent(self) -> str:
        return self.cfg.ipaws.user_agent

    def _ipaws_url(self) -> str:
        return self.cfg.ipaws.url

    def _ipaws_full_events(self) -> set[str]:
        return set(self.cfg.ipaws.full_events)

    def _ipaws_voice_events(self) -> set[str]:
        return set(self.cfg.ipaws.voice_events)

    def _ipaws_ern_dedup_ttl(self) -> int:
        return self.cfg.ipaws.ern_dedup_ttl_seconds

    # ---- ERN/GWES SAME monitor toggles ----
    def _ern_enabled(self) -> bool:
        return self.cfg.ern.enabled

    def _ern_dryrun(self) -> bool:
        return self.cfg.ern.dryrun

    def _ern_url(self) -> str:
        return self.cfg.ern.url.strip()

    def _ern_relay_enabled(self) -> bool:
        return self.cfg.ern.relay.enabled

    def _ern_relay_events(self) -> set[str]:
        return {e.strip().upper() for e in self.cfg.ern.relay.events if e.strip()}

    def _ern_relay_min_confidence(self) -> float:
        return self.cfg.ern.relay.min_confidence

    def _ern_relay_cooldown_seconds(self) -> int:
        return self.cfg.ern.relay.cooldown_seconds

    def _ern_relay_senders(self) -> set[str]:
        senders = [s.strip().upper() for s in self.cfg.ern.relay.senders if s.strip()]
        return set(senders)

    # ---- SAME toggles ----
    # ---- RWT/RMT scheduler toggles ----
    def _tests_enabled(self) -> bool:
        return self.cfg.tests.enabled

    def _tests_postpone_minutes(self) -> int:
        return self.cfg.tests.postpone_minutes

    def _tests_max_postpone_hours(self) -> int:
        return self.cfg.tests.max_postpone_hours

    def _tests_jitter_seconds(self) -> int:
        return self.cfg.tests.jitter_seconds

    def _tests_toneout_cooldown_seconds(self) -> int:
        return self.cfg.tests.toneout_cooldown_seconds

    def _tests_cap_block_seconds(self) -> int:
        return self.cfg.tests.cap_block_seconds

    def _tests_ern_block_seconds(self) -> int:
        return self.cfg.tests.ern_block_seconds

    async def _local_test_presentation(self, code: str, same_codes: list[str] | None = None) -> tuple[str, str, str]:
        event_text = _same_label_or_code(code)
        station_name = str(self.cfg.station.name or "SeasonalWeather").strip() or "SeasonalWeather"
        service_area_name = str(self.cfg.station.service_area_name or "service area").strip() or "service area"
        codes = [str(x).strip() for x in (same_codes or []) if str(x).strip()]

        auto_area_text = ""
        if codes:
            try:
                auto_area_text = await self._sf_area_text_from_same_codes(codes)
            except Exception:
                auto_area_text = ""
        auto_area_text = auto_area_text or service_area_name

        pres = self.cfg.tests.presentation
        fmt_ctx = {
            "code": str(code or "").strip().upper(),
            "event": event_text,
            "station_name": station_name,
            "service_area_name": service_area_name,
            "auto_area_text": auto_area_text,
        }
        headline = format_test_presentation_template(
            pres.headline_template,
            **fmt_ctx,
        ) or f"{event_text} for the {service_area_name}"
        area_text = format_test_presentation_template(
            pres.area_text,
            **fmt_ctx,
        ) or auto_area_text
        discord_area_text = format_test_presentation_template(
            pres.discord_area_text,
            **{**fmt_ctx, "area_text": area_text},
        ) or area_text
        return headline, area_text, discord_area_text

    def _test_same_codes_for_presentation(self) -> list[str]:
        try:
            codes = sorted(getattr(self, "_same_fips_allow_set", None) or [])
        except Exception:
            codes = []
        return [str(x).strip() for x in codes if str(x).strip()]

    def _tests_gate(self, event_code: str = "") -> tuple[bool, str]:
        now = dt.datetime.now(tz=self._tz)
        code = str(event_code or "").strip().upper()
        test_cfg = self.cfg.tests.rwt if code == "RWT" else self.cfg.tests.rmt
        gate = test_cfg.gate

        if gate.block_heightened and self.heightened_until and now < self.heightened_until:
            return (False, "heightened mode active")
        if gate.block_recent_toneout and self.last_toneout_at:
            if (now - self.last_toneout_at).total_seconds() < self._tests_toneout_cooldown_seconds():
                return (False, "recent tone-out cooldown")

        if gate.block_recent_severe_cap and self.cap_last_severe_at:
            if (now - self.cap_last_severe_at).total_seconds() < self._tests_cap_block_seconds():
                return (False, "recent severe CAP match")

        if gate.block_recent_ern and self.ern_last_tone_at:
            if (now - self.ern_last_tone_at).total_seconds() < self._tests_ern_block_seconds():
                return (False, "recent ERN SAME activity")

        return (True, "ok")

    async def _originate_required_test(self, event_code: str) -> None:
        """
        Originates a local RWT/RMT using the existing SAME+audio pipeline.
        Does NOT trigger heightened mode.
        """
        code = (event_code or "").strip().upper()
        if code not in {"RWT", "RMT"}:
            return

        # Script: use config override when provided, else fall back to built-in default.
        _cfg_lines: tuple[str, ...] = (
            self.cfg.tests.rwt.script_lines if code == "RWT" else self.cfg.tests.rmt.script_lines
        )
        if _cfg_lines:
            lines = list(_cfg_lines)
        else:
            lines = default_test_script_lines(code)

        spoken = "\n".join(lines).strip()

        dummy = SimpleNamespace(product_type=code, awips_id=None, wfo="KLWX", raw_text="")
        out_wav = await self.audio_originator.render_alert_audio(dummy, spoken)

        async with self._cycle_lock:
            try:
                self.telnet.flush_cycle()
            except Exception:
                pass
            tkey = "rwt" if code == "RWT" else "rmt"
            title = self._np_alert_title(tkey, event="")
            meta = self._np_meta(title=title, kind="test", extra={"sw_alert_source": "local", "sw_event_code": code})
            self.telnet.push_alert(str(out_wav), meta=meta)

        # --- Station feed note (radio UI/API handled-alerts feed) ---
        local_test_same_codes = self._test_same_codes_for_presentation()
        test_headline = f"Required {'Weekly' if code == 'RWT' else 'Monthly'} Test"
        test_area_text = str(self.cfg.station.service_area_name or "SeasonalWeather").strip() or "SeasonalWeather"
        discord_test_area_text = test_area_text
        try:
            test_headline, test_area_text, discord_test_area_text = await self._local_test_presentation(
                code,
                local_test_same_codes,
            )
        except Exception:
            pass

        _station_feed_note_required_test(
            code=code,
            headline=test_headline,
            area_text=test_area_text,
            same_codes=local_test_same_codes,
            out_wav=str(out_wav),
        )

        self._schedule_cycle_refill("post-test")
        log.info("Originated %s test (audio=%s)", code, out_wav)
        # _TEST_DL_v3_
        self.discord.alert_aired(
            code=code,
            event=f"Required {'Weekly' if code == 'RWT' else 'Monthly'} Test",
            source="SeasonalWeather (local)",
            mode="full",
            area=discord_test_area_text,
            same_codes=local_test_same_codes,
            is_test=True,
        )

    async def run(self) -> None:
        work, audio, cache, logs = self._paths()
        for p in (work, audio, cache, logs):
            p.mkdir(parents=True, exist_ok=True)

        await self._wait_for_liquidsoap()
        self.discord.service_started(
            cap_enabled=self._cap_enabled(),
            ern_enabled=self._ern_enabled(),
            tests_enabled=self._tests_enabled(),
            mode=self.mode,
        )

        # --- Persistent alert state: restore from disk, drop expired ---
        try:
            _loaded = self.alert_tracker.load()
            _purged = self.alert_tracker.purge_expired()
            log.info(
                "AlertTracker: loaded %d entries, purged %d expired on startup",
                _loaded, _purged,
            )
        except Exception:
            log.exception("AlertTracker: startup load/purge failed")
        # _TRACKER_DL_
        try:
            self.discord.alerttracker_lifecycle(
                loaded=_loaded,
                purged=_purged,
                active=len(self.alert_tracker.get_cycle_alerts()),
            )
        except Exception:
            pass

        try:
            _sf_restored_tracker = _station_feed_seed_from_alert_tracker(self.alert_tracker)
            if _sf_restored_tracker and _sf_enabled():
                log.info(
                    "Station feed: restored %d alerts from AlertTracker into SQLite read model on startup",
                    _sf_restored_tracker,
                )
        except Exception:
            log.exception("Station feed: startup restore from tracker failed")

        tasks: list[asyncio.Task] = []

        async def _health_probe_cap_api() -> None:
            await self.api.active_alerts(self.cycle_builder.alert_areas)

        async def _health_probe_nws_api() -> None:
            await self.api.latest_product_id("HWO", "LWX")

        self.health_state.register_probe("cap_api", _health_probe_cap_api)
        self.health_state.register_probe("nws_api", _health_probe_nws_api)

        if self.cfg.nwws.credentials_defaulted or not self.cfg.nwws.enabled:
            self.health_state.mark_disabled("nwws_oi", "nwws_disabled")
        if not self._cap_enabled():
            self.health_state.mark_disabled("cap_api", "cap_disabled")

        def _health_changed(_ctx) -> None:
            try:
                self.refresher.trigger_immediate("id", "health", "status")
                self._schedule_cycle_refill("health-state-change")
            except Exception:
                log.exception("Health state change refresh failed")

        tasks.append(asyncio.create_task(self.health_state.run_forever(on_change=_health_changed), name="health_state"))

        # CycleConductor + SegmentRefresher own routine cycle scheduling.
        tasks.append(asyncio.create_task(self.conductor.run(), name="conductor"))
        tasks.append(asyncio.create_task(self.refresher.run(), name="segment_refresher"))
        tasks.append(asyncio.create_task(self._pns_backfill_loop(), name="pns_api_backfill"))

        if self.cfg.nwws.credentials_defaulted:
            log.warning(
                "NWWS-OI disabled because NWWS_JID/NWWS_PASSWORD are unset or still use the example CHANGEME values; "
                "update /etc/seasonalweather/seasonalweather.env to enable NWWS-OI."
            )
        elif not self.cfg.nwws.enabled:
            log.info("NWWS-OI disabled (set nwws.enabled: true in config.yaml to enable)")
        else:
            xmpp = NWWSClient(
                self.jid, self.password, self.nwws_server, self.nwws_port, self.nwws_queue,
                room_jid=self.cfg.nwws.room,
                nick=self.cfg.nwws.nick,
                # TODO: wire stall/reconnect callbacks to self.discord.nwws_stall() / .nwws_reconnected() once NWWSClient exposes them
                stall_seconds=self.cfg.nwws.resiliency.stall_seconds,
                muc_confirm_seconds=self.cfg.nwws.resiliency.muc_confirm_seconds,
                start_wait_seconds=self.cfg.nwws.resiliency.start_wait_seconds,
                join_wait_seconds=self.cfg.nwws.resiliency.join_wait_seconds,
                backoff_max_seconds=self.cfg.nwws.resiliency.backoff_max_seconds,
            )
            tasks.append(asyncio.create_task(xmpp.run_forever(), name="nwws_xmpp"))
            tasks.append(asyncio.create_task(self._consume_nwws(), name="nwws_consumer"))
        # CycleConductor runs the cycle continuously.


        if self._cap_enabled():
            if NwsCapPoller is None or CapAlertEvent is None:
                log.warning("CAP enabled but cap_nws.py import failed; CAP is disabled.")
            else:
                kwargs = dict(
                    out_queue=self.cap_queue,
                    same_fips_allow=self.cfg.service_area.same_fips_all,
                    poll_seconds=self._cap_poll_seconds(),
                    user_agent=self._cap_user_agent(),
                    ledger_path=self.cfg.cap.ledger_path,
                    ledger_max_age_days=self.cfg.cap.ledger_max_age_days,
                    database=self.database,
                )
                url = self._cap_url().strip()
                if url:
                    kwargs["url"] = url  # type: ignore[assignment]

                cap = NwsCapPoller(**kwargs)  # type: ignore[arg-type]
                tasks.append(asyncio.create_task(cap.run_forever(), name="cap_poller"))
                tasks.append(asyncio.create_task(self._consume_cap(), name="cap_consumer"))
                log.info("CAP ingest enabled (dryrun=%s full=%s voice=%s)", self._cap_dryrun(), self._cap_full_enabled(), self._cap_voice_enabled())
        else:
            log.info("CAP ingest disabled (set cap.enabled: true in config.yaml to enable)")

        if self._ipaws_enabled():
            if IpawsCapPoller is None or IpawsCapEvent is None:
                log.warning("IPAWS enabled but ipaws_cap.py import failed; IPAWS is disabled.")
            else:
                ipaws_poller = IpawsCapPoller(
                    out_queue=self.ipaws_queue,
                    same_fips_allow=self.cfg.service_area.same_fips_all,
                    poll_seconds=self._ipaws_poll_seconds(),
                    user_agent=self._ipaws_user_agent(),
                    url=self._ipaws_url(),
                    ledger_path=self.cfg.ipaws.ledger_path,
                    ledger_max_age_days=self.cfg.ipaws.ledger_max_age_days,
                    database=self.database,
                )
                tasks.append(asyncio.create_task(ipaws_poller.run_forever(), name="ipaws_poller"))
                tasks.append(asyncio.create_task(self.ipaws_runtime.run(), name="ipaws_consumer"))
                log.info(
                    "IPAWS ingest enabled (dryrun=%s full_events=%s)",
                    self._ipaws_dryrun(),
                    ",".join(sorted(self._ipaws_full_events())),
                )
        else:
            log.info("IPAWS ingest disabled (set ipaws.enabled: true in config.yaml to enable)")

        if self._ern_enabled():
            if ErnGwesMonitor is None or ErnSameEvent is None:
                log.warning("ERN enabled but ern_gwes.py import failed; ERN is disabled.")
            else:
                url = self._ern_url()
                if not url:
                    log.warning("ERN enabled but SEASONAL_ERN_URL is empty; ERN is disabled.")
                else:
                    ern_cfg = self.cfg.ern
                    mon = ErnGwesMonitor(
                        out_queue=self.ern_queue,
                        same_fips_allow=self.cfg.service_area.same_fips_all,
                        url=url,
                        sample_rate=ern_cfg.sample_rate,
                        dedupe_seconds=ern_cfg.dedupe_seconds,
                        trigger_ratio=ern_cfg.trigger_ratio,
                        tail_seconds=ern_cfg.tail_seconds,
                        confidence_min=ern_cfg.confidence_min,
                        name=ern_cfg.name,
                        decoder_backend=ern_cfg.decoder_backend,
                    )
                    tasks.append(asyncio.create_task(mon.run_forever(), name="ern_monitor"))
                    tasks.append(asyncio.create_task(self.ern_relay_runtime.run(), name="ern_consumer"))
                    log.info(
                        "ERN monitor enabled (dryrun=%s url=%s relay=%s decoder=%s)",
                        self._ern_dryrun(),
                        url,
                        self._ern_relay_enabled(),
                        ern_cfg.decoder_backend,
                    )
        else:
            log.info("ERN monitor disabled (set ern.enabled: true in config.yaml to enable)")

        if self._tests_enabled():
            try:
                state_path = str(Path(self.cfg.paths.work_dir) / "rwt_rmt_state.json")

                sched = RwtRmtSchedule(
                    enabled=True,
                    tz_name=self.cfg.station.timezone,

                    rwt_enabled=True,
                    rwt_weekday=self.cfg.tests.rwt.weekday,
                    rwt_hour=self.cfg.tests.rwt.hour,
                    rwt_minute=self.cfg.tests.rwt.minute,

                    rmt_enabled=True,
                    rmt_nth=self.cfg.tests.rmt.nth,
                    rmt_weekday=self.cfg.tests.rmt.weekday,
                    rmt_hour=self.cfg.tests.rmt.hour,
                    rmt_minute=self.cfg.tests.rmt.minute,

                    jitter_seconds=self._tests_jitter_seconds(),
                    postpone_minutes=self._tests_postpone_minutes(),
                    max_postpone_hours=self._tests_max_postpone_hours(),
                    state_path=state_path,
                    state_key="rwt_rmt",
                    rwt_postpone_policy=self.cfg.tests.rwt.postpone_policy,
                    rwt_postpone_minutes=self.cfg.tests.rwt.postpone_minutes,
                    rwt_max_postpone_hours=self.cfg.tests.rwt.max_postpone_hours,
                    rwt_max_postpone_days=self.cfg.tests.rwt.max_postpone_days,
                    rmt_postpone_policy=self.cfg.tests.rmt.postpone_policy,
                    rmt_postpone_minutes=self.cfg.tests.rmt.postpone_minutes,
                    rmt_max_postpone_hours=self.cfg.tests.rmt.max_postpone_hours,
                    rmt_max_postpone_days=self.cfg.tests.rmt.max_postpone_days,
                )

                def _rlog(s: str) -> None:
                    log.info("%s", s)

                rsch = RwtRmtScheduler(
                    schedule=sched,
                    gate_fn=self._tests_gate,
                    fire_fn=self._originate_required_test,
                    log_fn=_rlog,
                    database=self.database,
                )
                tasks.append(asyncio.create_task(rsch.run_forever(), name="rwt_rmt_scheduler"))
                log.info("RWT/RMT scheduler enabled (state=%s)", state_path)
            except Exception:
                log.exception("Failed to start RWT/RMT scheduler")
        else:
            log.info("RWT/RMT scheduler disabled (set tests.enabled: true in config.yaml to enable)")

        if self.db_housekeeper is not None:
            tasks.append(asyncio.create_task(self.db_housekeeper.run_forever(), name="database_housekeeping"))

        tasks.append(asyncio.create_task(self.discord.start(), name="discord_log_drain"))
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            exc = t.exception()
            if exc:
                for p in pending:
                    p.cancel()
                raise exc

    async def _consume_nwws(self) -> None:
        while True:
            raw = await self.nwws_queue.get()
            self.health_state.mark_success("nwws_oi")

            # Flood-gate: allow the client logs for the first N messages, then silence them.
            self._nwws_raw_seen += 1
            if self._nwws_rx_log_first_n > 0 and self._nwws_raw_seen == self._nwws_rx_log_first_n:
                self._nwws_logger.setLevel(logging.WARNING)
                log.info(
                    "NWWS RX logging throttled after %d messages (seasonalweather.nwws -> WARNING)",
                    self._nwws_rx_log_first_n,
                )

            parsed = parse_product_text(raw)
            if not parsed:
                continue

            self._nwws_seen += 1

            allowed_wfo = (not self._nwws_allowed_wfos) or (parsed.wfo in self._nwws_allowed_wfos)
            toneout = parsed.product_type in self.cfg.policy.toneout_product_types
            raw_vtec = self._extract_vtec(parsed.raw_text or "")
            vtec_lifecycle = (not toneout) and self._vtec_matches_configured_toneout_code(raw_vtec)

            first_n = max(0, int(self._nwws_decision_log_first_n))
            every = max(0, int(self._nwws_decision_log_every))
            if (first_n and self._nwws_seen <= first_n) or (every and (self._nwws_seen % every) == 0):
                log.info(
                    "NWWS decision: #%d type=%s awips=%s wfo=%s allowed=%s toneout=%s vtec_lifecycle=%s",
                    self._nwws_seen,
                    parsed.product_type,
                    parsed.awips_id or "",
                    parsed.wfo,
                    allowed_wfo,
                    toneout,
                    vtec_lifecycle,
                )

            self.last_product_desc = f"{parsed.product_type} ({parsed.awips_id or ''})"

            # Trigger out-of-band segment refreshes for products whose content
            # feeds specific cycle segments so listeners hear fresh data sooner.
            try:
                _pt = (parsed.product_type or "").strip().upper()
                if _pt == "HWO":
                    self.refresher.trigger_immediate("hwo")
                elif _pt in {"RWS", "AFD", "SYN"}:
                    self.refresher.trigger_immediate("zfp")
                elif _pt == "RWR":
                    self.refresher.trigger_immediate("obs")
                    self.refresher.trigger_immediate("marine_obs")
                elif _pt in {"CWF", "CWA"}:
                    self.refresher.trigger_immediate("cwf")
                elif _pt in {"SWO", "SWS"}:
                    self.refresher.trigger_immediate("spc")
            except AttributeError:
                pass  # refresher not yet initialised

            if not allowed_wfo:
                continue

            if toneout or vtec_lifecycle:
                if vtec_lifecycle:
                    log.info(
                        "NWWS lifecycle carrier: type=%s awips=%s wfo=%s vtec=%s; routing to toneout handler",
                        parsed.product_type,
                        parsed.awips_id or "",
                        parsed.wfo,
                        ",".join(raw_vtec[:3]),
                    )
                await self._handle_toneout(parsed)

            # MWS (Marine Weather Statement) — Special Marine Warning lifecycle VTEC carrier.
            # SM.W (Special Marine Warning) is only issued ONCE by the NWS; all subsequent
            # lifecycle actions (CON, EXT, CAN, EXP) come through MWS products carrying
            # MA.W VTEC.  Routine MWSs without VTEC are silently ignored to avoid flooding
            # the schedule with routine marine text.
            elif (parsed.product_type or "").strip().upper() == "MWS":
                try:
                    _mws_vtec = self._extract_vtec(parsed.raw_text or "")
                    _mws_has_marine = any(
                        ".MA.W." in v or ".MA.A." in v
                        for v in _mws_vtec
                    )
                    if _mws_has_marine:
                        log.info(
                            "NWWS MWS: marine VTEC detected (%s), routing to toneout handler wfo=%s awips=%s",
                            ",".join(_mws_vtec[:3]),
                            parsed.wfo,
                            parsed.awips_id or "",
                        )
                        await self._handle_toneout(parsed)
                    else:
                        log.debug(
                            "NWWS MWS: no marine VTEC, ignoring wfo=%s awips=%s",
                            parsed.wfo,
                            parsed.awips_id or "",
                        )
                except Exception:
                    log.exception("NWWS MWS VTEC check failed wfo=%s awips=%s", parsed.wfo, parsed.awips_id or "")

            # PNS cycle injection — delegated to the configurable PNS state machine.
            # Only explicitly allowed, coherent PNS subtypes become cycle audio.
            elif (parsed.product_type or "").strip().upper() == "PNS":
                try:
                    official_pns = parsed.raw_text
                    try:
                        pid_pns = await self.api.latest_product_id("PNS", parsed.wfo[1:] if parsed.wfo.startswith("K") else parsed.wfo)
                        if pid_pns:
                            prod_pns = await self.api.get_product(pid_pns)
                            if prod_pns and prod_pns.product_text:
                                same_product = self._nwws_api_product_matches_raw(parsed, prod_pns.product_text)
                                same_issuance = pns_text_same_issuance(
                                    parsed.raw_text or "",
                                    prod_pns.product_text,
                                    raw_fallback=getattr(parsed, "issued", None),
                                    candidate_fallback=getattr(prod_pns, "issuance_time", None),
                                )
                                if same_product and same_issuance:
                                    official_pns = prod_pns.product_text
                                else:
                                    raw_issued = parse_nws_header_issued_dt(
                                        parsed.raw_text or "", fallback=getattr(parsed, "issued", None)
                                    )
                                    api_issued = parse_nws_header_issued_dt(
                                        prod_pns.product_text, fallback=getattr(prod_pns, "issuance_time", None)
                                    )
                                    log.warning(
                                        "PNS API override rejected (stale/mismatched product): wfo=%s awips=%s pid=%s same_product=%s raw_issued=%s api_issued=%s",
                                        parsed.wfo,
                                        parsed.awips_id or "",
                                        pid_pns,
                                        same_product,
                                        raw_issued.isoformat() if raw_issued else "",
                                        api_issued.isoformat() if api_issued else "",
                                    )
                    except Exception:
                        log.exception("PNS official-text resolution failed; falling back to raw NWWS payload")

                    decision = self.pns_state.evaluate(
                        official_pns,
                        wfo=parsed.wfo or "",
                        awips_id=parsed.awips_id or "",
                        issued=getattr(parsed, "issued", None),
                    )
                    queued = await self._queue_pns_decision(
                        decision,
                        wfo=parsed.wfo or "",
                        awips_id=parsed.awips_id or "",
                        context="nwws",
                    )
                    if not queued:
                        continue
                except Exception:
                    log.exception("PNS handler error wfo=%s", parsed.wfo)

    async def _queue_pns_decision(self, decision, *, wfo: str, awips_id: str, context: str) -> bool:
        """Queue an accepted PNS decision as cycle-only active-alert state."""
        if not decision.is_audio:
            log.info(
                "PNS audio suppressed; source=%s wfo=%s awips=%s action=%s subtype=%s reason=%s signals=%s",
                context,
                wfo,
                awips_id or "",
                decision.action,
                decision.subtype,
                decision.reason,
                ",".join(decision.signals) or "-",
            )
            return False

        ok_pns, _ = await self._dedupe_reserve([decision.key])
        if not ok_pns:
            log.info("PNS skipped (dedupe); source=%s id=%s subtype=%s wfo=%s", context, decision.key, decision.subtype, wfo)
            return False

        pns_exp_utc = decision.expires_utc or (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=4))
        pns_issued_utc = decision.issued_utc or dt.datetime.now(dt.timezone.utc)
        pns_ae = ActiveAlert(
            id=decision.key,
            source="PNS_CYCLE",
            event=decision.event,
            code=decision.code,
            vtec=[],
            headline=decision.headline or decision.event,
            script_text=decision.script_text,
            audio_path=None,
            expires=pns_exp_utc.isoformat(),
            issued=pns_issued_utc.isoformat(),
            same_locs=list(self.cfg.service_area.same_fips_all or []),
            cycle_only=True,
        )
        self.alert_tracker.add_or_update(pns_ae)
        self._schedule_cycle_refill(f"pns-{decision.subtype}")
        log.info(
            "PNS queued for cycle; source=%s id=%s subtype=%s event=%s wfo=%s awips=%s expires=%s signals=%s",
            context,
            decision.key,
            decision.subtype,
            decision.event,
            wfo,
            awips_id or "",
            pns_exp_utc.isoformat(),
            ",".join(decision.signals) or "-",
        )
        return True

    def _pns_backfill_wfos(self) -> list[str]:
        """Return 3-letter offices to poll for latest PNS backfill."""
        offices: set[str] = set()
        for raw in self._nwws_allowed_wfos:
            s = str(raw or "").strip().upper()
            if len(s) == 4 and s.startswith("K"):
                offices.add(s[1:])
            elif len(s) == 3:
                offices.add(s)
        return sorted(offices)

    async def _pns_backfill_latest_once(self) -> int:
        """Fetch latest API PNS products and queue any still-current audio-worthy PNS."""
        if not getattr(self.cfg.pns, "enabled", True):
            return 0

        queued = 0
        offices = self._pns_backfill_wfos()
        if not offices:
            log.debug("PNS API backfill skipped; no nwws.allowed_wfos configured")
            return 0

        for office in offices:
            try:
                pid = await self.api.latest_product_id("PNS", office)
                if not pid:
                    continue
                prod = await self.api.get_product(pid)
                if not prod or not prod.product_text:
                    continue

                parsed = parse_product_text(prod.product_text)
                wfo = (getattr(parsed, "wfo", None) or f"K{office}").strip().upper() if parsed else f"K{office}"
                awips_id = (getattr(parsed, "awips_id", None) or f"PNS{office}").strip().upper() if parsed else f"PNS{office}"

                decision = self.pns_state.evaluate(
                    prod.product_text,
                    wfo=wfo,
                    awips_id=awips_id,
                    issued=getattr(prod, "issuance_time", None),
                )
                if await self._queue_pns_decision(decision, wfo=wfo, awips_id=awips_id, context=f"api-backfill:{pid}"):
                    queued += 1
            except Exception:
                log.exception("PNS API backfill failed for office=%s", office)
        return queued

    async def _pns_backfill_loop(self) -> None:
        """Small recovery poller for missed NWWS-OI PNS products."""
        await asyncio.sleep(15)
        while True:
            queued = await self._pns_backfill_latest_once()
            if queued:
                log.info("PNS API backfill queued %d product(s)", queued)
            await asyncio.sleep(120)

    def _cap_is_actionable(self, ev: "CapAlertEvent") -> bool:  # type: ignore[name-defined]
        try:
            if str(ev.status or "").strip().lower() != "actual":
                return False
            mt = str(ev.message_type or "").strip().lower()
            if mt and mt not in {"alert", "update", "cancel"}:
                return False
        except Exception:
            return False
        return True

    def _cap_severity_str(self, ev: "CapAlertEvent") -> str:  # type: ignore[name-defined]
        return str(ev.severity or "").strip().lower()

    def _cap_event_to_same_code(self, event: str) -> str:
        e = (event or "").strip()
        m: dict[str, str] = {
            "Tornado Warning": "TOR",
            "Tornado Watch": "TOA",
            "Severe Thunderstorm Warning": "SVR",
            "Severe Thunderstorm Watch": "SVA",
            "Flash Flood Warning": "FFW",
            "Flash Flood Watch": "FFA",
            "Flood Warning": "FLW",
            "Flood Watch": "FLA",
            "Flood Advisory": "FLA",
            "Winter Storm Warning": "WSW",
            "Winter Storm Watch": "WSA",
            "Blizzard Warning": "BZW",
            "Blizzard Watch": "BZA",
            "Ice Storm Warning": "ISW",
            "Ice Storm Watch": "ISA",
            "Freeze Warning": "FZW",
            "Freeze Watch": "FZA",
            "Flash Freeze Warning": "FSW",
            "Winter Weather Advisory": "SPS",
            "Hurricane Warning": "HUW",
            "Hurricane Watch": "HUA",
            "Tropical Storm Warning": "TRW",
            "Tropical Storm Watch": "TRA",
            "Storm Surge Warning": "SSW",
            "Storm Surge Watch": "SSA",
            "High Wind Warning": "HWW",
            "High Wind Watch": "HWA",
            "Extreme Wind Warning": "EWW",
            "Wind Chill Warning": "WCW",
            "Wind Chill Watch": "WCA",
            "Snow Squall Warning": "SQW",
            "Special Weather Statement": "SPS",
            "Severe Weather Statement": "SVS",
            "Flood Statement": "FLS",
            "Flash Flood Statement": "FFS",
            "Hurricane Statement": "HLS",
        }
        if e in m:
            return m[e]

        words = [w for w in re.split(r"\s+", e) if w]
        if len(words) >= 1:
            code = "".join(ch for ch in "".join(w[0] for w in words[:3]) if ch.isalnum()).upper()
            if len(code) >= 3:
                return code[:3]

        return "SPS"


    def _vtec_matches_configured_toneout_code(self, vtec: list[str]) -> bool:
        """True when VTEC maps to a configured toneout event code.

        Lifecycle products such as SVS/FFS/FLS/MWS often carry the VTEC for
        the underlying warning/watch while their AWIPS product type is only a
        statement carrier.  This lets CON/EXT/CAN/EXP products follow the same
        handling path as the original NEW issuance without adding every carrier
        product to policy.toneout_product_types.
        """
        if not vtec:
            return False
        allowed_codes = {str(x).strip().upper() for x in self.cfg.policy.toneout_product_types if str(x).strip()}
        if not allowed_codes:
            return False
        return bool(set(_vtec_same_codes_for_vtec(vtec)) & allowed_codes)

    def _cap_should_full(self, ev: "CapAlertEvent") -> bool:  # type: ignore[name-defined]
        if not self._cap_full_enabled():
            return False
        if not self._cap_is_actionable(ev):
            return False

        # VTEC-aware gate for CAP Update:
        # CAP can send VTEC CON/EXT/COR/ROU as msgType=Update.
        # Do NOT FULL-tone those unless VTEC indicates a FULL-worthy action.
        try:
            mt = str(ev.message_type or "").strip().lower()
        except Exception:
            mt = ""

        if mt == "update":
            try:
                _cap_upd_policy = _vtec_toneout_policy(self._cap_vtec_list(ev))
                if _cap_upd_policy.mode != "FULL":
                    return False
            except Exception:
                # Parsing failed; be conservative on updates.
                return False

        event = (ev.event or "").strip()
        if event and event in self._cap_full_events():
            return True

        sev = self._cap_severity_str(ev)
        if sev and sev in self._cap_full_severities():
            return True

        return False

    def _cap_should_voice(self, ev: "CapAlertEvent") -> bool:  # type: ignore[name-defined]
        if not self._cap_voice_enabled():
            return False
        if not self._cap_is_actionable(ev):
            return False
        allow_events = self._cap_voice_events()
        if allow_events and (ev.event or "").strip() not in allow_events:
            return False
        return True

    def _cap_should_update(self, ev: "CapAlertEvent") -> bool:  # type: ignore[name-defined]
        """
        True for CAP messageType=Update with CON/EXT/CAN/EXP actions on events we
        already watch-and-warn on.  These get voice-only narration (no SAME tones).
        """
        if not self._cap_full_enabled():
            return False
        if not self._cap_is_actionable(ev):
            return False
        mt = str(ev.message_type or "").strip().lower()
        if mt not in {"update", "cancel"}:
            return False
        event = (ev.event or "").strip()
        vtec = self._cap_vtec_list(ev)
        if event not in self._cap_full_events() and not self._vtec_matches_configured_toneout_code(vtec):
            return False
        tracks = self._vtec_tracks(vtec)
        update_actions = {"CON", "EXT", "CAN", "EXP"}
        vtec_actions = {act for (_t, act) in tracks} if tracks else set()
        return bool(vtec_actions & update_actions)

    async def _consume_cap(self) -> None:
        while True:
            ev = await self.cap_queue.get()

            vtec = self._cap_vtec_list(ev)
            tracks = self._vtec_tracks(vtec)
            cap_mt = str(getattr(ev, "message_type", None) or "").strip().lower()
            cap_ref_ids = _sf_cap_reference_ids(ev)

            if cap_mt == "cancel" and not tracks:
                try:
                    same_code = self._cap_event_to_same_code((ev.event or "").strip())
                    same_locs = self._filter_same_locations_to_service_area(
                        list(getattr(ev, "same_fips", None) or [])
                    )
                    self.alert_tracker.remove(self._alert_tracker_id_for_cap(ev, same_code))
                    self._remove_matching_ipaws_state(
                        code=same_code,
                        same_locs=same_locs,
                        reason=f"cap-cancel:{getattr(ev, 'alert_id', '')}",
                    )
                    self._remove_shadowed_ern_state(
                        code=same_code,
                        same_locs=same_locs,
                        reason=f"cap-cancel:{getattr(ev, 'alert_id', '')}",
                    )
                except Exception:
                    log.exception("AlertTracker: failed handling CAP cancel without VTEC id=%s", getattr(ev, "alert_id", None))
                _sf_remove_ids(cap_ref_ids + [getattr(ev, "alert_id", None)])
                log.info("CAP cancel: evicted state without airing id=%s refs=%s", getattr(ev, "alert_id", None), ",".join(cap_ref_ids[:4]))
                continue

            log.info(
                "CAP match: event=%s severity=%s urgency=%s certainty=%s status=%s msgType=%s sent=%s same=%s headline=%s id=%s vtec=%s tracks=%s",
                ev.event,
                ev.severity,
                ev.urgency,
                ev.certainty,
                ev.status,
                ev.message_type,
                ev.sent,
                ",".join(ev.same_fips[:12]) + ("..." if len(ev.same_fips) > 12 else ""),
                ev.headline,
                ev.alert_id,
                ",".join(vtec[:2]) if vtec else "",
                ",".join(t for (t, _a) in tracks[:2]) if tracks else "",
            )

            try:
                sev = str(ev.severity or "").strip().lower()
                if sev in {"severe", "extreme"}:
                    self.cap_last_severe_at = dt.datetime.now(tz=self._tz)
            except Exception:
                pass

            if self._cap_dryrun():
                continue

            if self._cap_should_full(ev):
                await self._air_cap_full(ev)
                continue

            # CON/EXT/CAN/EXP for watched/warned events → voice-only update narration
            if self._cap_should_update(ev):
                await self._air_cap_update(ev)
                continue

            if self._cap_should_voice(ev):
                await self._air_cap_voice(ev)

    async def _air_cap_full(self, ev: "CapAlertEvent") -> None:  # type: ignore[name-defined]
        now = dt.datetime.now(tz=self._tz)

        key = (str(ev.alert_id or "").strip(), str(ev.sent or "").strip())
        last = self._cap_full_last_by_key.get(key)
        if last and (now - last).total_seconds() < self._cap_full_cooldown_seconds():
            log.info("CAP full: cooldown active; skipping id=%s sent=%s event=%s", ev.alert_id, ev.sent, ev.event)
            return

        vtec = self._cap_vtec_list(ev)
        tracks = self._vtec_tracks(vtec)
        vtec_actions = {act for (_t, act) in tracks} if tracks else set()

        ev_event = (ev.event or "").strip()
        _WATCH_EVENTS = {"Tornado Watch", "Severe Thunderstorm Watch"}
        is_watch = ev_event in _WATCH_EVENTS

        # ---- Determine watch number and kind for TOA/SVA ----
        watch_number: int | None = None
        watch_kind = "tornado"
        if is_watch:
            for v in vtec:
                m = _VTEC_PARSE_RE.search(v)
                if not m:
                    continue
                phen = (m.group("phen") or "").upper()
                sig = (m.group("sig") or "").upper()
                if sig != "A":
                    continue
                if phen == "TO":
                    watch_kind = "tornado"
                elif phen == "SV":
                    watch_kind = "severe"
                else:
                    continue
                try:
                    watch_number = int(m.group("etn"))
                except Exception:
                    pass
                break

        # ---- Route to appropriate script builder ----
        if is_watch:
            if vtec_actions & {"EXA", "EXB"}:
                # Watch expansion: full announcement with SAME for added counties
                script = self._build_watch_expansion_script(ev)
            else:
                # NEW or UPG watch
                script = self._build_cap_watch_script(ev, mode="full")
            if not script.strip():
                script = self._build_cap_full_script(ev)
        else:
            script = self._build_cap_full_script(ev)

        if not script.strip():
            return

        same_code = _vtec_toneout_policy(vtec).same_code or self._cap_event_to_same_code(ev_event)
        same_locs_raw = list(ev.same_fips) if getattr(ev, "same_fips", None) else []
        same_locs = self._filter_same_locations_to_service_area(same_locs_raw)

        keys: list[str] = []

        # Track/action-level dedupe prevents CAP vs NWWS double-air for the
        # same lifecycle action while allowing later EXA/EXB updates on the
        # same VTEC track to air when policy says they are FULL-worthy.
        for track_id, act in tracks:
            keys.append(f"TRACKFULL:{track_id}:{act or 'UNK'}")

        # Also keep raw VTEC strings (fine-grain)
        for v in vtec:
            keys.append(f"VTEC:{v}")

        # Functional FULL dedupe is only safe when we do NOT have a concrete
        # VTEC track.  Otherwise, two distinct warnings for the same counties can
        # collide (for example TO.W.0004 vs TO.W.0005).
        if not tracks:
            fkey = self._dedupe_func_full_key(same_code, same_locs)
            if fkey:
                keys.append(fkey)

        fips_part = ",".join(sorted(set(str(x).strip() for x in (same_locs or []) if str(x).strip())))[:800]
        keys.append(f"CAPFULL:{(ev.event or '').strip()}:{(ev.sent or '').strip()}:{self._sha1_12((ev.alert_id or '') + '|' + fips_part)}")

        ok, hit = await self._dedupe_reserve(keys)
        if not ok:
            log.info(
                "CAP full skipped (dedupe hit=%s) id=%s sent=%s event=%s vtec=%s",
                hit,
                ev.alert_id,
                ev.sent,
                ev.event,
                ",".join(vtec[:2]) if vtec else "",
            )
            return

        try:
            dummy = SimpleNamespace(product_type=same_code, awips_id=None, wfo="CAP", raw_text="")
            out_wav = await self.audio_originator.render_alert_audio(dummy, script, same_locations=same_locs if same_locs else None)

            async with self._cycle_lock:
                try:
                    self.telnet.flush_cycle()
                except Exception:
                    pass
                event_label = (ev.event or "").strip() or "Weather alert"
                title = self._np_alert_title("cap_full", event=event_label)
                meta = self._np_meta(
                    title=title,
                    kind="alert",
                    extra={
                        "sw_alert_source": "cap",
                        "sw_alert_mode": "full",
                        "sw_event": event_label,
                        "sw_event_code": (same_code or "").strip().upper(),
                        "sw_alert_id": str(ev.alert_id or "").strip(),
                    },
                )
                self.telnet.push_alert(str(out_wav), meta=meta)

            self._cap_full_last_by_key[key] = now
            self.last_product_desc = f"CAP {ev.event}".strip()

            try:
                code_u = (same_code or "").strip().upper()
                if code_u and code_u in self.cfg.policy.toneout_product_types:
                    self.last_toneout_at = now
                    self.last_heightened_at = now
                    self.heightened_until = now + dt.timedelta(seconds=self.cfg.cycle.min_heightened_seconds)
                    self._update_mode()
            except Exception:
                pass

            self._schedule_cycle_refill("post-cap-full")

            log.info("CAP ACTION: aired FULL event=%s code=%s id=%s sent=%s vtec=%s audio=%s", ev.event, same_code, ev.alert_id, ev.sent, ",".join(vtec[:2]) if vtec else "", out_wav)
            # _CAP_FULL_DL_
            self.discord.alert_aired(
                code=same_code,
                event=(ev.event or "").strip(),
                source="CAP",
                mode="full",
                area=getattr(ev, "area_desc", "") or "",
                vtec=vtec[:2],
                expires=self._fmt_local_from_utc_iso(
                    str(getattr(ev, "expires", "") or "")
                ),
            )
            _station_feed_note_cap(ev, mode="FULL", same_locations=(same_locs if same_locs else same_locs_raw), out_wav=str(out_wav), same_code=same_code, vtec=vtec)

            # ---- Register to AlertTracker for active cycle rotation ----
            try:
                tracker_id = self._alert_tracker_id_for_cap(ev, same_code)
                expires_iso = self._alert_expires_from_cap(ev, vtec)
                _is_watch = (ev.event or "").strip() in {"Tornado Watch", "Severe Thunderstorm Watch"}
                _watch_num: int | None = None
                if _is_watch:
                    for _v in vtec:
                        _m = _VTEC_PARSE_RE.search(_v)
                        if _m and (_m.group("sig") or "").upper() == "A":
                            try:
                                _watch_num = int(_m.group("etn"))
                            except Exception:
                                pass
                            break
                alert_entry = ActiveAlert(
                    id=tracker_id,
                    source="CAP",
                    event=str(ev.event or ""),
                    code=same_code,
                    vtec=vtec,
                    headline=str(ev.headline or ""),
                    script_text=script,
                    audio_path=str(out_wav),
                    expires=expires_iso,
                    issued=str(ev.sent or dt.datetime.now(dt.timezone.utc).isoformat()),
                    same_locs=same_locs,
                    cycle_only=False,
                    watch_number=_watch_num,
                )
                self.alert_tracker.add_or_update(alert_entry)
                self.alert_tracker.mark_aired(tracker_id)
                self._remove_shadowed_ern_state(
                    code=same_code,
                    same_locs=same_locs,
                    reason=f"cap-full:{tracker_id}",
                )
                log.info("AlertTracker: registered CAP FULL id=%s event=%s expires=%s", tracker_id, ev.event, expires_iso)
            except Exception:
                log.exception("AlertTracker: failed to register CAP FULL event=%s", ev.event)
        except Exception:
            await self._dedupe_release(keys)
            raise

    async def _air_cap_voice(self, ev: "CapAlertEvent") -> None:  # type: ignore[name-defined]
        now = dt.datetime.now(tz=self._tz)

        key = (str(ev.alert_id or "").strip(), str(ev.sent or "").strip())
        last = self._cap_voice_last_by_key.get(key)
        if last and (now - last).total_seconds() < self._cap_voice_cooldown_seconds():
            log.info("CAP voice: cooldown active; skipping id=%s sent=%s event=%s", ev.alert_id, ev.sent, ev.event)
            return

        script = self._build_cap_voice_script(ev)
        if not script.strip():
            return

        vtec = self._cap_vtec_list(ev)
        tracks = self._vtec_tracks(vtec)

        same_code = _vtec_toneout_policy(vtec).same_code or self._cap_event_to_same_code((ev.event or "").strip())
        same_locs_raw = list(ev.same_fips) if getattr(ev, "same_fips", None) else []
        same_locs = self._filter_same_locations_to_service_area(same_locs_raw)

        keys: list[str] = []
        for track_id, _act in tracks:
            keys.append(f"TRACKVOICE:{track_id}")

        fips_part = ",".join(sorted(set(str(x).strip() for x in (ev.same_fips or []) if str(x).strip())))[:800]
        keys.append(f"CAPVOICE:{(ev.event or '').strip()}:{(ev.sent or '').strip()}:{self._sha1_12((ev.alert_id or '') + '|' + fips_part)}")

        ok, hit = await self._dedupe_reserve(keys)
        if not ok:
            log.info(
                "CAP voice skipped (dedupe hit=%s) id=%s sent=%s event=%s vtec=%s",
                hit,
                ev.alert_id,
                ev.sent,
                ev.event,
                ",".join(vtec[:2]) if vtec else "",
            )
            return

        try:
            out_wav = await self.audio_originator.render_voice_only_audio(script, prefix="capvoice")

            async with self._cycle_lock:
                try:
                    self.telnet.flush_cycle()
                except Exception:
                    pass
                event_label = (ev.event or "").strip() or "Weather alert"
                title = self._np_alert_title("cap_update", event=event_label)
                meta = self._np_meta(
                    title=title,
                    kind="alert",
                    extra={
                        "sw_alert_source": "cap",
                        "sw_alert_mode": "voice",
                        "sw_event": event_label,
                        "sw_event_code": (same_code or "").strip().upper(),
                        "sw_alert_id": str(ev.alert_id or "").strip(),
                    },
                )
                self.telnet.push_alert(str(out_wav), meta=meta)

            self._cap_voice_last_by_key[key] = now
            self.last_product_desc = f"CAP {ev.event}".strip()

            self._schedule_cycle_refill("post-cap-voice")

            log.info("CAP ACTION: aired voice-only event=%s id=%s sent=%s audio=%s", ev.event, ev.alert_id, ev.sent, out_wav)
            # _CAP_VOICE_DL_
            self.discord.alert_aired(
                code=same_code,
                event=(ev.event or "").strip(),
                source="CAP",
                mode="voice",
                area=getattr(ev, "area_desc", "") or "",
                vtec=vtec[:2],
            )

            # Register to AlertTracker (cycle_only → no SAME retone on cycle replay)
            try:
                vtec_v = self._cap_vtec_list(ev)
                tracker_id_v = self._alert_tracker_id_for_cap(ev, same_code)
                expires_iso_v = self._alert_expires_from_cap(ev, vtec_v)
                _ae = ActiveAlert(
                    id=tracker_id_v,
                    source="CAP",
                    event=str(ev.event or ""),
                    code=same_code,
                    vtec=vtec_v,
                    headline=str(ev.headline or ""),
                    script_text=script,
                    audio_path=str(out_wav),
                    expires=expires_iso_v,
                    issued=str(ev.sent or dt.datetime.now(dt.timezone.utc).isoformat()),
                    same_locs=same_locs,
                    cycle_only=True,
                )
                self.alert_tracker.add_or_update(_ae)
                self.alert_tracker.mark_aired(tracker_id_v)
                self._remove_shadowed_ern_state(
                    code=same_code,
                    same_locs=same_locs,
                    reason=f"cap-voice:{tracker_id_v}",
                )
            except Exception:
                log.exception("AlertTracker: failed to register CAP VOICE event=%s", ev.event)
            _station_feed_note_cap(
                ev,
                mode="VOICE",
                same_locations=(same_locs if same_locs else same_locs_raw),
                out_wav=str(out_wav),
                same_code=same_code,
                vtec=vtec,
            )
        except Exception:
            await self._dedupe_release(keys)
            raise


    async def _air_cap_update(self, ev: "CapAlertEvent") -> None:  # type: ignore[name-defined]
        """
        Voice-only narration for VTEC CON/EXT/CAN/EXP on already-aired events.
        No SAME tones.  Removes entry from AlertTracker on CAN/EXP.
        """
        now = dt.datetime.now(tz=self._tz)
        vtec = self._cap_vtec_list(ev)
        tracks = self._vtec_tracks(vtec)
        vtec_actions = {act for (_t, act) in tracks} if tracks else set()

        ev_event = (ev.event or "").strip()
        _WATCH_EVENTS = {"Tornado Watch", "Severe Thunderstorm Watch"}
        is_watch = ev_event in _WATCH_EVENTS

        # Determine watch number/kind for watches
        watch_number: int | None = None
        watch_kind = "tornado"
        if is_watch:
            for v in vtec:
                m = _VTEC_PARSE_RE.search(v)
                if not m:
                    continue
                phen = (m.group("phen") or "").upper()
                sig = (m.group("sig") or "").upper()
                if sig != "A":
                    continue
                watch_kind = "tornado" if phen == "TO" else "severe"
                try:
                    watch_number = int(m.group("etn"))
                except Exception:
                    pass
                break

        if is_watch:
            script = self._build_watch_vtec_action_script(ev, vtec_actions, tracks, watch_number, watch_kind)
        elif self._cap_prefers_statement_update_script(ev_event, vtec_actions):
            script = self._build_statement_vtec_action_script(ev, vtec_actions, tracks)
        else:
            script = self._build_warning_vtec_action_script(ev, vtec_actions, tracks)

        if not script.strip():
            log.info("CAP update: empty script, skipping event=%s vtec_actions=%s", ev_event, vtec_actions)
            return

        same_code = self._cap_event_to_same_code(ev_event)
        same_locs_raw = list(ev.same_fips) if getattr(ev, "same_fips", None) else []
        same_locs = self._filter_same_locations_to_service_area(same_locs_raw)

        key_str = f"CAPUPDATE:{(ev.alert_id or '').strip()}:{(ev.sent or '').strip()}"
        keys = [key_str]
        for track_id, act in tracks:
            keys.append(f"TRACKVOICE:{track_id}:{act or 'UNK'}")

        ok, hit = await self._dedupe_reserve(keys)
        if not ok:
            log.info("CAP update skipped (dedupe hit=%s) event=%s vtec_actions=%s", hit, ev_event, vtec_actions)
            return

        try:
            out_wav = await self.audio_originator.render_voice_only_audio(script, prefix="capupdate")
            async with self._cycle_lock:
                try:
                    self.telnet.flush_cycle()
                except Exception:
                    pass
                event_label = ev_event or "Weather alert"
                title = self._np_alert_title("cap_update", event=event_label)
                meta = self._np_meta(
                    title=title,
                    kind="alert",
                    extra={
                        "sw_alert_source": "cap",
                        "sw_alert_mode": "update",
                        "sw_event": event_label,
                        "sw_event_code": (same_code or "").strip().upper(),
                        "sw_alert_id": str(ev.alert_id or "").strip(),
                    },
                )
                self.telnet.push_alert(str(out_wav), meta=meta)

            self.last_product_desc = f"CAP {ev_event}".strip()
            self._schedule_cycle_refill("post-cap-update")

            # Update or remove from AlertTracker
            try:
                tracker_id = self._alert_tracker_id_for_cap(ev, same_code)
                if vtec_actions & {"CAN", "EXP"}:
                    removed = self.alert_tracker.remove(tracker_id)
                    if not removed:
                        # Try by VTEC track
                        track_ids = {_vtec_track_id(v) for v in vtec if _vtec_track_id(v)}
                        self.alert_tracker.remove_by_vtec_tracks(
                            track_ids,  # type: ignore[arg-type]
                            reason=f"cap-update:{','.join(sorted(vtec_actions))}",
                        )
                    self._remove_matching_ipaws_state(
                        code=same_code,
                        same_locs=same_locs,
                        reason=f"cap-update:{','.join(sorted(vtec_actions))}",
                    )
                    self._remove_shadowed_ern_state(
                        code=same_code,
                        same_locs=same_locs,
                        reason=f"cap-update:{','.join(sorted(vtec_actions))}",
                    )
                    log.info("AlertTracker: removed id=%s event=%s action=%s", tracker_id, ev_event, vtec_actions)
                else:
                    # CON/EXT/EXA: update the stored script to latest narration
                    existing = self.alert_tracker.find_by_vtec_track(tracker_id.replace("CAP:", "")) or self.alert_tracker._alerts.get(tracker_id)
                    if existing:
                        self.alert_tracker.update_script(existing.id, script)
                        self.alert_tracker.mark_aired(existing.id)
                    log.info("AlertTracker: updated id=%s event=%s action=%s", tracker_id, ev_event, vtec_actions)
            except Exception:
                log.exception("AlertTracker: failed to update/remove on CAP update event=%s", ev_event)

            log.info("CAP ACTION: aired UPDATE event=%s code=%s id=%s vtec_actions=%s audio=%s",
                     ev_event, same_code, ev.alert_id, vtec_actions, out_wav)
            # _CAP_UPDATE_DL_
            if vtec_actions & {"CAN", "EXP"}:
                self.discord.alert_expired(
                    code=same_code,
                    event=ev_event,
                    vtec_action=next(iter(vtec_actions & {"CAN", "EXP"})),
                    source="CAP",
                    area=getattr(ev, "area_desc", "") or "",
                    vtec=vtec[:2],
                )
            else:
                self.discord.alert_updated(
                    code=same_code,
                    event=ev_event,
                    vtec_action=next(iter(vtec_actions), "CON"),
                    source="CAP",
                    area=getattr(ev, "area_desc", "") or "",
                    vtec=vtec[:2],
                )
            _station_feed_note_cap(ev, mode="VOICE", same_locations=(same_locs if same_locs else same_locs_raw),
                                   out_wav=str(out_wav), same_code=same_code, vtec=vtec)
        except Exception:
            await self._dedupe_release(keys)
            raise

    def _clean_cap_text(self, s: str, *, limit: int = 900) -> str:
        return self.cap_text._clean_cap_text(s, limit=limit)


    def _build_cap_watch_script(self, ev: "CapAlertEvent", *, mode: str = "full") -> str:  # type: ignore[name-defined]
        return self.cap_text._build_cap_watch_script(ev, mode=mode)


    # ------------------------------------------------------------------ #
    #  VTEC-action script builders (NWR-style update/cancel narration)    #
    # ------------------------------------------------------------------ #

    def _parse_cap_area_by_state(self, area_desc: str) -> tuple[dict[str, list[str]], list[str], list[str]]:
        return self.cap_text._parse_cap_area_by_state(area_desc)

    def _join_oxford(self, items: list[str]) -> str:
        return self.cap_text._join_oxford(items)

    def _fmt_local_from_utc_iso(self, iso_str: str) -> str:
        return self.cap_text._fmt_local_from_utc_iso(iso)

    def _alert_tracker_id_for_cap(self, ev: "CapAlertEvent", same_code: str) -> str:  # type: ignore[name-defined]
        """
        Return a stable AlertTracker ID for a CAP event.
        Prefers the first VTEC track id so updates slot into the same entry.
        """
        vtec = self._cap_vtec_list(ev)
        for v in vtec:
            tid = _vtec_track_id(v)
            if tid:
                return f"CAP:{tid}"
        return f"CAP:{(ev.alert_id or '').strip()}"

    def _alert_expires_from_cap(self, ev: "CapAlertEvent", vtec: list[str]) -> str:  # type: ignore[name-defined]
        """Best-effort expiry ISO string from VTEC end time or CAP expires field."""
        exp_utc = self._best_expiry_from_vtec(vtec)
        if exp_utc:
            return exp_utc.isoformat()
        raw = getattr(ev, "expires", None) or getattr(ev, "ends", None)
        if raw:
            return str(raw).strip()
        # Fallback: 6 hours from now
        return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=6)).isoformat()

    def _alert_expires_from_ern(self, ev) -> str:
        """Best-effort expiry ISO for an ERN SAME relay from JJJHHMM + TTTT."""
        now_utc = dt.datetime.now(dt.timezone.utc)
        start_utc = _ern_same_jday_to_utc(getattr(ev, "jjjhhmm", None), now_utc=now_utc)
        duration_min = _ern_parse_duration_minutes(getattr(ev, "tttt", None))
        if start_utc is not None and duration_min is not None:
            end_utc = start_utc + dt.timedelta(minutes=duration_min)
            if end_utc > now_utc:
                return end_utc.isoformat()
        if duration_min is not None and duration_min > 0:
            return (now_utc + dt.timedelta(minutes=duration_min)).isoformat()
        return (now_utc + dt.timedelta(hours=1)).isoformat()

    def _remove_shadowed_ern_state(self, *, code: str | None, same_locs, reason: str) -> int:
        """Remove ERN relay copies once authoritative CAP/NWWS/IPAWS state covers them."""
        removed = 0
        try:
            removed += self.alert_tracker.remove_matching_source(
                source="ERN",
                code=code,
                same_locs=list(same_locs or []),
                reason=reason,
            )
        except Exception:
            log.exception("AlertTracker: failed removing ERN relay state reason=%s", reason)
        try:
            removed += _sf_remove_ern_relays_matching(code, same_locs)
        except Exception:
            log.exception("Station feed: failed removing ERN relay state reason=%s", reason)
        return removed

    def _remove_matching_ipaws_state(self, *, code: str | None, same_locs, reason: str) -> int:
        """Remove IPAWS active-cycle entries when a CAP/IPAWS cancel kills them."""
        try:
            return self.alert_tracker.remove_matching_source(
                source="IPAWS",
                code=code,
                same_locs=list(same_locs or []),
                reason=reason,
            )
        except Exception:
            log.exception("AlertTracker: failed removing IPAWS state reason=%s", reason)
            return 0

    def _cap_prefers_statement_update_script(self, event: str, vtec_actions: set[str]) -> bool:
        return self.cap_text._cap_prefers_statement_update_script(event, vtec_actions)

    def _cap_expiry_summary_line(self, text: str) -> str:
        return self.cap_text._cap_expiry_summary_line(text)

    def _build_statement_vtec_action_script(
        self,
        ev: "CapAlertEvent",  # type: ignore[name-defined]
        vtec_actions: set[str],
        tracks: list[tuple[str, str]],
    ) -> str:
        return self.cap_text._build_statement_vtec_action_script(
            ev,
            vtec_actions=vtec_actions,
            tracks=tracks,
        )

    def _build_warning_vtec_action_script(
        self,
        ev: "CapAlertEvent",  # type: ignore[name-defined]
        vtec_actions: set[str],
        tracks: list[tuple[str, str]],
    ) -> str:
        return self.cap_text._build_warning_vtec_action_script(
            ev,
            vtec_actions=vtec_actions,
            tracks=tracks,
        )

    def _build_watch_vtec_action_script(
        self,
        ev: "CapAlertEvent",  # type: ignore[name-defined]
        vtec_actions: set[str],
        tracks: list[tuple[str, str]],
        watch_number: int | None,
        kind: str,
    ) -> str:
        return self.cap_text._build_watch_vtec_action_script(
            ev,
            vtec_actions=vtec_actions,
            tracks=tracks,
            watch_number=watch_number,
            kind=kind,
        )

    def _build_watch_expansion_script(self, ev: "CapAlertEvent") -> str:  # type: ignore[name-defined]
        return self.cap_text._build_watch_expansion_script(ev)

    def _build_cap_full_script(self, ev: "CapAlertEvent") -> str:  # type: ignore[name-defined]
        return self.cap_text._build_cap_full_script(ev)

    def _build_cap_voice_script(self, ev: "CapAlertEvent") -> str:  # type: ignore[name-defined]
        return self.cap_text._build_cap_voice_script(ev)

    async def _handle_toneout(self, parsed: ParsedProduct) -> None:
        log.info("NWWS toneout candidate: type=%s awips=%s wfo=%s", parsed.product_type, parsed.awips_id or "", parsed.wfo)

        official_text, pid = await self._resolve_nwws_official_text(parsed)

        # --- NEW: derive SAME targeting from UGC zones (NWWS-only) ---
        zones, in_area_same, src, mapped_ok, ugc_expires_utc = await self._nwws_same_targets_from_texts(parsed.raw_text or "", official_text or "")

        if zones:
            log.info(
                "NWWS targeting: ugc_zones=%d src=%s mapped_ok=%s in_area_same=%d",
                len(zones),
                src,
                mapped_ok,
                len(in_area_same),
            )

        vtec_preview = self._extract_vtec(official_text)
        _pre_policy = _vtec_toneout_policy(vtec_preview)
        _pre_is_wcn_watch = (
            (parsed.product_type or "").strip().upper() == "WCN"
            and (_pre_policy.same_code or "").strip().upper() in {"SVA", "TOA"}
        )
        if _pre_is_wcn_watch and not in_area_same:
            area_same = await self._nwws_wcn_watch_same_targets_from_area_desc(official_text or "")
            if area_same:
                in_area_same = area_same
                mapped_ok = True
                src = "wcn-area"
                log.info(
                    "NWWS WCN watch targeting recovered from county block: in_area_same=%d",
                    len(in_area_same),
                )

        # If we successfully mapped zones -> SOME SAME codes, and none are in-area => out-of-area, skip entirely.
        if zones and mapped_ok and not in_area_same:
            preview = ",".join(zones[:20]) + ("..." if len(zones) > 20 else "")
            log.info(
                "NWWS out-of-area: type=%s wfo=%s ugc_zones=%s (no intersection with service area); skipping",
                parsed.product_type,
                parsed.wfo,
                preview,
            )
            return

        vtec = vtec_preview
        tracks = self._vtec_tracks(vtec)
        exp_utc = self._best_expiry_from_vtec(vtec)

        # VTEC toneout policy — vtec.py is authoritative for FULL vs VOICE.
        # This replaces the inline action-only check that ignored significance
        # (e.g. CF.Y.NEW was incorrectly treated as FULL before this fix).
        _nw_policy = _vtec_toneout_policy(vtec)
        vtec_actions = {act for (_t, act) in tracks} if tracks else set()
        should_full = (_nw_policy.mode == "FULL")
        log.debug("NWWS vtec policy: %s", _nw_policy.reason)


        # Critical safety gate:
        # If we have UGC zones but could not map ANY of them to SAME, we do NOT air FULL.
        # This prevents blind tone-outs (especially marine zones) when mapping fails.
        if zones and (not mapped_ok) and should_full:
            log.warning(
                "NWWS SAME targeting failed (no zone->SAME mapping). Forcing voice-only type=%s wfo=%s zones=%s",
                parsed.product_type,
                parsed.wfo,
                ",".join(zones[:12]) + ("..." if len(zones) > 12 else ""),
            )
            should_full = False

        # WCN watch county notifications can legitimately have clean watch text
        # and VTEC, but no UGC block in the NWWS carrier.  When that happens we
        # cannot build SAME headers from NWWS alone.  Do not air a SAME-less FULL
        # and reserve TRACKFULL, because the follow-up CAP alert usually carries
        # explicit SAME/FIPS codes and must remain eligible to tone out.
        _nw_is_wcn_watch_carrier = (
            (parsed.product_type or "").strip().upper() == "WCN"
            and should_full
            and (_nw_policy.same_code or "").strip().upper() in {"SVA", "TOA"}
        )
        if _nw_is_wcn_watch_carrier and not in_area_same:
            log.warning(
                "NWWS WCN watch has no SAME targets; forcing voice-only to preserve CAP full-tone path type=%s wfo=%s vtec=%s",
                parsed.product_type,
                parsed.wfo,
                ",".join(vtec[:2]) if vtec else "",
            )
            should_full = False

        keys: list[str] = []

        # Track/action-level dedupe prevents CAP+NWWS double-air for the same
        # lifecycle action while still allowing later CON -> EXP/CAN updates for
        # the same VTEC track.
        for track_id, act in tracks:
            keys.append(f"{'TRACKFULL' if should_full else 'TRACKVOICE'}:{track_id}:{act or 'UNK'}")

        # Keep VTEC strings too (helps when track parse fails on weird edge cases)
        for v in vtec:
            keys.append(f"VTEC:{v}")

        # Functional FULL dedupe is only safe when we do NOT have a concrete
        # VTEC track.  Otherwise, distinct warnings for the same SAME footprint
        # can suppress each other.
        if should_full and not tracks:
            fkey2 = self._dedupe_func_full_key(parsed.product_type, in_area_same)
            if fkey2:
                keys.append(fkey2)

        # Message-level fallback key
        keys.append(f"NWWS:{parsed.product_type}:{parsed.wfo}:{self._sha1_12(official_text[:1200])}:{'FULL' if should_full else 'VOICE'}")

        ok, hit = await self._dedupe_reserve(keys)
        if not ok:
            log.info(
                "NWWS %s skipped (dedupe hit=%s) type=%s awips=%s wfo=%s vtec=%s",
                "FULL" if should_full else "VOICE",
                hit,
                parsed.product_type,
                parsed.awips_id or "",
                parsed.wfo,
                ",".join(vtec[:2]) if vtec else "",
            )
            return

        try:
            spoken = build_spoken_alert(parsed, official_text)

            sf_issued_dt = _sf_nwws_best_issued_dt(parsed, official_text)
            sf_event_label = _sf_nwws_event_label(parsed.product_type, vtec_list=vtec, text=official_text)
            sf_area_text = ""
            if in_area_same:
                try:
                    sf_area_text = await self._sf_area_text_from_same_codes(list(in_area_same))
                except Exception:
                    sf_area_text = ""
            if not sf_area_text:
                sf_area_text = _sf_nwws_area_from_text(official_text)
            sf_headline = _sf_nwws_make_headline(
                sf_event_label,
                issued_dt=sf_issued_dt,
                end_dt=exp_utc,
                issuer=_sf_nwws_extract_issuer(official_text, fallback_wfo=parsed.wfo),
            )

            try:
                rendered_script = render_nwws_product_script(
                    product_type=parsed.product_type,
                    base_script=spoken.script,
                    official_text=official_text,
                    vtec=vtec,
                    vtec_actions=vtec_actions,
                    has_tracks=bool(tracks),
                    should_full=should_full,
                    event_text=sf_event_label,
                    area_text=sf_area_text,
                    headline=sf_headline,
                    local_tz=self._tz,
                )
                spoken.script = rendered_script.script
                if rendered_script.changed:
                    log.info(
                        "NWWS script normalized by %s (act=%s type=%s awips=%s wfo=%s)",
                        rendered_script.renderer,
                        ','.join(sorted(vtec_actions))[:64],
                        parsed.product_type,
                        parsed.awips_id or '',
                        parsed.wfo,
                    )
                for note in rendered_script.notes:
                    if note.startswith("warning:"):
                        log.warning("NWWS script renderer note: %s", note[8:].strip())
                    else:
                        log.info("NWWS script renderer note: %s", note)
            except Exception:
                log.exception("NWWS script normalization failed; continuing with original script")

            if should_full:
                # If we have in-area SAME targets, use them. Otherwise, AIR WITHOUT SAME (no 67 fallback).
                same_for_render: list[str] = list(in_area_same) if in_area_same else []
                if zones and not mapped_ok:
                    log.warning(
                        "NWWS SAME targeting unavailable (zone map failed); airing without SAME headers type=%s wfo=%s",
                        parsed.product_type,
                        parsed.wfo,
                    )
                render_parsed = parsed
                if _nw_policy.same_code and _nw_policy.same_code != (parsed.product_type or "").strip().upper():
                    render_parsed = ParsedProduct(
                        product_type=_nw_policy.same_code,
                        wfo=parsed.wfo,
                        awips_id=parsed.awips_id,
                        vtec=parsed.vtec,
                        raw_text=parsed.raw_text,
                    )
                out_wav = await self.audio_originator.render_alert_audio(render_parsed, spoken.script, same_locations=same_for_render)
            else:
                # Voice-only always has no SAME headers by design.
                out_wav = await self.audio_originator.render_voice_only_audio(spoken.script, prefix="nwwsvoice")

            async with self._cycle_lock:
                try:
                    self.telnet.flush_cycle()
                except Exception:
                    pass
                event_label = sf_event_label
                if (not should_full) and (("EXP" in vtec_actions) or ("CAN" in vtec_actions)):
                    tkey = "nwws_end"
                elif should_full:
                    tkey = "nwws_full"
                else:
                    tkey = "nwws_update"
                title = self._np_alert_title(tkey, event=event_label)
                meta = self._np_meta(
                    title=title,
                    kind="alert",
                    extra={
                        "sw_alert_source": "nwws",
                        "sw_alert_mode": ("full" if should_full else "voice"),
                        "sw_event_code": (_nw_policy.same_code or parsed.product_type or "").strip().upper(),
                        "sw_event": event_label,
                        "sw_wfo": (parsed.wfo or "").strip(),
                        "sw_awips": (parsed.awips_id or "").strip(),
                    },
                )
                self.telnet.push_alert(str(out_wav), meta=meta)

            now = dt.datetime.now(tz=self._tz)

            # Only FULL toneouts should push heightened mode + toneout timestamp.
            if should_full:
                self.last_toneout_at = now
                self.last_heightened_at = now
                self.heightened_until = now + dt.timedelta(seconds=self.cfg.cycle.min_heightened_seconds)
                self._update_mode()

            self._nwws_acted += 1
            log.info(
                "NWWS ACTION: aired %s #%d/%d type=%s awips=%s wfo=%s vtec=%s tracks=%s audio=%s",
                "FULL" if should_full else "VOICE",
                self._nwws_acted,
                self._nwws_seen,
                parsed.product_type,
                parsed.awips_id or "",
                parsed.wfo,
                ",".join(vtec[:2]) if vtec else "",
                ",".join(t for (t, _a) in tracks[:2]) if tracks else "",
                out_wav,
            )
            # _NWWS_DL_
            _dl_vtec_acts = vtec_actions
            _dl_mode = "full" if should_full else "voice"
            if _dl_vtec_acts & {"CAN", "EXP"}:
                self.discord.alert_expired(
                    code=parsed.product_type,
                    event=sf_event_label,
                    vtec_action=next(iter(_dl_vtec_acts & {"CAN", "EXP"})),
                    source="NWWS-OI",
                    area=sf_area_text,
                    vtec=vtec[:2],
                )
            elif not should_full and _dl_vtec_acts & {"CON", "EXT", "EXA", "EXB"}:
                self.discord.alert_updated(
                    code=parsed.product_type,
                    event=sf_event_label,
                    vtec_action=next(iter(_dl_vtec_acts & {"CON", "EXT", "EXA", "EXB"})),
                    source="NWWS-OI",
                    area=sf_area_text,
                    vtec=vtec[:2],
                )
            else:
                self.discord.alert_aired(
                    code=parsed.product_type,
                    event=sf_event_label,
                    source="NWWS-OI",
                    mode=_dl_mode,
                    area=sf_area_text,
                    vtec=vtec[:2],
                    is_test=(parsed.product_type in {"RWT", "RMT"}),
                )
            _sf_mode = getattr(spoken, "mode", ("FULL" if should_full else "VOICE"))
            _station_feed_note_nwws(
                parsed,
                mode=_sf_mode,
                same_locations=list(in_area_same or []),
                out_wav=str(out_wav),
                product_id=pid,
                expires_at=exp_utc,
                vtec=vtec,
                official_text=official_text,
                issued_at=sf_issued_dt,
                event_text=sf_event_label,
                headline=sf_headline,
                area_text=sf_area_text,
            )

            self._schedule_cycle_refill("post-alert")

            # Register / update / remove from AlertTracker
            try:
                _nw_vtec = vtec
                _nw_tracks = tracks
                _nw_vtec_actions = vtec_actions
                _nw_exp_utc = self._best_expiry_from_vtec(_nw_vtec)
                _nw_expires_iso = _nw_exp_utc.isoformat() if _nw_exp_utc else (
                    dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=6)).isoformat()
                _nw_same_code = _nw_policy.same_code or _safe_event_code(parsed.product_type)
                _nw_same_locs = list(in_area_same) if in_area_same else []
                _nw_track_ids = {_vtec_track_id(v) for v in _nw_vtec if _vtec_track_id(v)}
                _nw_event_label = sf_event_label
                _nw_headline = sf_headline
                if _nw_vtec_actions & {"CAN", "EXP"} and not should_full:
                    # Partial cancel awareness: use the cancel_tracks frozenset from
                    # toneout_policy() — only tracks with a CAN/EXP action are removed.
                    # Tracks still active (CON/EXT in the same product) are left in place.
                    cancel_track_ids: frozenset[str] = _nw_policy.cancel_tracks
                    continuation_track_ids: frozenset[str] = _nw_policy.continuation_tracks

                    # Fall back to all tracks if vtec.py didn't differentiate
                    # (e.g. a product with only CAN and no CON).
                    if not cancel_track_ids and not continuation_track_ids:
                        cancel_track_ids = frozenset(_nw_track_ids)

                    if cancel_track_ids:
                        removed_n = self.alert_tracker.remove_by_vtec_tracks(
                            cancel_track_ids,
                            reason=f"nwws:{parsed.product_type}:{','.join(sorted(_nw_vtec_actions & {'CAN','EXP'}))}",
                        )
                        self._remove_matching_ipaws_state(
                            code=_nw_same_code,
                            same_locs=_nw_same_locs,
                            reason=f"nwws:{parsed.product_type}:{','.join(sorted(_nw_vtec_actions & {'CAN','EXP'}))}",
                        )
                        self._remove_shadowed_ern_state(
                            code=_nw_same_code,
                            same_locs=_nw_same_locs,
                            reason=f"nwws:{parsed.product_type}:{','.join(sorted(_nw_vtec_actions & {'CAN','EXP'}))}",
                        )
                        log.info(
                            "AlertTracker: removed %d entries for NWWS CAN/EXP type=%s awips=%s tracks=%s",
                            removed_n,
                            parsed.product_type,
                            parsed.awips_id or "",
                            ",".join(sorted(cancel_track_ids)),
                        )

                    if continuation_track_ids:
                        # Some tracks are still active — keep them in the tracker
                        # with an updated script so the cycle reflects the partial state.
                        _nw_tid_str = next(iter(continuation_track_ids), None)
                        _nw_tracker_id = f"NWWS:{_nw_tid_str}" if _nw_tid_str else (
                            f"NWWS:{parsed.product_type}:{parsed.wfo}:{(parsed.awips_id or '').strip()}")
                        _nw_issued = sf_issued_dt or dt.datetime.now(dt.timezone.utc)
                        if _nw_issued.tzinfo is None:
                            _nw_issued = _nw_issued.replace(tzinfo=dt.timezone.utc)
                        _ae_nw_con = ActiveAlert(
                            id=_nw_tracker_id,
                            source="NWWS",
                            event=_nw_event_label,
                            code=_nw_same_code,
                            vtec=_nw_vtec,
                            headline=_nw_headline,
                            script_text=spoken.script,
                            audio_path=str(out_wav),
                            expires=_nw_expires_iso,
                            issued=_nw_issued.isoformat(),
                            same_locs=_nw_same_locs,
                            cycle_only=True,
                        )
                        self.alert_tracker.add_or_update(_ae_nw_con)
                        self.alert_tracker.mark_aired(_nw_tracker_id)
                        self._remove_shadowed_ern_state(
                            code=_nw_same_code,
                            same_locs=_nw_same_locs,
                            reason=f"nwws-continuation:{_nw_tracker_id}",
                        )
                        log.info(
                            "AlertTracker: kept continuing NWWS id=%s type=%s (partial cancel) tracks=%s",
                            _nw_tracker_id,
                            parsed.product_type,
                            ",".join(sorted(continuation_track_ids)),
                        )
                else:
                    # New issuance, update, or continuation
                    _nw_tid_str = next(iter(_nw_track_ids), None)
                    _nw_tracker_id = f"NWWS:{_nw_tid_str}" if _nw_tid_str else (
                        f"NWWS:{parsed.product_type}:{parsed.wfo}:{(parsed.awips_id or '').strip()}")
                    _nw_is_cycle_only = not should_full
                    _nw_issued = sf_issued_dt or dt.datetime.now(dt.timezone.utc)
                    if _nw_issued.tzinfo is None:
                        _nw_issued = _nw_issued.replace(tzinfo=dt.timezone.utc)
                    _ae_nw = ActiveAlert(
                        id=_nw_tracker_id,
                        source="NWWS",
                        event=_nw_event_label,
                        code=_nw_same_code,
                        vtec=_nw_vtec,
                        headline=_nw_headline,
                        script_text=spoken.script,
                        audio_path=str(out_wav),
                        expires=_nw_expires_iso,
                        issued=_nw_issued.isoformat(),
                        same_locs=_nw_same_locs,
                        cycle_only=_nw_is_cycle_only,
                    )
                    self.alert_tracker.add_or_update(_ae_nw)
                    self.alert_tracker.mark_aired(_nw_tracker_id)
                    self._remove_shadowed_ern_state(
                        code=_nw_same_code,
                        same_locs=_nw_same_locs,
                        reason=f"nwws:{_nw_tracker_id}",
                    )
                    log.info("AlertTracker: registered NWWS id=%s type=%s should_full=%s expires=%s",
                             _nw_tracker_id, parsed.product_type, should_full, _nw_expires_iso)
            except Exception:
                log.exception("AlertTracker: failed to register NWWS type=%s", parsed.product_type)

        except Exception:
            await self._dedupe_release(keys)
            raise

    def _manual_full_eas_should_heighten(self) -> bool:
        return self.cfg.api.manual_full_eas_heightens


    async def _push_manual_originated_audio(
        self,
        *,
        wav_path: Path,
        headline: str,
        event_code: str,
        voice_mode: str,
        sender: str | None = None,
        actor: str | None = None,
        interrupt_policy: str = "interrupt_then_refill",
        same_locations: list[str] | None = None,
        expires_in_minutes: int | None = None,
        heightened_override: bool | None = None,
    ) -> dict[str, object]:
        policy = (interrupt_policy or "interrupt_then_refill").strip().lower()
        if policy != "interrupt_then_refill":
            raise ValueError(f"Unsupported interrupt policy: {interrupt_policy}")

        mode = (voice_mode or "voice_only").strip().lower()
        if mode not in {"voice_only", "full_eas"}:
            raise ValueError(f"Unsupported voice mode: {voice_mode}")

        same_codes = self._filter_same_locations_to_service_area(same_locations, allow_statewide_input=False)

        async with self._cycle_lock:
            try:
                self.telnet.flush_cycle()
            except Exception:
                pass

            title = (headline or "Manual message").strip() or "Manual message"
            meta = self._np_meta(
                title=title,
                kind="alert",
                extra={
                    "sw_alert_source": "api",
                    "sw_alert_mode": ("full" if mode == "full_eas" else "voice"),
                    "sw_event_code": _safe_event_code(event_code),
                    "sw_event": title,
                    "sw_sender": (sender or "").strip(),
                    "sw_actor": (actor or "").strip(),
                },
            )
            self.telnet.push_alert(str(wav_path), meta=meta)

        now = dt.datetime.now(tz=self._tz)
        self.last_product_desc = title[:200]
        if mode == "full_eas":
            self.last_toneout_at = now
        # Tristate heightened override:
        #   True  → always heighten (works for voice_only too)
        #   False → suppress even if config says to heighten
        #   None  → fall back to station config (manual_full_eas_heightens, full_eas only)
        if heightened_override is not None:
            _should_heighten = heightened_override
        else:
            _should_heighten = mode == "full_eas" and self._manual_full_eas_should_heighten()
        if _should_heighten:
            self.last_heightened_at = now
            self.heightened_until = now + dt.timedelta(seconds=self.cfg.cycle.min_heightened_seconds)
            self._update_mode()

        self._schedule_cycle_refill("post-api-origination")
        manual_area_text = ""
        if same_codes:
            try:
                manual_area_text = await self._sf_area_text_from_same_codes(same_codes)
            except Exception:
                manual_area_text = ""
        if not manual_area_text:
            manual_area_text = str(self.cfg.station.service_area_name or "Unknown area").strip() or "Unknown area"
        _station_feed_note_manual(
            event_code=event_code,
            headline=headline,
            voice_mode=mode,
            same_codes=same_codes,
            area_text=manual_area_text,
            out_wav=str(wav_path),
            sender=sender or self.cfg.station.name or "SeasonalWeather",
            expires_in_minutes=expires_in_minutes,
            actor=actor,
        )

        return {
            "ok": True,
            "headline": title,
            "event_code": _safe_event_code(event_code),
            "voice_mode": mode,
            "audio_path": str(wav_path),
            "same_codes": same_codes,
            "actor": (actor or "").strip(),
        }

    async def originate_manual_text(
        self,
        *,
        event_code: str,
        headline: str,
        script_text: str,
        voice_mode: str = "voice_only",
        same_locations: list[str] | None = None,
        sender: str | None = None,
        actor: str | None = None,
        interrupt_policy: str = "interrupt_then_refill",
        expires_in_minutes: int | None = None,
        heightened_override: bool | None = None,
    ) -> dict[str, object]:
        code = _safe_event_code(event_code)
        mode = (voice_mode or "voice_only").strip().lower()
        if mode == "full_eas":
            filtered_same = self._filter_same_locations_to_service_area(same_locations, allow_statewide_input=False)
            dummy = SimpleNamespace(product_type=code, awips_id=None, wfo="LOCAL", raw_text="")
            wav_path = await self.audio_originator.render_alert_audio(dummy, script_text, same_locations=filtered_same)
        else:
            filtered_same = []
            wav_path = await self.audio_originator.render_voice_only_audio(script_text, prefix="api_text")

        result = await self._push_manual_originated_audio(
            wav_path=wav_path,
            headline=headline,
            event_code=code,
            voice_mode=mode,
            sender=sender,
            actor=actor,
            interrupt_policy=interrupt_policy,
            same_locations=filtered_same,
            expires_in_minutes=expires_in_minutes,
            heightened_override=heightened_override,
        )
        result["script_text"] = script_text
        return result

    async def originate_manual_audio(
        self,
        *,
        event_code: str,
        headline: str,
        wav_path: str | Path,
        voice_mode: str = "voice_only",
        same_locations: list[str] | None = None,
        sender: str | None = None,
        actor: str | None = None,
        interrupt_policy: str = "interrupt_then_refill",
        expires_in_minutes: int | None = None,
        heightened_override: bool | None = None,
    ) -> dict[str, object]:
        code = _safe_event_code(event_code)
        mode = (voice_mode or "voice_only").strip().lower()

        path = Path(str(wav_path))
        if not path.exists():
            raise FileNotFoundError(str(path))
        self.audio_originator.assert_station_wav_format(path)

        if mode == "full_eas":
            filtered_same = self._filter_same_locations_to_service_area(same_locations, allow_statewide_input=False)
            out_wav = await self.audio_originator.render_pre_recorded_alert_audio(
                event_code=code,
                source_wav=path,
                same_locations=filtered_same,
            )
        elif mode == "voice_only":
            filtered_same = []
            out_wav = path
        else:
            raise ValueError(f"Unsupported voice mode: {voice_mode}")

        return await self._push_manual_originated_audio(
            wav_path=out_wav,
            headline=headline,
            event_code=code,
            voice_mode=mode,
            sender=sender,
            actor=actor,
            interrupt_policy=interrupt_policy,
            same_locations=filtered_same,
            expires_in_minutes=expires_in_minutes,
            heightened_override=heightened_override,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/etc/seasonalweather/config.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    _setup_logging(cfg)
    orch = Orchestrator(cfg)
    asyncio.run(orch.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# _DISCORD_LOG_ALL_HOOKS_APPLIED_
