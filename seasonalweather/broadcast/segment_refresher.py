"""
broadcast/segment_refresher.py — SegmentRefresher: background segment refresh engine.

Replaces the monolithic "build everything then push everything" pattern from
``_queue_cycle_once``.  Each segment now has its own cadence:

  Segment    Interval   Trigger
  ─────────  ─────────  ──────────────────────────────────────────────────────
  id         60 s       Mode change (normal → heightened) also triggers early.
  status     3 min      Reflects active watches/warnings; kept fresh.
  hwo        60 min     HWO arrives via NWWS → call trigger_immediate("hwo").
  spc        30 min     SPC day1 data; convective products trigger early.
  zfp        60 min     RWS/AFD products trigger early.
  fcst       30 min     ZFP zone data; stable between issuances.
  cwf        2 h        CWF; marine; slow cadence.
  obs        15 min     RWR/ASOS; medium cadence, most time-sensitive content.

Alert-tracker segments (_alert_{id}) are synthesised on demand when new
entries appear and pruned when entries expire or are cancelled.

Wiring in main.py (summary)
----------------------------
  In Orchestrator.__init__:
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
    )

  In Orchestrator.run():
    tasks.append(asyncio.create_task(
        self.refresher.run(), name="segment_refresher",
    ))

  In _consume_nwws / _handle_toneout, after relevant products arrive:
    self.refresher.trigger_immediate("hwo")   # HWO received
    self.refresher.trigger_immediate("zfp")   # RWS/AFD received
    self.refresher.trigger_immediate("obs")   # RWR received
    self.refresher.trigger_immediate("spc")   # SPC product
    self.refresher.trigger_immediate("id")    # mode changed

  Add to Orchestrator:
    def _make_cycle_ctx(self) -> CycleContext:
        return CycleContext(
            mode=self.mode,
            last_heightened_ago=self._heightened_ago_str(),
            last_product_desc=self.last_product_desc,
        )
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

from ..alerts.active import AlertTracker
from ..tts.tts import TTS
from .cycle import CycleBuilder, CycleContext, CycleSegment
from .segment_store import SegmentStore

log = logging.getLogger("seasonalweather.segment_refresher")

# Default refresh intervals (seconds).  Override via refresh_intervals kwarg.
_DEFAULT_INTERVALS: Dict[str, int] = {
    "id":         60,
    "status":     180,
    "hwo":        3600,
    "spc":        1800,
    "zfp":        3600,
    "fcst":       1800,
    "cwf":        7200,
    "marine_obs": 900,   # same cadence as land obs — RWR updates hourly
    "obs":        900,
}

# How often the refresher polls for stale segments (regardless of events).
_TICK_S: float = 30.0

# Reuse a recent full build_segments() result to amortise API calls when
# multiple segments are stale at the same time (e.g. cold start).
_BUILD_CACHE_TTL_S: float = 20.0

# All segment keys managed by the refresher (excluding live "time" and
# dynamic "_alert_*" keys which have their own paths).
_ALL_CONTENT_KEYS: List[str] = [
    "id", "status", "hwo", "spc", "zfp", "fcst", "cwf", "obs", "marine_obs",
]

_SEGMENT_TITLES: Dict[str, str] = {
    "id":         "Station identification.",
    "status":     "Overall station status and alerts.",
    "hwo":        "Hazardous weather outlook for the service area.",
    "spc":        "Severe weather outlook for the service area.",
    "zfp":        "Weather synopsis for the area.",
    "fcst":       "The forecast for the service area.",
    "cwf":        "Coastal and marine weather forecast.",
    "marine_obs": "Marine observations for the service area.",
    "obs":        "Current conditions in our area.",
}


class SegmentRefresher:
    """
    Background engine that keeps SegmentStore content fresh.

    On startup it performs a full cold-start population of all segments.
    Afterwards it runs a tick loop that re-synthesises any segment that
    has passed its refresh interval.  External callers can request an
    immediate out-of-band refresh via ``trigger_immediate(*keys)``.

    Internally, ``build_segments()`` results are cached for a short
    window (*_BUILD_CACHE_TTL_S*) so that when several segments are
    stale simultaneously the NWS API is only hit once.
    """

    def __init__(
        self,
        *,
        store: SegmentStore,
        cycle_builder: CycleBuilder,
        tts: TTS,
        alert_tracker: AlertTracker,
        ctx_fn: Callable[[], CycleContext],
        station_name: str,
        service_area_name: str,
        disclaimer: str,
        tz: ZoneInfo,
        sample_rate: int,
        seg_gap_s: float = 0.45,
        refresh_intervals: Optional[Dict[str, int]] = None,
        tick_s: float = _TICK_S,
    ) -> None:
        self._store = store
        self._builder = cycle_builder
        self._tts = tts
        self._alert_tracker = alert_tracker
        self._ctx_fn = ctx_fn
        self._station_name = station_name
        self._service_area_name = service_area_name
        self._disclaimer = disclaimer
        self._tz = tz
        self._sample_rate = sample_rate
        self._seg_gap_s = seg_gap_s
        self._intervals: Dict[str, int] = dict(_DEFAULT_INTERVALS)
        if refresh_intervals:
            self._intervals.update(refresh_intervals)
        self._tick_s = tick_s

        # Immediate-refresh request queue
        self._pending: Set[str] = set()
        self._wake_event: asyncio.Event = asyncio.Event()

        # build_segments() result cache (amortises API calls on multi-stale ticks)
        self._seg_cache: Optional[List[CycleSegment]] = None
        self._seg_cache_ts: float = 0.0
        self._seg_cache_mode: str = ""

        # Alert segment tracking
        self._known_alert_ids: Set[str] = set()

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def trigger_immediate(self, *keys: str) -> None:
        """
        Request an out-of-band refresh for one or more segment keys.
        Safe to call from any async context.
        """
        for k in keys:
            self._pending.add(k)
        self._wake_event.set()

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        log.info("SegmentRefresher: starting (tick=%.0fs)", self._tick_s)

        # Cold start: populate every segment before the conductor begins
        await self._populate_all()

        while True:
            self._wake_event.clear()

            # Process immediately-triggered keys first
            if self._pending:
                pending = set(self._pending)
                self._pending.clear()
                for key in pending:
                    await self._refresh_one(key)

            # Regular stale-check pass
            for key in _ALL_CONTENT_KEYS:
                if self._store.is_stale(key):
                    await self._refresh_one(key)

            # Sync alert-tracker voice segments
            await self._sync_alert_segments()

            # Sleep until next tick or woken by trigger_immediate
            try:
                await asyncio.wait_for(
                    self._wake_event.wait(),
                    timeout=self._tick_s,
                )
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------------
    #  Cold-start population
    # ------------------------------------------------------------------

    async def _populate_all(self) -> None:
        """Perform initial synthesis of all content segments."""
        log.info("SegmentRefresher: cold-start population beginning")
        for key in _ALL_CONTENT_KEYS:
            try:
                await self._refresh_one(key)
            except Exception:
                log.exception("SegmentRefresher: cold-start refresh failed key=%s", key)
        log.info("SegmentRefresher: cold-start population complete")

    # ------------------------------------------------------------------
    #  Segment refresh dispatch
    # ------------------------------------------------------------------

    async def _refresh_one(self, key: str) -> None:
        """Fetch fresh text and re-synthesise audio for *key*."""
        log.debug("SegmentRefresher: refreshing key=%s", key)
        try:
            if key == "id":
                await self._refresh_id()
            elif key == "status":
                await self._refresh_via_build("status")
            elif key == "hwo":
                await self._refresh_via_build("hwo")
            elif key == "spc":
                await self._refresh_via_build("spc")
            elif key == "zfp":
                await self._refresh_via_build("zfp")
            elif key == "fcst":
                await self._refresh_via_build("fcst")
            elif key == "cwf":
                await self._refresh_via_build("cwf")
            elif key == "marine_obs":
                await self._refresh_via_build("marine_obs")
            elif key == "obs":
                await self._refresh_via_build("obs")
            else:
                log.warning("SegmentRefresher: unrecognised key=%s, skipping", key)
        except Exception:
            log.exception("SegmentRefresher: refresh failed key=%s", key)

    # ------------------------------------------------------------------
    #  Per-segment builders
    # ------------------------------------------------------------------

    async def _refresh_id(self) -> None:
        """
        Rebuild the station ID segment.  The time sentence is intentionally
        omitted — the conductor synthesises the "time" segment live at push
        time so the spoken time is always accurate.
        """
        ctx = self._ctx_fn()
        if ctx.mode == "heightened":
            text = (
                f"This is the SeasonalNet I P Weather Radio Station, {self._station_name}, "
                f"with station programming and streaming facilities originating from SeasonalNet, "
                f"providing weather information for {self._service_area_name}. "
                f"Due to severe weather affecting the service area, normal broadcasts have been "
                f"curtailed to bring you the latest severe weather information. "
                f"{self._disclaimer}"
            )
        else:
            text = (
                f"This is the SeasonalNet I P Weather Radio Station, {self._station_name}, "
                f"with station programming and streaming facilities originating from SeasonalNet, "
                f"providing weather information for {self._service_area_name}. "
                f"{self._disclaimer}"
            )

        await self._synth(
            key="id",
            text=text,
            title=_SEGMENT_TITLES["id"],
            interval=self._intervals["id"],
        )

    async def _refresh_via_build(self, key: str) -> None:
        """
        Refresh a segment whose text comes from CycleBuilder.build_segments().
        Uses a short-lived result cache to amortise API calls when several
        segments are stale in the same tick.
        """
        ctx = self._ctx_fn()
        segs = await self._get_segments(ctx)

        # Resolve the segment key — "hwo-unavailable" collapses to a placeholder
        seg = next((s for s in segs if s.key == key), None)

        if seg is None or not seg.text.strip():
            await self._store.mark_placeholder(
                key,
                _SEGMENT_TITLES.get(key, key),
                self._intervals.get(key, 0),
            )
            log.debug("SegmentRefresher: key=%s not available — marked placeholder", key)
            return

        await self._synth(
            key=key,
            text=seg.text,
            title=_SEGMENT_TITLES.get(key, seg.title or key),
            interval=self._intervals.get(key, 0),
        )

    # ------------------------------------------------------------------
    #  build_segments() cache
    # ------------------------------------------------------------------

    async def _get_segments(self, ctx: CycleContext) -> List[CycleSegment]:
        """
        Return a recent CycleBuilder.build_segments() result, re-fetching
        only when the cache has expired or the mode has changed.
        """
        now = time.time()
        if (
            self._seg_cache is not None
            and (now - self._seg_cache_ts) < _BUILD_CACHE_TTL_S
            and self._seg_cache_mode == ctx.mode
        ):
            return self._seg_cache

        segs = await self._builder.build_segments(
            station_name=self._station_name,
            service_area_name=self._service_area_name,
            disclaimer=self._disclaimer,
            ctx=ctx,
        )
        self._seg_cache = segs
        self._seg_cache_ts = now
        self._seg_cache_mode = ctx.mode
        return segs

    # ------------------------------------------------------------------
    #  Alert-tracker segment sync
    # ------------------------------------------------------------------

    async def _sync_alert_segments(self) -> None:
        """
        Ensure every active AlertTracker voice entry has a synthesised store
        entry (``_alert_{id}``), and mark departed entries as placeholders so
        the conductor skips them on the next rotation.
        """
        try:
            active = self._alert_tracker.get_cycle_alerts()
            active_ids: Set[str] = {ae.id for ae in active}

            # Synthesise newly-appeared entries
            for ae in active:
                store_key = f"_alert_{ae.id}"
                if not self._store.is_ready(store_key):
                    if ae.script_text.strip():
                        log.info(
                            "SegmentRefresher: synthesising alert segment id=%s event=%s",
                            ae.id, ae.event,
                        )
                        await self._synth(
                            key=store_key,
                            text=ae.script_text,
                            title=f"{ae.event}." if ae.event else "Active alert.",
                            interval=0,  # on-demand only; tracker owns expiry
                        )
                    else:
                        await self._store.mark_placeholder(
                            store_key,
                            ae.event or "Active alert.",
                            refresh_interval_s=0,
                        )

            # Update text if it changed (e.g. CON/EXT updated the script)
            for ae in active:
                store_key = f"_alert_{ae.id}"
                existing = self._store.get(store_key)
                if (
                    existing
                    and not existing.is_placeholder
                    and ae.script_text.strip()
                    and existing.text != ae.script_text
                ):
                    log.info(
                        "SegmentRefresher: alert script changed, re-synthesising id=%s",
                        ae.id,
                    )
                    await self._synth(
                        key=store_key,
                        text=ae.script_text,
                        title=f"{ae.event}." if ae.event else "Active alert.",
                        interval=0,
                    )

            # Mark departed entries as placeholders
            departed = self._known_alert_ids - active_ids
            for alert_id in departed:
                store_key = f"_alert_{alert_id}"
                existing = self._store.get(store_key)
                if existing and not existing.is_placeholder:
                    await self._store.mark_placeholder(
                        store_key,
                        existing.title or "Expired alert.",
                        refresh_interval_s=0,
                    )
                    log.info(
                        "SegmentRefresher: alert segment expired/cancelled id=%s",
                        alert_id,
                    )

            self._known_alert_ids = active_ids

        except Exception:
            log.exception("SegmentRefresher: alert segment sync failed")

    # ------------------------------------------------------------------
    #  Audio synthesis helper
    # ------------------------------------------------------------------

    async def _synth(
        self,
        key: str,
        text: str,
        title: str,
        interval: int,
    ) -> None:
        """
        Synthesise *text* for *key* and update the store.

        Delegates to ``SegmentStore.synth_and_update`` which runs the
        blocking TTS call in a thread executor and then atomically replaces
        the stable WAV file.
        """
        dur = await self._store.synth_and_update(
            self._tts,
            key=key,
            title=title,
            text=text,
            refresh_interval_s=interval,
            sample_rate=self._sample_rate,
            seg_gap_s=self._seg_gap_s,
        )
        log.info(
            "SegmentRefresher: synthesised key=%s dur=%.1fs title=%r",
            key, dur, title,
        )
