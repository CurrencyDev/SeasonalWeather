"""
broadcast/conductor.py — CycleConductor: continuous broadcast cycle driver.

Replaces ``_cycle_loop`` + ``_queue_cycle_once`` in main.py.

Design
------
The conductor maintains a soft real-time estimate of how much audio is
buffered ahead in Liquidsoap's cycle queue.  Every *tick* it pushes the
next segment(s) until the estimated buffer depth exceeds *lookahead_s*.

Key differences from the old architecture
------------------------------------------
- No "outro" segment.  The cycle wraps from obs back to id continuously,
  just like a real NWR BMH.
- No repeat multiplier.  Segments are pushed one at a time as the buffer
  drains, so the queue depth stays bounded and stays fresh.
- The "time" segment is synthesised at push time (never cached) so the
  spoken time is always within a few seconds of accurate.
- After an alert flushes the cycle queue, call ``notify_flush()``; the
  conductor sees estimated_remaining_s ≈ 0 and immediately refills.
- Alert-tracker voice segments are injected right after "time" each
  rotation, replicating NWR rebroadcast behaviour without re-toning.

Wiring in main.py (summary)
----------------------------
  In Orchestrator.__init__:
    self.conductor = CycleConductor(store=..., telnet=..., ...)

  In Orchestrator.run():
    tasks.append(asyncio.create_task(self.conductor.run(), name="conductor"))

  Replace every ``self._schedule_cycle_refill(reason)`` call with:
    self.conductor.notify_flush()

  Remove:  _cycle_loop, _queue_cycle_once, _schedule_cycle_refill,
           _cycle_refill_task, _last_cycle_seq_dur,
           _live_time_loop, _live_time_wav_path, _render_live_time_wav_once,
           _live_time_text, live_time_enabled, live_time_interval_seconds,
           the live_time task in run(), and the _TIME_SENTENCE_RE strip
           in _queue_cycle_once.

  _cycle_lock in main.py is KEPT — it still guards concurrent alert
  handlers from stepping on flush_cycle() + push_alert() calls.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from ..alerts.active import AlertTracker
from ..liquidsoap_telnet import LiquidsoapTelnet
from ..tts.audio import concat_wavs, wav_duration_seconds, write_silence_wav
from ..tts.tts import TTS
from .segment_store import SegmentStore

log = logging.getLogger("seasonalweather.conductor")

# How many seconds of audio to keep ahead in Liquidsoap's cycle queue.
_LOOKAHEAD_S: float = 75.0

# How often the conductor checks the buffer level (seconds).
_TICK_S: float = 4.0

# Silence padding each side of the live time segment.
_SEG_GAP_S: float = 0.45

# Number of time WAV buffers to rotate through (avoids overwriting a
# file that Liquidsoap may still have queued).
_TIME_BUF_COUNT: int = 2

# Maximum segments to push in a single tick (safety valve).
_MAX_PUSHES_PER_TICK: int = 30

# Fixed base content order — alert segments are injected after "time".
_BASE_CONTENT: List[str] = ["hwo", "spc", "zfp", "fcst", "cwf", "obs"]

# Metadata titles for the Now-Playing / IP-RDS display.
_NP_TITLES: Dict[str, str] = {
    "id":     "Station identification.",
    "time":   "The current time in our service area.",
    "status": "Overall station status and alerts.",
    "hwo":    "Hazardous weather outlook for the service area.",
    "spc":    "Severe weather outlook for the service area.",
    "zfp":    "Weather synopsis for the area.",
    "fcst":   "The forecast for the service area.",
    "cwf":    "Coastal and marine weather forecast.",
    "obs":    "Current conditions in our area.",
}


# ---------------------------------------------------------------------------
#  Time formatting (mirrors cycle.py — kept local to avoid a circular import)
# ---------------------------------------------------------------------------

_TZ_NAME_MAP: Dict[str, str] = {
    "EST": "Eastern Standard Time",
    "EDT": "Eastern Daylight Time",
    "CST": "Central Standard Time",
    "CDT": "Central Daylight Time",
    "MST": "Mountain Standard Time",
    "MDT": "Mountain Daylight Time",
    "PST": "Pacific Standard Time",
    "PDT": "Pacific Daylight Time",
    "AKST": "Alaska Standard Time",
    "AKDT": "Alaska Daylight Time",
    "HST": "Hawaii Standard Time",
    "AST": "Atlantic Standard Time",
    "ADT": "Atlantic Daylight Time",
    "UTC": "Coordinated Universal Time",
    "GMT": "Greenwich Mean Time",
}


def _fmt_time(now: dt.datetime) -> str:
    return now.strftime("%-I:%M %p")


def _short_tz(now: dt.datetime) -> str:
    tok = (now.tzname() or "").strip()
    return _TZ_NAME_MAP.get(tok.upper(), tok or "local")


# ---------------------------------------------------------------------------
#  CycleConductor
# ---------------------------------------------------------------------------

class CycleConductor:
    """
    Continuously feeds Liquidsoap's cycle queue one segment at a time.

    The conductor tracks an estimated buffer depth and pushes the next
    segment whenever depth falls below *lookahead_s*.  This produces a
    continuous, ever-rolling broadcast with no outgoing silence gaps and
    no need for the old repeat-count arithmetic.
    """

    def __init__(
        self,
        *,
        store: SegmentStore,
        telnet: LiquidsoapTelnet,
        tts: TTS,
        alert_tracker: AlertTracker,
        tz: ZoneInfo,
        audio_dir: Path,
        sample_rate: int,
        np_meta_fn: Callable[..., Dict[str, str]],
        seg_gap_s: float = _SEG_GAP_S,
        lookahead_s: float = _LOOKAHEAD_S,
        tick_s: float = _TICK_S,
        discord_fn: Optional[Callable[..., Any]] = None,
        active_alerts_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        """
        Parameters
        ----------
        store:
            The shared SegmentStore (populated/maintained by SegmentRefresher).
        telnet:
            Live Liquidsoap telnet client.
        tts:
            TTS engine (used only for the live time segment).
        alert_tracker:
            AlertTracker for injecting active-alert voice segments.
        tz:
            Station timezone (used for the live time announcement).
        audio_dir:
            Directory for time-segment WAV buffers.
        sample_rate:
            PCM sample rate that matches the rest of the audio chain.
        np_meta_fn:
            ``Orchestrator._np_meta`` callable — builds Now-Playing metadata.
        seg_gap_s:
            Silence padding each side of the live time WAV.
        lookahead_s:
            Target buffer depth to maintain in Liquidsoap (seconds).
        tick_s:
            Conductor poll interval (seconds).
        """
        self._store = store
        self._telnet = telnet
        self._tts = tts
        self._alert_tracker = alert_tracker
        self._tz = tz
        self._audio_dir = Path(audio_dir)
        self._sample_rate = sample_rate
        self._np_meta_fn = np_meta_fn
        self._discord_fn = discord_fn            # optional: fires cycle_rebuilt embed
        self._active_alerts_fn = active_alerts_fn  # optional: count for embed
        self._seg_gap_s = seg_gap_s
        self._lookahead_s = lookahead_s
        self._tick_s = tick_s

        # Buffer accounting
        self._push_start_ts: float = time.time()
        self._total_pushed_s: float = 0.0

        # Cycle position tracking
        self._position_in_rotation: int = 0
        self._cycle_order: List[str] = []   # rebuilt at each rotation start

        # Double-buffer for live time WAV (prevents overwriting while queued)
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        self._time_bufs: List[Path] = [
            self._audio_dir / f"cycle_time_{i}.wav"
            for i in range(_TIME_BUF_COUNT)
        ]
        self._time_buf_idx: int = 0

        # Flush notification event (set by notify_flush to wake the loop early)
        self._flush_event: asyncio.Event = asyncio.Event()

        # Rotation accounting (for Discord embed + heartbeat log)
        self._rotation_count: int = 0
        self._rotation_seg_count: int = 0
        self._rotation_start_ts: float = time.time()

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def notify_flush(self) -> None:
        """
        Call this immediately after ``telnet.flush_cycle()`` in any alert
        handler.  Resets buffer tracking so the conductor refills the
        (now-empty) cycle queue as fast as possible.
        """
        self._push_start_ts = time.time()
        self._total_pushed_s = 0.0
        self._flush_event.set()
        log.debug("CycleConductor: notify_flush — buffer reset, refill imminent")

    @property
    def estimated_remaining_s(self) -> float:
        """Estimated seconds of audio buffered ahead in Liquidsoap."""
        consumed = time.time() - self._push_start_ts
        return max(0.0, self._total_pushed_s - consumed)

    # ------------------------------------------------------------------
    #  Main async loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        log.info(
            "CycleConductor: starting (lookahead=%.0fs tick=%.1fs)",
            self._lookahead_s,
            self._tick_s,
        )
        while True:
            try:
                pushes = 0
                while self.estimated_remaining_s < self._lookahead_s:
                    pushed = await self._push_next_segment()
                    pushes += 1
                    if not pushed:
                        # Nothing was ready — don't busy-spin
                        await asyncio.sleep(1.0)
                        break
                    if pushes >= _MAX_PUSHES_PER_TICK:
                        break
            except Exception:
                log.exception("CycleConductor: unhandled error in push loop")

            # Sleep until next tick, or wake early on a flush notification
            self._flush_event.clear()
            try:
                await asyncio.wait_for(self._flush_event.wait(), timeout=self._tick_s)
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------------
    #  Segment push orchestration
    # ------------------------------------------------------------------

    def _rebuild_cycle_order(self) -> None:
        """
        Recalculate the segment sequence for the upcoming rotation.
        Called once at position 0 (the start of each new cycle pass).

        Order:
          id → time → [active alert tracker segments] → hwo → spc → …
        """
        now = time.time()

        # Fire rotation-complete summary on every rotation after the first
        if self._rotation_count > 0:
            elapsed = now - self._rotation_start_ts
            active_alerts = 0
            try:
                if self._active_alerts_fn:
                    active_alerts = self._active_alerts_fn()
            except Exception:
                pass

            log.info(
                "CycleConductor: rotation #%d complete — %d segments pushed, "
                "elapsed=%.0fs, buffer=%.0fs, active_alerts=%d",
                self._rotation_count,
                self._rotation_seg_count,
                elapsed,
                self.estimated_remaining_s,
                active_alerts,
            )

            # Fire Discord cycle_rebuilt embed (mirrors old _queue_cycle_once behaviour)
            try:
                if self._discord_fn:
                    self._discord_fn(
                        reason="rotation",
                        mode="continuous",
                        interval=int(elapsed),
                        seq_dur=elapsed,
                        segments=self._rotation_seg_count,
                        active_alerts=active_alerts,
                    )
            except Exception:
                log.debug("CycleConductor: discord_fn failed (non-fatal)", exc_info=True)

        self._rotation_count += 1
        self._rotation_seg_count = 0
        self._rotation_start_ts = now

        order: List[str] = ["id", "time"]

        try:
            for ae in self._alert_tracker.get_cycle_alerts():
                order.append(f"_alert_{ae.id}")
        except Exception:
            log.exception("CycleConductor: could not fetch cycle alerts")

        order.extend(_BASE_CONTENT)
        self._cycle_order = order

        alert_count = len(order) - 2 - len(_BASE_CONTENT)
        log.info(
            "CycleConductor: starting rotation #%d — %d segments (%d active alerts)",
            self._rotation_count,
            len(order),
            alert_count,
        )

    async def _push_next_segment(self) -> bool:
        """
        Advance the position by one and push that segment to Liquidsoap.
        Returns True if audio was actually pushed, False if nothing was ready.
        """
        if self._position_in_rotation == 0 or not self._cycle_order:
            self._rebuild_cycle_order()

        key = self._cycle_order[self._position_in_rotation]
        self._position_in_rotation = (
            (self._position_in_rotation + 1) % len(self._cycle_order)
        )

        try:
            if key == "time":
                dur = await self._push_live_time()
            elif key.startswith("_alert_"):
                dur = self._push_tracker_alert(key[len("_alert_"):])
            else:
                dur = self._push_cached(key)
        except Exception:
            log.exception("CycleConductor: error pushing segment key=%s", key)
            return False

        if dur > 0.0:
            self._total_pushed_s += dur
            return True

        return False

    # ------------------------------------------------------------------
    #  Push helpers
    # ------------------------------------------------------------------

    def _push_cached(self, key: str) -> float:
        """
        Push a SegmentStore-cached segment to the cycle queue.
        Returns the segment duration, or 0.0 if the segment is unavailable
        (placeholder, missing audio).  Missing segments are silently skipped
        so a temporarily unavailable product never stops the broadcast.
        """
        entry = self._store.get(key)
        if entry is None or entry.is_placeholder:
            log.debug("CycleConductor: skipping unavailable segment key=%s", key)
            return 0.0

        audio = Path(entry.audio_path)
        if not audio.exists():
            log.warning(
                "CycleConductor: audio file missing key=%s path=%s",
                key, audio,
            )
            return 0.0

        title = _NP_TITLES.get(key, entry.title or key)
        meta = self._np_meta_fn(
            title=title,
            kind="cycle",
            extra={"sw_cycle_key": key},
        )
        try:
            self._telnet.push_cycle(str(audio), meta=meta)
        except Exception:
            log.exception("CycleConductor: telnet push_cycle failed key=%s", key)
            return 0.0

        self._rotation_seg_count += 1
        log.info("CycleConductor: → %s (%.1fs)", key, entry.duration_s)
        return entry.duration_s

    def _push_tracker_alert(self, alert_id: str) -> float:
        """
        Push a pre-synthesised AlertTracker voice segment from the store.
        The SegmentRefresher is responsible for keeping these in sync.
        Returns duration or 0.0 if not ready.
        """
        store_key = f"_alert_{alert_id}"
        entry = self._store.get(store_key)

        if entry is None or entry.is_placeholder:
            log.debug(
                "CycleConductor: alert segment not ready id=%s", alert_id,
            )
            return 0.0

        audio = Path(entry.audio_path)
        if not audio.exists():
            return 0.0

        try:
            # Best-effort title from the tracker
            ae = self._alert_tracker._alerts.get(alert_id)
            np_title = f"{ae.event}." if ae else entry.title or "Active alert."
            meta = self._np_meta_fn(
                title=np_title,
                kind="cycle",
                extra={"sw_cycle_key": "alert", "sw_alert_id": alert_id},
            )
            self._telnet.push_cycle(str(audio), meta=meta)
        except Exception:
            log.exception(
                "CycleConductor: failed pushing tracker alert id=%s", alert_id,
            )
            return 0.0

        self._rotation_seg_count += 1
        log.info(
            "CycleConductor: → alert_%s (%.1fs)", alert_id, entry.duration_s,
        )
        return entry.duration_s

    async def _push_live_time(self) -> float:
        """
        Synthesise the current time *right now* and push to Liquidsoap.

        Because synthesis happens at push time (not at build time), the spoken
        time is always within a few seconds of accurate when it plays — no
        drift from cache age or queue depth.

        Double-buffering (cycle_time_0.wav / cycle_time_1.wav) ensures we
        never overwrite a file Liquidsoap has already queued but not yet played.
        The cycle is ~15-30 minutes; the two buffers are always stale before
        reuse.
        """
        now = dt.datetime.now(tz=self._tz)
        text = f"The current time is, {_fmt_time(now)}, {_short_tz(now)}."

        # Rotate to the next buffer slot
        wav = self._time_bufs[self._time_buf_idx]
        self._time_buf_idx = (self._time_buf_idx + 1) % _TIME_BUF_COUNT

        # All intermediate work uses uniquely-named temp files
        tag = uuid.uuid4().hex[:6]
        tts_tmp = self._audio_dir / f"cycle_time_{tag}_tts.tmp.wav"
        gap_tmp = self._audio_dir / f"cycle_time_{tag}_gap.tmp.wav"
        out_tmp = self._audio_dir / f"cycle_time_{tag}_out.tmp.wav"

        try:
            # TTS is blocking — run in thread executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self._tts.synth_to_wav, text, tts_tmp,
            )
            write_silence_wav(gap_tmp, self._seg_gap_s, self._sample_rate)
            concat_wavs(out_tmp, [gap_tmp, tts_tmp, gap_tmp])
            dur = wav_duration_seconds(out_tmp)
            os.replace(str(out_tmp), str(wav))

            meta = self._np_meta_fn(
                title=_NP_TITLES["time"],
                kind="cycle",
                extra={"sw_cycle_key": "time"},
            )
            self._telnet.push_cycle(str(wav), meta=meta)

            self._rotation_seg_count += 1
            log.info("CycleConductor: → time (%.1fs) %r", dur, text)
            return dur

        except Exception:
            log.exception("CycleConductor: failed to synth/push live time segment")
            return 0.0

        finally:
            for p in (tts_tmp, gap_tmp, out_tmp):
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
