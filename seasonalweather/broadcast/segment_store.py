"""
broadcast/segment_store.py — Persistent cycle segment registry.

Each cacheable segment (hwo, fcst, obs, id, etc.) has:
  - A *stable* audio file path  (cycle_seg_{key}.wav)
  - Metadata (text, duration, freshness) persisted to JSON

Audio files use stable names so Liquidsoap's push queue can hold a path
reference that is atomically replaced on refresh — no timestamp-named files
accumulating in the audio directory for cycle content.

The ``render_segment_wav`` helper synthesises text to WAV with silence
padding using fully-temp intermediate files, then atomically replaces the
stable output path.  It is safe to call while Liquidsoap has the previous
version of the file queued, because Liquidsoap opens files at play time.
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..tts.audio import concat_wavs, wav_duration_seconds, write_silence_wav
from ..tts.tts import TTS

log = logging.getLogger("seasonalweather.segment_store")

_DEFAULT_SEG_GAP_S: float = 0.45


# ---------------------------------------------------------------------------
#  WAV rendering helper (module-level, importable by the refresher)
# ---------------------------------------------------------------------------

def render_segment_wav(
    tts: TTS,
    text: str,
    output_path: Path,
    *,
    sample_rate: int,
    seg_gap_s: float = _DEFAULT_SEG_GAP_S,
) -> float:
    """
    Synthesise *text* to WAV, pad both sides with silence, and atomically
    replace *output_path*.  Returns total duration in seconds.

    All intermediate work is done via uniquely-named temp files so that
    *output_path* is never partially written.  On failure the previous
    version of *output_path* (if any) is preserved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    stem = output_path.stem

    tts_tmp = output_path.parent / f"{stem}_{tag}_tts.tmp.wav"
    gap_tmp = output_path.parent / f"{stem}_{tag}_gap.tmp.wav"
    out_tmp = output_path.parent / f"{stem}_{tag}_out.tmp.wav"

    try:
        tts.synth_to_wav(text, tts_tmp)
        write_silence_wav(gap_tmp, seg_gap_s, sample_rate)
        concat_wavs(out_tmp, [gap_tmp, tts_tmp, gap_tmp])
        dur = wav_duration_seconds(out_tmp)
        os.replace(str(out_tmp), str(output_path))
        return dur
    finally:
        for p in (tts_tmp, gap_tmp, out_tmp):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
#  Data model
# ---------------------------------------------------------------------------

@dataclass
class SegmentEntry:
    """
    Metadata for one cycle segment.  Audio bytes live on disk at *audio_path*;
    this dataclass only carries index information that is persisted to JSON.
    """
    key: str
    title: str
    text: str                  # last synthesised text (for change detection)
    audio_path: str            # stable path — atomically replaced on refresh
    duration_s: float
    last_updated_ts: float     # unix epoch
    refresh_interval_s: int    # 0 = never auto-stale (live / on-demand only)
    is_placeholder: bool = False

    def is_stale(self) -> bool:
        """True when content is old enough to warrant a refresh."""
        if self.refresh_interval_s <= 0:
            return False
        return (time.time() - self.last_updated_ts) >= self.refresh_interval_s


# ---------------------------------------------------------------------------
#  Store
# ---------------------------------------------------------------------------

class SegmentStore:
    """
    In-memory registry of cycle segment metadata, backed by a JSON index.

    Thread/task safety model
    ------------------------
    *Reads* (``get``, ``is_stale``, ``is_ready``) are safe to call from any
    async task without acquiring the lock — Python dict reads are atomic at
    the CPython level and the conductor only reads, never writes.

    *Writes* (``update``, ``mark_placeholder``) take ``_lock`` and then call
    ``_persist_unlocked`` before releasing.  Only the SegmentRefresher writes,
    so there is at most one concurrent writer in practice.
    """

    INDEX_FILENAME = "segment_store.json"

    def __init__(self, work_dir: Path, audio_dir: Path) -> None:
        self._work_dir = Path(work_dir)
        self._audio_dir = Path(audio_dir)
        self._index_path = self._work_dir / self.INDEX_FILENAME
        self._entries: Dict[str, SegmentEntry] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    #  Stable path derivation
    # ------------------------------------------------------------------

    def audio_path_for(self, key: str) -> Path:
        """Return the canonical stable WAV path for *key*."""
        safe = "".join(ch for ch in key if ch.isalnum() or ch == "_")
        return self._audio_dir / f"cycle_seg_{safe}.wav"

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def load(self) -> int:
        """
        Load index from disk at startup.  Missing audio files are flagged
        as placeholders so the refresher re-synthesises them.
        Returns the number of entries restored.
        """
        if not self._index_path.exists():
            return 0
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
            loaded = 0
            for item in raw.get("entries") or []:
                try:
                    e = SegmentEntry(**item)
                    if not Path(e.audio_path).exists():
                        e.is_placeholder = True
                    self._entries[e.key] = e
                    loaded += 1
                except Exception:
                    log.warning("segment_store: skipped malformed index entry: %s", item)
            log.info(
                "segment_store: loaded %d entries from %s",
                loaded, self._index_path,
            )
            return loaded
        except Exception:
            log.exception("segment_store: failed to load index from %s", self._index_path)
            return 0

    def _persist_unlocked(self) -> None:
        """
        Write index JSON.  Must be called with ``_lock`` held (or at init
        before any async tasks run).  Uses write-then-replace for safety.
        """
        self._work_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._index_path.with_suffix(".tmp")
        payload = {"entries": [asdict(e) for e in self._entries.values()]}
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(self._index_path))

    # ------------------------------------------------------------------
    #  Read API  (no lock required)
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[SegmentEntry]:
        return self._entries.get(key)

    def all_keys(self) -> List[str]:
        return list(self._entries.keys())

    def is_stale(self, key: str) -> bool:
        """True if the entry is missing or past its refresh interval."""
        e = self._entries.get(key)
        return e is None or e.is_stale()

    def is_ready(self, key: str) -> bool:
        """True if entry exists, has audio on disk, and is not a placeholder."""
        e = self._entries.get(key)
        if e is None or e.is_placeholder:
            return False
        return Path(e.audio_path).exists()

    # ------------------------------------------------------------------
    #  Write API  (async, takes lock)
    # ------------------------------------------------------------------

    async def update(
        self,
        key: str,
        title: str,
        text: str,
        audio_path: Path,
        duration_s: float,
        refresh_interval_s: int,
        *,
        is_placeholder: bool = False,
    ) -> None:
        """Register or replace a segment entry and persist the index."""
        async with self._lock:
            self._entries[key] = SegmentEntry(
                key=key,
                title=title,
                text=text,
                audio_path=str(audio_path),
                duration_s=duration_s,
                last_updated_ts=time.time(),
                refresh_interval_s=refresh_interval_s,
                is_placeholder=is_placeholder,
            )
            self._persist_unlocked()
        log.debug(
            "segment_store: updated key=%s dur=%.1fs placeholder=%s",
            key, duration_s, is_placeholder,
        )

    async def mark_placeholder(
        self,
        key: str,
        title: str,
        refresh_interval_s: int,
    ) -> None:
        """
        Register *key* as known-unavailable.  Sets ``last_updated_ts=0`` so
        the entry is immediately stale and the refresher will retry.
        """
        async with self._lock:
            self._entries[key] = SegmentEntry(
                key=key,
                title=title,
                text="",
                audio_path=str(self.audio_path_for(key)),
                duration_s=0.0,
                last_updated_ts=0.0,   # immediately stale → refresher retries
                refresh_interval_s=refresh_interval_s,
                is_placeholder=True,
            )
            self._persist_unlocked()
        log.debug("segment_store: marked placeholder key=%s", key)

    # ------------------------------------------------------------------
    #  Async synthesis helper
    # ------------------------------------------------------------------

    async def synth_and_update(
        self,
        tts: TTS,
        key: str,
        title: str,
        text: str,
        refresh_interval_s: int,
        *,
        sample_rate: int,
        seg_gap_s: float = _DEFAULT_SEG_GAP_S,
    ) -> float:
        """
        Synthesise *text* for *key* in a thread executor (TTS is blocking),
        then atomically update the store entry.  Returns duration in seconds.

        Uses ``render_segment_wav`` so all temp I/O happens off the event loop.
        """
        wav_path = self.audio_path_for(key)
        loop = asyncio.get_event_loop()
        dur: float = await loop.run_in_executor(
            None,
            functools.partial(
                render_segment_wav,
                tts,
                text,
                wav_path,
                sample_rate=sample_rate,
                seg_gap_s=seg_gap_s,
            ),
        )
        await self.update(
            key=key,
            title=title,
            text=text,
            audio_path=wav_path,
            duration_s=dur,
            refresh_interval_s=refresh_interval_s,
        )
        return dur
