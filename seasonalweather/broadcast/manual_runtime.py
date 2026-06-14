from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from .audio_origination import safe_event_code as _safe_event_code
from .station_feed_runtime import note_manual as _station_feed_note_manual

if TYPE_CHECKING:  # pragma: no cover
    from seasonalweather.main import Orchestrator


class ManualOriginationRuntime:
    """Runtime service for API/manual audio origination.

    The control/API layer calls through Orchestrator compatibility shims, but
    the actual manual-origination side effects live here instead of in main.py.
    """

    def __init__(self, orchestrator: "Orchestrator") -> None:
        self.orch = orchestrator

    def manual_full_eas_should_heighten(self) -> bool:
        return self.orch.cfg.api.manual_full_eas_heightens

    async def push_manual_originated_audio(
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

        same_codes = self.orch.target_resolver._filter_same_locations_to_service_area(
            same_locations,
            allow_statewide_input=False,
        )

        title = (headline or "Manual message").strip() or "Manual message"
        meta = self.orch._np_meta(
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
        await self.orch._push_interrupt_audio(wav_path, meta=meta, full=(mode == "full_eas"))

        now = dt.datetime.now(tz=self.orch._tz)
        title = (headline or "Manual message").strip() or "Manual message"
        self.orch.last_product_desc = title[:200]
        if mode == "full_eas":
            self.orch.last_toneout_at = now
        # Tristate heightened override:
        #   True  → always heighten (works for voice_only too)
        #   False → suppress even if config says to heighten
        #   None  → fall back to station config (manual_full_eas_heightens, full_eas only)
        if heightened_override is not None:
            should_heighten = heightened_override
        else:
            should_heighten = mode == "full_eas" and self.manual_full_eas_should_heighten()
        if should_heighten:
            self.orch.last_heightened_at = now
            self.orch.heightened_until = now + dt.timedelta(seconds=self.orch.cfg.cycle.min_heightened_seconds)
            self.orch._update_mode()

        self.orch._schedule_cycle_refill("post-api-origination")
        manual_area_text = ""
        if same_codes:
            try:
                manual_area_text = await self.orch.target_resolver._sf_area_text_from_same_codes(same_codes)
            except Exception:
                manual_area_text = ""
        if not manual_area_text:
            manual_area_text = (
                str(self.orch.cfg.station.service_area_name or "Unknown area").strip()
                or "Unknown area"
            )
        _station_feed_note_manual(
            event_code=event_code,
            headline=headline,
            voice_mode=mode,
            same_codes=same_codes,
            area_text=manual_area_text,
            out_wav=str(wav_path),
            sender=sender or self.orch.cfg.station.name or "SeasonalWeather",
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

    async def originate_text(
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
            filtered_same = self.orch.target_resolver._filter_same_locations_to_service_area(
                same_locations,
                allow_statewide_input=False,
            )
            dummy = SimpleNamespace(product_type=code, awips_id=None, wfo="LOCAL", raw_text="")
            wav_path = await self.orch.audio_originator.render_alert_audio(
                dummy,
                script_text,
                same_locations=filtered_same,
            )
        else:
            filtered_same = []
            wav_path = await self.orch.audio_originator.render_voice_only_audio(
                script_text,
                prefix="api_text",
            )

        result = await self.push_manual_originated_audio(
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

    async def originate_audio(
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
        self.orch.audio_originator.assert_station_wav_format(path)

        if mode == "full_eas":
            filtered_same = self.orch.target_resolver._filter_same_locations_to_service_area(
                same_locations,
                allow_statewide_input=False,
            )
            out_wav = await self.orch.audio_originator.render_pre_recorded_alert_audio(
                event_code=code,
                source_wav=path,
                same_locations=filtered_same,
            )
        elif mode == "voice_only":
            filtered_same = []
            out_wav = path
        else:
            raise ValueError(f"Unsupported voice mode: {voice_mode}")

        return await self.push_manual_originated_audio(
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
