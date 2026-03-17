from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import uuid
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

from .api_models import InterruptPolicy, OriginateAudioRequest, OriginateTextRequest, VoiceMode
from .config import AppConfig, load_config
from .cycle import CycleBuilder
from .tts import TTS


class ControlError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int = 422, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class NotFoundError(ControlError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(code, message, status_code=404, details=details)


class ConflictError(ControlError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(code, message, status_code=409, details=details)


class DependencyUnavailableError(ControlError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(code, message, status_code=503, details=details)


class OrchestratorControl:
    def __init__(self, orch: Any, *, config_path: str) -> None:
        self.orch = orch
        self.config_path = str(config_path)
        self._assets_lock = asyncio.Lock()
        self._supported_interrupt_policies = {InterruptPolicy.INTERRUPT_THEN_REFILL.value}

    def _now_utc(self) -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc)

    def _now_local(self) -> dt.datetime:
        tz = getattr(self.orch, "_tz", dt.timezone.utc)
        return dt.datetime.now(tz=tz)

    def _work_paths(self) -> tuple[Path, Path, Path, Path]:
        return self.orch._paths()

    def _api_work_dir(self) -> Path:
        work_dir, _audio_dir, _cache_dir, _log_dir = self._work_paths()
        path = work_dir / "api"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _audio_asset_dir(self) -> Path:
        path = self._api_work_dir() / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _sha256_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _config_file_hash(self) -> str | None:
        try:
            return self._sha256_file(Path(self.config_path))
        except Exception:
            return None

    def _station_feed_path(self) -> Path:
        return Path(self.orch.cfg.station_feed.path)

    def _asset_expiry_seconds(self) -> int:
        return max(300, min(self.orch.cfg.api.audio_ttl_seconds, 7 * 86400))

    def _asset_max_size_bytes(self) -> int:
        return max(1024, self.orch.cfg.api.audio_max_bytes)

    def _asset_max_duration_seconds(self) -> float:
        return max(1.0, min(float(self.orch.cfg.api.audio_max_seconds), 3600.0))


    def _station_sample_rate(self) -> int:
        try:
            return int(getattr(self.orch.cfg.audio, "sample_rate", 16000) or 16000)
        except Exception:
            return 16000

    def _ffmpeg_bin(self) -> str:
        return self.orch.cfg.api.ffmpeg_bin or "ffmpeg"

    def _require_ffmpeg(self) -> str:
        ffmpeg = self._ffmpeg_bin()
        resolved = shutil.which(ffmpeg)
        if not resolved:
            raise DependencyUnavailableError(
                "ffmpeg_missing",
                "ffmpeg is required for uploaded-audio normalization but was not found in PATH.",
                details={"binary": ffmpeg},
            )
        return resolved

    def _probe_station_wav(self, path: Path) -> dict[str, Any]:
        try:
            with wave.open(str(path), "rb") as wf:
                channels = int(wf.getnchannels())
                sample_rate_hz = int(wf.getframerate())
                frames = int(wf.getnframes())
                sampwidth = int(wf.getsampwidth())
        except wave.Error as exc:
            raise ControlError("invalid_wav", "Normalized WAV could not be parsed.", details={"path": str(path)}) from exc

        duration_seconds = float(frames) / float(sample_rate_hz or 1)
        expected_rate = self._station_sample_rate()
        if channels != 2 or sampwidth != 2 or sample_rate_hz != expected_rate:
            raise ControlError(
                "normalized_wav_mismatch",
                "Normalized WAV does not match the station playout format.",
                details={
                    "path": str(path),
                    "channels": channels,
                    "sample_width_bytes": sampwidth,
                    "sample_rate_hz": sample_rate_hz,
                    "expected_channels": 2,
                    "expected_sample_width_bytes": 2,
                    "expected_sample_rate_hz": expected_rate,
                },
            )
        if duration_seconds <= 0.0:
            raise ControlError("invalid_wav_duration", "Uploaded WAV has zero duration.")
        if duration_seconds > self._asset_max_duration_seconds():
            raise ControlError(
                "wav_too_long",
                "Uploaded WAV exceeds the configured duration limit.",
                details={"max_seconds": self._asset_max_duration_seconds(), "duration_seconds": duration_seconds},
            )
        return {
            "duration_seconds": duration_seconds,
            "sample_rate_hz": sample_rate_hz,
            "channels": channels,
            "frames": frames,
            "sample_width_bytes": sampwidth,
        }

    def _normalize_uploaded_wav(self, *, src_path: Path, dest_path: Path) -> dict[str, Any]:
        ffmpeg = self._require_ffmpeg()
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src_path),
            "-map_metadata",
            "-1",
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(self._station_sample_rate()),
            "-c:a",
            "pcm_s16le",
            str(dest_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not dest_path.exists():
            stderr = (proc.stderr or proc.stdout or "").strip()
            raise ControlError(
                "audio_normalization_failed",
                "Uploaded WAV could not be normalized to the station playout format.",
                details={"stderr": stderr[-1200:]},
            )
        return self._probe_station_wav(dest_path)

    def _serialize_dt(self, value: dt.datetime | None) -> str | None:
        if value is None:
            return None
        return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()

    def _ensure_backend_ready(self) -> None:
        try:
            ok = bool(self.orch.telnet.ping())
        except Exception as exc:
            raise DependencyUnavailableError("liquidsoap_unreachable", "Liquidsoap telnet backend is unavailable.") from exc
        if not ok:
            raise DependencyUnavailableError("liquidsoap_unreachable", "Liquidsoap telnet backend is unavailable.")

    def _same_codes_in_service_area(self, same_codes: list[str]) -> list[str]:
        allow = getattr(self.orch, "_same_fips_allow_set", set())
        disallowed = [code for code in same_codes if code not in allow]
        if disallowed:
            raise ControlError(
                "same_code_out_of_service_area",
                "One or more SAME codes are outside the configured service area.",
                details={"same_codes": disallowed},
            )
        return same_codes

    def _validate_interrupt_policy(self, policy: str) -> None:
        if policy not in self._supported_interrupt_policies:
            raise ControlError(
                "interrupt_policy_unsupported",
                "This Liquidsoap integration currently supports only interrupt_then_refill.",
                details={"supported": sorted(self._supported_interrupt_policies), "requested": policy},
            )

    def _manual_full_eas_should_heighten(self) -> bool:
        return self.orch.cfg.api.full_eas_heightened

    async def get_health(self) -> dict[str, Any]:
        try:
            liquidsoap_ok = bool(self.orch.telnet.ping())
        except Exception:
            liquidsoap_ok = False
        return {
            "ok": liquidsoap_ok,
            "liquidsoap_telnet": {"reachable": liquidsoap_ok},
            "api": {"version": "1.0"},
        }

    async def get_status(self) -> dict[str, Any]:
        self.orch._update_mode()
        try:
            liquidsoap_ok = bool(self.orch.telnet.ping())
        except Exception:
            liquidsoap_ok = False
        return {
            "mode": getattr(self.orch, "mode", "unknown"),
            "heightened_until": self._serialize_dt(getattr(self.orch, "heightened_until", None)),
            "last_heightened_at": self._serialize_dt(getattr(self.orch, "last_heightened_at", None)),
            "last_product_desc": getattr(self.orch, "last_product_desc", None),
            "liquidsoap_telnet_reachable": liquidsoap_ok,
            "cycle_refill_pending": bool(getattr(self.orch, "_cycle_refill_task", None) and not self.orch._cycle_refill_task.done()),
            "live_time_enabled": bool(getattr(self.orch, "live_time_enabled", False)),
            "rebroadcast_enabled": bool(getattr(self.orch, "rebroadcast_enabled", False)),
            "nwws_queue_size": int(getattr(self.orch, "nwws_queue", asyncio.Queue()).qsize()),
            "cap_queue_size": int(getattr(self.orch, "cap_queue", asyncio.Queue()).qsize()),
            "ern_queue_size": int(getattr(self.orch, "ern_queue", asyncio.Queue()).qsize()),
            "config_sha256": self._config_file_hash(),
        }

    async def get_station_feed(self) -> dict[str, Any]:
        path = self._station_feed_path()
        if not path.exists():
            raise NotFoundError(
                "station_feed_missing",
                "Station feed JSON was not found.",
                details={"path": str(path)},
            )
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ControlError(
                "station_feed_invalid_json",
                "Station feed JSON exists but could not be parsed.",
                details={"path": str(path)},
            ) from exc

    async def get_config_summary(self) -> dict[str, Any]:
        cfg = self.orch.cfg
        return {
            "config_path": self.config_path,
            "config_sha256": self._config_file_hash(),
            "station": {
                "name": cfg.station.name,
                "service_area_name": cfg.station.service_area_name,
                "timezone": cfg.station.timezone,
            },
            "cycle": {
                "normal_interval_seconds": cfg.cycle.normal_interval_seconds,
                "heightened_interval_seconds": cfg.cycle.heightened_interval_seconds,
                "min_heightened_seconds": cfg.cycle.min_heightened_seconds,
                "reference_point_count": len(cfg.cycle.reference_points),
            },
            "observations": {"stations": list(cfg.observations.stations)},
            "nwws": {
                "server": cfg.nwws.server,
                "port": cfg.nwws.port,
                "allowed_wfos": list(cfg.nwws.allowed_wfos),
            },
            "policy": {
                "toneout_product_types": list(cfg.policy.toneout_product_types),
                "min_tone_gap_seconds": cfg.policy.min_tone_gap_seconds,
            },
            "tts": {
                "backend": cfg.tts.backend,
                "voice": cfg.tts.voice,
                "rate_wpm": cfg.tts.rate_wpm,
                "volume": cfg.tts.volume,
            },
            "audio": {
                "sample_rate": cfg.audio.sample_rate,
                "attention_tone_hz": cfg.audio.attention_tone_hz,
                "attention_tone_seconds": cfg.audio.attention_tone_seconds,
                "post_alert_silence_seconds": cfg.audio.post_alert_silence_seconds,
            },
            "service_area": {
                "same_fips_count": len(cfg.service_area.same_fips_all),
                "transmitter_count": len(cfg.service_area.transmitters),
            },
            "features": {
                "station_feed_enabled": self.orch.cfg.station_feed.enabled,
                "live_time_enabled": bool(getattr(self.orch, "live_time_enabled", False)),
                "rebroadcast_enabled": bool(getattr(self.orch, "rebroadcast_enabled", False)),
            },
        }

    async def rebuild_cycle(self, *, reason: str | None, actor: str) -> dict[str, Any]:
        self._ensure_backend_ready()
        reason_text = (reason or "admin-request").strip() or "admin-request"
        await self.orch._queue_cycle_once(reason=f"api-{reason_text[:48]}")
        return {
            "ok": True,
            "reason": reason_text,
            "actor": actor,
            "mode": getattr(self.orch, "mode", "unknown"),
        }

    async def set_heightened_mode(self, *, minutes: int, reason: str, actor: str) -> dict[str, Any]:
        now = self._now_local()
        self.orch.last_heightened_at = now
        self.orch.heightened_until = now + dt.timedelta(minutes=minutes)
        self.orch._update_mode()
        self.orch.last_product_desc = f"Manual heightened mode: {reason}"[:200]
        await self.orch._queue_cycle_once(reason="api-heightened")
        return {
            "ok": True,
            "mode": getattr(self.orch, "mode", "unknown"),
            "heightened_until": self._serialize_dt(self.orch.heightened_until),
            "reason": reason,
            "actor": actor,
        }

    async def clear_heightened_mode(self, *, reason: str | None, actor: str) -> dict[str, Any]:
        self.orch.heightened_until = None
        self.orch._update_mode()
        self.orch.last_product_desc = (f"Manual heightened mode cleared: {reason}" if reason else "Manual heightened mode cleared")[:200]
        await self.orch._queue_cycle_once(reason="api-clear-heightened")
        return {
            "ok": True,
            "mode": getattr(self.orch, "mode", "unknown"),
            "reason": reason,
            "actor": actor,
        }

    async def originate_test(self, *, event_code: str, actor: str) -> dict[str, Any]:
        self._ensure_backend_ready()
        allowed, why = self.orch._tests_gate()
        if not allowed:
            raise ConflictError("test_gate_blocked", "Required test origination is currently blocked.", details={"reason": why})
        await self.orch._originate_required_test(event_code)
        return {"ok": True, "event_code": event_code, "actor": actor}

    async def stage_wav_upload(self, *, filename: str, content_type: str, data: bytes, actor: str) -> dict[str, Any]:
        if not data:
            raise ControlError("empty_upload", "Uploaded audio file was empty.")
        if len(data) > self._asset_max_size_bytes():
            raise ControlError(
                "upload_too_large",
                "Uploaded audio exceeds the configured size limit.",
                status_code=413,
                details={"max_bytes": self._asset_max_size_bytes()},
            )

        filename_clean = Path(filename or "upload.wav").name or "upload.wav"
        ext = Path(filename_clean).suffix.lower()
        if ext != ".wav":
            raise ControlError("unsupported_audio_type", "Only .wav uploads are supported in v1.")
        if content_type and content_type.lower() not in {"audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"}:
            raise ControlError("unsupported_audio_type", "Only WAV uploads are supported in v1.")

        asset_id = f"aud_{uuid.uuid4().hex[:20]}"
        asset_dir = self._audio_asset_dir()
        src_path = asset_dir / f"{asset_id}.upload.wav"
        wav_path = asset_dir / f"{asset_id}.wav"
        meta_path = asset_dir / f"{asset_id}.json"
        src_path.write_bytes(data)

        try:
            probe = self._normalize_uploaded_wav(src_path=src_path, dest_path=wav_path)
        finally:
            src_path.unlink(missing_ok=True)

        sha256 = self._sha256_file(wav_path)
        uploaded_at = self._now_utc().replace(microsecond=0)
        expires_at = uploaded_at + dt.timedelta(seconds=self._asset_expiry_seconds())
        meta = {
            "asset_id": asset_id,
            "filename": filename_clean,
            "content_type": content_type or "audio/wav",
            "duration_seconds": round(float(probe["duration_seconds"]), 3),
            "sample_rate_hz": int(probe["sample_rate_hz"]),
            "target_sample_rate_hz": self._station_sample_rate(),
            "channels": int(probe["channels"]),
            "sample_width_bytes": int(probe["sample_width_bytes"]),
            "frames": int(probe["frames"]),
            "normalized": True,
            "sha256": sha256,
            "uploaded_at": uploaded_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "path": str(wav_path),
            "actor": actor,
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "asset_id": asset_id,
            "filename": filename_clean,
            "content_type": content_type or "audio/wav",
            "duration_seconds": round(float(probe["duration_seconds"]), 3),
            "sample_rate_hz": int(probe["sample_rate_hz"]),
            "target_sample_rate_hz": self._station_sample_rate(),
            "channels": int(probe["channels"]),
            "sample_width_bytes": int(probe["sample_width_bytes"]),
            "frames": int(probe["frames"]),
            "normalized": True,
            "sha256": sha256,
            "uploaded_at": uploaded_at.isoformat(),
            "expires_at": expires_at.isoformat(),
        }

    def _load_audio_asset(self, asset_id: str) -> dict[str, Any]:
        meta_path = self._audio_asset_dir() / f"{asset_id}.json"
        if not meta_path.exists():
            raise NotFoundError("audio_asset_not_found", "Audio asset was not found.", details={"audio_asset_id": asset_id})
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expires_at = dt.datetime.fromisoformat(meta["expires_at"])
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=dt.timezone.utc)
        if self._now_utc() > expires_at.astimezone(dt.timezone.utc):
            raise NotFoundError("audio_asset_expired", "Audio asset has expired.", details={"audio_asset_id": asset_id})
        wav_path = Path(meta["path"])
        if not wav_path.exists():
            raise NotFoundError("audio_asset_missing_file", "Audio asset metadata exists but the WAV file is missing.")
        return meta

    async def _push_manual_audio(
        self,
        *,
        wav_path: Path,
        headline: str,
        event_code: str,
        voice_mode: str,
        sender: str | None,
        interrupt_policy: str,
        actor: str,
    ) -> None:
        self._validate_interrupt_policy(interrupt_policy)
        self._ensure_backend_ready()

        async with self.orch._cycle_lock:
            if interrupt_policy == InterruptPolicy.INTERRUPT_THEN_REFILL.value:
                try:
                    self.orch.telnet.flush_cycle()
                except Exception:
                    pass

            meta = self.orch._np_meta(
                title=headline,
                kind="alert",
                extra={
                    "sw_alert_source": "api",
                    "sw_alert_mode": voice_mode,
                    "sw_event_code": event_code,
                    "sw_sender": sender or "",
                    "sw_actor": actor,
                },
            )
            self.orch.telnet.push_alert(str(wav_path), meta=meta)

        now = self._now_local()
        self.orch.last_product_desc = headline[:200]
        if voice_mode == VoiceMode.FULL_EAS.value:
            self.orch.last_toneout_at = now
            if self._manual_full_eas_should_heighten():
                self.orch.last_heightened_at = now
                self.orch.heightened_until = now + dt.timedelta(seconds=self.orch.cfg.cycle.min_heightened_seconds)
                self.orch._update_mode()
        self.orch._schedule_cycle_refill("post-api-origination")

    async def originate_text(self, req: OriginateTextRequest, *, actor: str) -> dict[str, Any]:
        same_codes = list(req.same_codes)
        if req.voice_mode == VoiceMode.FULL_EAS.value:
            self._same_codes_in_service_area(same_codes)

        try:
            return await self.orch.originate_manual_text(
                event_code=req.event_code,
                headline=req.headline,
                script_text=req.text,
                voice_mode=req.voice_mode,
                same_locations=same_codes,
                sender=req.sender,
                actor=actor,
                interrupt_policy=req.interrupt_policy,
                expires_in_minutes=req.expires_in_minutes,
            )
        except NotImplementedError as exc:
            raise ControlError("manual_origination_not_supported", str(exc)) from exc
        except FileNotFoundError as exc:
            raise NotFoundError("manual_audio_missing", "Manual origination audio source is missing.", details={"path": str(exc)}) from exc
        except ValueError as exc:
            raise ControlError("invalid_manual_origination", str(exc)) from exc

    async def originate_audio(self, req: OriginateAudioRequest, *, actor: str) -> dict[str, Any]:
        same_codes = list(req.same_codes)
        if req.voice_mode == VoiceMode.FULL_EAS.value:
            self._same_codes_in_service_area(same_codes)
        meta = self._load_audio_asset(req.audio_asset_id)
        source_wav = Path(meta["path"])
        _work_dir, audio_dir, _cache_dir, _logs_dir = self._work_paths()
        ts = self._now_local().strftime("%Y%m%d-%H%M%S")
        out_path = audio_dir / f"api_audio_{ts}_{req.audio_asset_id}.wav"
        shutil.copy2(source_wav, out_path)

        try:
            result = await self.orch.originate_manual_audio(
                event_code=req.event_code,
                headline=req.headline,
                wav_path=out_path,
                voice_mode=req.voice_mode,
                same_locations=same_codes,
                sender=req.sender,
                actor=actor,
                interrupt_policy=req.interrupt_policy,
                expires_in_minutes=req.expires_in_minutes,
            )
        except FileNotFoundError as exc:
            out_path.unlink(missing_ok=True)
            raise NotFoundError("manual_audio_missing", "Manual origination audio source is missing.", details={"path": str(exc)}) from exc
        except ValueError as exc:
            out_path.unlink(missing_ok=True)
            raise ControlError("invalid_manual_origination", str(exc)) from exc

        result["audio_asset_id"] = req.audio_asset_id
        result["audio_path"] = str(out_path)
        return result

    async def reload_config(self, *, actor: str, reason: str | None = None) -> dict[str, Any]:
        old_hash = self._config_file_hash()
        new_cfg = load_config(self.config_path)
        old_cfg = self.orch.cfg

        self.orch.cfg = new_cfg
        self.orch._tz = ZoneInfo(new_cfg.station.timezone)
        self.orch.local_tz = self.orch._tz
        self.orch.tts = TTS(
            backend=new_cfg.tts.backend,
            voice=new_cfg.tts.voice,
            rate_wpm=new_cfg.tts.rate_wpm,
            volume=new_cfg.tts.volume,
            sample_rate=new_cfg.audio.sample_rate,
            text_overrides=new_cfg.tts.text_overrides,
            vtp_cfg=new_cfg.tts.voicetext_paul,
        )
        self.orch.cycle_builder = CycleBuilder(
            api=self.orch.api,
            tz_name=new_cfg.station.timezone,
            obs_stations=new_cfg.observations.stations,
            reference_points=new_cfg.cycle.reference_points,
            same_fips_all=new_cfg.service_area.same_fips_all,
        )
        self.orch._same_fips_allow_set = {str(x).strip() for x in new_cfg.service_area.same_fips_all if str(x).strip()}
        self.orch._nwws_allowed_wfos = self.orch._norm_wfo_set(getattr(new_cfg.nwws, "allowed_wfos", []))
        self.orch.live_time_enabled = bool(getattr(self.orch, "live_time_enabled", False))
        self.orch.last_product_desc = (f"Config reloaded: {reason}" if reason else "Config reloaded")[:200]
        await self.orch._queue_cycle_once(reason="api-config-reload")

        caveats: list[str] = []
        if old_cfg.nwws.server != new_cfg.nwws.server or old_cfg.nwws.port != new_cfg.nwws.port:
            caveats.append("NWWS client tasks keep their existing connection settings until the process restarts.")
        if old_cfg.paths != new_cfg.paths:
            caveats.append("Changed paths are only partly hot-applied; a process restart is safer for path changes.")

        return {
            "ok": True,
            "old_config_sha256": old_hash,
            "new_config_sha256": self._config_file_hash(),
            "actor": actor,
            "reason": reason,
            "warnings": caveats,
        }
