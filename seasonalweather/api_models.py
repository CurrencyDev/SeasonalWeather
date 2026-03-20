from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


ALLOWED_MANUAL_EVENT_CODES = {"ADR", "DMO", "RWT", "RMT", "SPS"}
_SAME_RE = re.compile(r"^\d{6}$")
_SENDER_RE = re.compile(r"^[A-Z0-9_-]{3,16}$")
_PRINTABLE_RE = re.compile(r"^[\x20-\x7E\n\r\t]+$")


class VoiceMode(str, Enum):
    VOICE_ONLY = "voice_only"
    FULL_EAS = "full_eas"


class InterruptPolicy(str, Enum):
    INTERRUPT_THEN_REFILL = "interrupt_then_refill"
    QUEUE_AFTER_CURRENT_ALERT = "queue_after_current_alert"


class CommandStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, use_enum_values=True)


class RebuildCycleRequest(ApiModel):
    reason: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _PRINTABLE_RE.fullmatch(value):
            raise ValueError("reason contains unsupported characters")
        return value


class SetHeightenedModeRequest(ApiModel):
    minutes: int = Field(ge=1, le=240)
    reason: str = Field(min_length=3, max_length=160)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        if not _PRINTABLE_RE.fullmatch(value):
            raise ValueError("reason contains unsupported characters")
        return value


class ClearHeightenedModeRequest(ApiModel):
    reason: str | None = Field(default=None, min_length=3, max_length=160)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _PRINTABLE_RE.fullmatch(value):
            raise ValueError("reason contains unsupported characters")
        return value


class OriginateTestRequest(ApiModel):
    event_code: Literal["RWT", "RMT"]


class OriginateBaseRequest(ApiModel):
    event_code: str = Field(min_length=3, max_length=3)
    headline: str = Field(min_length=1, max_length=160)
    voice_mode: VoiceMode = Field(default=VoiceMode.VOICE_ONLY)
    same_codes: list[str] = Field(default_factory=list, max_length=31)
    sender: str | None = Field(default=None, min_length=3, max_length=16)
    expires_in_minutes: int = Field(default=30, ge=1, le=360)
    interrupt_policy: InterruptPolicy = Field(default=InterruptPolicy.INTERRUPT_THEN_REFILL)
    heightened: bool | None = Field(
        default=None,
        description=(
            "Override heightened-mode behaviour for this origination. "
            "True forces heightened mode on; False suppresses it even if the "
            "station config would normally enable it. "
            "Omit (null) to use the station default (manual_full_eas_heightens)."
        ),
    )

    @field_validator("event_code")
    @classmethod
    def _validate_event_code(cls, value: str) -> str:
        code = "".join(ch for ch in value.upper() if ch.isalnum())
        if len(code) != 3:
            raise ValueError("event_code must be exactly three alphanumeric characters")
        if code not in ALLOWED_MANUAL_EVENT_CODES:
            raise ValueError(f"event_code must be one of: {', '.join(sorted(ALLOWED_MANUAL_EVENT_CODES))}")
        return code

    @field_validator("headline")
    @classmethod
    def _validate_headline(cls, value: str) -> str:
        if not _PRINTABLE_RE.fullmatch(value):
            raise ValueError("headline contains unsupported characters")
        return value

    @field_validator("same_codes")
    @classmethod
    def _validate_same_codes(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in value:
            code = str(raw).strip()
            if not _SAME_RE.fullmatch(code):
                raise ValueError("same_codes must contain only 6-digit SAME/FIPS codes")
            if code in seen:
                continue
            seen.add(code)
            out.append(code)
        return out

    @field_validator("sender")
    @classmethod
    def _validate_sender(cls, value: str | None) -> str | None:
        if value is None:
            return None
        sender = value.strip().upper()
        if not _SENDER_RE.fullmatch(sender):
            raise ValueError("sender must match [A-Z0-9_-]{3,16}")
        return sender

    @model_validator(mode="after")
    def _validate_voice_mode_targeting(self) -> "OriginateBaseRequest":
        if self.voice_mode == VoiceMode.FULL_EAS and not self.same_codes:
            raise ValueError("same_codes must be provided for full_eas origination")
        if self.voice_mode == VoiceMode.VOICE_ONLY and self.same_codes:
            raise ValueError("same_codes are only valid for full_eas origination")
        return self


class OriginateTextRequest(OriginateBaseRequest):
    text: str = Field(min_length=1, max_length=4000)

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not _PRINTABLE_RE.fullmatch(value):
            raise ValueError("text contains unsupported characters")
        return value.strip()


class OriginateAudioRequest(OriginateBaseRequest):
    audio_asset_id: str = Field(min_length=8, max_length=64)

    @field_validator("audio_asset_id")
    @classmethod
    def _validate_asset_id(cls, value: str) -> str:
        v = value.strip()
        if not re.fullmatch(r"^[A-Za-z0-9_-]{8,64}$", v):
            raise ValueError("audio_asset_id contains unsupported characters")
        return v


class ConfigReloadRequest(ApiModel):
    reason: str | None = Field(default=None, min_length=3, max_length=160)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _PRINTABLE_RE.fullmatch(value):
            raise ValueError("reason contains unsupported characters")
        return value


class CommandAccepted(ApiModel):
    command_id: str
    command_type: str
    status: CommandStatus
    accepted_at: str
    idempotent_replay: bool = False
    request_id: str


class CommandSnapshot(ApiModel):
    command_id: str
    command_type: str
    status: CommandStatus
    accepted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    idempotency_key: str
    actor: str
    idempotent_replay_count: int = 0
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class AudioUploadAccepted(ApiModel):
    asset_id: str
    filename: str
    content_type: str
    duration_seconds: float
    sample_rate_hz: int
    target_sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    frames: int
    normalized: bool = True
    sha256: str
    uploaded_at: str
    expires_at: str


class ErrorEnvelope(ApiModel):
    error: dict[str, Any]
    request_id: str
