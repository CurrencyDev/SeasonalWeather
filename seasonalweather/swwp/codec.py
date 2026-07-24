"""Bounded duplicate-safe canonical JSON codec for SWWP/1."""

from __future__ import annotations

import json
import math
from typing import Any

from .constants import DEFAULT_LIMITS, PROTOCOL_VERSION, ProtocolErrorCategory, ProtocolLimits
from .messages import PAYLOAD_BY_TYPE, Envelope

_PROHIBITED_KEYS = {
    "authorization",
    "credential",
    "exception",
    "password",
    "secret",
    "token",
    "traceback",
}


class ProtocolCodecError(ValueError):
    def __init__(self, category: ProtocolErrorCategory, summary: str) -> None:
        self.category = category
        self.summary = summary[:256]
        super().__init__(self.summary)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolCodecError(ProtocolErrorCategory.MALFORMED_JSON, "duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ProtocolCodecError(ProtocolErrorCategory.MALFORMED_JSON, "non-finite JSON number")


def _bounded(value: Any, limits: ProtocolLimits, *, depth: int = 0) -> None:
    if depth > limits.max_depth:
        raise ProtocolCodecError(ProtocolErrorCategory.OVERSIZED, "JSON nesting limit exceeded")
    if isinstance(value, str):
        if len(value) > limits.max_string_chars:
            raise ProtocolCodecError(ProtocolErrorCategory.OVERSIZED, "JSON string limit exceeded")
        return
    if isinstance(value, bool | int) or value is None:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolCodecError(ProtocolErrorCategory.MALFORMED_JSON, "non-finite JSON number")
        return
    if isinstance(value, list):
        _bounded_sequence(value, limits, depth)
        return
    if isinstance(value, dict):
        _bounded_mapping(value, limits, depth)
        return
    raise ProtocolCodecError(ProtocolErrorCategory.INVALID_PAYLOAD, "non-JSON value")


def _bounded_sequence(value: list[Any], limits: ProtocolLimits, depth: int) -> None:
    if len(value) > limits.max_collection_items:
        raise ProtocolCodecError(ProtocolErrorCategory.OVERSIZED, "JSON collection limit exceeded")
    for item in value:
        _bounded(item, limits, depth=depth + 1)


def _bounded_mapping(value: dict[Any, Any], limits: ProtocolLimits, depth: int) -> None:
    if len(value) > limits.max_map_items:
        raise ProtocolCodecError(ProtocolErrorCategory.OVERSIZED, "JSON map limit exceeded")
    for key, item in value.items():
        if not isinstance(key, str):
            raise ProtocolCodecError(ProtocolErrorCategory.INVALID_ENVELOPE, "JSON keys must be strings")
        if key.lower() in _PROHIBITED_KEYS:
            raise ProtocolCodecError(ProtocolErrorCategory.INVALID_PAYLOAD, "prohibited data is not allowed")
        _bounded(key, limits, depth=depth + 1)
        _bounded(item, limits, depth=depth + 1)


def encode(envelope: Envelope, *, limits: ProtocolLimits = DEFAULT_LIMITS) -> bytes:
    try:
        raw = envelope.model_dump(mode="json")
    except (TypeError, ValueError) as exc:
        raise ProtocolCodecError(ProtocolErrorCategory.INVALID_PAYLOAD, "message contains a non-JSON value") from exc
    _bounded(raw, limits)
    encoded = json.dumps(
        raw,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > limits.max_message_bytes:
        raise ProtocolCodecError(ProtocolErrorCategory.OVERSIZED, "encoded message limit exceeded")
    return encoded


def decode(data: bytes, *, limits: ProtocolLimits = DEFAULT_LIMITS) -> Envelope:
    if not isinstance(data, bytes):
        raise ProtocolCodecError(ProtocolErrorCategory.INVALID_ENVELOPE, "SWWP frames must be UTF-8 JSON bytes")
    if len(data) > limits.max_message_bytes:
        raise ProtocolCodecError(ProtocolErrorCategory.OVERSIZED, "encoded message limit exceeded")
    raw = _parse_json(data)
    if not isinstance(raw, dict):
        raise ProtocolCodecError(ProtocolErrorCategory.INVALID_ENVELOPE, "SWWP message must be a JSON object")
    _bounded(raw, limits)
    return _typed_envelope(raw)


def _parse_json(data: bytes) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolCodecError(ProtocolErrorCategory.MALFORMED_JSON, "message is not valid UTF-8") from exc
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except ProtocolCodecError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProtocolCodecError(ProtocolErrorCategory.MALFORMED_JSON, "malformed JSON object") from exc


def _typed_envelope(raw: dict[str, Any]) -> Envelope:
    message_type = raw.get("message_type")
    if not isinstance(message_type, str) or message_type not in PAYLOAD_BY_TYPE:
        raise ProtocolCodecError(ProtocolErrorCategory.UNKNOWN_MESSAGE_TYPE, "unknown SWWP message type")
    if raw.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolCodecError(ProtocolErrorCategory.UNSUPPORTED_VERSION, "unsupported SWWP wire version")
    payload_raw = raw.get("payload")
    if not isinstance(payload_raw, dict):
        raise ProtocolCodecError(ProtocolErrorCategory.INVALID_PAYLOAD, "message payload must be an object")
    try:
        payload = PAYLOAD_BY_TYPE[message_type].model_validate(payload_raw)
    except (TypeError, ValueError) as exc:
        raise ProtocolCodecError(ProtocolErrorCategory.INVALID_PAYLOAD, "invalid typed message payload") from exc
    try:
        return Envelope.model_validate(raw | {"payload": payload})
    except (TypeError, ValueError) as exc:
        raise ProtocolCodecError(ProtocolErrorCategory.INVALID_ENVELOPE, "invalid SWWP envelope") from exc
