from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import AppConfig, LogsRuntimeConfig


_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_VALID_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


class _RuntimeMessageFilter(logging.Filter):
    """Filter routine steady-state log chatter based on config.yaml toggles."""

    def __init__(self, runtime_cfg: "LogsRuntimeConfig") -> None:
        super().__init__()
        self._runtime_cfg = runtime_cfg

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()

        if not self._runtime_cfg.cap_poll_summary and message.startswith("CAP poll: emitted "):
            return False
        if not self._runtime_cfg.ipaws_poll_summary and message.startswith("IPAWS poll: emitted="):
            return False

        if not self._runtime_cfg.conductor_alert_push and message.startswith("CycleConductor: → alert_"):
            return False
        if not self._runtime_cfg.conductor_live_time_push and message.startswith("CycleConductor: → time "):
            return False
        if not self._runtime_cfg.conductor_cycle_push and message.startswith("CycleConductor: → "):
            return False

        if not self._runtime_cfg.segment_refresher_synth and (
            message.startswith("SegmentRefresher: synthesising alert segment id=")
            or message.startswith("SegmentRefresher: alert script changed, re-synthesising id=")
            or message.startswith("SegmentRefresher: synthesised key=")
        ):
            return False
        if not self._runtime_cfg.segment_refresher_alert_lifecycle and (
            message.startswith("SegmentRefresher: alert segment expired/cancelled id=")
        ):
            return False

        return True


def _normalize_level(value: str | None, *, default: str) -> str:
    level = str(value or default).strip().upper()
    return level if level in _VALID_LEVELS else default


def _apply_level(logger_name: str, level_name: str) -> None:
    logging.getLogger(logger_name).setLevel(getattr(logging, level_name, logging.INFO))


def setup_logging(cfg: "AppConfig | None" = None) -> None:
    runtime = getattr(getattr(cfg, "logs", None), "runtime", None)
    root_level = _normalize_level(getattr(runtime, "level", None), default="INFO")

    logging.basicConfig(
        level=getattr(logging, root_level, logging.INFO),
        format=_DEFAULT_FORMAT,
        stream=sys.stdout,
        force=True,
    )

    if runtime is None:
        return

    root_logger = logging.getLogger()
    runtime_filter = _RuntimeMessageFilter(runtime)
    for handler in root_logger.handlers:
        handler.addFilter(runtime_filter)

    _apply_level("httpx", _normalize_level(runtime.httpx_level, default="WARNING"))
    _apply_level("httpcore", _normalize_level(runtime.httpcore_level, default="WARNING"))
    _apply_level("uvicorn.access", _normalize_level(runtime.uvicorn_access_level, default="WARNING"))
    _apply_level("uvicorn.error", _normalize_level(runtime.uvicorn_error_level, default="INFO"))
    _apply_level("asyncio", _normalize_level(runtime.asyncio_level, default="WARNING"))
    _apply_level("slixmpp", _normalize_level(runtime.slixmpp_level, default="WARNING"))
    _apply_level("slixmpp.xmlstream", _normalize_level(runtime.slixmpp_xmlstream_level, default="WARNING"))

    for logger_name, level_name in (runtime.logger_levels or {}).items():
        name = str(logger_name).strip()
        if not name:
            continue
        _apply_level(name, _normalize_level(level_name, default="INFO"))
