from __future__ import annotations

import io
import logging
import sys
from types import SimpleNamespace

from seasonalweather.config import load_config
from seasonalweather.logging_config import setup_logging


def test_config_loads_runtime_log_color(monkeypatch) -> None:
    monkeypatch.setenv("ICECAST_SOURCE_PASSWORD", "test-source")
    cfg = load_config("config/config.yaml")

    assert cfg.logs.runtime.color == "never"


def test_log_color_never_avoids_ansi(monkeypatch) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    cfg = SimpleNamespace(logs=SimpleNamespace(runtime=SimpleNamespace(
        level="INFO",
        color="never",
        httpx_level="WARNING",
        httpcore_level="WARNING",
        uvicorn_access_level="WARNING",
        uvicorn_error_level="INFO",
        asyncio_level="WARNING",
        slixmpp_level="WARNING",
        slixmpp_xmlstream_level="WARNING",
        logger_levels={},
        cap_poll_summary=True,
        ipaws_poll_summary=True,
        conductor_cycle_push=True,
        conductor_alert_push=True,
        conductor_live_time_push=True,
        segment_refresher_synth=True,
        segment_refresher_alert_lifecycle=True,
    )))

    setup_logging(cfg)
    logging.getLogger("seasonalweather.test").info("plain message")

    assert "\x1b[" not in stream.getvalue()


def test_log_color_always_adds_ansi(monkeypatch) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    cfg = SimpleNamespace(logs=SimpleNamespace(runtime=SimpleNamespace(
        level="INFO",
        color="always",
        httpx_level="WARNING",
        httpcore_level="WARNING",
        uvicorn_access_level="WARNING",
        uvicorn_error_level="INFO",
        asyncio_level="WARNING",
        slixmpp_level="WARNING",
        slixmpp_xmlstream_level="WARNING",
        logger_levels={},
        cap_poll_summary=True,
        ipaws_poll_summary=True,
        conductor_cycle_push=True,
        conductor_alert_push=True,
        conductor_live_time_push=True,
        segment_refresher_synth=True,
        segment_refresher_alert_lifecycle=True,
    )))

    setup_logging(cfg)
    logging.getLogger("seasonalweather.test").warning("colored message")

    assert "\x1b[" in stream.getvalue()
