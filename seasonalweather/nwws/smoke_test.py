#!/usr/bin/env python3
from __future__ import annotations

import inspect
import logging
import os
import re
import signal
import time
from pathlib import Path
from typing import Dict, Optional

import slixmpp

# --------------------------------------------------------------------------------------
# Logging (avoid dumping internals; never print creds)
# --------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("slixmpp").setLevel(logging.WARNING)
logging.getLogger("slixmpp.xmlstream").setLevel(logging.WARNING)
log = logging.getLogger("seasonalweather.nwws_smoke")

ENV_PATH = Path("/etc/seasonalweather/seasonalweather.env")


def _strip_quotes(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == "'") or (v[0] == v[-1] == '"')):
        return v[1:-1]
    return v


def load_env_file(path: Path) -> Dict[str, str]:
    """
    Parse a simple KEY=VALUE env file.
    Supports:
      - comments (# ...)
      - blank lines
      - optional leading 'export '
      - quoted values
    """
    if not path.exists():
        raise FileNotFoundError(f"env file not found: {path}")

    data: Dict[str, str] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        k, v = line.split("=", 1)
        k = k.strip()
        v = _strip_quotes(v.strip())
        if not k:
            continue

        # Don't allow a later blank assignment to clobber a good value.
        if k in data and data[k] and not v:
            continue

        data[k] = v

    return data


def _summarize(body: str) -> str:
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    first_line = body.split("\n", 1)[0].strip()
    first_line = re.sub(r"\s+", " ", first_line)
    if len(first_line) > 120:
        first_line = first_line[:117] + "..."
    return f'len={len(body)} first_line="{first_line}"'


class NWWSSmoke(slixmpp.ClientXMPP):
    """
    A minimal NWWS-OI smoke test:
      - connect STARTTLS
      - join the NWWS MUC
      - print safe summaries of received messages
    """

    def __init__(
        self,
        jid: str,
        password: str,
        server_host: str,
        server_port: int,
        duration_s: int,
        room_jid: str,
        room_password: Optional[str],
        nick: str,
    ) -> None:
        super().__init__(jid, password)

        # Avoid attribute name "server" to dodge Slixmpp deprecated property warnings.
        self.server_host = server_host
        self.server_port = server_port
        self.duration_s = duration_s

        self.room_jid = room_jid
        self.room_password = room_password
        self.nick = nick

        self._rx_total = 0
        self._rx_groupchat = 0
        self._rx_chat = 0
        self._start_ts = 0.0

        self.register_plugin("xep_0030")  # Service Discovery
        self.register_plugin("xep_0199")  # XMPP Ping
        self.register_plugin("xep_0045")  # Multi-User Chat (MUC)

        self.add_event_handler("session_start", self._on_session_start)
        self.add_event_handler("message", self._on_message)
        self.add_event_handler("failed_auth", self._on_failed_auth)
        self.add_event_handler("disconnected", self._on_disconnected)

    def _on_failed_auth(self, event) -> None:
        log.error("NWWS-OI authentication failed")
        self.disconnect()

    def _on_disconnected(self, event) -> None:
        log.info(
            "Disconnected. Totals: total=%d groupchat=%d chat=%d",
            self._rx_total,
            self._rx_groupchat,
            self._rx_chat,
        )

    async def _on_session_start(self, event) -> None:
        self._start_ts = time.time()
        self.send_presence()

        try:
            r = self.get_roster()
            if inspect.isawaitable(r):
                await r
        except Exception:
            pass

        log.info("NWWS-OI session started")

        # Stop after duration_s no matter what
        self.schedule("stop_after", self.duration_s, self._stop, repeat=False)

        # Join NWWS room (Slixmpp versions vary; try both signatures)
        try:
            res = self.plugin["xep_0045"].join_muc(
                self.room_jid,
                self.nick,
                password=self.room_password or None,
            )
        except TypeError:
            # Older signature: join_muc(room, nick, password=None, ...)
            res = self.plugin["xep_0045"].join_muc(
                self.room_jid,
                self.nick,
                self.room_password or None,
            )

        if inspect.isawaitable(res):
            await res

        log.info(
            "Join MUC requested: %s (nick=%s, pw=%s)",
            self.room_jid,
            self.nick,
            "set" if self.room_password else "none",
        )

    def _stop(self) -> None:
        log.info("Stopping smoke test now.")
        self.disconnect()

    def _on_message(self, msg) -> None:
        body = str(msg.get("body") or "").strip()
        if not body:
            return

        self._rx_total += 1
        mtype = str(msg.get("type") or "")
        from_jid = str(msg.get("from") or "")

        if mtype == "groupchat":
            self._rx_groupchat += 1
        elif mtype == "chat":
            self._rx_chat += 1

        log.info("RX #%d type=%s from=%s %s", self._rx_total, mtype, from_jid, _summarize(body))


def main() -> int:
    env = load_env_file(ENV_PATH)

    # Require only creds.
    if not env.get("NWWS_JID") or not env.get("NWWS_PASSWORD"):
        log.error("Missing required keys in %s: NWWS_JID and/or NWWS_PASSWORD", str(ENV_PATH))
        return 2

    server_host = (env.get("NWWS_SERVER") or "nwws-oi.weather.gov").strip()
    try:
        server_port = int((env.get("NWWS_PORT") or "5222").strip())
    except Exception:
        log.error("NWWS_PORT is not an int: %r", env.get("NWWS_PORT"))
        return 2

    duration_s = int(os.environ.get("NWWS_SMOKE_DURATION_SECONDS", "120"))

    # Standard NWWS room. (You can override via env if you ever need to.)
    room_jid = (env.get("NWWS_ROOM") or "NWWS@conference.nwws-oi.weather.gov").strip()

    # Password behavior: room pw -> room password -> fallback to account password (Pidgin-style)
    room_password = (
        env.get("NWWS_ROOM_PW")
        or env.get("NWWS_ROOM_PASSWORD")
        or env.get("NWWS_PASSWORD")
        or ""
    ).strip() or None

    nick = (env.get("NWWS_NICK") or "SeasonalWeatherSmoke").strip()

    log.info(
        "Starting NWWS-OI smoke test (server=%s port=%d duration=%ds join_room=yes)",
        server_host,
        server_port,
        duration_s,
    )

    xmpp = NWWSSmoke(
        jid=env["NWWS_JID"],
        password=env["NWWS_PASSWORD"],
        server_host=server_host,
        server_port=server_port,
        duration_s=duration_s,
        room_jid=room_jid,
        room_password=room_password,
        nick=nick,
    )

    # Clean SIGINT so Ctrl+C doesn't wedge you.
    def _sigint(*_args):
        log.info("SIGINT received; disconnecting...")
        xmpp.disconnect()

    signal.signal(signal.SIGINT, _sigint)

    res = xmpp.connect((server_host, server_port), use_ssl=False, force_starttls=True)
    if res is False:
        log.error("connect() returned False (DNS/socket/port issue)")
        return 1

    xmpp.process(forever=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
