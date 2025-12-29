from __future__ import annotations

# =========================================================================================
#      MP"""""`MM                                                       dP              MM'"""'YMM
#      M  mmmmm..M                                                       88              M' .mmm. `M
#      M.      `YM .d8888b. .d8888b. .d8888b. .d8888b. 88d888b. .d8888b. 88              M  MMMMMooM dP    dP 88d888b. 88d888b. .d8888b. 88d888b. .d8888b. dP    dP
#      MMMMMMM.  M 88ooood8 88'  `88 Y8ooooo. 88'  `88 88'  `88 88'  `88 88              M  MMMMMMMM 88    88 88'  `88 88'  `88 88ooood8 88'  `88 88'  `"" 88    88
#      M. .MMM'  M 88.  ... 88.  .88       88 88.  .88 88    88 88.  .88 88              M. `MMM' .M 88.  .88 88       88       88.  ... 88    88 88.  ... 88.  .88
#      Mb.     .dM `88888P' `88888P8 `88888P' `88888P' dP    dP `88888P8 dP              MM.     .dM `88888P' dP       dP       `88888P' dP    dP `88888P' `8888P88
#      MMMMMMMMMMM                                                Seasonal_Currency      MMMMMMMMMMM                                                            .88
#                                                                                                                                                           d8888P.
# =========================================================================================

import asyncio
import inspect
import logging
import os
import random
import re
import threading
import time
from typing import Optional, Tuple

import slixmpp

log = logging.getLogger("seasonalweather.nwws")

_WMO_RE = re.compile(r"^[A-Z]{4}\d{2}\s+[A-Z]{4}\s+\d{6}(?:\s+[A-Z]{3})?$")
_AWIPS_RE = re.compile(r"^[A-Z0-9]{6,9}$")
_NUMONLY_RE = re.compile(r"^\d{1,6}$")


def _env_int(key: str, default: int) -> int:
    try:
        v = os.environ.get(key, "").strip()
        return int(v) if v else default
    except Exception:
        return default


class NWWSClient(slixmpp.ClientXMPP):
    """
    NWWS-OI XMPP client that:
      - Connects with STARTTLS
      - Joins the NWWS MUC (XEP-0045)
      - Extracts full NWWS payloads when present
      - Emits payload strings into an asyncio.Queue[str] owned by the main loop

    Slixmpp is asyncio-based. To avoid conflicting with your orchestrator loop, we run Slixmpp
    in a dedicated thread with its own event loop, then shuttle messages back to the main loop
    via call_soon_threadsafe().

    Resiliency:
      - If we fail to confirm MUC join (no groupchat traffic) within SEASONAL_NWWS_MUC_CONFIRM_SECONDS, we restart.
      - If MUC is confirmed but we see no RX traffic for SEASONAL_NWWS_STALL_SECONDS, we restart.
    """

    def __init__(
        self,
        jid: str,
        password: str,
        server: str,
        port: int,
        out_queue: "asyncio.Queue[str]",
        *,
        room_jid: str = "NWWS@conference.nwws-oi.weather.gov",
        room_password: Optional[str] = None,
        nick: str = "SeasonalWeather",
    ) -> None:
        super().__init__(jid, password)

        # Avoid attribute name "server" (Slixmpp historically had a deprecated property)
        self.server_host = server
        self.server_port = int(port)

        self.out_queue = out_queue

        self.room_jid = room_jid
        # If not provided, default to account password (Pidgin-style behavior)
        self.room_password = room_password if room_password is not None else password
        self.nick = nick

        # Main orchestrator loop (set when run_forever() is called)
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        # Thread runner
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._started_evt = threading.Event()

        # State
        self._rx_count = 0
        self._muc_confirmed = False
        self._last_rx_monotonic: Optional[float] = None

        # Plugins
        self.register_plugin("xep_0030")  # Service Discovery
        self.register_plugin("xep_0199")  # XMPP Ping
        self.register_plugin("xep_0045")  # Multi-User Chat (MUC)

        # Events
        self.add_event_handler("session_start", self._on_session_start)
        self.add_event_handler("message", self._on_message)
        self.add_event_handler("failed_auth", self._on_failed_auth)
        self.add_event_handler("disconnected", self._on_disconnected)

    async def _on_session_start(self, event) -> None:
        self.send_presence()
        self._last_rx_monotonic = time.monotonic()

        # Enable keepalive ONLY once we're on the correct loop (thread loop).
        # This helps, but the stall watchdog is still the main safety net.
        try:
            self["xep_0199"].enable_keepalive(interval=30, timeout=10)
        except Exception:
            pass

        try:
            roster = self.get_roster()
            if inspect.isawaitable(roster):
                await roster
        except Exception:
            pass

        log.info("NWWS-OI session started; joining MUC %s as %s", self.room_jid, self.nick)

        # join_muc signature varies by slixmpp version; never pass wait=...
        try:
            res = self.plugin["xep_0045"].join_muc(
                self.room_jid,
                self.nick,
                password=self.room_password or None,
            )
        except TypeError:
            # Older signature sometimes uses positional password
            res = self.plugin["xep_0045"].join_muc(
                self.room_jid,
                self.nick,
                self.room_password or None,
            )

        if inspect.isawaitable(res):
            await res

        log.info(
            "MUC join requested: %s (pw=%s)",
            self.room_jid,
            "set" if self.room_password else "none",
        )
        self._started_evt.set()

    def _on_failed_auth(self, event) -> None:
        log.error("NWWS-OI authentication failed")
        self._started_evt.set()
        self.disconnect()

    def _on_disconnected(self, event) -> None:
        log.info("NWWS-OI disconnected")

    # ----------------------------
    # Message parsing + summarizing
    # ----------------------------
    @staticmethod
    def _looks_like_banner(body: str, msg_type: str, from_jid: str) -> bool:
        # Big legal WARNING banner tends to arrive as type=normal from the server host.
        if msg_type == "normal" and body.startswith("**WARNING**"):
            return True
        if msg_type == "normal" and "**WARNING**WARNING**" in body[:80]:
            return True
        return False

    @staticmethod
    def _extract_nwws_payload(msg) -> Optional[str]:
        """
        Try to extract the full NWWS product payload from the stanza.

        In NWWS-OI, the raw product text is commonly inside an <x> element in an NWWS namespace.
        If found, prefer that over msg['body'] which may be a short headline.
        """
        try:
            xml = msg.xml
            # We accept any <x> element whose tag namespace contains "nwws"
            for node in xml.iter():
                tag = str(getattr(node, "tag", "") or "")
                if not tag:
                    continue
                if not tag.endswith("}x"):
                    continue
                if "nwws" not in tag.lower():
                    continue
                raw = (node.text or "").strip()
                if raw:
                    return raw
        except Exception:
            pass
        return None

    @staticmethod
    def _clean_lines(raw: str) -> list[str]:
        txt = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = txt.split("\n")

        while lines and not lines[0].strip():
            lines.pop(0)

        while lines and _NUMONLY_RE.match(lines[0].strip()):
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)

        return [ln.rstrip("\n") for ln in lines]

    @classmethod
    def _summarize_payload(cls, raw: str) -> Tuple[str, str, str]:
        lines = cls._clean_lines(raw)
        wmo = ""
        awips = ""
        title = ""

        scan = [ln.strip() for ln in lines[:40] if ln.strip()]

        for ln in scan:
            if _WMO_RE.match(ln):
                wmo = ln
                break

        for ln in scan:
            if ln in {"000", "$$"}:
                continue
            if _AWIPS_RE.match(ln) and not ln.isdigit():
                awips = ln
                break

        for ln in scan:
            if _WMO_RE.match(ln):
                continue
            if _AWIPS_RE.match(ln) or _NUMONLY_RE.match(ln):
                continue
            if ln.startswith("**WARNING**"):
                continue
            if "national weather service" in ln.lower():
                continue
            if len(ln) < 6 or len(ln) > 90:
                continue
            if sum(ch.isalpha() for ch in ln) < 6:
                continue
            title = ln
            break

        return (wmo, awips, title)

    # ----------------------------
    # Queue emission
    # ----------------------------
    def _emit(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        if self._main_loop is not None:

            def _put() -> None:
                try:
                    self.out_queue.put_nowait(text)
                except asyncio.QueueFull:
                    log.warning("NWWS queue full; dropping message")
                except Exception:
                    log.exception("NWWS queue push failed")

            self._main_loop.call_soon_threadsafe(_put)
            return

        try:
            self.out_queue.put_nowait(text)
        except asyncio.QueueFull:
            log.warning("NWWS queue full; dropping message")
        except Exception:
            log.exception("NWWS queue push failed")

    def _should_log_rx(self, msg_type: str, summary: str) -> bool:
        if self._rx_count <= 10:
            return True
        if msg_type != "groupchat":
            return True
        return (self._rx_count % 10) == 0

    def _on_message(self, msg) -> None:
        msg_type = str(msg.get("type") or "").strip()
        from_jid = str(msg.get("from") or "").strip()
        body = str(msg.get("body") or "").strip()

        # Prefer embedded payload over body (payload-only stanzas exist)
        payload = self._extract_nwws_payload(msg)
        src = "payload" if payload else "body"
        text = (payload if payload else body).strip()

        if not text:
            return

        if body and self._looks_like_banner(body, msg_type, from_jid):
            log.info("Ignoring NWWS server banner (type=%s from=%s)", msg_type, from_jid)
            return

        self._rx_count += 1
        self._last_rx_monotonic = time.monotonic()

        if (not self._muc_confirmed) and msg_type == "groupchat" and self.room_jid.lower() in from_jid.lower():
            self._muc_confirmed = True
            log.info("NWWS MUC join CONFIRMED via groupchat traffic")

        wmo, awips, title = self._summarize_payload(text)
        parts = []
        if awips:
            parts.append(awips)
        if wmo:
            parts.append(wmo)
        if title:
            parts.append(title)
        summary = " | ".join(parts) if parts else (title or (wmo or "message"))

        if self._should_log_rx(msg_type, summary):
            log.info(
                "NWWS RX#%d type=%s src=%s from=%s len=%d summary=%s",
                self._rx_count,
                msg_type,
                src,
                from_jid,
                len(text),
                summary,
            )

        self._emit(text)

    # ----------------------------
    # Threaded runner
    # ----------------------------
    def _thread_main(self) -> None:
        """
        Runs Slixmpp's process loop in a dedicated thread.

        CRITICAL: Slixmpp stores the loop on self.loop; we must override it to the
        thread's event loop or it will try to run the already-running main loop.
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Force Slixmpp to use THIS thread loop
            try:
                self.loop = loop  # slixmpp/xmlstream uses self.loop.run_forever()
            except Exception:
                pass
            if hasattr(self, "event_loop"):
                try:
                    setattr(self, "event_loop", loop)
                except Exception:
                    pass

            log.info("NWWS thread starting: connect %s:%d", self.server_host, self.server_port)

            res = self.connect(
                (self.server_host, self.server_port),
                use_ssl=False,
                force_starttls=True,
            )
            if res is False:
                log.error("Slixmpp connect() returned False (DNS/socket/port issue)")
                self._started_evt.set()
                return

            # Blocks until disconnect
            self.process(forever=True)

        except Exception:
            log.exception("NWWS thread crashed")
            self._started_evt.set()
        finally:
            try:
                self.disconnect()
            except Exception:
                pass
            self._stop_evt.set()

    async def run_forever(self) -> None:
        """
        Start the NWWS client and keep it running until cancelled.

        Env knobs:
          - SEASONAL_NWWS_STALL_SECONDS (default 60): after MUC confirmed, if no RX in this window -> restart
          - SEASONAL_NWWS_MUC_CONFIRM_SECONDS (default 30): if no MUC-confirming groupchat within this window -> restart
        """
        self._main_loop = asyncio.get_running_loop()

        stall_seconds = _env_int("SEASONAL_NWWS_STALL_SECONDS", 60)
        muc_confirm_seconds = _env_int("SEASONAL_NWWS_MUC_CONFIRM_SECONDS", 30)

        backoff = 2.0
        max_backoff = 60.0

        try:
            while True:
                # Reset run-state
                self._stop_evt.clear()
                self._started_evt.clear()
                self._muc_confirmed = False
                self._last_rx_monotonic = None

                self._thread = threading.Thread(target=self._thread_main, name="NWWSClient", daemon=True)
                self._thread.start()

                # Wait for session_start / failed_auth best-effort
                await asyncio.to_thread(self._started_evt.wait, 15)

                started_at = time.monotonic()
                if self._last_rx_monotonic is None:
                    self._last_rx_monotonic = started_at

                try:
                    while True:
                        await asyncio.sleep(5)

                        if self._stop_evt.is_set():
                            raise RuntimeError("NWWS disconnected / thread stopped")

                        now = time.monotonic()

                        if (not self._muc_confirmed) and muc_confirm_seconds > 0:
                            if (now - started_at) > float(muc_confirm_seconds):
                                raise RuntimeError(f"NWWS MUC join not confirmed within {muc_confirm_seconds}s")

                        if self._muc_confirmed and stall_seconds > 0 and self._last_rx_monotonic is not None:
                            age = now - self._last_rx_monotonic
                            if age > float(stall_seconds):
                                raise RuntimeError(f"NWWS stalled: no RX for {age:.1f}s (threshold {stall_seconds}s)")

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.warning("NWWS supervisor restarting: %s", e)

                # Clean stop before restart
                self.stop()

                await asyncio.sleep(backoff + random.uniform(0, backoff * 0.25))
                backoff = min(max_backoff, backoff * 1.6)

        except asyncio.CancelledError:
            self.stop()
            raise
        finally:
            self.stop()

    def stop(self) -> None:
        try:
            self.disconnect()
        except Exception:
            pass

        t = self._thread
        if t and t.is_alive():
            try:
                t.join(timeout=3.0)
            except Exception:
                pass

        self._thread = None
