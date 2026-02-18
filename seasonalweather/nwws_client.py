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
from dataclasses import dataclass
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


def _env_str(key: str, default: str = "") -> str:
    v = os.environ.get(key, "")
    return v.strip() if isinstance(v, str) else default


async def _wait_evt(evt: threading.Event, timeout: float) -> bool:
    timeout = max(0.0, float(timeout))
    return await asyncio.to_thread(evt.wait, timeout)


@dataclass
class _SharedState:
    """
    Shared cross-thread state (thread-safe).
    The NWWS worker thread writes these; the async supervisor reads them.
    """
    lock: threading.Lock

    rx_count: int = 0
    muc_joined: bool = False
    last_rx_monotonic: float = 0.0
    last_any_monotonic: float = 0.0
    last_error: str = ""

    def reset(self) -> None:
        with self.lock:
            self.rx_count = 0
            self.muc_joined = False
            now = time.monotonic()
            self.last_rx_monotonic = 0.0
            self.last_any_monotonic = now
            self.last_error = ""

    def mark_any(self) -> None:
        with self.lock:
            self.last_any_monotonic = time.monotonic()

    def mark_rx(self) -> None:
        with self.lock:
            now = time.monotonic()
            self.last_rx_monotonic = now
            self.last_any_monotonic = now
            self.rx_count += 1

    def set_muc(self, v: bool) -> None:
        with self.lock:
            self.muc_joined = v
            self.last_any_monotonic = time.monotonic()

    def set_error(self, msg: str) -> None:
        with self.lock:
            self.last_error = (msg or "").strip()
            self.last_any_monotonic = time.monotonic()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "rx_count": self.rx_count,
                "muc_joined": self.muc_joined,
                "last_rx_monotonic": self.last_rx_monotonic,
                "last_any_monotonic": self.last_any_monotonic,
                "last_error": self.last_error,
            }


class _NWWSXMPP(slixmpp.ClientXMPP):
    """
    Slixmpp client that runs ONLY inside the worker thread's event loop.
    """

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        room_jid: str,
        room_password: Optional[str],
        nick: str,
        shared: _SharedState,
        muc_joined_evt: threading.Event,
        started_evt: threading.Event,
        stop_evt: threading.Event,
        emit_cb,  # callable(text:str)->None (thread-safe)
        muc_confirm_timeout_seconds: int,
    ) -> None:
        super().__init__(jid, password)

        self.room_jid = room_jid
        self.room_password = room_password if room_password is not None else password
        self.nick = nick

        self._shared = shared
        self._muc_joined_evt = muc_joined_evt
        self._started_evt = started_evt
        self._stop_evt = stop_evt
        self._emit_cb = emit_cb

        self._muc_confirm_timeout_seconds = max(5, int(muc_confirm_timeout_seconds))

        # Must be created on the worker loop.
        self._muc_self_presence_evt: Optional[asyncio.Event] = None

        # Plugins
        self.register_plugin("xep_0030")  # Service Discovery
        self.register_plugin("xep_0199")  # XMPP Ping
        self.register_plugin("xep_0045")  # Multi-User Chat (MUC)

        # Events
        self.add_event_handler("session_start", self._on_session_start)
        self.add_event_handler("message", self._on_message)
        self.add_event_handler("presence", self._on_presence)
        self.add_event_handler("failed_auth", self._on_failed_auth)
        self.add_event_handler("disconnected", self._on_disconnected)

    async def _on_session_start(self, event) -> None:
        self._shared.mark_any()

        # IMPORTANT: session_start means auth/bind succeeded.
        # Signal "started" NOW so the supervisor doesn't time out while waiting on MUC presence.
        self._started_evt.set()

        self._muc_self_presence_evt = asyncio.Event()
        self.send_presence()

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

        # Portable join (no join_muc_wait, no maxhistory).
        try:
            res = self.plugin["xep_0045"].join_muc(
                self.room_jid,
                self.nick,
                password=(self.room_password or None),
            )
        except TypeError:
            res = self.plugin["xep_0045"].join_muc(
                self.room_jid,
                self.nick,
                self.room_password or None,
            )

        if inspect.isawaitable(res):
            await res

        # Prefer self-presence confirmation, but don't brick the whole client if it never arrives.
        try:
            assert self._muc_self_presence_evt is not None
            await asyncio.wait_for(
                self._muc_self_presence_evt.wait(),
                timeout=float(self._muc_confirm_timeout_seconds),
            )
            self._shared.set_muc(True)
            self._muc_joined_evt.set()
            log.info("NWWS MUC join CONFIRMED (self-presence) for %s", self.room_jid)
        except Exception as e:
            # Not fatal: groupchat traffic can confirm join too.
            self._shared.set_error(f"MUC self-presence confirm not seen yet: {e!r}")
            log.warning("NWWS MUC self-presence not seen (will accept groupchat confirm): %r", e)

    def _on_failed_auth(self, event) -> None:
        self._shared.set_error("NWWS-OI authentication failed")
        log.error("NWWS-OI authentication failed")
        self._started_evt.set()
        try:
            self.disconnect(wait=False)
        except TypeError:
            self.disconnect()

    def _on_disconnected(self, event) -> None:
        self._shared.mark_any()
        log.info("NWWS-OI disconnected")

    def _on_presence(self, pres) -> None:
        self._shared.mark_any()
        try:
            frm = pres.get("from")
            if frm is None:
                return

            s = str(frm)
            bare = getattr(frm, "bare", None) or s.split("/")[0]
            resource = getattr(frm, "resource", None)
            if resource is None:
                resource = s.split("/", 1)[1] if "/" in s else ""

            if str(bare).lower() == self.room_jid.lower() and str(resource) == self.nick:
                if self._muc_self_presence_evt is not None and not self._muc_self_presence_evt.is_set():
                    self._muc_self_presence_evt.set()
        except Exception:
            pass

    # ----------------------------
    # Message parsing + summarizing
    # ----------------------------
    @staticmethod
    def _looks_like_banner(body: str, msg_type: str, from_jid: str) -> bool:
        if msg_type == "normal" and body.startswith("**WARNING**"):
            return True
        if msg_type == "normal" and "**WARNING**WARNING**" in body[:80]:
            return True
        return False

    @staticmethod
    def _extract_nwws_payload(msg) -> Optional[str]:
        try:
            xml = msg.xml
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

    def _should_log_rx(self, msg_type: str, rx_count: int) -> bool:
        if rx_count <= 10:
            return True
        if msg_type != "groupchat":
            return True
        return (rx_count % 10) == 0

    def _on_message(self, msg) -> None:
        self._shared.mark_any()

        msg_type = str(msg.get("type") or "").strip()
        from_jid = str(msg.get("from") or "").strip()
        body = str(msg.get("body") or "").strip()

        payload = self._extract_nwws_payload(msg)
        src = "payload" if payload else "body"
        text = (payload if payload else body).strip()

        if not text:
            return

        if body and self._looks_like_banner(body, msg_type, from_jid):
            log.info("Ignoring NWWS server banner (type=%s from=%s)", msg_type, from_jid)
            return

        self._shared.mark_rx()
        snap = self._shared.snapshot()
        rx_count = int(snap["rx_count"])

        # If we see groupchat traffic from the room, that is a valid join confirmation.
        if (not snap["muc_joined"]) and msg_type == "groupchat" and self.room_jid.lower() in from_jid.lower():
            self._shared.set_muc(True)
            self._muc_joined_evt.set()
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

        if self._should_log_rx(msg_type, rx_count):
            log.info(
                "NWWS RX#%d type=%s src=%s from=%s len=%d summary=%s",
                rx_count,
                msg_type,
                src,
                from_jid,
                len(text),
                summary,
            )

        try:
            self._emit_cb(text)
        except Exception:
            log.exception("NWWS emit callback failed")

    def request_stop(self) -> None:
        self._shared.mark_any()
        try:
            self.disconnect(wait=False)
        except TypeError:
            self.disconnect()


class NWWSClient:
    """
    NWWS-OI supervisor:
      - Slixmpp runs in a dedicated thread + dedicated asyncio loop
      - Confirms MUC join via self-presence OR groupchat fallback
      - Emits payloads into an asyncio.Queue[str] owned by the orchestrator loop
      - Restarts on disconnect, startup failure, or stall
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
        self.jid = jid
        self.password = password
        self.server_host = server
        self.server_port = int(port)

        self.out_queue = out_queue
        self.room_jid = room_jid
        self.room_password = room_password
        self.nick = nick

        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._started_evt = threading.Event()
        self._muc_joined_evt = threading.Event()

        self._shared = _SharedState(lock=threading.Lock())

        self._xmpp_lock = threading.Lock()
        self._xmpp: Optional[_NWWSXMPP] = None
        self._xmpp_loop: Optional[asyncio.AbstractEventLoop] = None

        self._worker_id = 0

    def _emit(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        loop = self._main_loop
        if loop is None:
            try:
                self.out_queue.put_nowait(text)
            except Exception:
                pass
            return

        def _put() -> None:
            try:
                self.out_queue.put_nowait(text)
            except asyncio.QueueFull:
                log.warning("NWWS queue full; dropping message")
            except Exception:
                log.exception("NWWS queue push failed")

        loop.call_soon_threadsafe(_put)

    def _thread_main(self, worker_id: int, muc_confirm_timeout_seconds: int) -> None:
        loop: Optional[asyncio.AbstractEventLoop] = None
        xmpp: Optional[_NWWSXMPP] = None

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._shared.mark_any()

            xmpp = _NWWSXMPP(
                self.jid,
                self.password,
                room_jid=self.room_jid,
                room_password=self.room_password,
                nick=self.nick,
                shared=self._shared,
                muc_joined_evt=self._muc_joined_evt,
                started_evt=self._started_evt,
                stop_evt=self._stop_evt,
                emit_cb=self._emit,
                muc_confirm_timeout_seconds=muc_confirm_timeout_seconds,
            )

            # Force Slixmpp to use THIS loop
            try:
                xmpp.loop = loop
            except Exception:
                pass
            if hasattr(xmpp, "event_loop"):
                try:
                    setattr(xmpp, "event_loop", loop)
                except Exception:
                    pass

            with self._xmpp_lock:
                self._xmpp = xmpp
                self._xmpp_loop = loop

            log.info("NWWS worker[%d] connecting %s:%d", worker_id, self.server_host, self.server_port)

            res = xmpp.connect(
                (self.server_host, self.server_port),
                use_ssl=False,
                force_starttls=True,
            )
            if res is False:
                self._shared.set_error("Slixmpp connect() returned False (DNS/socket/port issue)")
                log.error("NWWS worker[%d] connect() returned False", worker_id)
                self._started_evt.set()
                return

            xmpp.process(forever=True)

        except Exception as e:
            self._shared.set_error(f"NWWS worker crashed: {e!r}")
            log.exception("NWWS worker[%d] crashed", worker_id)
            self._started_evt.set()

        finally:
            if xmpp is not None:
                try:
                    xmpp.disconnect(wait=False)
                except TypeError:
                    try:
                        xmpp.disconnect()
                    except Exception:
                        pass
                except Exception:
                    pass

            if loop is not None:
                try:
                    pending = asyncio.all_tasks(loop=loop)
                except Exception:
                    pending = set()

                for t in list(pending):
                    try:
                        t.cancel()
                    except Exception:
                        pass

                if pending:
                    try:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    except Exception:
                        pass

                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass

                try:
                    loop.close()
                except Exception:
                    pass

            with self._xmpp_lock:
                self._xmpp = None
                self._xmpp_loop = None

            self._stop_evt.set()
            self._shared.mark_any()

    async def run_forever(self) -> None:
        """
        Iterative supervisor loop (no recursion).

        Env knobs:
          - SEASONAL_NWWS_STALL_SECONDS (default 60)
          - SEASONAL_NWWS_MUC_CONFIRM_SECONDS (default 30)
          - SEASONAL_NWWS_START_WAIT_SECONDS (default 25)   # wait for session_start (auth ok)
          - SEASONAL_NWWS_JOIN_WAIT_SECONDS (default 35)    # wait for MUC join confirm (presence or groupchat)
          - SEASONAL_NWWS_BACKOFF_MAX_SECONDS (default 90)
        """
        self._main_loop = asyncio.get_running_loop()

        stall_seconds = _env_int("SEASONAL_NWWS_STALL_SECONDS", 60)
        muc_confirm_seconds = _env_int("SEASONAL_NWWS_MUC_CONFIRM_SECONDS", 30)
        start_wait_seconds = _env_int("SEASONAL_NWWS_START_WAIT_SECONDS", 25)
        join_wait_seconds = _env_int("SEASONAL_NWWS_JOIN_WAIT_SECONDS", 35)
        max_backoff = float(_env_int("SEASONAL_NWWS_BACKOFF_MAX_SECONDS", 90))

        backoff = 2.0

        while True:
            try:
                # Hard stop any previous worker before starting a new one
                self.stop()

                self._worker_id += 1
                wid = self._worker_id

                self._stop_evt.clear()
                self._started_evt.clear()
                self._muc_joined_evt.clear()
                self._shared.reset()

                t = threading.Thread(
                    target=self._thread_main,
                    args=(wid, muc_confirm_seconds),
                    name=f"NWWSClient[{wid}]",
                    daemon=True,
                )
                self._thread = t
                t.start()

                # 1) Wait for session_start (auth/bind ok)
                started_ok = await _wait_evt(self._started_evt, float(start_wait_seconds))
                snap = self._shared.snapshot()

                if not started_ok:
                    log.warning(
                        "NWWS supervisor restarting (session_start timeout after %ss)",
                        start_wait_seconds,
                    )
                    await asyncio.sleep(backoff + random.uniform(0, backoff * 0.25))
                    backoff = min(max_backoff, backoff * 1.6)
                    continue

                if snap.get('last_error') and 'authentication failed' in str(snap['last_error']).lower():
                    log.warning('NWWS supervisor restarting (auth failed): %s', snap['last_error'])
                    await asyncio.sleep(backoff + random.uniform(0, backoff * 0.25))
                    backoff = min(max_backoff, backoff * 1.6)
                    continue

                # 2) Wait for MUC join confirmation (presence OR groupchat)
                joined_ok = await _wait_evt(self._muc_joined_evt, float(join_wait_seconds))
                snap = self._shared.snapshot()

                if not joined_ok:
                    log.warning(
                        "NWWS supervisor restarting (MUC not confirmed after %ss). last_error=%r",
                        join_wait_seconds,
                        snap.get('last_error', ''),
                    )
                    await asyncio.sleep(backoff + random.uniform(0, backoff * 0.25))
                    backoff = min(max_backoff, backoff * 1.6)
                    continue

                log.info("NWWS supervisor: started worker[%d], MUC joined=True", wid)
                backoff = 2.0

                # 3) Monitor loop
                while True:
                    await asyncio.sleep(5)

                    if self._stop_evt.is_set():
                        raise RuntimeError("NWWS disconnected / worker stopped")

                    snap = self._shared.snapshot()
                    if snap.get('last_error') and 'authentication failed' in str(snap['last_error']).lower():
                        raise RuntimeError(f"NWWS auth error: {snap['last_error']}")

                    if stall_seconds > 0 and snap.get('muc_joined') and float(snap.get('last_rx_monotonic', 0.0)) > 0:
                        age = time.monotonic() - float(snap['last_rx_monotonic'])
                        if age > float(stall_seconds):
                            raise RuntimeError(
                                f"NWWS stalled: no RX for {age:.1f}s (threshold {stall_seconds}s)"
                            )

            except asyncio.CancelledError:
                self.stop()
                raise
            except Exception as e:
                log.warning("NWWS supervisor restarting: %s", e)
                self.stop()
                await asyncio.sleep(backoff + random.uniform(0, backoff * 0.25))
                backoff = min(max_backoff, backoff * 1.6)
                continue
    def stop(self) -> None:
        self._stop_evt.set()

        xmpp = None
        loop = None
        with self._xmpp_lock:
            xmpp = self._xmpp
            loop = self._xmpp_loop

        if xmpp is not None and loop is not None:
            try:
                loop.call_soon_threadsafe(xmpp.request_stop)
            except Exception:
                pass

        t = self._thread
        if t and t.is_alive():
            try:
                t.join(timeout=10.0)
            except Exception:
                pass
            if t.is_alive():
                log.warning("NWWS stop: worker thread did not exit within timeout")

        self._thread = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    jid = _env_str("NWWS_JID", "")
    pw = _env_str("NWWS_PASSWORD", "")
    if not jid or not pw:
        raise SystemExit("Set NWWS_JID and NWWS_PASSWORD env vars to smoke-test this module.")

    server = _env_str("NWWS_SERVER", "nwws-oi.weather.gov")
    port = _env_int("NWWS_PORT", 5222)
    room = _env_str("NWWS_ROOM", "NWWS@conference.nwws-oi.weather.gov")
    nick = _env_str("NWWS_NICK", "SeasonalWeather")

    async def _main() -> None:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=50)
        c = NWWSClient(jid, pw, server, port, q, room_jid=room, nick=nick)
        task = asyncio.create_task(c.run_forever())
        try:
            msg = await asyncio.wait_for(q.get(), timeout=60)
            print("GOT MESSAGE (first 500 chars):")
            print(msg[:500])
        finally:
            task.cancel()
            try:
                await task
            except Exception:
                pass
            c.stop()

    asyncio.run(_main())
