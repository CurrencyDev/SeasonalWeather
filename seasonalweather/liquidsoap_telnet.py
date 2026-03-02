from __future__ import annotations

import logging
import re
import socket
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("seasonalweather.liquidsoap")

# Liquidsoap telnet server typically ends a response with a line containing END
_END_RE = re.compile(r"(?:\r?\n)END(?:\r?\n)", re.M)

# Help output in many builds is shown as: "| command"
_HELP_PIPE_RE = re.compile(r"^\s*\|\s*(.+?)\s*$")


class LiquidsoapTelnet:
    """
    Liquidsoap control client (line protocol over TCP).

    Backwards compatible across Liquidsoap versions:
      - Old: cycle.push, alert.push, cycle.flush, alert.flush, alert.skip
      - New: request_queue.push, request_queue.1.push, request_queue.flush_and_skip, request_queue.skip, ...

    Also avoids telnetlib/telnetlib3: Liquidsoap's control socket isn't real RFC telnet;
    it's a simple line protocol with END terminators.
    """

    def __init__(self, host: str, port: int, timeout: float = 3.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

        self._lock = threading.Lock()
        self._discovered = False

        # Prefixes for the two request queues (cycle and alert)
        self._cycle_prefix: Optional[str] = None
        self._alert_prefix: Optional[str] = None

        # Derived commands for flush/skip
        self._cycle_flush_cmd: Optional[str] = None
        self._alert_flush_cmd: Optional[str] = None
        self._alert_skip_cmd: Optional[str] = None

    def _to_uri(self, wav_path: str) -> str:
        s = str(wav_path).strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1]

        # Already a URI? leave it alone.
        if "://" in s:
            return s

        # Resolve symlinks/binds (important for /var/lib -> /home data moves)
        return Path(s).resolve().as_uri()

    def _recv_until_end(self, sock: socket.socket, deadline: float) -> str:
        sock.settimeout(0.5)
        buf = bytearray()
        end_at = time.time() + float(deadline)
        while time.time() < end_at:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)
                # decode loosely for delimiter detection
                if _END_RE.search(buf.decode("utf-8", "ignore")):
                    break
            except socket.timeout:
                pass
        return buf.decode("utf-8", "replace")

    def _send(self, command: str, *, read_deadline: Optional[float] = None) -> str:
        cmd = (command or "").strip()
        if not cmd:
            return ""

        with socket.create_connection((self.host, self.port), timeout=self.timeout) as s:
            # banner/prompt might exist; ignore
            _ = self._recv_until_end(s, deadline=0.6)

            # CRLF is safest for line protocol servers
            s.sendall((cmd + "\r\n").encode("utf-8"))
            out = self._recv_until_end(s, deadline=(read_deadline if read_deadline is not None else self.timeout))

            try:
                s.sendall(b"quit\r\n")
            except Exception:
                pass

        # Don’t silently swallow liquidsoap failures.
        if "ERROR:" in out:
            raise RuntimeError(out.strip())

        return out

    def _parse_help_commands(self, help_text: str) -> set[str]:
        """
        Extract command names from help output.
        We support both:
          - piped help lines: "| request_queue.push <uri>"
          - plain list lines (fallback): "request_queue.push <uri>"
        """
        cmds: set[str] = set()

        for ln in help_text.splitlines():
            ln = ln.rstrip()
            m = _HELP_PIPE_RE.match(ln)
            if m:
                cmds.add(m.group(1).strip())
                continue

            # Fallback: accept non-empty non-header lines that look like commands
            if ln and not ln.lower().startswith(("available commands", "type ")) and not ln.startswith("END"):
                # keep it conservative
                if re.match(r"^[A-Za-z0-9_.-]+(?:\s+<.*?>)?$", ln.strip()):
                    cmds.add(ln.strip())

        return cmds

    def _discover(self) -> None:
        help_txt = self._send("help", read_deadline=4.0)
        cmds = self._parse_help_commands(help_txt)

        # 1) Prefer legacy explicit queues if present
        if "cycle.push <uri>" in cmds and "alert.push <uri>" in cmds:
            self._cycle_prefix = "cycle"
            self._alert_prefix = "alert"
        else:
            # 2) Newer style: request_queue and request_queue.N
            prefixes: list[tuple[int, str]] = []
            if "request_queue.push <uri>" in cmds:
                prefixes.append((0, "request_queue"))
            for item in cmds:
                m = re.match(r"^(request_queue\.(\d+))\.push <uri>$", item)
                if m:
                    prefixes.append((int(m.group(2)), m.group(1)))

            prefixes.sort(key=lambda t: t[0])
            if len(prefixes) >= 2:
                self._cycle_prefix = prefixes[0][1]
                self._alert_prefix = prefixes[1][1]
            elif len(prefixes) == 1:
                # Degenerate case: one queue exists; use it for both rather than dead air
                self._cycle_prefix = prefixes[0][1]
                self._alert_prefix = prefixes[0][1]
            else:
                raise RuntimeError("Liquidsoap help did not expose any known push commands")

        assert self._cycle_prefix and self._alert_prefix

        # Flush commands vary by version
        def pick_flush(prefix: str) -> str:
            # Prefer flush_and_skip, else flush, else skip as last-ditch
            if f"{prefix}.flush_and_skip" in cmds:
                return f"{prefix}.flush_and_skip"
            if f"{prefix}.flush" in cmds:
                return f"{prefix}.flush"
            if f"{prefix}.skip" in cmds:
                return f"{prefix}.skip"
            # If nothing exists, keep something predictable; sending it will raise
            return f"{prefix}.flush_and_skip"

        self._cycle_flush_cmd = pick_flush(self._cycle_prefix)
        self._alert_flush_cmd = pick_flush(self._alert_prefix)

        # Skip alert: prefer skip if present, else use alert flush behavior
        if f"{self._alert_prefix}.skip" in cmds:
            self._alert_skip_cmd = f"{self._alert_prefix}.skip"
        else:
            self._alert_skip_cmd = self._alert_flush_cmd

        self._discovered = True
        log.info(
            "Liquidsoap control discovered: cycle=%s alert=%s cycle_flush=%s alert_flush=%s alert_skip=%s",
            self._cycle_prefix,
            self._alert_prefix,
            self._cycle_flush_cmd,
            self._alert_flush_cmd,
            self._alert_skip_cmd,
        )

    def _ensure_discovered(self) -> None:
        with self._lock:
            if not self._discovered:
                self._discover()

    # -------- Public API --------

    def push_alert(self, wav_path: str) -> None:
        self._ensure_discovered()
        uri = self._to_uri(wav_path)
        self._send(f'{self._alert_prefix}.push {uri}')

    def push_cycle(self, wav_path: str) -> None:
        self._ensure_discovered()
        uri = self._to_uri(wav_path)
        self._send(f'{self._cycle_prefix}.push {uri}')

    def flush_cycle(self) -> None:
        self._ensure_discovered()
        assert self._cycle_flush_cmd is not None
        self._send(self._cycle_flush_cmd)

    def flush_alert(self) -> None:
        self._ensure_discovered()
        assert self._alert_flush_cmd is not None
        self._send(self._alert_flush_cmd)

    def skip_alert(self) -> None:
        self._ensure_discovered()
        assert self._alert_skip_cmd is not None
        try:
            self._send(self._alert_skip_cmd)
        except Exception:
            pass

    def ping(self) -> bool:
        try:
            # Short response, good liveness probe
            self._send("version", read_deadline=2.0)
            return True
        except Exception:
            return False
