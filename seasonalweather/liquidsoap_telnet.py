from __future__ import annotations
from pathlib import Path

import telnetlib
import time


class LiquidsoapTelnet:
    def _to_uri(self, wav_path: str) -> str:
        s = str(wav_path).strip()

        # If someone already wrapped it in quotes, strip them.
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1]

        # Already a URI? leave it alone.
        if "://" in s:
            return s

        # Convert filesystem path -> file:/// URI
        return Path(s).resolve().as_uri()

    def __init__(self, host: str, port: int, timeout: float = 3.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    def _send(self, command: str) -> str:
        # Liquidsoap telnet expects newline-terminated commands.
        with telnetlib.Telnet(self.host, self.port, timeout=self.timeout) as tn:
            tn.write(command.encode("utf-8") + b"\n")
            # read one response line (liquidsoap replies with something like "END")
            out = tn.read_until(b"END", timeout=self.timeout)
            return out.decode("utf-8", errors="replace")

    def push_alert(self, wav_path: str) -> None:
        uri = self._to_uri(wav_path)
        self._send(f"alert.push {uri}")

    def push_cycle(self, wav_path: str) -> None:
        uri = self._to_uri(wav_path)
        self._send(f"cycle.push {uri}")

    def flush_cycle(self) -> None:
        self._send("cycle.flush")

    def flush_alert(self) -> None:
        self._send("alert.flush")

    def skip_alert(self) -> None:
        # Best-effort. Some liquidsoap builds expose alert.skip.
        try:
            self._send("alert.skip")
        except Exception:
            pass

    def ping(self) -> bool:
        try:
            self._send("help")
            return True
        except Exception:
            return False
