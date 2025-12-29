from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple


log = logging.getLogger("seasonalweather")


@dataclass(frozen=True)
class ErnSameEvent:
    """
    Event emitted when SAME is decoded from an ERN/JON (or similar) node stream.
    This is intentionally narrow: we only carry decoded SAME header/EOM.
    """
    kind: str  # "header" or "eom"
    text: str
    confidence: float
    start_seconds: float

    org: Optional[str] = None
    event: Optional[str] = None
    locations: Tuple[str, ...] = ()
    tttt: Optional[str] = None
    jjjhhmm: Optional[str] = None
    sender: Optional[str] = None

    source: str = "ERN"
    url: str = ""


def _project_root() -> Path:
    # /opt/seasonalweather/app/seasonalweather/ern_gwes.py -> parents[1] == /opt/seasonalweather/app
    return Path(__file__).resolve().parents[1]


def _same_listen_module_cmd(url: str, *, sr: int, dedupe: float, trigger_ratio: float, tail: float) -> list[str]:
    return [
        sys.executable,
        "-m",
        "seasonalweather.same_listen",
        "--url",
        url,
        "--sr",
        str(int(sr)),
        "--dedupe",
        str(float(dedupe)),
        "--trigger-ratio",
        str(float(trigger_ratio)),
        "--tail",
        str(float(tail)),
        "--jsonl",
    ]


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v else default


class ErnGwesMonitor:
    """
    Spawns same_listen.py as a module, reads JSONL decoded SAME messages,
    filters to service area SAME/FIPS, and emits ErnSameEvent into an asyncio queue.

    This is a "Level 3" source: we do not try to fetch or synthesize official text.
    We just observe and surface SAME activity.
    """

    def __init__(
        self,
        *,
        out_queue: "asyncio.Queue[ErnSameEvent]",
        same_fips_allow: Sequence[str],
        url: str,
        sample_rate: int = 48000,
        dedupe_seconds: float = 20.0,
        trigger_ratio: float = 8.0,
        tail_seconds: float = 10.0,
        confidence_min: float = 0.25,
        name: str = "ERN/JON",
    ) -> None:
        self.out_queue = out_queue
        self.same_fips_allow = set(str(x).strip() for x in (same_fips_allow or []) if str(x).strip())
        self.url = str(url).strip()
        self.sample_rate = int(sample_rate)
        self.dedupe_seconds = float(dedupe_seconds)
        self.trigger_ratio = float(trigger_ratio)
        self.tail_seconds = float(tail_seconds)
        self.confidence_min = float(confidence_min)
        self.name = str(name)

        if not self.url:
            raise ValueError("ERN monitor requires a non-empty url")

    def _service_area_hit(self, locs: Sequence[str]) -> bool:
        if not self.same_fips_allow:
            return True  # if you ever run without config, don't hard-drop everything
        for x in locs or []:
            if x in self.same_fips_allow:
                return True
        return False

    def _parse_jsonl_line(self, line: str) -> Optional[ErnSameEvent]:
        try:
            obj = json.loads(line)
        except Exception:
            return None

        kind = str(obj.get("kind") or "").strip().lower()
        text = str(obj.get("text") or "").strip()
        if kind not in {"header", "eom"} or not text:
            return None

        conf = float(obj.get("confidence") or 0.0)
        start_seconds = float(obj.get("start_seconds") or 0.0)

        org = obj.get("org")
        event = obj.get("event")
        tttt = obj.get("tttt")
        jjjhhmm = obj.get("jjjhhmm")
        sender = obj.get("sender")
        locs = obj.get("locations") or []
        try:
            locs_t = tuple(str(x) for x in locs if str(x))
        except Exception:
            locs_t = ()

        return ErnSameEvent(
            kind=kind,
            text=text,
            confidence=conf,
            start_seconds=start_seconds,
            org=str(org) if org else None,
            event=str(event) if event else None,
            locations=locs_t,
            tttt=str(tttt) if tttt else None,
            jjjhhmm=str(jjjhhmm) if jjjhhmm else None,
            sender=str(sender) if sender else None,
            source=self.name,
            url=self.url,
        )

    async def run_forever(self) -> None:
        root = _project_root()

        # Make module import robust under systemd by pinning cwd + PYTHONPATH.
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(root)

        cmd = _same_listen_module_cmd(
            self.url,
            sr=self.sample_rate,
            dedupe=self.dedupe_seconds,
            trigger_ratio=self.trigger_ratio,
            tail=self.tail_seconds,
        )

        log.info("ERN monitor starting (%s): %s", self.name, " ".join(cmd))

        while True:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(root),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            assert proc.stdout is not None
            assert proc.stderr is not None

            async def _drain_stderr() -> None:
                try:
                    while True:
                        b = await proc.stderr.readline()
                        if not b:
                            return
                        s = b.decode("utf-8", "replace").rstrip()
                        if s:
                            log.warning("ERN same_listen stderr: %s", s)
                except Exception:
                    return

            stderr_task = asyncio.create_task(_drain_stderr(), name="ern_same_listen_stderr")

            try:
                while True:
                    b = await proc.stdout.readline()
                    if not b:
                        break
                    line = b.decode("utf-8", "replace").strip()
                    if not line:
                        continue

                    ev = self._parse_jsonl_line(line)
                    if not ev:
                        continue

                    # Confidence gate (keep it low; ERN audio quality can be… “heritage”)
                    if ev.confidence < self.confidence_min:
                        continue

                    # Only headers are service-area-filtered; EOM is informational.
                    # IMPORTANT: log out-of-area headers so it doesn't look like decoding failed.
                    if ev.kind == "header" and not self._service_area_hit(ev.locations):
                        log.info(
                            "ERN SAME header out-of-area (dropped): org=%s event=%s sender=%s conf=%.3f same=%s text=%s",
                            ev.org,
                            ev.event,
                            (ev.sender or "").strip(),
                            ev.confidence,
                            ",".join(ev.locations[:12]) + ("..." if len(ev.locations) > 12 else ""),
                            ev.text,
                        )
                        continue

                    # Non-blocking enqueue (drop if full rather than wedging the monitor)
                    try:
                        self.out_queue.put_nowait(ev)
                    except asyncio.QueueFull:
                        log.warning("ERN queue full; dropping event %s %s", ev.kind, ev.text[:32])

            finally:
                try:
                    stderr_task.cancel()
                except Exception:
                    pass
                try:
                    proc.kill()
                except Exception:
                    pass

            # Restart backoff
            log.warning("ERN monitor exited; restarting in 2s (%s)", self.name)
            await asyncio.sleep(2.0)


def defaults_from_env() -> dict:
    """
    Optional helper if you want to construct this from env vars.
    (main.py can also manage env itself; this is just convenience.)
    """
    return dict(
        sample_rate=_env_int("SEASONAL_ERN_SR", 48000),
        dedupe_seconds=_env_float("SEASONAL_ERN_DEDUPE_SECONDS", 20.0),
        trigger_ratio=_env_float("SEASONAL_ERN_TRIGGER_RATIO", 8.0),
        tail_seconds=_env_float("SEASONAL_ERN_TAIL_SECONDS", 10.0),
        confidence_min=_env_float("SEASONAL_ERN_CONFIDENCE_MIN", 0.25),
        name=_env_str("SEASONAL_ERN_NAME", "ERN/JON"),
    )
