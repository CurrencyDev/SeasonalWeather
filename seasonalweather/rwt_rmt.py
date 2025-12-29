from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class RwtRmtSchedule:
    enabled: bool
    tz_name: str

    rwt_enabled: bool
    rwt_weekday: int  # 0=Mon..6=Sun
    rwt_hour: int
    rwt_minute: int

    rmt_enabled: bool
    rmt_nth: int      # 1..5
    rmt_weekday: int  # 0=Mon..6=Sun
    rmt_hour: int
    rmt_minute: int

    jitter_seconds: int
    postpone_minutes: int
    max_postpone_hours: int

    state_path: str


@dataclass
class TestState:
    last_rwt_period: str = ""
    last_rmt_period: str = ""

    @staticmethod
    def load(path: str) -> "TestState":
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f) or {}
            return TestState(
                last_rwt_period=str(obj.get("last_rwt_period", "")),
                last_rmt_period=str(obj.get("last_rmt_period", "")),
            )
        except FileNotFoundError:
            return TestState()
        except Exception:
            # never crash the station because of a bad state file
            return TestState()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "last_rwt_period": self.last_rwt_period,
                    "last_rmt_period": self.last_rmt_period,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        os.replace(tmp, path)


def _weekly_period_key(now: dt.datetime) -> str:
    iso = now.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _monthly_period_key(now: dt.datetime) -> str:
    return f"{now.year}-{now.month:02d}"


def _next_weekly(now: dt.datetime, weekday: int, hour: int, minute: int) -> dt.datetime:
    base = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    days_ahead = (weekday - base.weekday()) % 7
    if days_ahead == 0 and base <= now:
        days_ahead = 7
    return base + dt.timedelta(days=days_ahead)


def _days_in_month(y: int, m: int) -> int:
    if m == 12:
        nxt = dt.date(y + 1, 1, 1)
    else:
        nxt = dt.date(y, m + 1, 1)
    return (nxt - dt.date(y, m, 1)).days


def _nth_weekday_of_month(y: int, m: int, nth: int, weekday: int) -> Optional[dt.date]:
    if nth < 1 or nth > 5:
        return None
    first = dt.date(y, m, 1)
    delta = (weekday - first.weekday()) % 7
    day = 1 + delta + (nth - 1) * 7
    if day > _days_in_month(y, m):
        return None
    return dt.date(y, m, day)


def _next_monthly_nth_weekday(now: dt.datetime, nth: int, weekday: int, hour: int, minute: int) -> dt.datetime:
    y, m = now.year, now.month
    for _ in range(0, 15):
        d = _nth_weekday_of_month(y, m, nth, weekday)
        if d is not None:
            cand = dt.datetime(y, m, d.day, hour, minute, 0, 0, tzinfo=now.tzinfo)
            if cand > now:
                return cand
        # advance month
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return now + dt.timedelta(days=30)


GateFn = Callable[[], tuple[bool, str]]
FireFn = Callable[[str], Awaitable[None]]
LogFn = Callable[[str], None]


class RwtRmtScheduler:
    def __init__(self, schedule: RwtRmtSchedule, gate_fn: GateFn, fire_fn: FireFn, log_fn: LogFn):
        self.s = schedule
        self.gate_fn = gate_fn
        self.fire_fn = fire_fn
        self.log = log_fn
        self.tz = ZoneInfo(self.s.tz_name)
        self.state = TestState.load(self.s.state_path)
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    def _now(self) -> dt.datetime:
        return dt.datetime.now(self.tz)

    def _jitter(self) -> dt.timedelta:
        j = int(self.s.jitter_seconds)
        if j <= 0:
            return dt.timedelta(0)
        return dt.timedelta(seconds=random.uniform(0, float(j)))

    async def run_forever(self) -> None:
        if not self.s.enabled:
            self.log("[RWT/RMT] disabled")
            return

        self.log("[RWT/RMT] scheduler started")

        while not self._stop.is_set():
            now = self._now()

            next_events: list[tuple[str, dt.datetime]] = []

            if self.s.rwt_enabled:
                due = _next_weekly(now, self.s.rwt_weekday, self.s.rwt_hour, self.s.rwt_minute) + self._jitter()
                next_events.append(("RWT", due))

            if self.s.rmt_enabled:
                due = _next_monthly_nth_weekday(now, self.s.rmt_nth, self.s.rmt_weekday, self.s.rmt_hour, self.s.rmt_minute) + self._jitter()
                next_events.append(("RMT", due))

            if not next_events:
                await asyncio.sleep(5)
                continue

            event_code, due = min(next_events, key=lambda x: x[1])
            self.log(f"[RWT/RMT] next={event_code} at {due.isoformat()}")

            # wait in chunks so stop is responsive
            while not self._stop.is_set():
                now = self._now()
                if now >= due:
                    break
                await asyncio.sleep(min(30.0, max(1.0, (due - now).total_seconds())))

            if self._stop.is_set():
                break

            # de-dupe per period
            if event_code == "RWT":
                period = _weekly_period_key(due)
                if self.state.last_rwt_period == period:
                    self.log(f"[RWT/RMT] skip {event_code}: already ran for {period}")
                    continue
            else:
                period = _monthly_period_key(due)
                if self.state.last_rmt_period == period:
                    self.log(f"[RWT/RMT] skip {event_code}: already ran for {period}")
                    continue

            await self._attempt_with_postpone(event_code, due)

        self.log("[RWT/RMT] scheduler stopped")

    async def _attempt_with_postpone(self, event_code: str, scheduled: dt.datetime) -> None:
        deadline = scheduled + dt.timedelta(hours=max(1, int(self.s.max_postpone_hours)))
        postpone = dt.timedelta(minutes=max(1, int(self.s.postpone_minutes)))

        while not self._stop.is_set():
            now = self._now()
            if now > deadline:
                self.log(f"[RWT/RMT] skip {event_code}: gate blocked past deadline")
                return

            ok, reason = self.gate_fn()
            if ok:
                break

            self.log(f"[RWT/RMT] gate blocked for {event_code}: {reason} (postpone {int(postpone.total_seconds()/60)}m)")
            await asyncio.sleep(max(30.0, postpone.total_seconds()))

        if self._stop.is_set():
            return

        self.log(f"[RWT/RMT] firing {event_code}")
        await self.fire_fn(event_code)

        # write state after success
        if event_code == "RWT":
            self.state.last_rwt_period = _weekly_period_key(self._now())
        else:
            self.state.last_rmt_period = _monthly_period_key(self._now())
        self.state.save(self.s.state_path)
        self.log(f"[RWT/RMT] completed {event_code}")
