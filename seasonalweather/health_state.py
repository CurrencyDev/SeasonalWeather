"""SeasonalWeather source-health state machine.

The state machine is deliberately separate from main.py: callers only mark
source success/failure and ask for a cycle-context snapshot.  It owns hysteresis,
severity selection, and spoken degraded/detached wording.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Mapping

log = logging.getLogger("seasonalweather.health")

Probe = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class HealthSourceConfig:
    name: str
    enabled: bool = True
    role: str = "general"  # alert | alert_redundant | forecast | observation | general
    stale_after_seconds: int = 600
    failure_threshold: int = 3
    critical: bool = False


@dataclass(frozen=True)
class HealthStateConfig:
    enabled: bool = True
    check_interval_seconds: int = 30
    min_hold_seconds: int = 300
    detached_loop_only: bool = True
    source_impaired_message: str = "SeasonalWeather is operating with reduced data-feed redundancy. Some information may be delayed."
    degraded_message: str = "SeasonalWeather is operating in a degraded mode. Some National Weather Service data may be delayed or unavailable."
    critical_message: str = "SeasonalWeather is operating in a degraded mode. Current watches, warnings, and advisories may be delayed or unavailable."
    detached_message: str = "SeasonalWeather is temporarily unable to receive current National Weather Service information. Please use another weather information source or visit weather.gov for the latest information."
    sources: tuple[HealthSourceConfig, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class HealthCycleContext:
    mode: str = "normal"  # normal | source_impaired | degraded | critical_degraded | detached
    notice: str = ""
    status_line: str = ""
    detached_loop_only: bool = False
    reasons: tuple[str, ...] = ()


@dataclass
class _SourceState:
    cfg: HealthSourceConfig
    last_success: dt.datetime | None = None
    last_failure: dt.datetime | None = None
    observed_since: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    consecutive_failures: int = 0
    disabled_reason: str = ""
    last_error: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled) and not self.disabled_reason


def _source_status(
    st: _SourceState,
    now: dt.datetime,
) -> tuple[str, str]:
    if not st.enabled:
        return "disabled", "disabled_by_configuration"
    reference = st.last_success or st.observed_since
    if (now - reference).total_seconds() > max(
        1,
        st.cfg.stale_after_seconds,
    ):
        return "degraded", "source_stale"
    if st.consecutive_failures >= max(1, st.cfg.failure_threshold):
        return "degraded", "failure_threshold_reached"
    if st.last_success is None:
        return "unknown", "awaiting_first_success"
    if st.last_failure is not None and st.last_failure > st.last_success:
        return "degraded", "recent_source_failure"
    return "healthy", "source_current"


def default_health_sources() -> tuple[HealthSourceConfig, ...]:
    return (
        HealthSourceConfig(name="nwws_oi", role="alert_redundant", stale_after_seconds=600, failure_threshold=2),
        HealthSourceConfig(name="cap_api", role="alert", stale_after_seconds=300, failure_threshold=3, critical=True),
        HealthSourceConfig(name="nws_api", role="forecast", stale_after_seconds=900, failure_threshold=3),
    )


def _sources_from_config(raw_sources) -> tuple[HealthSourceConfig, ...]:
    if not raw_sources:
        return default_health_sources()
    out: list[HealthSourceConfig] = []
    if isinstance(raw_sources, Mapping):
        iterator = raw_sources.items()
    else:
        iterator = ((getattr(s, "name", ""), s) for s in raw_sources)
    for name, raw in iterator:
        out.append(
            HealthSourceConfig(
                name=str(getattr(raw, "name", name) or name).strip(),
                enabled=bool(getattr(raw, "enabled", True)),
                role=str(getattr(raw, "role", "general") or "general").strip().lower(),
                stale_after_seconds=int(getattr(raw, "stale_after_seconds", 600)),
                failure_threshold=int(getattr(raw, "failure_threshold", 3)),
                critical=bool(getattr(raw, "critical", False)),
            )
        )
    return tuple(s for s in out if s.name)


def config_from_app(raw) -> HealthStateConfig:
    if raw is None:
        return HealthStateConfig(sources=default_health_sources())
    return HealthStateConfig(
        enabled=bool(getattr(raw, "enabled", True)),
        check_interval_seconds=int(getattr(raw, "check_interval_seconds", 30)),
        min_hold_seconds=int(getattr(raw, "min_hold_seconds", 300)),
        detached_loop_only=bool(getattr(raw, "detached_loop_only", True)),
        source_impaired_message=str(getattr(raw, "source_impaired_message", HealthStateConfig.source_impaired_message) or HealthStateConfig.source_impaired_message),
        degraded_message=str(getattr(raw, "degraded_message", HealthStateConfig.degraded_message) or HealthStateConfig.degraded_message),
        critical_message=str(getattr(raw, "critical_message", HealthStateConfig.critical_message) or HealthStateConfig.critical_message),
        detached_message=str(getattr(raw, "detached_message", HealthStateConfig.detached_message) or HealthStateConfig.detached_message),
        sources=_sources_from_config(getattr(raw, "sources", None)),
    )


class HealthStateMachine:
    def __init__(self, cfg) -> None:
        self.cfg = config_from_app(cfg)
        self._sources: dict[str, _SourceState] = {s.name: _SourceState(s) for s in self.cfg.sources}
        self._probes: dict[str, Probe] = {}
        self._mode = "normal"
        self._mode_since = dt.datetime.now(dt.timezone.utc)
        self._last_snapshot = HealthCycleContext()
        self._lock = asyncio.Lock()

    def register_probe(self, source: str, probe: Probe) -> None:
        if source in self._sources:
            self._probes[source] = probe

    def mark_disabled(self, source: str, reason: str = "disabled") -> None:
        st = self._sources.get(source)
        if st is None:
            return
        st.disabled_reason = reason or "disabled"
        log.info("Health source disabled; source=%s reason=%s", source, st.disabled_reason)

    def mark_success(self, source: str) -> None:
        st = self._sources.get(source)
        if st is None:
            return
        now = dt.datetime.now(dt.timezone.utc)
        st.last_success = now
        st.consecutive_failures = 0
        st.last_error = ""

    def mark_failure(self, source: str, error: object = "") -> None:
        st = self._sources.get(source)
        if st is None:
            return
        now = dt.datetime.now(dt.timezone.utc)
        st.last_failure = now
        st.consecutive_failures += 1
        st.last_error = str(error or "failure")[:180]

    def context(self) -> HealthCycleContext:
        return self._last_snapshot

    def source_snapshot(self, source: str) -> dict[str, object] | None:
        """Return bounded, non-secret current state for a configured source."""
        st = self._sources.get(source)
        if st is None:
            return None
        now = dt.datetime.now(dt.timezone.utc)
        observed_at = st.last_success or st.last_failure
        age_seconds = (
            max(0.0, (now - observed_at).total_seconds())
            if observed_at is not None
            else max(0.0, (now - st.observed_since).total_seconds())
        )
        state, reason = _source_status(st, now)
        return {
            "state": state,
            "reason": reason,
            "role": st.cfg.role,
            "observed_at": observed_at,
            "age_seconds": age_seconds,
            "consecutive_failures": st.consecutive_failures,
        }

    def _impaired_sources(self, now: dt.datetime) -> list[_SourceState]:
        impaired: list[_SourceState] = []
        for st in self._sources.values():
            if not st.enabled:
                continue
            stale = False
            reference = st.last_success or st.observed_since
            stale = (now - reference).total_seconds() > max(1, st.cfg.stale_after_seconds)
            failed = st.consecutive_failures >= max(1, st.cfg.failure_threshold)
            if stale or failed:
                impaired.append(st)
        return impaired

    def _compute_mode(self, impaired: list[_SourceState]) -> tuple[str, tuple[str, ...]]:
        if not impaired:
            return "normal", ()
        reasons = tuple(f"{st.cfg.name}:{st.cfg.role}" for st in impaired)
        alert_impaired = [st for st in impaired if st.cfg.role in {"alert", "alert_redundant"}]
        primary_alert_impaired = [st for st in impaired if st.cfg.role == "alert" or st.cfg.critical]
        forecast_impaired = [st for st in impaired if st.cfg.role in {"forecast", "observation"}]
        enabled_alert_sources = [st for st in self._sources.values() if st.enabled and st.cfg.role in {"alert", "alert_redundant"}]

        if enabled_alert_sources and len(alert_impaired) >= len(enabled_alert_sources):
            return "detached", reasons
        if primary_alert_impaired:
            return "critical_degraded", reasons
        if alert_impaired and forecast_impaired:
            return "degraded", reasons
        if alert_impaired:
            return "source_impaired", reasons
        return "degraded", reasons

    def _message_for_mode(self, mode: str) -> str:
        if mode == "source_impaired":
            return self.cfg.source_impaired_message
        if mode == "critical_degraded":
            return self.cfg.critical_message
        if mode == "detached":
            return self.cfg.detached_message
        if mode == "degraded":
            return self.cfg.degraded_message
        return ""

    async def evaluate(self) -> HealthCycleContext:
        if not self.cfg.enabled:
            self._last_snapshot = HealthCycleContext()
            return self._last_snapshot
        async with self._lock:
            now = dt.datetime.now(dt.timezone.utc)
            impaired = self._impaired_sources(now)
            new_mode, reasons = self._compute_mode(impaired)

            # Hysteresis: once degraded, hold that state briefly to avoid cycle flaps.
            if self._mode != new_mode:
                if self._mode != "normal" and new_mode == "normal":
                    held = (now - self._mode_since).total_seconds()
                    if held < max(0, self.cfg.min_hold_seconds):
                        new_mode = self._mode
                        reasons = self._last_snapshot.reasons
                if self._mode != new_mode:
                    log.warning("Health mode changed; old=%s new=%s reasons=%s", self._mode, new_mode, ",".join(reasons) or "-")
                    self._mode = new_mode
                    self._mode_since = now

            notice = self._message_for_mode(self._mode)
            status_line = ""
            if self._mode != "normal":
                status_line = f"Data feed status: {self._mode.replace('_', ' ')}. Affected sources: {', '.join(reasons) or 'unknown'}."
            self._last_snapshot = HealthCycleContext(
                mode=self._mode,
                notice=notice,
                status_line=status_line,
                detached_loop_only=(self._mode == "detached" and self.cfg.detached_loop_only),
                reasons=tuple(reasons),
            )
            return self._last_snapshot

    async def _run_probe_once(self, source: str, probe: Probe) -> None:
        try:
            await probe()
            self.mark_success(source)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.mark_failure(source, exc)
            log.warning("Health probe failed; source=%s error=%s", source, exc)

    async def run_forever(self, on_change: Callable[[HealthCycleContext], None] | None = None) -> None:
        if not self.cfg.enabled:
            log.info("Health state machine disabled")
            return
        log.info("Health state machine starting (check_interval=%ss)", self.cfg.check_interval_seconds)
        previous = self.context()
        while True:
            for source, probe in list(self._probes.items()):
                await self._run_probe_once(source, probe)
            current = await self.evaluate()
            if on_change and current.mode != previous.mode:
                try:
                    on_change(current)
                except Exception:
                    log.exception("Health on_change callback failed")
            previous = current
            await asyncio.sleep(max(5, int(self.cfg.check_interval_seconds)))
