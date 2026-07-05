from __future__ import annotations

import asyncio
import logging
from typing import Any

# Optional NWWS-OI XMPP client (depends on slixmpp)
try:
    from ..nwws.client import NWWSClient
except Exception:  # pragma: no cover
    NWWSClient = None  # type: ignore
from .station_feed_runtime import (
    hydrate_persisted_alerts as _sf_hydrate_persisted_alerts,
    purge_legacy_synthetic_alerts as _sf_purge_legacy_synthetic_alerts,
)

# Optional CAP (api.weather.gov/alerts/active)
try:
    from ..alerts.cap_nws import NwsCapPoller, CapAlertEvent
except Exception:  # pragma: no cover
    NwsCapPoller = None  # type: ignore
    CapAlertEvent = None  # type: ignore

# Optional IPAWS CAP (apps.fema.gov IPAWS Open feed)
try:
    from ..alerts.ipaws_cap import IpawsCapPoller, IpawsCapEvent
except Exception:  # pragma: no cover
    IpawsCapPoller = None  # type: ignore
    IpawsCapEvent = None  # type: ignore

# Optional ERN/GWES SAME monitor (Level 3 source)
try:
    from .ern_gwes import ErnGwesMonitor, ErnSameEvent
except Exception:  # pragma: no cover
    ErnGwesMonitor = None  # type: ignore
    ErnSameEvent = None  # type: ignore


log = logging.getLogger("seasonalweather")


class SeasonalWeatherServiceRuntime:
    """Owns SeasonalWeather task startup and first-exception supervision.

    The orchestrator still owns subsystem construction and source-specific runtimes;
    this class only starts them, registers health probes, and applies the same
    first-failure cancellation semantics previously held by Orchestrator.run().
    """

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    async def run(self) -> None:
        o = self.owner
        work, audio, cache, logs = o._paths()
        for p in (work, audio, cache, logs):
            p.mkdir(parents=True, exist_ok=True)

        await o._wait_for_liquidsoap()
        o._clear_liquidsoap_queues_on_startup()
        o.discord.service_started(
            cap_enabled=o.cfg.cap.enabled,
            ern_enabled=o.cfg.ern.enabled,
            tests_enabled=o.cfg.tests.enabled,
            mode=o.mode,
        )

        # --- Persistent alert state: restore from disk, drop expired ---
        _loaded = 0
        _purged = 0
        try:
            _loaded = o.alert_tracker.load()
            _purged = o.alert_tracker.purge_expired()
            log.info(
                "AlertTracker: loaded %d entries, purged %d expired on startup",
                _loaded, _purged,
            )
        except Exception:
            log.exception("AlertTracker: startup load/purge failed")
        # _TRACKER_DL_
        try:
            o.discord.alerttracker_lifecycle(
                loaded=_loaded,
                purged=_purged,
                active=len(o.alert_tracker.get_cycle_alerts()),
            )
        except Exception:
            pass

        try:
            _sf_removed_legacy = _sf_purge_legacy_synthetic_alerts()
            if _sf_removed_legacy:
                log.info(
                    "Station feed: removed %d legacy synthetic CAP row(s) on startup",
                    _sf_removed_legacy,
                )
            _sf_hydrated = _sf_hydrate_persisted_alerts()
            if _sf_hydrated:
                log.info(
                    "Station feed: hydrated %d persisted row(s) into runtime cache",
                    _sf_hydrated,
                )
        except Exception:
            log.exception("Station feed: persisted-state startup initialization failed")

        tasks: list[asyncio.Task] = []

        async def _health_probe_cap_api() -> None:
            await o.api.active_alerts(o.cycle_builder.alert_areas)

        async def _health_probe_nws_api() -> None:
            await o.api.latest_product_id("HWO", "LWX")

        o.health_state.register_probe("cap_api", _health_probe_cap_api)
        o.health_state.register_probe("nws_api", _health_probe_nws_api)

        if o.cfg.nwws.credentials_defaulted or not o.cfg.nwws.enabled:
            o.health_state.mark_disabled("nwws_oi", "nwws_disabled")
        if not o.cfg.cap.enabled:
            o.health_state.mark_disabled("cap_api", "cap_disabled")

        def _health_changed(_ctx) -> None:
            try:
                o.refresher.trigger_immediate("id", "health", "status")
                o._schedule_cycle_refill("health-state-change")
            except Exception:
                log.exception("Health state change refresh failed")

        tasks.append(asyncio.create_task(o.health_state.run_forever(on_change=_health_changed), name="health_state"))
        o.alert_audio.start(tasks)

        # CycleConductor + SegmentRefresher own routine cycle scheduling.
        tasks.append(asyncio.create_task(o.conductor.run(), name="conductor"))
        tasks.append(asyncio.create_task(o.refresher.run(), name="segment_refresher"))
        tasks.append(asyncio.create_task(o.pns_runtime.run_backfill_loop(), name="pns_api_backfill"))
        tasks.append(asyncio.create_task(o.now_runtime.run(), name="now_cycle_worker"))
        tasks.append(asyncio.create_task(o.now_runtime.run_backfill_loop(), name="now_api_backfill"))

        if o.cfg.nwws.credentials_defaulted:
            log.warning(
                "NWWS-OI disabled because NWWS_JID/NWWS_PASSWORD are unset or still use the example CHANGEME values; "
                "update /etc/seasonalweather/seasonalweather.env to enable NWWS-OI."
            )
        elif not o.cfg.nwws.enabled:
            log.info("NWWS-OI disabled (set nwws.enabled: true in config.yaml to enable)")
        else:
            if NWWSClient is None:
                log.warning("NWWS-OI enabled but nwws.client import failed; NWWS-OI is disabled.")
            else:
                xmpp = NWWSClient(
                    o.jid, o.password, o.nwws_server, o.nwws_port, o.nwws_queue,
                    room_jid=o.cfg.nwws.room,
                    nick=o.cfg.nwws.nick,
                    # TODO: wire stall/reconnect callbacks to o.discord.nwws_stall() / .nwws_reconnected() once NWWSClient exposes them
                    stall_seconds=o.cfg.nwws.resiliency.stall_seconds,
                    muc_confirm_seconds=o.cfg.nwws.resiliency.muc_confirm_seconds,
                    start_wait_seconds=o.cfg.nwws.resiliency.start_wait_seconds,
                    join_wait_seconds=o.cfg.nwws.resiliency.join_wait_seconds,
                    backoff_max_seconds=o.cfg.nwws.resiliency.backoff_max_seconds,
                )
                tasks.append(asyncio.create_task(xmpp.run_forever(), name="nwws_xmpp"))
                tasks.append(asyncio.create_task(o.nwws_runtime.run(), name="nwws_consumer"))
        # CycleConductor runs the cycle continuously.

        if o.cfg.cap.enabled:
            if NwsCapPoller is None or CapAlertEvent is None:
                log.warning("CAP enabled but cap_nws.py import failed; CAP is disabled.")
            else:
                kwargs = dict(
                    out_queue=o.cap_queue,
                    same_fips_allow=o.cfg.service_area.same_fips_all,
                    poll_seconds=o.cfg.cap.poll_seconds,
                    user_agent=o.cfg.cap.user_agent,
                    ledger_path=o.cfg.cap.ledger_path,
                    ledger_max_age_days=o.cfg.cap.ledger_max_age_days,
                    database=o.database,
                )
                url = o.cfg.cap.url.strip()
                if url:
                    kwargs["url"] = url  # type: ignore[assignment]

                cap = NwsCapPoller(**kwargs)  # type: ignore[arg-type]
                tasks.append(asyncio.create_task(cap.run_forever(), name="cap_poller"))
                tasks.append(asyncio.create_task(o.cap_runtime.run(), name="cap_consumer"))
                log.info("CAP ingest enabled (dryrun=%s full=%s voice=%s)", o.cfg.cap.dryrun, o.cfg.cap.full.enabled, o.cfg.cap.voice.enabled)
        else:
            log.info("CAP ingest disabled (set cap.enabled: true in config.yaml to enable)")

        if o.cfg.ipaws.enabled:
            if IpawsCapPoller is None or IpawsCapEvent is None:
                log.warning("IPAWS enabled but ipaws_cap.py import failed; IPAWS is disabled.")
            else:
                ipaws_poller = IpawsCapPoller(
                    out_queue=o.ipaws_queue,
                    same_fips_allow=o.cfg.service_area.same_fips_all,
                    poll_seconds=o.cfg.ipaws.poll_seconds,
                    user_agent=o.cfg.ipaws.user_agent,
                    url=o.cfg.ipaws.url,
                    ledger_path=o.cfg.ipaws.ledger_path,
                    ledger_max_age_days=o.cfg.ipaws.ledger_max_age_days,
                    database=o.database,
                )
                tasks.append(asyncio.create_task(ipaws_poller.run_forever(), name="ipaws_poller"))
                tasks.append(asyncio.create_task(o.ipaws_runtime.run(), name="ipaws_consumer"))
                log.info(
                    "IPAWS ingest enabled (dryrun=%s full_events=%s)",
                    o.cfg.ipaws.dryrun,
                    ",".join(sorted(set(o.cfg.ipaws.full_events))),
                )
        else:
            log.info("IPAWS ingest disabled (set ipaws.enabled: true in config.yaml to enable)")

        if o.cfg.ern.enabled:
            if ErnGwesMonitor is None or ErnSameEvent is None:
                log.warning("ERN enabled but ern_gwes.py import failed; ERN is disabled.")
            else:
                url = o.cfg.ern.url.strip()
                if not url:
                    log.warning("ERN enabled but SEASONAL_ERN_URL is empty; ERN is disabled.")
                else:
                    ern_cfg = o.cfg.ern
                    mon = ErnGwesMonitor(
                        out_queue=o.ern_queue,
                        same_fips_allow=o.cfg.service_area.same_fips_all,
                        url=url,
                        sample_rate=ern_cfg.sample_rate,
                        dedupe_seconds=ern_cfg.dedupe_seconds,
                        trigger_ratio=ern_cfg.trigger_ratio,
                        tail_seconds=ern_cfg.tail_seconds,
                        confidence_min=ern_cfg.confidence_min,
                        name=ern_cfg.name,
                        decoder_backend=ern_cfg.decoder_backend,
                    )
                    tasks.append(asyncio.create_task(mon.run_forever(), name="ern_monitor"))
                    tasks.append(asyncio.create_task(o.ern_relay_runtime.run(), name="ern_consumer"))
                    log.info(
                        "ERN monitor enabled (dryrun=%s url=%s relay=%s decoder=%s)",
                        o.cfg.ern.dryrun,
                        url,
                        o.cfg.ern.relay.enabled,
                        ern_cfg.decoder_backend,
                    )
        else:
            log.info("ERN monitor disabled (set ern.enabled: true in config.yaml to enable)")

        o.tests_runtime.start_scheduler(tasks)

        if o.db_housekeeper is not None:
            tasks.append(asyncio.create_task(o.db_housekeeper.run_forever(), name="database_housekeeping"))

        tasks.append(asyncio.create_task(o.discord.start(), name="discord_log_drain"))
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            exc = t.exception()
            if exc:
                for p in pending:
                    p.cancel()
                raise exc
