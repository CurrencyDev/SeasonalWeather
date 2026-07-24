from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from collections.abc import Callable
from typing import Any

import uvicorn

from ..auth import AuthenticationRepository, AuthenticationService
from ..config import AuthMode, load_config
from ..control import OrchestratorControl
from ..database.bootstrap import bootstrap_database_from_config
from ..health_service import build_runtime_health_service
from ..lifecycle import Lifecycle, LifecycleState, TaskSupervisor
from ..main import Orchestrator, _setup_logging
from .api import create_app
from .commands import CommandStore

log = logging.getLogger("seasonalweather.api")


class _ControllerOwnedUvicornServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:
        """The SeasonalWeather controller is the sole signal owner."""


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    callback: Callable[[], None],
) -> Callable[[], None]:
    installed: list[signal.Signals] = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, callback)
        installed.append(sig)

    def remove() -> None:
        for sig in installed:
            loop.remove_signal_handler(sig)

    return remove


async def run_api_server(*, config_path: str, host: str, port: int) -> None:
    cfg = load_config(config_path)
    _setup_logging(cfg)
    lifecycle = Lifecycle(cfg.lifecycle)
    supervisor = TaskSupervisor(lifecycle)
    orch = Orchestrator(
        cfg,
        lifecycle=lifecycle,
        supervisor=supervisor,
    )
    control = OrchestratorControl(orch, config_path=config_path)
    db = bootstrap_database_from_config(cfg) if getattr(cfg.database, "enabled", True) else None
    if cfg.api.auth.mode in {AuthMode.EXCHANGE, AuthMode.HYBRID} and db is None:
        raise RuntimeError("Exchange authentication requires the controller SQLite database.")
    auth_service = (
        AuthenticationService(
            AuthenticationRepository(db),
            cfg.api.auth.exchange,
        )
        if db is not None and cfg.api.auth.mode in {AuthMode.EXCHANGE, AuthMode.HYBRID}
        else None
    )
    command_store = CommandStore(
        database=db,
        lifecycle=lifecycle,
    )
    health_service = build_runtime_health_service(
        orch,
        command_store=command_store,
        auth_service=auth_service,
    )
    app = create_app(
        control,
        store=command_store,
        auth_service=auth_service,
        health_service=health_service,
        lifecycle=lifecycle,
    )

    server = _ControllerOwnedUvicornServer(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            proxy_headers=False,
            forwarded_allow_ips="",
            timeout_graceful_shutdown=max(
                1,
                int(cfg.lifecycle.active_request_seconds),
            ),
        )
    )
    # The controller below is the sole signal owner. Uvicorn remains
    # responsible for its supported active-request drain contract.

    def request_shutdown() -> None:
        lifecycle.request_shutdown()
        server.should_exit = True

    remove_signal_handlers = _install_signal_handlers(
        asyncio.get_running_loop(),
        request_shutdown,
    )
    primary_failure: BaseException | None = None
    try:
        api_task = supervisor.create_task(
            server.serve(),
            name="seasonalweather-api",
            required=True,
        )
        supervisor.create_task(
            orch.run(),
            name="seasonalweather-orchestrator",
            required=True,
        )
        await lifecycle.wait_for_shutdown()
        server.should_exit = True
        if lifecycle.state is LifecycleState.FAILED:
            primary_failure = await supervisor.wait_for_fatal()

        async def shutdown() -> None:
            if not lifecycle.force_requested and not api_task.done():
                await _wait_task_or_force(
                    lifecycle,
                    api_task,
                    timeout=cfg.lifecycle.active_request_seconds,
                )
            if not lifecycle.force_requested:
                alert_idle = await _wait_alert_or_force(
                    lifecycle,
                    orch,
                    timeout=cfg.lifecycle.tts_stop_seconds,
                )
                if alert_idle is False:
                    log.warning("alert_audio_drain_timeout")
            if not lifecycle.force_requested:
                publication_idle = await _wait_publication_or_force(
                    lifecycle,
                    orch,
                    timeout=cfg.lifecycle.publication_seconds,
                )
                if publication_idle is False:
                    log.warning("publication_fence_timeout")
            if lifecycle.state is LifecycleState.DRAINING:
                lifecycle.mark_stopping()
            await supervisor.stop()
            await _close_resources(
                orch=orch,
                database=db,
                timeout_seconds=cfg.lifecycle.resource_close_seconds,
                tts_timeout_seconds=cfg.lifecycle.tts_stop_seconds,
            )

        try:
            await asyncio.wait_for(
                shutdown(),
                timeout=cfg.lifecycle.total_seconds,
            )
        except TimeoutError:
            log.error("controller_shutdown_deadline_exceeded")

        if primary_failure is None:
            lifecycle.mark_stopped()
        else:
            raise primary_failure
    finally:
        remove_signal_handlers()


async def _wait_task_or_force(
    lifecycle: Lifecycle,
    task: asyncio.Task[object],
    *,
    timeout: float,
) -> None:
    force_task = asyncio.create_task(
        lifecycle.wait_for_force(),
        name="lifecycle-force-wait",
    )
    try:
        await asyncio.wait(
            {task, force_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        force_task.cancel()
        await asyncio.gather(force_task, return_exceptions=True)


async def _wait_publication_or_force(
    lifecycle: Lifecycle,
    orch: Orchestrator,
    *,
    timeout: float,
) -> bool | None:
    publication_task = asyncio.create_task(
        orch.publication_fence.wait_idle(timeout),
        name="publication-fence-wait",
    )
    force_task = asyncio.create_task(
        lifecycle.wait_for_force(),
        name="lifecycle-force-wait",
    )
    try:
        done, _ = await asyncio.wait(
            {publication_task, force_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if publication_task in done:
            return publication_task.result()
        return None
    finally:
        for task in (publication_task, force_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(
            publication_task,
            force_task,
            return_exceptions=True,
        )


async def _wait_alert_or_force(
    lifecycle: Lifecycle,
    orch: Orchestrator,
    *,
    timeout: float,
) -> bool | None:
    alert_task = asyncio.create_task(
        orch.alert_audio.wait_idle(timeout),
        name="alert-audio-drain-wait",
    )
    force_task = asyncio.create_task(
        lifecycle.wait_for_force(),
        name="lifecycle-force-wait",
    )
    try:
        done, _ = await asyncio.wait(
            {alert_task, force_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if alert_task in done:
            return alert_task.result()
        return None
    finally:
        for task in (alert_task, force_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(
            alert_task,
            force_task,
            return_exceptions=True,
        )


async def _close_resources(
    *,
    orch: Orchestrator,
    database,
    timeout_seconds: float,
    tts_timeout_seconds: float,
) -> None:
    try:
        await asyncio.wait_for(
            orch.api.aclose(),
            timeout=timeout_seconds,
        )
    except Exception:
        log.warning("controller_resource_close_failed resource=nws_api")
    if database is not None:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(database.checkpoint),
                timeout=timeout_seconds,
            )
        except Exception:
            log.warning("controller_resource_close_failed resource=sqlite")
    try:
        loop: Any = asyncio.get_running_loop()
        await loop.shutdown_default_executor(timeout=tts_timeout_seconds)
    except Exception:
        log.warning("controller_resource_close_failed resource=executor")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=("Run the SeasonalWeather orchestrator with the localhost control API."))
    ap.add_argument("--config", default="/etc/seasonalweather/config.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9080)
    args = ap.parse_args(argv)

    asyncio.run(
        run_api_server(
            config_path=args.config,
            host=args.host,
            port=args.port,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
