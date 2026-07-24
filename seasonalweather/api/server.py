from __future__ import annotations

import argparse
import asyncio
import logging

import uvicorn

from ..auth import AuthenticationRepository, AuthenticationService
from ..config import AuthMode, load_config
from ..control import OrchestratorControl
from ..database.bootstrap import bootstrap_database_from_config
from ..health_service import build_runtime_health_service
from ..main import Orchestrator, _setup_logging
from .api import create_app
from .commands import CommandStore

log = logging.getLogger("seasonalweather.api")


async def run_api_server(*, config_path: str, host: str, port: int) -> None:
    cfg = load_config(config_path)
    _setup_logging(cfg)
    orch = Orchestrator(cfg)
    control = OrchestratorControl(orch, config_path=config_path)
    db = bootstrap_database_from_config(cfg) if getattr(cfg.database, "enabled", True) else None
    if cfg.api.auth.mode in {AuthMode.EXCHANGE, AuthMode.HYBRID} and db is None:
        raise RuntimeError("Exchange authentication requires the controller SQLite database.")
    auth_service = (
        AuthenticationService(AuthenticationRepository(db), cfg.api.auth.exchange)
        if db is not None and cfg.api.auth.mode in {AuthMode.EXCHANGE, AuthMode.HYBRID}
        else None
    )
    command_store = CommandStore(database=db)
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
    )

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            proxy_headers=False,
            forwarded_allow_ips="",
        )
    )

    async with asyncio.TaskGroup() as tg:
        tg.create_task(orch.run(), name="seasonalweather-orchestrator")
        tg.create_task(server.serve(), name="seasonalweather-api")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the SeasonalWeather orchestrator with the localhost control API.")
    ap.add_argument("--config", default="/etc/seasonalweather/config.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9080)
    args = ap.parse_args(argv)

    try:
        asyncio.run(run_api_server(config_path=args.config, host=args.host, port=args.port))
    except KeyboardInterrupt:
        log.info("SeasonalWeather API server interrupted by keyboard signal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
