from __future__ import annotations

import argparse
import asyncio
import logging

import uvicorn

from .api import create_app
from .config import load_config
from .control import OrchestratorControl
from .main import Orchestrator, _setup_logging


log = logging.getLogger("seasonalweather.api")


async def run_api_server(*, config_path: str, host: str, port: int) -> None:
    cfg = load_config(config_path)
    orch = Orchestrator(cfg)
    control = OrchestratorControl(orch, config_path=config_path)
    app = create_app(control)

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
    _setup_logging()
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
