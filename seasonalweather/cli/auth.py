from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from typing import Any

from ..auth import AuthenticationError, AuthenticationRepository, AuthenticationService
from ..config import load_config
from ..database.bootstrap import bootstrap_database_from_config


def _json(payload: Any) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seasonalweather auth")
    parser.add_argument("--config", default="/etc/seasonalweather/config.yaml")
    parser.add_argument("--json", action="store_true", dest="machine_readable")
    auth = parser.add_subparsers(dest="resource", required=True)
    client = auth.add_parser("client")
    commands = client.add_subparsers(dest="action", required=True)

    create = commands.add_parser("create")
    create.add_argument("--subject", required=True)
    create.add_argument("--scope", action="append", required=True, dest="scopes")
    route = create.add_mutually_exclusive_group(required=True)
    route.add_argument("--route-prefix", action="append", dest="route_prefixes")
    route.add_argument("--unrestricted-routes", action="store_true")
    create.add_argument("--cidr", action="append", required=True, dest="cidrs")
    create.add_argument("--expires-at")

    commands.add_parser("list")
    show = commands.add_parser("show")
    show.add_argument("client_id")
    for name in ("rotate", "disable", "enable", "revoke"):
        command = commands.add_parser(name)
        command.add_argument("client_id")
    return parser


def _expiration(value: str | None) -> dt.datetime | None:
    if value is None:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuthenticationError("invalid_expiration", "Client expiration is invalid.", status_code=400) from exc
    return parsed


def _service(config_path: str) -> AuthenticationService:
    cfg = load_config(config_path)
    if not cfg.database.enabled:
        raise AuthenticationError(
            "auth_store_unavailable",
            "Authentication administration requires the controller SQLite database.",
            status_code=503,
        )
    database = bootstrap_database_from_config(cfg)
    return AuthenticationService(AuthenticationRepository(database), cfg.api.auth.exchange)


def _emit_client(record: Any, *, machine_readable: bool) -> None:
    payload = record.public_dict()
    if machine_readable:
        _json(payload)
        return
    for key, value in payload.items():
        rendered = json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
        print(f"{key}: {rendered}")


def _emit_issued(issued: Any, machine_readable: bool) -> None:
    payload = issued.client.public_dict() | {"client_credential": issued.credential}
    if machine_readable:
        _json(payload)
        return
    _emit_client(issued.client, machine_readable=False)
    print(f"client_credential: {issued.credential}")


def _create(service: AuthenticationService, args: Any) -> None:
    issued = service.create_client(
        subject=args.subject,
        scopes=args.scopes,
        route_prefixes=args.route_prefixes or (),
        unrestricted_routes=args.unrestricted_routes,
        cidrs=args.cidrs,
        expires_at=_expiration(args.expires_at),
    )
    _emit_issued(issued, args.machine_readable)


def _list(service: AuthenticationService, args: Any) -> None:
    records = [record.public_dict() for record in service.list_clients()]
    if args.machine_readable:
        _json({"clients": records})
        return
    for record in records:
        print(f"{record['client_id']}\t{record['status']}\t{record['subject']}")


def _show(service: AuthenticationService, args: Any) -> None:
    _emit_client(service.show_client(args.client_id), machine_readable=args.machine_readable)


def _rotate(service: AuthenticationService, args: Any) -> None:
    _emit_issued(service.rotate_client(args.client_id), args.machine_readable)


def _state_change(service: AuthenticationService, args: Any) -> None:
    operation = {
        "disable": service.disable_client,
        "enable": service.enable_client,
        "revoke": service.revoke_client,
    }[args.action]
    _emit_client(operation(args.client_id), machine_readable=args.machine_readable)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        service = _service(args.config)
        handler = {
            "create": _create,
            "list": _list,
            "show": _show,
            "rotate": _rotate,
            "disable": _state_change,
            "enable": _state_change,
            "revoke": _state_change,
        }[args.action]
        handler(service, args)
    except AuthenticationError as exc:
        print(f"auth error [{exc.code}]: {exc}", file=sys.stderr)
        return 2
    return 0
