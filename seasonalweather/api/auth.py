from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import Depends, Header, HTTPException, Request, status

from ..config import AuthMode, StaticCredentialConfig


@dataclass(frozen=True)
class ApiPrincipal:
    subject: str
    scopes: frozenset[str]
    client_host: str | None

    def has_scope(self, required: str) -> bool:
        return "*" in self.scopes or required in self.scopes

    def require(self, *required_scopes: str) -> None:
        missing = [scope for scope in required_scopes if not self.has_scope(scope)]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "insufficient_scope",
                    "message": "Missing required API scope.",
                    "details": {"missing_scopes": missing},
                },
            )


@dataclass(frozen=True)
class RouteAuthPolicy:
    scopes: frozenset[str] = frozenset()

    @property
    def public(self) -> bool:
        return not self.scopes


def _protected(*scopes: str) -> RouteAuthPolicy:
    return RouteAuthPolicy(frozenset(scopes))


PUBLIC_ROUTE = RouteAuthPolicy()

ROUTE_AUTH_POLICIES: dict[tuple[str, str], RouteAuthPolicy] = {
    ("GET", "/openapi.json"): PUBLIC_ROUTE,
    ("HEAD", "/openapi.json"): PUBLIC_ROUTE,
    ("GET", "/docs"): PUBLIC_ROUTE,
    ("HEAD", "/docs"): PUBLIC_ROUTE,
    ("GET", "/docs/oauth2-redirect"): PUBLIC_ROUTE,
    ("HEAD", "/docs/oauth2-redirect"): PUBLIC_ROUTE,
    ("GET", "/redoc"): PUBLIC_ROUTE,
    ("HEAD", "/redoc"): PUBLIC_ROUTE,
    ("GET", "/healthz"): _protected("read:health"),
    ("GET", "/v1/health"): _protected("read:health"),
    ("GET", "/v1/status"): _protected("read:status"),
    ("GET", "/v1/handled-alerts"): PUBLIC_ROUTE,
    ("GET", "/v1/station-feed"): _protected("read:alerts"),
    ("GET", "/v1/config/summary"): _protected("read:config"),
    ("GET", "/v1/commands/{command_id}"): _protected("read:status"),
    ("POST", "/v1/cycle/rebuild"): _protected("control:cycle"),
    ("POST", "/v1/mode/heightened"): _protected("control:mode"),
    ("DELETE", "/v1/mode/heightened"): _protected("control:mode"),
    ("POST", "/v1/tests/originate"): _protected("control:tests"),
    ("POST", "/v1/uploads/audio"): _protected("control:audio"),
    ("POST", "/v1/inserts/text"): _protected("control:inserts"),
    ("POST", "/v1/inserts/audio"): _protected("control:inserts"),
    ("GET", "/v1/inserts"): _protected("control:inserts"),
    ("GET", "/v1/inserts/{insert_id}"): _protected("control:inserts"),
    ("DELETE", "/v1/inserts/{insert_id}"): _protected("control:inserts"),
    ("POST", "/v1/originate/text"): _protected("control:originate"),
    ("POST", "/v1/originate/audio"): _protected("control:originate"),
    ("POST", "/v1/config/reload"): _protected("control:config"),
    ("GET", "/v1/events"): _protected("read:status"),
}


def _is_loopback(host: str | None) -> bool:
    if host is None:
        return False
    return host in {"127.0.0.1", "::1", "localhost"}


def _load_static_credentials() -> tuple[StaticCredentialConfig, ...]:
    from ..main import _APP_CFG  # late import to avoid circular dependency

    if _APP_CFG is None:
        return ()
    auth = _APP_CFG.api.auth
    if auth.mode is not AuthMode.STATIC:
        return ()
    return auth.credentials


def _authenticate_token(token: str) -> StaticCredentialConfig | None:
    for credential in _load_static_credentials():
        if hmac.compare_digest(credential.token, token):
            return credential
    return None


async def get_api_principal(
    request: Request,
    authorization: str | None = Header(default=None),
) -> ApiPrincipal:
    from ..main import _APP_CFG  # late import to avoid circular dependency
    allow_remote = (_APP_CFG.api.allow_remote if _APP_CFG else False)
    client_host = request.client.host if request.client else None
    if not allow_remote and not _is_loopback(client_host):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "remote_client_forbidden",
                "message": "This API only accepts loopback clients unless SEASONAL_API_ALLOW_REMOTE is enabled.",
                "details": {"client_host": client_host},
            },
        )

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "missing_authorization", "message": "Missing Authorization header."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_authorization", "message": "Authorization must be a Bearer token."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    credential = _authenticate_token(token.strip())
    if credential is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_token", "message": "Bearer token was not recognized."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    return ApiPrincipal(
        subject=credential.subject,
        scopes=credential.scopes,
        client_host=client_host,
    )


def require_scopes(*scopes: str) -> Callable[..., Any]:
    async def _dependency(principal: ApiPrincipal = Depends(get_api_principal)) -> ApiPrincipal:
        principal.require(*scopes)
        return principal

    return _dependency


def route_auth_policy(method: str, path: str) -> RouteAuthPolicy:
    key = (method.strip().upper(), path)
    try:
        return ROUTE_AUTH_POLICIES[key]
    except KeyError as exc:
        raise RuntimeError(f"No authentication policy is declared for {key[0]} {path}.") from exc


def require_route_policy(method: str, path: str) -> Callable[..., Any]:
    policy = route_auth_policy(method, path)
    if policy.public:
        raise RuntimeError(f"Public route {method.strip().upper()} {path} must not install an auth dependency.")
    dependency = require_scopes(*sorted(policy.scopes))
    dependency.__dict__["__seasonalweather_auth_policy__"] = policy
    return dependency
