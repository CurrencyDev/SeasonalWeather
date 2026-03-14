from __future__ import annotations

import hmac
import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import Depends, Header, HTTPException, Request, status


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


_DEFAULT_ADMIN_SCOPES = frozenset(
    {
        "read:health",
        "read:status",
        "read:alerts",
        "read:config",
        "control:cycle",
        "control:mode",
        "control:tests",
        "control:originate",
        "control:audio",
        "control:config",
    }
)


def _is_loopback(host: str | None) -> bool:
    if host is None:
        return False
    return host in {"127.0.0.1", "::1", "localhost"}


def _load_token_map() -> dict[str, dict[str, Any]]:
    from .main import _APP_CFG  # late import to avoid circular dependency
    secrets = _APP_CFG.secrets if _APP_CFG else None

    raw_json = (secrets.api_tokens_json if secrets else "") or ""
    raw_json = raw_json.strip()
    if raw_json:
        try:
            obj = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("SEASONAL_API_TOKENS_JSON is not valid JSON") from exc
        if not isinstance(obj, dict):
            raise RuntimeError("SEASONAL_API_TOKENS_JSON must be a JSON object")
        out: dict[str, dict[str, Any]] = {}
        for token, meta in obj.items():
            if not isinstance(token, str) or not token:
                continue
            if isinstance(meta, dict):
                subject = str(meta.get("subject") or "api-user")
                scopes_raw = meta.get("scopes") or ["*"]
                scopes = [str(s).strip() for s in scopes_raw if str(s).strip()]
            else:
                subject = "api-user"
                scopes = ["*"]
            out[token] = {"subject": subject, "scopes": scopes}
        return out

    single = (secrets.api_token if secrets else "") or ""
    single = single.strip()
    if not single:
        raise RuntimeError(
            "No API token configured. Set SEASONAL_API_TOKEN or SEASONAL_API_TOKENS_JSON in seasonalweather.env."
        )

    api_cfg = _APP_CFG.api if _APP_CFG else None
    scopes_raw = (api_cfg.scopes if api_cfg else "") or ""
    scopes = [s.strip() for s in scopes_raw.split(",") if s.strip()] if scopes_raw else sorted(_DEFAULT_ADMIN_SCOPES)
    subject = (api_cfg.subject if api_cfg else "local-admin") or "local-admin"
    return {single: {"subject": subject, "scopes": scopes}}


def _authenticate_token(token: str) -> dict[str, Any] | None:
    token_map = _load_token_map()
    for configured, meta in token_map.items():
        if hmac.compare_digest(configured, token):
            return meta
    return None


async def get_api_principal(
    request: Request,
    authorization: str | None = Header(default=None),
) -> ApiPrincipal:
    from .main import _APP_CFG  # late import to avoid circular dependency
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

    meta = _authenticate_token(token.strip())
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_token", "message": "Bearer token was not recognized."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    scopes = frozenset(str(s).strip() for s in (meta.get("scopes") or []) if str(s).strip())
    return ApiPrincipal(subject=str(meta.get("subject") or "api-user"), scopes=scopes, client_host=client_host)


def require_scopes(*scopes: str) -> Callable[..., Any]:
    async def _dependency(principal: ApiPrincipal = Depends(get_api_principal)) -> ApiPrincipal:
        principal.require(*scopes)
        return principal

    return _dependency
