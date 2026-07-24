from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def to_iso(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


def from_iso(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.UTC)


@dataclass(frozen=True)
class ClientRecord:
    client_id: str
    subject: str
    scopes: frozenset[str]
    route_prefixes: tuple[str, ...]
    unrestricted_routes: bool
    cidrs: tuple[str, ...]
    enabled: bool
    created_at: dt.datetime
    updated_at: dt.datetime
    expires_at: dt.datetime | None
    revoked_at: dt.datetime | None
    last_used_at: dt.datetime | None
    generation: int
    verifier_algorithm: str = field(repr=False)
    verifier_digest: str = field(repr=False)

    def public_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "subject": self.subject,
            "status": "revoked" if self.revoked_at else ("enabled" if self.enabled else "disabled"),
            "scopes": sorted(self.scopes),
            "route_prefixes": list(self.route_prefixes),
            "unrestricted_routes": self.unrestricted_routes,
            "cidrs": list(self.cidrs),
            "created_at": to_iso(self.created_at),
            "updated_at": to_iso(self.updated_at),
            "expires_at": to_iso(self.expires_at),
            "revoked_at": to_iso(self.revoked_at),
            "last_used_at": to_iso(self.last_used_at),
            "generation": self.generation,
        }


@dataclass(frozen=True)
class AccessTokenRecord:
    token_id: str
    client_id: str
    scopes: frozenset[str]
    issued_at: dt.datetime
    expires_at: dt.datetime
    revoked_at: dt.datetime | None
    last_used_at: dt.datetime | None
    write_capable: bool
    client_generation: int
    verifier_algorithm: str = field(repr=False)
    verifier_digest: str = field(repr=False)


@dataclass(frozen=True)
class AuthenticatedClient:
    client_id: str
    subject: str
    source_ip: str


@dataclass(frozen=True)
class AccessPrincipal:
    client_id: str
    token_id: str
    subject: str
    scopes: frozenset[str]
    source_ip: str
