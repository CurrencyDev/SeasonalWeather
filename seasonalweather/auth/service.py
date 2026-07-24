from __future__ import annotations

import datetime as dt
import ipaddress
import logging
import posixpath
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, NoReturn

from .credentials import (
    generate_access_token,
    generate_client_credential,
    parse_access_token,
    parse_client_credential,
    secret_digest,
    verify_secret,
)
from .models import AccessPrincipal, AccessTokenRecord, AuthenticatedClient, ClientRecord, to_iso, utc_now
from .policy import KNOWN_API_SCOPES, SCOPE_RE, scopes_are_write_capable
from .repository import AuthenticationRepository

log = logging.getLogger("seasonalweather.auth")


class AuthenticationError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(message)


@dataclass(frozen=True)
class IssuedCredential:
    client: ClientRecord
    credential: str


@dataclass(frozen=True)
class IssuedAccessToken:
    access_token: str
    expires_in: int
    scopes: tuple[str, ...]


def normalize_scopes(values: Iterable[str]) -> frozenset[str]:
    raw = list(values)
    if not raw:
        raise AuthenticationError("invalid_scope", "At least one scope is required.", status_code=400)
    if len(raw) != len(set(raw)):
        raise AuthenticationError("invalid_scope", "Scope values must not be duplicated.", status_code=400)
    if any(
        not isinstance(value, str)
        or value != value.strip()
        or not SCOPE_RE.fullmatch(value)
        or value not in KNOWN_API_SCOPES
        for value in raw
    ):
        raise AuthenticationError("invalid_scope", "One or more scope values are invalid.", status_code=400)
    return frozenset(raw)


def _basic_route_prefix_valid(value: Any) -> bool:
    forbidden = ("?", "#", "\\", "//", "*")
    return (
        isinstance(value, str)
        and value.startswith("/")
        and value == value.strip()
        and not any(marker in value for marker in forbidden)
    )


def normalize_route_prefix(value: str) -> str:
    if not _basic_route_prefix_valid(value):
        raise AuthenticationError("invalid_route_prefix", "A route prefix is invalid.", status_code=400)
    parts = value.split("/")
    if any(part in {".", ".."} for part in parts):
        raise AuthenticationError("invalid_route_prefix", "A route prefix is invalid.", status_code=400)
    normalized = posixpath.normpath(value)
    if normalized != value.rstrip("/") and value != "/":
        raise AuthenticationError("invalid_route_prefix", "A route prefix is not canonical.", status_code=400)
    return normalized


def route_is_allowed(path: str, prefixes: tuple[str, ...], unrestricted: bool) -> bool:
    if unrestricted:
        return True
    return any(prefix == "/" or path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes)


def normalize_cidrs(values: Iterable[str]) -> tuple[str, ...]:
    raw = list(values)
    if not raw:
        raise AuthenticationError("invalid_cidr", "At least one client CIDR is required.", status_code=400)
    try:
        networks = [ipaddress.ip_network(value, strict=True) for value in raw]
    except (TypeError, ValueError) as exc:
        raise AuthenticationError("invalid_cidr", "One or more client CIDRs are invalid.", status_code=400) from exc
    canonical = [str(network) for network in networks]
    if len(canonical) != len(set(canonical)):
        raise AuthenticationError("invalid_cidr", "Client CIDRs must not be duplicated.", status_code=400)
    return tuple(sorted(canonical))


def _source_allowed(source_ip: str, cidrs: tuple[str, ...]) -> bool:
    try:
        address = ipaddress.ip_address(source_ip)
    except ValueError:
        return False
    return any(address in ipaddress.ip_network(cidr) for cidr in cidrs)


class AuthenticationService:
    def __init__(
        self,
        repository: AuthenticationRepository,
        policy: Any,
        *,
        clock: Any = utc_now,
    ) -> None:
        self.repository = repository
        self.policy = policy
        self.clock = clock
        self._last_used_lock = threading.Lock()
        self._last_used_persisted: dict[tuple[str, str], dt.datetime] = {}

    def create_client(
        self,
        *,
        subject: str,
        scopes: Iterable[str],
        route_prefixes: Iterable[str] = (),
        unrestricted_routes: bool = False,
        cidrs: Iterable[str],
        expires_at: dt.datetime | None = None,
        actor: str = "local-cli",
    ) -> IssuedCredential:
        now = self.clock()
        clean_subject = self._normalize_subject(subject)
        clean_scopes, clean_prefixes, clean_cidrs = self._normalize_client_policy(
            scopes, route_prefixes, unrestricted_routes, cidrs
        )
        expires_at = self._normalize_expiration(expires_at, now)
        client_id, secret, raw_credential = generate_client_credential()
        client = self.repository.create_client(
            client_id=client_id,
            subject=clean_subject,
            verifier_digest=secret_digest(secret),
            scopes=clean_scopes,
            route_prefixes=clean_prefixes,
            unrestricted_routes=unrestricted_routes,
            cidrs=clean_cidrs,
            expires_at=to_iso(expires_at),
            now=to_iso(now) or "",
            actor=actor,
        )
        return IssuedCredential(client, raw_credential)

    @staticmethod
    def _normalize_subject(subject: str) -> str:
        value = subject.strip() if isinstance(subject, str) else ""
        if not value or len(value) > 128:
            raise AuthenticationError("invalid_subject", "Client subject is invalid.", status_code=400)
        return value

    @staticmethod
    def _normalize_client_policy(
        scopes: Iterable[str],
        route_prefixes: Iterable[str],
        unrestricted_routes: bool,
        cidrs: Iterable[str],
    ) -> tuple[frozenset[str], tuple[str, ...], tuple[str, ...]]:
        raw_prefixes = list(route_prefixes)
        if unrestricted_routes == bool(raw_prefixes):
            raise AuthenticationError(
                "invalid_route_policy",
                "Select either explicit route prefixes or unrestricted routes.",
                status_code=400,
            )
        prefixes = tuple(sorted({normalize_route_prefix(value) for value in raw_prefixes}))
        return normalize_scopes(scopes), prefixes, normalize_cidrs(cidrs)

    @staticmethod
    def _normalize_expiration(value: dt.datetime | None, now: dt.datetime) -> dt.datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None or value <= now:
            raise AuthenticationError("invalid_expiration", "Client expiration must be in the future.", status_code=400)
        return value.astimezone(dt.UTC)

    def list_clients(self) -> list[ClientRecord]:
        return self.repository.list_clients()

    def show_client(self, client_id: str) -> ClientRecord:
        client = self.repository.get_client(client_id)
        if client is None:
            raise AuthenticationError("client_not_found", "Client was not found.", status_code=404)
        return client

    def rotate_client(self, client_id: str, *, actor: str = "local-cli") -> IssuedCredential:
        public_id, secret, credential = generate_client_credential()
        if public_id == client_id:
            raise RuntimeError("secure random client identifier collision")
        credential = f"swc_{client_id}.{secret}"
        client = self.repository.rotate_client(
            client_id,
            verifier_digest=secret_digest(secret),
            now=to_iso(self.clock()) or "",
            actor=actor,
        )
        if client is None:
            raise AuthenticationError("client_not_found", "Active client was not found.", status_code=404)
        return IssuedCredential(client, credential)

    def disable_client(self, client_id: str, *, actor: str = "local-cli") -> ClientRecord:
        return self._set_enabled(client_id, enabled=False, actor=actor)

    def enable_client(self, client_id: str, *, actor: str = "local-cli") -> ClientRecord:
        return self._set_enabled(client_id, enabled=True, actor=actor)

    def _set_enabled(self, client_id: str, *, enabled: bool, actor: str) -> ClientRecord:
        client = self.repository.set_client_enabled(
            client_id,
            enabled=enabled,
            now=to_iso(self.clock()) or "",
            actor=actor,
        )
        if client is None:
            raise AuthenticationError("client_not_found", "Active client was not found.", status_code=404)
        return client

    def revoke_client(self, client_id: str, *, actor: str = "local-cli") -> ClientRecord:
        client = self.repository.revoke_client(
            client_id,
            now=to_iso(self.clock()) or "",
            actor=actor,
        )
        if client is None:
            raise AuthenticationError("client_not_found", "Client was not found.", status_code=404)
        return client

    def authenticate_client(self, raw_credential: str, *, source_ip: str) -> tuple[ClientRecord, AuthenticatedClient]:
        parsed = parse_client_credential(raw_credential)
        if parsed is None:
            self._invalid_client()
            raise AssertionError("unreachable")
        client = self.repository.get_client(parsed.public_id)
        now = self.clock()
        if (
            client is None
            or client.verifier_algorithm != "sha256"
            or not verify_secret(parsed.secret, client.verifier_digest)
            or not client.enabled
            or client.revoked_at is not None
            or (client.expires_at is not None and client.expires_at <= now)
        ):
            self._invalid_client()
        if not _source_allowed(source_ip, client.cidrs):
            raise AuthenticationError("client_policy_denied", "Client policy denied this request.", status_code=403)
        return client, AuthenticatedClient(client.client_id, client.subject, source_ip)

    @staticmethod
    def _invalid_client() -> NoReturn:
        raise AuthenticationError("invalid_client", "Client authentication failed.", status_code=401)

    def issue_token(
        self,
        *,
        client_credential: str,
        source_ip: str,
        requested_scopes: Iterable[str] | None = None,
        requested_ttl: int | None = None,
        request_id: str | None = None,
    ) -> IssuedAccessToken:
        client, caller = self.authenticate_client(client_credential, source_ip=source_ip)
        scopes = self._requested_scopes(client, requested_scopes)
        write_capable = scopes_are_write_capable(scopes)
        now = self.clock()
        ttl = self._requested_ttl(requested_ttl, write_capable)
        ttl, expires_at = self._token_expiration(client, now, ttl)
        token_id, secret, raw_token = generate_access_token()
        self.repository.issue_token(
            token_id=token_id,
            client=client,
            verifier_digest=secret_digest(secret),
            scopes=scopes,
            issued_at=to_iso(now) or "",
            expires_at=to_iso(expires_at) or "",
            write_capable=write_capable,
            actor=caller.subject,
            source_ip=source_ip,
            request_id=request_id,
        )
        return IssuedAccessToken(raw_token, ttl, tuple(sorted(scopes)))

    @staticmethod
    def _requested_scopes(client: ClientRecord, requested_scopes: Iterable[str] | None) -> frozenset[str]:
        scopes = client.scopes if requested_scopes is None else normalize_scopes(requested_scopes)
        if not scopes.issubset(client.scopes) and "*" not in client.scopes:
            raise AuthenticationError("scope_not_allowed", "Requested scopes exceed client policy.", status_code=403)
        return scopes

    def _requested_ttl(self, requested_ttl: int | None, write_capable: bool) -> int:
        ttl = self.policy.default_ttl_seconds if requested_ttl is None else requested_ttl
        maximum = self.policy.maximum_write_ttl_seconds if write_capable else self.policy.maximum_read_ttl_seconds
        valid_type = isinstance(ttl, int) and not isinstance(ttl, bool)
        if not valid_type or not self.policy.minimum_ttl_seconds <= ttl <= maximum:
            raise AuthenticationError(
                "invalid_token_ttl", "Requested access-token TTL is not permitted.", status_code=400
            )
        return ttl

    def _token_expiration(self, client: ClientRecord, now: dt.datetime, ttl: int) -> tuple[int, dt.datetime]:
        expires_at = now + dt.timedelta(seconds=ttl)
        if client.expires_at is None or client.expires_at >= expires_at:
            return ttl, expires_at
        effective = int((client.expires_at - now).total_seconds())
        if effective < self.policy.minimum_ttl_seconds:
            raise AuthenticationError("client_expiring", "Client expires too soon to issue a token.", status_code=403)
        return effective, client.expires_at

    def revoke_token(
        self,
        *,
        client_credential: str,
        target_token: str,
        source_ip: str,
        request_id: str | None = None,
    ) -> None:
        client, caller = self.authenticate_client(client_credential, source_ip=source_ip)
        parsed = parse_access_token(target_token)
        if parsed is None:
            raise AuthenticationError("invalid_revocation_target", "Revocation target is malformed.", status_code=400)
        token = self.repository.get_token(parsed.public_id)
        if (
            token is None
            or token.client_id != client.client_id
            or token.verifier_algorithm != "sha256"
            or not verify_secret(parsed.secret, token.verifier_digest)
        ):
            return
        self.repository.revoke_owned_token(
            token_id=parsed.public_id,
            client_id=client.client_id,
            now=to_iso(self.clock()) or "",
            actor=caller.subject,
            source_ip=source_ip,
            request_id=request_id,
        )

    def authorize_access(self, raw_token: str, *, source_ip: str, path: str) -> AccessPrincipal:
        token = self._valid_token(raw_token)
        client = self._valid_token_client(token)
        self._enforce_access_policy(client, source_ip, path)
        self._record_last_used(client.client_id, token.token_id)
        return AccessPrincipal(client.client_id, token.token_id, client.subject, token.scopes, source_ip)

    def _valid_token(self, raw_token: str) -> AccessTokenRecord:
        parsed = parse_access_token(raw_token)
        if parsed is None:
            self._invalid_token()
        token = self.repository.get_token(parsed.public_id)
        now = self.clock()
        if (
            token is None
            or token.verifier_algorithm != "sha256"
            or not verify_secret(parsed.secret, token.verifier_digest)
            or token.revoked_at is not None
            or token.expires_at <= now
        ):
            self._invalid_token()
        return token

    def _valid_token_client(self, token: AccessTokenRecord) -> ClientRecord:
        client = self.repository.get_client(token.client_id)
        now = self.clock()
        if (
            client is None
            or not client.enabled
            or client.revoked_at is not None
            or (client.expires_at is not None and client.expires_at <= now)
            or token.client_generation != client.generation
        ):
            self._invalid_token()
        return client

    @staticmethod
    def _enforce_access_policy(client: ClientRecord, source_ip: str, path: str) -> None:
        if not _source_allowed(source_ip, client.cidrs):
            raise AuthenticationError("client_policy_denied", "Client policy denied this request.", status_code=403)
        if not route_is_allowed(path, client.route_prefixes, client.unrestricted_routes):
            raise AuthenticationError("client_policy_denied", "Client policy denied this request.", status_code=403)

    def _record_last_used(self, client_id: str, token_id: str) -> None:
        now = self.clock()
        key = (client_id, token_id)
        with self._last_used_lock:
            previous = self._last_used_persisted.get(key)
            if previous is not None and now - previous < dt.timedelta(seconds=60):
                return
        threshold = now - dt.timedelta(seconds=60)
        try:
            self.repository.coalesce_last_used(
                client_id=client_id,
                token_id=token_id,
                now=to_iso(now) or "",
                threshold=to_iso(threshold) or "",
            )
        except Exception:
            log.exception("Authentication last-used persistence failed")
        else:
            with self._last_used_lock:
                self._last_used_persisted[key] = now

    @staticmethod
    def _invalid_token() -> NoReturn:
        raise AuthenticationError("invalid_token", "Bearer token was not recognized.", status_code=401)
