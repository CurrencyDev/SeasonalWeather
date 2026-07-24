from __future__ import annotations

import json
from typing import Any

from ..database.core import SeasonalDatabase
from .models import AccessTokenRecord, ClientRecord, from_iso


def _json(values: list[str] | tuple[str, ...] | frozenset[str]) -> str:
    return json.dumps(sorted(values), separators=(",", ":"), ensure_ascii=True)


class AuthenticationRepository:
    def __init__(self, database: SeasonalDatabase) -> None:
        self.database = database
        self.database.bootstrap()

    @staticmethod
    def _client(row: Any) -> ClientRecord:
        return ClientRecord(
            client_id=str(row["client_id"]),
            subject=str(row["subject"]),
            scopes=frozenset(json.loads(row["scopes_json"])),
            route_prefixes=tuple(json.loads(row["route_prefixes_json"])),
            unrestricted_routes=bool(row["unrestricted_routes"]),
            cidrs=tuple(json.loads(row["cidrs_json"])),
            enabled=bool(row["enabled"]),
            created_at=from_iso(row["created_at"]),  # type: ignore[arg-type]
            updated_at=from_iso(row["updated_at"]),  # type: ignore[arg-type]
            expires_at=from_iso(row["expires_at"]),
            revoked_at=from_iso(row["revoked_at"]),
            last_used_at=from_iso(row["last_used_at"]),
            generation=int(row["generation"]),
            verifier_algorithm=str(row["verifier_algorithm"]),
            verifier_digest=str(row["verifier_digest"]),
        )

    @staticmethod
    def _token(row: Any) -> AccessTokenRecord:
        return AccessTokenRecord(
            token_id=str(row["token_id"]),
            client_id=str(row["client_id"]),
            scopes=frozenset(json.loads(row["scopes_json"])),
            issued_at=from_iso(row["issued_at"]),  # type: ignore[arg-type]
            expires_at=from_iso(row["expires_at"]),  # type: ignore[arg-type]
            revoked_at=from_iso(row["revoked_at"]),
            last_used_at=from_iso(row["last_used_at"]),
            write_capable=bool(row["write_capable"]),
            client_generation=int(row["client_generation"]),
            verifier_algorithm=str(row["verifier_algorithm"]),
            verifier_digest=str(row["verifier_digest"]),
        )

    @staticmethod
    def _audit(
        conn: Any,
        *,
        event_type: str,
        occurred_at: str,
        client_id: str,
        token_id: str | None = None,
        actor: str,
        source_ip: str | None = None,
        request_id: str | None = None,
        reason_code: str = "success",
    ) -> None:
        conn.execute(
            """
            INSERT INTO auth_audit_events (
                event_type, outcome, occurred_at, client_id, token_id,
                actor, source_ip, request_id, reason_code
            ) VALUES (?, 'success', ?, ?, ?, ?, ?, ?, ?)
            """,
            (event_type, occurred_at, client_id, token_id, actor, source_ip, request_id, reason_code),
        )

    def create_client(
        self,
        *,
        client_id: str,
        subject: str,
        verifier_digest: str,
        scopes: frozenset[str],
        route_prefixes: tuple[str, ...],
        unrestricted_routes: bool,
        cidrs: tuple[str, ...],
        expires_at: str | None,
        now: str,
        actor: str,
    ) -> ClientRecord:
        with self.database.transaction() as conn:
            conn.execute(
                """
                INSERT INTO auth_clients (
                    client_id, subject, verifier_algorithm, verifier_digest,
                    scopes_json, route_prefixes_json, unrestricted_routes,
                    cidrs_json, enabled, created_at, updated_at, expires_at,
                    generation
                ) VALUES (?, ?, 'sha256', ?, ?, ?, ?, ?, 1, ?, ?, ?, 1)
                """,
                (
                    client_id,
                    subject,
                    verifier_digest,
                    _json(scopes),
                    _json(route_prefixes),
                    int(unrestricted_routes),
                    _json(cidrs),
                    now,
                    now,
                    expires_at,
                ),
            )
            self._audit(conn, event_type="client.created", occurred_at=now, client_id=client_id, actor=actor)
            row = conn.execute("SELECT * FROM auth_clients WHERE client_id = ?", (client_id,)).fetchone()
        return self._client(row)

    def get_client(self, client_id: str) -> ClientRecord | None:
        with self.database.connect() as conn:
            row = conn.execute("SELECT * FROM auth_clients WHERE client_id = ?", (client_id,)).fetchone()
        return self._client(row) if row else None

    def list_clients(self) -> list[ClientRecord]:
        with self.database.connect() as conn:
            rows = conn.execute("SELECT * FROM auth_clients ORDER BY client_id").fetchall()
        return [self._client(row) for row in rows]

    def rotate_client(
        self,
        client_id: str,
        *,
        verifier_digest: str,
        now: str,
        actor: str,
    ) -> ClientRecord | None:
        with self.database.transaction() as conn:
            row = conn.execute("SELECT * FROM auth_clients WHERE client_id = ?", (client_id,)).fetchone()
            if row is None or row["revoked_at"] is not None:
                return None
            conn.execute(
                """
                UPDATE auth_clients
                   SET verifier_digest = ?, generation = generation + 1, updated_at = ?
                 WHERE client_id = ?
                """,
                (verifier_digest, now, client_id),
            )
            conn.execute(
                "UPDATE auth_access_tokens SET revoked_at = COALESCE(revoked_at, ?) WHERE client_id = ?",
                (now, client_id),
            )
            self._audit(conn, event_type="client.rotated", occurred_at=now, client_id=client_id, actor=actor)
            updated = conn.execute("SELECT * FROM auth_clients WHERE client_id = ?", (client_id,)).fetchone()
        return self._client(updated)

    def set_client_enabled(
        self,
        client_id: str,
        *,
        enabled: bool,
        now: str,
        actor: str,
    ) -> ClientRecord | None:
        with self.database.transaction() as conn:
            row = conn.execute("SELECT * FROM auth_clients WHERE client_id = ?", (client_id,)).fetchone()
            if row is None or row["revoked_at"] is not None:
                return None
            conn.execute(
                "UPDATE auth_clients SET enabled = ?, updated_at = ? WHERE client_id = ?",
                (int(enabled), now, client_id),
            )
            if not enabled:
                conn.execute(
                    "UPDATE auth_access_tokens SET revoked_at = COALESCE(revoked_at, ?) WHERE client_id = ?",
                    (now, client_id),
                )
            self._audit(
                conn,
                event_type="client.enabled" if enabled else "client.disabled",
                occurred_at=now,
                client_id=client_id,
                actor=actor,
            )
            updated = conn.execute("SELECT * FROM auth_clients WHERE client_id = ?", (client_id,)).fetchone()
        return self._client(updated)

    def revoke_client(self, client_id: str, *, now: str, actor: str) -> ClientRecord | None:
        with self.database.transaction() as conn:
            row = conn.execute("SELECT * FROM auth_clients WHERE client_id = ?", (client_id,)).fetchone()
            if row is None:
                return None
            if row["revoked_at"] is None:
                conn.execute(
                    "UPDATE auth_clients SET enabled = 0, revoked_at = ?, updated_at = ? WHERE client_id = ?",
                    (now, now, client_id),
                )
                conn.execute(
                    "UPDATE auth_access_tokens SET revoked_at = COALESCE(revoked_at, ?) WHERE client_id = ?",
                    (now, client_id),
                )
                self._audit(conn, event_type="client.revoked", occurred_at=now, client_id=client_id, actor=actor)
            updated = conn.execute("SELECT * FROM auth_clients WHERE client_id = ?", (client_id,)).fetchone()
        return self._client(updated)

    def issue_token(
        self,
        *,
        token_id: str,
        client: ClientRecord,
        verifier_digest: str,
        scopes: frozenset[str],
        issued_at: str,
        expires_at: str,
        write_capable: bool,
        actor: str,
        source_ip: str,
        request_id: str | None,
    ) -> AccessTokenRecord:
        with self.database.transaction() as conn:
            current = conn.execute("SELECT * FROM auth_clients WHERE client_id = ?", (client.client_id,)).fetchone()
            if (
                current is None
                or int(current["generation"]) != client.generation
                or not bool(current["enabled"])
                or current["revoked_at"] is not None
            ):
                raise RuntimeError("client state changed during token issuance")
            conn.execute(
                """
                INSERT INTO auth_access_tokens (
                    token_id, client_id, verifier_algorithm, verifier_digest,
                    scopes_json, issued_at, expires_at, write_capable,
                    client_generation
                ) VALUES (?, ?, 'sha256', ?, ?, ?, ?, ?, ?)
                """,
                (
                    token_id,
                    client.client_id,
                    verifier_digest,
                    _json(scopes),
                    issued_at,
                    expires_at,
                    int(write_capable),
                    client.generation,
                ),
            )
            conn.execute(
                "UPDATE auth_clients SET last_used_at = ?, updated_at = ? WHERE client_id = ?",
                (issued_at, issued_at, client.client_id),
            )
            self._audit(
                conn,
                event_type="token.issued",
                occurred_at=issued_at,
                client_id=client.client_id,
                token_id=token_id,
                actor=actor,
                source_ip=source_ip,
                request_id=request_id,
            )
            row = conn.execute("SELECT * FROM auth_access_tokens WHERE token_id = ?", (token_id,)).fetchone()
        return self._token(row)

    def get_token(self, token_id: str) -> AccessTokenRecord | None:
        with self.database.connect() as conn:
            row = conn.execute("SELECT * FROM auth_access_tokens WHERE token_id = ?", (token_id,)).fetchone()
        return self._token(row) if row else None

    def revoke_owned_token(
        self,
        *,
        token_id: str,
        client_id: str,
        now: str,
        actor: str,
        source_ip: str,
        request_id: str | None,
    ) -> bool:
        with self.database.transaction() as conn:
            row = conn.execute(
                "SELECT revoked_at FROM auth_access_tokens WHERE token_id = ? AND client_id = ?",
                (token_id, client_id),
            ).fetchone()
            if row is None or row["revoked_at"] is not None:
                return False
            conn.execute("UPDATE auth_access_tokens SET revoked_at = ? WHERE token_id = ?", (now, token_id))
            self._audit(
                conn,
                event_type="token.revoked",
                occurred_at=now,
                client_id=client_id,
                token_id=token_id,
                actor=actor,
                source_ip=source_ip,
                request_id=request_id,
            )
        return True

    def coalesce_last_used(
        self,
        *,
        client_id: str,
        token_id: str,
        now: str,
        threshold: str,
    ) -> None:
        with self.database.transaction() as conn:
            conn.execute(
                """
                UPDATE auth_clients SET last_used_at = ?, updated_at = ?
                 WHERE client_id = ? AND (last_used_at IS NULL OR last_used_at <= ?)
                """,
                (now, now, client_id, threshold),
            )
            conn.execute(
                """
                UPDATE auth_access_tokens SET last_used_at = ?
                 WHERE token_id = ? AND (last_used_at IS NULL OR last_used_at <= ?)
                """,
                (now, token_id, threshold),
            )

    def list_audit_events(self) -> list[dict[str, Any]]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT event_type, outcome, occurred_at, client_id, token_id,
                       actor, source_ip, request_id, reason_code
                  FROM auth_audit_events ORDER BY event_id
                """
            ).fetchall()
        return [dict(row) for row in rows]
