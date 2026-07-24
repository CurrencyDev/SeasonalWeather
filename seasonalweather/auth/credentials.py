from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass, field

_ID_RE = r"[a-f0-9]{24}"
_BODY_RE = r"[A-Za-z0-9_-]{43}"
_CLIENT_RE = re.compile(rf"^swc_({_ID_RE})\.({_BODY_RE})$")
_ACCESS_RE = re.compile(rf"^swa_({_ID_RE})\.({_BODY_RE})$")


@dataclass(frozen=True)
class ParsedCredential:
    public_id: str
    secret: str = field(repr=False)


def _generate(prefix: str) -> tuple[str, str, str]:
    public_id = secrets.token_hex(12)
    secret = secrets.token_urlsafe(32)
    return public_id, secret, f"{prefix}{public_id}.{secret}"


def generate_client_credential() -> tuple[str, str, str]:
    return _generate("swc_")


def generate_access_token() -> tuple[str, str, str]:
    return _generate("swa_")


def parse_client_credential(value: str) -> ParsedCredential | None:
    match = _CLIENT_RE.fullmatch(value)
    return ParsedCredential(match.group(1), match.group(2)) if match else None


def parse_access_token(value: str) -> ParsedCredential | None:
    match = _ACCESS_RE.fullmatch(value)
    return ParsedCredential(match.group(1), match.group(2)) if match else None


def secret_digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("ascii")).hexdigest()


def verify_secret(secret: str, expected_digest: str) -> bool:
    return hmac.compare_digest(secret_digest(secret), expected_digest)
