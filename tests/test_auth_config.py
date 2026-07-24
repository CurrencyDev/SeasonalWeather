from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from seasonalweather.config import AuthConfigurationError, AuthMode, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_config(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    raw = yaml.safe_load((REPO_ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None] | None = None,
):
    monkeypatch.setenv("ICECAST_SOURCE_PASSWORD", "synthetic-source-password")
    return load_config(str(_write_config(tmp_path, mutate)))


def test_explicit_static_mode_builds_typed_redacted_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _load(monkeypatch, tmp_path)

    assert cfg.api.auth.mode is AuthMode.STATIC
    assert cfg.api.auth.legacy_mode_normalized is False
    assert cfg.api.auth.legacy_scope_normalized is False
    assert cfg.api.auth.credentials[0].subject == "local-admin"
    assert "read:status" in cfg.api.auth.credentials[0].scopes
    assert "synthetic-test-api-token" not in repr(cfg)


def test_legacy_omitted_mode_normalizes_only_with_one_valid_credential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def legacy(raw: dict[str, Any]) -> None:
        raw["api"].pop("auth")
        raw["api"]["subject"] = "legacy-admin"
        raw["api"]["scopes"] = "read:status control:cycle"

    cfg = _load(monkeypatch, tmp_path, legacy)

    assert cfg.api.auth.mode is AuthMode.STATIC
    assert cfg.api.auth.legacy_mode_normalized is True
    assert cfg.api.auth.credentials[0].subject == "legacy-admin"
    assert cfg.api.auth.credentials[0].scopes == frozenset({"read:status", "control:cycle"})


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        ("unsupported", "unknown_value"),
        ("", "empty_value"),
        ("   ", "empty_value"),
        (42, "invalid_type"),
        (None, "invalid_type"),
    ],
)
def test_invalid_mode_values_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: Any,
    kind: str,
) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["api"]["auth"]["mode"] = value

    with pytest.raises(AuthConfigurationError) as exc_info:
        _load(monkeypatch, tmp_path, mutate)

    assert exc_info.value.kind == kind
    assert exc_info.value.path == "api.auth.mode"


@pytest.mark.parametrize("token", [None, "", "   ", "bad token", "CHANGEME"])
def test_explicit_static_rejects_absent_or_invalid_single_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    token: str | None,
) -> None:
    if token is None:
        monkeypatch.delenv("SEASONAL_API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("SEASONAL_API_TOKEN", token)

    with pytest.raises(AuthConfigurationError) as exc_info:
        _load(monkeypatch, tmp_path)

    assert exc_info.value.kind == "invalid_credential"
    assert exc_info.value.path == "SEASONAL_API_TOKEN"


def test_omitted_mode_without_token_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("SEASONAL_API_TOKEN", raising=False)

    def legacy(raw: dict[str, Any]) -> None:
        raw["api"].pop("auth")

    with pytest.raises(AuthConfigurationError) as exc_info:
        _load(monkeypatch, tmp_path, legacy)

    assert exc_info.value.kind == "invalid_credential"


def test_current_and_legacy_auth_fields_conflict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["api"]["scopes"] = "read:status"

    with pytest.raises(AuthConfigurationError) as exc_info:
        _load(monkeypatch, tmp_path, mutate)

    assert exc_info.value.kind == "conflicting_fields"
    assert exc_info.value.path == "api.auth"


def test_single_and_multi_token_sources_conflict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "SEASONAL_API_TOKENS_JSON",
        json.dumps(
            {
                "second-synthetic-token": {
                    "subject": "test",
                    "scopes": ["read:status"],
                }
            }
        ),
    )

    with pytest.raises(AuthConfigurationError) as exc_info:
        _load(monkeypatch, tmp_path)

    assert exc_info.value.kind == "conflicting_credentials"


@pytest.mark.parametrize("mode", ["exchange", "hybrid"])
def test_exchange_dependent_modes_are_explicitly_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["api"]["auth"]["mode"] = mode

    with pytest.raises(AuthConfigurationError) as exc_info:
        _load(monkeypatch, tmp_path, mutate)

    error = exc_info.value
    assert error.kind == "authenticator_unavailable"
    assert error.path == "api.auth.mode"
    assert error.details == {"mode": mode, "available_modes": ["static"]}


def test_legacy_comma_scopes_normalize_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["api"]["auth"]["scopes"] = "read:status,control:cycle"

    cfg = _load(monkeypatch, tmp_path, mutate)

    assert cfg.api.auth.legacy_scope_normalized is True
    assert cfg.api.auth.credentials[0].scopes == frozenset({"read:status", "control:cycle"})


@pytest.mark.parametrize(
    ("scopes", "kind"),
    [
        ("", "empty_scope"),
        (" read:status", "empty_scope"),
        ("read:status  control:cycle", "empty_scope"),
        ("read:status,", "empty_scope"),
        ("read:status, control:cycle", "mixed_scope_delimiters"),
        ("read:status read:status", "duplicate_scope"),
        ("read:stat", "malformed_scope"),
        ("read:status:extra", "malformed_scope"),
    ],
)
def test_invalid_single_token_scope_forms_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scopes: str,
    kind: str,
) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["api"]["auth"]["scopes"] = scopes

    with pytest.raises(AuthConfigurationError) as exc_info:
        _load(monkeypatch, tmp_path, mutate)

    assert exc_info.value.kind == kind


def test_multi_token_scope_arrays_are_validated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("SEASONAL_API_TOKEN", raising=False)
    monkeypatch.setenv(
        "SEASONAL_API_TOKENS_JSON",
        json.dumps(
            {
                "synthetic-multi-token": {
                    "subject": "multi-client",
                    "scopes": ["read:status", "control:cycle"],
                }
            }
        ),
    )

    cfg = _load(monkeypatch, tmp_path)

    assert cfg.api.auth.credentials[0].scopes == frozenset({"read:status", "control:cycle"})


def test_configuration_errors_and_rendering_redact_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sentinel = "SENTINEL-STATIC-CREDENTIAL"
    monkeypatch.setenv("SEASONAL_API_TOKEN", sentinel)
    cfg = _load(monkeypatch, tmp_path)

    assert sentinel not in repr(cfg)
    assert sentinel not in repr(cfg.api)
    assert sentinel not in repr(cfg.api.auth)
    assert sentinel not in repr(cfg.api.auth.credentials[0])

    monkeypatch.setenv("SEASONAL_API_TOKEN", f"{sentinel} invalid")
    with pytest.raises(AuthConfigurationError) as exc_info:
        _load(monkeypatch, tmp_path)

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)
