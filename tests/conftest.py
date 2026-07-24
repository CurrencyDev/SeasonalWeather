from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _synthetic_static_api_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep repository-config tests on the explicit static auth contract."""
    monkeypatch.setenv("SEASONAL_API_TOKEN", "synthetic-test-api-token")
    monkeypatch.delenv("SEASONAL_API_TOKENS_JSON", raising=False)
