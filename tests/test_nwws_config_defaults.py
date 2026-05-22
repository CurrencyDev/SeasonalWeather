from pathlib import Path

from seasonalweather.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _set_required_env(monkeypatch, *, nwws_jid: str, nwws_password: str) -> None:
    monkeypatch.setenv("NWWS_JID", nwws_jid)
    monkeypatch.setenv("NWWS_PASSWORD", nwws_password)
    monkeypatch.setenv("ICECAST_SOURCE_PASSWORD", "test-source-password")


def test_default_nwws_example_credentials_disable_nwws(monkeypatch) -> None:
    _set_required_env(
        monkeypatch,
        nwws_jid="CHANGEME@nwws-oi.weather.gov",
        nwws_password="CHANGEME",
    )

    cfg = load_config(str(REPO_ROOT / "config/config.yaml"))

    assert cfg.nwws.enabled is False
    assert cfg.nwws.credentials_defaulted is True
    assert cfg.secrets.nwws_jid == "CHANGEME@nwws-oi.weather.gov"


def test_real_nwws_credentials_keep_nwws_enabled(monkeypatch) -> None:
    _set_required_env(
        monkeypatch,
        nwws_jid="test-user@nwws-oi.weather.gov",
        nwws_password="not-the-default",
    )

    cfg = load_config(str(REPO_ROOT / "config/config.yaml"))

    assert cfg.nwws.enabled is True
    assert cfg.nwws.credentials_defaulted is False


def test_same_native_encoder_defaults_to_safe_python_fallback(monkeypatch) -> None:
    _set_required_env(
        monkeypatch,
        nwws_jid="test-user@nwws-oi.weather.gov",
        nwws_password="not-the-default",
    )

    cfg = load_config(str(REPO_ROOT / "config/config.yaml"))

    assert cfg.same.native_encoder.enabled is False
    assert cfg.same.native_encoder.bin == "samegen"
    assert cfg.same.native_encoder.fallback_to_python is True
