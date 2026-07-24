from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from seasonalweather.cli.auth import main

REPO_ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path) -> Path:
    raw = yaml.safe_load((REPO_ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    raw["database"]["path"] = str(tmp_path / "auth.sqlite3")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_cli_client_lifecycle_and_one_time_secret_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ICECAST_SOURCE_PASSWORD", "synthetic-source-password")
    config = _config(tmp_path)
    common = ["--config", str(config), "--json", "client"]
    assert (
        main(
            common
            + [
                "create",
                "--subject",
                "cli-client",
                "--scope",
                "read:status",
                "--route-prefix",
                "/v1/status",
                "--cidr",
                "127.0.0.1/32",
            ]
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    client_id = created["client_id"]
    credential = created["client_credential"]
    assert credential.startswith("swc_")

    assert main(common + ["list"]) == 0
    listed = capsys.readouterr().out
    assert credential not in listed
    assert client_id in listed
    assert main(common + ["show", client_id]) == 0
    assert credential not in capsys.readouterr().out
    assert main(common + ["disable", client_id]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "disabled"
    assert main(common + ["enable", client_id]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "enabled"
    assert main(common + ["rotate", client_id]) == 0
    rotated = json.loads(capsys.readouterr().out)
    assert rotated["client_credential"].startswith("swc_")
    assert rotated["client_credential"] != credential
    assert main(common + ["revoke", client_id]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "revoked"
