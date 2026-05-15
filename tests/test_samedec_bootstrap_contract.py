from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_samedec_installer_script_is_shell_valid() -> None:
    subprocess.run(["bash", "-n", str(ROOT / "scripts/install-samedec.sh")], check=True)


def test_bootstrap_script_is_shell_valid() -> None:
    subprocess.run(["bash", "-n", str(ROOT / "scripts/00-bootstrap.sh")], check=True)


def test_samedec_installer_uses_pinned_crate_and_expected_paths() -> None:
    script = (ROOT / "scripts/install-samedec.sh").read_text()

    assert 'SAMEDEC_VERSION_DEFAULT="0.4.2"' in script
    assert 'SAMEDEC_ROOT="${SEASONAL_SAMEDEC_ROOT:-/opt/seasonalweather/samedec}"' in script
    assert 'SAMEDEC_BIN="${SEASONAL_SAMEDEC_BIN:-/usr/local/bin/samedec}"' in script
    assert "cargo install" in script
    assert "--locked" in script
    assert "--force" in script
    assert 'printf \'%s\\n\' "${SAMEDEC_VERSION}" >"${SAMEDEC_ROOT}/VERSION"' in script


def test_bootstrap_installs_samedec_by_default_with_explicit_opt_out() -> None:
    script = (ROOT / "scripts/00-bootstrap.sh").read_text()

    assert 'if [[ "${SEASONAL_SAMEDEC:-1}" == "1" ]]; then' in script
    assert "bash /opt/seasonalweather/app/scripts/install-samedec.sh" in script
    assert "Skipping samedec" in script
