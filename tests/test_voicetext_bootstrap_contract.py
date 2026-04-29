from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_provisions_voicetext_runtime_dirs_under_canonical_state_root() -> None:
    bootstrap = (REPO_ROOT / "scripts/00-bootstrap.sh").read_text()

    assert 'VTP_STATE_BASE="${SEASONALWEATHER_DATA_BASE:-/var/lib/seasonalweather}"' in bootstrap
    assert 'install -d -o seasonalweather -g seasonalweather "${VTP_STATE_BASE}/voices"' in bootstrap
    assert 'install -d -o "${VTP_USER}"    -g "${VTP_USER}"    "${VTP_STATE_BASE}/wineprefixes"' in bootstrap
    assert 'install -d -m 700 -o "${VTP_USER}" -g "${VTP_USER}" "${VTP_STATE_BASE}/tmp"' in bootstrap


def test_bootstrap_does_not_recursively_clobber_voicetext_owned_state() -> None:
    bootstrap = (REPO_ROOT / "scripts/00-bootstrap.sh").read_text()

    assert 'chown -R seasonalweather:seasonalweather /var/lib/seasonalweather /var/log/seasonalweather' not in bootstrap
    assert 'chown seasonalweather:seasonalweather /var/lib/seasonalweather /var/lib/seasonalweather/audio /var/lib/seasonalweather/cache /var/log/seasonalweather' in bootstrap


def test_bootstrap_installs_and_enables_voicetext_xvfb_service() -> None:
    bootstrap = (REPO_ROOT / "scripts/00-bootstrap.sh").read_text()

    assert 'dpkg --add-architecture i386' in bootstrap
    assert 'apt-get install -y --no-install-recommends wine wine64 wine32:i386 unzip xvfb x11-utils' in bootstrap
    assert 'cp /opt/seasonalweather/app/systemd/seasonalweather-voicetext-xvfb.service /etc/systemd/system/seasonalweather-voicetext-xvfb.service' in bootstrap
    assert 'systemctl enable --now seasonalweather-voicetext-xvfb.service' in bootstrap
    assert 'VTP_SYSTEMD_DROPIN="${VTP_SYSTEMD_DROPIN_DIR}/10-voicetext-paul.conf"' in bootstrap
    assert 'After=seasonalweather-voicetext-xvfb.service' in bootstrap


def test_bootstrap_smoke_tests_voicetext_paul_before_service_use() -> None:
    bootstrap = (REPO_ROOT / "scripts/00-bootstrap.sh").read_text()

    assert 'SEASONAL_VOICETEXT_PAUL_SOURCE' in bootstrap
    assert 'SEASONAL_VOICETEXT_PAUL_SHA256' in bootstrap
    assert 'SEASONAL_VOICETEXT_PAUL_SMOKE:-1' in bootstrap
    assert 'VoiceText Paul smoke test failed; the backend is not safe to enable yet' in bootstrap
    assert 'runuser -u "${VTP_USER}" -- env' in bootstrap
