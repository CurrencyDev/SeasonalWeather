from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_voicetext_runtime_defaults_use_canonical_state_root() -> None:
    synth = (REPO_ROOT / "scripts/wrappers/voicetext_paul_synth").read_text()
    kill = (REPO_ROOT / "scripts/wrappers/voicetext_paul_wineserver_kill").read_text()
    tts = (REPO_ROOT / "seasonalweather/tts/tts.py").read_text()

    assert "/var/lib/seasonalweather" in synth
    assert "/var/lib/seasonalweather" in kill
    assert 'os.getenv("SEASONALWEATHER_DATA_BASE", "/var/lib/seasonalweather")' in tts
    assert "/home/seasonalweather-data/var-lib-seasonalweather" not in synth
    assert "/home/seasonalweather-data/var-lib-seasonalweather" not in kill
    assert "/home/seasonalweather-data/var-lib-seasonalweather" not in tts


def test_voicetext_wrappers_default_to_headless_display_service() -> None:
    synth = (REPO_ROOT / "scripts/wrappers/voicetext_paul_synth").read_text()
    kill = (REPO_ROOT / "scripts/wrappers/voicetext_paul_wineserver_kill").read_text()
    unit = (REPO_ROOT / "systemd/seasonalweather-voicetext-xvfb.service").read_text()

    assert 'VOICETEXT_PAUL_DISPLAY:-:99' in synth
    assert 'VOICETEXT_PAUL_DISPLAY:-:99' in kill
    assert 'DISPLAY=${DISPLAY} is not available' in synth
    assert 'ExecStart=/usr/bin/Xvfb :99 -screen 0 1024x768x24 -nolisten tcp -noreset -ac' in unit


def test_voicetext_wrapper_defaults_to_32_bit_prefix() -> None:
    synth = (REPO_ROOT / "scripts/wrappers/voicetext_paul_synth").read_text()
    kill = (REPO_ROOT / "scripts/wrappers/voicetext_paul_wineserver_kill").read_text()

    assert 'REQUESTED_WINEARCH="${VOICETEXT_PAUL_WINEARCH:-win32}"' in synth
    assert '[[ ! -f "${PREFIX}/system.reg" && "${REQUESTED_WINEARCH}" != "auto" ]]' in synth
    assert 'VOICETEXT_PAUL_WINEARCH:-win32' in kill
    assert 'VOICETEXT_PAUL_WINEDLLOVERRIDES:-mscoree,mshtml=' in synth
    assert 'wineboot --init' in synth
    assert 'unset WINEARCH' in synth


def test_voicetext_wrapper_retries_stateful_wine_crashes() -> None:
    synth = (REPO_ROOT / "scripts/wrappers/voicetext_paul_synth").read_text()
    kill = (REPO_ROOT / "scripts/wrappers/voicetext_paul_wineserver_kill").read_text()

    assert 'ATTEMPTS="${VOICETEXT_PAUL_ATTEMPTS:-2}"' in synth
    assert 'RETRY_RCS=" ${VOICETEXT_PAUL_RETRY_RCS:-134 139} "' in synth
    assert 'wineserver -k >/dev/null 2>&1 || true' in synth
    assert 'wine attempt ${attempt}/${ATTEMPTS} failed rc=${rc1}' in synth
    assert 'HOME="${pw_home}"' in synth
    assert 'HOME="${pw_home}"' in kill


def test_voicetext_runtime_serializes_python_side_access_to_shared_wav() -> None:
    tts = (REPO_ROOT / "seasonalweather/tts/tts.py").read_text()

    assert 'import fcntl' in tts
    assert 'def _flock_path' in tts
    assert 'state_base / ".voicetext_paul_tts.lock"' in tts
