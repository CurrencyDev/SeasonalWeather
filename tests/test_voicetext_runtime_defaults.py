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
    assert 'ExecStart=/usr/bin/Xvfb :99 -screen 0 1024x768x24 -nolisten tcp -noreset' in unit
