from seasonalweather.config import load_config


def test_default_alert_focus_policy_inherits_toneout_without_tests(monkeypatch) -> None:
    monkeypatch.setenv("ICECAST_SOURCE_PASSWORD", "test-source")
    cfg = load_config("config/config.yaml")

    assert "TOR" in cfg.cycle.alert_focus.hold_event_codes
    assert "EAN" in cfg.cycle.alert_focus.hold_event_codes
    assert "TRA" in cfg.cycle.alert_focus.hold_event_codes
    assert "RWT" not in cfg.cycle.alert_focus.hold_event_codes
    assert "PNS_CYCLE" in cfg.cycle.alert_focus.excluded_sources
    assert "SMW" in cfg.cycle.alert_focus.marine_hold_event_codes
    assert "GLW" in cfg.cycle.alert_focus.marine_event_codes
