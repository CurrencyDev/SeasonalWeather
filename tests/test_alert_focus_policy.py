import datetime as dt

from seasonalweather.alerts.active import ActiveAlert
from seasonalweather.alerts.focus import AlertFocusPolicy, alert_holds_focus


def _alert(
    id_: str,
    source: str,
    event: str,
    code: str,
    *,
    vtec: list[str] | None = None,
    headline: str | None = None,
) -> ActiveAlert:
    return ActiveAlert(
        id=id_,
        source=source,
        event=event,
        code=code,
        vtec=list(vtec or []),
        headline=headline or event,
        script_text=event,
        audio_path=None,
        expires=(dt.datetime(2026, 6, 6, 12, tzinfo=dt.timezone.utc)).isoformat(),
        issued=(dt.datetime(2026, 6, 6, 0, tzinfo=dt.timezone.utc)).isoformat(),
        same_locs=["024003"],
        cycle_only=True,
    )


def _policy(*, hold_codes: tuple[str, ...] | None = None) -> AlertFocusPolicy:
    return AlertFocusPolicy(
        hold_event_codes=hold_codes
        or (
            "TOA",
            "TOR",
            "SVA",
            "SVR",
            "SMW",
            "FFA",
            "FFW",
            "FLW",
            "TRA",
            "TRW",
            "HUA",
            "HUW",
            "WSW",
            "BZW",
            "SQW",
            "HWW",
            "CFW",
            "CDW",
            "CEM",
            "EAN",
            "EWW",
            "LAE",
            "TOE",
            # Included here to prove lower-priority/test/marine guards still win.
            "GLW",
            "SCY",
            "SPS",
            "RWT",
        )
    )


def test_focus_hold_allows_warning_watch_and_special_marine_warning() -> None:
    policy = _policy()

    assert alert_holds_focus(
        _alert("tor", "CAP", "Tornado Warning", "TOR", vtec=["/O.NEW.KLWX.TO.W.0001.260606T0000Z-260606T0100Z/"]),
        policy,
    )
    assert alert_holds_focus(
        _alert("sva", "NWWS", "Severe Thunderstorm Watch", "SVA", vtec=["/O.NEW.KLWX.SV.A.0123.260606T0000Z-260606T0600Z/"]),
        policy,
    )
    assert alert_holds_focus(
        _alert("smw", "CAP", "Special Marine Warning", "SMW", vtec=["/O.NEW.KLWX.MA.W.0002.260606T0000Z-260606T0030Z/"]),
        policy,
    )


def test_focus_hold_excludes_advisory_statement_and_long_lived_marine() -> None:
    policy = _policy()

    assert not alert_holds_focus(
        _alert("sca", "CAP", "Small Craft Advisory", "SCY", vtec=["/O.NEW.KLWX.SC.Y.0001.260606T0000Z-260607T0000Z/"]),
        policy,
    )
    assert not alert_holds_focus(
        _alert("sps", "NWWS", "Special Weather Statement", "SPS", vtec=["/O.NEW.KLWX.SP.S.0001.260606T0000Z-260606T0300Z/"]),
        policy,
    )
    assert not alert_holds_focus(
        _alert("glw", "CAP", "Gale Warning", "GLW", vtec=["/O.NEW.KLWX.GL.W.0001.260606T0000Z-260607T0000Z/"]),
        policy,
    )


def test_focus_hold_keeps_operationally_significant_non_vtec_codes() -> None:
    policy = _policy()

    for code in ("EAN", "LAE", "CDW", "CEM", "TOE"):
        assert alert_holds_focus(_alert(code.lower(), "IPAWS", code, code, vtec=[]), policy)


def test_focus_hold_keeps_tropical_watch_warning_codes() -> None:
    policy = _policy()

    for code, phen_sig, label in (
        ("TRA", "TR.A", "Tropical Storm Watch"),
        ("TRW", "TR.W", "Tropical Storm Warning"),
        ("HUA", "HU.A", "Hurricane Watch"),
        ("HUW", "HU.W", "Hurricane Warning"),
    ):
        assert alert_holds_focus(
            _alert(code.lower(), "CAP", label, code, vtec=[f"/O.NEW.KLWX.{phen_sig}.0001.260606T0000Z-260606T0600Z/"]),
            policy,
        )


def test_focus_hold_excludes_pns_and_ern_tests() -> None:
    policy = _policy()

    assert not alert_holds_focus(_alert("pns", "PNS_CYCLE", "Tornado Warning", "TOR"), policy)
    assert not alert_holds_focus(_alert("ern-rwt", "ERN", "Required Weekly Test", "RWT"), policy)
    assert not alert_holds_focus(_alert("ern-test", "ERN", "Test Tornado Warning", "TOR"), policy)
