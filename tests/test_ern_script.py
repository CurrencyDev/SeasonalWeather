import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from seasonalweather.broadcast.ern_script import build_ern_relay_script


def test_ern_script_includes_test_metadata_and_area():
    ev = SimpleNamespace(
        org="EAS",
        event="RWT",
        locations=("024003", "024031"),
        tttt="0015",
        jjjhhmm="1480130",
        sender="WJON/TV",
    )

    script = build_ern_relay_script(
        ev,
        same_locations=ev.locations,
        area_text="Anne Arundel County, MD; Montgomery County, MD",
        tz=ZoneInfo("America/New_York"),
        now_utc=dt.datetime(2026, 5, 28, 2, 0, tzinfo=dt.timezone.utc),
    )

    assert "relay of a Required Weekly Test" in script
    assert "This is only a test." in script
    assert "An EAS participant has issued a Required Weekly Test." in script
    assert "Anne Arundel County, MD; Montgomery County, MD" in script
    assert "SAME header time: 9:30 PM EDT on Wednesday, May 27." in script
    assert "Valid until 9:45 PM EDT on Wednesday, May 27." in script
    assert "Sender: WJON/TV." in script
    assert script.endswith("End of message.")


def test_ern_script_includes_warning_metadata_without_area_lookup():
    ev = SimpleNamespace(
        org="WXR",
        event="SVR",
        locations=("024003", "024031"),
        tttt="0030",
        jjjhhmm="1480130",
        sender="WJON/TV",
    )

    script = build_ern_relay_script(
        ev,
        same_locations=ev.locations,
        tz=dt.timezone.utc,
        now_utc=dt.datetime(2026, 5, 28, 2, 0, tzinfo=dt.timezone.utc),
    )

    assert "relay of a Severe Thunderstorm Warning" in script
    assert "The National Weather Service has issued a Severe Thunderstorm Warning." in script
    assert "Affected SAME locations: 024003 and 024031." in script
    assert "SAME header time: 1:30 AM UTC on Thursday, May 28." in script
    assert "Valid until 2:00 AM UTC on Thursday, May 28." in script
    assert "This is only a test." not in script
