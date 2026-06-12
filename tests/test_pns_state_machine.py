import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from seasonalweather.broadcast.pns import PnsStateMachine


def _cfg():
    return SimpleNamespace(
        enabled=True,
        default_expire_hours=4,
        hard_stop_delimiter="&&",
        suppress_unknown_audio=True,
        reject_audio_keywords=["spotter reports", "storm reports", "metadata"],
        subtypes=[],
    )


def test_pns_transmitter_outage_becomes_cycle_audio() -> None:
    text = """
NOUS41 KLWX 211300
PNSLWX

Public Information Statement
National Weather Service Baltimore MD/Washington DC
900 AM EDT Thu May 21 2026

...Pikesville NOAA Weather Radio Off the Air Until Further Notice...

NOAA Weather Radio transmitter KEC-83 located in Baltimore,
Maryland, and broadcasting on a frequency of 162.400 Megahertz,
will be offline until further notice due to technical difficulties.

$$
"""
    pns = PnsStateMachine(_cfg(), tz=ZoneInfo("America/New_York"))
    decision = pns.evaluate(
        text,
        wfo="KLWX",
        awips_id="PNSLWX",
        issued=dt.datetime(2026, 5, 21, 13, 0, tzinfo=dt.timezone.utc).isoformat(),
        now=dt.datetime(2026, 5, 21, 14, 0, tzinfo=dt.timezone.utc),
    )

    assert decision.action == "audio"
    assert decision.subtype == "nwr_transmitter_outage"
    assert "service announcement" in decision.script_text.lower()
    assert "NOAA Weather Radio transmitter" in decision.script_text
    assert decision.expires_utc is not None


def test_spotter_report_pns_is_not_audio_even_before_metadata() -> None:
    text = """
NOUS41 KLWX 211300
PNSLWX

Public Information Statement
Spotter Reports
National Weather Service Baltimore MD/Washington DC
900 AM EDT Thu May 21 2026

**************PEAK WIND GUST (AT LEAST 39 MPH)**************

LOCATION             MAX WIND      TIME/DATE  COMMENTS
                         GUST       MEASURED
                        (mph)

MARYLAND

...Baltimore County...
  Owings Mills             40   430 PM  5/21  Mesonet
  Relay                    40   543 PM  5/21  Mesonet
  Towson                   41   555 PM  5/21  AWOS
  Dundalk                  47   630 PM  5/21  NDBC
  Essex                    44   706 PM  5/21  NOS-PORTS
  Pikesville               43   517 PM  5/21  Mesonet
  Catonsville              39   625 PM  5/21  ASOS
  Parkville                42   654 PM  5/21  Mesonet

&&

*****METADATA*****
:5/21/2026, 430 PM, MD, Baltimore, Owings Mills, 39.40, -76.78, PKGUST, 40, mph, Mesonet,

$$
"""
    pns = PnsStateMachine(_cfg(), tz=ZoneInfo("America/New_York"))
    decision = pns.evaluate(
        text,
        wfo="KLWX",
        awips_id="PNSLWX",
        issued=dt.datetime(2026, 5, 21, 13, 0, tzinfo=dt.timezone.utc).isoformat(),
        now=dt.datetime(2026, 5, 21, 14, 0, tzinfo=dt.timezone.utc),
    )

    assert decision.action == "ui_only"
    assert not decision.is_audio
    assert {"metadata", "table_header", "aligned_rows", "reject_keyword"}.intersection(decision.signals)


def test_recent_severe_weather_safety_rules_pns_with_overnight_ugc_expiry_is_audio() -> None:
    text = """464
NOUS41 KLWX 112052
PNSLWX
DCZ001-MDZ003>006-008-011-013-014-016>018-501>510-VAZ025>031-
036>040-050-051-053>057-501>508-526-527-WVZ050>053-055-501>506-
120200-

Public Information Statement
National Weather Service Baltimore MD/Washington DC
452 PM EDT Thu Jun 11 2026

...SEVERE WEATHER SAFETY RULES...

Severe thunderstorms capable of producing damaging winds and large
hail are possible this afternoon into this evening across eastern
West Virginia, northern and central Virginia, most of Maryland, and
the District of Columbia.

Residents in these areas should monitor this situation very closely
and ensure your NOAA weather radios are set to alert mode. Severe
weather warnings for imminent damaging storms may become necessary.

Here are some safety rules to keep in mind when severe weather is
expected or is occurring. If a warning is issued, seek shelter
indoors immediately.

Stay tuned to NOAA weather radio, commercial radio, information
sources from your phone, or television for the latest on this
potential severe weather event. Additional weather information can
be found at weather.gov/washington or weather.gov/baltimore.

$$
"""
    pns = PnsStateMachine(_cfg(), tz=ZoneInfo("America/New_York"))
    decision = pns.evaluate(
        text,
        wfo="KLWX",
        awips_id="PNSLWX",
        now=dt.datetime(2026, 6, 11, 20, 52, 11, tzinfo=dt.timezone.utc),
    )

    assert decision.action == "audio"
    assert decision.subtype == "severe_weather_safety_rules"
    assert decision.issued_utc == dt.datetime(2026, 6, 11, 20, 52, tzinfo=dt.timezone.utc)
    assert decision.expires_utc == dt.datetime(2026, 6, 12, 2, 0, tzinfo=dt.timezone.utc)


def test_pns_api_candidate_must_match_nwws_issuance() -> None:
    from seasonalweather.broadcast.pns import pns_text_same_issuance

    raw_text = """NOUS41 KLWX 112052
PNSLWX
DCZ001-
120200-

Public Information Statement
National Weather Service Baltimore MD/Washington DC
452 PM EDT Thu Jun 11 2026

...SEVERE WEATHER SAFETY RULES...
$$
"""
    stale_api_text = """NOUS41 KLWX 111300
PNSLWX
DCZ001-
111700-

Public Information Statement
National Weather Service Baltimore MD/Washington DC
900 AM EDT Thu Jun 11 2026

...SEVERE WEATHER SAFETY RULES...
$$
"""
    matching_api_text = raw_text.replace("DCZ001-", "DCZ001-MDZ003-")

    assert not pns_text_same_issuance(raw_text, stale_api_text)
    assert pns_text_same_issuance(raw_text, matching_api_text)
