import datetime as dt
from zoneinfo import ZoneInfo

from seasonalweather.broadcast.product_text import build_nwws_watch_vtec_script
from seasonalweather.same.ugc import extract_ugc_zones


WCN_SVA_NEW = """WWUS61 KLWX 201639
WCNLWX

WATCH COUNTY NOTIFICATION FOR WATCH 234
NATIONAL WEATHER SERVICE BALTIMORE MD/WASHINGTON DC
1239 PM EDT WED MAY 20 2026

DCC001-MDC031-VAC013-210000-
/O.NEW.KLWX.SV.A.0234.260520T1639Z-260521T0000Z/

THE NATIONAL WEATHER SERVICE HAS ISSUED SEVERE THUNDERSTORM WATCH
234 IN EFFECT UNTIL 8 PM EDT THIS EVENING FOR THE FOLLOWING
AREAS

THE DISTRICT OF COLUMBIA

IN MARYLAND THIS WATCH INCLUDES 1 COUNTY

IN CENTRAL MARYLAND

MONTGOMERY

IN VIRGINIA THIS WATCH INCLUDES 1 COUNTY

IN NORTHERN VIRGINIA

ARLINGTON

THIS INCLUDES THE CITIES OF ARLINGTON AND ROCKVILLE.

$$
"""


def test_wcn_ugc_county_block_is_parseable():
    assert extract_ugc_zones(WCN_SVA_NEW) == ["DCC001", "MDC031", "VAC013"]


def test_wcn_sva_new_uses_watch_specific_narration():
    script = build_nwws_watch_vtec_script(
        WCN_SVA_NEW,
        ["/O.NEW.KLWX.SV.A.0234.260520T1639Z-260521T0000Z/"],
        local_tz=ZoneInfo("America/New_York"),
        now=dt.datetime(2026, 5, 20, 12, 39, tzinfo=ZoneInfo("America/New_York")),
    )

    assert "The National Weather Service has issued Severe Thunderstorm Watch Number 234." in script
    assert "Effective until 8 PM this evening." in script
    assert "This watch includes the District of Columbia." in script
    assert "in Maryland: Montgomery" in script
    assert "in Virginia: Arlington" in script
    assert "WATCH COUNTY NOTIFICATION" not in script
    assert "DCC001" not in script
    assert script.endswith("End of message.")


def test_wcn_watch_script_prefers_resolved_area_text_when_available():
    script = build_nwws_watch_vtec_script(
        WCN_SVA_NEW,
        ["/O.NEW.KLWX.SV.A.0234.260520T1639Z-260521T0000Z/"],
        local_tz=ZoneInfo("America/New_York"),
        area_text="Montgomery, MD; Arlington, VA",
        now=dt.datetime(2026, 5, 20, 12, 39, tzinfo=ZoneInfo("America/New_York")),
    )

    assert "in Maryland: Montgomery" in script
    assert "in Virginia: Arlington" in script
    assert "the District of Columbia" not in script

WCN_SVA_NO_UGC = """WWUS61 KPHI 201810
WCNPHI

WATCH COUNTY NOTIFICATION FOR WATCH 235
NATIONAL WEATHER SERVICE MOUNT HOLLY NJ
210 PM EDT WED MAY 20 2026

/O.NEW.KPHI.SV.A.0235.260520T1810Z-260521T0100Z/

THE NATIONAL WEATHER SERVICE HAS ISSUED SEVERE THUNDERSTORM WATCH
235 IN EFFECT UNTIL 9 PM EDT THIS EVENING FOR THE FOLLOWING
AREAS

IN DELAWARE THIS WATCH INCLUDES 3 COUNTIES

KENT              NEW CASTLE        SUSSEX

IN MARYLAND THIS WATCH INCLUDES 4 COUNTIES

CAROLINE          KENT              QUEEN ANNE'S
TALBOT

THIS INCLUDES THE CITIES OF DOVER, GEORGETOWN, AND WILMINGTON.

$$
"""


def test_wcn_watch_without_ugc_can_format_text_but_has_no_same_target_source():
    """Regression fixture for live KPHI.SV.A.0235: CAP must not be deduped by a SAME-less WCN FULL."""
    assert extract_ugc_zones(WCN_SVA_NO_UGC) == []

    script = build_nwws_watch_vtec_script(
        WCN_SVA_NO_UGC,
        ["/O.NEW.KPHI.SV.A.0235.260520T1810Z-260521T0100Z/"],
        local_tz=ZoneInfo("America/New_York"),
        now=dt.datetime(2026, 5, 20, 14, 10, tzinfo=ZoneInfo("America/New_York")),
    )

    assert "The National Weather Service has issued Severe Thunderstorm Watch Number 235." in script
    assert "Effective until 9 PM tonight." in script
    assert "in Delaware: Kent New Castle Sussex" in script
    assert "in Maryland: Caroline Kent Queen Anne'S and Talbot" in script
