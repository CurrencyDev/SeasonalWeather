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
