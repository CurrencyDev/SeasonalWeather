from seasonalweather.alerts.vtec import VTEC_FIND_RE, toneout_policy
from seasonalweather.broadcast.product_text import (
    build_nwws_partial_cancel_script,
    parse_nwws_product_segments,
)


PARTIAL_CAN_CON_SVS = """403
WWUS51 KLWX 132251
SVSLWX

Severe Weather Statement
National Weather Service Baltimore MD/Washington DC
651 PM EDT Wed May 13 2026

WVC027-065-132301-
/O.CAN.KLWX.SV.W.0041.000000T0000Z-260513T2300Z/
Morgan WV-Hampshire WV-
651 PM EDT Wed May 13 2026

...THE SEVERE THUNDERSTORM WARNING FOR SOUTHEASTERN MORGAN AND
SOUTHEASTERN HAMPSHIRE COUNTIES IS CANCELLED...

The storms which prompted the warning have moved out of the warned
area. Therefore, the warning has been cancelled.

&&

LAT...LON 3957 7783 3956 7784 3956 7789 3954 7787
TIME...MOT...LOC 2251Z 279DEG 35KT 3951 7802 3925 7815 3910 7839

$$

VAC043-069-840-WVC003-037-132300-
/O.CON.KLWX.SV.W.0041.000000T0000Z-260513T2300Z/
Frederick VA-Clarke VA-City of Winchester VA-Jefferson WV-
Berkeley WV-
651 PM EDT Wed May 13 2026

...A SEVERE THUNDERSTORM WARNING REMAINS IN EFFECT UNTIL 700 PM EDT
FOR SOUTHERN FREDERICK AND NORTHWESTERN CLARKE COUNTIES IN
NORTHWESTERN VIRGINIA...CENTRAL JEFFERSON AND CENTRAL BERKELEY
COUNTIES IN THE PANHANDLE OF WEST VIRGINIA AND THE CITY OF
WINCHESTER...

At 651 PM EDT, severe thunderstorms were located along a line
extending from near Martinsburg to near Winchester to near Star
Tannery, moving east at 40 mph.

HAZARD...60 mph wind gusts and quarter size hail.

SOURCE...Radar indicated.

IMPACT...Damaging winds will cause some trees and large branches to
         fall. This could injure those outdoors, as well as damage
         homes and vehicles.

Locations impacted include...
Winchester, Martinsburg, Charles Town, Shepherdstown, Millwood Pike,
Ranson, Berryville, Inwood, Stephens City, Kearneysville.

PRECAUTIONARY/PREPAREDNESS ACTIONS...

For your protection move to an interior room on the lowest floor of a
building.

&&

LAT...LON 3957 7783 3956 7784 3956 7789 3954 7787
TIME...MOT...LOC 2251Z 279DEG 35KT 3951 7802 3925 7815 3910 7839

HAIL THREAT...RADAR INDICATED
MAX HAIL SIZE...1.00 IN
WIND THREAT...RADAR INDICATED
MAX WIND GUST...60 MPH

$$

KLW
"""


def test_partial_can_con_vtec_policy_tracks_both_actions():
    vtec = VTEC_FIND_RE.findall(PARTIAL_CAN_CON_SVS)
    policy = toneout_policy(vtec)

    assert vtec == [
        "/O.CAN.KLWX.SV.W.0041.000000T0000Z-260513T2300Z/",
        "/O.CON.KLWX.SV.W.0041.000000T0000Z-260513T2300Z/",
    ]
    assert policy.mode == "VOICE"
    assert policy.same_code is None
    assert policy.cancel_tracks == frozenset({"KLWX.SV.W.0041"})
    assert policy.continuation_tracks == frozenset({"KLWX.SV.W.0041"})


def test_partial_can_con_svs_segments_include_cancel_and_continuation_text():
    segments = parse_nwws_product_segments(PARTIAL_CAN_CON_SVS)

    assert len(segments) == 2
    assert segments[0].actions == {"CAN"}
    assert segments[0].area_text == "Morgan WV; Hampshire WV"
    assert "warning has been cancelled" in segments[0].reason_text

    assert segments[1].actions == {"CON"}
    assert segments[1].area_text == "Frederick VA; Clarke VA; City of Winchester VA; Jefferson WV; Berkeley WV"
    assert segments[1].expiry_phrase == "700 PM EDT"
    assert "60 mph wind gusts and quarter size hail" in segments[1].reason_text
    assert "For your protection move to an interior room" in segments[1].precautions
    assert "Winchester, Martinsburg" not in segments[1].reason_text


def test_partial_can_con_script_says_cancelled_and_remains_in_effect():
    segments = parse_nwws_product_segments(PARTIAL_CAN_CON_SVS)
    script = build_nwws_partial_cancel_script("Severe Thunderstorm Warning", segments)

    assert "has been cancelled for the following areas: Morgan WV; Hampshire WV" in script
    assert "remains in effect until 700 PM EDT" in script
    assert "Frederick VA; Clarke VA; City of Winchester VA; Jefferson WV; Berkeley WV" in script
    assert script.endswith("End of message.")

KCTP_PARTIAL_CAN_CON_SVS = """930
WWUS51 KCTP 201726
SVSCTP

Severe Weather Statement
National Weather Service State College PA
126 PM EDT Wed May 20 2026

PAC057-201745-
/O.CAN.KCTP.SV.W.0064.000000T0000Z-260520T1745Z/
Fulton PA-
126 PM EDT Wed May 20 2026

...THE SEVERE THUNDERSTORM WARNING FOR FULTON COUNTY IS CANCELLED...

CANCELLED

The severe thunderstorm which prompted the warning has moved out of
The warned area. Therefore, the warning has been cancelled.
A Severe Thunderstorm Watch remains in effect until 800 PM EDT for
south central Pennsylvania.

&&

LAT...LON 3990 7800
TIME...MOT...LOC 1725Z 260DEG 22KT 3980 7800

$$

PAC055-201745-
/O.CON.KCTP.SV.W.0064.000000T0000Z-260520T1745Z/
Franklin PA-
126 PM EDT Wed May 20 2026

...A SEVERE THUNDERSTORM WARNING REMAINS IN EFFECT UNTIL 145 PM EDT
FOR SOUTHWESTERN FRANKLIN COUNTY...

FOR SOUTHWESTERN FRANKLIN COUNTY...

At 125 PM EDT, a severe thunderstorm was located over Claylick,
moving east at 25 mph.

HAZARD...60 mph wind gusts and quarter size hail.

SOURCE...Radar indicated.

IMPACT...Hail damage to vehicles is expected. Expect wind damage to
         roofs, siding, and trees.

Locations impacted include...
St. Thomas, Mercersburg, Claylick, Williamson, Upton, and Whitetail
Ski Area.

PRECAUTIONARY/PREPAREDNESS ACTIONS...

Stay inside a well built structure and keep away from windows.

&&

LAT...LON 3990 7800
TIME...MOT...LOC 1725Z 260DEG 22KT 3980 7800

$$
"""


def test_partial_can_con_svs_preserves_sentence_boundaries_and_skips_labels():
    segments = parse_nwws_product_segments(KCTP_PARTIAL_CAN_CON_SVS)
    script = build_nwws_partial_cancel_script("Severe Thunderstorm Warning", segments)

    assert "CANCELLED The severe thunderstorm" not in script
    assert "warning has been cancelled. A Severe Thunderstorm Watch" in script
    assert "COUNTY At 125 PM" not in script
    assert "moving east at 25 mph. Hazard: 60 mph wind gusts" in script
    assert "Hazard: 60 mph wind gusts and quarter size hail. Source: Radar indicated. Impact: Hail damage" in script
    assert "Expect wind damage to roofs, siding, and trees." in script
    assert "Stay inside a well built structure and keep away from windows." in script
