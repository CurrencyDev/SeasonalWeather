from seasonalweather.alerts.vtec import VTEC_FIND_RE, toneout_policy
from seasonalweather.broadcast.product_text import (
    build_nwws_partial_cancel_script,
    build_nwws_terminal_cancel_expiry_script,
    expiry_summary_script,
    parse_nwws_product_segments,
)
from seasonalweather.tts.tts import clean_for_tts


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
    assert "Locations impacted include. Winchester, Martinsburg" in segments[1].reason_text


def test_partial_can_con_script_says_cancelled_and_remains_in_effect():
    segments = parse_nwws_product_segments(PARTIAL_CAN_CON_SVS)
    script = build_nwws_partial_cancel_script("Severe Thunderstorm Warning", segments)

    assert "has been cancelled for the following areas: Morgan WV; Hampshire WV" not in script
    assert "The severe thunderstorm warning for southeastern morgan and southeastern hampshire counties is cancelled." in script
    assert "A severe thunderstorm warning remains in effect until 700 PM EDT" in script
    assert "The Severe Thunderstorm Warning remains in effect" not in script
    assert "Locations impacted include. Winchester, Martinsburg" in script
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



FLS_CAN_FA_W = """147
WGUS81 KLWX 271737
FLSLWX

Flood Statement
National Weather Service Baltimore MD/Washington DC
137 PM EDT Wed May 27 2026

MDC001-271747-
/O.CAN.KLWX.FA.W.0003.000000T0000Z-260527T1900Z/
/00000.0.ER.000000T0000Z.000000T0000Z.000000T0000Z.OO/
Allegany MD-
137 PM EDT Wed May 27 2026

...FLOOD WARNING IS CANCELLED...

The Flood Warning is cancelled for a portion of western Maryland,
including the following area, Allegany.

Flood waters have receded. The heavy rain has ended. Flooding is no
longer expected to pose a threat. Please continue to heed remaining
road closures.

&&

LAT...LON 3955 7899 3953 7899 3951 7899 3950 7901
      3947 7903 3948 7904 3948 7905 3947 7905
      3947 7906 3948 7906 3951 7905 3954 7903
      3956 7903 3955 7901

$$

KJP
"""


def test_fls_terminal_cancel_keeps_scoped_cancellation_prose():
    assert expiry_summary_script(FLS_CAN_FA_W) == "The heavy rain has ended.\nEnd of message."

    script = build_nwws_terminal_cancel_expiry_script("Flood Warning", FLS_CAN_FA_W)

    assert "The Flood Warning is cancelled for a portion of western Maryland" in script
    assert "including the following area, Allegany" in script
    assert "Flood waters have receded" in script
    assert "Flooding is no longer expected to pose a threat" in script
    assert "Please continue to heed remaining road closures" in script
    assert "Flood warning is cancelled.\nThe Flood Warning is cancelled" not in script
    assert script.endswith("End of message.")

EXP_SVS = """246
WWUS51 KLWX 201855
SVSLWX

Severe Weather Statement
National Weather Service Baltimore MD/Washington DC
255 PM EDT Wed May 20 2026

VAC069-WVC027-201904-
/O.EXP.KLWX.SV.W.0053.000000T0000Z-260520T1900Z/
Frederick VA-Hampshire WV-
255 PM EDT Wed May 20 2026

...THE SEVERE THUNDERSTORM WARNING FOR WEST CENTRAL FREDERICK COUNTY
IN NORTHWESTERN VIRGINIA AND SOUTHEASTERN HAMPSHIRE COUNTIES IN
EASTERN WEST VIRGINIA WILL EXPIRE AT 300 PM EDT...

The storm which prompted the warning has weakened below severe
limits, and no longer poses an immediate threat to life or property.
Therefore, the warning will be allowed to expire.

A Severe Thunderstorm Watch remains in effect until 800 PM EDT for
northwestern Virginia...and eastern West Virginia.

&&

LAT...LON 3917 7858 3923 7860 3935 7838 3917 7832
TIME...MOT...LOC 1854Z 253DEG 18KT 3924 7842

$$

Belak
"""


def test_expiry_summary_script_sentence_cases_all_caps_svs_headline_for_tts():
    script = expiry_summary_script(EXP_SVS)

    assert script is not None
    assert "WILL EXPIRE AT" not in script
    assert "will expire at 300 PM EDT." in script

    spoken = clean_for_tts(script)
    assert "will expire at 3:00 PM EDT." in spoken
    assert " AT " not in spoken

KLWX_CON_SVS = """435
WWUS51 KLWX 201903
SVSLWX

Severe Weather Statement
National Weather Service Baltimore MD/Washington DC
303 PM EDT Wed May 20 2026

VAC139-171-201915-
/O.CON.KLWX.SV.W.0054.000000T0000Z-260520T1915Z/
Shenandoah VA-Page VA-
303 PM EDT Wed May 20 2026

...A SEVERE THUNDERSTORM WARNING REMAINS IN EFFECT UNTIL 315 PM EDT
FOR SOUTH CENTRAL SHENANDOAH AND NORTH CENTRAL PAGE COUNTIES...

At 303 PM EDT, a severe thunderstorm was located near Luray, or 9
miles south of Woodstock, moving east at 20 mph.

HAZARD...60 mph wind gusts.

SOURCE...Radar indicated.

IMPACT...Damaging winds will cause some trees and large branches to
         fall. This could injure those outdoors, as well as damage
         homes and vehicles. Roadways may become blocked by downed
         trees. Localized power outages are possible. Unsecured
         light objects may become projectiles.

Locations impacted include...
Luray and Kings Crossing.

PRECAUTIONARY/PREPAREDNESS ACTIONS...

For your protection move to an interior room on the lowest floor of a
building.

&&

LAT...LON 3866 7857 3873 7858 3881 7846 3867 7842
TIME...MOT...LOC 1903Z 253DEG 18KT 3874 7851

$$

Belak
"""


def test_spoken_alert_sentence_cases_all_caps_svs_continuation_body_for_tts():
    from seasonalweather.alerts.builder import build_spoken_alert
    from seasonalweather.alerts.product import parse_product_text

    parsed = parse_product_text(KLWX_CON_SVS)
    assert parsed is not None

    spoken = build_spoken_alert(parsed, KLWX_CON_SVS)

    assert "FOR SOUTH CENTRAL" not in spoken.script
    assert "A SEVERE THUNDERSTORM WARNING" not in spoken.script
    assert "Shenandoah VA-Page VA-" not in spoken.script
    assert "For south central shenandoah" in spoken.script
    assert "Hazard: 60 mph wind gusts." in spoken.script
    assert "Source: Radar indicated." in spoken.script
    assert "Impact: Damaging winds will cause" in spoken.script


KLWX_PARTIAL_CAN_CON_SVS = """704
WWUS51 KLWX 202021
SVSLWX

Severe Weather Statement
National Weather Service Baltimore MD/Washington DC
421 PM EDT Wed May 20 2026

VAC043-202031-
/O.CAN.KLWX.SV.W.0060.000000T0000Z-260520T2045Z/
Clarke VA-
421 PM EDT Wed May 20 2026

...THE SEVERE THUNDERSTORM WARNING FOR SOUTHEASTERN CLARKE COUNTY IS
CANCELLED...

The severe thunderstorm which prompted the warning has moved out of
The warned area. Therefore, the warning has been cancelled.
A Severe Thunderstorm Watch remains in effect until 800 PM EDT for
northern Virginia.

&&

LAT...LON 3895 7783 3907 7788 3913 7757 3898 7748
TIME...MOT...LOC 2021Z 255DEG 15KT 3901 7776

$$

VAC061-107-202045-
/O.CON.KLWX.SV.W.0060.000000T0000Z-260520T2045Z/
Loudoun VA-Fauquier VA-
421 PM EDT Wed May 20 2026

...A SEVERE THUNDERSTORM WARNING REMAINS IN EFFECT UNTIL 445 PM EDT
FOR CENTRAL LOUDOUN AND NORTH CENTRAL FAUQUIER COUNTIES...

At 421 PM EDT, a severe thunderstorm was located over Middleburg, or
12 miles west of Brambleton, moving east at 20 mph.

HAZARD...60 mph wind gusts.

SOURCE...Radar indicated.

IMPACT...Damaging winds will cause some trees and large branches to
         fall. This could injure those outdoors, as well as damage
         homes and vehicles. Roadways may become blocked by downed
         trees. Localized power outages are possible. Unsecured
         light objects may become projectiles.

Locations impacted include...
Leesburg, Broadlands, Brambleton, Ashburn, Middleburg, Oatlands,
Saint Louis, Gleedsville, Aldie, Philomont, and Hughesville.

PRECAUTIONARY/PREPAREDNESS ACTIONS...

For your protection move to an interior room on the lowest floor of a
building.

&&

LAT...LON 3895 7783 3907 7788 3913 7757 3898 7748
TIME...MOT...LOC 2021Z 255DEG 15KT 3901 7776

HAIL THREAT...RADAR INDICATED
MAX HAIL SIZE...<.75 IN
WIND THREAT...RADAR INDICATED
MAX WIND GUST...60 MPH

$$

Belak
"""


def test_partial_can_con_klwx_preserves_nws_headlines_and_locations():
    segments = parse_nwws_product_segments(KLWX_PARTIAL_CAN_CON_SVS)
    script = build_nwws_partial_cancel_script("Severe Thunderstorm Warning", segments)

    assert "CANCELLED..." not in script
    assert "The severe thunderstorm warning for southeastern clarke county is cancelled." in script
    assert "The Severe Thunderstorm Warning has been cancelled for the following areas" not in script
    assert "A severe thunderstorm warning remains in effect until 445 PM EDT for central loudoun and north central fauquier counties." in script
    assert "The Severe Thunderstorm Warning remains in effect until 445 PM EDT for the following areas" not in script
    assert "Locations impacted include. Leesburg, Broadlands" in script
    assert "For your protection move to an interior room" in script
