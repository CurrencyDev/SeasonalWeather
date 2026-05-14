from seasonalweather.tts.tts import clean_for_tts, normalize_nws_spoken_times


def test_normalize_nws_spoken_times_adds_colons_to_compact_local_times() -> None:
    text = "At 651 PM EDT, storms remain in effect until 700 PM EDT."

    assert normalize_nws_spoken_times(text) == (
        "At 6:51 PM EDT, storms remain in effect until 7:00 PM EDT."
    )


def test_normalize_nws_spoken_times_handles_noon_midnight_style_hours() -> None:
    text = "The watch is in effect from 1200 AM to 115 PM and again at 1015 PM."

    assert normalize_nws_spoken_times(text) == (
        "The watch is in effect from 12:00 AM to 1:15 PM and again at 10:15 PM."
    )


def test_normalize_nws_spoken_times_leaves_utc_and_vtec_timestamps_alone() -> None:
    text = "/O.CON.KLWX.SV.W.0041.000000T0000Z-260513T2300Z/ TIME...MOT...LOC 2251Z"

    assert normalize_nws_spoken_times(text) == text


def test_clean_for_tts_normalizes_alert_times_before_synthesis() -> None:
    text = "At 651 PM EDT, severe thunderstorms were located near Winchester until 700 PM EDT."

    assert clean_for_tts(text) == (
        "At 6:51 PM EDT, severe thunderstorms were located near Winchester until 7:00 PM EDT."
    )
