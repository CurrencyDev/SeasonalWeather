from __future__ import annotations

from types import SimpleNamespace

from seasonalweather.broadcast.ipaws_text import build_ipaws_script
from seasonalweather.broadcast.product_text import NwsAlertTextInput, build_nws_full_alert_script


_FORBIDDEN_CLOSER = "End of message."


def test_central_nws_full_script_has_no_spoken_eom_closer() -> None:
    script = build_nws_full_alert_script(
        NwsAlertTextInput(
            event="Severe Thunderstorm Warning",
            description="A severe thunderstorm was located near Exampleville.",
            instruction="Move indoors and stay away from windows.",
        ),
        sps_preamble=lambda _sent: "The National Weather Service has issued a Severe Thunderstorm Warning.",
    )

    assert _FORBIDDEN_CLOSER not in script
    assert script.endswith("Move indoors and stay away from windows.")


def test_ipaws_script_has_no_spoken_eom_closer() -> None:
    script = build_ipaws_script(
        SimpleNamespace(
            sender_name_clean="Example County Emergency Management",
            event="Civil Emergency Message",
            headline="Shelter in place",
            description="Remain indoors until further notice.",
            instruction="Close all windows and doors.",
        )
    )

    assert _FORBIDDEN_CLOSER not in script
    assert script.endswith("Close all windows and doors.")
