from seasonalweather.broadcast.cap_policy import cap_event_to_same_code, cap_should_full
from seasonalweather.broadcast.cap_runtime import CapRuntime
from seasonalweather.broadcast.ipaws_runtime import IpawsRuntime
from seasonalweather.broadcast.nwws_runtime import NwwsRuntime
from seasonalweather.broadcast.pns_runtime import PnsRuntime
from seasonalweather.broadcast.tests_runtime import RequiredTestRuntime
from seasonalweather.broadcast.manual_runtime import ManualOriginationRuntime
from seasonalweather.broadcast.service_runtime import SeasonalWeatherServiceRuntime


def test_source_runtime_modules_importable() -> None:
    assert cap_event_to_same_code("Tornado Warning") == "TOR"
    assert cap_should_full is not None
    assert CapRuntime is not None
    assert IpawsRuntime is not None
    assert NwwsRuntime is not None
    assert PnsRuntime is not None
    assert RequiredTestRuntime is not None
    assert ManualOriginationRuntime is not None
    assert SeasonalWeatherServiceRuntime is not None

def test_nwws_runtime_does_not_reference_removed_orchestrator_shims() -> None:
    import inspect

    from seasonalweather.broadcast import nwws_runtime

    source = inspect.getsource(nwws_runtime.NwwsRuntime)
    assert "self._vtec_matches_configured_toneout_code" not in source
    assert "self._nwws_same_targets_from_texts" not in source
    assert "self._nwws_wcn_watch_same_targets_from_area_desc" not in source
    assert "self._sf_area_text_from_same_codes" not in source

def test_runtime_modules_do_not_call_removed_same_targeting_orchestrator_shims() -> None:
    import inspect

    from seasonalweather.broadcast import (
        cap_runtime,
        ern_relay_runtime,
        ipaws_runtime,
        manual_runtime,
        tests_runtime,
    )

    modules = [
        cap_runtime,
        ern_relay_runtime,
        ipaws_runtime,
        manual_runtime,
        tests_runtime,
    ]
    forbidden = (
        "host._filter_same_locations_to_service_area",
        "o._filter_same_locations_to_service_area",
        "orch._filter_same_locations_to_service_area",
        "self.orch._filter_same_locations_to_service_area",
        "host._sf_area_text_from_same_codes",
        "o._sf_area_text_from_same_codes",
        "orch._sf_area_text_from_same_codes",
        "self.orch._sf_area_text_from_same_codes",
    )
    source = "\n".join(inspect.getsource(module) for module in modules)
    for needle in forbidden:
        assert needle not in source
