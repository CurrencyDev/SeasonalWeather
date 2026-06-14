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
