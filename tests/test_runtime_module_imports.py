from seasonalweather.broadcast.cap_runtime import CapRuntime
from seasonalweather.broadcast.ipaws_runtime import IpawsRuntime
from seasonalweather.broadcast.nwws_runtime import NwwsRuntime
from seasonalweather.broadcast.pns_runtime import PnsRuntime


def test_source_runtime_modules_importable() -> None:
    assert CapRuntime is not None
    assert IpawsRuntime is not None
    assert NwwsRuntime is not None
    assert PnsRuntime is not None
