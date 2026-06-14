from seasonalweather.broadcast.cap_runtime import CapRuntime
from seasonalweather.broadcast.ipaws_runtime import IpawsRuntime


def test_source_runtime_modules_importable() -> None:
    assert CapRuntime is not None
    assert IpawsRuntime is not None
