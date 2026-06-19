import asyncio

from seasonalweather.broadcast import conductor as conductor_module
from seasonalweather.broadcast.conductor import CycleConductor


class _FakeTelnet:
    def __init__(self, *, active=None, reset_ok=True) -> None:
        self.active = active
        self.reset_ok = reset_ok
        self.status_calls = 0
        self.reset_calls = 0

    def interrupt_active(self):
        self.status_calls += 1
        return self.active

    def reset_cycle_safely(self) -> bool:
        self.reset_calls += 1
        return self.reset_ok


def _bare_conductor(telnet: _FakeTelnet) -> CycleConductor:
    conductor = CycleConductor.__new__(CycleConductor)
    conductor._telnet = telnet
    conductor._flush_event = asyncio.Event()
    conductor._interrupt_hold = False
    conductor._interrupt_expected_end = 0.0
    conductor._interrupt_reason = ""
    conductor._push_start_ts = 1.0
    conductor._total_pushed_s = 123.0
    conductor._cycle_order = ["id", "time", "obs"]
    conductor._position_in_rotation = 2
    return conductor


def test_interrupt_admission_holds_cycle_and_accumulates_serial_audio(monkeypatch) -> None:
    now = iter((100.0, 105.0))
    monkeypatch.setattr(conductor_module.time, "monotonic", lambda: next(now))
    conductor = _bare_conductor(_FakeTelnet())

    conductor.notify_interrupt_started(duration_s=20.0, reason="voice-interrupt")
    assert conductor._interrupt_hold is True
    assert conductor._interrupt_expected_end == 120.0
    assert conductor._total_pushed_s == 0.0
    assert conductor._cycle_order == []
    assert conductor._position_in_rotation == 0

    # A FULL alert admitted five seconds later preempts/resumes the VOICE plane,
    # so its duration is additive rather than replacing the existing deadline.
    conductor.notify_interrupt_started(duration_s=30.0, reason="full-interrupt")
    assert conductor._interrupt_expected_end == 150.0


def test_interrupt_release_resets_cycle_and_requests_immediate_refill(monkeypatch) -> None:
    monkeypatch.setattr(conductor_module.time, "monotonic", lambda: 200.0)
    telnet = _FakeTelnet(active=False)
    conductor = _bare_conductor(telnet)
    conductor._interrupt_hold = True
    conductor._interrupt_expected_end = 199.0
    conductor._interrupt_reason = "full-interrupt"

    asyncio.run(conductor._wait_for_interrupt_end())

    assert telnet.status_calls == 1
    assert telnet.reset_calls == 1
    assert conductor._interrupt_hold is False
    assert conductor._interrupt_expected_end == 0.0
    assert conductor._interrupt_reason == ""
    assert conductor._total_pushed_s == 0.0
    assert conductor._cycle_order == []
    assert conductor._position_in_rotation == 0
    assert conductor._flush_event.is_set()


def test_interrupt_release_waits_while_liquidsoap_still_has_alert_audio(monkeypatch) -> None:
    monkeypatch.setattr(conductor_module.time, "monotonic", lambda: 300.0)
    telnet = _FakeTelnet(active=True)
    conductor = _bare_conductor(telnet)
    conductor._interrupt_hold = True
    conductor._interrupt_expected_end = 299.0

    asyncio.run(conductor._wait_for_interrupt_end())

    assert telnet.status_calls == 1
    assert telnet.reset_calls == 0
    assert conductor._interrupt_hold is True
    assert conductor._interrupt_expected_end == 300.5
