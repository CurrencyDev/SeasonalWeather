"""Deterministic in-memory SWWP peers and bounded transport fault controls."""

from __future__ import annotations

import datetime as dt
from collections import deque
from dataclasses import dataclass, field

from seasonalweather.capabilities.hysteresis import (
    CapabilityHysteresis,
    CapabilityObservation,
)
from seasonalweather.swwp.capability_adapter import record_to_wire
from seasonalweather.swwp.controller import ControllerSession
from seasonalweather.swwp.messages import Envelope
from seasonalweather.swwp.worker import WorkerSession


class SimulatedClock:
    def __init__(self, now: dt.datetime) -> None:
        self.now = now

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += dt.timedelta(seconds=seconds)


class DeterministicIds:
    def __init__(self) -> None:
        self._next = 0

    def __call__(self, kind: str) -> str:
        self._next += 1
        return f"{kind}_{self._next:08d}"


@dataclass
class SimulatedTransport:
    maximum_frames: int = 128
    controller_inbox: deque[Envelope] = field(default_factory=deque)
    worker_inbox: deque[Envelope] = field(default_factory=deque)
    connected: bool = True

    def _put(self, queue: deque[Envelope], frame: Envelope) -> None:
        if not self.connected:
            return
        if len(queue) >= self.maximum_frames:
            raise OverflowError("simulated SWWP queue is full")
        queue.append(frame)

    def to_controller(self, frame: Envelope) -> None:
        self._put(self.controller_inbox, frame)

    def to_worker(self, frame: Envelope) -> None:
        self._put(self.worker_inbox, frame)

    def duplicate_to_controller(self, index: int = 0) -> None:
        self._put(self.controller_inbox, self.controller_inbox[index])

    def duplicate_to_worker(self, index: int = 0) -> None:
        self._put(self.worker_inbox, self.worker_inbox[index])

    def reorder_controller(self) -> None:
        self.controller_inbox.reverse()

    def reorder_worker(self) -> None:
        self.worker_inbox.reverse()

    def drop_controller(self, index: int = 0) -> Envelope:
        frame = self.controller_inbox[index]
        del self.controller_inbox[index]
        return frame

    def drop_worker(self, index: int = 0) -> Envelope:
        frame = self.worker_inbox[index]
        del self.worker_inbox[index]
        return frame

    def disconnect(self) -> None:
        self.connected = False

    def reconnect(self) -> None:
        self.connected = True


@dataclass
class SimulatedPeers:
    controller: ControllerSession
    worker: WorkerSession
    transport: SimulatedTransport = field(default_factory=SimulatedTransport)

    def start(self) -> None:
        self.transport.to_controller(self.worker.connect())

    def pump_once(self) -> bool:
        if self.transport.controller_inbox:
            incoming = self.transport.controller_inbox.popleft()
            for response in self.controller.receive(incoming):
                self.transport.to_worker(response)
            return True
        if self.transport.worker_inbox:
            incoming = self.transport.worker_inbox.popleft()
            for response in self.worker.receive(incoming):
                self.transport.to_controller(response)
            return True
        return False

    def pump(self, limit: int = 128) -> None:
        for _ in range(limit):
            if not self.pump_once():
                return
        raise RuntimeError("simulated SWWP pump exceeded bounded steps")

    def deliver_assignment(self) -> Envelope | None:
        frame = self.controller.assign_next()
        if frame is not None:
            self.transport.to_worker(frame)
        return frame

    def disconnect(self) -> None:
        self.transport.disconnect()
        self.controller.transport_lost()
        self.worker.transport_lost()


@dataclass
class SimulatedCapabilityObserver:
    worker: WorkerSession
    machines: dict[str, CapabilityHysteresis]
    validity_seconds: int = 60

    def observe(
        self,
        name: str,
        observation: CapabilityObservation,
    ) -> Envelope | None:
        published = self.machines[name].observe(observation)
        if published is None:
            return None
        return self.worker.capability_update(
            changed=(record_to_wire(published),),
            validity_seconds=self.validity_seconds,
        )
