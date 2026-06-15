from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

log = logging.getLogger("seasonalweather.alert_audio")

AlertAudioMode = Literal["full", "voice"]
RenderCallable = Callable[[], Awaitable[Path]]
PushCallable = Callable[[Path], Awaitable[None]]
StaleCheck = Callable[[], bool]


@dataclass(order=True)
class _QueuedAlertAudioJob:
    priority: int
    sequence: int
    created_monotonic: float = field(compare=False)
    mode: AlertAudioMode = field(compare=False)
    source: str = field(compare=False)
    render: RenderCallable = field(compare=False)
    push: PushCallable = field(compare=False)
    future: asyncio.Future[Path] = field(compare=False)
    stale_check: StaleCheck | None = field(default=None, compare=False)


class AlertAudioDispatcher:
    """Priority worker for rendered interrupt audio.

    This intentionally starts small: it centralises alert audio render/push order
    without turning SeasonalWeather into a distributed job system.  The source
    runtimes may still await completion so their existing dedupe, tracker, and
    station-feed writes remain ordered and reversible.
    """

    FULL_PRIORITY = 0
    VOICE_PRIORITY = 10

    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[_QueuedAlertAudioJob] = asyncio.PriorityQueue()
        self._seq = itertools.count()
        self._started = False

    def start(self, tasks: list[asyncio.Task]) -> None:
        if self._started:
            return
        self._started = True
        tasks.append(asyncio.create_task(self.run_forever(), name="alert_audio_dispatcher"))
        log.info("AlertAudioDispatcher: started")

    async def run_forever(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._run_job(job)
            finally:
                self._queue.task_done()

    def pending_count(self) -> int:
        return self._queue.qsize()

    async def render_and_push_full(
        self,
        *,
        source: str,
        render: RenderCallable,
        push: PushCallable,
        stale_check: StaleCheck | None = None,
    ) -> Path:
        return await self.submit(
            priority=self.FULL_PRIORITY,
            mode="full",
            source=source,
            render=render,
            push=push,
            stale_check=stale_check,
        )

    async def render_and_push_voice(
        self,
        *,
        source: str,
        render: RenderCallable,
        push: PushCallable,
        stale_check: StaleCheck | None = None,
    ) -> Path:
        return await self.submit(
            priority=self.VOICE_PRIORITY,
            mode="voice",
            source=source,
            render=render,
            push=push,
            stale_check=stale_check,
        )

    async def submit(
        self,
        *,
        priority: int,
        mode: AlertAudioMode,
        source: str,
        render: RenderCallable,
        push: PushCallable,
        stale_check: StaleCheck | None = None,
    ) -> Path:
        if not self._started:
            # Unit tests and direct manual use do not always run service_runtime.
            # Preserve the old synchronous contract unless the worker has been
            # explicitly started by the service supervisor.
            return await self._run_inline(
                mode=mode,
                source=source,
                render=render,
                push=push,
                stale_check=stale_check,
            )

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Path] = loop.create_future()
        await self._queue.put(
            _QueuedAlertAudioJob(
                priority=int(priority),
                sequence=next(self._seq),
                created_monotonic=time.monotonic(),
                mode=mode,
                source=str(source or "unknown"),
                render=render,
                push=push,
                future=fut,
                stale_check=stale_check,
            )
        )
        return await fut

    async def _run_inline(
        self,
        *,
        mode: AlertAudioMode,
        source: str,
        render: RenderCallable,
        push: PushCallable,
        stale_check: StaleCheck | None,
    ) -> Path:
        if stale_check is not None and stale_check():
            raise RuntimeError(f"alert audio job stale before render: mode={mode} source={source}")
        path = await render()
        if stale_check is not None and stale_check():
            raise RuntimeError(f"alert audio job stale before push: mode={mode} source={source}")
        await push(path)
        return path

    async def _run_job(self, job: _QueuedAlertAudioJob) -> None:
        if job.future.cancelled():
            return
        wait_ms = int((time.monotonic() - job.created_monotonic) * 1000)
        try:
            if job.stale_check is not None and job.stale_check():
                raise RuntimeError(f"alert audio job stale before render: mode={job.mode} source={job.source}")

            render_start = time.monotonic()
            path = await job.render()
            render_ms = int((time.monotonic() - render_start) * 1000)

            if job.stale_check is not None and job.stale_check():
                raise RuntimeError(f"alert audio job stale before push: mode={job.mode} source={job.source}")

            push_start = time.monotonic()
            await job.push(path)
            push_ms = int((time.monotonic() - push_start) * 1000)

            if not job.future.cancelled():
                job.future.set_result(path)
            log.debug(
                "AlertAudioDispatcher: completed mode=%s source=%s wait_ms=%d render_ms=%d push_ms=%d audio=%s",
                job.mode,
                job.source,
                wait_ms,
                render_ms,
                push_ms,
                path,
            )
        except Exception as exc:
            if not job.future.cancelled():
                job.future.set_exception(exc)
            log.exception("AlertAudioDispatcher: failed mode=%s source=%s", job.mode, job.source)
