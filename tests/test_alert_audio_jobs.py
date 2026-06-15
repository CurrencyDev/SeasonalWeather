import asyncio
from pathlib import Path

from seasonalweather.broadcast.alert_audio_jobs import AlertAudioDispatcher


def test_alert_audio_dispatcher_runs_inline_until_started(tmp_path: Path) -> None:
    disp = AlertAudioDispatcher()
    events: list[str] = []

    async def render() -> Path:
        events.append("render")
        out = tmp_path / "voice.wav"
        out.write_bytes(b"RIFFfakeWAVE")
        return out

    async def push(path: Path) -> None:
        events.append(f"push:{path.name}")

    result = asyncio.run(disp.render_and_push_voice(source="test", render=render, push=push))

    assert result.name == "voice.wav"
    assert events == ["render", "push:voice.wav"]


def test_alert_audio_dispatcher_prioritizes_full_before_voice(tmp_path: Path) -> None:
    async def scenario() -> list[str]:
        disp = AlertAudioDispatcher()
        tasks: list[asyncio.Task] = []
        disp.start(tasks)
        events: list[str] = []

        async def render_voice() -> Path:
            events.append("render_voice")
            out = tmp_path / "voice.wav"
            out.write_bytes(b"RIFFfakeWAVE")
            return out

        async def render_full() -> Path:
            events.append("render_full")
            out = tmp_path / "full.wav"
            out.write_bytes(b"RIFFfakeWAVE")
            return out

        async def push(path: Path) -> None:
            events.append(f"push:{path.name}")

        voice_task = asyncio.create_task(
            disp.render_and_push_voice(source="voice", render=render_voice, push=push)
        )
        full_task = asyncio.create_task(
            disp.render_and_push_full(source="full", render=render_full, push=push)
        )

        await asyncio.gather(voice_task, full_task)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return events

    events = asyncio.run(scenario())

    assert events[:2] == ["render_full", "push:full.wav"]
    assert events[2:] == ["render_voice", "push:voice.wav"]
