from __future__ import annotations

import wave
from datetime import datetime, timezone

from seasonalweather.same.same import SameHeader, chunk_locations, render_same_bursts_wav


def test_same_header_normalizes_codes_locations_and_sender() -> None:
    header = SameHeader(
        org="wxr-extra",
        event="svr!",
        locations=("24033", "024031", "bad"),
        duration_minutes=17,
        sender="seasonal-weather",
        issued_utc=datetime(2026, 5, 22, 13, 45, tzinfo=timezone.utc),
    )

    assert header.as_ascii() == "ZCZC-WXR-SVR-024033-024031+0030-1421345-SEASONAL-"


def test_same_header_falls_back_for_bad_three_char_codes() -> None:
    header = SameHeader(
        org="WX",
        event="S",
        locations=(),
        duration_minutes=15,
        sender="SEASNWXR",
        issued_utc=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
    )

    assert header.as_ascii() == "ZCZC-WXR-CEM-000000+0015-0010000-SEASNWXR-"


def test_chunk_locations_deduplicates_and_chunks_to_31() -> None:
    locs = [f"024{i:03d}" for i in range(1, 35)] + ["024001"]
    chunks = chunk_locations(locs)

    assert len(chunks) == 2
    assert len(chunks[0]) == 31
    assert len(chunks[1]) == 3
    assert chunks[0][0] == "024001"


def test_render_same_bursts_wav_writes_stereo_pcm(tmp_path) -> None:
    out = tmp_path / "same.wav"

    render_same_bursts_wav(
        out,
        "ZCZC-WXR-RWT-024033+0015-1421345-SEASNWXR-",
        sample_rate=48000,
        burst_count=1,
        inter_burst_pause_seconds=0.0,
    )

    with wave.open(str(out), "rb") as wf:
        assert wf.getnchannels() == 2
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 48000
        assert wf.getnframes() > 0
