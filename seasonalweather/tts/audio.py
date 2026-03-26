from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Iterable


def write_sine_wav(path: Path, freq_hz: float, seconds: float, sample_rate: int, amplitude: float = 0.25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(seconds * sample_rate)
    amp = max(0.0, min(1.0, amplitude))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        for i in range(n_frames):
            t = i / sample_rate
            v = math.sin(2 * math.pi * freq_hz * t)
            s = int(v * 32767 * amp)
            wf.writeframesraw(s.to_bytes(2, "little", signed=True) * 2)


def write_silence_wav(path: Path, seconds: float, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * 2 * n_frames)


def wav_duration_seconds(path: Path) -> float:
    """
    Fast duration probe (no ffmpeg). Assumes a valid WAV file.
    """
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            return 0.0
        return float(frames) / float(rate)


def concat_wavs(out_path: Path, parts: Iterable[Path]) -> None:
    parts = [Path(p) for p in parts]
    if not parts:
        raise ValueError("concat_wavs: no parts")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(parts[0]), "rb") as r0:
        nch = r0.getnchannels()
        sw = r0.getsampwidth()
        rate = r0.getframerate()

    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(sw)
        w.setframerate(rate)
        for p in parts:
            with wave.open(str(p), "rb") as r:
                if r.getnchannels() != nch or r.getsampwidth() != sw or r.getframerate() != rate:
                    raise ValueError(f"WAV format mismatch: {p}")
                while True:
                    frames = r.readframes(8192)
                    if not frames:
                        break
                    w.writeframes(frames)
