# =========================================================================================
#      MP"""""`MM                                                       dP              MM'"""'YMM
#      M  mmmmm..M                                                       88              M' .mmm. `M
#      M.      `YM .d8888b. .d8888b. .d8888b. .d8888b. 88d888b. .d8888b. 88              M  MMMMMooM dP    dP 88d888b. 88d888b. .d8888b. 88d888b. .d8888b. dP    dP
#      MMMMMMM.  M 88ooood8 88'  `88 Y8ooooo. 88'  `88 88'  `88 88'  `88 88              M  MMMMMMMM 88    88 88'  `88 88'  `88 88ooood8 88'  `88 88'  `"" 88    88
#      M. .MMM'  M 88.  ... 88.  .88       88 88.  .88 88    88 88.  .88 88              M. `MMM' .M 88.  .88 88       88       88.  ... 88    88 88.  ... 88.  .88
#      Mb.     .dM `88888P' `88888P8 `88888P' `88888P' dP    dP `88888P8 dP              MM.     .dM `88888P' dP       dP       `88888P' dP    dP `88888P' `8888P88
#      MMMMMMMMMMM                                                Seasonal_Currency      MMMMMMMMMMM                                                            .88
#                                                                                                                                                           d8888P.
# =========================================================================================

from __future__ import annotations

import math
import wave
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


# NWSI 10-1712 Appendix A (SAME on NWR)
SAME_BITRATE = 520.83               # bits per second
SAME_BIT_PERIOD = 1.0 / SAME_BITRATE
SAME_FREQ_ZERO = 1562.5             # logic 0
SAME_FREQ_ONE = 2083.3              # logic 1
SAME_PREAMBLE_BYTE = 0xAB
SAME_PREAMBLE_LEN = 16              # bytes
SAME_MAX_LOCS = 31                  # max PSSCCC blocks per header

DEFAULT_ORG = "WXR"
DEFAULT_SENDER = "SEASNWXR"         # must be 8 chars in the header; we clamp/pad


def _clamp_sender(sender: str) -> str:
    s = (sender or "").strip().upper()
    if not s:
        s = DEFAULT_SENDER
    s = "".join(ch for ch in s if ch.isalnum())
    if len(s) >= 8:
        return s[:8]
    return (s + (" " * 8))[:8]


def _julian_day(dt_utc: datetime) -> int:
    return int(dt_utc.timetuple().tm_yday)


def _round_up_to_15_minutes(minutes: int) -> int:
    m = max(0, int(minutes))
    if m % 15 == 0:
        return m
    return m + (15 - (m % 15))


def _format_tttt(duration_minutes: int) -> str:
    # TTTT = HHMM, minutes are typically multiples of 15 for SAME
    m = _round_up_to_15_minutes(duration_minutes)
    h = m // 60
    mm = m % 60
    h = max(0, min(99, h))
    mm = max(0, min(59, mm))
    return f"{h:02d}{mm:02d}"


def chunk_locations(locs: Sequence[str], max_per: int = SAME_MAX_LOCS) -> List[List[str]]:
    cleaned: List[str] = []
    for x in locs or []:
        s = str(x).strip()
        if not s:
            continue
        # SAME PSSCCC are 6 digits (P + state + county); keep numeric only, zero-pad
        s = "".join(ch for ch in s if ch.isdigit())
        if not s:
            continue
        cleaned.append(s.zfill(6)[:6])
    cleaned = sorted(set(cleaned))
    if not cleaned:
        return [[]]
    out: List[List[str]] = []
    for i in range(0, len(cleaned), max_per):
        out.append(cleaned[i : i + max_per])
    return out


@dataclass(frozen=True)
class SameHeader:
    org: str
    event: str
    locations: Tuple[str, ...]
    duration_minutes: int
    sender: str
    issued_utc: datetime

    def as_ascii(self) -> str:
        org = (self.org or DEFAULT_ORG).strip().upper()[:3]
        event = (self.event or "CEM").strip().upper()[:3]

        dt = self.issued_utc.astimezone(timezone.utc)
        jjj = _julian_day(dt)
        hhmm = f"{dt.hour:02d}{dt.minute:02d}"
        tttt = _format_tttt(self.duration_minutes)
        llllllll = _clamp_sender(self.sender)

        # ZCZC-ORG-EEE-PSSCCC-PSSCCC+TTTT-JJJHHMM-LLLLLLLL-
        # NOTE: The spec’s spaces are for clarity only; do not include spaces.
        loc_part = "-".join(self.locations) if self.locations else "000000"
        return f"ZCZC-{org}-{event}-{loc_part}+{tttt}-{jjj:03d}{hhmm}-{llllllll}-"


def _bytes_lsb_first(bits: Iterable[int]) -> List[int]:
    # helper for testing / clarity; unused in final render path
    return [1 if b else 0 for b in bits]


def _iter_afsk_bits_for_bytes(payload: bytes) -> Iterable[int]:
    # Each byte is sent LSB first, 8 bits, no start/stop/parity.
    for byte in payload:
        for i in range(8):
            yield (byte >> i) & 0x01


def _render_afsk_bits_to_pcm(
    bits: Iterable[int],
    *,
    sample_rate: int,
    amplitude: float,
) -> array:
    """
    Render SAME AFSK as 16-bit PCM mono samples.
    Uses a fractional-sample accumulator so average bitrate stays correct at e.g. 48kHz.
    """
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError("sample_rate must be > 0")

    amp = float(amplitude)
    amp = max(0.0, min(1.0, amp))
    peak = int(32767 * amp)

    samples = array("h")
    phase = 0.0

    spb = sr / SAME_BITRATE  # samples per bit (fractional at 48k)
    acc = 0.0

    for bit in bits:
        freq = SAME_FREQ_ONE if bit else SAME_FREQ_ZERO
        acc += spb
        n = int(acc)
        acc -= n
        if n <= 0:
            continue

        step = (2.0 * math.pi * freq) / sr
        for _ in range(n):
            phase += step
            # keep phase bounded
            if phase > 1e9:
                phase = math.fmod(phase, 2.0 * math.pi)
            v = math.sin(phase)
            samples.append(int(v * peak))

    return samples


def _write_pcm_mono_to_stereo_wav(path: Path, pcm_mono: array, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(sample_rate))

        # interleave mono -> stereo
        out = array("h")
        out.extend((0, 0))  # tiny pad for some picky decoders
        for s in pcm_mono:
            out.append(s)
            out.append(s)

        wf.writeframes(out.tobytes())


def _silence_pcm(seconds: float, sample_rate: int) -> array:
    n = int(max(0.0, float(seconds)) * int(sample_rate))
    return array("h", [0] * n)


def _concat_pcm(parts: Sequence[array]) -> array:
    out = array("h")
    for p in parts:
        out.extend(p)
    return out


def render_same_bursts_wav(
    out_wav: Path,
    message_ascii: str,
    *,
    sample_rate: int,
    amplitude: float = 0.35,
    burst_count: int = 3,
    inter_burst_pause_seconds: float = 1.0,
) -> None:
    """
    Render a SAME message as NWR-style bursts:
      (preamble + ascii header) x3 with 1s pauses
    """
    msg = (message_ascii or "").strip()
    if not msg:
        raise ValueError("message_ascii is empty")

    # Preamble + 7-bit ASCII (8th bit always 0 -> mask with 0x7F)
    pre = bytes([SAME_PREAMBLE_BYTE] * SAME_PREAMBLE_LEN)
    payload = bytes((ord(ch) & 0x7F) for ch in msg)

    bits = list(_iter_afsk_bits_for_bytes(pre + payload))
    burst_pcm = _render_afsk_bits_to_pcm(bits, sample_rate=sample_rate, amplitude=amplitude)
    pause_pcm = _silence_pcm(inter_burst_pause_seconds, sample_rate)

    parts: List[array] = []
    for i in range(max(1, int(burst_count))):
        parts.append(burst_pcm)
        if i != burst_count - 1:
            parts.append(pause_pcm)

    pcm = _concat_pcm(parts)
    _write_pcm_mono_to_stereo_wav(out_wav, pcm, sample_rate)


def render_same_eom_wav(
    out_wav: Path,
    *,
    sample_rate: int,
    amplitude: float = 0.35,
    burst_count: int = 3,
    inter_burst_pause_seconds: float = 1.0,
) -> None:
    # EOM is “NNNN” with preamble, repeated three times
    render_same_bursts_wav(
        out_wav,
        "NNNN",
        sample_rate=sample_rate,
        amplitude=amplitude,
        burst_count=burst_count,
        inter_burst_pause_seconds=inter_burst_pause_seconds,
    )
