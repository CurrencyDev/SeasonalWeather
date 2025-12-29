from __future__ import annotations

import argparse
import json
import math
import re
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


# SAME / EAS AFSK parameters (spec)
SAME_BITRATE = 520.83          # bits/sec
SAME_MARK_HZ = 2083.3          # logic 1
SAME_SPACE_HZ = 1562.5         # logic 0
SAME_PREAMBLE_BYTE = 0xAB
SAME_PREAMBLE_LEN = 16         # bytes (16 * 0xAB)
SAME_MAX_LOCS = 31             # max PSSCCC blocks per header (spec)

# Header parsing (fairly strict, but sender allows spaces and '/')
HEADER_RE = re.compile(
    r"^ZCZC-"
    r"(?P<org>[A-Z0-9]{3})-"
    r"(?P<event>[A-Z0-9]{3})-"
    r"(?P<locs>[0-9]{6}(?:-[0-9]{6})*)\+"
    r"(?P<tttt>[0-9]{4})-"
    r"(?P<jjjhhmm>[0-9]{7})-"
    r"(?P<sender>[A-Z0-9 /]{8})-"
    r"$"
)

EPS = 1e-12


@dataclass(frozen=True)
class DecodedSAME:
    kind: str               # "header" or "eom"
    text: str               # raw decoded text (e.g., ZCZC-...- or NNNN)
    start_seconds: float    # approximate start time in file
    confidence: float       # 0..1-ish (higher = cleaner)

    # Parsed fields (header only)
    org: Optional[str] = None
    event: Optional[str] = None
    locations: Tuple[str, ...] = ()
    tttt: Optional[str] = None
    jjjhhmm: Optional[str] = None
    sender: Optional[str] = None


def _read_wav_mono(path: Path) -> Tuple[int, array]:
    """
    Read PCM WAV (8-bit unsigned or 16-bit signed) into mono float samples in [-1, 1].
    Lightweight and strict on purpose.
    """
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        nch = int(wf.getnchannels())
        sw = int(wf.getsampwidth())
        nframes = int(wf.getnframes())
        raw = wf.readframes(nframes)

    if sr <= 0:
        raise ValueError("Invalid sample rate")
    if nch <= 0:
        raise ValueError("Invalid channel count")
    if sw not in (1, 2):
        raise ValueError(f"Unsupported sample width {sw} (supported: 8-bit, 16-bit PCM)")

    mono = array("f")

    if sw == 1:
        # 8-bit unsigned PCM
        data = array("B")
        data.frombytes(raw)
        if nch == 1:
            for b in data:
                mono.append((b - 128) / 128.0)
        else:
            step = nch
            for i in range(0, len(data), step):
                acc = 0
                for c in range(step):
                    acc += int(data[i + c]) - 128
                mono.append((acc / step) / 128.0)

    else:
        # 16-bit signed PCM
        data16 = array("h")
        data16.frombytes(raw)
        # wave is little-endian; array('h') follows host endianness. On x86 this is fine.
        # If you ever run this on a big-endian system, you'd want byteswap().
        if nch == 1:
            for s in data16:
                mono.append(s / 32768.0)
        else:
            step = nch
            for i in range(0, len(data16), step):
                acc = 0
                for c in range(step):
                    acc += int(data16[i + c])
                mono.append((acc / step) / 32768.0)

    # Remove tiny DC offset (helps Goertzel a bit)
    if mono:
        mean = sum(mono) / float(len(mono))
        if abs(mean) > 1e-6:
            for i in range(len(mono)):
                mono[i] -= mean

    return sr, mono


def _coeff(freq_hz: float, sample_rate: int) -> float:
    # Goertzel coefficient
    w = (2.0 * math.pi * float(freq_hz)) / float(sample_rate)
    return 2.0 * math.cos(w)


def _goertzel_pair_power(
    x: Sequence[float],
    start: int,
    end: int,
    coeff_space: float,
    coeff_mark: float,
) -> Tuple[float, float]:
    """
    Compute Goertzel power for (space, mark) tones over x[start:end].
    One pass, two states.
    """
    s1s = s2s = 0.0
    s1m = s2m = 0.0

    for i in range(start, end):
        v = float(x[i])

        ns = v + coeff_space * s1s - s2s
        s2s = s1s
        s1s = ns

        nm = v + coeff_mark * s1m - s2m
        s2m = s1m
        s1m = nm

    ps = s2s * s2s + s1s * s1s - coeff_space * s1s * s2s
    pm = s2m * s2m + s1m * s1m - coeff_mark * s1m * s2m
    return ps, pm


class _BitReader:
    """
    Bit-timed reader matching SAME's 520.83 bps using a fractional-sample accumulator.
    This mirrors how your encoder renders bits, so boundaries stay aligned.
    """
    __slots__ = ("_x", "_sr", "_spb", "_i", "_acc", "_c_space", "_c_mark")

    def __init__(
        self,
        samples: Sequence[float],
        sample_rate: int,
        start_sample: int,
        *,
        coeff_space: float,
        coeff_mark: float,
    ) -> None:
        self._x = samples
        self._sr = int(sample_rate)
        self._spb = float(sample_rate) / SAME_BITRATE
        self._i = int(start_sample)
        self._acc = 0.0
        self._c_space = float(coeff_space)
        self._c_mark = float(coeff_mark)

    def tell(self) -> int:
        return self._i

    def read_bit(self) -> Optional[Tuple[int, float]]:
        # Determine bit window length using fractional accumulator
        self._acc += self._spb
        n = int(self._acc)
        self._acc -= n

        if n <= 0:
            return None

        j = self._i + n
        if j <= self._i or j > len(self._x):
            return None

        ps, pm = _goertzel_pair_power(self._x, self._i, j, self._c_space, self._c_mark)
        self._i = j

        tot = ps + pm + EPS
        conf = (pm - ps) / tot  # + => mark dominates
        bit = 1 if conf >= 0.0 else 0
        return bit, abs(conf)

    def read_byte_lsb_first(self) -> Optional[Tuple[int, float]]:
        b = 0
        csum = 0.0
        for k in range(8):
            r = self.read_bit()
            if r is None:
                return None
            bit, conf = r
            b |= (bit & 1) << k
            csum += conf
        return b, (csum / 8.0)


def _score_preamble(reader: _BitReader, *, bytes_to_check: int) -> Tuple[int, float]:
    good = 0
    csum = 0.0
    for _ in range(bytes_to_check):
        r = reader.read_byte_lsb_first()
        if r is None:
            break
        b, conf = r
        csum += conf
        if b == SAME_PREAMBLE_BYTE:
            good += 1
    avg_conf = csum / float(max(1, bytes_to_check))
    return good, avg_conf


def _find_candidate_regions(
    x: Sequence[float],
    sr: int,
    coeff_space: float,
    coeff_mark: float,
) -> List[Tuple[int, int]]:
    """
    Coarse scan: find regions with strong energy near SAME tones.
    Returns sample-index ranges [start, end).
    """
    if not x:
        return []

    block = max(256, int(sr * 0.02))     # 20 ms
    step = max(128, block // 2)          # 50% overlap

    scores: List[Tuple[int, float]] = []
    for i in range(0, len(x) - block, step):
        ps, pm = _goertzel_pair_power(x, i, i + block, coeff_space, coeff_mark)
        scores.append((i, ps + pm))

    if not scores:
        return []

    max_s = max(s for _, s in scores)
    if max_s <= 0:
        return []

    thr = max_s * 0.15  # burst-ish
    hot = [i for i, s in scores if s >= thr]
    if not hot:
        return []

    regions: List[Tuple[int, int]] = []
    cur_s = hot[0]
    cur_e = cur_s + block
    for i in hot[1:]:
        if i <= cur_e + step:
            cur_e = i + block
        else:
            regions.append((max(0, cur_s - block), min(len(x), cur_e + block)))
            cur_s = i
            cur_e = i + block

    regions.append((max(0, cur_s - block), min(len(x), cur_e + block)))
    return regions


def _try_decode_message_at(
    x: Sequence[float],
    sr: int,
    start_sample: int,
    coeff_space: float,
    coeff_mark: float,
    *,
    min_preamble_bytes: int,
    max_chars: int,
) -> Optional[DecodedSAME]:
    r = _BitReader(
        x,
        sr,
        start_sample,
        coeff_space=coeff_space,
        coeff_mark=coeff_mark,
    )

    # Full preamble check (and keep reader timing aligned for the payload)
    good, pre_conf = _score_preamble(r, bytes_to_check=SAME_PREAMBLE_LEN)
    if good < min_preamble_bytes:
        return None

    # Decode ASCII payload
    buf: List[str] = []
    csum = 0.0
    ccount = 0

    for _ in range(max_chars):
        rb = r.read_byte_lsb_first()
        if rb is None:
            break
        b, conf = rb
        csum += conf
        ccount += 1

        ch = chr(b & 0x7F)

        # SAME payload is generally printable ASCII. Stop on hard garbage.
        if ch in ("\r", "\n"):
            continue
        if not (32 <= ord(ch) <= 126):
            break

        buf.append(ch)

        s = "".join(buf)

        # EOM: preamble + "NNNN"
        if s == "NNNN":
            avg_conf = (pre_conf + (csum / max(1, ccount))) / 2.0
            return DecodedSAME(
                kind="eom",
                text="NNNN",
                start_seconds=float(start_sample) / float(sr),
                confidence=float(avg_conf),
            )

        # Header: ends with '-' and should parse cleanly
        if s.endswith("-") and s.startswith("Z"):
            m = HEADER_RE.match(s)
            if m:
                locs_raw = m.group("locs")
                locs = tuple(locs_raw.split("-"))[:SAME_MAX_LOCS]
                sender = (m.group("sender") or "").rstrip()
                avg_conf = (pre_conf + (csum / max(1, ccount))) / 2.0
                return DecodedSAME(
                    kind="header",
                    text=s,
                    start_seconds=float(start_sample) / float(sr),
                    confidence=float(avg_conf),
                    org=m.group("org"),
                    event=m.group("event"),
                    locations=locs,
                    tttt=m.group("tttt"),
                    jjjhhmm=m.group("jjjhhmm"),
                    sender=sender,
                )

        # Fast-fail: if the first few chars aren't plausible, bail early
        if len(buf) == 6:
            head = "".join(buf)
            if not (head.startswith("ZCZC-") or head.startswith("NNNN")):
                return None

    return None


def decode_same_wav(
    wav_path: Path,
    *,
    min_preamble_bytes: int = 14,
    max_chars: int = 320,
) -> List[DecodedSAME]:
    """
    Decode SAME headers/EOM from a WAV file.
    - min_preamble_bytes: how many of the 16 preamble bytes must match 0xAB
    - max_chars: safety cap for payload length
    """
    sr, x = _read_wav_mono(wav_path)
    if not x:
        return []

    c_space = _coeff(SAME_SPACE_HZ, sr)
    c_mark = _coeff(SAME_MARK_HZ, sr)

    regions = _find_candidate_regions(x, sr, c_space, c_mark)
    if not regions:
        return []

    spb = float(sr) / SAME_BITRATE
    fine_step = max(1, int(spb / 10.0))  # ~9-10 samples at 48k

    hits: List[DecodedSAME] = []

    for (rs, re_) in regions:
        # Give ourselves a little runway before the region to catch the true start
        rs2 = max(0, rs - int(sr * 0.25))
        re2 = min(len(x), re_ + int(sr * 0.05))

        # Fine scan for preamble starts
        last_accept = -10**9
        for s in range(rs2, re2, fine_step):
            # De-duplicate near-identical starts
            if s - last_accept < int(spb * 8):  # within ~1 byte window
                continue

            # Quick preamble sniff (first 4 bytes)
            r_quick = _BitReader(x, sr, s, coeff_space=c_space, coeff_mark=c_mark)
            good4, _ = _score_preamble(r_quick, bytes_to_check=4)
            if good4 < 3:
                continue

            msg = _try_decode_message_at(
                x,
                sr,
                s,
                c_space,
                c_mark,
                min_preamble_bytes=min_preamble_bytes,
                max_chars=max_chars,
            )
            if msg:
                hits.append(msg)
                last_accept = s

    # Deduplicate by (kind,text) and keep highest-confidence instance
    best: dict[Tuple[str, str], DecodedSAME] = {}
    for h in hits:
        key = (h.kind, h.text)
        cur = best.get(key)
        if cur is None or h.confidence > cur.confidence:
            best[key] = h

    out = sorted(best.values(), key=lambda d: d.start_seconds)
    return out


def _as_jsonable(d: DecodedSAME) -> dict:
    return {
        "kind": d.kind,
        "text": d.text,
        "start_seconds": d.start_seconds,
        "confidence": d.confidence,
        "org": d.org,
        "event": d.event,
        "locations": list(d.locations) if d.locations else [],
        "tttt": d.tttt,
        "jjjhhmm": d.jjjhhmm,
        "sender": d.sender,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Decode SAME headers/EOM from a WAV file (lightweight, stdlib-only).")
    p.add_argument("wav", type=str, help="Path to WAV file")
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.add_argument("--min-preamble-bytes", type=int, default=14, help="0..16 (default: 14)")
    p.add_argument("--max-chars", type=int, default=120, help="Max decoded payload chars (default: 120)")
    args = p.parse_args(argv)

    wav_path = Path(args.wav)
    msgs = decode_same_wav(
        wav_path,
        min_preamble_bytes=max(0, min(16, int(args.min_preamble_bytes))),
        max_chars=max(16, int(args.max_chars)),
    )

    if args.json:
        print(json.dumps([_as_jsonable(m) for m in msgs], indent=2, sort_keys=False))
        return 0

    if not msgs:
        print("No SAME messages found.")
        return 1

    for m in msgs:
        ts = f"{m.start_seconds:8.3f}s"
        q = f"{m.confidence:0.3f}"
        if m.kind == "eom":
            print(f"{ts}  q={q}  EOM: NNNN")
        else:
            locs = ",".join(m.locations[:5]) + ("..." if len(m.locations) > 5 else "")
            sender = (m.sender or "").strip()
            print(f"{ts}  q={q}  {m.org}-{m.event}  T={m.tttt}  J={m.jjjhhmm}  SENDER={sender!r}  LOCS={locs}")
            print(f"          {m.text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
