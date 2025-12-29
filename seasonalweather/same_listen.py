from __future__ import annotations

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

import argparse
import json
import math
import os
import select
import subprocess
import sys
import time
from array import array
from dataclasses import asdict
from typing import Iterable, Iterator

from . import same_decode as sd

DecodedSAME = sd.DecodedSAME


def decode_same_from_samples(
    samples,
    sr: int,
    *,
    min_preamble_bytes: int = 14,
    max_chars: int = 320,
):
    if samples is None:
        return None
    try:
        n = len(samples)
    except Exception:
        return None
    if n <= 0:
        return None

    c_space = sd._coeff(sd.SAME_SPACE_HZ, sr)
    c_mark = sd._coeff(sd.SAME_MARK_HZ, sr)

    regions = sd._find_candidate_regions(samples, sr, c_space, c_mark)
    if not regions:
        return None

    spb = float(sr) / sd.SAME_BITRATE
    fine_step = max(1, int(spb / 10.0))

    best = None
    best_conf = -1.0

    for (rs, re_) in regions:
        rs2 = max(0, rs - int(sr * 0.25))
        re2 = min(n, re_ + int(sr * 0.05))

        last_accept = -10**9
        for start in range(rs2, re2, fine_step):
            if start - last_accept < int(spb * 8):
                continue

            r_quick = sd._BitReader(samples, sr, start, coeff_space=c_space, coeff_mark=c_mark)
            good4, _ = sd._score_preamble(r_quick, bytes_to_check=4)
            if good4 < 3:
                continue

            msg = sd._try_decode_message_at(
                samples,
                sr,
                start,
                c_space,
                c_mark,
                min_preamble_bytes=min_preamble_bytes,
                max_chars=max_chars,
            )
            if msg is not None:
                last_accept = start
                if msg.confidence > best_conf:
                    best = msg
                    best_conf = msg.confidence

    return best


try:
    from .same import SAME_FREQ_ONE as SAME_MARK_HZ  # type: ignore
    from .same import SAME_FREQ_ZERO as SAME_SPACE_HZ  # type: ignore
except Exception:
    SAME_MARK_HZ = 2083.3
    SAME_SPACE_HZ = 1562.5

BYTES_PER_SAMPLE = 2  # s16le mono


def _force_line_buffered_stdout() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _goertzel_power(samples: Iterable[float], sr: int, freq_hz: float) -> float:
    w = 2.0 * math.pi * (freq_hz / float(sr))
    coeff = 2.0 * math.cos(w)
    s_prev = 0.0
    s_prev2 = 0.0
    for x in samples:
        s = x + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    return s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2


def _pcm16le_bytes_to_floats(chunk: bytes) -> array:
    a = array("h")
    a.frombytes(chunk)
    out = array("f")
    inv = 1.0 / 32768.0
    for v in a:
        out.append(v * inv)
    return out


def _spawn_ffmpeg(url: str, sr: int, debug: bool = False) -> subprocess.Popen:
    stderr = None if debug else subprocess.DEVNULL
    return subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            url,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "s16le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=stderr,
        bufsize=0,
    )


def _iter_pcm_chunks(proc: subprocess.Popen, chunk_bytes: int, *, idle_timeout_seconds: float = 45.0) -> Iterator[bytes]:
    """
    Yields aligned PCM chunks, and restarts ffmpeg if it goes silent (no stdout bytes).
    """
    assert proc.stdout is not None
    fd = proc.stdout.fileno()

    carry = b""
    last_data_at = time.monotonic()

    while True:
        if proc.poll() is not None:
            return

        r, _, _ = select.select([fd], [], [], 1.0)
        if not r:
            if (time.monotonic() - last_data_at) > float(idle_timeout_seconds):
                raise TimeoutError(f"ffmpeg stalled: no PCM bytes for {idle_timeout_seconds}s")
            continue

        data = os.read(fd, chunk_bytes)
        if not data:
            return

        last_data_at = time.monotonic()

        data = carry + data
        rem = len(data) % BYTES_PER_SAMPLE
        if rem:
            carry = data[-rem:]
            data = data[:-rem]
        else:
            carry = b""

        if data:
            yield data


def _emit(obj: dict, jsonl: bool) -> None:
    if jsonl:
        print(json.dumps(obj, ensure_ascii=False), flush=True)
    else:
        kind = obj.get("kind", "?")
        txt = obj.get("text", "")
        conf = obj.get("confidence", 0.0)
        print(f"{kind} conf={conf:.3f} {txt}", flush=True)


def listen_stream(
    url: str,
    sr: int = 48000,
    tail_seconds: float = 10.0,
    dedupe_seconds: float = 20.0,
    trigger_ratio: float = 8.0,
    jsonl: bool = False,
    debug: bool = False,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    hot_frames_needed: int = 3,
    lookback_seconds: float = 2.5,
    lookahead_seconds: float = 0.3,
    min_preamble_bytes: int = 12,
    idle_bytes_seconds: float = 45.0,
) -> None:
    mark = float(SAME_MARK_HZ)
    space = float(SAME_SPACE_HZ)

    frame_len = max(1, int(sr * (frame_ms / 1000.0)))
    hop_len = max(1, int(sr * (hop_ms / 1000.0)))

    max_buf = int(sr * tail_seconds)

    baseline = 1e-12
    baseline_alpha = 0.001

    last_emit_at: dict[tuple[str, str], float] = {}

    while True:
        proc: subprocess.Popen | None = None
        try:
            proc = _spawn_ffmpeg(url, sr, debug=debug)

            if debug:
                print(f"[same_listen] ffmpeg started for {url} @ {sr}Hz", file=sys.stderr, flush=True)

            buf = array("f")
            base_idx = 0

            frame_cursor = 0
            hot = 0

            chunk_bytes = 16384

            for chunk in _iter_pcm_chunks(proc, chunk_bytes, idle_timeout_seconds=idle_bytes_seconds):
                floats = _pcm16le_bytes_to_floats(chunk)
                buf.extend(floats)

                if len(buf) > max_buf + sr:
                    drop = len(buf) - (max_buf + sr)
                    del buf[:drop]
                    base_idx += drop

                while frame_cursor + frame_len <= base_idx + len(buf):
                    local_start = frame_cursor - base_idx
                    if local_start < 0:
                        frame_cursor = base_idx
                        continue

                    frame = buf[local_start : local_start + frame_len]

                    p_mark = _goertzel_power(frame, sr, mark)
                    p_space = _goertzel_power(frame, sr, space)
                    tone_energy = p_mark + p_space

                    baseline = (1.0 - baseline_alpha) * baseline + baseline_alpha * tone_energy

                    abs_floor = 1e-10
                    thresh = max(abs_floor, baseline * trigger_ratio)

                    if tone_energy >= thresh:
                        hot += 1
                    else:
                        hot = 0

                    if hot >= hot_frames_needed:
                        win_end = frame_cursor + frame_len + int(sr * lookahead_seconds)
                        win_start = max(base_idx, win_end - int(sr * lookback_seconds))

                        local_ws = win_start - base_idx
                        local_we = min(len(buf), win_end - base_idx)

                        if local_we > local_ws + int(sr * 0.2):
                            window = buf[local_ws:local_we]

                            decoded: DecodedSAME | None = decode_same_from_samples(
                                window,
                                sr=sr,
                                min_preamble_bytes=min_preamble_bytes,
                            )

                            if decoded is not None:
                                now = time.time()
                                key = (decoded.kind, decoded.text)
                                last = last_emit_at.get(key, 0.0)

                                if (now - last) >= dedupe_seconds:
                                    obj = asdict(decoded)
                                    obj["stream_sample_index"] = int(win_start)
                                    obj["start_seconds"] = float(win_start) / float(sr)
                                    obj["url"] = url
                                    obj["sr"] = int(sr)

                                    _emit(obj, jsonl=jsonl)
                                    last_emit_at[key] = now

                        hot = 0
                        frame_cursor = win_end
                        continue

                    frame_cursor += hop_len

            if debug:
                print("[same_listen] ffmpeg EOF / stream ended, restarting soon...", file=sys.stderr, flush=True)

        except KeyboardInterrupt:
            if debug:
                print("[same_listen] KeyboardInterrupt, exiting.", file=sys.stderr, flush=True)
            return
        except Exception as e:
            print(f"[same_listen] ERROR: {e!r} (restarting)", file=sys.stderr, flush=True)
        finally:
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass

        time.sleep(1.0)


def main() -> None:
    _force_line_buffered_stdout()

    ap = argparse.ArgumentParser(description="Listen to an audio stream and decode SAME (EAS/SAME) headers/EOM.")
    ap.add_argument("--url", required=True, help="Stream URL (http(s)://...)")
    ap.add_argument("--sr", type=int, default=48000, help="Decode sample rate (Hz)")
    ap.add_argument("--tail", type=float, default=10.0, help="Seconds of audio to retain for backtracking")
    ap.add_argument("--dedupe", type=float, default=20.0, help="Seconds to suppress duplicate (kind,text)")
    ap.add_argument("--trigger-ratio", type=float, default=8.0, help="Trigger ratio above baseline tone energy")
    ap.add_argument("--jsonl", action="store_true", help="Emit JSON lines (one per decode)")
    ap.add_argument("--debug", action="store_true", help="More stderr logging")
    ap.add_argument("--min-preamble-bytes", type=int, default=12, help="Min AB bytes required to accept a preamble")
    ap.add_argument("--idle-bytes", type=float, default=45.0, help="Restart ffmpeg if no PCM bytes arrive for this many seconds")
    args = ap.parse_args()

    listen_stream(
        url=args.url,
        sr=args.sr,
        tail_seconds=args.tail,
        dedupe_seconds=args.dedupe,
        trigger_ratio=args.trigger_ratio,
        jsonl=args.jsonl,
        debug=args.debug,
        min_preamble_bytes=args.min_preamble_bytes,
        idle_bytes_seconds=args.idle_bytes,
    )


if __name__ == "__main__":
    main()
