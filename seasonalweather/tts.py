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

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

_SPACE_RE = re.compile(r"[ \t]+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)", re.IGNORECASE)
_ANGLE_URL_RE = re.compile(r"<(https?://[^>]+)>", re.IGNORECASE)

# Common “NWS product wrapper” / footer junk we do NOT want spoken
_SKIP_LINE_RE = re.compile(r"^\s*(?:\$\$|&&|NNNN|0{3,})\s*$")

# WMO-style header line (ex: FXUS61 KLWX 201925)
_WMO_HEADER_RE = re.compile(r"^[A-Z]{3,6}\d{2}\s+[A-Z]{4}\s+\d{6}(?:\s+[A-Z]{3})?$")

# A line that is mostly uppercase/digits/punct and short -> often metadata, not prose
_METAISH_RE = re.compile(r"^[A-Z0-9 \-\/\.\(\):;,+#]{1,50}$")

# Stuff that tends to appear in footers
_FOOTER_PREFIXES = (
    "visit us at",
    "for more information",
    "follow us on",
    "facebook",
    "twitter",
    "youtube",
    "weather.gov",
)

# HWO section lines like:
#   .DAY ONE...Tonight
#   .DAYS TWO THROUGH SEVEN...Sunday through Friday
#   .SPOTTER INFORMATION STATEMENT...
_HWO_SECTION_LINE_RE = re.compile(r"^\.(?P<title>[^.]+?)\.\.\.(?P<rest>.*)$")

# Zone-code-ish line like: MDZ003>006-503-505-...-212115-
_ZONEISH_CHARS_RE = re.compile(r"^[A-Z0-9>\-.,]+$")


def _looks_like_hwo(text: str) -> bool:
    low = (text or "").lower()
    if "hazardous weather outlook" in low:
        return True
    if re.search(
        r"^\.(day one|days two through seven|spotter information statement)\.\.\.",
        text or "",
        re.IGNORECASE | re.MULTILINE,
    ):
        return True
    return False


def _is_zoneish_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if " " in s:
        return False
    if len(s) < 10:
        return False
    if not _ZONEISH_CHARS_RE.fullmatch(s):
        return False
    if "-" not in s and ">" not in s:
        return False
    if not any(ch.isdigit() for ch in s):
        return False
    # common pattern like MDZ003 / VAZ053 / WVZ050 / ANZ530 etc
    if not re.search(r"[A-Z]{2,4}\d{3}", s):
        return False
    return True


def clean_for_tts(text: str) -> str:
    """
    De-noise NWS-ish content without rewriting meaning too aggressively.
    This runs on *everything* before it gets spoken, so keep it conservative.
    """
    if not text:
        return ""

    # Normalize newlines first (DECtalk clause/line mode benefits from real line breaks)
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # HWO needs special handling: it's ALL CAPS + huge zone blocks that should not be spoken.
    is_hwo = _looks_like_hwo(t)
    skip_hwo_zone_block = False

    # Strip a few common formatting artifacts
    t = t.replace("*", " ")
    t = t.replace("\u2022", " ")  # bullet
    t = t.replace("\u2013", "-")  # en-dash
    t = t.replace("\u2014", "-")  # em-dash

    # Markdown links: [label](url) -> label
    t = _MD_LINK_RE.sub(r"\1", t)
    # <https://...> -> (remove)
    t = _ANGLE_URL_RE.sub("", t)

    lines_out: list[str] = []
    for raw in t.split("\n"):
        line = raw.strip()
        if not line:
            continue

        # Kill pure control/footer markers
        if _SKIP_LINE_RE.match(line):
            continue

        # Remove URLs that are embedded in a line
        line = _URL_RE.sub("", line).strip()

        # Drop "link" or similar orphan words after URL removal
        if line.lower() in {"link", "links"}:
            continue

        # Drop obvious WMO header lines
        if _WMO_HEADER_RE.match(line):
            continue

        low = line.lower()

        # Drop very common footer lines
        if any(low.startswith(p) for p in _FOOTER_PREFIXES):
            continue

        # HWO: skip the entire zone-name wall between the zone-code line and the start of prose
        # (LWX HWOs repeat this block for multiple area groupings.)
        if is_hwo:
            if skip_hwo_zone_block:
                if low.startswith("this hazardous weather outlook is for"):
                    skip_hwo_zone_block = False
                    # fall through and process this line
                else:
                    continue

            if _is_zoneish_line(line):
                skip_hwo_zone_block = True
                continue

            # HWO: make section lines speakable
            if line.startswith("."):
                m = _HWO_SECTION_LINE_RE.match(line)
                if m:
                    title = m.group("title").strip().title()
                    rest = (m.group("rest") or "").strip()
                    if rest:
                        if not rest.endswith((".", "!", "?")):
                            rest += "."
                        line = f"{title}. {rest}"
                    else:
                        line = f"{title}."

        # Drop short “meta-ish” lines that look like headers, IDs, or routing metadata
        # BUT: HWO prose is frequently ALL CAPS per-line. Don't delete it just for being uppercase.
        if _METAISH_RE.match(line) and (" " not in line or line == line.upper()):
            if not is_hwo:
                # Try to keep things that look like real sentences
                if not any(ch in line for ch in (".", ",", "!", "?", "'")):
                    continue
            else:
                # In HWO, only drop meta-ish if it's a pure token (no spaces)
                if " " not in line and not any(ch in line for ch in (".", ",", "!", "?", "'")):
                    continue

        # Normalize whitespace *within* the line
        line = _SPACE_RE.sub(" ", line).strip()
        lines_out.append(line)

    # Keep line breaks for pacing, but collapse excessive blankness
    out = "\n".join(lines_out).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def _festival_voice_expr(voice: str) -> str:
    """
    Accept:
      - kal_diphone
      - voice_kal_diphone
      - (voice_kal_diphone)
    Return a safe Festival expression like: (voice_kal_diphone)
    """
    v = (voice or "").strip()
    if not v:
        v = "kal_diphone"

    if v.startswith("(") and v.endswith(")"):
        v = v[1:-1].strip()

    if not v.startswith("voice_"):
        v = f"voice_{v}"

    return f"({v})"


def _duration_stretch_from_wpm(rate_wpm: int, baseline_wpm: int = 175) -> float:
    """
    Festival speed knob:
      Duration_Stretch > 1.0 => slower
      Duration_Stretch < 1.0 => faster
    We map requested WPM roughly around a baseline.
    """
    wpm = max(80, min(400, int(rate_wpm)))
    stretch = baseline_wpm / float(wpm)
    return max(0.5, min(2.0, stretch))


def _clamp_int(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(val)))


@dataclass
class TTS:
    backend: str
    voice: str
    rate_wpm: int
    volume: float
    sample_rate: int

    def synth_to_wav(self, text: str, out_wav: Path) -> None:
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        msg = clean_for_tts(text)
        tmp_wav = out_wav.with_suffix(".tmp.wav")

        try:
            if self.backend == "piper":
                if not shutil.which("piper"):
                    raise RuntimeError("piper backend selected but piper binary not found")

                cmd = ["piper", "-m", self.voice, "-f", str(tmp_wav), "-r", str(self.sample_rate)]
                subprocess.run(cmd, input=msg.encode("utf-8"), check=True)

            elif self.backend == "festival":
                if not shutil.which("text2wave"):
                    raise RuntimeError("festival backend selected but text2wave not found")

                voice_expr = _festival_voice_expr(self.voice)
                stretch = _duration_stretch_from_wpm(self.rate_wpm)

                with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as tf:
                    tf.write(msg + "\n")
                    text_path = tf.name

                try:
                    cmd = [
                        "text2wave",
                        "-eval",
                        f"(Parameter.set 'Duration_Stretch {stretch})",
                        "-eval",
                        voice_expr,
                        "-o",
                        str(tmp_wav),
                        text_path,
                    ]
                    subprocess.run(cmd, check=True)
                finally:
                    try:
                        Path(text_path).unlink(missing_ok=True)
                    except Exception:
                        pass

            elif self.backend == "dectalk":
                say_bin = Path("/opt/dectalk/runtime/say")
                if not say_bin.exists():
                    raise RuntimeError("dectalk backend selected but /opt/dectalk/runtime/say not found")

                dectalk_env = shutil.which("dectalk-env")
                if not dectalk_env:
                    raise RuntimeError("dectalk backend selected but dectalk-env not found in PATH")

                # voice is speaker 0-9 for your build
                try:
                    speaker = int(str(self.voice).strip())
                except Exception:
                    speaker = 0
                speaker = _clamp_int(speaker, 0, 9)

                # DECtalk rate range 75-600
                rate = _clamp_int(self.rate_wpm, 75, 600)

                # volume float 0..1 -> percent 0..100
                vol = float(self.volume)
                if vol <= 0:
                    vol_pct = 0
                elif vol >= 1.0:
                    vol_pct = 100
                else:
                    vol_pct = _clamp_int(round(vol * 100), 0, 100)

                # Write 16-bit mono 11k PCM to tmp_wav, then we normalize w/ ffmpeg below.
                cmd = [
                    dectalk_env,
                    str(say_bin),
                    "-l",
                    "us",
                    "-s",
                    str(speaker),
                    "-r",
                    str(rate),
                    "-v",
                    str(vol_pct),
                    "-e",
                    "1",
                    "-fo",
                    str(tmp_wav),
                    "-c",
                    "-",  # stdin in clause-mode
                ]
                subprocess.run(cmd, input=(msg + "\n").encode("utf-8"), check=True)

            else:
                # default: espeak-ng
                if not shutil.which("espeak-ng"):
                    raise RuntimeError("espeak-ng not found")

                cmd = ["espeak-ng", "-v", self.voice, "-s", str(int(self.rate_wpm)), "-w", str(tmp_wav), msg]
                subprocess.run(cmd, check=True)

            # Normalize to <sample_rate> stereo 16-bit for clean concatenation
            if not shutil.which("ffmpeg"):
                raise RuntimeError("ffmpeg not found")

            ff_cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(tmp_wav),
                "-ar",
                str(int(self.sample_rate)),
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
            ]

            # Optional gain
            vol = float(self.volume)
            if vol > 0 and abs(vol - 1.0) > 1e-3:
                ff_cmd += ["-filter:a", f"volume={vol}"]

            ff_cmd.append(str(out_wav))
            subprocess.run(ff_cmd, check=True)

        finally:
            try:
                tmp_wav.unlink(missing_ok=True)
            except Exception:
                pass
