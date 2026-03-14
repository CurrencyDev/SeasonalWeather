from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Pattern

# Split text vs tags so we never “rewrite inside markup”.
_TAG_SPLIT_RE = re.compile(r"(<[^>]+>)")

# _env_enabled() removed — vtml_lexicon is now passed as a function argument.

@dataclass(frozen=True)
class Rule:
    rx: Pattern[str]
    repl: Callable[[re.Match[str]], str]

def _sub_alias(alias: str) -> Callable[[re.Match[str]], str]:
    # Keep the original surface form (case/punct) inside the tag.
    def _r(m: re.Match[str]) -> str:
        return f'<vtml_sub alias="{alias}">{m.group(0)}</vtml_sub>'
    return _r

def _phoneme_x_cmu(ph: str) -> Callable[[re.Match[str]], str]:
    def _r(m: re.Match[str]) -> str:
        # x-cmu uses CMU dict phonemes with stress numbers (ASCII-friendly).
        return f'<vtml_phoneme alphabet="x-cmu" ph="{ph}">{m.group(0)}</vtml_phoneme>'
    return _r

# Direction abbreviations -> full words, but ONLY when used as wind direction.
_DIR_MAP = {
    "N": "north", "S": "south", "E": "east", "W": "west",
    "NE": "northeast", "NW": "northwest", "SE": "southeast", "SW": "southwest",
    "NNE": "north-northeast", "ENE": "east-northeast", "ESE": "east-southeast", "SSE": "south-southeast",
    "SSW": "south-southwest", "WSW": "west-southwest", "WNW": "west-northwest", "NNW": "north-northwest",
}

def _wind_dir_repl(m: re.Match[str]) -> str:
    tok = m.group("dir")
    alias = _DIR_MAP.get(tok.upper())
    if not alias:
        return tok
    return f'<vtml_sub alias="{alias}">{tok}</vtml_sub>'

_RULES: list[Rule] = [
    # --- Common wind-direction abbreviations, only in "NW winds ..." contexts ---
    Rule(
        re.compile(
            r"\b(?P<dir>NNE|ENE|ESE|SSE|SSW|WSW|WNW|NNW|NE|NW|SE|SW|N|S|E|W)\b(?=\s+wind(?:s|['’]s)?\b)",
            re.IGNORECASE,
        ),
        _wind_dir_repl,
    ),

    # --- The big one: "winds / wind's" homograph (verb vs weather noun) ---
    # Force weather-noun "winds" and "wind's" => /wɪndz/
    # Avoid common verb cases like "winds up ..." (except "up to ...") and "winds down" / "winds its ...".
    Rule(
        re.compile(
            r"\b(?:winds|wind['’]s)\b"
            r"(?!\s+up\b(?!\s+to\b))"
            r"(?!\s+down\b)"
            r"(?!\s+(?:its|their|his|her|my|your|our)\b)",
            re.IGNORECASE,
        ),
        _phoneme_x_cmu("W IH1 N D Z"),
    ),

   # Bare singular "wind" (noun) => /wɪnd/
   # Mirror the same exclusions as the winds/wind's rule below.
    Rule(
        re.compile(
            r"\bwind\b"
            r"(?!\s+up\b(?!\s+to\b))"
            r"(?!\s+down\b)"
            r"(?!\s+(?:its|their|his|her|my|your|our)\b)",
            re.IGNORECASE,
        ),
        _phoneme_x_cmu("W IH1 N D"),
    ),

    # --- Units (spoken nicely) ---
    Rule(re.compile(r"\bmph\b", re.IGNORECASE), _sub_alias("miles per hour")),
    Rule(re.compile(r"\bkts\b", re.IGNORECASE), _sub_alias("knots")),
    Rule(re.compile(r"\bkt\b",  re.IGNORECASE), _sub_alias("knots")),

    Rule(re.compile(r"\bhpa\b", re.IGNORECASE), _sub_alias("hecto pascals")),
    Rule(re.compile(r"\bmb\b",  re.IGNORECASE), _sub_alias("millibars")),

    # Degree symbol forms
    Rule(re.compile(r"°\s*F\b"), _sub_alias("degrees Fahrenheit")),
    Rule(re.compile(r"°\s*C\b"), _sub_alias("degrees Celsius")),

    # Measurements only when they look like units after a number.
    Rule(re.compile(r"(\d+(?:\.\d+)?)\s*(in\.)\b", re.IGNORECASE), lambda m: f'{m.group(1)} <vtml_sub alias="inches">{m.group(2)}</vtml_sub>'),
    Rule(re.compile(r"(\d+(?:\.\d+)?)\s*(ft\.)\b", re.IGNORECASE), lambda m: f'{m.group(1)} <vtml_sub alias="feet">{m.group(2)}</vtml_sub>'),
    Rule(re.compile(r"(\d+(?:\.\d+)?)\s*(mi\.)\b", re.IGNORECASE), lambda m: f'{m.group(1)} <vtml_sub alias="miles">{m.group(2)}</vtml_sub>'),

    # --- A few NWS-ish abbreviations that show up in headers/closures ---
    # NOAA: VoiceText Paul reads it as "N-O-A-A" letters without this rule.
    Rule(re.compile(r"\bNOAA\b"), _sub_alias("noah")),
    Rule(re.compile(r"\bNWS\b"), _sub_alias("National Weather Service")),
    Rule(re.compile(r"\bTSTM\b", re.IGNORECASE), _sub_alias("thunderstorm")),
    Rule(re.compile(r"\bTSTMS\b", re.IGNORECASE), _sub_alias("thunderstorms")),

    # Time zone abbreviations that may still appear in headers or time announcements
    Rule(re.compile(r"\bEST\b"), _sub_alias("Eastern Standard Time")),
    Rule(re.compile(r"\bEDT\b"), _sub_alias("Eastern Daylight Time")),

    # Aviation categories sometimes appear in AFD/TAF-style text
    Rule(re.compile(r"\bVFR\b"),  _sub_alias("V F R")),
    Rule(re.compile(r"\bMVFR\b"), _sub_alias("M V F R")),
    Rule(re.compile(r"\bIFR\b"),  _sub_alias("I F R")),
    Rule(re.compile(r"\bLIFR\b"), _sub_alias("L I F R")),
]

def apply_voicetext_paul_vtml(text: str, vtml_lexicon: bool = True) -> str:
    """
    Apply small, meteorology-focused VTML tweaks.
    - Does NOT touch existing <...> tags.
    - No newlines are introduced (your wrapper still flattens anyway).
    - vtml_lexicon: set False to skip all substitutions (pass from cfg.tts.voicetext_paul.vtml_lexicon).
    """
    if not text or not vtml_lexicon:
        return text or ""


    parts = _TAG_SPLIT_RE.split(text)
    for i in range(0, len(parts), 2):  # even indexes = plain text between tags
        s = parts[i]
        for r in _RULES:
            s = r.rx.sub(r.repl, s)
        parts[i] = s

    return "".join(parts)
