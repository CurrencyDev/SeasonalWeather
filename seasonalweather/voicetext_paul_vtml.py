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

# State/territory abbreviations -> spoken names, but only in place-name contexts
# like "Baltimore MD", "Smith Point VA", "Washington DC", etc.
_STATE_MAP = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "PR": "Puerto Rico",
    "VI": "U.S. Virgin Islands",
    "GU": "Guam",
    "AS": "American Samoa",
    "MP": "Northern Mariana Islands",
}

_AMBIGUOUS_STATE_CODES = {"IN", "OR", "ME", "HI", "OK"}

# If one of the ambiguous state codes appears after one of these words,
# it's probably normal sentence text, not a place name.
_PLACE_STOPWORDS = {
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BECOME", "BECOMING", "BEEN", "BEING",
    "BY", "CAN", "COULD", "EXPECTED", "FOR", "FROM", "HAS", "HAVE", "IN", "INTO",
    "IS", "IT", "ITS", "LIKELY", "MAY", "MOVING", "NEAR", "OF", "ON", "OR",
    "POSSIBLE", "REMAIN", "REMAINS", "REMAINING", "SHOULD", "THE", "THAT", "THESE",
    "THIS", "THOSE", "TO", "WAS", "WERE", "WILL", "WITH",
}

_PLACE_STATE_RE = re.compile(
    r"(?P<prefix>\b(?:[A-Z][A-Za-z.'’\-]*)(?:[ ,]+[A-Z][A-Za-z.'’\-]*){0,5}[ ,]+)"
    r"(?P<st>"
    + "|".join(sorted(_STATE_MAP, key=len, reverse=True))
    + r")"
    r"(?=(?:\s+(?:to|and)\b)|(?:\s*(?:/|,|\.{1,3}|;|:|\)|$)))"
)

_PLACE_WORD_RE = re.compile(r"[A-Z][A-Za-z.'’\-]*")

def _place_state_repl(m: re.Match[str]) -> str:
    prefix = m.group("prefix")
    st = m.group("st")
    alias = _STATE_MAP.get(st.upper())
    if not alias:
        return m.group(0)

    words = _PLACE_WORD_RE.findall(prefix.rstrip(" ,"))
    if not words:
        return m.group(0)

    last_word = words[-1]
    last_upper = last_word.upper()

    # Peek at following text so we can be stricter for ambiguous codes.
    tail = m.string[m.end():m.end() + 16]

    # Extra guardrail for ambiguous state codes.
    if st.upper() in _AMBIGUOUS_STATE_CODES:
        if last_upper in _PLACE_STOPWORDS:
            return m.group(0)

        # If the next token is a connector like "to" / "and", be stricter.
        if re.match(r"\s+(?:to|and)\b", tail, re.IGNORECASE):
            if len(words) == 1 and last_upper in _PLACE_STOPWORDS:
                return m.group(0)

    return f'{prefix}<vtml_sub alias="{alias}">{st}</vtml_sub>'

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
    Rule(re.compile(r"(\d+(?:\.\d+)?)\s*((?:in)\.?)(?=\s|$|[,:;!?])", re.IGNORECASE), lambda m: f'{m.group(1)} <vtml_sub alias="inches">{m.group(2)}</vtml_sub>'),
    Rule(re.compile(r"(\d+(?:\.\d+)?)\s*((?:ft)\.?)(?=\s|$|[,:;!?])", re.IGNORECASE), lambda m: f'{m.group(1)} <vtml_sub alias="feet">{m.group(2)}</vtml_sub>'),
    Rule(re.compile(r"(\d+(?:\.\d+)?)\s*((?:mi)\.?)(?=\s|$|[,:;!?])", re.IGNORECASE), lambda m: f'{m.group(1)} <vtml_sub alias="miles">{m.group(2)}</vtml_sub>'),
    Rule(re.compile(r"(\d+(?:\.\d+)?)\s*((?:nm)\.?)(?=\s|$|[,:;!?])", re.IGNORECASE), lambda m: f'{m.group(1)} <vtml_sub alias="nautical miles">{m.group(2)}</vtml_sub>'),

    # State abbreviations only when they look like place-name suffixes.
    Rule(_PLACE_STATE_RE, _place_state_repl),

    # --- A few NWS-ish abbreviations that show up in headers/closures ---
    # NOAA: VoiceText Paul reads it as "N-O-A-A" letters without this rule.
    Rule(re.compile(r"\bNOAA\b"), _sub_alias("noah")),
    Rule(re.compile(r"\bNWS\b"), _sub_alias("National Weather Service")),
    Rule(re.compile(r"\bthunderstorms\b", re.IGNORECASE), _phoneme_x_cmu("TH AH1 N D ER0 S T AO2 R M Z")),
    Rule(re.compile(r"\bthunderstorm\b", re.IGNORECASE), _phoneme_x_cmu("TH AH1 N D ER0 S T AO2 R M")),
    Rule(re.compile(r"\bTSTMS\b", re.IGNORECASE), _phoneme_x_cmu("TH AH1 N D ER0 S T AO2 R M Z")),
    Rule(re.compile(r"\bTSTM\b", re.IGNORECASE), _phoneme_x_cmu("TH AH1 N D ER0 S T AO2 R M")),

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
