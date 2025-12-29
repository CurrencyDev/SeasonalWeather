from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os

import yaml


@dataclass(frozen=True)
class StationConfig:
    name: str
    service_area_name: str
    timezone: str
    disclaimer: str


@dataclass(frozen=True)
class StreamConfig:
    icecast_host: str
    icecast_port: int
    icecast_mount: str


@dataclass(frozen=True)
class CycleConfig:
    normal_interval_seconds: int
    heightened_interval_seconds: int
    min_heightened_seconds: int
    reference_points: List[Tuple[float, float, str]]


@dataclass(frozen=True)
class ObservationsConfig:
    stations: List[str]


@dataclass(frozen=True)
class NWWSConfig:
    server: str
    port: int
    allowed_wfos: List[str]


@dataclass(frozen=True)
class PolicyConfig:
    toneout_product_types: List[str]
    min_tone_gap_seconds: float


@dataclass(frozen=True)
class TTSConfig:
    backend: str
    voice: str
    rate_wpm: int
    volume: float


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    attention_tone_hz: int
    attention_tone_seconds: float
    eom_beep_hz: int
    eom_beep_seconds: float
    inter_segment_silence_seconds: float
    post_alert_silence_seconds: float


@dataclass(frozen=True)
class PathsConfig:
    work_dir: str
    audio_dir: str
    cache_dir: str
    config_dir: str
    log_dir: str


@dataclass(frozen=True)
class ServiceAreaConfig:
    same_fips_all: List[str]
    transmitters: Dict[str, List[Dict[str, str]]]


@dataclass(frozen=True)
class AppConfig:
    station: StationConfig
    stream: StreamConfig
    cycle: CycleConfig
    observations: ObservationsConfig
    nwws: NWWSConfig
    policy: PolicyConfig
    tts: TTSConfig
    audio: AudioConfig
    paths: PathsConfig
    service_area: ServiceAreaConfig


def _env(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def load_config(path: str) -> AppConfig:
    raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    station = StationConfig(**raw["station"])
    stream = StreamConfig(**raw["stream"])
    cycle = CycleConfig(
        normal_interval_seconds=int(raw["cycle"]["normal_interval_seconds"]),
        heightened_interval_seconds=int(raw["cycle"]["heightened_interval_seconds"]),
        min_heightened_seconds=int(raw["cycle"]["min_heightened_seconds"]),
        reference_points=[(float(a), float(b), str(lbl)) for a, b, lbl in raw["cycle"]["reference_points"]],
    )
    observations = ObservationsConfig(stations=list(raw["observations"]["stations"]))
    nwws = NWWSConfig(**raw["nwws"])
    policy = PolicyConfig(
        toneout_product_types=list(raw["policy"]["toneout_product_types"]),
        min_tone_gap_seconds=float(raw["policy"].get("min_tone_gap_seconds", 2.0)),
    )
    tts = TTSConfig(**raw["tts"])
    audio = AudioConfig(**raw["audio"])
    paths = PathsConfig(**raw["paths"])

    transmitters = raw["service_area"]["transmitters"]
    same_all: List[str] = []
    for tx, items in transmitters.items():
        for item in items:
            same_all.append(str(item["same_fips"]).zfill(6))
    same_all = sorted(set(same_all))
    service_area = ServiceAreaConfig(same_fips_all=same_all, transmitters=transmitters)

    return AppConfig(
        station=station,
        stream=stream,
        cycle=cycle,
        observations=observations,
        nwws=nwws,
        policy=policy,
        tts=tts,
        audio=audio,
        paths=paths,
        service_area=service_area,
    )
