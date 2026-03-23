"""
SeasonalWeather configuration loader.

config.yaml is the single source of truth for all behaviour.

The only things that belong in the environment (seasonalweather.env) are:
  - NWWS_JID / NWWS_PASSWORD          — XMPP credentials
  - ICECAST_SOURCE_PASSWORD            — Icecast source secret
  - ICECAST_ADMIN_PASSWORD             — Icecast admin secret   (optional)
  - ICECAST_RELAY_PASSWORD             — Icecast relay secret   (optional)
  - SEASONAL_API_TOKEN                 — API bearer token       (optional)
  - SEASONAL_API_TOKENS_JSON           — multi-token JSON blob  (optional)
  - LIQUIDSOAP_TELNET_HOST             — deployment topology    (optional, default 127.0.0.1)
  - LIQUIDSOAP_TELNET_PORT             — deployment topology    (optional, default 1234)

Everything else lives in config.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Helpers — only used here for the surviving env-sourced secrets
# ---------------------------------------------------------------------------

def _env(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def _env_required(key: str) -> str:
    v = os.environ.get(key, "").strip()
    if not v:
        raise RuntimeError(
            f"Required environment variable {key!r} is not set. "
            "Check /etc/seasonalweather/seasonalweather.env."
        )
    return v


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key, "").strip()
    return v if v else default


# ---------------------------------------------------------------------------
# Dataclasses — one per logical subsystem
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StationConfig:
    name: str
    service_area_name: str
    timezone: str
    disclaimer: str
    # Governs which broadcast segments are generated.
    # land         - ZFP land zones only, no marine segment
    # coastal      - CWF marine segment only (pure marine station)
    # land_coastal - ZFP + CWF (default for Canonical SeasonalWeather)
    # land_marine  - ZFP + CWF for inland/bay waters (non-ocean)
    # marine       - CWF / marine products only
    deployment_type: str = "land"  # land|coastal|land_coastal|land_marine|marine


@dataclass(frozen=True)
class StreamConfig:
    icecast_host: str
    icecast_port: int
    icecast_mount: str


# --- cycle sub-sections ---

@dataclass(frozen=True)
class CycleSpcConfig:
    enabled: bool
    wfos: List[str]
    days: int
    min_dn: int
    timeout_s: float


@dataclass(frozen=True)
class CycleFcConfig:
    use_short: bool
    periods_normal: int
    periods_per_group: int
    max_points_normal: int
    max_points_7day: int
    point_max_chars: int
    line_max_chars: int
    rotate_period_s: int
    rotate_step: int
    # Ordered list of (zone_id, display_label) pairs.  When non-empty
    # the ZFP /zones/forecast/{zoneId}/forecast path is used instead of
    # gridpoint lat/lon fetches.  Falls back to gridpoint if empty.
    forecast_zones: List[Tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class CycleObsConfig:
    max_normal: int
    rotate_period_s: int
    rotate_step: int
    aliases: Dict[str, str]


@dataclass(frozen=True)
class CycleProductConfig:
    """Generic per-product character-limit config (AFD, SYN, etc.)."""
    max_chars_normal: int
    max_chars_heightened: int


@dataclass(frozen=True)
class CycleHwoConfig:
    max_chars_normal: int
    max_chars_heightened: int
    speak_unavailable: bool


@dataclass(frozen=True)
class CycleCwfConfig:
    # Coastal Waters Forecast segment configuration.
    enabled: bool
    offices: List[str]          # WFO offices to pull CWF from, e.g. ["LWX"]
    max_chars_normal: int       # 0 = unlimited
    max_chars_heightened: int


@dataclass(frozen=True)
class CycleRwrConfig:
    # Regional Weather Roundup (RWR) observations segment configuration.
    enabled: bool
    office: str                 # WFO to pull RWR from (e.g. 'LWX')
    staleness_minutes: int      # fall back to ASOS if RWR older than this
    # Station names matching the raw RWR product (upper-case) that get
    # full-detail treatment (wind, pressure, dew point, humidity).
    # Empty list = auto-derive from first station in first section.
    anchor_stations: List[str]
    # ASOS station IDs to use when RWR is stale/unavailable.
    # Empty = use observations.stations from the main config.
    fallback_stations: List[str]
    pressure_trend_threshold_inhg: float
    pressure_cache_hours: float
    max_compact_per_section: int
    # Optional spoken-name overrides for RWR station abbreviations.
    # {RAW_UPPER: spoken_name}  e.g. {'DULLES': 'Dulles Airport'}
    station_names: Dict[str, str]


@dataclass(frozen=True)
class CycleConfig:
    normal_interval_seconds: int
    heightened_interval_seconds: int
    min_heightened_seconds: int
    lead_time_seconds: int
    reference_points: List[Tuple[float, float, str]]
    last_product_max_chars: int
    spc: CycleSpcConfig
    fc: CycleFcConfig
    obs: CycleObsConfig
    hwo: CycleHwoConfig
    afd: CycleProductConfig
    syn: CycleProductConfig
    cwf: CycleCwfConfig
    rwr: CycleRwrConfig


@dataclass(frozen=True)
class ObservationsConfig:
    stations: List[str]


# --- nwws ---

@dataclass(frozen=True)
class NWWSResiliencyConfig:
    stall_seconds: int
    muc_confirm_seconds: int
    start_wait_seconds: int
    join_wait_seconds: int
    backoff_max_seconds: int
    rx_log_first_n: int
    decision_log_first_n: int
    decision_log_every: int


@dataclass(frozen=True)
class NWWSConfig:
    server: str
    port: int
    room: str
    nick: str
    allowed_wfos: List[str]
    resiliency: NWWSResiliencyConfig


# --- nws (api.weather.gov calls) ---

@dataclass(frozen=True)
class NwsConfig:
    user_agent: str


# --- policy ---

@dataclass(frozen=True)
class PolicyConfig:
    toneout_product_types: List[str]
    min_tone_gap_seconds: float


# --- same ---

@dataclass(frozen=True)
class SameConfig:
    enabled: bool
    sender: str
    duration_minutes: int
    amplitude: float


# --- cap ---

@dataclass(frozen=True)
class CapFullConfig:
    enabled: bool
    severities: List[str]
    events: List[str]
    cooldown_seconds: int


@dataclass(frozen=True)
class CapVoiceConfig:
    enabled: bool
    events: List[str]
    cooldown_seconds: int


@dataclass(frozen=True)
class CapConfig:
    enabled: bool
    dryrun: bool
    poll_seconds: int
    user_agent: str
    url: str
    ledger_path: str
    ledger_max_age_days: int
    full: CapFullConfig
    voice: CapVoiceConfig


# --- ern ---

@dataclass(frozen=True)
class ErnRelayConfig:
    enabled: bool
    events: List[str]
    min_confidence: float
    cooldown_seconds: int
    senders: List[str]


@dataclass(frozen=True)
class ErnConfig:
    enabled: bool
    dryrun: bool
    url: str
    name: str
    sample_rate: int
    tail_seconds: float
    trigger_ratio: float
    dedupe_seconds: float
    confidence_min: float
    relay: ErnRelayConfig


# --- samedec subprocess ---

@dataclass(frozen=True)
class SameDecConfig:
    bin: str
    confidence: float
    start_delay_s: float


# --- tests (RWT/RMT scheduling) ---

@dataclass(frozen=True)
class TestsScheduleConfig:
    weekday: int
    hour: int
    minute: int


@dataclass(frozen=True)
class TestsRmtConfig:
    nth: int
    weekday: int
    hour: int
    minute: int


@dataclass(frozen=True)
class TestsConfig:
    enabled: bool
    postpone_minutes: int
    max_postpone_hours: int
    jitter_seconds: int
    toneout_cooldown_seconds: int
    cap_block_seconds: int
    ern_block_seconds: int
    rwt: TestsScheduleConfig
    rmt: TestsRmtConfig


# --- zonecounty ---

@dataclass(frozen=True)
class ZoneCountyConfig:
    enabled: bool
    dbx_url: str
    cache_days: int
    index_url: str
    base_url: str


# --- mareas ---

@dataclass(frozen=True)
class MareasConfig:
    enabled: bool
    url: str
    cache_days: int


# --- station_feed ---

@dataclass(frozen=True)
class StationFeedHousekeepingConfig:
    enabled: bool
    interval_sec: int
    grace_sec: int
    keep_unparseable: bool
    housekeep_seconds: int


@dataclass(frozen=True)
class StationFeedNwwsConfig:
    vtec_event_labels: dict[str, str]
    tz_abbrev_overrides: dict[str, str]


@dataclass(frozen=True)
class StationFeedConfig:
    enabled: bool
    path: str
    station_id: str
    source: str
    max_items: int
    ttl_seconds: int
    min_write_seconds: float
    fetch_nws: bool
    debug: bool
    ern_area_names: bool
    housekeeping: StationFeedHousekeepingConfig
    nwws: StationFeedNwwsConfig


# --- rebroadcast ---

@dataclass(frozen=True)
class RebroadcastConfig:
    enabled: bool
    interval_seconds: int
    min_gap_seconds: int
    ttl_seconds: int
    max_items: int
    include_voice: bool


# --- api ---

@dataclass(frozen=True)
class ApiConfig:
    allow_remote: bool
    audio_max_bytes: int
    audio_max_seconds: int
    audio_ttl_seconds: int
    ffmpeg_bin: str
    full_eas_heightened: bool
    scopes: str
    subject: str
    manual_full_eas_heightens: bool


# --- live_time ---

@dataclass(frozen=True)
class LiveTimeConfig:
    enabled: bool
    interval_seconds: int


# --- dedupe ---

@dataclass(frozen=True)
class DedupeConfig:
    ttl_seconds: int


# --- tts ---



@dataclass(frozen=True)
class LogsDiscordConfig:
    """Discord webhook logging knobs. URLs come from .env, not config.yaml."""
    enabled: bool = False

    # Per-channel on/off toggles.  A channel only fires if both `enabled` AND
    # the corresponding env var is set.
    alerts_enabled: bool = True
    ops_enabled: bool = True
    api_enabled: bool = True
    errors_enabled: bool = True

    # Webhook URLs (populated from .env by load_config; kept as empty strings
    # here so the dataclass can be constructed without env vars in tests).
    alerts_url: str = ""
    ops_url: str = ""
    api_url: str = ""
    errors_url: str = ""

    # Rate limiting — max embeds per webhook per minute.
    # Discord allows ~30/min per webhook; stay conservative.
    rate_limit_per_minute: int = 20

    # Content knobs
    post_tests: bool = True           # Post RWT/RMT test originations to alerts channel
    post_voice_only: bool = True      # Post voice-only cut-ins (SPS, CAP voice, etc.)
    cycle_rebuild_log: bool = True    # Post cycle rebuild events to ops channel
    # AlertTracker lifecycle (load/purge on startup) — slightly noisy, off by default
    alerttracker_lifecycle_log: bool = False

    # Base URL for the Lucide icon CDN, e.g. "https://cdn.seasonalnet.org"
    # Leave empty to omit thumbnails from all embeds.
    icon_cdn_url: str = ""


@dataclass(frozen=True)
class LogsConfig:
    discord: LogsDiscordConfig = field(default_factory=LogsDiscordConfig)

@dataclass(frozen=True)
class VoiceTextPaulConfig:
    run_as: str
    retries: int
    retry_sleep_ms: int
    reset_every: int
    kill_before: bool
    vtml_lexicon: bool
    alias_overrides: List[Dict[str, Any]] = field(default_factory=list)
    phoneme_overrides_x_cmu: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TTSConfig:
    backend: str
    voice: str
    rate_wpm: int
    volume: float
    voicetext_paul: VoiceTextPaulConfig
    text_overrides: List[Dict[str, Any]] = field(default_factory=list)


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


# ---------------------------------------------------------------------------
# Secrets — sourced from environment, not yaml
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SecretsConfig:
    """
    The nine values that legitimately live in the environment file.
    Loaded once at startup by load_config() and kept here so no other
    module ever needs to call os.getenv() for credentials.
    """
    nwws_jid: str
    nwws_password: str
    icecast_source_password: str
    icecast_admin_password: str    # may be empty
    icecast_relay_password: str    # may be empty
    api_token: str                 # may be empty
    api_tokens_json: str           # may be empty — JSON blob for multi-token
    liquidsoap_host: str
    liquidsoap_port: int


# ---------------------------------------------------------------------------
# Top-level AppConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    # core
    station: StationConfig
    stream: StreamConfig
    cycle: CycleConfig
    observations: ObservationsConfig
    nwws: NWWSConfig
    nws: NwsConfig
    policy: PolicyConfig
    tts: TTSConfig
    audio: AudioConfig
    paths: PathsConfig
    service_area: ServiceAreaConfig

    # subsystems
    same: SameConfig
    cap: CapConfig
    ern: ErnConfig
    samedec: SameDecConfig
    tests: TestsConfig
    zonecounty: ZoneCountyConfig
    mareas: MareasConfig
    station_feed: StationFeedConfig
    rebroadcast: RebroadcastConfig
    api: ApiConfig
    live_time: LiveTimeConfig
    dedupe: DedupeConfig
    logs: LogsConfig

    # secrets (from environment)
    secrets: SecretsConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safe nested get with a fallback default."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, None)
        if cur is None:
            return default
    return cur


def load_config(path: str) -> AppConfig:
    raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # station
    # ------------------------------------------------------------------
    st = raw["station"]
    station = StationConfig(
        name=str(st["name"]),
        service_area_name=str(st["service_area_name"]),
        timezone=str(st["timezone"]),
        disclaimer=str(st["disclaimer"]),
        deployment_type=str(st.get("deployment_type", "land")),
    )

    # ------------------------------------------------------------------
    # stream
    # ------------------------------------------------------------------
    stream = StreamConfig(**raw["stream"])

    # ------------------------------------------------------------------
    # cycle
    # ------------------------------------------------------------------
    cy = raw["cycle"]
    spc_raw = cy.get("spc", {})
    fc_raw = cy.get("fc", {})
    obs_raw = cy.get("obs", {})
    hwo_raw = cy.get("hwo", {})
    afd_raw = cy.get("afd", {})
    syn_raw = cy.get("syn", {})
    cwf_raw = cy.get("cwf", {})
    rwr_raw = cy.get("rwr", {})

    cycle = CycleConfig(
        normal_interval_seconds=int(cy["normal_interval_seconds"]),
        heightened_interval_seconds=int(cy["heightened_interval_seconds"]),
        min_heightened_seconds=int(cy["min_heightened_seconds"]),
        lead_time_seconds=int(cy.get("lead_time_seconds", 90)),
        reference_points=[
            (float(a), float(b), str(lbl))
            for a, b, lbl in cy["reference_points"]
        ],
        last_product_max_chars=int(cy.get("last_product_max_chars", 260)),
        spc=CycleSpcConfig(
            enabled=bool(spc_raw.get("enabled", False)),
            wfos=[str(w).upper() for w in spc_raw.get("wfos", ["LWX"])],
            days=int(spc_raw.get("days", 3)),
            min_dn=int(spc_raw.get("min_dn", 3)),
            timeout_s=float(spc_raw.get("timeout_s", 6.0)),
        ),
        fc=CycleFcConfig(
            use_short=bool(fc_raw.get("use_short", True)),
            periods_normal=int(fc_raw.get("periods_normal", 14)),
            periods_per_group=int(fc_raw.get("periods_per_group", 4)),
            max_points_normal=int(fc_raw.get("max_points_normal", 6)),
            max_points_7day=int(fc_raw.get("max_points_7day", 2)),
            point_max_chars=int(fc_raw.get("point_max_chars", 1600)),
            line_max_chars=int(fc_raw.get("line_max_chars", 1600)),
            rotate_period_s=int(fc_raw.get("rotate_period_s", 300)),
            rotate_step=int(fc_raw.get("rotate_step", 0)),
            forecast_zones=[
                (str(z["id"]).upper().strip(), str(z["label"]).strip())
                for z in (fc_raw.get("forecast_zones") or [])
                if isinstance(z, dict) and z.get("id") and z.get("label")
            ],
        ),
        obs=CycleObsConfig(
            max_normal=int(obs_raw.get("max_normal", 0)),
            rotate_period_s=int(obs_raw.get("rotate_period_s", 300)),
            rotate_step=int(obs_raw.get("rotate_step", 0)),
            aliases=dict(obs_raw.get("aliases", {})),
        ),
        hwo=CycleHwoConfig(
            max_chars_normal=int(hwo_raw.get("max_chars_normal", 0)),
            max_chars_heightened=int(hwo_raw.get("max_chars_heightened", 1200)),
            speak_unavailable=bool(hwo_raw.get("speak_unavailable", True)),
        ),
        afd=CycleProductConfig(
            max_chars_normal=int(afd_raw.get("max_chars_normal", 0)),
            max_chars_heightened=int(afd_raw.get("max_chars_heightened", 1000)),
        ),
        syn=CycleProductConfig(
            max_chars_normal=int(syn_raw.get("max_chars_normal", 1500)),
            max_chars_heightened=int(syn_raw.get("max_chars_heightened", 900)),
        ),
        cwf=CycleCwfConfig(
            enabled=bool(cwf_raw.get("enabled", False)),
            offices=[str(o).upper().strip() for o in (cwf_raw.get("offices") or [])],
            max_chars_normal=int(cwf_raw.get("max_chars_normal", 2000)),
            max_chars_heightened=int(cwf_raw.get("max_chars_heightened", 1200)),
        ),
        rwr=CycleRwrConfig(
            enabled=bool(rwr_raw.get("enabled", False)),
            office=str(rwr_raw.get("office", "LWX")).upper().strip(),
            staleness_minutes=int(rwr_raw.get("staleness_minutes", 75)),
            anchor_stations=[
                str(s).upper().strip()
                for s in (rwr_raw.get("anchor_stations") or [])
                if str(s).strip()
            ],
            fallback_stations=[
                str(s).upper().strip()
                for s in (rwr_raw.get("fallback_stations") or [])
                if str(s).strip()
            ],
            pressure_trend_threshold_inhg=float(
                rwr_raw.get("pressure_trend_threshold_inhg", 0.02)
            ),
            pressure_cache_hours=float(rwr_raw.get("pressure_cache_hours", 3.0)),
            max_compact_per_section=int(rwr_raw.get("max_compact_per_section", 8)),
            station_names={
                str(k).upper().strip(): str(v).strip()
                for k, v in (rwr_raw.get("station_names") or {}).items()
                if str(k).strip() and str(v).strip()
            },
        ),
    )

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------
    observations = ObservationsConfig(stations=list(raw["observations"]["stations"]))

    # ------------------------------------------------------------------
    # nwws
    # ------------------------------------------------------------------
    nw = raw["nwws"]
    res_raw = nw.get("resiliency", {})
    nwws = NWWSConfig(
        server=str(nw.get("server", "nwws-oi.weather.gov")),
        port=int(nw.get("port", 5222)),
        room=str(nw.get("room", "NWWS@conference.nwws-oi.weather.gov")),
        nick=str(nw.get("nick", "SeasonalWeather")),
        allowed_wfos=list(nw.get("allowed_wfos", [])),
        resiliency=NWWSResiliencyConfig(
            stall_seconds=int(res_raw.get("stall_seconds", 60)),
            muc_confirm_seconds=int(res_raw.get("muc_confirm_seconds", 30)),
            start_wait_seconds=int(res_raw.get("start_wait_seconds", 25)),
            join_wait_seconds=int(res_raw.get("join_wait_seconds", 35)),
            backoff_max_seconds=int(res_raw.get("backoff_max_seconds", 90)),
            rx_log_first_n=int(res_raw.get("rx_log_first_n", 20)),
            decision_log_first_n=int(res_raw.get("decision_log_first_n", 20)),
            decision_log_every=int(res_raw.get("decision_log_every", 0)),
        ),
    )

    # ------------------------------------------------------------------
    # nws
    # ------------------------------------------------------------------
    nws_raw = raw.get("nws", {})
    nws = NwsConfig(user_agent=str(nws_raw.get("user_agent", "")))

    # ------------------------------------------------------------------
    # policy
    # ------------------------------------------------------------------
    pol = raw["policy"]
    policy = PolicyConfig(
        toneout_product_types=list(pol["toneout_product_types"]),
        min_tone_gap_seconds=float(pol.get("min_tone_gap_seconds", 2.0)),
    )

    # ------------------------------------------------------------------
    # same
    # ------------------------------------------------------------------
    sa = raw.get("same", {})
    same = SameConfig(
        enabled=bool(sa.get("enabled", False)),
        sender=str(sa.get("sender", "SEASNWXR")),
        duration_minutes=int(sa.get("duration_minutes", 60)),
        amplitude=float(sa.get("amplitude", 0.35)),
    )

    # ------------------------------------------------------------------
    # cap
    # ------------------------------------------------------------------
    cap_raw = raw.get("cap", {})
    cap_full_raw = cap_raw.get("full", {})
    cap_voice_raw = cap_raw.get("voice", {})
    cap = CapConfig(
        enabled=bool(cap_raw.get("enabled", False)),
        dryrun=bool(cap_raw.get("dryrun", True)),
        poll_seconds=int(cap_raw.get("poll_seconds", 60)),
        user_agent=str(cap_raw.get("user_agent", "SeasonalWeather (CAP monitor)")),
        url=str(cap_raw.get("url", "")),
        ledger_path=str(cap_raw.get("ledger_path", "/var/lib/seasonalweather/cap_ledger.json")),
        ledger_max_age_days=int(cap_raw.get("ledger_max_age_days", 14)),
        full=CapFullConfig(
            enabled=bool(cap_full_raw.get("enabled", True)),
            severities=[str(s) for s in cap_full_raw.get("severities", ["Severe", "Extreme"])],
            events=[str(e) for e in cap_full_raw.get("events", [])],
            cooldown_seconds=int(cap_full_raw.get("cooldown_seconds", 180)),
        ),
        voice=CapVoiceConfig(
            enabled=bool(cap_voice_raw.get("enabled", False)),
            events=[str(e) for e in cap_voice_raw.get("events", ["Special Weather Statement"])],
            cooldown_seconds=int(cap_voice_raw.get("cooldown_seconds", 600)),
        ),
    )

    # ------------------------------------------------------------------
    # ern
    # ------------------------------------------------------------------
    ern_raw = raw.get("ern", {})
    ern_relay_raw = ern_raw.get("relay", {})
    ern = ErnConfig(
        enabled=bool(ern_raw.get("enabled", False)),
        dryrun=bool(ern_raw.get("dryrun", True)),
        url=str(ern_raw.get("url", "")),
        name=str(ern_raw.get("name", "ERN/JON")),
        sample_rate=int(ern_raw.get("sample_rate", 48000)),
        tail_seconds=float(ern_raw.get("tail_seconds", 10.0)),
        trigger_ratio=float(ern_raw.get("trigger_ratio", 8.0)),
        dedupe_seconds=float(ern_raw.get("dedupe_seconds", 20.0)),
        confidence_min=float(ern_raw.get("confidence_min", 0.25)),
        relay=ErnRelayConfig(
            enabled=bool(ern_relay_raw.get("enabled", False)),
            events=[str(e) for e in ern_relay_raw.get("events", ["RWT", "RMT"])],
            min_confidence=float(ern_relay_raw.get("min_confidence", 0.80)),
            cooldown_seconds=int(ern_relay_raw.get("cooldown_seconds", 300)),
            senders=[str(s) for s in ern_relay_raw.get("senders", [])],
        ),
    )

    # ------------------------------------------------------------------
    # samedec
    # ------------------------------------------------------------------
    sd_raw = raw.get("samedec", {})
    samedec = SameDecConfig(
        bin=str(sd_raw.get("bin", "/usr/local/bin/samedec")),
        confidence=float(sd_raw.get("confidence", 0.85)),
        start_delay_s=float(sd_raw.get("start_delay_s", 1.4)),
    )

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------
    tst_raw = raw.get("tests", {})
    rwt_raw = tst_raw.get("rwt", {})
    rmt_raw = tst_raw.get("rmt", {})
    tests = TestsConfig(
        enabled=bool(tst_raw.get("enabled", False)),
        postpone_minutes=int(tst_raw.get("postpone_minutes", 15)),
        max_postpone_hours=int(tst_raw.get("max_postpone_hours", 6)),
        jitter_seconds=int(tst_raw.get("jitter_seconds", 60)),
        toneout_cooldown_seconds=int(
            tst_raw.get("toneout_cooldown_seconds", cycle.min_heightened_seconds)
        ),
        cap_block_seconds=int(tst_raw.get("cap_block_seconds", 3600)),
        ern_block_seconds=int(tst_raw.get("ern_block_seconds", 3600)),
        rwt=TestsScheduleConfig(
            weekday=int(rwt_raw.get("weekday", 2)),
            hour=int(rwt_raw.get("hour", 11)),
            minute=int(rwt_raw.get("minute", 0)),
        ),
        rmt=TestsRmtConfig(
            nth=int(rmt_raw.get("nth", 1)),
            weekday=int(rmt_raw.get("weekday", 2)),
            hour=int(rmt_raw.get("hour", 11)),
            minute=int(rmt_raw.get("minute", 0)),
        ),
    )

    # ------------------------------------------------------------------
    # zonecounty
    # ------------------------------------------------------------------
    zc_raw = raw.get("zonecounty", {})
    zonecounty = ZoneCountyConfig(
        enabled=bool(zc_raw.get("enabled", True)),
        dbx_url=str(zc_raw.get("dbx_url", "")),
        cache_days=int(zc_raw.get("cache_days", 30)),
        index_url=str(zc_raw.get("index_url", "https://www.weather.gov/gis/ZoneCounty")),
        base_url=str(zc_raw.get("base_url", "https://www.weather.gov/source/gis/Shapefiles/County/")),
    )

    # ------------------------------------------------------------------
    # mareas
    # ------------------------------------------------------------------
    mr_raw = raw.get("mareas", {})
    mareas = MareasConfig(
        enabled=bool(mr_raw.get("enabled", True)),
        url=str(mr_raw.get("url", "")),
        cache_days=int(mr_raw.get("cache_days", 30)),
    )

    # ------------------------------------------------------------------
    # station_feed
    # ------------------------------------------------------------------
    sf_raw = raw.get("station_feed", {})
    sf_hk_raw = sf_raw.get("housekeeping", {})
    sf_nwws_raw = sf_raw.get("nwws", {}) if isinstance(sf_raw.get("nwws", {}), dict) else {}

    sf_nwws_labels_raw = sf_nwws_raw.get("vtec_event_labels", {})
    if not isinstance(sf_nwws_labels_raw, dict):
        sf_nwws_labels_raw = {}
    sf_nwws_labels = {
        str(k).strip().upper(): str(v).strip()
        for k, v in sf_nwws_labels_raw.items()
        if str(k).strip() and str(v).strip()
    }

    sf_nwws_tz_raw = sf_nwws_raw.get("tz_abbrev_overrides", {})
    if not isinstance(sf_nwws_tz_raw, dict):
        sf_nwws_tz_raw = {}
    sf_nwws_tz = {
        str(k).strip().upper(): str(v).strip()
        for k, v in sf_nwws_tz_raw.items()
        if str(k).strip() and str(v).strip()
    }

    station_feed = StationFeedConfig(
        enabled=bool(sf_raw.get("enabled", False)),
        path=str(sf_raw.get("path", "/srv/seasonalweather/api/station/handled-alerts.json")),
        station_id=str(sf_raw.get("station_id", "seasonalweather")),
        source=str(sf_raw.get("source", "seasonalweather")),
        max_items=int(sf_raw.get("max_items", 24)),
        ttl_seconds=int(sf_raw.get("ttl_seconds", 7200)),
        min_write_seconds=float(sf_raw.get("min_write_seconds", 0.5)),
        fetch_nws=bool(sf_raw.get("fetch_nws", False)),
        debug=bool(sf_raw.get("debug", False)),
        ern_area_names=bool(sf_raw.get("ern_area_names", True)),
        housekeeping=StationFeedHousekeepingConfig(
            enabled=bool(sf_hk_raw.get("enabled", True)),
            interval_sec=int(sf_hk_raw.get("interval_sec", 60)),
            grace_sec=int(sf_hk_raw.get("grace_sec", 5)),
            keep_unparseable=bool(sf_hk_raw.get("keep_unparseable", True)),
            housekeep_seconds=int(sf_hk_raw.get("housekeep_seconds", 30)),
        ),
        nwws=StationFeedNwwsConfig(
            vtec_event_labels=sf_nwws_labels,
            tz_abbrev_overrides=sf_nwws_tz,
        ),
    )

    # ------------------------------------------------------------------
    # rebroadcast
    # ------------------------------------------------------------------
    rb_raw = raw.get("rebroadcast", {})
    rebroadcast = RebroadcastConfig(
        enabled=bool(rb_raw.get("enabled", False)),
        interval_seconds=int(rb_raw.get("interval_seconds", 300)),
        min_gap_seconds=int(rb_raw.get("min_gap_seconds", 300)),
        ttl_seconds=int(rb_raw.get("ttl_seconds", 3600)),
        max_items=int(rb_raw.get("max_items", 6)),
        include_voice=bool(rb_raw.get("include_voice", False)),
    )

    # ------------------------------------------------------------------
    # api
    # ------------------------------------------------------------------
    api_raw = raw.get("api", {})
    api = ApiConfig(
        allow_remote=bool(api_raw.get("allow_remote", False)),
        audio_max_bytes=int(api_raw.get("audio_max_bytes", 20971520)),
        audio_max_seconds=int(api_raw.get("audio_max_seconds", 180)),
        audio_ttl_seconds=int(api_raw.get("audio_ttl_seconds", 86400)),
        ffmpeg_bin=str(api_raw.get("ffmpeg_bin", "ffmpeg")),
        full_eas_heightened=bool(api_raw.get("full_eas_heightened", False)),
        scopes=str(api_raw.get("scopes", "")),
        subject=str(api_raw.get("subject", "local-admin")),
        manual_full_eas_heightens=bool(api_raw.get("manual_full_eas_heightens", True)),
    )

    # ------------------------------------------------------------------
    # live_time
    # ------------------------------------------------------------------
    lt_raw = raw.get("live_time", {})
    live_time = LiveTimeConfig(
        enabled=bool(lt_raw.get("enabled", True)),
        interval_seconds=int(lt_raw.get("interval_seconds", 45)),
    )

    # ------------------------------------------------------------------
    # dedupe
    # ------------------------------------------------------------------
    dd_raw = raw.get("dedupe", {})
    dedupe = DedupeConfig(ttl_seconds=int(dd_raw.get("ttl_seconds", 900)))

    # ------------------------------------------------------------------
    # tts
    # ------------------------------------------------------------------
    tts_raw = raw["tts"]
    vtp_raw = tts_raw.get("voicetext_paul", {})
    tts = TTSConfig(
        backend=str(tts_raw["backend"]),
        voice=str(tts_raw.get("voice", "9")),
        rate_wpm=int(tts_raw.get("rate_wpm", 165)),
        volume=float(tts_raw.get("volume", 1.0)),
        text_overrides=list(tts_raw.get("text_overrides", []) or []),
        voicetext_paul=VoiceTextPaulConfig(
            run_as=str(vtp_raw.get("run_as", "voicetext")),
            retries=int(vtp_raw.get("retries", 1)),
            retry_sleep_ms=int(vtp_raw.get("retry_sleep_ms", 150)),
            reset_every=int(vtp_raw.get("reset_every", 0)),
            kill_before=bool(vtp_raw.get("kill_before", False)),
            vtml_lexicon=bool(vtp_raw.get("vtml_lexicon", True)),
            alias_overrides=list(vtp_raw.get("alias_overrides", []) or []),
            phoneme_overrides_x_cmu=list(vtp_raw.get("phoneme_overrides_x_cmu", []) or []),
        ),
    )

    # ------------------------------------------------------------------
    # audio
    # ------------------------------------------------------------------
    audio = AudioConfig(**raw["audio"])

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    paths = PathsConfig(**raw["paths"])

    # ------------------------------------------------------------------
    # service_area
    # ------------------------------------------------------------------
    transmitters = raw["service_area"]["transmitters"]
    same_all: List[str] = []
    for _tx, items in transmitters.items():
        for item in items:
            same_all.append(str(item["same_fips"]).zfill(6))
    same_all = sorted(set(same_all))
    service_area = ServiceAreaConfig(same_fips_all=same_all, transmitters=transmitters)

    # ------------------------------------------------------------------
    # secrets — read from environment, only place in the codebase that
    # touches os.environ for these keys
    # ------------------------------------------------------------------
    secrets = SecretsConfig(
        nwws_jid=_env_required("NWWS_JID"),
        nwws_password=_env_required("NWWS_PASSWORD"),
        icecast_source_password=_env_required("ICECAST_SOURCE_PASSWORD"),
        icecast_admin_password=_env_str("ICECAST_ADMIN_PASSWORD", ""),
        icecast_relay_password=_env_str("ICECAST_RELAY_PASSWORD", ""),
        api_token=_env_str("SEASONAL_API_TOKEN", ""),
        api_tokens_json=_env_str("SEASONAL_API_TOKENS_JSON", ""),
        liquidsoap_host=_env_str("LIQUIDSOAP_TELNET_HOST", "127.0.0.1"),
        liquidsoap_port=_env_int("LIQUIDSOAP_TELNET_PORT", 1234),
    )


    # ------------------------------------------------------------------
    # logs — Discord webhook URLs (from .env, not config.yaml)
    # ------------------------------------------------------------------
    _ld = _get(raw, "logs", "discord") or {}
    _logs_discord = LogsDiscordConfig(
        enabled=bool(_get(_ld, "enabled", default=False)),
        alerts_enabled=bool(_get(_ld, "alerts_enabled", default=True)),
        ops_enabled=bool(_get(_ld, "ops_enabled", default=True)),
        api_enabled=bool(_get(_ld, "api_enabled", default=True)),
        errors_enabled=bool(_get(_ld, "errors_enabled", default=True)),
        # URLs come exclusively from env; not from config.yaml (no auth on webhooks)
        alerts_url=_env_str("SEASONAL_DISCORD_ALERTS_WEBHOOK", ""),
        ops_url=_env_str("SEASONAL_DISCORD_OPS_WEBHOOK", ""),
        api_url=_env_str("SEASONAL_DISCORD_API_WEBHOOK", ""),
        errors_url=_env_str("SEASONAL_DISCORD_ERRORS_WEBHOOK", ""),
        rate_limit_per_minute=int(_get(_ld, "rate_limit_per_minute", default=20)),
        post_tests=bool(_get(_ld, "post_tests", default=True)),
        post_voice_only=bool(_get(_ld, "post_voice_only", default=True)),
        cycle_rebuild_log=bool(_get(_ld, "cycle_rebuild_log", default=True)),
        alerttracker_lifecycle_log=bool(
            _get(_ld, "alerttracker_lifecycle_log", default=False)
        ),
        icon_cdn_url=str(_get(_ld, "icon_cdn_url", default="") or "").strip(),
    )
    logs = LogsConfig(discord=_logs_discord)

    return AppConfig(
        station=station,
        stream=stream,
        cycle=cycle,
        observations=observations,
        nwws=nwws,
        nws=nws,
        policy=policy,
        tts=tts,
        audio=audio,
        paths=paths,
        service_area=service_area,
        same=same,
        cap=cap,
        ern=ern,
        samedec=samedec,
        tests=tests,
        zonecounty=zonecounty,
        mareas=mareas,
        station_feed=station_feed,
        rebroadcast=rebroadcast,
        api=api,
        live_time=live_time,
        dedupe=dedupe,
        secrets=secrets,
        logs=logs,
    )
