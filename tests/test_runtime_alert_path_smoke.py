import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from seasonalweather.alerts.cap_nws import CapAlertEvent
from seasonalweather.alerts.product import ParsedProduct
from seasonalweather.config import load_config
from seasonalweather.main import Orchestrator


class _FakeAudioOriginator:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.full_calls = []
        self.voice_calls = []

    async def render_alert_audio(self, parsed, script: str, *, same_locations=None):
        self.full_calls.append((parsed, script, list(same_locations or [])))
        path = self.base / f"full-{len(self.full_calls)}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFFfakeWAVE")
        return path

    async def render_voice_only_audio(self, script: str, *, prefix: str = "voice"):
        self.voice_calls.append((prefix, script))
        path = self.base / f"{prefix}-{len(self.voice_calls)}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFFfakeWAVE")
        return path


class _FakeTelnet:
    def __init__(self) -> None:
        self.pushed = []

    def flush_cycle(self) -> None:
        pass

    def push_alert(self, path: str, *, meta=None) -> None:
        self.pushed.append((path, meta or {}))


class _FakeDiscord:
    def __init__(self) -> None:
        self.calls = []

    def alert_aired(self, **kwargs) -> None:
        self.calls.append(("aired", kwargs))

    def alert_updated(self, **kwargs) -> None:
        self.calls.append(("updated", kwargs))

    def alert_expired(self, **kwargs) -> None:
        self.calls.append(("expired", kwargs))


class _FakeTargetResolver:
    def __init__(self, same_codes=None) -> None:
        self.same_codes = list(same_codes or ["024031"])

    def _filter_same_locations_to_service_area(self, locations):
        allowed = set(self.same_codes)
        return [str(x) for x in (locations or []) if str(x) in allowed]

    async def _nwws_same_targets_from_texts(self, raw_text: str, official_text: str):
        return ["MDC031"], list(self.same_codes), "test", True, None

    async def _nwws_wcn_watch_same_targets_from_area_desc(self, official_text: str):
        return list(self.same_codes)

    async def _sf_area_text_from_same_codes(self, codes):
        return "; ".join(str(x) for x in (codes or []))


class _FakeRefresher:
    def trigger_immediate(self, *_args, **_kwargs) -> None:
        pass


def _minimal_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ICECAST_SOURCE_PASSWORD", "test-source")
    monkeypatch.setenv("NWWS_JID", "changeme@nwws-oi.weather.gov")
    monkeypatch.setenv("NWWS_PASSWORD", "CHANGEME")
    cfg = load_config("config/config.yaml")
    return replace(
        cfg,
        paths=replace(
            cfg.paths,
            work_dir=str(tmp_path / "work"),
            audio_dir=str(tmp_path / "audio"),
            cache_dir=str(tmp_path / "cache"),
            config_dir=str(tmp_path / "config"),
            log_dir=str(tmp_path / "log"),
        ),
        database=replace(cfg.database, enabled=False),
        station_feed=replace(cfg.station_feed, enabled=False),
    )


def _orchestrator(tmp_path, monkeypatch) -> Orchestrator:
    orch = Orchestrator(_minimal_config(tmp_path, monkeypatch))
    orch.audio_originator = _FakeAudioOriginator(tmp_path / "audio")
    orch.telnet = _FakeTelnet()
    orch.discord = _FakeDiscord()
    orch.target_resolver = _FakeTargetResolver(["024031"])
    orch.targeting = orch.target_resolver
    orch.refresher = _FakeRefresher()
    return orch


def _cap_event(*, event="Severe Thunderstorm Warning", vtec_action="NEW") -> CapAlertEvent:
    vtec = f"/O.{vtec_action}.KLWX.SV.W.0123.260614T2000Z-260614T2030Z/"
    return CapAlertEvent(
        alert_id=f"urn:oid:2.49.0.1.840.0.test-{vtec_action}",
        sent="2026-06-14T20:00:00+00:00",
        status="Actual",
        message_type="Alert",
        event=event,
        severity="Severe",
        urgency="Immediate",
        certainty="Likely",
        headline=f"{event} issued June 14 at 4:00PM EDT",
        area_desc="Montgomery County",
        description="At 4:00 PM EDT, a severe thunderstorm was located near Rockville.",
        instruction="For your protection move to an interior room on the lowest floor of a building.",
        effective="2026-06-14T20:00:00+00:00",
        onset="2026-06-14T20:00:00+00:00",
        ends="2026-06-14T20:30:00+00:00",
        expires="2026-06-14T20:30:00+00:00",
        references=[],
        same_fips=["024031"],
        parameters={"VTEC": [vtec]},
        vtec=[vtec],
    )


def _nwws_product(*, product_type: str, action: str) -> tuple[ParsedProduct, str]:
    raw = f"""WUUS51 KLWX 142000
{product_type}LWX

Severe Thunderstorm Warning
National Weather Service Baltimore MD/Washington DC
400 PM EDT Sun Jun 14 2026

MDC031-142030-
/O.{action}.KLWX.SV.W.0123.260614T2000Z-260614T2030Z/
Montgomery MD-
400 PM EDT Sun Jun 14 2026

...A SEVERE THUNDERSTORM WARNING {'REMAINS IN EFFECT' if action == 'CON' else 'IS IN EFFECT'} UNTIL 430 PM EDT FOR MONTGOMERY COUNTY...

At 400 PM EDT, a severe thunderstorm was located near Rockville.

PRECAUTIONARY/PREPAREDNESS ACTIONS...
Move to an interior room on the lowest floor of a building.

$$
"""
    return ParsedProduct(product_type=product_type, wfo="KLWX", awips_id=f"{product_type}LWX", vtec=None, raw_text=raw), raw


def test_cap_full_runtime_path_smoke(tmp_path, monkeypatch):
    orch = _orchestrator(tmp_path, monkeypatch)

    asyncio.run(orch.cap_runtime.air_full(_cap_event()))

    assert orch.audio_originator.full_calls
    parsed, _script, same_locations = orch.audio_originator.full_calls[0]
    assert parsed.product_type == "SVR"
    assert same_locations == ["024031"]
    assert orch.telnet.pushed
    assert any(kind == "aired" for kind, _payload in orch.discord.calls)


def test_cap_voice_runtime_path_smoke(tmp_path, monkeypatch):
    orch = _orchestrator(tmp_path, monkeypatch)

    asyncio.run(orch.cap_runtime.air_voice(_cap_event(vtec_action="CON")))

    assert not orch.audio_originator.full_calls
    assert orch.audio_originator.voice_calls
    prefix, script = orch.audio_originator.voice_calls[0]
    assert prefix == "capvoice"
    assert "Severe Thunderstorm Warning" in script
    assert orch.telnet.pushed


def test_nwws_full_runtime_path_smoke(tmp_path, monkeypatch):
    orch = _orchestrator(tmp_path, monkeypatch)
    parsed, official_text = _nwws_product(product_type="SVR", action="NEW")

    async def _resolve(_parsed):
        return official_text, "test-product-id"

    orch._resolve_nwws_official_text = _resolve  # type: ignore[method-assign]

    asyncio.run(orch.nwws_runtime._handle_toneout(parsed))

    assert orch.audio_originator.full_calls
    render_parsed, _script, same_locations = orch.audio_originator.full_calls[0]
    assert render_parsed.product_type == "SVR"
    assert same_locations == ["024031"]
    assert orch.telnet.pushed


def test_nwws_voice_runtime_path_smoke(tmp_path, monkeypatch):
    orch = _orchestrator(tmp_path, monkeypatch)
    parsed, official_text = _nwws_product(product_type="SVS", action="CON")

    async def _resolve(_parsed):
        return official_text, "test-product-id"

    orch._resolve_nwws_official_text = _resolve  # type: ignore[method-assign]

    asyncio.run(orch.nwws_runtime._handle_toneout(parsed))

    assert not orch.audio_originator.full_calls
    assert orch.audio_originator.voice_calls
    prefix, script = orch.audio_originator.voice_calls[0]
    assert prefix == "nwwsvoice"
    assert "severe thunderstorm warning" in script.lower()
    assert orch.telnet.pushed
