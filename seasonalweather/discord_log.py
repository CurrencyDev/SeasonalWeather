"""
seasonalweather/discord_log.py

Centralised, non-blocking, rate-limited Discord embed poster.

All public methods are synchronous fire-and-forget: they enqueue a payload
and return immediately so the alert path is never delayed.

A single asyncio drain task (started by Orchestrator.run()) pulls from the
queue, applies per-URL token-bucket rate limiting, and posts to Discord.
On HTTP 429 the payload is re-enqueued after Retry-After expires.
The queue never drops; during severe outbreaks it drains at the rate Discord
allows (~20/min per webhook).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from typing import Any

import httpx

log = logging.getLogger("seasonalweather.discord_log")

# ---------------------------------------------------------------------------
# NWS hazard-map sidebar colors (6-char hex, no #)
# Source: https://www.weather.gov/help-map
# ---------------------------------------------------------------------------
_NWS_COLORS: dict[str, str] = {
    # Tornado
    "TOR": "FF0000",
    "TOA": "FFFF00",
    # Severe Thunderstorm
    "SVR": "FF8C00",
    "SVA": "DB7093",
    # Flash Flood / Flood
    "FFW": "8B0000",
    "FFA": "2E8B57",
    "FLW": "00FF00",
    "FLA": "2E8B57",
    "FLS": "00FF00",
    "FFS": "8B0000",
    # Marine
    "SMW": "FF8C00",
    "CFW": "6495ED",
    "CFA": "6495ED",
    # Winter
    "WSW": "FF69B4",
    "WSA": "4682B4",
    "BZW": "FF4500",
    "SQW": "C0C0C0",
    "WW":  "7B68EE",
    "ISW": "8B008B",
    # Wind
    "HWW": "DAA520",
    "HWA": "DAA520",
    "EWW": "FF8C00",
    "DSW": "FFE4B5",
    # Hurricane / Tropical / Surge
    "HUW": "DC143C",
    "HUA": "FF8C00",
    "TRW": "B22222",
    "TRA": "F4A460",
    "HLS": "FFE4B5",
    "SSW": "9400D3",
    "SSA": "DB7093",
    # Fire
    "FRW": "FF1493",
    "FWW": "FF0000",
    "FWA": "FF6347",
    # Civil / Emergency
    "CAE": "00FFFF",
    "CDW": "FFE4B5",
    "CEM": "FFD700",
    "LAE": "C0C0C0",
    "EAN": "000000",
    "EVI": "FF6347",
    "TOE": "C0C0C0",
    "SPW": "C0C0C0",
    "LEW": "C0C0C0",
    "BLU": "1E90FF",
    "ADR": "C0C0C0",
    "NMN": "C0C0C0",
    "MEP": "FF69B4",
    # Hazmat / Nuke / Radiological
    "HMW": "4B0082",
    "NUW": "4B0082",
    "RHW": "4B0082",
    # Earthquake / Volcano
    "EQW": "8B4513",
    "VOW": "8B4513",
    "VOA": "A0522D",
    # Tsunami / Avalanche
    "TSW": "1C00FF",
    "TSA": "00CED1",
    "AVW": "1E90FF",
    "AVA": "87CEEB",
    # Tests
    "RWT": "C0C0C0",
    "RMT": "A9A9A9",
    "DMO": "A9A9A9",
    # Statements / voice-only
    "SPS": "FFE4B5",
    "SVS": "00FF7F",
}

_DEFAULT_ALERT_COLOR = "4169E1"
_OPS_COLOR           = "378ADD"
_API_SUCCESS_COLOR   = "639922"
_API_FAIL_COLOR      = "E24B4A"
_WARN_COLOR          = "BA7517"
_ERROR_COLOR         = "E24B4A"

# ---------------------------------------------------------------------------
# EAS event code → Lucide icon name
# All icons confirmed present in lucide-static / lucide-react.
# ---------------------------------------------------------------------------
_ICON_MAP: dict[str, str] = {
    # Tornado / convective
    "TOR": "siren",
    "TOA": "siren",
    # Severe Thunderstorm
    "SVR": "cloud-lightning",
    "SVA": "cloud-lightning",
    # Flood
    "FFW": "waves",
    "FFA": "waves",
    "FLW": "waves",
    "FLA": "waves",
    "FLS": "waves",
    "FFS": "waves",
    # Marine
    "SMW": "anchor",
    "CFW": "anchor",
    "CFA": "anchor",
    # Winter / ice / snow / squall
    "WSW": "snowflake",
    "WSA": "snowflake",
    "BZW": "snowflake",
    "SQW": "snowflake",
    "WW":  "snowflake",
    "ISW": "snowflake",
    # Wind
    "HWW": "wind",
    "HWA": "wind",
    "EWW": "wind",
    "DSW": "wind",
    # Hurricane / tropical / surge
    "HUW": "wind",
    "HUA": "wind",
    "TRW": "wind",
    "TRA": "wind",
    "HLS": "wind",
    "SSW": "waves",
    "SSA": "waves",
    # Fire
    "FRW": "flame",
    "FWW": "flame",
    "FWA": "flame",
    # Civil / emergency
    "CAE": "shield-alert",
    "CDW": "shield-alert",
    "CEM": "shield-alert",
    "LAE": "shield-alert",
    "EAN": "shield-alert",
    "EVI": "shield-alert",
    "TOE": "phone-missed",
    "SPW": "shield",
    "LEW": "shield",
    "BLU": "shield",
    "MEP": "shield-alert",
    # Hazmat / nuke / rad
    "HMW": "triangle-alert",
    "NUW": "triangle-alert",
    "RHW": "triangle-alert",
    # Earthquake / volcano
    "EQW": "mountain",
    "VOW": "mountain",
    "VOA": "mountain",
    # Tsunami / avalanche
    "TSW": "waves",
    "TSA": "waves",
    "AVW": "mountain-snow",
    "AVA": "mountain-snow",
    # Tests
    "RWT": "radio",
    "RMT": "radio",
    "DMO": "radio",
    # Statements / voice-only
    "SPS": "bell",
    "SVS": "bell",
    # Internal sentinels
    "_voice":   "bell",
    "_expired": "bell-off",
    "_test":    "radio",
    "_ops":     "activity",
    "_api":     "terminal",
    "_error":   "circle-alert",
    "_startup": "power",
    "_stall":   "wifi-off",
    "_default": "bell-ring",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _color_int(hex_str: str) -> int:
    try:
        return int(hex_str.lstrip("#"), 16)
    except (ValueError, AttributeError):
        return 0x4169E1


def _alert_color_int(code: str) -> int:
    return _color_int(_NWS_COLORS.get((code or "").strip().upper(), _DEFAULT_ALERT_COLOR))


def _alert_color_hex(code: str) -> str:
    return _NWS_COLORS.get((code or "").strip().upper(), _DEFAULT_ALERT_COLOR)


def _icon_name(code: str, *, sentinel: str | None = None) -> str:
    if sentinel:
        return _ICON_MAP.get(sentinel, _ICON_MAP["_default"])
    return _ICON_MAP.get((code or "").strip().upper(), _ICON_MAP["_default"])


def _icon_url(cdn_base: str, icon: str, hex_color: str) -> str | None:
    if not cdn_base:
        return None
    return f"{cdn_base.rstrip('/')}/icon?icon={icon}&hex={hex_color.lstrip('#').upper()}"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _area_display(area: str) -> str:
    """Format a semicolon-separated area string as ' · ' joined inline code."""
    parts = [p.strip() for p in (area or "").split(";") if p.strip()]
    joined = " · ".join(parts) if parts else (area or "").strip()
    return f"`{joined}`" if joined else "—"


# ---------------------------------------------------------------------------
# Token bucket — per webhook URL
# ---------------------------------------------------------------------------

class _TokenBucket:
    """
    Leaky token bucket: `capacity` = max burst, refills at `rate_per_minute / 60`
    tokens per second. consume() returns 0.0 if a token is available immediately,
    or the fractional seconds to wait until one is.
    """

    def __init__(self, rate_per_minute: int) -> None:
        self._capacity = float(max(1, rate_per_minute))
        self._tokens   = self._capacity
        self._rate     = self._capacity / 60.0   # tokens / second
        self._last     = time.monotonic()

    def consume(self) -> float:
        now = time.monotonic()
        self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return 0.0
        return (1.0 - self._tokens) / self._rate


# ---------------------------------------------------------------------------
# DiscordLogger
# ---------------------------------------------------------------------------

class DiscordLogger:
    """
    Fire-and-forget Discord embed poster for SeasonalWeather.

    Usage:
        # In Orchestrator.__init__:
        self.discord = DiscordLogger.from_config(cfg.logs.discord)

        # In Orchestrator.run(), before awaiting tasks:
        tasks.append(asyncio.create_task(self.discord.start(), name="discord_log_drain"))

        # Anywhere in the codebase:
        self.discord.alert_aired(code="TOR", event="Tornado Warning", ...)
    """

    def __init__(
        self,
        *,
        alerts_url: str = "",
        alerts_enabled: bool = True,
        ops_url: str = "",
        ops_enabled: bool = True,
        api_url: str = "",
        api_enabled: bool = True,
        errors_url: str = "",
        errors_enabled: bool = True,
        rate_limit_per_minute: int = 20,
        post_tests: bool = True,
        post_voice_only: bool = True,
        cycle_rebuild_log: bool = True,
        alerttracker_lifecycle_log: bool = False,
        icon_cdn_url: str = "",
    ) -> None:
        self._alerts_url     = alerts_url.strip()
        self._alerts_enabled = alerts_enabled and bool(self._alerts_url)
        self._ops_url        = ops_url.strip()
        self._ops_enabled    = ops_enabled and bool(self._ops_url)
        self._api_url        = api_url.strip()
        self._api_enabled    = api_enabled and bool(self._api_url)
        self._errors_url     = errors_url.strip()
        self._errors_enabled = errors_enabled and bool(self._errors_url)

        self._rate_limit     = max(1, rate_limit_per_minute)
        self._post_tests     = post_tests
        self._post_voice     = post_voice_only
        self._cycle_log      = cycle_rebuild_log
        self._tracker_log    = alerttracker_lifecycle_log
        self._icon_cdn       = icon_cdn_url.strip()

        self._queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self._buckets: dict[str, _TokenBucket] = {}
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_config(cls, cfg: Any) -> "DiscordLogger":
        """Construct from a LogsDiscordConfig dataclass."""
        if not cfg.enabled:
            return cls()  # All URLs empty → nothing fires
        return cls(
            alerts_url=cfg.alerts_url,
            alerts_enabled=cfg.alerts_enabled,
            ops_url=cfg.ops_url,
            ops_enabled=cfg.ops_enabled,
            api_url=cfg.api_url,
            api_enabled=cfg.api_enabled,
            errors_url=cfg.errors_url,
            errors_enabled=cfg.errors_enabled,
            rate_limit_per_minute=cfg.rate_limit_per_minute,
            post_tests=cfg.post_tests,
            post_voice_only=cfg.post_voice_only,
            cycle_rebuild_log=cfg.cycle_rebuild_log,
            alerttracker_lifecycle_log=cfg.alerttracker_lifecycle_log,
            icon_cdn_url=cfg.icon_cdn_url,
        )

    def _any_enabled(self) -> bool:
        return any([self._alerts_enabled, self._ops_enabled,
                    self._api_enabled, self._errors_enabled])

    def _bucket(self, url: str) -> _TokenBucket:
        if url not in self._buckets:
            self._buckets[url] = _TokenBucket(self._rate_limit)
        return self._buckets[url]

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
                headers={"User-Agent": "SeasonalWeather/1.0"},
            )
        return self._client

    def _enqueue(self, url: str, payload: dict) -> None:
        if not url:
            return
        self._queue.put_nowait((url, payload))

    # ------------------------------------------------------------------
    # Background drain task
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Long-running drain coroutine. Schedule as an asyncio Task:
            asyncio.create_task(self.discord.start(), name="discord_log_drain")
        """
        if not self._any_enabled():
            log.info("Discord webhook logging: disabled (no URLs configured or all disabled)")
            return

        log.info(
            "Discord webhook logging started "
            "(alerts=%s ops=%s api=%s errors=%s rate=%d/min icon_cdn=%s)",
            self._alerts_enabled, self._ops_enabled,
            self._api_enabled, self._errors_enabled,
            self._rate_limit,
            self._icon_cdn or "(none)",
        )

        while True:
            try:
                url, payload = await self._queue.get()
                bucket = self._bucket(url)
                wait = bucket.consume()
                if wait > 0:
                    await asyncio.sleep(wait)

                try:
                    client = await self._ensure_client()
                    r = await client.post(url, json=payload)
                    if r.status_code == 429:
                        try:
                            retry_after = float(r.headers.get("Retry-After", "5"))
                        except (ValueError, TypeError):
                            retry_after = 5.0
                        log.warning(
                            "Discord webhook 429; re-enqueuing after %.1fs (url=%.50s...)",
                            retry_after, url,
                        )
                        await asyncio.sleep(retry_after)
                        self._enqueue(url, payload)
                    elif r.status_code not in (200, 204):
                        log.warning(
                            "Discord webhook unexpected status %d (url=%.50s...)",
                            r.status_code, url,
                        )
                except Exception:
                    log.exception("Discord webhook post failed (url=%.50s...)", url)
            except Exception:
                log.exception("Discord log drain loop error")

    # ------------------------------------------------------------------
    # Internal embed / payload builders
    # ------------------------------------------------------------------

    def _icon_url(self, icon: str, hex_color: str) -> str | None:
        return _icon_url(self._icon_cdn, icon, hex_color)

    @staticmethod
    def _field(name: str, value: str, *, inline: bool = True) -> dict:
        return {"name": name, "value": (value or "—")[:1024], "inline": inline}

    @staticmethod
    def _embed(
        *,
        color: int,
        title: str,
        description: str = "",
        fields: list[dict] | None = None,
        footer_text: str = "",
        thumbnail_url: str | None = None,
        author_name: str = "",
        author_icon_url: str | None = None,
        timestamp: str | None = None,
    ) -> dict:
        e: dict[str, Any] = {"color": color, "title": title[:256]}
        if description:
            e["description"] = description[:4096]
        if fields:
            e["fields"] = fields[:25]
        if author_name:
            a: dict[str, Any] = {"name": author_name[:256]}
            if author_icon_url:
                a["icon_url"] = author_icon_url
            e["author"] = a
        if thumbnail_url:
            e["thumbnail"] = {"url": thumbnail_url}
        if footer_text:
            e["footer"] = {"text": footer_text[:2048]}
        e["timestamp"] = timestamp or _now_iso()
        return e

    @staticmethod
    def _payload(embeds: list[dict]) -> dict:
        return {"username": "SeasonalWeather", "embeds": embeds[:10]}

    # ------------------------------------------------------------------
    # ── ALERTS CHANNEL ────────────────────────────────────────────────
    # ------------------------------------------------------------------

    def alert_aired(
        self,
        *,
        code: str,
        event: str,
        source: str,
        mode: str,
        area: str = "",
        vtec: list[str] | None = None,
        expires: str = "",
        is_test: bool = False,
        is_ern: bool = False,
    ) -> None:
        """
        A full or voice-only alert has been aired.

        mode:  "full" | "voice" | "voice_only"
        source: human-readable label, e.g. "CAP + NWWS-OI", "NWWS-OI", "ERN/GWES"
        """
        if not self._alerts_enabled:
            return
        if is_test and not self._post_tests:
            return
        mode_l = (mode or "").strip().lower()
        is_voice = mode_l in {"voice", "voice_only"}
        if is_voice and not self._post_voice:
            return

        code_u = (code or "SPS").strip().upper()
        hex_c  = _alert_color_hex(code_u)
        color  = _color_int(hex_c)

        if is_test:
            icon = _icon_name(code_u, sentinel="_test")
        elif is_voice:
            # voice-only: use event-specific icon but prefer the "quiet" variant
            icon = _icon_name(code_u, sentinel="_voice")
        else:
            icon = _icon_name(code_u)

        thumb = self._icon_url(icon, hex_c)

        event_str    = (event or code_u).strip()
        mode_display = (
            "Originated" if is_test
            else ("FULL + SAME" if not is_voice else "Voice-only")
        )
        if is_ern:
            mode_display += " (ERN relay)"

        title = f"{event_str} — {'originated' if is_test else 'aired'}"

        fields: list[dict] = [
            self._field("Source", source, inline=True),
            self._field("Mode", mode_display, inline=True),
            self._field("Code", code_u, inline=True),
        ]
        if area:
            fields.append(self._field("Area", _area_display(area), inline=False))
        if expires:
            fields.append(self._field("Expires", expires, inline=True))
        for v in (vtec or [])[:2]:
            fields.append(self._field("VTEC", f"`{v}`", inline=False))

        src_slug = (
            "local" if is_test
            else "ern" if is_ern
            else "cap" if "cap" in source.lower()
            else "nwws"
        )
        embed = self._embed(
            color=color,
            title=title,
            fields=fields,
            footer_text=f"SeasonalWeather · {src_slug} · alerts",
            thumbnail_url=thumb,
        )
        self._enqueue(self._alerts_url, self._payload([embed]))

    def alert_updated(
        self,
        *,
        code: str,
        event: str,
        vtec_action: str,
        source: str = "CAP",
        area: str = "",
        vtec: list[str] | None = None,
    ) -> None:
        """CON/EXT/EXA/EXB — voice-only continuation/extension, no retone."""
        if not self._alerts_enabled or not self._post_voice:
            return

        code_u = (code or "SPS").strip().upper()
        hex_c  = _alert_color_hex(code_u)
        color  = _color_int(hex_c)
        thumb  = self._icon_url(_icon_name(code_u, sentinel="_voice"), hex_c)

        action_labels = {
            "CON": "continuing", "EXT": "extended",
            "EXA": "expanded", "EXB": "expanded",
        }
        action_word = action_labels.get(vtec_action.upper(), vtec_action.lower())
        title = f"{(event or code_u).strip()} — {action_word}"

        fields: list[dict] = [
            self._field("VTEC action", vtec_action.upper(), inline=True),
            self._field("Source", source, inline=True),
            self._field("Mode", "Voice-only (no retone)", inline=True),
        ]
        if area:
            fields.append(self._field("Area", _area_display(area), inline=False))
        for v in (vtec or [])[:1]:
            fields.append(self._field("VTEC", f"`{v}`", inline=False))

        embed = self._embed(
            color=color, title=title, fields=fields,
            footer_text=f"SeasonalWeather · {source.lower()} update · alerts",
            thumbnail_url=thumb,
        )
        self._enqueue(self._alerts_url, self._payload([embed]))

    def alert_expired(
        self,
        *,
        code: str,
        event: str,
        vtec_action: str,
        source: str = "CAP",
        area: str = "",
        vtec: list[str] | None = None,
    ) -> None:
        """CAN/EXP — sidebar gray, icon bell-off."""
        if not self._alerts_enabled:
            return

        code_u = (code or "SPS").strip().upper()
        color  = _color_int("888888")
        thumb  = self._icon_url(_ICON_MAP["_expired"], "888888")

        action_labels = {"CAN": "cancelled", "EXP": "expired"}
        action_word = action_labels.get(vtec_action.upper(), vtec_action.lower())
        title = f"{(event or code_u).strip()} — {action_word}"

        fields: list[dict] = [
            self._field("VTEC action", vtec_action.upper(), inline=True),
            self._field("Source", source, inline=True),
            self._field("Mode", "Voice-only (no retone)", inline=True),
        ]
        if area:
            fields.append(self._field("Area", _area_display(area), inline=False))
        for v in (vtec or [])[:1]:
            fields.append(self._field("VTEC", f"`{v}`", inline=False))

        embed = self._embed(
            color=color, title=title, fields=fields,
            footer_text=f"SeasonalWeather · {source.lower()} · alerts",
            thumbnail_url=thumb,
        )
        self._enqueue(self._alerts_url, self._payload([embed]))

    # ------------------------------------------------------------------
    # ── OPS CHANNEL ───────────────────────────────────────────────────
    # ------------------------------------------------------------------

    def service_started(
        self,
        *,
        cap_enabled: bool = False,
        ern_enabled: bool = False,
        tests_enabled: bool = False,
        mode: str = "normal",
    ) -> None:
        if not self._ops_enabled:
            return
        thumb = self._icon_url(_ICON_MAP["_startup"], _OPS_COLOR)
        fields = [
            self._field("CAP",     "Enabled"  if cap_enabled   else "Disabled", inline=True),
            self._field("ERN",     "Enabled"  if ern_enabled   else "Disabled", inline=True),
            self._field("RWT/RMT", "Enabled"  if tests_enabled else "Disabled", inline=True),
            self._field("Mode",    mode.capitalize(), inline=True),
        ]
        embed = self._embed(
            color=_color_int(_OPS_COLOR),
            title="Service started",
            description="Orchestrator online. Liquidsoap reachable.",
            fields=fields,
            footer_text="SeasonalWeather · startup · ops",
            thumbnail_url=thumb,
        )
        self._enqueue(self._ops_url, self._payload([embed]))

    def cycle_rebuilt(
        self,
        *,
        reason: str = "scheduled",
        mode: str = "normal",
        interval: int = 300,
        seq_dur: float = 0.0,
        segments: int = 0,
        active_alerts: int = 0,
    ) -> None:
        if not self._ops_enabled or not self._cycle_log:
            return
        thumb = self._icon_url(_ICON_MAP["_ops"], _OPS_COLOR)
        dur_str = f"{int(seq_dur // 60)}m {int(seq_dur % 60)}s" if seq_dur > 0 else "—"
        fields: list[dict] = [
            self._field("Reason",       reason,           inline=True),
            self._field("Mode",         mode.capitalize(),inline=True),
            self._field("Interval",     f"{interval}s",   inline=True),
            self._field("Seq duration", dur_str,          inline=True),
            self._field("Segments",     str(segments),    inline=True),
        ]
        if active_alerts:
            fields.append(self._field("Active alerts", str(active_alerts), inline=True))
        embed = self._embed(
            color=_color_int(_OPS_COLOR),
            title="Cycle rebuilt",
            fields=fields,
            footer_text="SeasonalWeather · cycle · ops",
            thumbnail_url=thumb,
        )
        self._enqueue(self._ops_url, self._payload([embed]))

    def mode_changed(
        self,
        *,
        old_mode: str,
        new_mode: str,
        reason: str = "",
    ) -> None:
        if not self._ops_enabled:
            return
        hex_c = _WARN_COLOR if new_mode == "heightened" else _OPS_COLOR
        thumb = self._icon_url(_ICON_MAP["_ops"], hex_c)
        fields: list[dict] = [
            self._field("Old mode", old_mode.capitalize(), inline=True),
            self._field("New mode", new_mode.capitalize(), inline=True),
        ]
        if reason:
            fields.append(self._field("Reason", reason, inline=False))
        embed = self._embed(
            color=_color_int(hex_c),
            title=f"Mode changed → {new_mode}",
            fields=fields,
            footer_text="SeasonalWeather · mode · ops",
            thumbnail_url=thumb,
        )
        self._enqueue(self._ops_url, self._payload([embed]))

    def nwws_stall(self, *, backoff_attempt: int = 1) -> None:
        if not self._ops_enabled:
            return
        thumb = self._icon_url(_ICON_MAP["_stall"], _WARN_COLOR)
        embed = self._embed(
            color=_color_int(_WARN_COLOR),
            title="NWWS stall — reconnecting",
            description=(
                "No messages received from nwws-oi.weather.gov within the "
                "configured stall window. Initiating reconnect with backoff."
            ),
            fields=[self._field("Backoff attempt", f"#{backoff_attempt}", inline=True)],
            footer_text="SeasonalWeather · nwws · ops",
            thumbnail_url=thumb,
        )
        self._enqueue(self._ops_url, self._payload([embed]))

    def nwws_reconnected(self) -> None:
        if not self._ops_enabled:
            return
        embed = self._embed(
            color=_color_int(_OPS_COLOR),
            title="NWWS reconnected",
            footer_text="SeasonalWeather · nwws · ops",
            thumbnail_url=self._icon_url(_ICON_MAP["_ops"], _OPS_COLOR),
        )
        self._enqueue(self._ops_url, self._payload([embed]))

    def alerttracker_lifecycle(
        self,
        *,
        loaded: int,
        purged: int,
        active: int,
    ) -> None:
        """Startup AlertTracker restore summary. Gated by alerttracker_lifecycle_log knob."""
        if not self._ops_enabled or not self._tracker_log:
            return
        embed = self._embed(
            color=_color_int(_OPS_COLOR),
            title="AlertTracker restored",
            fields=[
                self._field("Loaded",  str(loaded),  inline=True),
                self._field("Purged",  str(purged),  inline=True),
                self._field("Active",  str(active),  inline=True),
            ],
            footer_text="SeasonalWeather · alerttracker · ops",
            thumbnail_url=self._icon_url(_ICON_MAP["_ops"], _OPS_COLOR),
        )
        self._enqueue(self._ops_url, self._payload([embed]))

    # ------------------------------------------------------------------
    # ── API CHANNEL ───────────────────────────────────────────────────
    # ------------------------------------------------------------------

    def api_action(
        self,
        *,
        method: str,
        endpoint: str,
        actor: str,
        status: str,
        command_id: str = "",
        headline: str = "",
        details: dict | None = None,
        success: bool = True,
    ) -> None:
        if not self._api_enabled:
            return
        hex_c = _API_SUCCESS_COLOR if success else _API_FAIL_COLOR
        color = _color_int(hex_c)
        thumb = self._icon_url(_ICON_MAP["_api"], hex_c)

        title = f"{method.upper()} {endpoint}"
        if not success:
            title += " — rejected"

        fields: list[dict] = [
            self._field("Actor",  actor or "unknown", inline=True),
            self._field("Status", status,             inline=True),
        ]
        if command_id:
            fields.append(self._field("Command ID", f"`{command_id}`", inline=True))
        if headline:
            fields.append(self._field("Headline", headline, inline=False))
        if details:
            for k, v in list(details.items())[:4]:
                fields.append(self._field(str(k), str(v)[:100], inline=True))

        embed = self._embed(
            color=color, title=title, fields=fields,
            footer_text="SeasonalWeather · api",
            thumbnail_url=thumb,
        )
        self._enqueue(self._api_url, self._payload([embed]))

    # ------------------------------------------------------------------
    # ── ERRORS CHANNEL ────────────────────────────────────────────────
    # ------------------------------------------------------------------

    def error(
        self,
        *,
        title: str,
        module: str = "",
        exception_type: str = "",
        message: str = "",
        context: dict | None = None,
        fallback: str = "",
    ) -> None:
        if not self._errors_enabled:
            return
        thumb = self._icon_url(_ICON_MAP["_error"], _ERROR_COLOR)
        fields: list[dict] = []
        if module:
            fields.append(self._field("Module", module, inline=True))
        if exception_type:
            fields.append(self._field("Exception", exception_type, inline=True))
        if fallback:
            fields.append(self._field("Fallback", fallback, inline=False))
        if context:
            for k, v in list(context.items())[:4]:
                fields.append(self._field(str(k), str(v)[:100], inline=True))
        embed = self._embed(
            color=_color_int(_ERROR_COLOR),
            title=title,
            description=(message[:300] if message else ""),
            fields=fields or None,
            footer_text="SeasonalWeather · errors",
            thumbnail_url=thumb,
        )
        self._enqueue(self._errors_url, self._payload([embed]))

    def warning(
        self,
        *,
        title: str,
        module: str = "",
        message: str = "",
        context: dict | None = None,
    ) -> None:
        if not self._errors_enabled:
            return
        thumb = self._icon_url(_ICON_MAP["_error"], _WARN_COLOR)
        fields: list[dict] = []
        if module:
            fields.append(self._field("Module", module, inline=True))
        if context:
            for k, v in list(context.items())[:4]:
                fields.append(self._field(str(k), str(v)[:100], inline=True))
        embed = self._embed(
            color=_color_int(_WARN_COLOR),
            title=title,
            description=(message[:300] if message else ""),
            fields=fields or None,
            footer_text="SeasonalWeather · errors",
            thumbnail_url=thumb,
        )
        self._enqueue(self._errors_url, self._payload([embed]))
