# SeasonalWeather

SeasonalWeather is a Python-based, internet-delivered weather/alert radio stream inspired by NOAA Weather Radio (NWR) broadcast workflows: continuous cycle audio plus interrupting alert cut-ins (SAME header bursts, 1050 Hz attention tone, spoken content, EOM).

It's designed for homelab / hobby IP-based stream use with a focus on resiliency, dedupe, and "don't accidentally target every county in the service area."

> **Not affiliated with NOAA / NWS / FEMA.**
> This is an **unofficial** hobby project. **Do not** rely on it for life safety.
> Always use official sources for real warnings (e.g., NOAA Weather Radio, weather.gov, local authorities).

> **Please note, SeasonalWeather is a project coded with the use of generative AI.**
> Keep this in mind if you're against certain projects coded with the use of such tools.

## Safety, scope, and acceptable use

SeasonalWeather can generate **valid SAME headers** and **1050 Hz attention tones**. That power comes with responsibility.

- **Intended use:** homelab / internal / IP-based streaming and testing.
- **Not intended for:** over-the-air broadcasting, public alert origination, or any use that could be confused with an official EAS/NWR source.
- **You are responsible** for compliance with your local laws and regulations and for preventing misuse.

## Support and warranty

- Provided **as-is**, with **no warranty** of any kind.
- Support is **best-effort** and may be limited depending on available time.

---

## What it does

### Sources / ingestion

- **NWWS-OI (XMPP)** ingest — room monitoring for alert payloads from NWS.
- **NWS API** ingest — forecasts, observations, hazardous weather outlooks, area forecast discussions.
- **CAP ingest** — active alerts polling via `api.weather.gov/alerts/active`.
- **GWES-ERN / ERN-JON** — Level 3 stream monitoring; decodes live SAME headers from a radio stream and relays qualifying events.

### Alert behavior

Full cut-ins (when enabled) follow:

1. **SAME header burst(s)** targeted to the alert's affected SAME/FIPS codes
2. **1050 Hz attention tone**
3. **TTS narration**
4. **SAME EOM**
5. Return to normal cycle

Low-severity CAP alerts can optionally trigger **voice-only interruptions** (no SAME/tone/EOM) with per-event cooldowns.

Freeze products are supported natively. `FZ.W` maps to SAME `FZW` (**Freeze Warning**), `FZ.A` maps to SAME `FZA` (**Freeze Watch**), and `FSW` (**Flash Freeze Warning**) is recognized as a native event code. The repo default keeps **Freeze Watch** voice-only and leaves **Freeze Warning** / **Flash Freeze Warning** commented out in the tone-out list so operators can opt in deliberately.

### Dedupe and safety gates

- Cross-source deduplication prevents the same alert from airing twice via NWWS then CAP.
- Ledger-based CAP tracking prevents restart spam.
- Tests (RWT/RMT) are gated behind configurable cooldowns and per-test postpone policies (`none`, `fixed_delay`, `delay_window`, `next_day`, `skip_day`, `skip_week`).
- No "global fallback" targeting — alerts that don't match the service area are dropped.

### Output

- Audio is produced for **Liquidsoap**, which feeds **Icecast**.
- Typical mount: `/seasonalweather.ogg`

### Ops

- Runs under **systemd** (units included in `systemd/`).
- HTTP control API (localhost-only by default) for status, cycle control, and audio injection.
- Public handled-alerts feed API (`/v1/handled-alerts`) for external UI consumption, with `handled-alerts.json` kept as a legacy compatibility mirror.

---

## Repo layout

```
config/
  config.yaml       — repo template/example for runtime behaviour
  example.env       — secrets template (copy to /etc/seasonalweather/seasonalweather.env)
docs/               — design notes, state machine documentation, runtime wrapper notes
liquidsoap/         — Liquidsoap script(s)
scripts/            — bootstrap + helper scripts
  wrappers/         — version-controlled runtime wrapper scripts installed into /usr/local/bin
tools/              — standalone dev/debug utilities
systemd/            — systemd service unit files
seasonalweather/    — Python application
  main.py           — Orchestrator (hub, stays at root of package)
  control.py        — API → Orchestrator bridge
  config.py         — config loader
  discord_log.py    — Discord embed logger
  logging_config.py — central runtime/systemd logging policy
  liquidsoap_telnet.py
  cli/              — CLI tools (inject)
  same/             — SAME/EAS subsystem (encoder, decoder, event codes, listeners)
  alerts/           — alert lifecycle (CAP, NWWS products, VTEC, active registry)
  broadcast/        — cycle content generation (cycle, RWR, RWT/RMT, ERN, station feed)
  tts/              — speech synthesis (TTS engine, audio utils, VoiceText Paul VTML)
  api/              — HTTP control API (routes, models, auth, server entrypoint)
  nwws/             — NWWS-OI XMPP client and smoke-test
```

---

## Configuration

SeasonalWeather uses a **two-file configuration split**:

| File | Contains | Committed to repo? |
|---|---|---|
| `config.yaml` | All behaviour — service area, TTS, CAP, ERN, tests, cycle tuning, etc. | Yes (example) |
| `seasonalweather.env` | Secrets only — NWWS credentials, Icecast password, API tokens | **Never** |

### config.yaml

This is the single source of truth for all runtime behaviour. The file is well-commented and self-documenting. Top-level sections:

| Section | What it controls |
|---|---|
| `station` | Name, service area description, timezone, disclaimer |
| `stream` | Icecast host, port, mount |
| `cycle` | Broadcast interval, reference points, SPC/forecast/obs/HWO tuning |
| `observations` | ASOS/AWOS station IDs for current conditions |
| `nwws` | NWWS-OI server, allowed WFOs, resiliency knobs |
| `policy` | Product types that trigger tone-out |
| `cap.voice.events` | Voice-only CAP events; repo default includes `Freeze Watch` (`FZA`) here |
| `same` | SAME encoding: enabled, sender ID, amplitude |
| `cap` | CAP polling: enabled, dryrun, poll interval, voice/full thresholds |
| `ern` | ERN/GWES stream monitoring and relay settings |
| `samedec` | Path and confidence for the `samedec` binary |
| `tests` | RWT/RMT scheduling, gating, and postpone policy |
| `zonecounty` | NWS UGC → SAME FIPS crosswalk |
| `mareas` | Marine zone → SAME crosswalk |
| `station_feed` | Public handled-alerts feed state, SQLite persistence, and legacy JSON mirror output |
| `rebroadcast` | Periodic voice-only rebroadcast of active alerts |
| `api` | HTTP control API settings |
| `live_time` | Live time WAV update interval |
| `dedupe` | Cross-source deduplication window |
| `tts` | TTS backend selection and VoiceText Paul tuning |
| `audio` | Sample rate, tone frequencies, silence durations |
| `paths` | Working directories |
| `service_area` | Transmitter SAME/FIPS lists (the most important section) |

### seasonalweather.env

Only the following belong in this file:

```bash
# Required
NWWS_JID=yourjid@nwws-oi.weather.gov
NWWS_PASSWORD=yourpassword
ICECAST_SOURCE_PASSWORD=yourpassword

# Optional — only needed if using the HTTP API with auth
SEASONAL_API_TOKEN=yourtoken
SEASONAL_API_TOKENS_JSON='{ ... }'

# Optional — only if you changed the Liquidsoap telnet port
LIQUIDSOAP_TELNET_HOST=127.0.0.1
LIQUIDSOAP_TELNET_PORT=1234

# VoiceText Paul only — Wine subprocess environment vars
# (not read by Python; passed to the wine process by the synth wrapper)
VOICETEXT_PAUL_WINEDEBUG=-all,+seh,+tid,+timestamp
# ... etc
```

Everything else that was previously in `.env` (SEASONAL_CAP_*, SEASONAL_ERN_*, SEASONAL_TESTS_*, SEASONAL_CYCLE_*, VOICETEXT_PAUL_RETRIES, etc.) is now in `config.yaml`.

For the wrapper install/runtime contract, see `docs/runtime-wrappers.md`.

---

## Quick start

### 1) Clone to a staging path

Do **not** clone directly into `/opt/seasonalweather/app`. Clone to any temporary or staging path and run bootstrap from there. The bootstrapper rsyncs the repo into `/opt/seasonalweather/app` automatically.

```bash
git clone https://git.seasonalnet.org/Seasonal_Currency/SeasonalWeather /tmp/SeasonalWeather
cd /tmp/SeasonalWeather
```

### 2) Bootstrap

```bash
sudo bash scripts/00-bootstrap.sh
```

Optional backend installs:

```bash
SEASONAL_DECTALK=1 sudo -E bash scripts/00-bootstrap.sh
SEASONAL_VOICETEXT_PAUL=1 sudo -E bash scripts/00-bootstrap.sh
```

VoiceText Paul is smoke-tested during bootstrap. The repo-owned wrapper retries
stateful Wine crash exits such as rc=134/139 after resetting `wineserver`; if all
attempts fail, refresh from a known-good runtime with `SEASONAL_VOICETEXT_PAUL_SOURCE`
and `SEASONAL_VOICETEXT_PAUL_REFRESH=1`.

VoiceText Paul fresh installs use Wine's default amd64/WOW64-capable prefix
(`VOICETEXT_PAUL_WINEARCH=auto`) and install both `wine64` and `wine32`. This is
intentional: pure `win32` prefixes have been observed to segfault the bundled
VoiceText/Cygwin runtime on Debian trixie/Wine 10.

### 3) Configure the live files

```bash
# Behaviour — this is the LIVE config the service reads
sudo nano /etc/seasonalweather/config.yaml

# Secrets — fill in NWWS_JID, NWWS_PASSWORD, ICECAST_SOURCE_PASSWORD
sudo nano /etc/seasonalweather/seasonalweather.env
```

Do not edit `/opt/seasonalweather/app/config/config.yaml` and expect the running service to change. That file is the repo template/example only.

### 4) Start

```bash
sudo systemctl enable --now icecast2
sudo systemctl enable --now seasonalweather-liquidsoap
sudo systemctl enable --now seasonalweather
```

### 5) Check logs

```bash
journalctl -u seasonalweather -f
```

### Runtime logging policy

SeasonalWeather now has a central runtime logging policy in `seasonalweather/logging_config.py`, driven by `logs.runtime` in `config.yaml`.

The defaults are intentionally quieter for systemd/journalctl:
- suppress `httpx` and `httpcore` request lines unless you opt back in
- suppress routine CAP/IPAWS zero-change poll summaries
- suppress routine cycle conductor push chatter and segment refresher synthesis chatter
- keep normal `INFO`/`WARNING`/`ERROR` application logs for real state changes and failures

Example live config snippet:

```yaml
logs:
  runtime:
    level: INFO
    httpx_level: WARNING
    httpcore_level: WARNING
    uvicorn_access_level: WARNING
    uvicorn_error_level: INFO
    cap_poll_summary: false
    ipaws_poll_summary: false
    conductor_cycle_push: false
    conductor_alert_push: false
    conductor_live_time_push: false
    segment_refresher_synth: false
    segment_refresher_alert_lifecycle: false
    logger_levels:
      seasonalweather.nwws: INFO
```

Set the per-logger levels back to `INFO` or enable the boolean toggles when you want the firehose during troubleshooting.

### 6) Listen

```
http://<your-ip>:8000/seasonalweather.ogg
```

---

## Debug / test tool: inject

Generate a test alert WAV and optionally push it into the Liquidsoap queue:

```bash
/opt/seasonalweather/venv/bin/python -m seasonalweather.cli.inject --help
/opt/seasonalweather/venv/bin/python -m seasonalweather.cli.inject --event DMO --loc 024021 test-alert "hello world"
```

---

## GWES-ERN SAME decoding

SeasonalWeather uses `samedec` (Rust) for fast SAME decoding from the ERN stream by default. To switch back to the pure-Python decoder, edit `seasonalweather/broadcast/ern_gwes.py` and find `_same_listen_module_cmd` — change `same.listen_samedec` to `same.listen` and restart.

The `samedec` binary path and confidence threshold are configured in `config.yaml` under the `samedec:` section.
