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

### Dedupe and safety gates

- Cross-source deduplication prevents the same alert from airing twice via NWWS then CAP.
- Ledger-based CAP tracking prevents restart spam.
- Tests (RWT/RMT) are gated behind configurable cooldowns and blocked during active severe weather.
- No "global fallback" targeting — alerts that don't match the service area are dropped.

### Output

- Audio is produced for **Liquidsoap**, which feeds **Icecast**.
- Typical mount: `/seasonalweather.ogg`

### Ops

- Runs under **systemd** (units included in `systemd/`).
- HTTP control API (localhost-only by default) for status, cycle control, and audio injection.
- Station feed JSON (`handled-alerts.json`) for external UI consumption.

---

## Repo layout

```
config/
  config.yaml       — all behaviour configuration (edit this for your deployment)
  example.env       — secrets template (copy to /etc/seasonalweather/seasonalweather.env)
docs/               — design notes, state machine documentation
liquidsoap/         — Liquidsoap script(s)
scripts/            — bootstrap + helper scripts
seasonalweather/    — Python application
systemd/            — systemd service unit files
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
| `same` | SAME encoding: enabled, sender ID, amplitude |
| `cap` | CAP polling: enabled, dryrun, poll interval, voice/full thresholds |
| `ern` | ERN/GWES stream monitoring and relay settings |
| `samedec` | Path and confidence for the `samedec` binary |
| `tests` | RWT/RMT scheduling and gating |
| `zonecounty` | NWS UGC → SAME FIPS crosswalk |
| `mareas` | Marine zone → SAME crosswalk |
| `station_feed` | Alert feed JSON output for UI consumption |
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

---

## Quick start

### 1) System deps

```bash
# Debian/Ubuntu
sudo apt-get install -y python3 python3-venv ffmpeg icecast2 liquidsoap espeak-ng
```

### 2) Bootstrap (installs, creates user, sets up dirs)

```bash
sudo bash scripts/00-bootstrap.sh
```

### 3) Configure

```bash
# Behaviour — edit for your service area, TTS backend, schedule, etc.
sudo nano /etc/seasonalweather/config.yaml

# Secrets — fill in NWWS_JID, NWWS_PASSWORD, ICECAST_SOURCE_PASSWORD
sudo nano /etc/seasonalweather/seasonalweather.env
```

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

SeasonalWeather uses `samedec` (Rust) for fast SAME decoding from the ERN stream by default. To switch back to the pure-Python decoder, edit `seasonalweather/ern_gwes.py` and find `_same_listen_module_cmd` — change `same_listen_samedec` to `same_listen` and restart.

The `samedec` binary path and confidence threshold are configured in `config.yaml` under the `samedec:` section.
