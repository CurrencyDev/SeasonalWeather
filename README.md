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
- **NWS JSON-LD/GeoJSON ingest** — active alerts polling via `api.weather.gov/alerts/active`.
- **FEMA CAP ingest** - CAP XML formatted alerts polling via `https://apps.fema.gov/IPAWSOPEN_EAS_SERVICE/rest/eas/recent/`
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
- Liquidsoap uses separate interrupt planes for FULL alerts, VOICE alerts, and routine cycle audio; FULL alerts have the highest priority and can preempt lower-priority VOICE updates.
- NWS spoken-alert prose is centralized in the broadcast product text helpers; CAP/JSON-LD, IPAWS, NWWS-OI, and API backfill remain transport/parser paths feeding the same NWS presentation layer.
- Interrupt alert audio render/push work runs through a priority dispatcher so FULL jobs are admitted ahead of queued VOICE jobs before Liquidsoap playout.
- While interrupt audio is active, routine cycle production is held; after the FULL/VOICE planes become idle, a guarded cycle-only reset discards paused/queued stale audio and rebuilds from station ID and the current time.
- Typical mounts: `/seasonalweather.ogg`, `/seasonalweather.mp3`

### Ops

- Runs under **systemd** (units included in `systemd/`).
- HTTP control API (localhost-only by default) for status, cycle control, bounded broadcast inserts, and audio injection.
- OpenAPI 3.1 API document at `/openapi.json`; Swagger UI remains available at `/docs`.
- RFC 9457 Problem Details error responses (`application/problem+json`) with `code`, `details`, and `request_id` extensions for operator/debug use.
- Public handled-alerts feed API (`/v1/handled-alerts`) for external UI consumption, backed by SQLite.

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
| `cycle` | Broadcast reference points, SPC/forecast/obs/HWO tuning, heightened-mode policy, and alert-focus hold policy |
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
| `station_feed` | Public handled-alerts feed state backed by the SQLite read model |
| `api` | HTTP control API settings |
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

## HTTP API contract

The SeasonalWeather API publishes an OpenAPI 3.1 document at `/openapi.json` and interactive Swagger UI at `/docs`. The document includes the public station-feed routes and authenticated control-plane routes.

Successful JSON endpoints use `application/json`. API errors use RFC 9457 Problem Details with `application/problem+json`; callers should read the standard `type`, `title`, `status`, `detail`, and `instance` members and may also use the SeasonalWeather extensions `code`, `details`, `errors`, and `request_id`. The same request identifier is returned in the `X-Request-ID` response header.

`/v1/handled-alerts` remains public and cacheable for SPA/radio clients. Authenticated routes require Bearer tokens and mutating JSON routes require an `Idempotency-Key` header. Scheduled broadcast inserts live under `/v1/inserts/*` and require the `control:inserts` scope; they add bounded, non-SAME text or uploaded-audio segments into the normal cycle without flushing the Liquidsoap cycle queue.

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

The bootstrapper is interactive by default when attached to a TTY. It tries to
use `dialog` or `whiptail` for a terminal menu, then falls back to numbered
stdout/stdin prompts with a notice if those helpers are unavailable. Automation
can keep using env vars or pass `--non-interactive`.

Install profiles:

```bash
sudo bash scripts/00-bootstrap.sh --profile standard
sudo bash scripts/00-bootstrap.sh --profile voicetext-paul
sudo bash scripts/00-bootstrap.sh --profile dectalk
sudo bash scripts/00-bootstrap.sh --profile minimal
```

Optional feature flags remain available for scripted installs:

```bash
SEASONAL_ESPEAK=1 SEASONAL_SAMEDEC=1 sudo -E bash scripts/00-bootstrap.sh --non-interactive
SEASONAL_PIPER=1 sudo -E bash scripts/00-bootstrap.sh --non-interactive
SEASONAL_FESTIVAL=1 sudo -E bash scripts/00-bootstrap.sh --non-interactive
SEASONAL_DECTALK=1 sudo -E bash scripts/00-bootstrap.sh --non-interactive
SEASONAL_VOICETEXT_PAUL=1 sudo -E bash scripts/00-bootstrap.sh --non-interactive
```

The base Python requirements no longer install the Piper/ONNX/NumPy stack.
Piper is installed only when `SEASONAL_PIPER=1` or the custom interactive
selection enables it. Festival and DECtalk build dependencies are likewise
feature-selected instead of being installed for every host.

The configuration assistant defaults to preserving an existing live config when
`/etc/seasonalweather/config.yaml` already exists. Use `--profile standard`,
`--profile voicetext-paul`, or another explicit profile only when intentionally
regenerating those high-level config choices.

The bootstrap installs the pinned Rust `samedec` decoder by default in the
standard, DECtalk, and VoiceText Paul profiles because ERN/GWES monitoring uses
it when available. To skip it on a minimal/native-decoder-only host, run:

```bash
SEASONAL_SAMEDEC=0 sudo -E bash scripts/00-bootstrap.sh --non-interactive
```

To deliberately refresh or change the pinned crate version:

```bash
SEASONAL_SAMEDEC_VERSION=0.4.2 sudo -E bash scripts/00-bootstrap.sh --non-interactive
```

VoiceText Paul is smoke-tested during bootstrap. The repo-owned wrapper retries
stateful Wine crash exits such as rc=134/139 after resetting `wineserver`; if all
attempts fail, refresh from a known-good runtime with `SEASONAL_VOICETEXT_PAUL_SOURCE`
and `SEASONAL_VOICETEXT_PAUL_REFRESH=1`.

VoiceText Paul fresh installs use a 32-bit Wine prefix
(`VOICETEXT_PAUL_WINEARCH=win32`) and install the 32-bit Wine stack. Bootstrap
also runs `loginctl enable-linger voicetext` so Wine subprocesses keep a stable
systemd user runtime instead of having logind tear the user session state down
between short-lived synth invocations. Operators may set
`VOICETEXT_PAUL_WINEARCH=auto` only as a known-good fallback.

### 3) Configure the live files

Use the configuration assistant to generate a candidate live config:

```bash
sudo seasonalweather-configure
```

The assistant writes `/etc/seasonalweather/config.yaml.new`, validates that the
application can load it, and offers to back up and apply it. It is profile-driven
and covers common operational settings; advanced operators may still edit the
live YAML directly. Generated candidate YAML does not preserve comments from the
source file.

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
    color: never        # never|auto|always; ANSI output is presentation-only
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

Set the per-logger levels back to `INFO` or enable the boolean toggles when you want the firehose during troubleshooting. Set `logs.runtime.color` to `auto` for local foreground runs, or `always` only when you deliberately want ANSI color preserved in journal output.

### 6) Listen

```
http://<your-ip>:8000/seasonalweather.ogg
http://<your-ip>:8000/seasonalweather.mp3
```

---

## Debug / test tool: inject

Generate a test alert WAV and optionally push it into the Liquidsoap queue:

```bash
/opt/seasonalweather/venv/bin/python -m seasonalweather.cli.inject --help
/opt/seasonalweather/venv/bin/python -m seasonalweather.cli.inject --event DMO --loc 024021 test-alert "hello world"
```

---

## SAME encoding

SeasonalWeather uses a custom Rust `samegen` binary for fast SAME encoding for alerts it generates with SAME tones when `native_encoder` is enabled and the binary is installed and present. 

The `samegen` binary is located in `tools/samegen` from the repo root, and is currently not provisioned and installed by the bootstrapper. 

Configuration flags:

- `enabled:` — use `samegen` when available, otherwise, fall back to the Python encoder.
- `bin:` — What custom binary to use.
- `timeout_seconds:` — self explanatory time out for waiting for the Rust encoder to finish.
- `fallback_to_python:` — use Python native encoder if `samegen` binary is not present but enabled.

## GWES-ERN SAME decoding

SeasonalWeather uses the Rust `samedec` binary for fast SAME decoding from the ERN stream when `ern.decoder_backend` is `auto` and the configured binary exists. Fresh bootstrap installs a pinned `samedec` crate release into `/opt/seasonalweather/samedec` and publishes `/usr/local/bin/samedec`.

The `samedec` binary path, confidence threshold, and timing offset are configured in `config.yaml` under the `samedec:` section. The ERN decoder backend is configured with `ern.decoder_backend`:

- `auto` — use `samedec` when available, otherwise fall back to the native Python decoder.
- `samedec` — require the Rust decoder.
- `native` — force the pure-Python decoder.

Operational checks:

```bash
/usr/local/bin/samedec --version
/opt/seasonalweather/venv/bin/python -m seasonalweather.same.listen_samedec --help
```
