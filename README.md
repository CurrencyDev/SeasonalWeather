# SeasonalWeather

SeasonalWeather is a Python-based, internet-delivered weather/alert radio stream inspired by NOAA Weather Radio (NWR) broadcast workflows: continuous cycle audio plus interrupting alert cut-ins (SAME header bursts, 1050 Hz, spoken content, EOM).

It’s designed for homelab / hobby IP based stream use with a focus on resiliency, dedupe, and “don’t spam the entire service area by accident.”

> **Not affiliated with NOAA / NWS / FEMA.**
> This is an **unofficial** hobby project. **Do not** rely on it for life safety.
> Always use official sources for real warnings (e.g., NOAA Weather Radio, NWS/NOAA alerts, local authorities).

> **Please note, SeasonalWeather is a project coded with the use of generative AI, with models like ChatGPT.**
> Keep this in mind if you're against certain projects coded with the use of such tools.

## Safety, scope, and acceptable use
SeasonalWeather can generate **valid SAME headers** and **1050 Hz attention tones**. That power comes with responsibility.

- **Intended use:** homelab / internal / IP-based streaming and testing.
- **Not intended for:** over-the-air broadcasting, public alert origination, or any use that could be confused with an official EAS/NWR source.
- **You are responsible** for compliance with your local laws/regulations and for preventing misuse.

## Support and warranty
- Provided **as-is**, with **no warranty** of any kind.
- Support is **best-effort** and may be limited depending on available time.

---

## What it does (current behavior)

### Sources / ingestion
- **NWWS-OI (XMPP)** ingest (room monitoring) for alert payloads.
- **NWS API** ingest for supporting products (forecasts, observations, text products).
- **CAP ingest** via the NWS alerts API (active alerts polling).

### Alert behavior
- Full cut-ins (when enabled) typically follow:
  1) **SAME header burst(s)** targeted to the alert’s affected SAME/FIPS
  2) **1050 Hz tone**
  3) **TTS narration**
  4) **SAME EOM**
  5) return to normal cycle

- Low-severity CAP can optionally do **voice-only interruptions** (no SAME/1050/EOM) with cooldowns.

### Dedupe & safety gates
- Cross-source **de-duping** to avoid immediately repeating the same alert via multiple feeds (e.g., NWWS then CAP).
- Ledger-based tracking for CAP so previously processed alerts aren’t re-issued repeatedly.
- “No global fallback” targeting discipline: alerts should not accidentally target *every* county in the service area.

### Output
- Audio output is produced for **Liquidsoap**, which feeds **Icecast**.
- Typical mount: `/seasonalweather.ogg` (your setup may vary).

### Hardening / ops
- Designed to run under **systemd** (units included in `systemd/`).
- Disk usage is managed via pruning/retention scripts in production deployments (see notes below).

---

## Repo layout

- `seasonalweather/` — the Python application (orchestrator, decoders, ingestors)
- `config/`
  - `config.yaml` — service area + behavior config (SAME/FIPS lists live here)
  - `example.env` — environment variable template (copy this for your deployment)
- `liquidsoap/` — Liquidsoap script(s)
- `systemd/` — service unit files
- `scripts/` — bootstrap + helper scripts
- `docs/` — design notes / state machine docs

---

## Quick start (dev / local)

### 1) System deps
You’ll need at least:
- Python 3.11+ (3.10 may work depending on requirements)
- `ffmpeg`
- (If using Liquidsoap/Icecast locally) `liquidsoap`, `icecast2`

### 2) Python env
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Debug / test tool: inject

This repo ships an inject helper used to generate a test alert WAV (and optionally push it into the Liquidsoap alert queue).

Run from the repo root using the SeasonalWeather venv Python:

```bash
/opt/seasonalweather/venv/bin/python -m seasonalweather.cli.inject --help
/opt/seasonalweather/venv/bin/python -m seasonalweather.cli.inject --event DMO --loc 024021 test-alert "hello world"
```
