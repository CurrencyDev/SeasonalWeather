# SeasonalWeather (VM build) — KLWX / LWX (WXM42, WXM43, KHB36, KEC83)

SeasonalWeather is an **unofficial** automated IP weather broadcast:
- Listens to **NWWS-OI** (XMPP) for NWS products from **KLWX**
- Fetches supporting context from **api.weather.gov**
- Produces audio (TTS) and streams it via **Liquidsoap → Icecast**

**Important:** This is *not* NOAA Weather Radio and should not be treated as an emergency alerting system.

## What you get (v0.9)
- Two-layer playout (like a mini NWR):
  - **cycle** queue: routine info loop
  - **alert** queue: interrupts with an attention tone + spoken alert + EOM beep
- A tiny state machine:
  - NORMAL cycle (default 5 min)
  - HEIGHTENED cycle (shorter loop for a bit after severe products)
- Service area filter: union of SAME/FIPS codes for LWX transmitters:
  - **KEC-83** Baltimore
  - **KHB-36** Manassas
  - **WXM-42** Hagerstown
  - **WXM-43** Frostburg

## Quick start (Debian 12 / Ubuntu 24.04)
1) Copy this repo to the VM (or scp the zip contents)
2) Run bootstrap:
   ```bash
   sudo bash scripts/00-bootstrap.sh
   ```
3) Edit env:
   ```bash
   sudo nano /etc/seasonalweather/seasonalweather.env
   ```
   Set:
   - `NWWS_JID`
   - `NWWS_PASSWORD`

4) Start services:
   ```bash
   sudo systemctl enable --now icecast2
   sudo systemctl enable --now seasonalweather-liquidsoap
   sudo systemctl enable --now seasonalweather
   ```

5) Listen:
   - `http://<vm-ip>:8000/seasonalweather.ogg`

## Logs
- Orchestrator: `journalctl -u seasonalweather -f`
- Liquidsoap: `/var/log/seasonalweather/liquidsoap.log`
- Icecast: `journalctl -u icecast2 -f`

## Config
- `/etc/seasonalweather/config.yaml` (copied from `config/config.yaml`)
- `/etc/seasonalweather/radio.liq` (Liquidsoap)

## Dev notes
- TTS backend default: `espeak-ng` → normalized to 48 kHz stereo WAVs for safe concatenation.
- The NWWS parser is intentionally conservative; it keys off the AWIPS ID (e.g., `SVRLWX`) and WFO (e.g., `KLWX`).
