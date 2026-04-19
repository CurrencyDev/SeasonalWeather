# ISSUES.md — SeasonalWeather Known Issues

Known bugs, deployment gaps, and their suspected root causes.
Intended to reduce support overhead and give contributors or AI agents a clear starting point.

---

## ISSUE-001 — Service ignores config edits; stuck on `voicetext_paul`

**Severity:** High — service fails to start on every fresh install with default config  
**Component:** `config/config.yaml`, `scripts/00-bootstrap.sh`

### Symptom

```
RuntimeError: voicetext_paul backend selected but
/home/seasonalweather-data/.../voicetext_paul.exe not found
```

Appears even after editing `config.yaml` to `backend: "espeak-ng"` and restarting the service.

### Root Cause

Two separate copies of `config.yaml` exist:

| Path | Role |
|------|------|
| `/opt/seasonalweather/app/config/config.yaml` | Repo template — never read by the running service |
| `/etc/seasonalweather/config.yaml` | Live runtime config — the only file the service reads |

The bootstrapper copies the repo default to `/etc/` only on first install (idempotent; will not overwrite an existing file). The repo default ships with `backend: "voicetext_paul"`, so the live config starts life with that value.

Users editing the repo copy (e.g. via VS Code remote to `/opt/seasonalweather/app/config/config.yaml`) change nothing at runtime. The VS Code breadcrumb path `opt › seasonalweather › app › config › config.yaml` makes this distinction non-obvious.

### Fix Applied

`config/config.yaml` default changed from `voicetext_paul` to `espeak-ng`. New installs now get a working `/etc/seasonalweather/config.yaml` without requiring Wine or any proprietary binary.

### Workaround (existing installs already bootstrapped)

Edit the **live** config, not the repo copy:

```bash
sudo nano /etc/seasonalweather/config.yaml
# Change:  backend: "voicetext_paul"
# To:      backend: "espeak-ng"
sudo systemctl restart seasonalweather
```

---

## ISSUE-002 — DECTalk bootstrapper build fails: wrong build system, missing packages

**Severity:** High — `scripts/00-bootstrap.sh` aborts with `make: *** No targets specified and no makefile found. Stop.`  
**Component:** `scripts/00-bootstrap.sh`

### Symptom

```
make: *** No targets specified and no makefile found.  Stop.
```

### Root Cause

The `dectalk/dectalk` repo (`develop` branch) uses **autoconf/automake** — there is no `Makefile` at the repo root. The bootstrapper was running `make -j$(nproc)` in `${DECTALK_SRC}` (the repo root), which fails immediately.

Correct build sequence:

```bash
cd "${DECTALK_SRC}/src"
./autogen.sh
./configure
make -j$(nproc)
```

Additionally, the bootstrapper did not install the required build-time packages (`autoconf`, `automake`, `libasound2-dev`, `libpulse-dev`), and the `DECTALK_SAY` sentinel path was wrong (`dist/say` at the repo root rather than `src/dist/say` where the build actually produces output).

### Fix Applied

Bootstrapper updated to:
- Install `autoconf automake libasound2-dev libpulse-dev`
- Build from `src/` via `autogen.sh` + `configure` + `make`
- Detect the built `say` binary and install it to `/opt/dectalk/dectalk/dist/` (the path the wrapper scripts expect)

---

## ISSUE-003 — VoiceText Paul backend: no install procedure in bootstrapper

**Severity:** Medium — selecting `voicetext_paul` in config without manual setup is a guaranteed crash  
**Component:** `scripts/00-bootstrap.sh`, `seasonalweather/tts/tts.py`

### Symptom

```
RuntimeError: voicetext_paul backend selected but
/home/seasonalweather-data/.../voicetext_paul.exe not found

RuntimeError: voicetext_paul backend selected but wrapper
/usr/local/bin/voicetext_paul_synth not found
```

### Root Cause

The VoiceText Paul backend requires all of the following, none of which existed in the bootstrapper:

1. `wine` installed
2. A dedicated `voicetext` system user (low-privilege Wine runner)
3. Binary archive downloaded from `https://cdn.dondaplayer.com/WeatherRadioSuite-LIB.zip`
4. Archive extracted to `/home/seasonalweather-data/var-lib-seasonalweather/voices/voicetext_paul/WeatherRadioSuite-LIB/binary/`
5. `/usr/local/bin/voicetext_paul_synth` — wrapper that runs the `.exe` under Wine as the `voicetext` user
6. `/usr/local/bin/voicetext_paul_wineserver_kill` — wineserver teardown wrapper
7. Sudoers entry granting `seasonalweather` passwordless sudo to both wrappers as the `voicetext` user

The underlying `.exe` is from the open-source `dondaplayer1/weather-radio-suite` project
(`https://cdn.dondaplayer.com/WeatherRadioSuite-LIB.zip`).

### Fix Applied

Bootstrapper now includes a VoiceText Paul install block gated on the `SEASONAL_VOICETEXT_PAUL=1`
environment variable:

```bash
SEASONAL_VOICETEXT_PAUL=1 sudo -E bash scripts/00-bootstrap.sh
```

This step installs Wine, creates the system user, downloads and extracts the binary, writes the
wrapper scripts, and configures sudoers. It is idempotent and safe to re-run.

After install, set `backend: "voicetext_paul"` in `/etc/seasonalweather/config.yaml` and restart
the service.

---

## ISSUE-004 — Bootstrapper deployment workflow is underdocumented

**Severity:** Low — causes first-time deployer confusion; not a runtime failure  
**Component:** `scripts/00-bootstrap.sh`, `README.md`

### Symptom

Users clone the repo into `/opt/seasonalweather/app/` directly, then manually copy files up one
level and delete the cloned directory. Or they run the bootstrapper before cloning, and it fails
because `SRC_DIR` is undefined relative to the script location.

### Root Cause

The bootstrapper derives `SRC_DIR` from `BASH_SOURCE[0]` and rsyncs the repo to
`/opt/seasonalweather/app/` automatically. This means:

- Clone to **any** temporary location first
- Run `sudo bash scripts/00-bootstrap.sh` from inside the clone
- The bootstrapper handles deploying to `/opt/` via rsync — no manual copying needed

This workflow is not stated clearly in `README.md` or the bootstrapper header comment.

### Correct Deployment Procedure

```bash
# 1. Clone to a staging location (NOT directly into /opt/seasonalweather/app)
git clone https://git.seasonalnet.org/Seasonal_Currency/SeasonalWeather /tmp/SeasonalWeather
cd /tmp/SeasonalWeather

# 2. Bootstrapper rsyncs to /opt/seasonalweather/app/ automatically
sudo bash scripts/00-bootstrap.sh

# Optional: VoiceText Paul backend
# SEASONAL_VOICETEXT_PAUL=1 sudo -E bash scripts/00-bootstrap.sh

# 3. Edit the LIVE config (not the repo copy in /opt/seasonalweather/app/config/)
sudo nano /etc/seasonalweather/config.yaml

# 4. Fill in credentials
sudo nano /etc/seasonalweather/seasonalweather.env

# 5. Enable and start services
sudo systemctl enable --now icecast2
sudo systemctl enable --now seasonalweather-liquidsoap
sudo systemctl enable --now seasonalweather

# 6. Verify
journalctl -u seasonalweather -f
```
