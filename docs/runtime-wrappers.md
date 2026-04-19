# Runtime wrappers

SeasonalWeather installs a small set of helper wrappers into `/usr/local/bin` during bootstrap.
These wrappers are part of the deployed runtime contract and are version-controlled in
`scripts/wrappers/` within this repo.

## Source of truth

- Repo source: `scripts/wrappers/`
- Installed runtime path: `/usr/local/bin/`
- Installer: `scripts/00-bootstrap.sh`

Do not hand-edit the installed copies on the target host and assume the repo now matches.
If a wrapper needs to change, change the repo copy first and re-run bootstrap.

## Installed wrappers

### DECtalk

- `/usr/local/bin/dectalk-env`
- `/usr/local/bin/dectalk-text2wav`

Bootstrap installs these when `SEASONAL_DECTALK=1` is set.

The canonical DECtalk `say` path used by this repo is:

```text
/opt/dectalk/dectalk/dist/say
```

`dectalk-env` sets the DECtalk runtime library path and then execs the requested command.
`dectalk-text2wav` is the stable wrapper interface for converting text to a WAV file.

Relevant environment variables:

- `DECTALK_DIST`
- `DECTALK_SAY_BIN`
- `DECTALK_VOICE`
- `DECTALK_RATE_WPM`

### VoiceText Paul

- `/usr/local/bin/voicetext_paul_synth`
- `/usr/local/bin/voicetext_paul_wineserver_kill`

Bootstrap installs these when `SEASONAL_VOICETEXT_PAUL=1` is set.

The synth wrapper is invoked by SeasonalWeather via `sudo -n -u <run_as>` and is expected to:

- read input text from stdin
- write `output.wav` in the VoiceText Paul engine binary directory
- serialize access with a lock so concurrent synth jobs do not clobber `input1.txt` and `output.wav`
- preserve useful failure artifacts when Wine crashes or times out

The wineserver-kill wrapper takes the same lock before calling `wineserver -k` so it does not kill Wine mid-synthesis.

Relevant environment variables:

- `SEASONALWEATHER_DATA_BASE`
- `VOICETEXT_PAUL_ENGINE_ROOT`
- `VOICETEXT_PAUL_BIN_DIR`
- `VOICETEXT_PAUL_PREFIX_BASE`
- `VOICETEXT_PAUL_PREFIX_NAME`
- `VOICETEXT_PAUL_WINEPREFIX`
- `VOICETEXT_PAUL_TMPDIR`
- `VOICETEXT_PAUL_WINEDEBUG`
- `VOICETEXT_PAUL_WINEESYNC`
- `VOICETEXT_PAUL_WINEFSYNC`
- `VOICETEXT_PAUL_DISABLE_WRITE_WATCH`
- `VOICETEXT_PAUL_LOCK_WAIT`
- `VOICETEXT_PAUL_CORE`
- `VOICETEXT_DEBUG`
- `VOICETEXT_VTML_PARA_PAUSE_MS`

## Deployment workflow

SeasonalWeather's bootstrapper expects to run from a clone in a staging location, then rsyncs the repo into `/opt/seasonalweather/app`.

Example:

```bash
git clone https://git.seasonalnet.org/Seasonal_Currency/SeasonalWeather /tmp/SeasonalWeather
cd /tmp/SeasonalWeather
sudo bash scripts/00-bootstrap.sh
```

Optional backend installs:

```bash
SEASONAL_DECTALK=1 sudo -E bash scripts/00-bootstrap.sh
SEASONAL_VOICETEXT_PAUL=1 sudo -E bash scripts/00-bootstrap.sh
```

The live runtime config is `/etc/seasonalweather/config.yaml`.
The repo copy at `/opt/seasonalweather/app/config/config.yaml` is only a template/example.
