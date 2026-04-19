#!/usr/bin/env bash
# SeasonalWeather bootstrap script
# Run as root from inside a fresh clone of the repo:
#
#   git clone https://git.seasonalnet.org/Seasonal_Currency/SeasonalWeather /tmp/SeasonalWeather
#   cd /tmp/SeasonalWeather
#   sudo bash scripts/00-bootstrap.sh
#
# The script rsyncs the repo to /opt/seasonalweather/app/ automatically.
# Do NOT clone directly into /opt/seasonalweather/app/ — clone elsewhere and let
# this script handle the deploy.
#
# Optional environment variables:
#   SEASONAL_VOICETEXT_PAUL=1   — also install the VoiceText Paul (Wine) TTS backend
#   SEASONAL_DECTALK=1          — install DECtalk (builds from source; slow)
#   SEASONAL_DECTALK_UPDATE=1   — pull latest DECtalk source before rebuilding
#
set -euo pipefail

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Run as root: sudo bash $0"
  exit 1
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log()  { echo "[+] $*"; }
warn() { echo "[!] $*" >&2; }

export DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------------------
# Base OS packages
# -----------------------------------------------------------------------------------------
log "Installing OS packages"
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  build-essential \
  pkg-config \
  autoconf \
  automake \
  libasound2-dev \
  libpulse-dev \
  python3 python3-venv \
  ffmpeg \
  icecast2 \
  liquidsoap \
  espeak-ng \
  sox \
  rsync \
  tzdata \
  festival \
  festvox-kallpc16k \
  plocate \
  sudo

# -----------------------------------------------------------------------------------------
# System user + directories
# -----------------------------------------------------------------------------------------
log "Creating user + directories"
if ! id -u seasonalweather >/dev/null 2>&1; then
  useradd --system --home /var/lib/seasonalweather --shell /usr/sbin/nologin seasonalweather
fi

install -d -o seasonalweather -g seasonalweather /var/lib/seasonalweather
install -d -o seasonalweather -g seasonalweather /var/lib/seasonalweather/audio
install -d -o seasonalweather -g seasonalweather /var/lib/seasonalweather/cache
install -d -o seasonalweather -g seasonalweather /var/log/seasonalweather
install -d -o root            -g root            /etc/seasonalweather

# -----------------------------------------------------------------------------------------
# Deploy repo to /opt/seasonalweather/app via rsync
# -----------------------------------------------------------------------------------------
log "Syncing app to /opt/seasonalweather/app"
install -d -o root -g root /opt/seasonalweather
install -d -o root -g root /opt/seasonalweather/app
rsync -a --delete --exclude ".git" "${SRC_DIR}/" /opt/seasonalweather/app/

# -----------------------------------------------------------------------------------------
# Python venv
# -----------------------------------------------------------------------------------------
log "Creating Python venv"
if [[ ! -d /opt/seasonalweather/venv ]]; then
  python3 -m venv /opt/seasonalweather/venv
fi
/opt/seasonalweather/venv/bin/python -m pip install --upgrade pip wheel
/opt/seasonalweather/venv/bin/pip install -r /opt/seasonalweather/app/requirements.txt

# -----------------------------------------------------------------------------------------
# Config files — never overwrite existing operator-edited files
# -----------------------------------------------------------------------------------------
log "Installing default config (won't overwrite existing)"

if [[ ! -f /etc/seasonalweather/config.yaml ]]; then
  cp /opt/seasonalweather/app/config/config.yaml /etc/seasonalweather/config.yaml
  warn "Installed default config.yaml — edit it before starting the service!"
  warn "  sudo nano /etc/seasonalweather/config.yaml"
  warn "  NOTE: This is the LIVE config. Do NOT edit the repo copy at"
  warn "        /opt/seasonalweather/app/config/config.yaml — the service does not read it."
else
  log "config.yaml already exists — not overwriting"
  log "  REMINDER: Live config is /etc/seasonalweather/config.yaml"
  log "            /opt/seasonalweather/app/config/config.yaml is the repo template; editing it does nothing."
fi

if [[ ! -f /etc/seasonalweather/radio.liq ]]; then
  cp /opt/seasonalweather/app/liquidsoap/radio.liq /etc/seasonalweather/radio.liq
fi

if [[ ! -f /etc/seasonalweather/seasonalweather.env ]]; then
  cp /opt/seasonalweather/app/config/example.env /etc/seasonalweather/seasonalweather.env
  chmod 600 /etc/seasonalweather/seasonalweather.env
  warn "Installed example seasonalweather.env — fill in credentials before starting!"
  warn "  sudo nano /etc/seasonalweather/seasonalweather.env"
else
  log "seasonalweather.env already exists — not overwriting"
fi

# -----------------------------------------------------------------------------------------
# DECtalk (optional — set SEASONAL_DECTALK=1 to install)
#
#   Build from source: github.com/dectalk/dectalk (develop branch, autoconf/automake)
#   Requires: autoconf automake libasound2-dev libpulse-dev (installed above)
#
#   Built say binary:   /opt/dectalk/dectalk/src/dist/say
#   Installed to:       /opt/dectalk/dectalk/dist/say  (path wrapper scripts expect)
#
#   Usage: SEASONAL_DECTALK=1 sudo -E bash scripts/00-bootstrap.sh
# -----------------------------------------------------------------------------------------
if [[ "${SEASONAL_DECTALK:-0}" == "1" ]]; then
  DECTALK_BASE="/opt/dectalk"
  DECTALK_SRC="${DECTALK_BASE}/dectalk"
  DECTALK_SAY_BUILT="${DECTALK_SRC}/src/dist/say"  # where autoconf build outputs say
  DECTALK_DIST="${DECTALK_SRC}/dist"                # where wrapper scripts expect it
  DECTALK_SAY="${DECTALK_DIST}/say"

  log "Ensuring DECtalk is installed at ${DECTALK_SRC}"
  install -d -o root -g root "${DECTALK_BASE}"

  if [[ ! -d "${DECTALK_SRC}/.git" ]]; then
    log "Cloning DECtalk repo (develop branch)"
    rm -rf "${DECTALK_SRC}" || true
    git clone --depth=1 --branch develop https://github.com/dectalk/dectalk "${DECTALK_SRC}"
  else
    if [[ "${SEASONAL_DECTALK_UPDATE:-0}" == "1" ]]; then
      log "Updating DECtalk repo (SEASONAL_DECTALK_UPDATE=1)"
      git -C "${DECTALK_SRC}" pull --ff-only
    else
      log "DECtalk repo already present (not updating; set SEASONAL_DECTALK_UPDATE=1 to pull)"
    fi
  fi

  if [[ ! -x "${DECTALK_SAY}" ]]; then
    log "Building DECtalk from src/ (autogen + configure + make)"
    if [[ ! -f "${DECTALK_SRC}/src/autogen.sh" ]]; then
      warn "DECtalk src/autogen.sh not found — repo may be on wrong branch or incomplete."
      warn "Expected develop branch: github.com/dectalk/dectalk (branch: develop)"
      warn "Skipping DECtalk build."
    else
      (
        cd "${DECTALK_SRC}/src"
        ./autogen.sh
        ./configure
        make -j"$(nproc)"
      )

      if [[ -x "${DECTALK_SAY_BUILT}" ]]; then
        install -d -o root -g root "${DECTALK_DIST}"
        install -m 755 "${DECTALK_SAY_BUILT}" "${DECTALK_SAY}"
        if [[ -d "${DECTALK_SRC}/src/dist/lib" ]]; then
          cp -a "${DECTALK_SRC}/src/dist/lib" "${DECTALK_DIST}/"
        fi
        log "DECtalk built and installed to ${DECTALK_DIST}"
      else
        warn "Build ran but ${DECTALK_SAY_BUILT} not found — check build output above."
        warn "DECtalk backend will not function."
      fi
    fi
  else
    log "DECtalk already built (${DECTALK_SAY} exists)"
  fi

  log "Ensuring DECtalk wrapper scripts exist"
  if [[ -f "/opt/seasonalweather/app/scripts/dectalk-env" ]]; then
    install -m 755 /opt/seasonalweather/app/scripts/dectalk-env /usr/local/bin/dectalk-env
  elif [[ ! -x /usr/local/bin/dectalk-env ]]; then
    cat > /usr/local/bin/dectalk-env <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
DECTALK_DIST="${DECTALK_DIST:-/opt/dectalk/dectalk/dist}"
export DECTALK_DIST
export LD_LIBRARY_PATH="${DECTALK_DIST}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${DECTALK_DIST}:${PATH}"
exec "$@"
EOF
    chmod 755 /usr/local/bin/dectalk-env
  fi

  if [[ -f "/opt/seasonalweather/app/scripts/dectalk-text2wav" ]]; then
    install -m 755 /opt/seasonalweather/app/scripts/dectalk-text2wav /usr/local/bin/dectalk-text2wav
  elif [[ ! -x /usr/local/bin/dectalk-text2wav ]]; then
    cat > /usr/local/bin/dectalk-text2wav <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

OUT=""
VOICE="${DECTALK_VOICE:-9}"
RATE="${DECTALK_RATE_WPM:-165}"
TEXT_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out) OUT="${2:-}"; shift 2;;
    -v|--voice) VOICE="${2:-}"; shift 2;;
    -r|--rate) RATE="${2:-}"; shift 2;;
    --) shift; TEXT_ARGS+=("$@"); break;;
    *) TEXT_ARGS+=("$1"); shift;;
  esac
done

if [[ -z "${OUT}" && ${#TEXT_ARGS[@]} -ge 2 ]]; then
  if [[ "${TEXT_ARGS[0]}" == *.wav ]]; then
    OUT="${TEXT_ARGS[0]}"; TEXT_ARGS=("${TEXT_ARGS[@]:1}")
  elif [[ "${TEXT_ARGS[-1]}" == *.wav ]]; then
    OUT="${TEXT_ARGS[-1]}"; unset 'TEXT_ARGS[-1]'
  fi
fi

if [[ -z "${OUT}" ]]; then
  echo "ERR: output wav path not provided. Use --out /path/file.wav" >&2; exit 2
fi

TEXT="${TEXT_ARGS[*]:-}"
if [[ -z "${TEXT}" ]]; then TEXT="$(cat)"; fi

SAY_BIN="${DECTALK_SAY_BIN:-/opt/dectalk/dectalk/dist/say}"
if [[ ! -x "${SAY_BIN}" ]]; then SAY_BIN="$(command -v say || true)"; fi
if [[ -z "${SAY_BIN}" || ! -x "${SAY_BIN}" ]]; then
  echo "ERR: DECtalk say binary not found." >&2; exit 3
fi

HELP="$("${SAY_BIN}" --help 2>&1 || true)"

OUTFLAG=""
if grep -qE '(^|[[:space:]])-fo([[:space:]]|$)' <<<"${HELP}"; then OUTFLAG="-fo"
elif grep -qE '(^|[[:space:]])-o([[:space:]]|$)' <<<"${HELP}"; then OUTFLAG="-o"
elif grep -qiE '(^|[[:space:]])--output([[:space:]]|$)' <<<"${HELP}"; then OUTFLAG="--output"
fi
if [[ -z "${OUTFLAG}" ]]; then echo "ERR: couldn't detect file-output flag." >&2; exit 4; fi

VOICE_ARGS=()
if grep -qE '(^|[[:space:]])-v([[:space:]]|$)' <<<"${HELP}"; then VOICE_ARGS=(-v "${VOICE}")
elif grep -qiE '(^|[[:space:]])--voice([[:space:]]|$)' <<<"${HELP}"; then VOICE_ARGS=(--voice "${VOICE}")
fi

RATE_ARGS=()
if grep -qE '(^|[[:space:]])-r([[:space:]]|$)' <<<"${HELP}"; then RATE_ARGS=(-r "${RATE}")
elif grep -qiE '(^|[[:space:]])--rate([[:space:]]|$)' <<<"${HELP}"; then RATE_ARGS=(--rate "${RATE}")
fi

/usr/local/bin/dectalk-env "${SAY_BIN}" "${VOICE_ARGS[@]}" "${RATE_ARGS[@]}" "${OUTFLAG}" "${OUT}" "${TEXT}"
EOF
    chmod 755 /usr/local/bin/dectalk-text2wav
  fi
else
  log "Skipping DECtalk (set SEASONAL_DECTALK=1 to install)"
fi

# -----------------------------------------------------------------------------------------
# VoiceText Paul (optional — set SEASONAL_VOICETEXT_PAUL=1 to install)
#
#   Requires Wine. Downloads WeatherRadioSuite-LIB.zip from cdn.dondaplayer.com.
#   Source project: github.com/dondaplayer1/weather-radio-suite (open source)
#
#   After install, set  tts.backend: "voicetext_paul"  in /etc/seasonalweather/config.yaml.
#
#   Usage: SEASONAL_VOICETEXT_PAUL=1 sudo -E bash scripts/00-bootstrap.sh
# -----------------------------------------------------------------------------------------
if [[ "${SEASONAL_VOICETEXT_PAUL:-0}" == "1" ]]; then
  log "Installing VoiceText Paul backend"

  apt-get install -y --no-install-recommends wine wine64 unzip

  VTP_USER="voicetext"
  VTP_HOME="/home/${VTP_USER}"
  VTP_BASE="/home/seasonalweather-data/var-lib-seasonalweather/voices/voicetext_paul"
  VTP_ENGINE_DIR="${VTP_BASE}/WeatherRadioSuite-LIB"
  VTP_BIN_DIR="${VTP_ENGINE_DIR}/binary"
  VTP_EXE="${VTP_BIN_DIR}/voicetext_paul.exe"
  VTP_CDN="https://cdn.dondaplayer.com/WeatherRadioSuite-LIB.zip"
  VTP_SYNTH="/usr/local/bin/voicetext_paul_synth"
  VTP_WSKILL="/usr/local/bin/voicetext_paul_wineserver_kill"
  VTP_SUDOERS="/etc/sudoers.d/seasonalweather-voicetext-paul"

  if ! id -u "${VTP_USER}" >/dev/null 2>&1; then
    useradd --system --home "${VTP_HOME}" --create-home --shell /usr/sbin/nologin "${VTP_USER}"
  fi
  install -d -o "${VTP_USER}" -g "${VTP_USER}" "${VTP_HOME}"

  install -d -o root          -g root          "$(dirname "${VTP_BASE}")"
  install -d -o root          -g root          "${VTP_BASE}"
  install -d -o "${VTP_USER}" -g "${VTP_USER}" "${VTP_ENGINE_DIR}"
  install -d -o "${VTP_USER}" -g "${VTP_USER}" "${VTP_BIN_DIR}"

  if [[ ! -f "${VTP_EXE}" ]]; then
    log "Downloading WeatherRadioSuite-LIB.zip"
    TMP_ZIP="$(mktemp /tmp/WeatherRadioSuite-LIB.XXXXXX.zip)"
    curl -fsSL --max-time 120 "${VTP_CDN}" -o "${TMP_ZIP}"
    unzip -o "${TMP_ZIP}" -d "${VTP_ENGINE_DIR}"
    rm -f "${TMP_ZIP}"
    chown -R "${VTP_USER}:${VTP_USER}" "${VTP_ENGINE_DIR}"
    log "VoiceText Paul binary extracted to ${VTP_BIN_DIR}"
  else
    log "VoiceText Paul binary already present — skipping download"
  fi

  # Synth wrapper — called by SeasonalWeather via: sudo -n -u voicetext /usr/local/bin/voicetext_paul_synth
  # Text arrives on stdin; the exe writes output.wav into its working directory (engine bin dir).
  if [[ ! -x "${VTP_SYNTH}" ]]; then
    cat > "${VTP_SYNTH}" <<VTSEOF
#!/usr/bin/env bash
# voicetext_paul_synth
# Run as the voicetext user (via sudo -n -u voicetext).
# Reads text from stdin. The exe writes output.wav to its working directory.
set -euo pipefail
ENGINE_DIR="${VTP_BIN_DIR}"
EXE="\${ENGINE_DIR}/voicetext_paul.exe"
if [[ ! -f "\${EXE}" ]]; then
  echo "voicetext_paul_synth: exe not found at \${EXE}" >&2
  exit 1
fi
export WINEPREFIX="\${HOME}/.wine"
export WINEDEBUG="-all"
export DISPLAY=""
cd "\${ENGINE_DIR}"
exec wine "\${EXE}"
VTSEOF
    chmod 755 "${VTP_SYNTH}"
  fi

  if [[ ! -x "${VTP_WSKILL}" ]]; then
    cat > "${VTP_WSKILL}" <<WSKEOF
#!/usr/bin/env bash
# voicetext_paul_wineserver_kill
# Run as the voicetext user (via sudo -n -u voicetext).
set -euo pipefail
export WINEPREFIX="\${HOME}/.wine"
export WINEDEBUG="-all"
wineserver -k 2>/dev/null || true
WSKEOF
    chmod 755 "${VTP_WSKILL}"
  fi

  if [[ ! -f "${VTP_SUDOERS}" ]]; then
    cat > "${VTP_SUDOERS}" <<SUDOEOF
# SeasonalWeather: voicetext_paul backend
# Allows the seasonalweather service user to invoke the Wine TTS wrappers as the voicetext user.
seasonalweather ALL=(${VTP_USER}) NOPASSWD: ${VTP_SYNTH}
seasonalweather ALL=(${VTP_USER}) NOPASSWD: ${VTP_WSKILL}
SUDOEOF
    chmod 440 "${VTP_SUDOERS}"
    if ! visudo -cf "${VTP_SUDOERS}" >/dev/null 2>&1; then
      warn "sudoers syntax check failed — removing ${VTP_SUDOERS}"
      rm -f "${VTP_SUDOERS}"
    else
      log "sudoers entry installed at ${VTP_SUDOERS}"
    fi
  else
    log "sudoers entry already exists — not overwriting"
  fi

  log "VoiceText Paul install complete"
  log "  Activate: set  tts.backend: \"voicetext_paul\"  in /etc/seasonalweather/config.yaml"
else
  log "Skipping VoiceText Paul (set SEASONAL_VOICETEXT_PAUL=1 to install)"
fi

# -----------------------------------------------------------------------------------------
# Icecast config
# -----------------------------------------------------------------------------------------
log "Configuring Icecast"
if [[ -f /etc/default/icecast2 ]]; then
  sed -i 's/^ENABLE=.*/ENABLE=true/' /etc/default/icecast2 || true
  grep -q '^ENABLE=true' /etc/default/icecast2 || echo 'ENABLE=true' >> /etc/default/icecast2
fi

ICECAST_SOURCE_PASSWORD="$(grep -E '^ICECAST_SOURCE_PASSWORD=' /etc/seasonalweather/seasonalweather.env 2>/dev/null | head -n1 | cut -d= -f2- || true)"
ICECAST_SOURCE_PASSWORD="${ICECAST_SOURCE_PASSWORD:-seasonal-source}"
ICECAST_ADMIN_PASSWORD="seasonal-admin"
ICECAST_RELAY_PASSWORD="seasonal-relay"

if [[ -f /etc/icecast2/icecast.xml ]]; then
  sed -i "s#<source-password>.*</source-password>#<source-password>${ICECAST_SOURCE_PASSWORD}</source-password>#g" /etc/icecast2/icecast.xml || true
  sed -i "s#<admin-password>.*</admin-password>#<admin-password>${ICECAST_ADMIN_PASSWORD}</admin-password>#g" /etc/icecast2/icecast.xml || true
  sed -i "s#<relay-password>.*</relay-password>#<relay-password>${ICECAST_RELAY_PASSWORD}</relay-password>#g" /etc/icecast2/icecast.xml || true
fi

if [[ -f /etc/seasonalweather/radio.liq ]]; then
  sed -i "s#password=\"[^\"]*\"#password=\"${ICECAST_SOURCE_PASSWORD}\"#g" /etc/seasonalweather/radio.liq || true
fi

# -----------------------------------------------------------------------------------------
# systemd
# -----------------------------------------------------------------------------------------
log "Installing systemd units"
cp /opt/seasonalweather/app/systemd/seasonalweather.service /etc/systemd/system/seasonalweather.service
cp /opt/seasonalweather/app/systemd/seasonalweather-liquidsoap.service /etc/systemd/system/seasonalweather-liquidsoap.service
systemctl daemon-reload

log "Installing helper scripts (if present in repo)"
for f in seasonalweather-preflight.sh seasonalweather-audio-prune.sh seasonalweather-prune-audio.sh; do
  if [[ -f "/opt/seasonalweather/app/scripts/${f}" ]]; then
    install -m 755 "/opt/seasonalweather/app/scripts/${f}" "/usr/local/sbin/${f}"
  fi
done
for f in seasonalweather-inject; do
  if [[ -f "/opt/seasonalweather/app/scripts/${f}" ]]; then
    install -m 755 "/opt/seasonalweather/app/scripts/${f}" "/usr/local/bin/${f}"
  fi
done

# -----------------------------------------------------------------------------------------
# Permissions
# -----------------------------------------------------------------------------------------
log "Permissions"
chown -R seasonalweather:seasonalweather /var/lib/seasonalweather /var/log/seasonalweather
chmod 755 /var/lib/seasonalweather /var/lib/seasonalweather/audio /var/lib/seasonalweather/cache /var/log/seasonalweather

echo
echo "======================================================="
echo " SeasonalWeather bootstrap complete"
echo "======================================================="
echo
echo " IMPORTANT — two config files exist; they are NOT the same:"
echo
echo "   LIVE (service reads this):  /etc/seasonalweather/config.yaml"
echo "   REPO TEMPLATE (ignored):    /opt/seasonalweather/app/config/config.yaml"
echo
echo "   Always edit the LIVE config. Editing the repo template has no effect"
echo "   on the running service."
echo
echo " Configuration:"
echo "   Behaviour  →  sudo nano /etc/seasonalweather/config.yaml"
echo "   Secrets    →  sudo nano /etc/seasonalweather/seasonalweather.env"
echo
echo " Next steps:"
echo "   1) Set your service area, TTS backend (default: espeak-ng),"
echo "      NWWS office, and other behaviour knobs in the LIVE config:"
echo "      sudo nano /etc/seasonalweather/config.yaml"
echo
echo "   2) Fill in credentials (NWWS_JID, NWWS_PASSWORD, ICECAST_SOURCE_PASSWORD):"
echo "      sudo nano /etc/seasonalweather/seasonalweather.env"
echo
echo "   3) Enable and start services:"
echo "      sudo systemctl enable --now icecast2"
echo "      sudo systemctl enable --now seasonalweather-liquidsoap"
echo "      sudo systemctl enable --now seasonalweather"
echo
echo "   4) Check logs:"
echo "      journalctl -u seasonalweather -f"
echo
echo " Listen:"
echo "   http://<your-ip>:8000/seasonalweather.ogg"
echo
echo " Optional backends (re-run bootstrapper with env var to install):"
echo "   DECtalk:        SEASONAL_DECTALK=1 sudo -E bash scripts/00-bootstrap.sh"
echo "   VoiceText Paul: SEASONAL_VOICETEXT_PAUL=1 sudo -E bash scripts/00-bootstrap.sh"
echo "======================================================="
