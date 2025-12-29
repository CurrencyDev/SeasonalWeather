#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Run as root: sudo bash $0"
  exit 1
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { echo "[+] $*"; }
warn() { echo "[!] $*" >&2; }

export DEBIAN_FRONTEND=noninteractive

log "Installing OS packages"
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates \
  git \
  build-essential \
  pkg-config \
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
  plocate

log "Creating user + directories"
if ! id -u seasonalweather >/dev/null 2>&1; then
  useradd --system --home /var/lib/seasonalweather --shell /usr/sbin/nologin seasonalweather
fi

install -d -o seasonalweather -g seasonalweather /var/lib/seasonalweather
install -d -o seasonalweather -g seasonalweather /var/lib/seasonalweather/audio
install -d -o seasonalweather -g seasonalweather /var/lib/seasonalweather/cache
install -d -o seasonalweather -g seasonalweather /var/log/seasonalweather
install -d -o root -g root /etc/seasonalweather

log "Syncing app to /opt/seasonalweather/app"
install -d -o root -g root /opt/seasonalweather
install -d -o root -g root /opt/seasonalweather/app
rsync -a --delete --exclude ".git" "${SRC_DIR}/" /opt/seasonalweather/app/

log "Creating Python venv"
if [[ ! -d /opt/seasonalweather/venv ]]; then
  python3 -m venv /opt/seasonalweather/venv
fi
/opt/seasonalweather/venv/bin/python -m pip install --upgrade pip wheel
/opt/seasonalweather/venv/bin/pip install -r /opt/seasonalweather/app/requirements.txt

log "Installing default config (won't overwrite existing)"
if [[ ! -f /etc/seasonalweather/config.yaml ]]; then
  cp /opt/seasonalweather/app/config/config.yaml /etc/seasonalweather/config.yaml
fi
if [[ ! -f /etc/seasonalweather/radio.liq ]]; then
  cp /opt/seasonalweather/app/liquidsoap/radio.liq /etc/seasonalweather/radio.liq
fi
if [[ ! -f /etc/seasonalweather/seasonalweather.env ]]; then
  cp /opt/seasonalweather/app/config/example.env /etc/seasonalweather/seasonalweather.env
  chmod 600 /etc/seasonalweather/seasonalweather.env
  echo "!!! Edit /etc/seasonalweather/seasonalweather.env and set NWWS creds"
fi

# -----------------------------------------------------------------------------------------
# DECtalk (keep same paths as your current VM)
#   - source: /opt/dectalk/dectalk
#   - wrappers: /usr/local/bin/dectalk-env + /usr/local/bin/dectalk-text2wav
# -----------------------------------------------------------------------------------------
DECTALK_BASE="/opt/dectalk"
DECTALK_SRC="${DECTALK_BASE}/dectalk"
DECTALK_SAY="${DECTALK_SRC}/dist/say"

log "Ensuring DECtalk is installed at ${DECTALK_SRC}"
install -d -o root -g root "${DECTALK_BASE}"

if [[ ! -d "${DECTALK_SRC}/.git" ]]; then
  log "Cloning DECtalk repo"
  rm -rf "${DECTALK_SRC}" || true
  git clone --depth=1 https://github.com/dectalk/dectalk "${DECTALK_SRC}"
else
  if [[ "${SEASONAL_DECTALK_UPDATE:-0}" == "1" ]]; then
    log "Updating DECtalk repo (SEASONAL_DECTALK_UPDATE=1)"
    git -C "${DECTALK_SRC}" pull --ff-only
  else
    log "DECtalk repo already present (not updating; set SEASONAL_DECTALK_UPDATE=1 to pull)"
  fi
fi

if [[ ! -x "${DECTALK_SAY}" ]]; then
  log "Building DECtalk (dist/say missing)"
  ( cd "${DECTALK_SRC}" && make -j"$(nproc)" && make install )
else
  log "DECtalk already built (dist/say exists)"
fi

log "Ensuring DECtalk wrapper scripts exist"
# Prefer repo-tracked wrappers if you add them; otherwise generate minimal ones ONCE.
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

# A small, forgiving wrapper.
# Supports:
#   dectalk-text2wav --out /path/file.wav --voice 9 --rate 165 "text..."
# Also tolerates:
#   dectalk-text2wav /path/file.wav "text..."
#   dectalk-text2wav "text..." /path/file.wav

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

# Heuristics: if OUT wasn't set, see if first/last arg looks like a wav path.
if [[ -z "${OUT}" && ${#TEXT_ARGS[@]} -ge 2 ]]; then
  if [[ "${TEXT_ARGS[0]}" == *.wav ]]; then
    OUT="${TEXT_ARGS[0]}"
    TEXT_ARGS=("${TEXT_ARGS[@]:1}")
  elif [[ "${TEXT_ARGS[-1]}" == *.wav ]]; then
    OUT="${TEXT_ARGS[-1]}"
    unset 'TEXT_ARGS[-1]'
  fi
fi

if [[ -z "${OUT}" ]]; then
  echo "ERR: output wav path not provided. Use --out /path/file.wav" >&2
  exit 2
fi

TEXT="${TEXT_ARGS[*]:-}"
if [[ -z "${TEXT}" ]]; then
  # Read stdin if no text args
  TEXT="$(cat)"
fi

SAY_BIN="${DECTALK_SAY_BIN:-/opt/dectalk/dectalk/dist/say}"
if [[ ! -x "${SAY_BIN}" ]]; then
  SAY_BIN="$(command -v say || true)"
fi
if [[ -z "${SAY_BIN}" || ! -x "${SAY_BIN}" ]]; then
  echo "ERR: DECtalk say binary not found. Expected /opt/dectalk/dectalk/dist/say" >&2
  exit 3
fi

HELP="$("${SAY_BIN}" --help 2>&1 || true)"

# Detect output flag
OUTFLAG=""
if grep -qE '(^|[[:space:]])-fo([[:space:]]|$)' <<<"${HELP}"; then
  OUTFLAG="-fo"
elif grep -qE '(^|[[:space:]])-o([[:space:]]|$)' <<<"${HELP}"; then
  OUTFLAG="-o"
elif grep -qiE '(^|[[:space:]])--output([[:space:]]|$)' <<<"${HELP}"; then
  OUTFLAG="--output"
fi

if [[ -z "${OUTFLAG}" ]]; then
  echo "ERR: couldn't detect file-output flag from say --help" >&2
  exit 4
fi

VOICE_ARGS=()
if grep -qE '(^|[[:space:]])-v([[:space:]]|$)' <<<"${HELP}"; then
  VOICE_ARGS=(-v "${VOICE}")
elif grep -qiE '(^|[[:space:]])--voice([[:space:]]|$)' <<<"${HELP}"; then
  VOICE_ARGS=(--voice "${VOICE}")
fi

RATE_ARGS=()
if grep -qE '(^|[[:space:]])-r([[:space:]]|$)' <<<"${HELP}"; then
  RATE_ARGS=(-r "${RATE}")
elif grep -qiE '(^|[[:space:]])--rate([[:space:]]|$)' <<<"${HELP}"; then
  RATE_ARGS=(--rate "${RATE}")
fi

# Run with env wrapper to ensure libs are visible.
# shellcheck disable=SC2086
/usr/local/bin/dectalk-env "${SAY_BIN}" "${VOICE_ARGS[@]}" "${RATE_ARGS[@]}" "${OUTFLAG}" "${OUT}" "${TEXT}"
EOF
  chmod 755 /usr/local/bin/dectalk-text2wav
fi

# -----------------------------------------------------------------------------------------
# Icecast config
# -----------------------------------------------------------------------------------------
log "Configuring Icecast"
if [[ -f /etc/default/icecast2 ]]; then
  sed -i 's/^ENABLE=.*/ENABLE=true/' /etc/default/icecast2 || true
  grep -q '^ENABLE=true' /etc/default/icecast2 || echo 'ENABLE=true' >> /etc/default/icecast2
fi

# Default passwords (you may override in /etc/seasonalweather/seasonalweather.env later)
ICECAST_SOURCE_PASSWORD="$(grep -E '^ICECAST_SOURCE_PASSWORD=' /etc/seasonalweather/seasonalweather.env 2>/dev/null | head -n1 | cut -d= -f2- || true)"
ICECAST_SOURCE_PASSWORD="${ICECAST_SOURCE_PASSWORD:-seasonal-source}"
ICECAST_ADMIN_PASSWORD="seasonal-admin"
ICECAST_RELAY_PASSWORD="seasonal-relay"

if [[ -f /etc/icecast2/icecast.xml ]]; then
  sed -i "s#<source-password>.*</source-password>#<source-password>${ICECAST_SOURCE_PASSWORD}</source-password>#g" /etc/icecast2/icecast.xml || true
  sed -i "s#<admin-password>.*</admin-password>#<admin-password>${ICECAST_ADMIN_PASSWORD}</admin-password>#g" /etc/icecast2/icecast.xml || true
  sed -i "s#<relay-password>.*</relay-password>#<relay-password>${ICECAST_RELAY_PASSWORD}</relay-password>#g" /etc/icecast2/icecast.xml || true
fi

# Keep Liquidsoap in sync with source password
if [[ -f /etc/seasonalweather/radio.liq ]]; then
  sed -i "s#password=\"[^\"]*\"#password=\"${ICECAST_SOURCE_PASSWORD}\"#g" /etc/seasonalweather/radio.liq || true
fi

log "Installing systemd units"
cp /opt/seasonalweather/app/systemd/seasonalweather.service /etc/systemd/system/seasonalweather.service
cp /opt/seasonalweather/app/systemd/seasonalweather-liquidsoap.service /etc/systemd/system/seasonalweather-liquidsoap.service
systemctl daemon-reload

log "Installing helper scripts (if present in repo)"
# These are optional: bootstrap won’t fail if they’re not in the repo.
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

log "Permissions"
chown -R seasonalweather:seasonalweather /var/lib/seasonalweather /var/log/seasonalweather
chmod 755 /var/lib/seasonalweather /var/lib/seasonalweather/audio /var/lib/seasonalweather/cache /var/log/seasonalweather

echo
echo "Next steps:"
echo "  1) sudo nano /etc/seasonalweather/seasonalweather.env   # set NWWS creds"
echo "  2) sudo systemctl enable --now icecast2"
echo "  3) sudo systemctl enable --now seasonalweather-liquidsoap"
echo "  4) sudo systemctl enable --now seasonalweather"
echo
echo "Listen:"
echo "  http://<vm-ip>:8000/seasonalweather.ogg"
