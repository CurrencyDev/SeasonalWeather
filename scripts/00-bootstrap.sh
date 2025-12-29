#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Run as root: sudo bash $0"
  exit 1
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[+] Installing OS packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates \
  python3 python3-venv \
  ffmpeg \
  icecast2 \
  liquidsoap \
  espeak-ng \
  sox \
  rsync \
  tzdata

echo "[+] Creating user + directories"
if ! id -u seasonalweather >/dev/null 2>&1; then
  useradd --system --home /var/lib/seasonalweather --shell /usr/sbin/nologin seasonalweather
fi

install -d -o seasonalweather -g seasonalweather /var/lib/seasonalweather
install -d -o seasonalweather -g seasonalweather /var/lib/seasonalweather/audio
install -d -o seasonalweather -g seasonalweather /var/lib/seasonalweather/cache
install -d -o seasonalweather -g seasonalweather /var/log/seasonalweather
install -d -o root -g root /etc/seasonalweather

echo "[+] Syncing app to /opt/seasonalweather/app"
install -d -o root -g root /opt/seasonalweather
install -d -o root -g root /opt/seasonalweather/app
rsync -a --delete --exclude ".git" "${SRC_DIR}/" /opt/seasonalweather/app/

echo "[+] Creating Python venv"
if [[ ! -d /opt/seasonalweather/venv ]]; then
  python3 -m venv /opt/seasonalweather/venv
fi
/opt/seasonalweather/venv/bin/python -m pip install --upgrade pip wheel
/opt/seasonalweather/venv/bin/pip install -r /opt/seasonalweather/app/requirements.txt

echo "[+] Installing default config (won't overwrite existing)"
if [[ ! -f /etc/seasonalweather/config.yaml ]]; then
  cp /opt/seasonalweather/app/config/config.yaml /etc/seasonalweather/config.yaml
fi
if [[ ! -f /etc/seasonalweather/radio.liq ]]; then
  cp /opt/seasonalweather/app/liquidsoap/radio.liq /etc/seasonalweather/radio.liq
fi
if [[ ! -f /etc/seasonalweather/seasonalweather.env ]]; then
  cp /opt/seasonalweather/app/config/example.env /etc/seasonalweather/seasonalweather.env
  chmod 600 /etc/seasonalweather/seasonalweather.env
  echo "!!! Edit /etc/seasonalweather/seasonalweather.env and set NWWS_JID / NWWS_PASSWORD"
fi

echo "[+] Configuring Icecast"
# Enable daemon
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

echo "[+] Installing systemd units"
cp /opt/seasonalweather/app/systemd/seasonalweather.service /etc/systemd/system/seasonalweather.service
cp /opt/seasonalweather/app/systemd/seasonalweather-liquidsoap.service /etc/systemd/system/seasonalweather-liquidsoap.service
systemctl daemon-reload

echo "[+] Permissions"
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

# Keep Liquidsoap in sync with source password
if [[ -f /etc/seasonalweather/radio.liq ]]; then
  sed -i "s#password=\"[^\"]*\"#password=\"${ICECAST_SOURCE_PASSWORD}\"#g" /etc/seasonalweather/radio.liq || true
fi
