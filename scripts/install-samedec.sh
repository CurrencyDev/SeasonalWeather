#!/usr/bin/env bash
# Build/install the Rust samedec SAME/EAS decoder used by ERN/GWES monitoring.
#
# This script intentionally installs a pinned crate version instead of relying on
# an unmanaged host-local binary. It is safe to run repeatedly.
set -euo pipefail

SAMEDEC_VERSION_DEFAULT="0.4.2"
SAMEDEC_CRATE="${SEASONAL_SAMEDEC_CRATE:-samedec}"
SAMEDEC_VERSION="${SEASONAL_SAMEDEC_VERSION:-${SAMEDEC_VERSION_DEFAULT}}"
SAMEDEC_ROOT="${SEASONAL_SAMEDEC_ROOT:-/opt/seasonalweather/samedec}"
SAMEDEC_BIN="${SEASONAL_SAMEDEC_BIN:-/usr/local/bin/samedec}"

log()  { echo "[+] $*"; }
warn() { echo "[!] $*" >&2; }

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Run as root: sudo bash $0" >&2
  exit 1
fi

have_requested_samedec() {
  [[ -x "${SAMEDEC_BIN}" ]] && "${SAMEDEC_BIN}" --version 2>/dev/null | grep -Eq "(^| )${SAMEDEC_VERSION}($| )"
}

ensure_cargo() {
  if command -v cargo >/dev/null 2>&1; then
    return 0
  fi

  log "Installing Rust build toolchain for samedec"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y --no-install-recommends cargo rustc pkg-config ca-certificates
}

if have_requested_samedec; then
  log "samedec ${SAMEDEC_VERSION} already installed at ${SAMEDEC_BIN}"
  exit 0
fi

ensure_cargo

log "Installing ${SAMEDEC_CRATE} ${SAMEDEC_VERSION} into ${SAMEDEC_ROOT}"
install -d -o root -g root "${SAMEDEC_ROOT}"

# Keep cargo's install root out of /root/.cargo/bin and make the resulting binary
# explicit. --locked uses the crate's committed lockfile when available.
cargo install \
  --locked \
  --force \
  --root "${SAMEDEC_ROOT}" \
  --version "${SAMEDEC_VERSION}" \
  "${SAMEDEC_CRATE}"

if [[ ! -x "${SAMEDEC_ROOT}/bin/samedec" ]]; then
  warn "cargo install completed, but ${SAMEDEC_ROOT}/bin/samedec was not created"
  exit 1
fi

install -m 755 "${SAMEDEC_ROOT}/bin/samedec" "${SAMEDEC_BIN}"
printf '%s\n' "${SAMEDEC_VERSION}" >"${SAMEDEC_ROOT}/VERSION"

log "Installed $(${SAMEDEC_BIN} --version 2>/dev/null || echo samedec) at ${SAMEDEC_BIN}"
