#!/usr/bin/env bash
# One-shot setup on a fresh Ubuntu/Debian VM. Run as root:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/lime-paris-rebalancing/main/deploy/setup_vm.sh | REPO=<owner>/lime-paris-rebalancing bash
set -euo pipefail
REPO="${REPO:?set REPO=owner/repo}"
apt-get update -q && apt-get install -y -q git python3 python3-venv
id -u lime >/dev/null 2>&1 || useradd -m -s /bin/bash lime
sudo -u lime bash -c "
  cd ~ && { [ -d lime-paris-rebalancing ] || git clone https://github.com/${REPO}.git; }
  cd lime-paris-rebalancing
  python3 -m venv .venv && .venv/bin/pip install -q requests
"
cp /home/lime/lime-paris-rebalancing/deploy/lime-collector.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now lime-collector
systemctl status lime-collector --no-pager
