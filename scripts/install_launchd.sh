#!/usr/bin/env bash
# Install (or with "uninstall", remove) the launchd agents for engine, guard and api.
# Usage: scripts/install_launchd.sh [install|uninstall]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HOME/Library/LaunchAgents"
mkdir -p "$DEST" "$ROOT/var"
for name in engine guard api; do
  label="com.tradingbot.$name"
  plist="$DEST/$label.plist"
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  if [ "${1:-install}" = "uninstall" ]; then
    rm -f "$plist"
    echo "removed $label"
    continue
  fi
  sed "s#__ROOT__#$ROOT#g" "$ROOT/deploy/$label.plist" > "$plist"
  plutil -lint "$plist" >/dev/null
  launchctl bootstrap "gui/$(id -u)" "$plist"
  echo "loaded $label (logs: $ROOT/var/$name.*.log)"
done
