#!/usr/bin/env bash
#
# Launch FluoroSim: the on-screen FLUORO simulation window AND the web control
# panel server from one process (fluoro_web.py, default window mode).
#
# The panel binds to 127.0.0.1 only: on a provisioned Pi the sole way in is the
# Cloudflare Tunnel (cloudflared on this machine -> localhost:5000), with
# Cloudflare Access enforcing the customer sign-in at the edge. Customers reach
# it via their portal link (https://sim-<device-id>.<device-domain>/).
#
# Wired to the desktop shortcut ~/Desktop/FluoroSim.desktop.

set -u
cd "$(dirname "$(readlink -f "$0")")"

# Log every launch (and all of fluoro_web.py's output) so a failed desktop
# double-click can be diagnosed even though no terminal is attached.
LOG="$(pwd)/launch.log"
exec >>"$LOG" 2>&1
echo "===== launch $(date '+%Y-%m-%d %H:%M:%S')  DISPLAY=${DISPLAY:-unset}  WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-unset}  PWD=$(pwd) ====="

# OpenCV's Qt GUI needs the X11/XWayland backend for the fullscreen toggle to
# work (fluoro_web.py also sets this, but be explicit for the desktop launch).
export QT_QPA_PLATFORM=xcb

# A desktop double-click usually has no DISPLAY for X apps under a Wayland
# session; fall back to :0 so the FLUORO window can open.
export DISPLAY="${DISPLAY:-:0}"

# Serves the panel and shows the FLUORO window (no --no-window).
# --host 127.0.0.1: only local cloudflared may connect (never expose :5000 on
# the LAN — the app has no auth of its own; Cloudflare Access is the gate).
# --http: cloudflared originates over plain HTTP; TLS terminates at the edge.
exec python3 fluoro_web.py --host 127.0.0.1 --http
