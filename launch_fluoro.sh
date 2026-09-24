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

# Size the fullscreen UI to the attached display (e.g. a 1024x600 touch
# monitor): ask xrandr for the current mode and pass it as --screen so the
# app composes natively at that resolution — letterboxed video, touch-sized
# buttons, and the on-screen keyboard laid out for the real panel. If xrandr
# is unavailable (or headless), fall back to the classic camera-sized layout.
#
# Right after boot the desktop's output manager (kanshi, ~/.config/kanshi/config)
# may still be switching the mode - the bench monitor is set to 720p there,
# because the Pi 3 paints 1080p too slowly - and a mode read mid-switch lays
# the UI out for the wrong resolution. So early on, wait for the mode to hold
# steady (and for xrandr to answer at all) before trusting it.
current_mode() { xrandr --current 2>/dev/null | awk '/\*/ {print $1; exit}'; }
SCREEN_ARGS=()
MODE=$(current_mode)
if (( $(cut -d. -f1 /proc/uptime) < 180 )); then
    for _ in 1 2 3 4 5 6; do
        sleep 3
        NEXT=$(current_mode)
        [[ -n "$NEXT" && "$NEXT" == "$MODE" ]] && break
        MODE=$NEXT
    done
fi
if [[ "${MODE:-}" =~ ^[0-9]+x[0-9]+$ ]]; then
    echo "display mode detected: $MODE"
    SCREEN_ARGS=(--screen "$MODE")
fi

# Serves the panel and shows the FLUORO window (no --no-window).
# --host 127.0.0.1: only local cloudflared may connect (never expose :5000 on
# the LAN — the app has no auth of its own; Cloudflare Access is the gate).
# --http: cloudflared originates over plain HTTP; TLS terminates at the edge.
exec python3 fluoro_web.py --host 127.0.0.1 --http "${SCREEN_ARGS[@]}" "$@"
