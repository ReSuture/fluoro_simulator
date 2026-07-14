#!/usr/bin/env python

'''
FluoroSim — Web control panel
=============================

Runs the FluoroSim fluoroscopy simulation (the same fullscreen ``FLUORO``
window and keyboard shortcuts as ``fluoro_simulator (3).py``) and, alongside it,
serves a small web page so the simulation can be toggled from a phone, tablet,
or any browser on the same network — handy for operating the demo from a tablet
while the monitor shows the fluoro view.

The web panel exposes the on/off toggles only (no numeric tuning sliders):

    Overlay, Equalize, Pedal mode, Pedal press, HUD, Record,
    plus Fullscreen / Windowed / Start / Stop actions and a live preview.
    Stop puts the simulation into standby (camera released, FLUORO window
    closed) while this web server keeps running, so Start can bring it back
    remotely without touching the Pi.

The FLUORO window itself is tabbed: **Fluoro** (the sim), **Remote Access**
(WiFi setup + registering the device to a customer email on the ReSuture
portal — see device_setup.py), and **Library** (browse/replay the session
recordings made with the Record toggle — see recording.py). The sim is fully
usable offline; the tabs never block launch.

Architecture
------------
The capture + processing loop and the OpenCV ``FLUORO`` window run on the main
thread (OpenCV's GUI must run on the main thread). Flask runs in a background
daemon thread. The two communicate only through a small ``state`` dict guarded
by ``state_lock`` (the toggle flags) and the JPEG buffer ``_latest_jpeg`` guarded
by ``_latest_lock`` (the latest frame for the preview). The web handlers never
touch OpenCV directly — they just flip flags that the main loop reads each frame.

HTTPS
-----
If ``cert.pem`` and ``key.pem`` are present next to this script, the panel is
served over HTTPS (so browsers that force secure connections can reach it). Pass
``--http`` to force plain HTTP. Generate a self-signed cert (valid ~2 years) with:

    openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem \\
        -days 825 -subj "/CN=FluoroSim" \\
        -addext "subjectAltName=IP:<your-lan-ip>,DNS:localhost,IP:127.0.0.1"

Usage:
    python fluoro_web.py [<video device number>] [--port 5000] [--host 0.0.0.0]
                         [--no-window] [--http]

Then open  https://<this-machine-ip>:<port>/  in a browser (http:// without a cert).

``--host 127.0.0.1`` restricts the panel to local connections only — used on
Pis exposed through a Cloudflare Tunnel, where the tunnel daemon on the same
machine is the only client allowed to reach the panel directly.
'''

from __future__ import print_function

import os
import sys
import time
import threading
import traceback

# Force OpenCV's Qt GUI onto the X11/XWayland backend. Under the native Wayland
# Qt backend, cv.setWindowProperty(FULLSCREEN) is a no-op (it logs
# "qt.qpa.wayland: Wayland does not support QWindow::requestActivate()") so the
# Fullscreen/Windowed controls can't change the FLUORO window. Set before cv2 is
# imported; respect an explicit override if the user already set one.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
import cv2 as cv
from flask import Flask, Response, jsonify, render_template_string, request

import device_setup
import recording
import uploader

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OVERLAY_IMAGE = os.path.join(BASE_DIR, "skel.jpg")
LOGO_IMAGE = os.path.join(BASE_DIR, "static", "logosign_white.png")

# Brightness threshold above which a pixel is treated as bright "white background"
# and composited mostly from the overlay. (Mirrors fluoro_simulator (3).py.)
MASK_THRESHOLD = 220
# White-background-removal freeze: once the raw camera frame has stayed largely
# the same for STABLE_FRAMES consecutive frames, the composite's white-background
# mask is frozen so the exact same removal is applied to every following frame
# (re-thresholding each live frame otherwise flips pixels near MASK_THRESHOLD
# between the two blend weights with sensor noise, making a steady scene pulse).
# The freeze holds until the frame drifts from the frozen reference frame by more
# than CHANGE_DIFF (camera pan, lighting change), then it can re-freeze once the
# scene settles again. Both thresholds are mean gray levels per pixel (0-255).
STABLE_FRAMES = 15
STABLE_DIFF = 3.0
CHANGE_DIFF = 10.0

# ── Camera-position → viewport calibration ────────────────────────────────────
# The full-resolution "master" background: a plain (NO-CONTRAST) radiograph of the
# torso, roughly brachiocephalic → femoral. The live camera feed superimposes the
# real vasculature on top, so the master must NOT show opacified vessels. As the
# camera moves, we crop a viewport out of this master at the camera's (x, y)
# position and scale it to the frame, so the anatomy pans like a C-arm.
#
# The camera position arrives as an (x_cm, y_cm) location measured in centimetres
# from an origin point. How it's produced (a separate engineer's motion
# controller) is not this module's concern — it's pushed in via POST /api/position
# and lives in state["pos_x_cm"] / state["pos_y_cm"].
#
# Calibrated against fulltorsofluoroimage.png (472×868), a plain no-contrast
# full-torso radiograph. Retune these four values if the master image changes.
MASTER_IMAGE = os.path.join(BASE_DIR, "fulltorsofluoroimage.png")
# Brightness scale applied to the master at load. <1.0 darkens it (pulls the
# bright/white areas down the most, since it's multiplicative); 1.0 = as-is.
MASTER_BRIGHTNESS = 1.0
# Master-image pixels per real centimetre of anatomy.
PX_PER_CM = 5.0
# The master pixel that camera coordinate (0, 0) maps to (viewport centre at origin).
ORIGIN_PX = (236, 434)          # centre of the 472×868 master
# Physical area the detector sees at once, (width_cm, height_cm). Keep 4:3 to match
# the camera frame. This is the ZOOM knob: larger = zoomed out (more anatomy shown),
# smaller = zoomed in. Tuned so one frame shows most of the body width (like
# skel.jpg filled the frame) while leaving room to pan.
FOV_CM = (80.0, 60.0)
# Flip a sign if increasing X (or Y) should pan the viewport the opposite way.
FLIP_X = 1.0
FLIP_Y = 1.0
# Zoom (Z) calibration: nominal camera height above the anatomy (cm). The field
# of view scales with (ZOOM_REF_CM + z) / ZOOM_REF_CM, so Z+ (raising the
# camera) shows more anatomy (zoom out) and Z− zooms in; z = 0 gives exactly
# FOV_CM. Flip FLIP_Z if the Z axis runs the other way.
ZOOM_REF_CM = 50.0
FLIP_Z = 1.0
# How far each on-screen nudge button / keyboard keypress moves the camera
# position (cm) — shared by the X/Y pan and Z zoom controls.
PAN_STEP_CM = 5.0

# ── Shared state ────────────────────────────────────────────────────────────────
# All values touched by both the processing loop and the web handlers live here,
# guarded by `state_lock`. Booleans mirror the keyboard toggles of the original.
state_lock = threading.Lock()
state = {
    "overlay": True,        # (2/5) anatomy overlay; off => full raw video
    "equalize": False,      # (6) CLAHE histogram equalisation
    "pedal_mode": False,    # (space) only capture while the pedal is pressed
    "pedal_pressed": False, # web stand-in for holding the foot pedal / 'b' key
    "hud": True,            # (7) on-screen text HUD
    "fullscreen": True,     # (3/4) FLUORO window fullscreen vs. windowed
    "running": True,        # simulation active; False = standby (web server stays up)
    "quit": False,          # exit the whole process (ESC key); web Stop uses "running"
    "pos_x_cm": 0.0,        # camera X position (cm from origin) — drives the viewport
    "pos_y_cm": 0.0,        # camera Y position (cm from origin) — drives the viewport
    "pos_z_cm": 0.0,        # camera Z position (cm from nominal height) — drives the zoom
    "ui_view": "fluoro",    # which tab the FLUORO window shows: fluoro|remote|library
    "recording": False,     # session recording to ~/fluorosim_recordings
}

# Latest processed frame, JPEG-encoded, for the MJPEG preview stream.
_latest_jpeg = None
_latest_lock = threading.Lock()


def get_state_snapshot():
    '''Return a thread-safe shallow copy of the shared state dict.

    The processing loop reads a snapshot once per frame so the flags can't change
    underneath it mid-frame, and so it doesn't hold the lock during heavy work.
    '''
    with state_lock:
        return dict(state)


# ── Flask app ───────────────────────────────────────────────────────────────────
app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>NAVISLab Controls</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background: #0b0f14; color: #e7edf3; }
  header { padding: 18px 16px; display: flex; flex-direction: column; align-items: center; gap: 10px;
           border-bottom: 1px solid #1d2733; position: sticky; top: 0; background: #0b0f14; }
  header .titlerow { display: flex; align-items: center; gap: 10px; }
  header h1 { font-size: 18px; margin: 0; font-weight: 600; }
  header img.logo { width: min(520px, 88%); height: auto; display: block; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: #f04438; }
  .dot.live { background: #12b76a; box-shadow: 0 0 8px #12b76a; }
  main { padding: 16px; max-width: 720px; margin: 0 auto; }
  .preview { width: 100%; background: #000; border-radius: 12px; overflow: hidden;
             border: 1px solid #1d2733; aspect-ratio: 4 / 3; display: flex; }
  .preview img { width: 100%; height: 100%; object-fit: contain; }
  /* Fullscreen: stack the video on top (filling the available space) and a
     compact, full-width control strip below it — the controls never cover the
     video, and the video only loses the strip's height. Leaving fullscreen
     returns everything to the normal stacked layout. */
  .stage:fullscreen, .stage:-webkit-full-screen {
      display: flex; flex-direction: column; width: 100vw; height: 100vh; background: #000; }
  .stage:fullscreen .preview, .stage:-webkit-full-screen .preview {
      flex: 1 1 auto; min-height: 0; width: 100%;
      border: 0; border-radius: 0; aspect-ratio: auto; }
  .stage:fullscreen .panel, .stage:-webkit-full-screen .panel {
      flex: 0 0 auto; padding: 10px 12px; background: #0b0f14; border-top: 1px solid #1d2733; }
  /* Lay the buttons out horizontally so the strip stays short. */
  .stage:fullscreen .panel .grid, .stage:-webkit-full-screen .panel .grid {
      grid-template-columns: repeat(5, 1fr); margin-top: 0; }
  .stage:fullscreen .panel .actions, .stage:-webkit-full-screen .panel .actions {
      grid-template-columns: repeat(4, 1fr); margin-top: 8px; }
  .stage:fullscreen .panel button, .stage:-webkit-full-screen .panel button {
      padding: 10px 8px; font-size: 14px; }
  .stage:fullscreen .panel button.quit, .stage:-webkit-full-screen .panel button.quit {
      grid-column: auto; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 16px; }
  button { font-size: 16px; padding: 16px 12px; border-radius: 12px; border: 1px solid #26323f;
           background: #131a22; color: #e7edf3; cursor: pointer; font-weight: 600;
           transition: background .12s, border-color .12s; }
  button:active { transform: translateY(1px); }
  button.on { background: #103b2a; border-color: #12b76a; color: #7af0b6; }
  button.rec.on { background: #3b1013; border-color: #f04438; color: #ff9a9a; }
  button.toggle .st { display: block; font-size: 12px; font-weight: 500; opacity: .7; margin-top: 2px; }
  .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
  button.quit { background: #2a1416; border-color: #5b2327; color: #ff9a9a; }
  button.quit.on { background: #3b1013; border-color: #f04438; }
  .hint { color: #7d8a99; font-size: 12px; margin: 14px 2px 0; }
  .liblink { color: #7af0b6; font-size: 14px; font-weight: 600; text-decoration: none;
             border: 1px solid #26323f; border-radius: 999px; padding: 6px 16px; }
  .liblink:active { transform: translateY(1px); }
  .posctl { display: flex; gap: 10px; align-items: center; margin-top: 16px; flex-wrap: wrap; }
  .posctl label { display: flex; align-items: center; gap: 6px; color: #7d8a99; font-size: 14px; }
  .posctl input { width: 90px; font-size: 16px; padding: 12px 10px; border-radius: 10px;
                  border: 1px solid #26323f; background: #131a22; color: #e7edf3; }
  .posctl button { padding: 12px 18px; }
</style>
</head>
<body>
<header>
  <img class="logo" src="/static/logosign_white.png" alt="NAVISLab logo">
  <div class="titlerow">
    <span id="dot" class="dot"></span>
    <h1>NAVISLab Controls</h1>
  </div>
  {% if library_url %}
  <a class="liblink" href="{{ library_url }}" target="_blank" rel="noopener">Session Library &#8599;</a>
  {% endif %}
</header>
<main>
  <div class="stage" id="stage">
    <div class="preview"><img id="feed" src="/video_feed" alt="live preview"></div>

    <div class="panel">
      <div class="grid" id="toggles">
        <button class="toggle" data-toggle="overlay">Overlay<span class="st">—</span></button>
        <button class="toggle" data-toggle="equalize">Equalize<span class="st">—</span></button>
        <button class="toggle" data-toggle="hud">HUD<span class="st">—</span></button>
        <button class="toggle" data-toggle="pedal_mode">Pedal mode<span class="st">—</span></button>
        <button class="toggle" data-toggle="pedal_pressed">Pedal press<span class="st">—</span></button>
        <button class="toggle rec" data-toggle="recording">Record<span class="st">—</span></button>
      </div>

      <div class="actions">
        <button class="fsbtn" data-action="fullscreen" data-fs="1">Fullscreen</button>
        <button class="fsbtn" data-action="windowed" data-fs="0">Windowed</button>
        <button class="runbtn" data-action="start" data-run="1">Start simulator</button>
        <button class="quit runbtn" data-action="stop" data-run="0">Stop simulator</button>
      </div>
    </div>
  </div>

  <div class="posctl">
    <label>X <input id="xin" type="number" step="0.5" value="0"> cm</label>
    <label>Y <input id="yin" type="number" step="0.5" value="0"> cm</label>
    <label>Z <input id="zin" type="number" step="0.5" value="0"> cm</label>
    <button id="gopos" type="button">Pan</button>
  </div>
  <p class="hint" id="pos">Camera: —</p>
  <p class="hint">Overlay off shows the full raw video. Pedal press only matters when Pedal mode is on.</p>
</main>

<script>
function applyState(s) {
  document.querySelectorAll('[data-toggle]').forEach(function (b) {
    var on = !!s[b.dataset.toggle];
    b.classList.toggle('on', on);
    b.querySelector('.st').textContent = on ? 'ON' : 'OFF';
  });
  document.querySelectorAll('.fsbtn').forEach(function (b) {
    b.classList.toggle('on', (b.dataset.fs === '1') === !!s.fullscreen);
  });
  document.querySelectorAll('.runbtn').forEach(function (b) {
    b.classList.toggle('on', (b.dataset.run === '1') === !!s.running);
  });
  var pos = document.getElementById('pos');
  if (pos) pos.textContent = 'Camera: x=' + (+s.pos_x_cm || 0).toFixed(1) +
                             '  y=' + (+s.pos_y_cm || 0).toFixed(1) +
                             '  z=' + (+s.pos_z_cm || 0).toFixed(1) + ' cm';
}
function refresh() {
  fetch('/api/state').then(function (r) { return r.json(); }).then(applyState).catch(function () {});
}
document.querySelectorAll('[data-toggle]').forEach(function (b) {
  b.addEventListener('click', function () {
    fetch('/api/toggle/' + b.dataset.toggle, { method: 'POST' })
      .then(function (r) { return r.json(); }).then(applyState);
  });
});
function browserFullscreen(on) {
  // Fullscreen the whole stage (preview + overlaid controls) in THIS browser
  // (must run inside a click handler).
  var el = document.getElementById('stage');
  try {
    if (on) {
      var req = el.requestFullscreen || el.webkitRequestFullscreen;
      if (req) req.call(el);
    } else {
      var exit = document.exitFullscreen || document.webkitExitFullscreen;
      if (exit && (document.fullscreenElement || document.webkitFullscreenElement)) exit.call(document);
    }
  } catch (e) {}
}
document.querySelectorAll('[data-action]').forEach(function (b) {
  b.addEventListener('click', function () {
    if (b.dataset.action === 'stop' && !confirm('Stop the simulator?')) return;
    // Fullscreen/Windowed also control this browser's preview, not just the
    // popup window on the computer running the simulator.
    if (b.dataset.action === 'fullscreen') browserFullscreen(true);
    if (b.dataset.action === 'windowed') browserFullscreen(false);
    fetch('/api/action/' + b.dataset.action, { method: 'POST' })
      .then(function (r) { return r.json(); }).then(applyState);
  });
});
function sendPos() {
  var x = parseFloat(document.getElementById('xin').value) || 0;
  var y = parseFloat(document.getElementById('yin').value) || 0;
  var z = parseFloat(document.getElementById('zin').value) || 0;
  fetch('/api/position', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x: x, y: y, z: z }) })
    .then(function (r) { return r.json(); }).then(applyState).catch(function () {});
}
document.getElementById('gopos').addEventListener('click', sendPos);
['xin', 'yin', 'zin'].forEach(function (id) {
  var el = document.getElementById(id);
  el.addEventListener('change', sendPos);                 // fires on blur / stepper arrows
  el.addEventListener('keydown', function (e) { if (e.key === 'Enter') sendPos(); });
});
var feed = document.getElementById('feed');
feed.addEventListener('load', function () { document.getElementById('dot').classList.add('live'); });
feed.addEventListener('error', function () { document.getElementById('dot').classList.remove('live'); });
// Tablets suspend a backgrounded tab's timers and its MJPEG connection, so the
// toggle chips go stale (showing e.g. ON for a state that has since changed)
// and the preview freezes. Re-sync and restart the stream the moment the page
// is visible again instead of waiting for the next poll.
function resume() {
  refresh();
  feed.src = '/video_feed?t=' + Date.now();
}
document.addEventListener('visibilitychange', function () { if (!document.hidden) resume(); });
window.addEventListener('pageshow', resume);
refresh();
setInterval(refresh, 1500);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    '''Serve the single-page control panel.'''
    # Provisioned devices link to their portal's session library (the tunneled
    # panel has no other path to it); bench boxes with no portal hide it.
    portal_url = device_setup.read_identity()["portal_url"]
    return render_template_string(
        PAGE, library_url=portal_url + "/library" if portal_url else None)


@app.route("/api/state")
def api_state():
    '''Return the current toggle states as JSON (polled by the page to stay in sync).'''
    return jsonify(get_state_snapshot())


@app.route("/api/toggle/<name>", methods=["POST"])
def api_toggle(name):
    '''Flip one boolean toggle and return the full updated state.

    Only the known toggle names are accepted; anything else is ignored so an
    arbitrary key can't be injected into the state dict.
    '''
    with state_lock:
        if name in ("overlay", "equalize", "pedal_mode", "pedal_pressed", "hud",
                    "recording"):
            state[name] = not state[name]
        snap = dict(state)
    return jsonify(snap)


@app.route("/api/action/<action>", methods=["POST"])
def api_action(action):
    '''Apply a one-shot action (fullscreen / windowed / start / stop) and return the state.

    "stop" puts the simulation into standby but keeps this server running so a
    later "start" can resume it remotely. ("quit" is kept as an alias of "stop"
    for old clients — the web panel no longer exits the process.)
    '''
    with state_lock:
        if action == "fullscreen":
            state["fullscreen"] = True
        elif action == "windowed":
            state["fullscreen"] = False
        elif action == "start":
            state["running"] = True
        elif action in ("stop", "quit"):
            state["running"] = False
        snap = dict(state)
    return jsonify(snap)


@app.route("/api/position", methods=["POST"])
def api_position():
    '''Set the current camera position, in centimetres from the origin.

    This is the single input hook for the motion source: the position provider
    POSTs the camera's (x, y) here and the processing loop pans the anatomy
    viewport to match. Accepts JSON ``{"x": <cm>, "y": <cm>}`` or plain form/query
    params ``x`` and ``y``. An optional ``z`` (cm) sets the zoom the same way;
    omitting it leaves the current zoom unchanged. Returns the full updated state.
    '''
    data = request.get_json(silent=True) or {}
    x = data.get("x", request.values.get("x"))
    y = data.get("y", request.values.get("y"))
    z = data.get("z", request.values.get("z"))
    try:
        x, y = float(x), float(y)
        z = float(z) if z is not None else None
    except (TypeError, ValueError):
        return jsonify({"error": "provide numeric x and y (cm), optional z (cm)"}), 400
    with state_lock:
        state["pos_x_cm"] = x
        state["pos_y_cm"] = y
        if z is not None:
            state["pos_z_cm"] = z
        snap = dict(state)
    return jsonify(snap)


def mjpeg_generator():
    '''Yield the latest processed frame as a multipart MJPEG stream.'''
    boundary = b"--frame"
    while True:
        with _latest_lock:
            buf = _latest_jpeg
        if buf is not None:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n"
        time.sleep(0.05)  # ~20 fps cap for the preview


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Image processing ────────────────────────────────────────────────────────────
def compute_viewport(master, x_cm, y_cm, z_cm=0.0):
    '''Crop the anatomy viewport out of the full-res master at a camera position.

    Maps the camera's (x_cm, y_cm) — centimetres from the origin — to a pixel
    window in ``master`` using the calibration constants, centred on that point.
    ``z_cm`` scales the window like camera height: Z+ widens the field of view
    (zoom out), Z− narrows it (zoom in); see ZOOM_REF_CM.
    The window is clamped to lie fully inside the image, so reaching the edge of
    travel pans up against the border instead of returning a truncated crop (which
    would otherwise distort when resized to the frame).

    Returns a view into ``master`` (no copy); the caller resizes it to the frame.
    '''
    mh, mw = master.shape[:2]
    zoom = max(0.05, (ZOOM_REF_CM + FLIP_Z * z_cm) / ZOOM_REF_CM)
    w = max(1, min(int(round(FOV_CM[0] * zoom * PX_PER_CM)), mw))
    h = max(1, min(int(round(FOV_CM[1] * zoom * PX_PER_CM)), mh))
    cx = ORIGIN_PX[0] + FLIP_X * x_cm * PX_PER_CM
    cy = ORIGIN_PX[1] + FLIP_Y * y_cm * PX_PER_CM
    x0 = max(0, min(int(round(cx - w / 2.0)), mw - w))
    y0 = max(0, min(int(round(cy - h / 2.0)), mh - h))
    return master[y0:y0 + h, x0:x0 + w]


def white_bg_mask(gray):
    '''Mask of bright "white background" pixels (0 where white background).'''
    _, m = cv.threshold(gray, MASK_THRESHOLD, 255, cv.THRESH_BINARY_INV)
    return cv.medianBlur(m, 5)


def composite_overlay(gray, overlay, equalize, bg_mask=None):
    '''Composite the anatomy overlay onto a grayscale camera frame.

    Mirrors the blend weights in fluoro_simulator (3).py: bright/white areas are
    30% video / 70% overlay; vasculature areas are 60% video / 40% overlay.

    ``bg_mask`` supplies a precomputed white-background mask (see white_bg_mask).
    Pass the frozen mask while the scene is steady: thresholding each live frame
    makes pixels near MASK_THRESHOLD flip between the two blend weights with
    sensor noise, which pulses. None = threshold ``gray`` (scene changing).
    '''
    if bg_mask is None:
        bg_mask = white_bg_mask(gray)

    ov = overlay.astype(np.float32)
    fr = gray.astype(np.float32)
    result = ov.copy()
    white = bg_mask == 0
    result[white] = 0.30 * fr[white] + 0.70 * ov[white]
    result[~white] = 0.60 * fr[~white] + 0.40 * ov[~white]
    out = np.clip(result, 0, 255).astype(np.uint8)

    if equalize:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(out)
    return out


def draw_str(dst, target, s):
    '''Draw white text with a black drop-shadow so it stays legible over any frame.'''
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


# ── On-screen control panel ───────────────────────────────────────────────────
# The same dark, clickable panel as fluoro_simulator (3).py, drawn on-screen next
# to / below the FLUORO window so the Pi has buttons even without a browser. It
# drives the same thread-safe `state` the web handlers use, so the on-screen
# buttons, the keyboard, and remote clients all stay in sync.

# Theme (BGR — matches the web panel's hex colours).
C_BG       = (20, 15, 11)     # #0b0f14
C_BTN_BG   = (34, 26, 19)     # #131a22
C_BTN_BD   = (63, 50, 38)     # #26323f
C_ON_BG    = (42, 59, 16)     # #103b2a
C_ON_BD    = (106, 183, 18)   # #12b76a
C_ON_TX    = (182, 240, 122)  # #7af0b6
C_TEXT     = (243, 237, 231)  # #e7edf3
C_SUBTEXT  = (153, 138, 125)  # #7d8a99
C_QUIT_BG  = (22, 20, 42)     # #2a1416
C_QUIT_BD  = (39, 35, 91)     # #5b2327
C_QUIT_TX  = (154, 154, 255)  # #ff9a9a
C_DOT_OFF  = (56, 68, 240)    # #f04438
C_DOT_ON   = (106, 183, 18)   # #12b76a
C_DOT_WAIT = (18, 180, 240)   # #f0b412 — upload pending/in progress

# Button hit-boxes (x, y, w, h, kind, name), rebuilt every render. `ctrl_buttons`
# are positions inside the separate CONTROLS window (windowed); `overlay_buttons`
# are positions on the FLUORO frame when the bar is stacked below it (fullscreen).
# `tab_buttons` is the persistent tab bar on top of the FLUORO frame, and
# `view_buttons` the active Remote Access / Library view's own buttons.
ctrl_buttons = []
overlay_buttons = []
tab_buttons = []
view_buttons = []

# UI-local state for the Remote Access / Library views (text-field contents,
# focus, list selection). Only ever touched from the OpenCV main thread — the
# mouse callback and cv.waitKey both run inside the main loop's GUI pump — so
# no lock is needed (unlike `state`, which the Flask thread also reads).
ui = {
    "focus": None,          # name of the focused text field: "password"|"email"
    "password": "",
    "email": "",
    "show_pw": False,
    "selected_ssid": None,  # network picked from the scan list
    "selected_secured": False,
    "net_page": 0,
    "email_editing": False, # entering/changing the registration email
}


def rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    '''Draw a (optionally filled) rounded rectangle to approximate the web buttons.'''
    if thickness < 0:
        cv.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
            cv.circle(img, (cx, cy), r, color, -1, cv.LINE_AA)
    else:
        cv.line(img, (x + r, y), (x + w - r, y), color, thickness, cv.LINE_AA)
        cv.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness, cv.LINE_AA)
        cv.line(img, (x, y + r), (x, y + h - r), color, thickness, cv.LINE_AA)
        cv.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness, cv.LINE_AA)
        cv.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, color, thickness, cv.LINE_AA)
        cv.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, color, thickness, cv.LINE_AA)
        cv.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, color, thickness, cv.LINE_AA)
        cv.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness, cv.LINE_AA)


def paste_rgba(dst, rgba, x, y, target_w):
    '''Alpha-composite an (RGBA or BGR) image onto dst, scaled to target_w. Returns drawn height.'''
    scale = target_w / float(rgba.shape[1])
    tw = target_w
    th = max(1, int(rgba.shape[0] * scale))
    r = cv.resize(rgba, (tw, th), interpolation=cv.INTER_AREA)
    if r.ndim == 3 and r.shape[2] == 4:
        a = r[:, :, 3:4].astype(np.float32) / 255.0
        bgr = r[:, :, :3].astype(np.float32)
    else:
        a = np.ones((th, tw, 1), np.float32)
        bgr = r.reshape(th, tw, -1)[:, :, :3].astype(np.float32)
    roi = dst[y:y + th, x:x + tw].astype(np.float32)
    dst[y:y + th, x:x + tw] = (a * bgr + (1.0 - a) * roi).astype(np.uint8)
    return th


def draw_button(img, x, y, w, h, label, sub=None, on=False, variant="normal"):
    '''Draw one themed button (fill + border + centred label, optional ON/OFF sub-label).'''
    if variant == "quit":
        bg, bd, tx = C_QUIT_BG, C_QUIT_BD, C_QUIT_TX
    elif on:
        bg, bd, tx = C_ON_BG, C_ON_BD, C_ON_TX
    else:
        bg, bd, tx = C_BTN_BG, C_BTN_BD, C_TEXT
    rounded_rect(img, x, y, w, h, 12, bg, -1)
    rounded_rect(img, x, y, w, h, 12, bd, 1)
    font = cv.FONT_HERSHEY_SIMPLEX
    if sub is None:
        (lw, lh), _ = cv.getTextSize(label, font, 0.55, 1)
        cv.putText(img, label, (x + (w - lw) // 2, y + (h + lh) // 2), font, 0.55, tx, 1, cv.LINE_AA)
    else:
        (lw, lh), _ = cv.getTextSize(label, font, 0.55, 1)
        cv.putText(img, label, (x + (w - lw) // 2, y + h // 2 - 2), font, 0.55, tx, 1, cv.LINE_AA)
        (sw, sh), _ = cv.getTextSize(sub, font, 0.42, 1)
        cv.putText(img, sub, (x + (w - sw) // 2, y + h // 2 + 16), font, 0.42,
                   tx if on else C_SUBTEXT, 1, cv.LINE_AA)


TAB_H = 44  # height of the persistent tab bar stacked on top of the FLUORO frame


def render_tab_bar(width, active_view, rec_elapsed=None):
    '''Render the persistent Fluoro / Remote Access / Library tab bar.

    Returns (image, buttons) with hit-boxes relative to the bar's top-left.
    `rec_elapsed` is a "mm:ss" string shown with a red dot while recording.
    '''
    img = np.full((TAB_H, width, 3), C_BG, np.uint8)
    cv.line(img, (0, TAB_H - 1), (width, TAB_H - 1), C_BTN_BD, 1, cv.LINE_AA)
    buttons = []
    pad, gap, bh = 5, 6, TAB_H - 10
    tw = min(190, (width - 2 * pad - 2 * gap) // 3)
    for i, (label, name) in enumerate([("Fluoro", "fluoro"),
                                       ("Remote Access", "remote"),
                                       ("Library", "library")]):
        bx = pad + i * (tw + gap)
        draw_button(img, bx, 5, tw, bh, label, None, active_view == name)
        buttons.append((bx, 5, tw, bh, "tab", name))
    if rec_elapsed is not None:
        font = cv.FONT_HERSHEY_SIMPLEX
        txt = "REC %s" % rec_elapsed
        (tw2, th2), _ = cv.getTextSize(txt, font, 0.55, 1)
        cv.circle(img, (width - tw2 - 28, TAB_H // 2), 7, C_DOT_OFF, -1, cv.LINE_AA)
        cv.putText(img, txt, (width - tw2 - 14, (TAB_H + th2) // 2), font, 0.55,
                   C_QUIT_TX, 1, cv.LINE_AA)
    return img, buttons


def draw_text_field(img, x, y, w, h, text, focused=False, masked=False,
                    placeholder=""):
    '''Draw a clickable text field (the keyboard types into it while focused).'''
    rounded_rect(img, x, y, w, h, 8, C_BTN_BG, -1)
    rounded_rect(img, x, y, w, h, 8, C_ON_BD if focused else C_BTN_BD, 1)
    font = cv.FONT_HERSHEY_SIMPLEX
    shown = ("*" * len(text)) if masked else text
    color = C_TEXT
    if focused:
        shown += "_"
    elif not shown and placeholder:
        shown, color = placeholder, C_SUBTEXT
    # Keep the tail visible when the text outgrows the box.
    while len(shown) > 1 and cv.getTextSize(shown, font, 0.55, 1)[0][0] > w - 20:
        shown = shown[1:]
    (_, th), _ = cv.getTextSize("Ag", font, 0.55, 1)
    cv.putText(img, shown, (x + 10, y + (h + th) // 2), font, 0.55, color, 1, cv.LINE_AA)


def render_controls(s, logo, live):
    '''Render the vertical control panel (logo + button grid) for the CONTROLS window.

    Returns (image, buttons) with hit-boxes relative to the panel's top-left.
    `s` is a state snapshot.
    '''
    W, m, gap, top_pad = 380, 16, 10, 18
    bt, ba, cap_h = 56, 52, 22

    lw = W - 2 * m
    lh = int(logo.shape[0] * (lw / float(logo.shape[1]))) if logo is not None else 0
    # rows: 3 toggle rows, then a location caption + nudge row, then Fullscreen/
    # Windowed, then Quit.
    H = (top_pad + lh + 14 + 26 + 14 + bt * 3 + gap * 3
         + cap_h + gap + ba + gap + ba + gap + ba + 16)

    img = np.full((H, W, 3), C_BG, np.uint8)
    buttons = []
    y = top_pad

    if logo is not None:
        paste_rgba(img, logo, m, y, lw)
    y += lh + 14

    title = "NAVISLab Controls"
    font = cv.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv.getTextSize(title, font, 0.6, 1)
    tx0 = (W - (tw + 18)) // 2
    cv.circle(img, (tx0 + 5, y + 9), 5, C_DOT_ON if live else C_DOT_OFF, -1, cv.LINE_AA)
    cv.putText(img, title, (tx0 + 18, y + 9 + th // 2), font, 0.6, C_TEXT, 1, cv.LINE_AA)
    y += 26 + 14

    cw = (W - 2 * m - gap) // 2
    toggles = [("Overlay", "overlay"), ("Equalize", "equalize"),
               ("HUD", "hud"), ("Pedal mode", "pedal_mode"),
               ("Pedal press", "pedal_pressed"), ("Record", "recording")]
    for i, (label, key) in enumerate(toggles):
        col, row = i % 2, i // 2
        bx = m + col * (cw + gap)
        by = y + row * (bt + gap)
        on = bool(s[key])
        draw_button(img, bx, by, cw, bt, label, "ON" if on else "OFF", on,
                    "quit" if key == "recording" and on else "normal")
        buttons.append((bx, by, cw, bt, "toggle", key))
    y += 3 * (bt + gap)

    # Background-location readout + nudge buttons (X−/X+/Y−/Y+/Z−/Z+).
    cap = "Background   X %.0f   Y %.0f   Z %.0f cm" % (
        s["pos_x_cm"], s["pos_y_cm"], s["pos_z_cm"])
    (capw, caph), _ = cv.getTextSize(cap, font, 0.5, 1)
    cv.putText(img, cap, ((W - capw) // 2, y + caph + 2), font, 0.5, C_SUBTEXT, 1, cv.LINE_AA)
    y += cap_h + gap
    pw = (W - 2 * m - 5 * gap) // 6
    for i, (label, name) in enumerate([("X-", "xm"), ("X+", "xp"), ("Y-", "ym"),
                                       ("Y+", "yp"), ("Z-", "zm"), ("Z+", "zp")]):
        bx = m + i * (pw + gap)
        draw_button(img, bx, y, pw, ba, label, None, False)
        buttons.append((bx, y, pw, ba, "pan", name))
    y += ba + gap

    draw_button(img, m, y, cw, ba, "Fullscreen", None, s["fullscreen"])
    buttons.append((m, y, cw, ba, "action", "fullscreen"))
    draw_button(img, m + cw + gap, y, cw, ba, "Windowed", None, not s["fullscreen"])
    buttons.append((m + cw + gap, y, cw, ba, "action", "windowed"))
    y += ba + gap

    draw_button(img, m, y, W - 2 * m, ba, "Stop simulator", None, False, "quit")
    buttons.append((m, y, W - 2 * m, ba, "action", "stop"))
    return img, buttons


# Control-bar geometry, shared with the Remote Access / Library views so their
# canvases match the fluoro composite's height exactly (no window resizing).
BAR_PAD, BAR_GAP, BAR_BH = 6, 6, 34
BAR_H = BAR_PAD + 3 * BAR_BH + 2 * BAR_GAP + BAR_PAD  # 3 rows: toggles, actions, location


def render_control_bar(s, width, live):
    '''Render a short, full-width control strip stacked below the video in fullscreen.

    Returns (image, buttons) with hit-boxes relative to the bar's top-left.
    '''
    pad, gap, bh = BAR_PAD, BAR_GAP, BAR_BH
    bar_h = BAR_H
    img = np.full((bar_h, width, 3), C_BG, np.uint8)
    cv.line(img, (0, 0), (width, 0), C_BTN_BD, 1, cv.LINE_AA)
    buttons = []

    toggles = [("Overlay", "overlay"), ("Equalize", "equalize"),
               ("HUD", "hud"), ("Pedal mode", "pedal_mode"),
               ("Pedal press", "pedal_pressed"), ("Record", "recording")]
    cw = (width - 2 * pad - 5 * gap) // 6
    y = pad
    for i, (label, key) in enumerate(toggles):
        bx = pad + i * (cw + gap)
        on = bool(s[key])
        draw_button(img, bx, y, cw, bh, label, None, on,
                    "quit" if key == "recording" and on else "normal")
        buttons.append((bx, y, cw, bh, "toggle", key))

    y += bh + gap
    aw = (width - 2 * pad - 2 * gap) // 3
    draw_button(img, pad, y, aw, bh, "Fullscreen", None, s["fullscreen"])
    buttons.append((pad, y, aw, bh, "action", "fullscreen"))
    draw_button(img, pad + aw + gap, y, aw, bh, "Windowed", None, not s["fullscreen"])
    buttons.append((pad + aw + gap, y, aw, bh, "action", "windowed"))
    draw_button(img, pad + 2 * (aw + gap), y, aw, bh, "Stop simulator", None, False, "quit")
    buttons.append((pad + 2 * (aw + gap), y, aw, bh, "action", "stop"))

    # Row 3: background-location readout + nudge buttons (X−/X+/Y−/Y+/Z−/Z+).
    y += bh + gap
    cap = "Bg  X %.0f  Y %.0f  Z %.0f cm" % (
        s["pos_x_cm"], s["pos_y_cm"], s["pos_z_cm"])
    cap_w = 240
    (_, caph), _ = cv.getTextSize(cap, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv.putText(img, cap, (pad, y + (bh + caph) // 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, C_SUBTEXT, 1, cv.LINE_AA)
    pw = (width - 2 * pad - cap_w - 5 * gap) // 6
    x = pad + cap_w
    for (label, name) in [("X-", "xm"), ("X+", "xp"), ("Y-", "ym"),
                          ("Y+", "yp"), ("Z-", "zm"), ("Z+", "zp")]:
        draw_button(img, x, y, pw, bh, label, None, False)
        buttons.append((x, y, pw, bh, "pan", name))
        x += pw + gap
    return img, buttons


# ── Remote Access view ────────────────────────────────────────────────────────
# WiFi + portal registration, drawn into the FLUORO window under the tab bar.
# All slow work (nmcli, portal HTTP) lives in device_setup.SetupManager on
# daemon threads; this section only renders its snapshot and posts button
# presses back to it.

setup_mgr = device_setup.SetupManager()

NET_ROWS = 5  # networks shown per page of the scan list


def _draw_signal_bars(img, x, y, signal):
    '''Four little bars, filled according to a 0-100 signal strength.'''
    for i in range(4):
        h = 5 + i * 4
        color = C_ON_BD if signal > i * 25 else C_BTN_BD
        cv.rectangle(img, (x + i * 7, y - h), (x + i * 7 + 4, y), color, -1)


def render_remote_view(s, W, H):
    '''Render the Remote Access view. Returns (canvas, buttons).'''
    setup_mgr.refresh_status()  # TTL-gated and non-blocking
    st = setup_mgr.snapshot()
    img = np.full((H, W, 3), C_BG, np.uint8)
    buttons = []
    m, gap, bh = 24, 8, 38
    font = cv.FONT_HERSHEY_SIMPLEX

    def text(txt, x, yy, scale=0.55, color=C_TEXT):
        cv.putText(img, txt, (x, yy), font, scale, color, 1, cv.LINE_AA)

    # ── WiFi ──
    y = 34
    text("WiFi", m, y, 0.7)
    cv.circle(img, (m + 78, y - 6), 6,
              C_DOT_ON if st["wifi_connected"] else C_DOT_OFF, -1, cv.LINE_AA)
    text("connected to %s" % st["wifi_ssid"] if st["wifi_connected"]
         else "not connected", m + 94, y, 0.55, C_SUBTEXT)
    if st["provisioned"]:
        chip = ("portal: online", C_ON_TX) if st["online"] else \
               ("portal: unreachable", C_QUIT_TX) if st["wifi_connected"] else None
        if chip:
            (cw_, _), _ = cv.getTextSize(chip[0], font, 0.5, 1)
            text(chip[0], W - m - cw_, y, 0.5, chip[1])
    y += 14

    scanning = st["wifi"] == "scanning"
    draw_button(img, m, y, 250, bh,
                "Scanning..." if scanning else "Scan for networks", None, scanning)
    if not scanning:
        buttons.append((m, y, 250, bh, "setup", "scan"))
    y += bh + gap

    # Scan-result list (paged).
    nets = st["networks"]
    page = min(ui["net_page"], max(0, (len(nets) - 1) // NET_ROWS))
    ui["net_page"] = page
    for i in range(page * NET_ROWS, min(len(nets), (page + 1) * NET_ROWS)):
        n = nets[i]
        selected = n["ssid"] == ui["selected_ssid"]
        draw_button(img, m, y, W - 2 * m, 32, "", None, selected)
        label = n["ssid"] if len(n["ssid"]) <= 40 else n["ssid"][:39] + "~"
        text(label, m + 14, y + 22, 0.55, C_ON_TX if selected else C_TEXT)
        _draw_signal_bars(img, W - m - 100, y + 24, n["signal"])
        if n["secured"]:
            text("secured", W - m - 70, y + 22, 0.45, C_SUBTEXT)
        buttons.append((m, y, W - 2 * m, 32, "setup", "pick:%d" % i))
        y += 32 + 4
    if len(nets) > NET_ROWS:
        pw_ = 90
        if page > 0:
            draw_button(img, m, y, pw_, 30, "Prev", None, False)
            buttons.append((m, y, pw_, 30, "setup", "page_prev"))
        if (page + 1) * NET_ROWS < len(nets):
            draw_button(img, m + pw_ + gap, y, pw_, 30, "Next", None, False)
            buttons.append((m + pw_ + gap, y, pw_, 30, "setup", "page_next"))
        y += 30 + gap

    # Selected network: password + connect.
    if ui["selected_ssid"]:
        connecting = st["wifi"] == "connecting"
        if connecting:
            text("Connecting to %s..." % ui["selected_ssid"], m, y + 24, 0.55, C_ON_TX)
            y += bh + gap
        else:
            fw_ = min(360, W - 2 * m - 3 * (90 + gap))
            if ui["selected_secured"]:
                draw_text_field(img, m, y, fw_, bh, ui["password"],
                                ui["focus"] == "password", not ui["show_pw"],
                                "password for %s" % ui["selected_ssid"])
                buttons.append((m, y, fw_, bh, "focus", "password"))
                bx = m + fw_ + gap
                draw_button(img, bx, y, 90, bh, "Hide" if ui["show_pw"] else "Show",
                            None, ui["show_pw"])
                buttons.append((bx, y, 90, bh, "setup", "show_pw"))
                bx += 90 + gap
            else:
                text("Open network: %s" % ui["selected_ssid"], m, y + 24, 0.55, C_SUBTEXT)
                bx = m + fw_ + 2 * gap + 90
            draw_button(img, bx, y, 110, bh, "Connect", None, True)
            buttons.append((bx, y, 110, bh, "setup", "connect"))
            bx += 110 + gap
            draw_button(img, bx, y, 90, bh, "Cancel", None, False)
            buttons.append((bx, y, 90, bh, "setup", "cancel_net"))
            y += bh + gap

    # ── Registration ──
    y += 10
    cv.line(img, (m, y), (W - m, y), C_BTN_BD, 1, cv.LINE_AA)
    y += 34
    text("Remote access", m, y, 0.7)
    y += 16

    if not st["provisioned"]:
        text("Remote registration unavailable — this device has not been",
             m, y + 20, 0.55, C_SUBTEXT)
        text("provisioned. Contact ReSuture. The simulator works normally.",
             m, y + 44, 0.55, C_SUBTEXT)
        y += 60
    elif st["registered_email"] and not ui["email_editing"]:
        suffix = "" if st["online"] else "  (cached - offline)"
        text("Registered to %s%s" % (st["registered_email"], suffix),
             m, y + 24, 0.6)
        draw_button(img, m, y + 38, 110, bh, "Change", None, False)
        buttons.append((m, y + 38, 110, bh, "setup", "change"))
        y += 38 + bh + gap
    else:
        claiming = st["portal"] == "claiming"
        if claiming:
            text("Registering...", m, y + 24, 0.6, C_ON_TX)
            y += bh + gap
        else:
            text("Enter your email to register this simulator to your account:",
                 m, y + 18, 0.5, C_SUBTEXT)
            y += 28
            fw_ = min(420, W - 2 * m - 2 * (120 + gap))
            draw_text_field(img, m, y, fw_, bh, ui["email"],
                            ui["focus"] == "email", False, "you@example.com")
            buttons.append((m, y, fw_, bh, "focus", "email"))
            bx = m + fw_ + gap
            if st["online"]:
                draw_button(img, bx, y, 120, bh, "Register", None, True)
                buttons.append((bx, y, 120, bh, "setup", "claim"))
            else:
                draw_button(img, bx, y, 120, bh, "Register", None, False)
                text("connect to WiFi first", bx + 130, y + 24, 0.5, C_SUBTEXT)
            bx += 120 + gap
            if st["registered_email"]:
                draw_button(img, bx, y, 100, bh, "Cancel", None, False)
                buttons.append((bx, y, 100, bh, "setup", "cancel_email"))
            y += bh + gap

    # Errors / notices at the bottom of the section.
    if st["error"]:
        text(str(st["error"])[:90], m, y + 24, 0.5, C_QUIT_TX)
    elif st["notice"]:
        text(str(st["notice"])[:90], m, y + 24, 0.5, C_ON_TX)
    return img, buttons


def handle_setup_button(name):
    '''Apply a Remote Access button press (runs on the main thread; instant).'''
    if name == "scan":
        ui["net_page"] = 0
        setup_mgr.start_scan()
    elif name.startswith("pick:"):
        nets = setup_mgr.snapshot()["networks"]
        idx = int(name.split(":", 1)[1])
        if 0 <= idx < len(nets):
            ui["selected_ssid"] = nets[idx]["ssid"]
            ui["selected_secured"] = nets[idx]["secured"]
            ui["password"] = ""
            ui["focus"] = "password" if nets[idx]["secured"] else None
    elif name == "connect":
        if ui["selected_ssid"]:
            setup_mgr.connect(ui["selected_ssid"], ui["password"])
    elif name == "cancel_net":
        ui["selected_ssid"], ui["password"], ui["focus"] = None, "", None
    elif name == "show_pw":
        ui["show_pw"] = not ui["show_pw"]
    elif name == "page_prev":
        ui["net_page"] = max(0, ui["net_page"] - 1)
    elif name == "page_next":
        ui["net_page"] += 1
    elif name == "change":
        ui["email_editing"], ui["email"], ui["focus"] = True, "", "email"
    elif name == "cancel_email":
        ui["email_editing"], ui["focus"] = False, None
    elif name == "claim":
        email = ui["email"].strip().lower()
        if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
            setup_mgr.set_error("That doesn't look like an email address.")
            return
        ui["email_editing"] = False
        setup_mgr.claim(email)


# ── Library view ──────────────────────────────────────────────────────────────
# Recorded-session browser + player. Only ever touched from the OpenCV main
# thread (like `ui`), except the duration cache, which a probe worker fills
# under its own lock.

recorder = recording.Recorder()

# Background auto-upload of finished recordings to the portal library (no-op
# until the device is provisioned). Skips whatever file the recorder still
# has open; poked whenever a recording stops so uploads start promptly.
uploads = uploader.Uploader(
    active_path_fn=lambda: recorder.path if recorder.active else None)

LIB_ROWS = 7  # recordings shown per page


class LibraryManager:
    def __init__(self):
        self.mode = "list"        # list | playing
        self.entries = []
        self.durations = {}
        self._dur_lock = threading.Lock()
        self.page = 0
        self.confirm_path = None  # Delete pressed once; Confirm? shown
        self.confirm_at = 0.0
        self.error = None
        # Playback state.
        self.cap = None
        self.path = None
        self.fps = recording.FPS
        self.total = 0
        self.idx = 0
        self.paused = False
        self.ended = False
        self.next_due = 0.0
        self.frame = None

    @property
    def playing(self):
        return self.mode == "playing" and not self.paused and not self.ended

    def enter(self):
        self.refresh()

    def leave(self):
        self.stop_playback()

    def refresh(self):
        self.entries = recording.list_recordings()
        self.error = recorder.error  # surface a failed writer here too
        with self._dur_lock:
            unknown = [e["path"] for e in self.entries
                       if e["path"] not in self.durations]
        if unknown:
            threading.Thread(
                target=recording.probe_durations,
                args=(unknown, self.durations, self._dur_lock),
                daemon=True).start()

    def duration(self, path):
        with self._dur_lock:
            return self.durations.get(path, "...")

    def handle(self, name):
        if name.startswith("play:"):
            i = int(name.split(":", 1)[1])
            if 0 <= i < len(self.entries):
                self.start_playback(self.entries[i]["path"])
        elif name.startswith("del:"):
            i = int(name.split(":", 1)[1])
            if 0 <= i < len(self.entries):
                self.confirm_path = self.entries[i]["path"]
                self.confirm_at = time.monotonic()
        elif name == "confirm_del":
            if self.confirm_path:
                if not recording.delete_recording(self.confirm_path):
                    self.error = "Could not delete the recording."
                self.confirm_path = None
                self.refresh()
        elif name == "refresh":
            self.refresh()
        elif name == "page_prev":
            self.page = max(0, self.page - 1)
        elif name == "page_next":
            self.page += 1
        elif name == "pause":
            self.paused = not self.paused
            if not self.paused:
                self.next_due = time.monotonic()
        elif name == "replay":
            if self.path:
                self.start_playback(self.path)
        elif name == "back":
            self.stop_playback()
            self.refresh()

    def start_playback(self, path):
        self.stop_playback()
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            self.error = "Cannot open %s" % os.path.basename(path)
            return
        self.cap, self.path = cap, path
        self.fps = cap.get(cv.CAP_PROP_FPS) or recording.FPS
        self.total = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
        self.idx, self.paused, self.ended = 0, False, False
        self.frame, self.next_due = None, time.monotonic()
        self.mode = "playing"

    def stop_playback(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.mode, self.paused, self.ended = "list", False, False

    def tick(self):
        '''Advance playback when the next frame is due (called every render).'''
        if not self.playing or self.cap is None:
            return
        now = time.monotonic()
        if now < self.next_due:
            return
        ok, f = self.cap.read()
        if not ok:
            self.ended = True
            return
        self.frame = f
        self.idx += 1
        # Never let the schedule fall far behind wall time (long stalls).
        self.next_due = max(self.next_due + 1.0 / self.fps, now - 0.25)


library = LibraryManager()


def _fmt_when(mtime):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))


def render_library_view(W, H):
    '''Render the Library view (list or playback). Returns (canvas, buttons).'''
    library.tick()
    img = np.full((H, W, 3), C_BG, np.uint8)
    buttons = []
    m, gap, bh = 24, 8, 38
    font = cv.FONT_HERSHEY_SIMPLEX

    def text(txt, x, yy, scale=0.55, color=C_TEXT):
        cv.putText(img, txt, (x, yy), font, scale, color, 1, cv.LINE_AA)

    if library.mode == "playing":
        ctrl_h = bh + 18
        area_h = H - ctrl_h - 40
        f = library.frame
        if f is not None:
            fh, fw = f.shape[:2]
            scale = min((W - 2 * m) / float(fw), area_h / float(fh))
            tw, th = max(1, int(fw * scale)), max(1, int(fh * scale))
            x0, y0 = (W - tw) // 2, 8 + (area_h - th) // 2
            img[y0:y0 + th, x0:x0 + tw] = cv.resize(f, (tw, th))
        if library.ended:
            text("Ended", W // 2 - 30, 40, 0.7, C_ON_TX)
        # Progress bar + time.
        py = H - ctrl_h - 24
        cv.rectangle(img, (m, py), (W - m, py + 6), C_BTN_BD, -1)
        if library.total > 0:
            frac = min(1.0, library.idx / float(library.total))
            cv.rectangle(img, (m, py), (m + int((W - 2 * m) * frac), py + 6),
                         C_ON_BD, -1)
        cur = int(library.idx / library.fps) if library.fps else 0
        tot = int(library.total / library.fps) if library.fps else 0
        text("%d:%02d / %d:%02d" % (cur // 60, cur % 60, tot // 60, tot % 60),
             W - m - 110, py - 8, 0.5, C_SUBTEXT)
        # Controls.
        y = H - ctrl_h + 4
        for label, name, w_ in ((("Resume" if library.paused else "Pause"),
                                 "pause", 110),
                                ("Replay", "replay", 110),
                                ("Back to list", "back", 140)):
            draw_button(img, m, y, w_, bh, label, None, name == "pause" and library.paused)
            buttons.append((m, y, w_, bh, "lib", name))
            m += w_ + gap
        return img, buttons

    # ── list mode ──
    y = 34
    text("Library - recorded sessions", m, y, 0.7)
    draw_button(img, W - m - 110, y - 24, 110, 34, "Refresh", None, False)
    buttons.append((W - m - 110, y - 24, 110, 34, "lib", "refresh"))
    y += 20

    if not library.entries:
        text("No recordings yet - press Record on the Fluoro tab.",
             m, y + 40, 0.6, C_SUBTEXT)
    # Auto-cancel a pending delete confirmation after 5 s.
    if library.confirm_path and time.monotonic() - library.confirm_at > 5.0:
        library.confirm_path = None

    page = min(library.page, max(0, (len(library.entries) - 1) // LIB_ROWS))
    library.page = page
    row_h = 44
    for i in range(page * LIB_ROWS,
                   min(len(library.entries), (page + 1) * LIB_ROWS)):
        e = library.entries[i]
        rounded_rect(img, m, y, W - 2 * m, row_h - 6, 8, C_BTN_BG, -1)
        rounded_rect(img, m, y, W - 2 * m, row_h - 6, 8, C_BTN_BD, 1)
        # Upload status: green check = in the portal library, ring = upload
        # in progress (percent sent inside once measurable, spinner during
        # the transcode), amber dot = queued/retrying, red X = failed for
        # good, grey dot = local only (unprovisioned).
        st = uploads.status_for(e["name"])
        cx, cy = m + 15, y + (row_h - 6) // 2
        if st == "uploaded":
            cv.line(img, (cx - 6, cy), (cx - 2, cy + 4), C_DOT_ON, 2, cv.LINE_AA)
            cv.line(img, (cx - 2, cy + 4), (cx + 6, cy - 5), C_DOT_ON, 2, cv.LINE_AA)
        elif st == "working":
            frac = uploads.progress_for(e["name"])
            if frac is None:
                a0 = int(time.monotonic() * 300) % 360
                cv.ellipse(img, (cx, cy), (8, 8), 0, a0, a0 + 270,
                           C_DOT_WAIT, 2, cv.LINE_AA)
            else:
                cv.ellipse(img, (cx, cy), (8, 8), 0, 0, 360,
                           C_BTN_BD, 2, cv.LINE_AA)
                cv.ellipse(img, (cx, cy), (8, 8), -90, 0, int(360 * frac),
                           C_DOT_ON, 2, cv.LINE_AA)
                pct = "%d" % int(frac * 100)
                (pw, ph), _ = cv.getTextSize(pct, font, 0.3, 1)
                cv.putText(img, pct, (cx - pw // 2, cy + ph // 2), font, 0.3,
                           C_TEXT, 1, cv.LINE_AA)
        elif st == "failed":
            cv.line(img, (cx - 5, cy - 5), (cx + 5, cy + 5), C_DOT_OFF, 2, cv.LINE_AA)
            cv.line(img, (cx - 5, cy + 5), (cx + 5, cy - 5), C_DOT_OFF, 2, cv.LINE_AA)
        elif st in ("queued", "retrying"):
            cv.circle(img, (cx, cy), 5, C_DOT_WAIT, -1, cv.LINE_AA)
        else:
            cv.circle(img, (cx, cy), 5, C_BTN_BD, -1, cv.LINE_AA)
        # Columns are laid out right-to-left from the buttons' edge so the
        # date and duration can never slide underneath Play/Delete; whatever
        # width is left after them belongs to the (truncated) file name.
        bx = W - m - 200
        date_x, dur_x = bx - 178, bx - 52
        text(_fmt_when(e["mtime"]), date_x, y + 25, 0.45, C_SUBTEXT)
        text(library.duration(e["path"]), dur_x, y + 25, 0.45, C_SUBTEXT)
        name = e["name"]
        name_w = date_x - (m + 28) - 10
        while len(name) > 4 and cv.getTextSize(name, font, 0.5, 1)[0][0] > name_w:
            name = name[:-2] + "~"
        text(name, m + 28, y + 25, 0.5)
        draw_button(img, bx, y + 4, 80, row_h - 14, "Play", None, False)
        buttons.append((bx, y + 4, 80, row_h - 14, "lib", "play:%d" % i))
        bx += 80 + gap
        if library.confirm_path == e["path"]:
            draw_button(img, bx, y + 4, 100, row_h - 14, "Confirm?", None, False, "quit")
            buttons.append((bx, y + 4, 100, row_h - 14, "lib", "confirm_del"))
        else:
            draw_button(img, bx, y + 4, 100, row_h - 14, "Delete", None, False)
            buttons.append((bx, y + 4, 100, row_h - 14, "lib", "del:%d" % i))
        y += row_h

    if len(library.entries) > LIB_ROWS:
        if page > 0:
            draw_button(img, m, y, 90, 30, "Prev", None, False)
            buttons.append((m, y, 90, 30, "lib", "page_prev"))
        if (page + 1) * LIB_ROWS < len(library.entries):
            draw_button(img, m + 98, y, 90, 30, "Next", None, False)
            buttons.append((m + 98, y, 90, 30, "lib", "page_next"))
        y += 38
    summary = uploads.snapshot()["summary"]
    if summary:
        text(str(summary)[:90], m, y + 20, 0.45, C_SUBTEXT)
        y += 18
    if library.error:
        text(str(library.error)[:90], m, y + 24, 0.5, C_QUIT_TX)
    return img, buttons


def apply_button(kind, name):
    '''Apply an on-screen button press to the shared `state` (under state_lock).'''
    if kind == "tab":
        with state_lock:
            state["ui_view"] = name
        on_view_enter(name)
        return
    if kind == "focus":
        ui["focus"] = name
        return
    if kind in ("setup", "lib"):
        # Remote Access / Library view buttons — handled outside state_lock
        # (they talk to their own managers, which have their own locks).
        handle_view_button(kind, name)
        return
    with state_lock:
        if kind == "toggle":
            state[name] = not state[name]
        elif kind == "pan":
            if name == "xm":
                state["pos_x_cm"] -= PAN_STEP_CM
            elif name == "xp":
                state["pos_x_cm"] += PAN_STEP_CM
            elif name == "ym":
                state["pos_y_cm"] -= PAN_STEP_CM
            elif name == "yp":
                state["pos_y_cm"] += PAN_STEP_CM
            elif name == "zm":
                state["pos_z_cm"] -= PAN_STEP_CM
            elif name == "zp":
                state["pos_z_cm"] += PAN_STEP_CM
        elif name == "fullscreen":
            state["fullscreen"] = True
        elif name == "windowed":
            state["fullscreen"] = False
        elif name in ("stop", "quit"):
            # Standby, not process exit: the web server stays up so the panel's
            # Start button (or the desktop shortcut) can bring the sim back.
            state["running"] = False


def _hit(buttons, x, y):
    '''Return the (kind, name) of the button containing (x, y), or None.'''
    for (bx, by, bw, bh, kind, name) in buttons:
        if bx <= x < bx + bw and by <= y < by + bh:
            return kind, name
    return None


def on_mouse_controls(event, x, y, flags, param):
    '''Click handler for the separate CONTROLS window (windowed mode).'''
    if event == cv.EVENT_LBUTTONDOWN:
        hit = _hit(ctrl_buttons, x, y)
        if hit:
            apply_button(*hit)


def on_mouse_fluoro(event, x, y, flags, param):
    '''Click handler for the FLUORO window: tab bar first, then the active
    view's buttons (the fluoro control bar, or the Remote Access / Library UI).'''
    if event == cv.EVENT_LBUTTONDOWN:
        hit = _hit(tab_buttons, x, y)
        if hit is None:
            with state_lock:
                view = state["ui_view"]
            hit = _hit(overlay_buttons if view == "fluoro" else view_buttons, x, y)
        if hit:
            apply_button(*hit)


def handle_ui_key(key):
    '''Route a keypress to the tab UI. Returns True when consumed.

    On the Fluoro view nothing is consumed (shortcuts work as always). On the
    other views a focused text field captures typing (so "wifi2024!" cannot pan
    the anatomy); with no focus, ESC returns to Fluoro and 'b' passes through
    so a foot pedal keeps working on any tab.
    '''
    with state_lock:
        view = state["ui_view"]
    if view == "fluoro":
        return False
    focus = ui["focus"]
    if focus:
        if 32 <= key <= 126:
            ui[focus] += chr(key)
        elif key in (8, 127):
            ui[focus] = ui[focus][:-1]
        elif key in (10, 13):
            ui["focus"] = None
            submit_field(focus)
        elif key == 27:
            ui["focus"] = None
        return True
    if key == 27:
        with state_lock:
            state["ui_view"] = "fluoro"
        return True
    if key == ord('b'):
        return False
    return True


def on_view_enter(name):
    '''Hook run when a tab is selected (refresh lists/status; never blocks).'''
    if name == "remote":
        setup_mgr.refresh_status()
    elif name == "library":
        library.enter()
    elif name == "fluoro":
        library.leave()


def submit_field(field):
    '''Enter pressed in a text field — same as clicking its action button.'''
    if field == "password":
        handle_view_button("setup", "connect")
    elif field == "email":
        handle_view_button("setup", "claim")


def handle_view_button(kind, name):
    '''Dispatch a Remote Access ("setup") or Library ("lib") button press.'''
    if kind == "setup":
        handle_setup_button(name)
    elif kind == "lib":
        library.handle(name)


def run_simulation(cam_index, show_window):
    '''Main capture/process/display loop — runs on the main thread until quit.

    Each iteration: read a frame, apply the overlay composite (optional), draw
    the HUD, then both show it in the FLUORO window and JPEG-encode it into
    ``_latest_jpeg`` for the web preview. The loop
    reads the shared toggle state once per frame via get_state_snapshot(), so the
    web buttons and the keyboard shortcuts drive exactly the same behaviour.

    cam_index   : int  — V4L2 camera device index
    show_window : bool — open the on-screen FLUORO window (False = web preview only)
    '''
    global _latest_jpeg

    def open_camera():
        c = cv.VideoCapture(cam_index, cv.CAP_V4L2)
        if not c.isOpened():
            print("Warning: unable to open video source:", cam_index)
        return c

    # Placeholder frame published to the preview while the sim is in standby.
    stopped_img = np.full((480, 640), 16, np.uint8)
    draw_str(stopped_img, (200, 230), "SIMULATION STOPPED")
    draw_str(stopped_img, (150, 260), "press Start simulator on the control panel")
    ok, _stopped_jpg = cv.imencode(".jpg", stopped_img, [cv.IMWRITE_JPEG_QUALITY, 80])
    stopped_buf = _stopped_jpg.tobytes() if ok else None

    # Full-resolution "master" anatomy background, kept untouched at full res so we
    # can crop a fresh viewport out of it every frame (see compute_viewport). Falls
    # back to the static skel.jpg (shown whole, no panning) if the master is absent.
    master = cv.imread(MASTER_IMAGE)
    if master is None:
        print("Master image not found at %s — falling back to %s (no panning)"
              % (MASTER_IMAGE, OVERLAY_IMAGE))
        master = cv.imread(OVERLAY_IMAGE)
        if master is None:
            raise FileNotFoundError("Cannot load image: %s" % OVERLAY_IMAGE)
    master = cv.cvtColor(master, cv.COLOR_BGR2GRAY)
    if MASTER_BRIGHTNESS != 1.0:
        master = cv.convertScaleAbs(master, alpha=MASTER_BRIGHTNESS, beta=0)

    # On-screen control panel: the logo image and a helper to (re)open the
    # separate CONTROLS window used in windowed mode.
    logo = cv.imread(LOGO_IMAGE, cv.IMREAD_UNCHANGED) if show_window else None

    def show_controls_window():
        cv.namedWindow("CONTROLS", cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback("CONTROLS", on_mouse_controls)
        cv.moveWindow("CONTROLS", 20, 20)

    def show_fluoro_window():
        cv.namedWindow("FLUORO", cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty("FLUORO", cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)
        # Clicks on the FLUORO window only matter in fullscreen, where the control
        # bar is stacked below the video (see overlay_buttons).
        cv.setMouseCallback("FLUORO", on_mouse_fluoro)

    if show_window and logo is None:
        print("Warning: logo not found at", LOGO_IMAGE)

    # The camera and the windows are opened lazily on the first running
    # iteration (and reopened after a standby stop) — see the loop below.
    cap = None
    prev_raw = None            # previous raw gray frame (freeze stability check)
    stable_count = 0           # consecutive largely-unchanged frames so far
    frozen_bg = None           # frozen reference frame (None = watching for stability)
    frozen_mask = None         # frozen white-background mask for the composite
    applied_fullscreen = None  # last fullscreen state pushed to the window (avoids redundant calls)
    res = None                 # last processed frame (shown + streamed)
    live = False               # True once a frame has been read (drives the status dot)

    while True:
        s = get_state_snapshot()
        if s["quit"]:
            break

        # Standby: release the camera and close the windows, publish the
        # "stopped" placeholder to the preview, and idle until the web panel's
        # Start button (POST /api/action/start) sets running again. The Flask
        # server stays up throughout, so start/stop work fully remotely.
        if not s["running"]:
            if recorder.active:
                recorder.stop()
                uploads.poke()
                with state_lock:
                    state["recording"] = False
            if cap is not None:
                cap.release()
                cap = None
                if show_window:
                    library.stop_playback()
                    cv.destroyAllWindows()
                    cv.waitKey(1)  # let the GUI process the window teardown
                applied_fullscreen = None
                prev_raw, stable_count = None, 0
                frozen_bg = frozen_mask = None
                res, live = None, False
            if stopped_buf is not None:
                with _latest_lock:
                    _latest_jpeg = stopped_buf
            time.sleep(0.2)
            continue

        # (Re)open the camera and windows on the first running iteration and
        # when resuming from standby.
        if cap is None:
            cap = open_camera()
            if show_window:
                show_fluoro_window()

        # Keep the OpenCV window's fullscreen state in sync with the toggle, and
        # move the on-screen controls between the separate CONTROLS window
        # (windowed) and a bar stacked below the video (fullscreen).
        if show_window and s["fullscreen"] != applied_fullscreen:
            cv.setWindowProperty(
                "FLUORO", cv.WND_PROP_FULLSCREEN,
                cv.WINDOW_FULLSCREEN if s["fullscreen"] else cv.WINDOW_NORMAL)
            if s["fullscreen"]:
                if applied_fullscreen is not None:   # close the window if it was open
                    cv.destroyWindow("CONTROLS")
            else:
                show_controls_window()
                overlay_buttons[:] = []
            applied_fullscreen = s["fullscreen"]

        # Keyboard: the tab UI (focused text fields, ESC-back) gets first look;
        # unconsumed keys keep their existing shortcut meaning on the Fluoro view.
        key = (cv.waitKey(1) & 0xFF) if show_window else 0xFF
        consumed = handle_ui_key(key) if key != 0xFF else False
        key_pedal = not consumed and key == ord('b')
        if key != 0xFF and not consumed:
            _handle_key(key)
        if key == 27 and not consumed:  # ESC (on the Fluoro view) exits
            with state_lock:
                state["quit"] = True
            break

        # Which tab is showing (read after waitKey so a click this iteration
        # counts). While a Library recording is actually playing, skip the
        # live processing so the CPU goes to decoding — one grab() keeps the
        # camera's buffer fresh so switching back to Fluoro is instant.
        with state_lock:
            view = state["ui_view"]
        lib_playing = show_window and view == "library" and library.playing
        if lib_playing and cap is not None:
            cap.grab()

        # In pedal mode, only grab a frame while the pedal/'b' is held; otherwise
        # capture continuously.
        pedal_down = s["pedal_pressed"] or key_pedal
        capture_now = (pedal_down or not s["pedal_mode"]) and not lib_playing

        if capture_now:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            live = True

            frame_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            frame_raw = frame_gray.copy()  # untouched copy shown when the overlay is off
            # Pan the anatomy: crop the viewport at the current camera position and
            # scale it to the frame. This is the background the vasculature sits on.
            crop = compute_viewport(master, s["pos_x_cm"], s["pos_y_cm"],
                                    s["pos_z_cm"])
            overlay_frame = cv.resize(crop, (frame_gray.shape[1], frame_gray.shape[0]))

            # Freeze the white-background removal once the scene has been steady
            # for STABLE_FRAMES frames; thaw it when the frame moves significantly
            # away from the frozen reference (pan, lighting change, ...).
            step = (cv.norm(frame_gray, prev_raw, cv.NORM_L1) / frame_gray.size
                    if prev_raw is not None else 0.0)
            if frozen_bg is None:
                if prev_raw is not None:
                    stable_count = stable_count + 1 if step < STABLE_DIFF else 0
                if stable_count >= STABLE_FRAMES:
                    # Steady scene: freeze the composite's white-background mask
                    # on the current frame so the identical removal is applied
                    # every frame (thresholding each live frame flips pixels near
                    # MASK_THRESHOLD with sensor noise; see composite_overlay).
                    frozen_bg = frame_gray.copy()
                    frozen_mask = white_bg_mask(frame_gray)
            elif cv.norm(frame_gray, frozen_bg, cv.NORM_L1) / frame_gray.size > CHANGE_DIFF:
                frozen_bg = None
                frozen_mask = None
                stable_count = 0
            prev_raw = frame_gray

            # Overlay off => show the full raw video (bright/white areas intact).
            if s["overlay"]:
                res = composite_overlay(frame_gray, overlay_frame, s["equalize"],
                                        frozen_mask)
            else:
                res = frame_raw

            if s["hud"]:
                draw_str(res, (20, 20), "Overlay:%s  Equalize:%s" %
                         (s["overlay"], s["equalize"]))
                draw_str(res, (20, 40), "Pedal mode:%s  HUD:%s" % (s["pedal_mode"], s["hud"]))
                draw_str(res, (20, 60), "Cam: x=%.1f  y=%.1f  z=%.1f cm" %
                         (s["pos_x_cm"], s["pos_y_cm"], s["pos_z_cm"]))
            if pedal_down:
                draw_str(res, (20, 80), "PEDAL ACTIVE")

        # Session recording: feed the processed frame to the recorder while
        # the toggle is on; a writer that fails to open flips the toggle back
        # off and surfaces its error on the Library tab.
        if s["recording"] and res is not None:
            if recorder.ensure_started(res.shape):
                if capture_now:
                    recorder.submit(res)
            else:
                with state_lock:
                    state["recording"] = False
                library.error = recorder.error
        elif not s["recording"] and recorder.active:
            recorder.stop()
            uploads.poke()

        rec_elapsed = recorder.elapsed_str() if recorder.active else None

        if show_window and view != "fluoro":
            # Remote Access / Library replace the video area under the tab
            # bar; the capture pipeline above keeps running so the preview
            # stays live and switching back is instant. A failure in tab code
            # must never take down the sim: fall back to the Fluoro view.
            try:
                fh, fw = res.shape[:2] if res is not None else (480, 640)
                canvas_h = fh + (BAR_H if s["fullscreen"] else 0)
                if view == "remote":
                    canvas, vbtns = render_remote_view(s, fw, canvas_h)
                else:
                    canvas, vbtns = render_library_view(fw, canvas_h)
                tab, tbtns = render_tab_bar(fw, view, rec_elapsed)
                tab_buttons[:] = tbtns
                view_buttons[:] = [(x, y + TAB_H, w, h, k, n)
                                   for (x, y, w, h, k, n) in vbtns]
                if not s["fullscreen"]:
                    panel, btns = render_controls(s, logo, live)
                    ctrl_buttons[:] = btns
                    cv.imshow("CONTROLS", panel)
                cv.imshow("FLUORO", np.vstack([tab, canvas]))
            except Exception:
                traceback.print_exc()
                with state_lock:
                    state["ui_view"] = "fluoro"
        elif res is not None and show_window:
            vid = cv.cvtColor(res, cv.COLOR_GRAY2BGR) if res.ndim == 2 else res
            tab, tbtns = render_tab_bar(vid.shape[1], view, rec_elapsed)
            tab_buttons[:] = tbtns
            if s["fullscreen"]:
                # Stack the video between the tab bar and a thin control bar.
                bar, bbtns = render_control_bar(s, vid.shape[1], live)
                composite = np.vstack([tab, vid, bar])
                overlay_buttons[:] = [(x, y + TAB_H + vid.shape[0], w, h, k, n)
                                      for (x, y, w, h, k, n) in bbtns]
                cv.imshow("FLUORO", composite)
            else:
                # Tabbed video in FLUORO, the vertical panel in CONTROLS.
                panel, btns = render_controls(s, logo, live)
                ctrl_buttons[:] = btns
                cv.imshow("CONTROLS", panel)
                cv.imshow("FLUORO", np.vstack([tab, vid]))
        if res is not None:
            # The web MJPEG preview always streams the clean frame (the browser has
            # its own HTML buttons), so encode `res`, not the composited view.
            ok, jpg = cv.imencode(".jpg", res, [cv.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with _latest_lock:
                    _latest_jpeg = jpg.tobytes()

    recorder.stop(wait=True)  # finalize the .avi before os._exit
    library.stop_playback()
    if cap is not None:
        cap.release()
    cv.destroyAllWindows()
    # Stop the process so the Flask daemon thread exits too.
    os._exit(0)


def _handle_key(key):
    '''Map FLUORO-window keypresses onto the shared state (keeps parity with the CLI).'''
    with state_lock:
        if key == ord('2') or key == ord('5'):
            state["overlay"] = not state["overlay"]
        elif key == ord('3'):
            state["fullscreen"] = True
        elif key == ord('4'):
            state["fullscreen"] = False
        elif key == ord('6'):
            state["equalize"] = not state["equalize"]
        elif key == ord('7'):
            state["hud"] = not state["hud"]
        elif key == ord(' '):
            state["pedal_mode"] = not state["pedal_mode"]
        elif key == ord('a'):            # pan the anatomy background: X−
            state["pos_x_cm"] -= PAN_STEP_CM
        elif key == ord('d'):            # X+
            state["pos_x_cm"] += PAN_STEP_CM
        elif key == ord('w'):            # Y−
            state["pos_y_cm"] -= PAN_STEP_CM
        elif key == ord('s'):            # Y+
            state["pos_y_cm"] += PAN_STEP_CM
        elif key == ord('q'):            # Z− (lower the camera = zoom in)
            state["pos_z_cm"] -= PAN_STEP_CM
        elif key == ord('e'):            # Z+ (raise the camera = zoom out)
            state["pos_z_cm"] += PAN_STEP_CM


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(__doc__)

    args = sys.argv[1:]
    port = 5000
    host = "0.0.0.0"
    show_window = True
    cam_index = 0
    force_http = False

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--port":
            port = int(args[i + 1]); i += 2; continue
        if a == "--host":          # bind address; 127.0.0.1 = local/tunnel only
            host = args[i + 1]; i += 2; continue
        if a == "--no-window":
            show_window = False; i += 1; continue
        if a == "--http":          # force plain HTTP even if a cert is present
            force_http = True; i += 1; continue
        cam_index = int(a); i += 1

    # Serve HTTPS with the self-signed cert if cert.pem/key.pem are present next to
    # this script (so browsers that force secure connections can reach the panel).
    # Falls back to plain HTTP if the cert is missing or --http is given.
    cert = os.path.join(BASE_DIR, "cert.pem")
    key = os.path.join(BASE_DIR, "key.pem")
    use_https = (not force_http) and os.path.exists(cert) and os.path.exists(key)
    ssl_context = (cert, key) if use_https else None
    scheme = "https" if use_https else "http"

    # Flask in a daemon thread; the simulation owns the main thread (OpenCV GUI rule).
    flask_thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True,
                               debug=False, use_reloader=False,
                               ssl_context=ssl_context),
        daemon=True)
    flask_thread.start()
    uploads.start()   # auto-upload finished recordings to the portal library
    print("Control panel:  %s://<this-machine-ip>:%d/" % (scheme, port))
    if use_https:
        print("(self-signed cert — your browser will show a one-time "
              "'not private' warning to click through)")

    run_simulation(cam_index, show_window)
