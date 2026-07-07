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

    Subtraction, Overlay, Equalize, Pedal mode, Pedal press, HUD,
    plus Fullscreen / Windowed / Quit actions and a live preview.

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
    python fluoro_web.py [<video device number>] [--port 5000] [--no-window] [--http]

Then open  https://<this-machine-ip>:<port>/  in a browser (http:// without a cert).
'''

from __future__ import print_function

import os
import sys
import time
import threading

# Force OpenCV's Qt GUI onto the X11/XWayland backend. Under the native Wayland
# Qt backend, cv.setWindowProperty(FULLSCREEN) is a no-op (it logs
# "qt.qpa.wayland: Wayland does not support QWindow::requestActivate()") so the
# Fullscreen/Windowed controls can't change the FLUORO window. Set before cv2 is
# imported; respect an explicit override if the user already set one.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
import cv2 as cv
from flask import Flask, Response, jsonify, render_template_string, request

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OVERLAY_IMAGE = os.path.join(BASE_DIR, "skel.jpg")
LOGO_IMAGE = os.path.join(BASE_DIR, "static", "logosign_white.png")

# Brightness threshold above which a pixel is treated as bright "white background"
# and composited mostly from the overlay. (Mirrors fluoro_simulator (3).py.)
MASK_THRESHOLD = 220
# Background-model learning rate for cv.accumulateWeighted.
ALPHA = 0.05

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
# How far each on-screen nudge button / WASD keypress moves the camera position (cm).
PAN_STEP_CM = 5.0

# ── Shared state ────────────────────────────────────────────────────────────────
# All values touched by both the processing loop and the web handlers live here,
# guarded by `state_lock`. Booleans mirror the keyboard toggles of the original.
state_lock = threading.Lock()
state = {
    "subtract": True,       # (1) background subtraction + inversion
    "overlay": True,        # (2/5) anatomy overlay; off => full raw video
    "equalize": True,       # (6) CLAHE histogram equalisation
    "pedal_mode": False,    # (space) only capture while the pedal is pressed
    "pedal_pressed": False, # web stand-in for holding the foot pedal / 'b' key
    "hud": True,            # (7) on-screen text HUD
    "fullscreen": True,     # (3/4) FLUORO window fullscreen vs. windowed
    "quit": False,          # set by the web Quit button to stop the loop
    "pos_x_cm": 0.0,        # camera X position (cm from origin) — drives the viewport
    "pos_y_cm": 0.0,        # camera Y position (cm from origin) — drives the viewport
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
<title>FluoroSim Controls</title>
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
      grid-template-columns: repeat(6, 1fr); margin-top: 0; }
  .stage:fullscreen .panel .actions, .stage:-webkit-full-screen .panel .actions {
      grid-template-columns: repeat(3, 1fr); margin-top: 8px; }
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
  button.toggle .st { display: block; font-size: 12px; font-weight: 500; opacity: .7; margin-top: 2px; }
  .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
  button.quit { grid-column: 1 / -1; background: #2a1416; border-color: #5b2327; color: #ff9a9a; }
  .hint { color: #7d8a99; font-size: 12px; margin: 14px 2px 0; }
  .posctl { display: flex; gap: 10px; align-items: center; margin-top: 16px; flex-wrap: wrap; }
  .posctl label { display: flex; align-items: center; gap: 6px; color: #7d8a99; font-size: 14px; }
  .posctl input { width: 90px; font-size: 16px; padding: 12px 10px; border-radius: 10px;
                  border: 1px solid #26323f; background: #131a22; color: #e7edf3; }
  .posctl button { padding: 12px 18px; }
</style>
</head>
<body>
<header>
  <img class="logo" src="/static/logosign_white.png" alt="FluoroSim logo">
  <div class="titlerow">
    <span id="dot" class="dot"></span>
    <h1>FluoroSim Controls</h1>
  </div>
</header>
<main>
  <div class="stage" id="stage">
    <div class="preview"><img id="feed" src="/video_feed" alt="live preview"></div>

    <div class="panel">
      <div class="grid" id="toggles">
        <button class="toggle" data-toggle="subtract">Subtraction<span class="st">—</span></button>
        <button class="toggle" data-toggle="overlay">Overlay<span class="st">—</span></button>
        <button class="toggle" data-toggle="equalize">Equalize<span class="st">—</span></button>
        <button class="toggle" data-toggle="hud">HUD<span class="st">—</span></button>
        <button class="toggle" data-toggle="pedal_mode">Pedal mode<span class="st">—</span></button>
        <button class="toggle" data-toggle="pedal_pressed">Pedal press<span class="st">—</span></button>
      </div>

      <div class="actions">
        <button class="fsbtn" data-action="fullscreen" data-fs="1">Fullscreen</button>
        <button class="fsbtn" data-action="windowed" data-fs="0">Windowed</button>
        <button class="quit" data-action="quit">Quit simulator</button>
      </div>
    </div>
  </div>

  <div class="posctl">
    <label>X <input id="xin" type="number" step="0.5" value="0"> cm</label>
    <label>Y <input id="yin" type="number" step="0.5" value="0"> cm</label>
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
  var pos = document.getElementById('pos');
  if (pos) pos.textContent = 'Camera: x=' + (+s.pos_x_cm || 0).toFixed(1) +
                             '  y=' + (+s.pos_y_cm || 0).toFixed(1) + ' cm';
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
    if (b.dataset.action === 'quit' && !confirm('Stop the simulator?')) return;
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
  fetch('/api/position', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x: x, y: y }) })
    .then(function (r) { return r.json(); }).then(applyState).catch(function () {});
}
document.getElementById('gopos').addEventListener('click', sendPos);
['xin', 'yin'].forEach(function (id) {
  var el = document.getElementById(id);
  el.addEventListener('change', sendPos);                 // fires on blur / stepper arrows
  el.addEventListener('keydown', function (e) { if (e.key === 'Enter') sendPos(); });
});
var feed = document.getElementById('feed');
feed.addEventListener('load', function () { document.getElementById('dot').classList.add('live'); });
feed.addEventListener('error', function () { document.getElementById('dot').classList.remove('live'); });
refresh();
setInterval(refresh, 1500);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    '''Serve the single-page control panel.'''
    return render_template_string(PAGE)


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
        if name in ("subtract", "overlay", "equalize", "pedal_mode", "pedal_pressed", "hud"):
            state[name] = not state[name]
        snap = dict(state)
    return jsonify(snap)


@app.route("/api/action/<action>", methods=["POST"])
def api_action(action):
    '''Apply a one-shot action (fullscreen / windowed / quit) and return the state.'''
    with state_lock:
        if action == "fullscreen":
            state["fullscreen"] = True
        elif action == "windowed":
            state["fullscreen"] = False
        elif action == "quit":
            state["quit"] = True
        snap = dict(state)
    return jsonify(snap)


@app.route("/api/position", methods=["POST"])
def api_position():
    '''Set the current camera position, in centimetres from the origin.

    This is the single input hook for the motion source: the position provider
    POSTs the camera's (x, y) here and the processing loop pans the anatomy
    viewport to match. Accepts JSON ``{"x": <cm>, "y": <cm>}`` or plain form/query
    params ``x`` and ``y``. Returns the full updated state.
    '''
    data = request.get_json(silent=True) or {}
    x = data.get("x", request.values.get("x"))
    y = data.get("y", request.values.get("y"))
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return jsonify({"error": "provide numeric x and y (cm)"}), 400
    with state_lock:
        state["pos_x_cm"] = x
        state["pos_y_cm"] = y
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
def compute_viewport(master, x_cm, y_cm):
    '''Crop the anatomy viewport out of the full-res master at a camera position.

    Maps the camera's (x_cm, y_cm) — centimetres from the origin — to a pixel
    window in ``master`` using the calibration constants, centred on that point.
    The window is clamped to lie fully inside the image, so reaching the edge of
    travel pans up against the border instead of returning a truncated crop (which
    would otherwise distort when resized to the frame).

    Returns a view into ``master`` (no copy); the caller resizes it to the frame.
    '''
    mh, mw = master.shape[:2]
    w = max(1, min(int(round(FOV_CM[0] * PX_PER_CM)), mw))
    h = max(1, min(int(round(FOV_CM[1] * PX_PER_CM)), mh))
    cx = ORIGIN_PX[0] + FLIP_X * x_cm * PX_PER_CM
    cy = ORIGIN_PX[1] + FLIP_Y * y_cm * PX_PER_CM
    x0 = max(0, min(int(round(cx - w / 2.0)), mw - w))
    y0 = max(0, min(int(round(cy - h / 2.0)), mh - h))
    return master[y0:y0 + h, x0:x0 + w]


def composite_overlay(gray, overlay, equalize):
    '''Composite the anatomy overlay onto a (subtracted) grayscale frame.

    Mirrors the blend weights in fluoro_simulator (3).py: bright/white areas are
    30% video / 70% overlay; vasculature areas are 60% video / 40% overlay.
    '''
    _, bg_mask = cv.threshold(gray, MASK_THRESHOLD, 255, cv.THRESH_BINARY_INV)
    bg_mask = cv.medianBlur(bg_mask, 5)

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

# Button hit-boxes (x, y, w, h, kind, name), rebuilt every render. `ctrl_buttons`
# are positions inside the separate CONTROLS window (windowed); `overlay_buttons`
# are positions on the FLUORO frame when the bar is stacked below it (fullscreen).
ctrl_buttons = []
overlay_buttons = []


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

    title = "FluoroSim Controls"
    font = cv.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv.getTextSize(title, font, 0.6, 1)
    tx0 = (W - (tw + 18)) // 2
    cv.circle(img, (tx0 + 5, y + 9), 5, C_DOT_ON if live else C_DOT_OFF, -1, cv.LINE_AA)
    cv.putText(img, title, (tx0 + 18, y + 9 + th // 2), font, 0.6, C_TEXT, 1, cv.LINE_AA)
    y += 26 + 14

    cw = (W - 2 * m - gap) // 2
    toggles = [("Subtraction", "subtract"), ("Overlay", "overlay"),
               ("Equalize", "equalize"), ("HUD", "hud"),
               ("Pedal mode", "pedal_mode"), ("Pedal press", "pedal_pressed")]
    for i, (label, key) in enumerate(toggles):
        col, row = i % 2, i // 2
        bx = m + col * (cw + gap)
        by = y + row * (bt + gap)
        on = bool(s[key])
        draw_button(img, bx, by, cw, bt, label, "ON" if on else "OFF", on)
        buttons.append((bx, by, cw, bt, "toggle", key))
    y += 3 * (bt + gap)

    # Background-location readout + nudge buttons (X−/X+/Y−/Y+).
    cap = "Background   X %.0f   Y %.0f cm" % (s["pos_x_cm"], s["pos_y_cm"])
    (capw, caph), _ = cv.getTextSize(cap, font, 0.5, 1)
    cv.putText(img, cap, ((W - capw) // 2, y + caph + 2), font, 0.5, C_SUBTEXT, 1, cv.LINE_AA)
    y += cap_h + gap
    pw = (W - 2 * m - 3 * gap) // 4
    for i, (label, name) in enumerate([("X-", "xm"), ("X+", "xp"), ("Y-", "ym"), ("Y+", "yp")]):
        bx = m + i * (pw + gap)
        draw_button(img, bx, y, pw, ba, label, None, False)
        buttons.append((bx, y, pw, ba, "pan", name))
    y += ba + gap

    draw_button(img, m, y, cw, ba, "Fullscreen", None, s["fullscreen"])
    buttons.append((m, y, cw, ba, "action", "fullscreen"))
    draw_button(img, m + cw + gap, y, cw, ba, "Windowed", None, not s["fullscreen"])
    buttons.append((m + cw + gap, y, cw, ba, "action", "windowed"))
    y += ba + gap

    draw_button(img, m, y, W - 2 * m, ba, "Quit simulator", None, False, "quit")
    buttons.append((m, y, W - 2 * m, ba, "action", "quit"))
    return img, buttons


def render_control_bar(s, width, live):
    '''Render a short, full-width control strip stacked below the video in fullscreen.

    Returns (image, buttons) with hit-boxes relative to the bar's top-left.
    '''
    pad, gap, bh = 6, 6, 34
    bar_h = pad + bh + gap + bh + gap + bh + pad   # 3 rows: toggles, actions, location
    img = np.full((bar_h, width, 3), C_BG, np.uint8)
    cv.line(img, (0, 0), (width, 0), C_BTN_BD, 1, cv.LINE_AA)
    buttons = []

    toggles = [("Subtraction", "subtract"), ("Overlay", "overlay"),
               ("Equalize", "equalize"), ("HUD", "hud"),
               ("Pedal mode", "pedal_mode"), ("Pedal press", "pedal_pressed")]
    cw = (width - 2 * pad - 5 * gap) // 6
    y = pad
    for i, (label, key) in enumerate(toggles):
        bx = pad + i * (cw + gap)
        draw_button(img, bx, y, cw, bh, label, None, bool(s[key]))
        buttons.append((bx, y, cw, bh, "toggle", key))

    y += bh + gap
    aw = (width - 2 * pad - 2 * gap) // 3
    draw_button(img, pad, y, aw, bh, "Fullscreen", None, s["fullscreen"])
    buttons.append((pad, y, aw, bh, "action", "fullscreen"))
    draw_button(img, pad + aw + gap, y, aw, bh, "Windowed", None, not s["fullscreen"])
    buttons.append((pad + aw + gap, y, aw, bh, "action", "windowed"))
    draw_button(img, pad + 2 * (aw + gap), y, aw, bh, "Quit simulator", None, False, "quit")
    buttons.append((pad + 2 * (aw + gap), y, aw, bh, "action", "quit"))

    # Row 3: background-location readout + nudge buttons (X−/X+/Y−/Y+).
    y += bh + gap
    cap = "Bg  X %.0f  Y %.0f cm" % (s["pos_x_cm"], s["pos_y_cm"])
    cap_w = 190
    (_, caph), _ = cv.getTextSize(cap, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv.putText(img, cap, (pad, y + (bh + caph) // 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, C_SUBTEXT, 1, cv.LINE_AA)
    pw = (width - 2 * pad - cap_w - 3 * gap) // 4
    x = pad + cap_w
    for (label, name) in [("X-", "xm"), ("X+", "xp"), ("Y-", "ym"), ("Y+", "yp")]:
        draw_button(img, x, y, pw, bh, label, None, False)
        buttons.append((x, y, pw, bh, "pan", name))
        x += pw + gap
    return img, buttons


def apply_button(kind, name):
    '''Apply an on-screen button press to the shared `state` (under state_lock).'''
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
        elif name == "fullscreen":
            state["fullscreen"] = True
        elif name == "windowed":
            state["fullscreen"] = False
        elif name == "quit":
            state["quit"] = True


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
    '''Click handler for buttons stacked below the video (fullscreen mode).'''
    if event == cv.EVENT_LBUTTONDOWN:
        hit = _hit(overlay_buttons, x, y)
        if hit:
            apply_button(*hit)


def run_simulation(cam_index, show_window):
    '''Main capture/process/display loop — runs on the main thread until quit.

    Each iteration: read a frame, apply background subtraction (optional) and the
    overlay composite (optional), draw the HUD, then both show it in the FLUORO
    window and JPEG-encode it into ``_latest_jpeg`` for the web preview. The loop
    reads the shared toggle state once per frame via get_state_snapshot(), so the
    web buttons and the keyboard shortcuts drive exactly the same behaviour.

    cam_index   : int  — V4L2 camera device index
    show_window : bool — open the on-screen FLUORO window (False = web preview only)
    '''
    global _latest_jpeg

    cap = cv.VideoCapture(cam_index, cv.CAP_V4L2)
    if not cap.isOpened():
        print("Warning: unable to open video source:", cam_index)

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

    # On-screen control panel: the logo image and a helper to (re)open the
    # separate CONTROLS window used in windowed mode.
    logo = cv.imread(LOGO_IMAGE, cv.IMREAD_UNCHANGED) if show_window else None

    def show_controls_window():
        cv.namedWindow("CONTROLS", cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback("CONTROLS", on_mouse_controls)
        cv.moveWindow("CONTROLS", 20, 20)

    if show_window:
        cv.namedWindow("FLUORO", cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty("FLUORO", cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)
        # Clicks on the FLUORO window only matter in fullscreen, where the control
        # bar is stacked below the video (see overlay_buttons).
        cv.setMouseCallback("FLUORO", on_mouse_fluoro)
        if logo is None:
            print("Warning: logo not found at", LOGO_IMAGE)

    background = None          # float32 running-average background model
    applied_fullscreen = None  # last fullscreen state pushed to the window (avoids redundant calls)
    res = None                 # last processed frame (shown + streamed)
    live = False               # True once a frame has been read (drives the status dot)

    while True:
        s = get_state_snapshot()
        if s["quit"]:
            break

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

        # Keyboard shortcuts still work on the FLUORO window.
        key = (cv.waitKey(1) & 0xFF) if show_window else 0xFF
        key_pedal = key == ord('b')
        if key != 0xFF:
            _handle_key(key)
        if key == 27:  # ESC
            with state_lock:
                state["quit"] = True
            break

        # In pedal mode, only grab a frame while the pedal/'b' is held; otherwise
        # capture continuously.
        pedal_down = s["pedal_pressed"] or key_pedal
        capture_now = pedal_down or not s["pedal_mode"]

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
            crop = compute_viewport(master, s["pos_x_cm"], s["pos_y_cm"])
            overlay_frame = cv.resize(crop, (frame_gray.shape[1], frame_gray.shape[0]))

            # Seed / read the running-background model used for subtraction.
            if background is None:
                background = frame_gray.astype("float")
            bg_uint8 = cv.convertScaleAbs(background)

            if s["subtract"]:
                # Difference vs. background, stretch to full range, then invert so
                # static background reads white and moving structures read dark.
                frame_gray = cv.absdiff(frame_gray, bg_uint8)
                cv.normalize(frame_gray, frame_gray, 0, 255, cv.NORM_MINMAX)
                frame_gray = cv.bitwise_not(frame_gray)

            # Slowly adapt the background model to gradual lighting changes.
            cv.accumulateWeighted(frame_gray, background, ALPHA)

            # Overlay off => show the full raw video (bright/white areas intact).
            if s["overlay"]:
                res = composite_overlay(frame_gray, overlay_frame, s["equalize"])
            else:
                res = frame_raw

            if s["hud"]:
                draw_str(res, (20, 20), "Subtraction:%s  Overlay:%s  Equalize:%s" %
                         (s["subtract"], s["overlay"], s["equalize"]))
                draw_str(res, (20, 40), "Pedal mode:%s  HUD:%s" % (s["pedal_mode"], s["hud"]))
                draw_str(res, (20, 60), "Cam: x=%.1f  y=%.1f cm" % (s["pos_x_cm"], s["pos_y_cm"]))
            if pedal_down:
                draw_str(res, (20, 80), "PEDAL ACTIVE")

        if res is not None:
            if show_window:
                if s["fullscreen"]:
                    # Stack the video on top and a thin control bar below it.
                    vid = cv.cvtColor(res, cv.COLOR_GRAY2BGR) if res.ndim == 2 else res
                    bar, bbtns = render_control_bar(s, vid.shape[1], live)
                    composite = np.vstack([vid, bar])
                    overlay_buttons[:] = [(x, y + vid.shape[0], w, h, k, n)
                                          for (x, y, w, h, k, n) in bbtns]
                    cv.imshow("FLUORO", composite)
                else:
                    # Clean video in FLUORO, the vertical panel in CONTROLS.
                    panel, btns = render_controls(s, logo, live)
                    ctrl_buttons[:] = btns
                    cv.imshow("CONTROLS", panel)
                    cv.imshow("FLUORO", res)
            # The web MJPEG preview always streams the clean frame (the browser has
            # its own HTML buttons), so encode `res`, not the composited view.
            ok, jpg = cv.imencode(".jpg", res, [cv.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with _latest_lock:
                    _latest_jpeg = jpg.tobytes()

    cap.release()
    cv.destroyAllWindows()
    # Stop the process so the Flask daemon thread exits too.
    os._exit(0)


def _handle_key(key):
    '''Map FLUORO-window keypresses onto the shared state (keeps parity with the CLI).'''
    with state_lock:
        if key == ord('1'):
            state["subtract"] = not state["subtract"]
        elif key == ord('2') or key == ord('5'):
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


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(__doc__)

    args = sys.argv[1:]
    port = 5000
    show_window = True
    cam_index = 0
    force_http = False

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--port":
            port = int(args[i + 1]); i += 2; continue
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
        target=lambda: app.run(host="0.0.0.0", port=port, threaded=True,
                               debug=False, use_reloader=False,
                               ssl_context=ssl_context),
        daemon=True)
    flask_thread.start()
    print("Control panel:  %s://<this-machine-ip>:%d/" % (scheme, port))
    if use_https:
        print("(self-signed cert — your browser will show a one-time "
              "'not private' warning to click through)")

    run_simulation(cam_index, show_window)
