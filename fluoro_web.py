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

The capture + processing loop and the OpenCV window run on the main thread
(OpenCV's GUI must run on the main thread). Flask runs in a background daemon
thread and only reads/writes a small shared-state dict guarded by a lock.

Usage:
    python fluoro_web.py [<video device number>] [--port 5000] [--no-window]

Then open  http://<this-machine-ip>:<port>/  in a browser.
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
from flask import Flask, Response, jsonify, render_template_string

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OVERLAY_IMAGE = os.path.join(BASE_DIR, "skel.jpg")

# Brightness threshold above which a pixel is treated as bright "white background"
# and composited mostly from the overlay. (Mirrors fluoro_simulator (3).py.)
MASK_THRESHOLD = 220
# Background-model learning rate for cv.accumulateWeighted.
ALPHA = 0.05

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
}

# Latest processed frame, JPEG-encoded, for the MJPEG preview stream.
_latest_jpeg = None
_latest_lock = threading.Lock()


def get_state_snapshot():
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
  header { padding: 14px 16px; display: flex; align-items: center; gap: 10px;
           border-bottom: 1px solid #1d2733; position: sticky; top: 0; background: #0b0f14; }
  header h1 { font-size: 18px; margin: 0; font-weight: 600; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: #f04438; }
  .dot.live { background: #12b76a; box-shadow: 0 0 8px #12b76a; }
  main { padding: 16px; max-width: 720px; margin: 0 auto; }
  .preview { width: 100%; background: #000; border-radius: 12px; overflow: hidden;
             border: 1px solid #1d2733; aspect-ratio: 4 / 3; display: flex; }
  .preview img { width: 100%; height: 100%; object-fit: contain; }
  .preview:fullscreen, .preview:-webkit-full-screen {
      width: 100vw; height: 100vh; border: 0; border-radius: 0; aspect-ratio: auto; }
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
</style>
</head>
<body>
<header>
  <span id="dot" class="dot"></span>
  <h1>FluoroSim Controls</h1>
</header>
<main>
  <div class="preview"><img id="feed" src="/video_feed" alt="live preview"></div>

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
  // Fullscreen the live preview in THIS browser (must run inside a click handler).
  var el = document.querySelector('.preview');
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
    return render_template_string(PAGE)


@app.route("/api/state")
def api_state():
    return jsonify(get_state_snapshot())


@app.route("/api/toggle/<name>", methods=["POST"])
def api_toggle(name):
    with state_lock:
        if name in ("subtract", "overlay", "equalize", "pedal_mode", "pedal_pressed", "hud"):
            state[name] = not state[name]
        snap = dict(state)
    return jsonify(snap)


@app.route("/api/action/<action>", methods=["POST"])
def api_action(action):
    with state_lock:
        if action == "fullscreen":
            state["fullscreen"] = True
        elif action == "windowed":
            state["fullscreen"] = False
        elif action == "quit":
            state["quit"] = True
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
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def run_simulation(cam_index, show_window):
    global _latest_jpeg

    cap = cv.VideoCapture(cam_index, cv.CAP_V4L2)
    if not cap.isOpened():
        print("Warning: unable to open video source:", cam_index)

    overlay = cv.imread(OVERLAY_IMAGE)
    if overlay is None:
        raise FileNotFoundError("Cannot load image: %s" % OVERLAY_IMAGE)
    overlay = cv.cvtColor(overlay, cv.COLOR_RGB2GRAY)

    if show_window:
        cv.namedWindow("FLUORO", cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty("FLUORO", cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)

    background = None
    applied_fullscreen = None
    res = None

    while True:
        s = get_state_snapshot()
        if s["quit"]:
            break

        # Keep the OpenCV window's fullscreen state in sync with the web toggle.
        if show_window and s["fullscreen"] != applied_fullscreen:
            cv.setWindowProperty(
                "FLUORO", cv.WND_PROP_FULLSCREEN,
                cv.WINDOW_FULLSCREEN if s["fullscreen"] else cv.WINDOW_NORMAL)
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

        pedal_down = s["pedal_pressed"] or key_pedal
        capture_now = pedal_down or not s["pedal_mode"]

        if capture_now:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            frame_raw = frame_gray.copy()
            overlay = cv.resize(overlay, (frame_gray.shape[1], frame_gray.shape[0]))

            if background is None:
                background = frame_gray.astype("float")
            bg_uint8 = cv.convertScaleAbs(background)

            if s["subtract"]:
                frame_gray = cv.absdiff(frame_gray, bg_uint8)
                cv.normalize(frame_gray, frame_gray, 0, 255, cv.NORM_MINMAX)
                frame_gray = cv.bitwise_not(frame_gray)

            cv.accumulateWeighted(frame_gray, background, ALPHA)

            # Overlay off => show the full raw video (bright/white areas intact).
            if s["overlay"]:
                res = composite_overlay(frame_gray, overlay, s["equalize"])
            else:
                res = frame_raw

            if s["hud"]:
                draw_str(res, (20, 20), "Subtraction:%s  Overlay:%s  Equalize:%s" %
                         (s["subtract"], s["overlay"], s["equalize"]))
                draw_str(res, (20, 40), "Pedal mode:%s  HUD:%s" % (s["pedal_mode"], s["hud"]))
            if pedal_down:
                draw_str(res, (20, 80), "PEDAL ACTIVE")

        if res is not None:
            if show_window:
                cv.imshow("FLUORO", res)
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
