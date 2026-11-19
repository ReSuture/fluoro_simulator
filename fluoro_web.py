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

    Overlay, Equalize, Pedal mode, HUD, Record,
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
by ``state_lock`` (the toggle flags) and the preview frame (``publish_preview`` /
``latest_preview``, guarded by ``_latest_lock``). The web handlers never touch
the camera or the windows — they just flip flags that the main loop reads each
frame, and JPEG-encode the preview frame the loop hands over.

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
                         [--no-window] [--http] [--screen WxH[+X+Y]]

``--screen 1024x600`` composes the fullscreen UI natively at the display's
resolution (e.g. the WIMAXIT touch monitor): the video is letterboxed instead
of stretched, buttons get real touch-sized hit areas, and tapping a text
field pops an on-screen keyboard. ``--screen 1024x600+1920+0`` also says where
that display sits in the desktop, so the window lands on it when a second
monitor is attached. launch_fluoro.sh finds the touch display (displays.py)
and passes this flag.

Then open  https://<this-machine-ip>:<port>/  in a browser (http:// without a cert).

``--host 127.0.0.1`` restricts the panel to local connections only — used on
Pis exposed through a Cloudflare Tunnel, where the tunnel daemon on the same
machine is the only client allowed to reach the panel directly.
'''

from __future__ import print_function

import os
import sys
import ctypes
import ctypes.util
import glob
import functools
import json
import math
import re
import time
import struct
import threading
import traceback

try:
    import fcntl          # Linux only: used to ask V4L2 what each device is
except ImportError:       # pragma: no cover - non-Linux dev boxes
    fcntl = None

# Force OpenCV's Qt GUI onto the X11/XWayland backend. Under the native Wayland
# Qt backend, cv.setWindowProperty(FULLSCREEN) is a no-op (it logs
# "qt.qpa.wayland: Wayland does not support QWindow::requestActivate()") so the
# Fullscreen/Windowed controls can't change the FLUORO window. Set before cv2 is
# imported; respect an explicit override if the user already set one.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
import cv2 as cv
from flask import Flask, Response, jsonify, render_template_string, request, send_file

import device_setup
import recording
import uploader

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OVERLAY_IMAGE = os.path.join(BASE_DIR, "skel.jpg")
LOGO_IMAGE = os.path.join(BASE_DIR, "static", "logosign_white.png")

# ── Attenuation compositing ───────────────────────────────────────────────────
# The composite models true fluoroscopy: every camera pixel is converted to a
# TRANSMISSION factor (0 = radiopaque, 1 = unattenuated beam) and the anatomy
# master is multiplied by it — exactly like stacking attenuators in an X-ray
# beam. Transmission is the live pixel divided by an estimate of the external
# illumination at that pixel (flat-field correction), so the light source —
# gradients, hot spots — divides out entirely rather than being masked or
# blended away: evenly-or-unevenly lit white background gives transmission ~1
# and shows the pure anatomy; the vasculature model attenuates moderately; the
# catheter attenuates hardest and is always the darkest object on screen
# (multiplying the anatomy by t <= 1 can only darken it).
#
# The illumination field is estimated per frame by a grayscale dilation (the
# local max rides over dark objects narrower than ILLUM_SPAN_PX and reads the
# background brightness behind them) followed by a Gaussian blur (ILLUM_SIGMA)
# that smooths it into a slowly-varying light field. Objects WIDER than
# ILLUM_SPAN_PX (a hand laid flat over the scene) start being treated as
# illumination and hollow out to their edges — raise ILLUM_SPAN_PX if large
# objects lose their interiors, at some CPU cost per frame.
ILLUM_SPAN_PX = 41
ILLUM_SIGMA = 21
# The light field is smooth by construction, so it's estimated on a
# 1/ILLUM_DOWNSCALE-size frame and upsampled — visually identical, ~16x
# cheaper (matters on the Pi).
ILLUM_DOWNSCALE = 4
# Contrast curve applied to transmission (t ** ATTEN_GAMMA). >1 deepens the
# dark end, separating the catheter from the vasculature model; 1.0 is
# physically linear.
ATTEN_GAMMA = 1.6

# ── C-arm field of view ───────────────────────────────────────────────────────
# Each view is masked to the shape of the detector's field. A flat-panel C-arm
# shows its square detector's image cut to a centred circle a little larger
# than the square, as on GE's OEC display: the middle of each side stays
# straight and each corner becomes one round arc, meeting the sides in a point
# ("flatpanel", the default). "square" is the bare detector square and
# "circle" an image intensifier's round input face (a bright disc with black
# corners). APERTURE_ON toggles the mask; APERTURE_SCALE is the field's width -
# the square's side or the disc's diameter - as a fraction of the frame's
# SHORTER side (1.0 just touches the top and bottom edges), so every shape is
# the same size. APERTURE_CORNER is the flat panel's circle radius as a
# multiple of half the width: 1.0 would leave just the disc and 1.414 the whole
# square; measured off GE's display it is about 1.12, which keeps half of each
# side straight and meets it at about 27°. APERTURE_FEATHER softens the edge by
# that fraction of half the width, so it is anti-aliased rather than jagged
# (0 = crisp edge).
APERTURE_ON = True
APERTURE_SHAPE = "flatpanel"  # "flatpanel"/"square" = flat-panel detector, "circle" = image intensifier
APERTURE_SCALE = 0.92
APERTURE_CORNER = 1.12
APERTURE_FEATHER = 0.015

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
# Calibrated against frontal_master.png (867×1275), a plain no-contrast frontal
# torso projection, shoulders → proximal femurs. Retune these four values if
# the master image changes. (PX_PER_CM is scaled from the previous master's
# by-eye tuning by the anatomy's size in pixels, so the default framing and
# pan feel carry over — the FOV is anchored to the body width, not to true
# anatomical centimetres.)
MASTER_IMAGE = os.path.join(BASE_DIR, "frontal_master.png")

# The lateral view's anatomy background, a lateral projection of the same
# torso. Optional: with no file here the simulator builds an obvious
# placeholder field instead (build_filler_master). It must have the same
# pixel dimensions as MASTER_IMAGE, since the cm -> pixel pan/zoom calibration
# below is expressed in that image's pixels.
LATERAL_MASTER_IMAGE = os.path.join(BASE_DIR, "lateral_master.png")
# Attenuation-based fluoro display is dense = dark: bone reads darker than
# tissue and the unattenuated background is bright. The masters are supplied
# in that polarity already (white air, dark bone), so they load as-is.
# MASTER_INVERT flips a film-convention radiograph (white bone, dark air) at
# load instead; MASTER_GAMMA is then applied on the 0-1 scale to tune the
# bone/tissue separation (>1 deepens bone and other dark structures, <1 lifts
# them, 1.0 = unchanged). The previous film radiograph used invert + 0.75; on
# these masters 0.75 washes the frontal spine out, so they are left untouched.
MASTER_INVERT = False
MASTER_GAMMA = 1.0
# Gaussian blur (sigma, px at master resolution) applied at load. Real fluoro
# is softened by scatter; this takes a crisp radiograph's bone edges off. The
# masters are already soft and are about the frame width at the default zoom,
# so any blur here shows nearly 1:1. 0 = off.
MASTER_SOFTEN = 0.0
# Brightness scale applied to the master at load. <1.0 darkens it (pulls the
# bright/white areas down the most, since it's multiplicative); 1.0 = as-is.
MASTER_BRIGHTNESS = 1.0
# Master-image pixels per real centimetre of anatomy.
PX_PER_CM = 7.0
# The master pixel that camera coordinate (0, 0) maps to (viewport centre at origin).
ORIGIN_PX = (433, 637)          # centre of the 867×1275 master (~T12/L1)
# Physical area the detector sees at once, (width_cm, height_cm). Keep 4:3 to match
# the camera frame. This is the ZOOM knob: larger = zoomed out (more anatomy shown),
# smaller = zoomed in. Widened from (80, 60) by 1.17 — the old view at z=+8.5 —
# so the resting z=0 view shows the full body width; z stays a pure offset from
# nominal camera height, so the motion controller's z=0 lands on this framing.
FOV_CM = (93.6, 70.2)
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
# ── Camera discovery and the main / lateral roles ─────────────────────────────
# The simulator drives two views: the MAIN (frontal) camera and an optional
# LATERAL one, mirroring a C-arm's AP and lateral projections. Which physical
# camera plays which role is decided by the USB PORT it is plugged into, not by
# the /dev/videoN number - Linux hands those out in probe order, so they shuffle
# between boots and replugs, while the port is a property of the bench.
#
# V4L2 reports the port in a device's bus_info ("usb-3f980000.usb-1.4"), which is
# what we key the roles on. The assignment lives in cameras.json next to the
# device identity cache; with no file (or a port that is not plugged in right
# now) the roles fall back to port order: first camera = main, second = lateral.
CAMERA_CONFIG = os.path.join(device_setup.CACHE_DIR, "cameras.json")

# VIDIOC_QUERYCAP: _IOR('V', 0, struct v4l2_capability) - asks a /dev/video*
# node what it actually is. Necessary because a Pi exposes a dozen or more
# video nodes and some report "video capture" - four of the bcm2835 ISP ones
# on a Pi 3: without this filter the simulator would think it had four cameras
# attached and no camera at all. (A Pi 5's pispbe ISP nodes report the
# multi-planar capture capability instead, so the caps check drops them.)
_VIDIOC_QUERYCAP = 0x80685600
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_STREAMING = 0x04000000
_V4L2_CAP_DEVICE_CAPS = 0x80000000

# The cameras are USB (UVC) webcams, whose bus_info always starts "usb-". The
# Pi's own codec/ISP plumbing reports "platform:..." (bcm2835-isp on a Pi 3;
# pispbe, rpi-hevc-dec and the CSI receiver on a Pi 5), so keeping only USB
# nodes filters it out on every model without listing each driver.
_CAMERA_BUS = "usb-"


def _querycap(path):
    '''Ask one /dev/video* node what it is, or None if it will not answer.

    Read-only: this works on a device another process is already streaming from,
    so scanning never disturbs the running capture.
    '''
    if fcntl is None:
        return None
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        buf = bytearray(104)
        fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
        driver, card, bus, _ver, caps, device_caps = struct.unpack(
            "16s32s32sIII", bytes(buf[:92]))
        text = lambda b: b.split(b"\x00")[0].decode("ascii", "replace").strip()
        return {"driver": text(driver), "name": text(card), "port": text(bus),
                "caps": device_caps if (caps & _V4L2_CAP_DEVICE_CAPS) else caps}
    except (OSError, IOError, struct.error):
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def list_cameras():
    '''Every real capture camera attached right now, ordered by port.

    Returns dicts of {device, index, port, name}. The order is stable for a
    given set of ports (it is the sort order of the port strings, not the probe
    order), so "first camera = main" means the same thing across reboots.
    '''
    cams = []
    for path in glob.glob("/dev/video*"):
        cap = _querycap(path)
        if cap is None:
            continue
        if not (cap["caps"] & _V4L2_CAP_VIDEO_CAPTURE):
            continue
        if not (cap["caps"] & _V4L2_CAP_STREAMING):
            continue
        if not cap["port"].startswith(_CAMERA_BUS):
            continue
        try:
            index = int(path.rsplit("video", 1)[1])
        except (IndexError, ValueError):
            continue
        cams.append({"device": path, "index": index,
                     "port": cap["port"], "name": cap["name"]})
    # A camera with several nodes (capture + metadata) shares one port; keep the
    # lowest-numbered node, which is the capture one.
    best = {}
    for cam in sorted(cams, key=lambda c: c["index"]):
        best.setdefault(cam["port"], cam)
    return sorted(best.values(), key=lambda c: c["port"])


def discovery_works():
    '''True when this machine can be trusted to enumerate its own cameras.

    Needs the V4L2 ioctl (Linux) and at least one video node to ask about. When
    this is False - a non-Linux dev box, or a /dev with no video nodes at all -
    the caller falls back to opening a device index blindly, as the simulator
    always did.
    '''
    return fcntl is not None and bool(glob.glob("/dev/video*"))


def read_camera_roles():
    '''The saved {"main": <port>, "lateral": <port>} assignment (may be empty).'''
    try:
        with open(CAMERA_CONFIG, encoding="utf-8") as fh:
            data = json.load(fh)
        return {k: data.get(k) for k in ("main", "lateral") if data.get(k)}
    except (OSError, ValueError):
        return {}


def write_camera_roles(roles):
    '''Persist the port assignment so it survives restarts. -> ok bool.'''
    try:
        os.makedirs(device_setup.CACHE_DIR, exist_ok=True)
        with open(CAMERA_CONFIG, "w", encoding="utf-8") as fh:
            json.dump(roles, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return True
    except OSError as exc:
        print("Could not save %s: %s" % (CAMERA_CONFIG, exc))
        return False


def match_camera(cams, spec):
    '''Find the attached camera a spec names: a port, a /dev path, or an index.'''
    if spec is None:
        return None
    spec = str(spec).strip()
    for cam in cams:
        if spec in (cam["port"], cam["device"], str(cam["index"])):
            return cam
    return None


def resolve_cameras(cams):
    '''Decide which attached camera drives each view.

    Saved ports win; anything they do not answer for falls back to port order
    (first = main, second = lateral). The two roles never collapse onto one
    camera, and the main view is always filled first: with a single camera
    attached there is no lateral view, even if that camera sits in the port
    assigned to the lateral role - a working frontal view beats a black screen.

    Returns (main, lateral) as camera records from list_cameras(), or None.
    '''
    if not cams:
        return None, None
    # Sorted here too, not just in list_cameras(): "first camera = main" is an
    # invariant of the role assignment, so it must not depend on the caller
    # handing them over in port order.
    cams = sorted(cams, key=lambda c: c["port"])
    roles = read_camera_roles()
    main = match_camera(cams, roles.get("main"))
    lateral = match_camera(cams, roles.get("lateral"))
    if lateral is main:
        lateral = None
    spare = [c for c in cams if c is not main and c is not lateral]
    if main is None:
        main = spare.pop(0) if spare else None
    if lateral is None:
        lateral = spare.pop(0) if spare else None
    if main is None:
        main, lateral = lateral, None
    return main, lateral


def describe_cameras():
    '''Human-readable inventory for --list-cameras.'''
    cams = list_cameras()
    roles = read_camera_roles()
    main_cam, lateral_cam = resolve_cameras(cams)
    lines = ["Cameras attached: %d" % len(cams)]
    if not cams:
        lines.append("  (none - the simulator will run with a NO CAMERA CONNECTED screen)")
    for cam in cams:
        role = ("main" if cam is main_cam else
                "lateral" if cam is lateral_cam else "-")
        lines.append("  %-8s %-30s %-13s %s"
                     % (role, cam["port"], cam["device"], cam["name"]))
    lines.append("")
    lines.append("Roles from %s: %s"
                 % (CAMERA_CONFIG, roles if roles else "(none saved - using port order)"))
    lines.append("Assign with:  python fluoro_web.py --main <port> --lateral <port>")
    return "\n".join(lines)


# All values touched by both the processing loop and the web handlers live here,
# guarded by `state_lock`. Booleans mirror the keyboard toggles of the original.
state_lock = threading.Lock()
state = {
    "overlay": True,        # (2/5) anatomy overlay; off => full raw video
    "equalize": False,      # (6) CLAHE histogram equalisation
    "pedal_mode": False,    # (space) only capture while the foot pedal ('b') is held
    "hud": True,            # (7) on-screen text HUD
    "fullscreen": True,     # (3/4) FLUORO window fullscreen vs. windowed
    "running": True,        # simulation active; False = standby (web server stays up)
    "quit": False,          # exit the whole process (ESC key); web Stop uses "running"
    "pos_x_cm": 0.0,        # camera X position (cm from origin) — drives the viewport
    "pos_y_cm": 0.0,        # camera Y position (cm from origin) — drives the viewport
    "pos_z_cm": 0.0,        # camera Z position (cm from nominal height) — drives the zoom
    "ui_view": "fluoro",    # which tab the FLUORO window shows: fluoro|remote|library
    "recording": False,     # session recording to ~/fluorosim_recordings
    "fluoro_sec": 0.0,      # fluoro time: seconds of beam-on (frames taken / pedal held)
    "camera": False,        # read-only: a camera is open and delivering frames
    "lateral": False,       # show the lateral camera instead of the main one
    "cameras": 0,           # read-only: how many cameras are attached right now
    "lateral_available": False,  # read-only: a distinct lateral camera is attached
}

# How often to retry opening the camera while none is attached (seconds). A
# missing camera is never fatal - the simulator starts anyway and stands a black
# "NO CAMERA CONNECTED" frame in for the video, so the control panel, the tabs
# and the Library stay usable - and a camera plugged in later is picked up on
# its own at the next retry.
CAMERA_RETRY_SEC = 3.0

# Fluoro time adds up the time between consecutive moments the beam is seen on
# (a frame taken; in Pedal mode, the pedal held), but a gap longer than this
# (seconds) is not beam time: the camera stalled, or the pedal came up between.
# It outlasts the keyboard's key-repeat delay (0.6 s by default), since where X
# cannot say whether the key is down, a held pedal is seen only as its repeats.
FLUORO_GAP_SEC = 0.75

# Consecutive failed reads before a camera is treated as gone (unplugged) and
# released, dropping back to the placeholder + retry path. A few dropped frames
# are normal, so this is deliberately not 1.
CAMERA_READ_FAIL_LIMIT = 30

# How often to re-inventory the attached cameras (seconds), so a second camera
# plugged in mid-session lights up "Lateral view" without a restart.
CAMERA_SCAN_SEC = 3.0

# How long a read waits for a frame newer than the last one handed out before
# counting as a failed read (seconds). A healthy camera delivers one every
# ~33ms, so this only trips on a stalled or vanished camera.
CAMERA_FRAME_TIMEOUT_SEC = 0.5


class LatestFrameCapture(object):
    '''A VideoCapture that always hands back the newest frame, never a queued one.

    V4L2 queues several frames inside the driver, and the pipeline is slower
    than the camera (~19 fps against 30), so a plain read() returns the oldest
    frame in that queue - the picture lags the bench by the whole queue, about
    200ms on a Pi 3. Worse in pedal mode: nothing is read while the pedal is up,
    so the first frames after a press show the moment it was last released.

    A daemon thread reads the camera continuously and keeps only the latest
    frame; read() returns that one, waiting briefly if the caller has already
    had it so no frame is processed twice. In split screen the two cameras are
    also read in parallel rather than one after the other.
    '''

    def __init__(self, cap):
        self._cap = cap
        self._cond = threading.Condition()
        self._ok, self._frame = False, None
        self._seq = 0          # bumped for every read the thread completes
        self._taken = 0        # the seq last handed out by read()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            ok, frame = self._cap.read()
            with self._cond:
                self._ok, self._frame = ok, frame
                self._seq += 1
                self._cond.notify_all()
            if not ok:
                time.sleep(0.01)   # a dead camera fails instantly; do not spin

    def isOpened(self):
        return self._cap.isOpened()

    def read(self):
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._taken,
                                       CAMERA_FRAME_TIMEOUT_SEC):
                return False, None
            self._taken = self._seq
            return self._ok, self._frame

    def grab(self):
        return True            # the thread keeps the camera drained

    def release(self):
        self._stop = True
        # Let a read in progress finish first: releasing a VideoCapture that
        # another thread is reading from is not safe in OpenCV.
        self._thread.join(timeout=2.0)
        self._cap.release()

# After a deliberate view switch, keep showing the last frame for this long
# while the other camera opens, so switching does not flash the black
# placeholder for the second or so a USB camera takes to come up.
# Split screen: each half is composited at no more than this width before
# being scaled into place. The pipeline costs ~60ms per view at 640x480 on a
# Pi 3, so compositing both halves at full capture resolution would put the
# split view into single-figure frame rates. The single frontal view is
# untouched and still composites at the camera's own resolution.
SPLIT_WORK_MAX_W = 480

# Width of the dark seam drawn between the two halves.
SPLIT_DIVIDER_PX = 3


def set_lateral(on):
    '''Switch the lateral view on/off, ignoring it when there is no second camera.

    The single gate for every caller (web panel, on-screen button, key), so an
    unavailable lateral view cannot be selected from any of them.
    '''
    with state_lock:
        if on and not state["lateral_available"]:
            return False
        state["lateral"] = bool(on)
        return True

# Compose the fullscreen UI natively at this (width, height) instead of at the
# camera frame's size — set by --screen for touch displays so the pixels (and
# therefore the tap targets) map 1:1 onto the panel. None = classic behaviour.
SCREEN_SIZE = None
# Where that display sits in the desktop layout, (x, y) — from --screen's
# "+X+Y" part. With a second monitor attached (a Pi 5 mirroring the touch
# panel) the FLUORO window is placed here before going fullscreen, so it
# fills the touch panel rather than whichever output the desktop picked.
SCREEN_POS = None

# Windowed mode on a touch panel (--screen) tiles the FLUORO window and the
# CONTROLS panel side by side across the panel, so both are on it and together
# fill it - the image as a square, full height, on the left and the GE-style
# panel in the rest - rather than at the camera's size and a fixed desktop spot,
# which is off the panel whenever it is not the leftmost output. labwc draws a
# title bar this tall above each window's contents (the Pi's PiXtrix theme).
TITLEBAR_PX = 26


def windowed_layout():
    '''Content boxes (x, y, w, h) of the FLUORO and CONTROLS windows for
    windowed mode on the touch panel, or None without --screen.'''
    if not SCREEN_SIZE:
        return None
    (sx, sy), (sw, sh) = SCREEN_POS or (0, 0), SCREEN_SIZE
    y, h = sy + TITLEBAR_PX, sh - TITLEBAR_PX
    side = min(h, int(sw * 0.6))    # a narrow screen still leaves the panel room
    return (sx, y, side, side), (sx + side, y, sw - side, h)


def window_open(name):
    '''False while a window is closed with its title-bar X. OpenCV only hides
    it then (minimized still counts as open), and the next cv.imshow shows
    it again, with its flags and click handler intact.'''
    try:
        return cv.getWindowProperty(name, cv.WND_PROP_VISIBLE) >= 1
    except cv.error:
        return False

# Latest processed frame for the MJPEG preview stream. The main loop only hands
# the frame over (publish_preview); it is JPEG-encoded on first request, in the
# web thread, so the capture loop never pays for encoding - and nothing is
# encoded at all while nobody is watching. _latest_seq identifies the frame so
# the stream sends each one once.
_latest_jpeg = None
_latest_frame = None
_latest_seq = 0
_latest_lock = threading.Lock()

# Minimum gap between preview frames (seconds): caps the stream at ~20 fps to
# keep the bandwidth sane over the tunnel.
PREVIEW_MIN_INTERVAL = 0.05


def publish_preview(frame=None, jpeg=None):
    '''Make a frame the current preview: a raw image, or an already-encoded JPEG.'''
    global _latest_jpeg, _latest_frame, _latest_seq
    with _latest_lock:
        if jpeg is not None and jpeg is _latest_jpeg:
            return     # the same static placeholder again: not a new frame
        _latest_frame, _latest_jpeg = frame, jpeg
        _latest_seq += 1


def latest_preview():
    '''The current preview frame as JPEG bytes. -> (seq, jpeg | None)'''
    global _latest_jpeg
    with _latest_lock:
        seq, frame, jpeg = _latest_seq, _latest_frame, _latest_jpeg
    if jpeg is None and frame is not None:
        ok, enc = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 80])
        jpeg = enc.tobytes() if ok else None
        with _latest_lock:
            if _latest_seq == seq:     # still current: share it with other viewers
                _latest_jpeg = jpeg
    return seq, jpeg


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
  /* The device's own controls as its touch screen draws them, GE OEC style:
     the colours are the GE_* palette and the shapes the _ge_* helpers in
     fluoro_web.py (glossy = a three-stop gradient, a light rim, a sheen). */
  @font-face { font-family: "Nunito Sans"; src: url("/ui-font.ttf") format("truetype");
               font-weight: 200 1000; font-stretch: 75% 125%; font-display: swap; }
  :root {
    color-scheme: dark;
    --bg: #090b0e; --white: #ffffff; --muted: #b8c0c8; --dim: #868e96; --oncap: #d2ffbe;
    --card: linear-gradient(#6a7680, #4d5862 50%, #424b54);
    --tab-on: #6a7680; --tab-off: #343b42;
    --yellow: #f5d728; --live: #12b76a; --dead: #f04438;
    --cyan: radial-gradient(circle at 38% 30%, #80d6ec 0%, #20a0c8 42.5%, #06527e 85%);
    --cyan-off: radial-gradient(circle at 38% 30%, #5096aa 0%, #266278 42.5%, #143846 85%);
    --red: radial-gradient(circle at 38% 30%, #ff9696 0%, #e64646 42.5%, #961e1e 85%);
    --red-off: radial-gradient(circle at 38% 30%, #aa6e78 0%, #824650 42.5%, #5a2d32 85%);
  }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  html, body { margin: 0; background: var(--bg); color: var(--white);
               font-family: "Nunito Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  button { font: inherit; color: inherit; border: 0; padding: 0; background: none; cursor: pointer; }
  button:disabled { cursor: not-allowed; }
  .gloss { position: relative; }
  .gloss::after { content: ""; position: absolute; left: 8%; right: 8%; top: 6%; height: 44%;
                  border-radius: 50%; pointer-events: none;
                  background: radial-gradient(ellipse at 50% 50%, rgba(255,255,255,.30), rgba(255,255,255,0) 70%); }

  /* Picture left, panel right - the Pi's windowed layout; stacked on a phone. */
  .app { display: flex; gap: 16px; justify-content: center; align-items: flex-start;
         padding: 16px; min-height: 100vh; background: var(--bg); }
  .view { flex: 0 1 auto; display: flex; flex-direction: column; gap: 10px;
          width: min(calc(100vh - 32px), calc(100vw - 32px - 450px - 16px)); }
  /* The square field alone (the frame's black sides cropped off); a split
     view or the no-camera card shows whole. */
  .field { position: relative; width: 100%; aspect-ratio: 1 / 1; background: #000; overflow: hidden; }
  .field > .feed { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
  .field.wide { aspect-ratio: auto; }
  .field.wide > .feed { position: static; display: block; height: auto; object-fit: contain; }
  .field .logo { position: absolute; left: 14px; top: 14px; width: clamp(60px, 17%, 120px); height: auto; }
  .field .live { position: absolute; right: 14px; top: 14px; width: 12px; height: 12px;
                 border-radius: 50%; background: var(--dead); }
  .field .live.on { background: var(--live); box-shadow: 0 0 8px var(--live); }
  .field .rec { position: absolute; right: 36px; top: 10px; display: none; align-items: center;
                gap: 7px; font-size: 15px; color: #ff9a9a; }
  .field .rec::before { content: ""; width: 12px; height: 12px; border-radius: 50%; background: var(--dead); }
  .field .rec.on { display: flex; }
  .field .note { position: absolute; bottom: 12px; font-size: 15px; display: none; }
  .field .note.on { display: block; }
  .field .pedal { left: 16px; color: var(--yellow); }
  .field .zoom { right: 16px; color: var(--muted); }
  .warn { margin: 0; padding: 10px 14px; border-radius: 14px; background: #2a1416;
          border: 1px solid #5b2327; color: #ff9a9a; font-size: 14px; }

  /* The panel, GE_W wide like the Pi's CONTROLS window. */
  .panel { flex: 0 0 450px; width: 450px; display: flex; flex-direction: column; gap: 10px; }
  .card { background: var(--card); border-radius: 14px; }
  .tabs { display: flex; align-items: flex-end; height: 44px; padding: 0 8px; margin-bottom: -2px;
          border-bottom: 3px solid var(--tab-on); }
  .tab { position: relative; flex: 0 0 162px; height: 39px; margin-right: -26px; display: flex;
         align-items: center; justify-content: center; gap: 6px; font-size: 16px;
         color: var(--muted); text-decoration: none; background: var(--tab-off);
         clip-path: polygon(18px 0, calc(100% - 18px) 0, 100% 100%, 0 100%); }
  .tab.on { background: var(--tab-on); color: var(--white); font-weight: 800; z-index: 1; }
  .tab small { font-size: 12px; }

  .toggles { display: flex; justify-content: space-evenly; padding: 10px 0 9px; }
  .tg { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .knob { width: 70px; height: 70px; border-radius: 50%; font-size: 13px; font-weight: 800;
          color: var(--muted); background: var(--cyan-off);
          box-shadow: inset 0 0 0 1.5px rgba(80,150,170,.55); }
  .knob.sm { font-size: 11.5px; }
  .knob.on { color: var(--white); background: var(--cyan);
             box-shadow: inset 0 0 0 1.5px rgba(128,214,236,.55), 0 0 18px 3px rgba(110,220,255,.7); }
  .knob.red { background: var(--red-off); box-shadow: inset 0 0 0 1.5px rgba(170,110,120,.55); }
  .knob.red.on { background: var(--red);
                 box-shadow: inset 0 0 0 1.5px rgba(255,150,150,.55), 0 0 18px 3px rgba(255,90,90,.7); }
  .knob:active, .pill:active { transform: translateY(1px); }
  .cap { font-size: 11px; color: var(--dim); }
  .cap.on { color: var(--oncap); }

  .readout { position: relative; display: grid; grid-template-columns: 80px 80px 80px 1fr;
             height: 84px; background: #07080a; border: 1.2px solid #d6dce1; border-radius: 14px; }
  .ax { position: relative; margin: 14px 0; border-right: 1px solid #3a4046; }
  .ax .lbl { position: absolute; left: 14px; top: 1px; font-size: 12px; color: var(--dim); white-space: pre; }
  .ax .val { position: absolute; left: 4px; right: 0; bottom: -3px; text-align: center;
             font-size: 28px; line-height: 1; }
  .clock { display: flex; align-items: flex-end; gap: 6px; padding: 0 0 15px 14px; }
  .clock b { font-size: 40px; line-height: 1; font-weight: 400; }
  .clock small { font-size: 12px; color: var(--dim); margin-bottom: 3px; }
  .pill { border-radius: 999px; font-weight: 800; }
  .reset { position: absolute; right: 12px; top: 8px; width: 58px; height: 24px; font-size: 11px;
           color: var(--muted); background: linear-gradient(#5096aa, #266278 50%, #143846);
           box-shadow: inset 0 0 0 1px rgba(80,150,170,.55); }

  .move { display: flex; align-items: center; gap: 14px; padding: 14px 12px 14px 16px; }
  .dpad { flex: 0 0 auto; width: 168px; height: 168px; overflow: visible; }
  .dpad text { font-family: inherit; font-weight: 800; font-size: 14px; fill: #fff; }
  .dpad .sec { fill: transparent; cursor: pointer; }
  .dpad .sec:active { fill: rgba(255,255,255,.2); }
  .rocker { flex: 0 0 auto; width: 62px; height: 172px; border-radius: 31px; overflow: hidden;
            background: linear-gradient(#c4cdfa, #8c96e2 50%, #565ca0);
            box-shadow: inset 0 0 0 1.5px rgba(196,205,250,.55); }
  .rocker button { position: absolute; left: 0; right: 0; height: 50%; display: flex;
                   flex-direction: column; align-items: center; gap: 6px; font-size: 15px; font-weight: 800; }
  .rocker button:active { background: rgba(255,255,255,.18); }
  .rocker .zp { top: 0; padding-top: 22px; }
  .rocker .zm { bottom: 0; flex-direction: column-reverse; padding-bottom: 22px; }
  .rocker .rule { position: absolute; left: 10px; right: 10px; top: calc(50% - 1px); height: 2px;
                  background: #545a96; }
  .chev { width: 18px; height: 18px; overflow: visible; }
  .chev polyline { fill: none; stroke: #fff; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
  .lateral { flex: 1 1 auto; height: 172px; border-radius: 14px; display: flex; flex-direction: column;
             align-items: center; justify-content: space-between; padding: 16px 6px 12px;
             background: linear-gradient(#a8acb0, #767a7e 50%, #4a4e52); }
  .lateral.na { background: linear-gradient(#666a6e, #4a4e52 50%, #323538); }
  .lateral .ttl { font-size: 17px; font-weight: 800; line-height: 1.35; text-align: center; }
  .lateral.na .ttl { color: var(--muted); }
  .switch { position: relative; width: 84px; height: 42px; border-radius: 21px; background: #32363a;
            box-shadow: inset 0 0 0 1.2px rgba(26,28,30,.6); }
  .switch.on { background: #20a0c8; }
  .switch:disabled { background: #3e4246; }
  .switch .k { position: absolute; top: 4px; left: 4px; width: 34px; height: 34px; border-radius: 50%;
               background: radial-gradient(circle at 38% 30%, #fff 0%, #e4e6e8 42.5%, #aeb2b6 85%);
               box-shadow: 1px 2px 3px rgba(16,18,20,.45), inset 0 0 0 1px rgba(255,255,255,.55);
               transition: left .15s; }
  .switch.on .k { left: 46px; }
  .switch:disabled .k { background: radial-gradient(circle at 38% 30%, #8e9296 0%, #707478 42.5%, #52565a 85%); }
  .lcap { font-size: 12px; color: var(--dim); }
  .lcap.on { color: var(--oncap); }
  .lcap.off { color: var(--muted); }

  .bottom { display: flex; align-items: center; justify-content: space-between; padding: 14px 14px 14px 18px; }
  .radio { display: flex; flex-direction: column; gap: 12px; }
  .radio button { display: flex; align-items: center; gap: 10px; font-size: 16px; color: var(--muted); }
  .radio button::before { content: ""; width: 18px; height: 18px; border-radius: 50%;
                          border: 2px solid var(--muted); }
  .radio button.on { color: var(--white); font-weight: 800; }
  .radio button.on::before { border-color: var(--white); background: radial-gradient(circle, #fff 0 4px, transparent 5px); }
  .run { width: 214px; height: 48px; font-size: 16px; color: #ffbebe;
         background: linear-gradient(#783c46, #5a242c 50%, #3c181e); box-shadow: inset 0 0 0 2px #f55a5a; }
  .run.start { color: #fff; background: linear-gradient(#80d6ec, #20a0c8 50%, #06527e);
               box-shadow: inset 0 0 0 1.5px rgba(128,214,236,.55), 0 0 14px rgba(110,220,255,.55); }

  details.adv { color: var(--dim); font-size: 13px; padding: 2px 4px 0; }
  details.adv summary { cursor: pointer; }
  details.adv .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 8px; }
  details.adv input { width: 72px; font: inherit; font-size: 15px; padding: 7px 8px; border-radius: 8px;
                      border: 1px solid #3a4046; background: #07080a; color: var(--white); }
  details.adv button { padding: 7px 16px; font-size: 14px; color: #fff;
                       background: linear-gradient(#5096aa, #266278 50%, #143846); }

  @media (max-width: 880px) {
    .app { flex-direction: column; align-items: center; padding: 10px; }
    .view { width: min(100%, 600px); }
    .panel { flex: none; width: min(100%, 450px); }
  }
  @media (max-width: 440px) {
    .tab { flex-basis: 140px; }
    .knob { width: 58px; height: 58px; font-size: 11px; }
    .knob.sm { font-size: 9.5px; }
    .readout { grid-template-columns: 64px 64px 64px 1fr; }
    .ax .val { font-size: 24px; }
    .clock b { font-size: 32px; }
    .move { gap: 8px; padding: 12px 8px; }
    .dpad { width: 140px; height: 140px; }
    .rocker { width: 54px; height: 150px; }
    .lateral { height: 150px; }
    .lateral .ttl { font-size: 15px; }
    .switch { width: 72px; height: 36px; }
    .switch .k { width: 28px; height: 28px; }
    .switch.on .k { left: 40px; }
    .run { width: 180px; font-size: 14px; }
  }
</style>
</head>
<body>
<div class="app" id="stage">
  <section class="view">
    <div class="field" id="field">
      <img class="feed" id="feed" src="/video_feed" alt="live picture">
      <img class="logo" src="/static/logosign_white.png" alt="ReSuture">
      <span class="rec" id="recnote">REC</span>
      <span class="live" id="dot"></span>
      <span class="note pedal" id="pedalnote">PEDAL MODE</span>
      <span class="note zoom" id="zoomnote">1.00x</span>
    </div>
    <p class="warn" id="camwarn" hidden>No camera connected &mdash; the simulator is running
      without video. Plug one in and it is picked up automatically.</p>
  </section>

  <aside class="panel">
    <nav class="tabs">
      <span class="tab on">FLUORO</span>
      {% if library_url %}
      <a class="tab" href="{{ library_url }}" target="_blank" rel="noopener">LIBRARY <small>&#8599;</small></a>
      {% endif %}
    </nav>

    <div class="card toggles">
      <div class="tg"><button class="knob sm gloss" data-toggle="overlay">OVERLAY</button><span class="cap">OFF</span></div>
      <div class="tg"><button class="knob sm gloss" data-toggle="equalize">EQUALIZE</button><span class="cap">OFF</span></div>
      <div class="tg"><button class="knob gloss" data-toggle="hud">HUD</button><span class="cap">OFF</span></div>
      <div class="tg"><button class="knob gloss" data-toggle="pedal_mode">PEDAL</button><span class="cap">OFF</span></div>
      <div class="tg"><button class="knob red gloss" data-toggle="recording">REC</button><span class="cap">OFF</span></div>
    </div>

    <div class="readout">
      <div class="ax"><span class="lbl">X  cm</span><span class="val" id="vx">0</span></div>
      <div class="ax"><span class="lbl">Y  cm</span><span class="val" id="vy">0</span></div>
      <div class="ax"><span class="lbl">Z  cm</span><span class="val" id="vz">0</span></div>
      <div class="clock"><b id="clock">00:00</b><small>mm:ss</small></div>
      <button class="pill reset gloss" id="reset">RESET</button>
    </div>

    <div class="card move">
      <svg class="dpad" viewBox="-84 -84 168 168" aria-label="pan">
        <defs>
          <radialGradient id="dpg" cx="38%" cy="30%" r="85%">
            <stop offset="0" stop-color="#96e2f0"/><stop offset=".5" stop-color="#28a8ce"/>
            <stop offset="1" stop-color="#085e8c"/></radialGradient>
          <radialGradient id="hubg" cx="38%" cy="30%" r="85%">
            <stop offset="0" stop-color="#5abad6"/><stop offset=".5" stop-color="#1888b0"/>
            <stop offset="1" stop-color="#064c74"/></radialGradient>
          <radialGradient id="sheen"><stop offset="0" stop-color="#fff" stop-opacity=".3"/>
            <stop offset=".7" stop-color="#fff" stop-opacity="0"/></radialGradient>
          <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="14"/></filter>
        </defs>
        <circle r="84" fill="#6edcff" opacity=".6" filter="url(#glow)"/>
        <circle r="84" fill="url(#dpg)"/>
        <circle r="83" fill="none" stroke="#96e2f0" stroke-opacity=".55" stroke-width="1.5"/>
        <ellipse cy="-37" rx="35" ry="18.5" fill="url(#sheen)"/>
        <g stroke="#146e96" stroke-width="3" stroke-linecap="round">
          <line x1="21.4" y1="21.4" x2="57.6" y2="57.6"/><line x1="-21.4" y1="21.4" x2="-57.6" y2="57.6"/>
          <line x1="-21.4" y1="-21.4" x2="-57.6" y2="-57.6"/><line x1="21.4" y1="-21.4" x2="57.6" y2="-57.6"/>
        </g>
        <g fill="none" stroke-linecap="round" stroke-linejoin="round">
          <g stroke="#0a466e" stroke-width="5">
            <polyline points="-7.8,-53.7 0,-63 7.8,-53.7"/><polyline points="53.7,-7.8 63,0 53.7,7.8"/>
            <polyline points="-7.8,53.7 0,63 7.8,53.7"/><polyline points="-53.7,-7.8 -63,0 -53.7,7.8"/></g>
          <g stroke="#fff" stroke-width="3">
            <polyline points="-7.8,-53.7 0,-63 7.8,-53.7"/><polyline points="53.7,-7.8 63,0 53.7,7.8"/>
            <polyline points="-7.8,53.7 0,63 7.8,53.7"/><polyline points="-53.7,-7.8 -63,0 -53.7,7.8"/></g>
        </g>
        <circle r="28" fill="url(#hubg)"/>
        <circle r="27.3" fill="none" stroke="#5abad6" stroke-opacity=".55" stroke-width="1.2"/>
        <text y="5" text-anchor="middle">PAN</text>
        <path class="sec" data-pan="ym" d="M-19.8 -19.8L-59.4 -59.4A84 84 0 0 1 59.4 -59.4L19.8 -19.8A28 28 0 0 0 -19.8 -19.8Z"/>
        <path class="sec" data-pan="xp" d="M19.8 -19.8L59.4 -59.4A84 84 0 0 1 59.4 59.4L19.8 19.8A28 28 0 0 0 19.8 -19.8Z"/>
        <path class="sec" data-pan="yp" d="M19.8 19.8L59.4 59.4A84 84 0 0 1 -59.4 59.4L-19.8 19.8A28 28 0 0 0 19.8 19.8Z"/>
        <path class="sec" data-pan="xm" d="M-19.8 19.8L-59.4 59.4A84 84 0 0 1 -59.4 -59.4L-19.8 -19.8A28 28 0 0 0 -19.8 19.8Z"/>
      </svg>

      <div class="rocker gloss">
        <button class="zp" data-pan="zp" aria-label="Z+ (raise the camera)">
          <svg class="chev" viewBox="-9 -9 18 18"><polyline points="-8,4 0,-5 8,4"/></svg><span>Z+</span></button>
        <i class="rule"></i>
        <button class="zm" data-pan="zm" aria-label="Z- (lower the camera)">
          <svg class="chev" viewBox="-9 -9 18 18"><polyline points="-8,-4 0,5 8,-4"/></svg><span>Z-</span></button>
      </div>

      <div class="lateral na" id="latplate">
        <div class="ttl">LATERAL<br>VIEW</div>
        <button class="switch" id="lateral" data-toggle="lateral" disabled aria-label="Lateral view"><i class="k"></i></button>
        <span class="lcap" id="latcap">connect 2nd camera</span>
      </div>
    </div>

    <div class="card bottom">
      <div class="radio">
        <button id="fs">FULLSCREEN</button>
        <button id="win">WINDOWED</button>
      </div>
      <button class="pill run gloss" id="run">STOP SIMULATOR</button>
    </div>

    <details class="adv">
      <summary>Set the camera position</summary>
      <div class="row">
        <label>X <input id="xin" type="number" step="0.5" value="0"></label>
        <label>Y <input id="yin" type="number" step="0.5" value="0"></label>
        <label>Z <input id="zin" type="number" step="0.5" value="0"></label>
        <span>cm</span>
        <button class="pill" id="gopos" type="button">GO</button>
      </div>
    </details>
  </aside>
</div>

<script>
var ZOOM_REF = {{ zoom_ref }}, FLIP_Z = {{ flip_z }};
var state = {};
function $(id) { return document.getElementById(id); }
function post(url, body) {
  var opt = { method: 'POST' };
  if (body) { opt.headers = { 'Content-Type': 'application/json' }; opt.body = JSON.stringify(body); }
  return fetch(url, opt).then(function (r) { return r.json(); }).then(applyState).catch(function () {});
}
// The readout rounds as the Pi's does (Python's round: halves to even).
function pyRound(v) {
  v = +v || 0;
  var r = Math.round(v);
  if (Math.abs(v % 1) === 0.5) r = 2 * Math.round(v / 2);
  return r === 0 ? 0 : r;
}
function mmss(sec) {
  sec = Math.floor(+sec || 0);
  var m = Math.floor(sec / 60), s = sec % 60;
  return (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
}
function applyState(s) {
  if (!s || s.error) return;
  state = s;
  document.querySelectorAll('.knob[data-toggle]').forEach(function (b) {
    var on = !!s[b.getAttribute('data-toggle')];
    b.classList.toggle('on', on);
    var cap = b.parentNode.querySelector('.cap');
    cap.textContent = on ? 'ON' : 'OFF';
    cap.classList.toggle('on', on);
  });
  var ok = !!s.lateral_available, sw = $('lateral'), lcap = $('latcap');
  sw.disabled = !ok;
  sw.classList.toggle('on', ok && !!s.lateral);
  $('latplate').classList.toggle('na', !ok);
  lcap.textContent = !ok ? 'connect 2nd camera' : (s.lateral ? 'ON' : 'OFF');
  lcap.className = 'lcap' + (!ok ? '' : (s.lateral ? ' on' : ' off'));
  $('vx').textContent = pyRound(s.pos_x_cm);
  $('vy').textContent = pyRound(s.pos_y_cm);
  $('vz').textContent = pyRound(s.pos_z_cm);
  $('clock').textContent = mmss(s.fluoro_sec);
  $('fs').classList.toggle('on', !!s.fullscreen);
  $('win').classList.toggle('on', !s.fullscreen);
  var run = $('run');
  run.classList.toggle('start', !s.running);
  run.textContent = s.running ? 'STOP SIMULATOR' : 'START SIMULATOR';
  var z = +s.pos_z_cm || 0;
  var zoom = ZOOM_REF / Math.max(0.05 * ZOOM_REF, ZOOM_REF + FLIP_Z * z);
  $('zoomnote').textContent = zoom.toFixed(2) + 'x';
  $('zoomnote').classList.toggle('on', !!s.hud);
  $('pedalnote').classList.toggle('on', !!(s.hud && s.pedal_mode));
  $('recnote').classList.toggle('on', !!s.recording);
  if ('rec_elapsed' in s) $('recnote').textContent = 'REC' + (s.rec_elapsed ? ' ' + s.rec_elapsed : '');
  $('field').classList.toggle('wide', !!s.lateral || !s.camera);
  $('camwarn').hidden = !(s.running && !s.camera);
}
function refresh() {
  fetch('/api/state').then(function (r) { return r.json(); }).then(applyState).catch(function () {});
}
document.querySelectorAll('[data-toggle]').forEach(function (b) {
  b.addEventListener('click', function () {
    if (!b.disabled) post('/api/toggle/' + b.getAttribute('data-toggle'));
  });
});
document.querySelectorAll('[data-pan]').forEach(function (b) {
  b.addEventListener('click', function () { post('/api/pan/' + b.getAttribute('data-pan')); });
});
$('reset').addEventListener('click', function () { post('/api/action/fluoro_reset'); });
function browserFullscreen(on) {
  // Fullscreen/Windowed also fullscreen THIS browser's view of the controls
  // (must run inside a click handler).
  var el = $('stage');
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
$('fs').addEventListener('click', function () { browserFullscreen(true); post('/api/action/fullscreen'); });
$('win').addEventListener('click', function () { browserFullscreen(false); post('/api/action/windowed'); });
$('run').addEventListener('click', function () {
  if (!state.running) { post('/api/action/start'); return; }
  if (confirm('Stop the simulator?')) post('/api/action/stop');
});
function sendPos() {
  post('/api/position', { x: parseFloat($('xin').value) || 0, y: parseFloat($('yin').value) || 0,
                          z: parseFloat($('zin').value) || 0 });
}
$('gopos').addEventListener('click', sendPos);
['xin', 'yin', 'zin'].forEach(function (id) {
  $(id).addEventListener('keydown', function (e) { if (e.key === 'Enter') sendPos(); });
});
var feed = $('feed');
feed.addEventListener('load', function () { $('dot').classList.add('on'); });
feed.addEventListener('error', function () { $('dot').classList.remove('on'); });
// Tablets suspend a backgrounded tab's timers and its MJPEG connection, so the
// controls go stale and the picture freezes. Re-sync and restart the stream
// the moment the page is visible again instead of waiting for the next poll.
function resume() {
  refresh();
  feed.src = '/video_feed?t=' + Date.now();
}
document.addEventListener('visibilitychange', function () { if (!document.hidden) resume(); });
window.addEventListener('pageshow', resume);
refresh();
setInterval(refresh, 1000);
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
        PAGE, library_url=portal_url + "/library" if portal_url else None,
        zoom_ref=ZOOM_REF_CM, flip_z=FLIP_Z)


@app.route("/ui-font.ttf")
def ui_font():
    '''The touch screen's font (Nunito Sans, Raspberry Pi OS's own) for the
    panel page, so it matches the device without fetching anything from
    outside it. Absent on other systems: the page falls back to system fonts.'''
    if not os.path.exists(_NUNITO):
        return ("", 404)
    return send_file(_NUNITO, mimetype="font/ttf", max_age=7 * 86400)


@app.route("/api/state")
def api_state():
    '''Return the current toggle states as JSON (polled by the page to stay in
    sync), plus the running recording's length, as the Pi's screen shows it.'''
    snap = get_state_snapshot()
    snap["rec_elapsed"] = recorder.elapsed_str() if recorder.active else None
    return jsonify(snap)


@app.route("/api/toggle/<name>", methods=["POST"])
def api_toggle(name):
    '''Flip one boolean toggle and return the full updated state.

    Only the known toggle names are accepted; anything else is ignored so an
    arbitrary key can't be injected into the state dict.
    '''
    if name == "lateral":
        # Gated: selecting a lateral view with no second camera does nothing.
        with state_lock:
            want = not state["lateral"]
        set_lateral(want)
        return jsonify(get_state_snapshot())
    with state_lock:
        if name in ("overlay", "equalize", "pedal_mode", "hud", "recording"):
            state[name] = not state[name]
        snap = dict(state)
    return jsonify(snap)


@app.route("/api/action/<action>", methods=["POST"])
def api_action(action):
    '''Apply a one-shot action (fullscreen / windowed / start / stop /
    fluoro_reset) and return the state.

    "stop" puts the simulation into standby but keeps this server running so a
    later "start" can resume it remotely. ("quit" is kept as an alias of "stop"
    for old clients — the web panel no longer exits the process.)
    "fluoro_reset" zeroes the fluoro timer.
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
        elif action == "fluoro_reset":
            state["fluoro_sec"] = 0.0
        snap = dict(state)
    return jsonify(snap)


@app.route("/api/position", methods=["POST"])
def api_position():
    '''Set the current camera position, in centimetres from the origin.

    This is the single input hook for the motion source: the position provider
    POSTs the camera's (x, y) here and the processing loop pans the anatomy
    viewport to match. Accepts JSON ``{"x": <cm>, "y": <cm>}`` or plain form/query
    params ``x`` and ``y``. An optional ``z`` (cm) sets the zoom the same way;
    omitting it leaves the current zoom unchanged. A position past the
    anatomy's edges is brought back to them (clamp_position) - the field can
    go no further - and the returned state carries the position as applied.
    '''
    data = request.get_json(silent=True) or {}
    x = data.get("x", request.values.get("x"))
    y = data.get("y", request.values.get("y"))
    z = data.get("z", request.values.get("z"))
    try:
        x, y = float(x), float(y)
        z = float(z) if z is not None else None
        if any(math.isnan(v) or math.isinf(v) for v in (x, y, z) if v is not None):
            raise ValueError("not a finite number")
    except (TypeError, ValueError):
        return jsonify({"error": "provide numeric x and y (cm), optional z (cm)"}), 400
    with state_lock:
        state["pos_x_cm"], state["pos_y_cm"], state["pos_z_cm"] = clamp_position(
            x, y, state["pos_z_cm"] if z is None else z)
        snap = dict(state)
    return jsonify(snap)


@app.route("/api/pan/<name>", methods=["POST"])
def api_pan(name):
    '''One step of the direction pad or the Z rocker (xm, xp, ym, yp, zm, zp):
    the same 5 cm grid and edge stops as the device's own controls (pan_step).
    Returns the full updated state.'''
    if name not in PAN_BUTTONS:
        return jsonify({"error": "no such pan button: %s" % name}), 404
    with state_lock:
        _pan_locked(*PAN_BUTTONS[name])
        snap = dict(state)
    return jsonify(snap)


def mjpeg_generator():
    '''Yield the latest processed frame as a multipart MJPEG stream.

    Only new frames are sent (plus a resend once a second so an idle stream is
    not dropped as dead). Re-sending an unchanged frame just queues bytes in
    front of the next real one on a slow link. Each part is followed at once
    by the next boundary: browsers display a part only when they see the
    boundary that closes it, so sending it up front would hold every frame
    back until the next one arrived.
    '''
    yield b"--frame\r\n"
    sent_seq, sent_at = None, 0.0
    while True:
        wait = sent_at + PREVIEW_MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        seq, buf = latest_preview()
        if buf is None or (seq == sent_seq and time.monotonic() - sent_at < 1.0):
            time.sleep(0.01)
            continue
        sent_seq, sent_at = seq, time.monotonic()
        yield (b"Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(buf)
               + buf + b"\r\n--frame\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache, no-store",
                             "X-Accel-Buffering": "no"})


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


# ── The anatomy's edges ───────────────────────────────────────────────────────
# compute_viewport stops the field at the edges of the master; the position, and
# so its readout, stops there too, or the numbers would run on while the picture
# stood still. (width, height) of the frontal master in px - the calibration is
# in its pixels - set once it is loaded; None until then (nothing is clamped).
MASTER_SIZE = None
ZOOM_MIN = 0.05        # the closest zoom compute_viewport allows


def set_master_size(master):
    global MASTER_SIZE
    MASTER_SIZE = (master.shape[1], master.shape[0])


def zoom_limits(size):
    '''(z_min, z_max) in cm: from the closest zoom to the widest field that
    still fits inside the anatomy whole - wider, it is cut to the master in one
    direction only and the picture is squeezed.'''
    mw, mh = size
    fit = min(mw / (FOV_CM[0] * PX_PER_CM), mh / (FOV_CM[1] * PX_PER_CM))
    ends = [(zoom * ZOOM_REF_CM - ZOOM_REF_CM) / FLIP_Z for zoom in (ZOOM_MIN, fit)]
    return min(ends), max(ends)


def travel_limits(size, z_cm):
    '''((x_min, x_max), (y_min, y_max)) in cm: how far the field can move at
    height z_cm before compute_viewport stops it at the anatomy's edge.'''
    zoom = max(ZOOM_MIN, (ZOOM_REF_CM + FLIP_Z * z_cm) / ZOOM_REF_CM)
    out = []
    for fov, origin, flip, extent in ((FOV_CM[0], ORIGIN_PX[0], FLIP_X, size[0]),
                                      (FOV_CM[1], ORIGIN_PX[1], FLIP_Y, size[1])):
        win = max(1, min(int(round(fov * zoom * PX_PER_CM)), extent))
        ends = [(centre - origin) / (flip * PX_PER_CM)
                for centre in (win / 2.0, extent - win / 2.0)]
        out.append((min(ends), max(ends)))
    return tuple(out)


def clamp_position(x, y, z):
    '''The position (cm) brought inside the anatomy's edges: Z first, then X
    and Y at that height, as zooming out leaves the field less room to move.'''
    if MASTER_SIZE is None:
        return x, y, z
    z_lo, z_hi = zoom_limits(MASTER_SIZE)
    z = min(max(z, z_lo), z_hi)
    (x_lo, x_hi), (y_lo, y_hi) = travel_limits(MASTER_SIZE, z)
    return min(max(x, x_lo), x_hi), min(max(y, y_lo), y_hi), z


def pan_step(pos, axis, direction):
    '''One pan/zoom step from pos (x, y, z): along axis 0/1/2, direction +1/-1.

    It goes to the next PAN_STEP_CM grid line at least half a step away, but
    no further than the anatomy's edge: the readout counts in steps, stops
    where the picture stops, and a step back from an edge lands on the grid.
    '''
    pos = list(pos)
    k = pos[axis] / PAN_STEP_CM
    pos[axis] = (math.ceil(k + 0.5) if direction > 0 else math.floor(k - 0.5)) * PAN_STEP_CM
    return clamp_position(*pos)


def estimate_illumination(gray):
    '''Smooth estimate of the external light field across the camera frame.

    Grayscale dilation takes the local max, so dark objects narrower than
    ILLUM_SPAN_PX are ridden over and read the background brightness behind
    them; the Gaussian blur then smooths the result into the slowly-varying
    light field the flat-field division needs. float32, floored at 1 so the
    caller can divide by it.
    '''
    h, w = gray.shape
    small = cv.resize(gray, (max(1, w // ILLUM_DOWNSCALE),
                             max(1, h // ILLUM_DOWNSCALE)),
                      interpolation=cv.INTER_AREA)
    span = max(3, ILLUM_SPAN_PX // ILLUM_DOWNSCALE) | 1
    k = cv.getStructuringElement(cv.MORPH_RECT, (span, span))
    bright = cv.dilate(small, k)
    bright = cv.GaussianBlur(bright, (0, 0),
                             max(1.0, ILLUM_SIGMA / ILLUM_DOWNSCALE))
    bright = cv.resize(bright, (w, h), interpolation=cv.INTER_LINEAR)
    return np.maximum(bright.astype(np.float32), 1.0)


# Transmission curve as a 256-entry table. The maths is t**ATTEN_GAMMA applied
# per pixel; done as a float power over a 300k-element array it cost ~30ms a
# frame on a Pi 3, which is most of a frame budget and twice that in split
# screen. Through a LUT in OpenCV's vectorised C it is the same curve at 8-bit
# precision: measured against the float version on the device, the largest
# difference anywhere in the image is 1 level out of 255.
_ATTEN_LUT = np.clip((np.arange(256) / 255.0) ** ATTEN_GAMMA * 255.0,
                     0, 255).astype(np.uint8)


def composite_overlay(gray, overlay, equalize):
    '''Composite the anatomy overlay with the camera frame by attenuation.

    Each camera pixel becomes a transmission factor — its brightness relative
    to the estimated illumination at that pixel — contrast-shaped by
    ATTEN_GAMMA, and the anatomy overlay is multiplied by it (see the
    "Attenuation compositing" constants for the model). The upshot: the
    external light source divides out, bone reads darker than tissue (from
    the polarity-inverted master), and the device is darkest of all.
    '''
    # Transmission as 0-255, then shaped through _ATTEN_LUT; cv.divide's scale
    # puts the ratio on the 8-bit scale and saturates rather than wrapping.
    ratio = cv.divide(gray.astype(np.float32), estimate_illumination(gray),
                      scale=255.0)
    trans = cv.LUT(np.clip(ratio, 0, 255).astype(np.uint8), _ATTEN_LUT)
    out = cv.multiply(overlay, trans, scale=1.0 / 255.0, dtype=cv.CV_8U)

    if equalize:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(out)
    return out


# Feathered aperture masks are cached per frame-shape: the field only depends on
# the frame dimensions, so we build the float mask once and reuse it every frame.
_aperture_cache = {}


def aperture_mask(h, w):
    '''Return a cached HxW float32 mask (1 inside the field, 0 outside, feathered edge).'''
    key = (h, w)
    mask = _aperture_cache.get(key)
    if mask is None:
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        half = min(h, w) / 2.0 * APERTURE_SCALE      # half the side / the radius
        feather = max(1.0, half * APERTURE_FEATHER)
        yy, xx = np.ogrid[:h, :w]
        # How far each pixel is outside the edge: measured square-wise for the
        # square (the larger of the two axis offsets) and radially for a circle.
        square = np.maximum(np.abs(yy - cy), np.abs(xx - cx)) - half
        disc = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        if APERTURE_SHAPE == "circle":
            out = disc - half
        elif APERTURE_SHAPE == "square":
            out = square
        else:
            # The flat panel: inside both the square and the larger circle.
            out = np.maximum(square, disc - half * APERTURE_CORNER)
        # Ramp from 1 (inside) to 0 (outside) across the feather band.
        mask = np.clip(0.5 - out / feather, 0.0, 1.0).astype(np.float32)
        _aperture_cache[key] = mask
    return mask


def apply_aperture(img):
    '''Mask the fluoro output to the C-arm's field: a centred, round-cornered
    or plain square (flat panel) or a disc (image intensifier), black outside,
    feathered edge.

    Works on both the single-channel composite and the 3-channel raw frame, and
    returns a new array (the source frame is left untouched).
    '''
    if not APERTURE_ON:
        return img
    h, w = img.shape[:2]
    mask = aperture_mask(h, w)
    if img.ndim == 3:
        mask = mask[:, :, None]
    return (img.astype(np.float32) * mask).astype(np.uint8)


def load_master(path, invert=None):
    '''Load an anatomy master and tone-map it into the compositing space.

    Returns None if the file is missing. The tone mapping (polarity, gamma,
    soften, brightness) is the tuned look — every master goes through it, so
    the frontal and lateral backgrounds match. ``invert`` overrides
    MASTER_INVERT for an image supplied in the other polarity.
    '''
    if invert is None:
        invert = MASTER_INVERT
    # Checked before imread: the lateral master is optional, and imread logs a
    # "can't open/read file" warning of its own for a file that is simply not
    # there yet, which reads like a fault in the launch log.
    if not os.path.exists(path):
        return None
    img = cv.imread(path)
    if img is None:
        return None
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if invert:
        img = 255 - img
    if MASTER_GAMMA != 1.0:
        lut = (np.clip((np.arange(256) / 255.0) ** MASTER_GAMMA, 0.0, 1.0)
               * 255.0).astype(np.uint8)
        img = cv.LUT(img, lut)
    if MASTER_SOFTEN > 0:
        img = cv.GaussianBlur(img, (0, 0), MASTER_SOFTEN)
    if MASTER_BRIGHTNESS != 1.0:
        img = cv.convertScaleAbs(img, alpha=MASTER_BRIGHTNESS, beta=0)
    return img


# ── UI typeface ───────────────────────────────────────────────────────────────
# All on-screen text is set in Nunito Sans - Raspberry Pi OS's own desktop font
# - through OpenCV's FreeType module, in Bold wherever text asks for a heavier
# stroke. Each list of (file, face index) is tried in order; with no FreeType
# module or none of the files, text falls back to OpenCV's own Hershey font,
# so a missing font never breaks the UI. Nunito Sans is a variable font, and
# FreeType opens one of its named weights when the face index carries the
# weight's number in its upper 16 bits: 3 is Regular and 6 Bold (index 0 would
# be ExtraLight; `fc-query <file>` lists them). Sizes are given as
# Hershey-style scales: UI_PX_PER_SCALE pixels of font height per unit, which
# gives the same capital height as Hershey Simplex at that scale.
_NUNITO = "/usr/share/fonts/truetype/nunito-sans/NunitoSans-VariableFont_YTLC,opsz,wdth,wght.ttf"
UI_FONT_FILES = {
    False: ((_NUNITO, 3 << 16),
            ("/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf", 0),
            ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 0),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0)),
    True: ((_NUNITO, 6 << 16),
           ("/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf", 0),
           ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 0),
           ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0)),
}
UI_PX_PER_SCALE = 30.0
_ui_faces = {}


def _ui_face(bold):
    '''The FreeType face for UI text, or None to fall back to Hershey.'''
    if bold not in _ui_faces:
        face = None
        for path, index in UI_FONT_FILES[bold] if hasattr(cv, "freetype") else ():
            if os.path.exists(path):
                try:
                    face = cv.freetype.createFreeType2()
                    face.loadFontData(path, index)
                    break
                except cv.error:
                    face = None
        _ui_faces[bold] = face
    return _ui_faces[bold]


def ui_text_size(text, scale, thick=1):
    '''cv.getTextSize in the UI typeface: ((width, cap height), baseline).'''
    face = _ui_face(thick > 1)
    if face is None:
        return cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, thick)
    return face.getTextSize(text, max(6, int(round(UI_PX_PER_SCALE * scale))), -1)


def ui_put_text(img, text, org, scale, color, thick=1):
    '''cv.putText in the UI typeface; `org` is the left end of the baseline.'''
    face = _ui_face(thick > 1)
    org = (int(org[0]), int(org[1]))
    if face is None:
        cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv.LINE_AA)
    else:
        face.putText(img, text, org, max(6, int(round(UI_PX_PER_SCALE * scale))), color, -1,
                     cv.LINE_AA, True)


def build_filler_master(shape, label="LATERAL MASTER"):
    '''Stand-in background for a view whose master image does not exist yet.

    Deliberately not anatomy: a plain light field, captioned often enough that
    any viewport crop lands on a caption, so nobody can mistake the filler for
    a real projection. Built at the same pixel dimensions as the frontal
    master so the cm -> pixel pan/zoom calibration applies unchanged, and
    already in the post-tone-mapping space that load_master returns.
    '''
    h, w = shape[:2]
    # Light field = little attenuation, so the camera image reads through it.
    yy, xx = np.ogrid[:h, :w]
    vignette = np.sqrt(((yy - h / 2.0) / (h / 2.0)) ** 2
                       + ((xx - w / 2.0) / (w / 2.0)) ** 2)
    img = (206 * np.clip(1.0 - 0.10 * vignette, 0.0, 1.0)).astype(np.uint8)

    # About half the default viewport crop's height, so a caption is always in
    # shot even zoomed in to half the field.
    step = max(120, int(0.52 * FOV_CM[1] * PX_PER_CM))
    for cy in range(step // 2, h, step):
        for row, (text, scale, thick) in enumerate(
                ((label, 1.5, 3), ("PLACEHOLDER - NO IMAGE YET", 0.8, 2))):
            (tw, th), _ = ui_text_size(text, scale, thick)
            ui_put_text(img, text, ((w - tw) // 2, cy + row * 58), scale, 150, thick)
    return img


def render_fluoro_view(frame, master, s, work_w=None):
    '''One camera frame + its anatomy master -> a finished fluoro image.

    The per-view half of the pipeline, shared by the single and split views:
    greyscale, pan the anatomy to the current position, composite by
    attenuation, and mask to the C-arm's field of view. No annotations - the
    caller draws those on the finished (possibly composed) image.

    ``work_w`` caps the width the compositing runs at; None = the camera's own
    resolution, which is what the single frontal view uses.
    '''
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    if work_w and gray.shape[1] > work_w:
        scale = float(work_w) / gray.shape[1]
        gray = cv.resize(gray, (work_w, max(1, int(round(gray.shape[0] * scale)))))
    if not s["overlay"]:
        # Overlay off => the full raw video (bright/white areas intact).
        return apply_aperture(gray)
    crop = compute_viewport(master, s["pos_x_cm"], s["pos_y_cm"], s["pos_z_cm"])
    overlay_frame = cv.resize(crop, (gray.shape[1], gray.shape[0]))
    return apply_aperture(composite_overlay(gray, overlay_frame, s["equalize"]))


def compose_split(left, right):
    '''Lay two finished views side by side with a seam between them.

    Both halves are brought to a common height so the pair reads as one image
    even when the two cameras run at different resolutions.
    '''
    h = max(left.shape[0], right.shape[0])
    halves = []
    for img in (left, right):
        if img.shape[0] != h:
            w = max(1, int(round(img.shape[1] * float(h) / img.shape[0])))
            img = cv.resize(img, (w, h))
        halves.append(img)
    seam = np.full((h, SPLIT_DIVIDER_PX), 45, np.uint8)
    return np.hstack([halves[0], seam, halves[1]])


def label_split(img, left_text, right_text):
    '''Caption each half down its outer edge, reading up the left half and down
    the right one, like book spines.

    In the black side bar beside each half's field, so the labels never sit
    over anatomy (a flat-panel field reaches the corners they used to have),
    and at opposite edges so the two can never be read against the
    wrong half. Not gated on the HUD toggle: mistaking a lateral image for a
    frontal one is exactly the confusion the simulator must never create.
    '''
    scale, thick = 0.6, 2
    h, w = img.shape[:2]
    for text, left in ((left_text, True), (right_text, False)):
        (tw, th), base = ui_text_size(text, scale, thick)
        tag = np.zeros((th + base + 4, tw + 4), np.uint8)
        ui_put_text(tag, text, (2, th + 2), scale, 235, thick)
        tag = cv.rotate(tag, cv.ROTATE_90_COUNTERCLOCKWISE if left
                        else cv.ROTATE_90_CLOCKWISE)
        tag_h, tag_w = tag.shape
        y0 = (h - tag_h) // 2
        x0 = 12 if left else w - tag_w - 12
        region = img[y0:y0 + tag_h, x0:x0 + tag_w]
        np.maximum(region, tag, out=region)    # light text onto the black bar
    return img


def draw_str(dst, target, s):
    '''Draw white text with a black drop-shadow so it stays legible over any frame.'''
    x, y = target
    ui_put_text(dst, s, (x + 1, y + 1), 0.45, (0, 0, 0))
    ui_put_text(dst, s, (x, y), 0.45, (255, 255, 255))


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
    "kb_shift": False,      # on-screen keyboard: next letter uppercase
    "kb_sym": False,        # on-screen keyboard: symbols/digits page
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
    '''Draw one themed button (fill + border + centred label, optional ON/OFF sub-label).

    variant="disabled" dims the button; callers that pass it also leave the
    button out of their hit-box list, so it cannot be pressed.
    '''
    if variant == "disabled":
        bg, bd, tx = C_BG, C_BTN_BD, C_SUBTEXT
    elif variant == "quit":
        bg, bd, tx = C_QUIT_BG, C_QUIT_BD, C_QUIT_TX
    elif on:
        bg, bd, tx = C_ON_BG, C_ON_BD, C_ON_TX
    else:
        bg, bd, tx = C_BTN_BG, C_BTN_BD, C_TEXT
    rounded_rect(img, x, y, w, h, 12, bg, -1)
    rounded_rect(img, x, y, w, h, 12, bd, 1)
    if sub is None:
        (lw, lh), _ = ui_text_size(label, 0.55)
        ui_put_text(img, label, (x + (w - lw) // 2, y + (h + lh) // 2), 0.55, tx)
    else:
        (lw, lh), _ = ui_text_size(label, 0.55)
        ui_put_text(img, label, (x + (w - lw) // 2, y + h // 2 - 2), 0.55, tx)
        (sw, sh), _ = ui_text_size(sub, 0.42)
        ui_put_text(img, sub, (x + (w - sw) // 2, y + h // 2 + 16), 0.42,
                    tx if on else C_SUBTEXT)


def _render_key(arg):
    '''A comparable stand-in for one render argument (see cache_last_render).'''
    if isinstance(arg, dict):
        return tuple(sorted(arg.items()))
    if isinstance(arg, np.ndarray):
        return id(arg)          # the logo: loaded once and never modified
    return arg


def cache_last_render(fn):
    '''Redraw a UI strip only when its inputs change; reuse the last image otherwise.

    The bars are drawn from a handful of flags that change a few times a
    minute, but used to be redrawn every frame - ~28ms a frame at 1920 wide on
    a Pi 3, more than the whole frame budget at 30 fps. The state snapshot is
    part of the key, so any flag change redraws. The image comes back
    read-only: it is shared between frames, so drawing on it would be a bug.
    '''
    last = {"key": None, "out": None}

    @functools.wraps(fn)
    def wrapper(*args):
        key = tuple(_render_key(a) for a in args)
        if last["out"] is None or key != last["key"]:
            img, buttons = fn(*args)
            img.flags.writeable = False
            last["key"], last["out"] = key, (img, tuple(buttons))
        img, buttons = last["out"]
        return img, list(buttons)
    return wrapper


def draw_text_field(img, x, y, w, h, text, focused=False, masked=False,
                    placeholder=""):
    '''Draw a clickable text field (the keyboard types into it while focused).'''
    rounded_rect(img, x, y, w, h, 8, C_BTN_BG, -1)
    rounded_rect(img, x, y, w, h, 8, C_ON_BD if focused else C_BTN_BD, 1)
    shown = ("*" * len(text)) if masked else text
    color = C_TEXT
    if focused:
        shown += "_"
    elif not shown and placeholder:
        shown, color = placeholder, C_SUBTEXT
    # Keep the tail visible when the text outgrows the box.
    while len(shown) > 1 and ui_text_size(shown, 0.55)[0][0] > w - 20:
        shown = shown[1:]
    (_, th), _ = ui_text_size("Ag", 0.55)
    ui_put_text(img, shown, (x + 10, y + (h + th) // 2), 0.55, color)


# ── GE OEC-style screens ──────────────────────────────────────────────────────
# Both modes are modelled on GE's OEC C-arm display. Windowed, the CONTROLS
# window is a touch panel - overlapping angled tabs, glossy round toggles, a
# big-digit readout of the camera position and the fluoro time, a direction
# pad for panning, a rocker for zoom and pill buttons for the rest - and FLUORO
# shows the image alone; the panel is drawn at its design size, GE_W x GE_H
# (the touch panel's tile), and scaled to fit any other window. Fullscreen
# stacks the image between a header strip (the tabs, window mode and Stop),
# TAB_H tall, and a control bar of the same controls, BAR_H tall; both strips
# are laid out for GE_FS_W, drawn wider on a wider screen and scaled down on a
# narrower one. Colours are BGR.
GE_W, GE_H = 450, 574
GE_FS_W, TAB_H, BAR_H = 1024, 44, 126
GE_BG = (14, 11, 9)
GE_CARD = ((128, 118, 106), (98, 88, 77), (84, 75, 66))         # top, middle, bottom
GE_TAB_ON, GE_TAB_OFF, GE_TAB_EDGE = (128, 118, 106), (66, 59, 52), (30, 26, 22)
GE_WHITE, GE_MUTED, GE_DIM = (255, 255, 255), (200, 192, 184), (150, 142, 134)
GE_CYAN = ((236, 214, 128), (200, 160, 32), (126, 82, 6))       # highlight, body, rim
GE_CYAN_OFF = ((170, 150, 80), (120, 98, 38), (70, 56, 20))
GE_CYAN_GLOW = (255, 220, 110)
GE_LAVENDER = ((250, 205, 196), (226, 150, 140), (160, 92, 86))
GE_GREY = ((176, 172, 168), (126, 122, 118), (82, 78, 74))       # Lateral view's plate
GE_GREY_OFF = ((110, 106, 102), (82, 78, 74), (56, 53, 50))
GE_KNOB = ((255, 255, 255), (232, 230, 228), (182, 178, 174))     # its switch knob
GE_KNOB_OFF = ((150, 146, 142), (120, 116, 112), (90, 86, 82))
GE_RED = ((150, 150, 255), (70, 70, 230), (30, 30, 150))
GE_RED_OFF = ((120, 110, 170), (80, 70, 130), (50, 45, 90))
GE_RED_GLOW = (90, 90, 255)
GE_STOP, GE_STOP_RIM = ((70, 60, 120), (44, 36, 90), (30, 24, 60)), (90, 90, 245)
GE_YELLOW = (40, 215, 245)
# The state flags the panel draws from (see ge_panel_state).
GE_PANEL_KEYS = ("overlay", "equalize", "hud", "pedal_mode", "recording", "lateral",
                 "lateral_available", "fullscreen", "ui_view",
                 "pos_x_cm", "pos_y_cm", "pos_z_cm")


def _blend(dst, alpha, color, x0, y0):
    '''Alpha-blend `color` (BGR, or an HxWx3 image) onto dst through `alpha` (HxW, 0-1).'''
    h, w = alpha.shape
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0 + w, dst.shape[1]), min(y0 + h, dst.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    a = alpha[y1 - y0:y2 - y0, x1 - x0:x2 - x0, None]
    col = np.asarray(color, np.float32)
    if col.ndim == 3:
        col = col[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    roi = dst[y1:y2, x1:x2].astype(np.float32)
    dst[y1:y2, x1:x2] = (roi * (1 - a) + col * a).astype(np.uint8)


def _box_sdf(w, h, r):
    '''Signed distance in px to a w x h box with corners of radius r (r = h/2: a pill).'''
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    qx = np.abs(xx - (w - 1) / 2.0) - (w / 2.0 - r)
    qy = np.abs(yy - (h - 1) / 2.0) - (h / 2.0 - r)
    out = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
    return out + np.minimum(np.maximum(qx, qy), 0) - r


def _shade(w, h, stops, radial):
    '''A w x h gradient through three colours: top to bottom, or radial from a
    highlight up and to the left of centre.'''
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if radial:
        t = np.sqrt(((xx - 0.38 * w) / w) ** 2 + ((yy - 0.30 * h) / h) ** 2) / 0.85
    else:
        t = yy / max(1, h - 1)
    t = np.clip(t, 0, 1)[..., None]
    c0, c1, c2 = (np.array(c, np.float32) for c in stops)
    return np.where(t < 0.5, c0 + (c1 - c0) * (t / 0.5), c1 + (c2 - c1) * ((t - 0.5) / 0.5))


def _glossy(img, x, y, w, h, r, stops, glow=None, radial=False):
    '''A glossy button: an optional soft glow, a gradient body, a light rim
    and a sheen across the top.'''
    if glow is not None:
        pad = int(0.35 * min(w, h)) + 6
        g = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
        g[pad:pad + h, pad:pad + w] = np.clip(0.5 - _box_sdf(w, h, r), 0, 1)
        _blend(img, cv.GaussianBlur(g, (0, 0), pad / 2.5) * 0.85, glow, x - pad, y - pad)
    d = _box_sdf(w, h, r)
    _blend(img, np.clip(0.5 - d, 0, 1), _shade(w, h, stops, radial), x, y)
    _blend(img, np.clip(1.0 - np.abs(d + 1.2), 0, 1) * 0.55, stops[0], x, y)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sheen = 1 - ((xx - w / 2.0) / (0.42 * w)) ** 2 - ((yy - 0.28 * h) / (0.22 * h)) ** 2
    _blend(img, np.clip(sheen, 0, 1) ** 1.5 * 0.30 * np.clip(-d / 3, 0, 1), GE_WHITE, x, y)


def _ge_text(img, text, cx, cy, max_w, scale, color=GE_WHITE, bold=True):
    '''Flat text centred on (cx, cy), shrunk to fit max_w.'''
    thick = 2 if bold else 1
    (tw, th), _ = ui_text_size(text, scale, thick)
    if tw > max_w:
        scale *= max_w / float(tw)
        (tw, th), _ = ui_text_size(text, scale, thick)
    ui_put_text(img, text, (cx - tw / 2.0, cy + th / 2.0), scale, color, thick)


def _ge_card(img, x, y, w, h):
    '''One of the slate panels the controls are grouped on.'''
    _blend(img, np.clip(0.5 - _box_sdf(w, h, 14), 0, 1), _shade(w, h, GE_CARD, False), x, y)


def _chevron(img, cx, cy, size, angle, color, thick):
    '''A ">" pointing at `angle` degrees (0 = right, 90 = down).'''
    a = np.deg2rad(angle)
    tip = np.array([cx + size * np.cos(a), cy + size * np.sin(a)])
    for da in (np.deg2rad(140), -np.deg2rad(140)):
        end = tip + size * 1.1 * np.array([np.cos(a + da), np.sin(a + da)])
        cv.line(img, tuple(int(v) for v in tip), tuple(int(v) for v in end), color, thick,
                cv.LINE_AA)


def _ge_dpad(img, cx, cy, R, buttons):
    '''The direction pad: four arrow sectors round a hub, panning like the
    keyboard - up is Y- (W), down Y+ (S), left X- (A), right X+ (D).'''
    _glossy(img, cx - R, cy - R, 2 * R, 2 * R, R,
            ((240, 226, 150), (206, 168, 40), (140, 94, 8)), glow=GE_CYAN_GLOW, radial=True)
    for ang in (45, 135, 225, 315):
        a = np.deg2rad(ang)
        cv.line(img, (int(cx + 0.36 * R * np.cos(a)), int(cy + 0.36 * R * np.sin(a))),
                (int(cx + 0.97 * R * np.cos(a)), int(cy + 0.97 * R * np.sin(a))),
                (150, 110, 20), 3, cv.LINE_AA)
    for ang, name in ((270, "ym"), (0, "xp"), (90, "yp"), (180, "xm")):
        a = np.deg2rad(ang)
        px, py = cx + 0.62 * R * np.cos(a), cy + 0.62 * R * np.sin(a)
        _chevron(img, px, py, 0.13 * R, ang, (110, 70, 10), 5)
        _chevron(img, px, py, 0.13 * R, ang, GE_WHITE, 3)
        buttons.append((int(px - 0.3 * R), int(py - 0.3 * R), int(0.6 * R), int(0.6 * R),
                        "pan", name))
    hub = int(0.33 * R)
    _glossy(img, cx - hub, cy - hub, 2 * hub, 2 * hub, hub,
            ((214, 186, 90), (176, 136, 24), (116, 76, 6)), radial=True)
    _ge_text(img, "PAN", cx, cy, 1.6 * hub, 0.5)


def fmt_fluoro(sec):
    '''Fluoro time as mm:ss (the minutes keep counting past 99).'''
    return "%02d:%02d" % divmod(int(sec), 60)


def ge_panel_state(s):
    '''What render_ge_panel draws from: the state flags, plus the fluoro time
    in whole seconds - passed like that, the cached panel is redrawn once a
    second while the beam is on rather than on every frame.'''
    p = {k: s[k] for k in GE_PANEL_KEYS}
    p["fluoro_sec"] = int(s["fluoro_sec"])
    return p


GE_TABS = (("FLUORO", "fluoro"), ("REMOTE", "remote"), ("LIBRARY", "library"))
GE_TOGGLES = (("OVERLAY", "overlay"), ("EQUALIZE", "equalize"), ("HUD", "hud"),
              ("PEDAL", "pedal_mode"), ("REC", "recording"))


def _ge_tabs(img, view, x, top, bottom, tw, lap, scale, buttons):
    '''The overlapping angled tabs from x, the selected one in front, standing
    on a shelf at `bottom` that spans the image.'''
    active = next((i for i, (_, n) in enumerate(GE_TABS) if n == view), 0)
    order = sorted(range(len(GE_TABS)), key=lambda i: -abs(i - active))   # back to front
    slant = (bottom - top) * 18 // 39
    for i in order:
        x0 = x + i * (tw - lap)
        poly = np.array([(x0, bottom), (x0 + slant, top), (x0 + tw - slant, top),
                         (x0 + tw, bottom)], np.int32)
        cv.fillPoly(img, [poly], GE_TAB_ON if i == active else GE_TAB_OFF, cv.LINE_AA)
        cv.polylines(img, [poly], False, GE_TAB_EDGE, 1, cv.LINE_AA)
        _ge_text(img, GE_TABS[i][0], x0 + tw / 2.0, (top + bottom) / 2.0 + 1, tw - 48, scale,
                 GE_WHITE if i == active else GE_MUTED, bold=(i == active))
    img[bottom:bottom + 3] = GE_TAB_ON
    for i in reversed(order):                   # front tab first: overlaps go to it
        buttons.append((x + i * (tw - lap), top, tw, bottom + 3 - top, "tab", GE_TABS[i][1]))


def _ge_toggles(img, p, x, y, w, dia, caption_dy, scale, caption_scale, buttons):
    '''The five glossy round toggles spread across w from x, tops at y, lit
    when on, each with ON/OFF caption_dy below its top.'''
    gap = (w - 5 * dia) / 6.0
    for i, (text, key) in enumerate(GE_TOGGLES):
        bx = int(round(x + gap + i * (dia + gap)))
        on = bool(p[key])
        if key == "recording":
            stops, glow = (GE_RED, GE_RED_GLOW) if on else (GE_RED_OFF, None)
        else:
            stops, glow = (GE_CYAN, GE_CYAN_GLOW) if on else (GE_CYAN_OFF, None)
        _glossy(img, bx, y, dia, dia, dia // 2, stops, glow=glow, radial=True)
        _ge_text(img, text, bx + dia / 2.0, y + dia / 2.0, dia - 12, scale,
                 GE_WHITE if on else GE_MUTED)
        _ge_text(img, "ON" if on else "OFF", bx + dia / 2.0, y + caption_dy, dia, caption_scale,
                 (190, 255, 210) if on else GE_DIM, bold=False)
        buttons.append((bx, y, dia, dia, "toggle", key))


def _ge_readout(img, p, x, y, w, h, col, scale, buttons):
    '''The readout band, w x h: the camera position (X, Y, Z in cm) in
    col-wide columns, then the fluoro time with its Reset button.'''
    d = _box_sdf(w, h, 14)
    _blend(img, np.clip(0.5 - d, 0, 1), (10, 8, 7), x, y)
    _blend(img, np.clip(1.2 - np.abs(d + 1.5), 0, 1), (225, 220, 214), x, y)
    for i, (axis, key) in enumerate((("X", "pos_x_cm"), ("Y", "pos_y_cm"), ("Z", "pos_z_cm"))):
        cx = x + i * col
        ui_put_text(img, axis + "  cm", (cx + 0.17 * col, y + 0.26 * h), 0.4 * scale, GE_DIM)
        val = "%d" % round(p[key])
        (vw, _), _ = ui_text_size(val, 0.95 * scale)
        ui_put_text(img, val, (cx + (col - vw) / 2.0 + 4 * scale, y + 0.76 * h), 0.95 * scale,
                    GE_WHITE)
        cv.line(img, (int(cx + col), int(y + 0.17 * h)), (int(cx + col), int(y + 0.83 * h)),
                (70, 64, 58), 1)
    tx = x + 3 * col
    clock = fmt_fluoro(p["fluoro_sec"])
    (cw, _), _ = ui_text_size(clock, 1.35 * scale)
    ui_put_text(img, clock, (tx + 14 * scale, y + 0.74 * h), 1.35 * scale, GE_WHITE)
    ui_put_text(img, "mm:ss", (tx + 20 * scale + cw, y + 0.74 * h), 0.4 * scale, GE_DIM)
    bw, bh = int(58 * scale), int(24 * scale)
    bx, by = x + w - bw - int(12 * scale), y + int(8 * scale)
    _glossy(img, bx, by, bw, bh, bh // 2, GE_CYAN_OFF)
    _ge_text(img, "RESET", bx + bw / 2.0, by + bh / 2.0, bw - 12 * scale, 0.36 * scale, GE_MUTED)
    buttons.append((bx - 4, by - 4, bw + 8, bh + 8, "action", "fluoro_reset"))


def _ge_zrocker(img, x, y, w, h, chev, scale, buttons):
    '''The lavender Z rocker: Z+ (raise the camera) above, Z- below.'''
    _glossy(img, x, y, w, h, w // 2, GE_LAVENDER)
    cv.line(img, (x + w // 6, y + h // 2), (x + w - w // 6, y + h // 2), (150, 90, 84), 2, cv.LINE_AA)
    _chevron(img, x + w / 2.0, y + 0.2 * h, chev, 270, GE_WHITE, 3 if chev > 8 else 2)
    _chevron(img, x + w / 2.0, y + 0.8 * h, chev, 90, GE_WHITE, 3 if chev > 8 else 2)
    _ge_text(img, "Z+", x + w / 2.0, y + 0.36 * h, w - 14, scale)
    _ge_text(img, "Z-", x + w / 2.0, y + 0.64 * h, w - 14, scale)
    buttons += [(x, y, w, h // 2, "pan", "zp"), (x, y + h // 2, w, h - h // 2, "pan", "zm")]


def _ge_switch(img, x, y, w, h, on, ok):
    '''An on/off slide switch: the knob sits right, on a lit track, when on.'''
    d = _box_sdf(w, h, h // 2)
    _blend(img, np.clip(0.5 - d, 0, 1),
           GE_CYAN[1] if on else (58, 54, 50) if ok else (70, 66, 62), x, y)
    _blend(img, np.clip(1.2 - np.abs(d + 1.2), 0, 1) * 0.6, (30, 28, 26), x, y)
    kd = int(h * 0.81)
    inset = (h - kd) // 2
    kx = x + (w - kd - inset if on else inset)
    _blend(img, np.clip(0.5 - _box_sdf(kd, kd, kd // 2), 0, 1) * 0.45, (20, 18, 16),
           kx + 1, y + inset + 2)
    _glossy(img, kx, y + inset, kd, kd, kd // 2, GE_KNOB if ok else GE_KNOB_OFF, radial=True)


def _ge_lateral_caption(p):
    '''What the Lateral view switch says under it, and in which colour.'''
    if not p["lateral_available"]:
        return "connect 2nd camera", GE_DIM
    return ("ON", (190, 255, 210)) if p["lateral"] else ("OFF", GE_MUTED)


def _ge_stop(img, x, y, w, h, scale, buttons):
    '''The Stop simulator pill: dark red with a red rim.'''
    _glossy(img, x, y, w, h, h // 2, GE_STOP)
    _blend(img, np.clip(1.4 - np.abs(_box_sdf(w, h, h // 2) + 1.5), 0, 1), GE_STOP_RIM, x, y)
    _ge_text(img, "STOP SIMULATOR", x + w / 2.0, y + h / 2.0, w - 24, scale, (190, 190, 255))
    buttons.append((x, y, w, h, "action", "stop"))


@cache_last_render
def render_ge_panel(p, width, height):
    '''The windowed CONTROLS panel, GE OEC style, fitted into width x height.

    `p` comes from ge_panel_state. Returns (image, buttons) with hit-boxes
    relative to the image's top-left.
    '''
    w, h, m = GE_W, GE_H, 12
    img = np.full((h, w, 3), GE_BG, np.uint8)
    buttons = []
    _ge_tabs(img, p["ui_view"], 8, 8, 47, 162, 26, 0.55, buttons)

    # Imaging toggles: glossy round buttons, lit when on.
    y = 56
    _ge_card(img, m, y, w - 2 * m, 106)
    _ge_toggles(img, p, m, y + 10, w - 2 * m, 70, 84, 0.5, 0.38, buttons)

    # Readout: the camera position and the fluoro time in big digits.
    y += 106 + 10
    _ge_readout(img, p, m, y, w - 2 * m, 84, 80, 1.0, buttons)

    # Movement: the direction pad for X/Y, a rocker for Z, and Lateral view.
    y += 84 + 10
    mh = 196
    _ge_card(img, m, y, w - 2 * m, mh)
    R = 84
    _ge_dpad(img, m + 16 + R, y + mh // 2, R, buttons)
    zx, zw, zh = m + 16 + 2 * R + 22, 62, 172
    zy = y + (mh - zh) // 2
    _ge_zrocker(img, zx, zy, zw, zh, 11, 0.45, buttons)
    # Lateral view: an on/off switch on a grey plate, dimmed and unpressable
    # with fewer than two cameras; the whole plate is the hit area.
    lx = zx + zw + 16
    lw = w - m - 16 - lx
    lat_ok = bool(p["lateral_available"])
    _blend(img, np.clip(0.5 - _box_sdf(lw, zh, 20), 0, 1),
           _shade(lw, zh, GE_GREY if lat_ok else GE_GREY_OFF, False), lx, zy)
    text_c = GE_WHITE if lat_ok else GE_DIM
    _ge_text(img, "LATERAL", lx + lw / 2.0, zy + 34, lw - 16, 0.55, text_c)
    _ge_text(img, "VIEW", lx + lw / 2.0, zy + 58, lw - 16, 0.55, text_c)
    _ge_switch(img, lx + (lw - 84) // 2, zy + 84, 84, 42, bool(p["lateral"]), lat_ok)
    cap, cap_c = _ge_lateral_caption(p)
    _ge_text(img, cap, lx + lw / 2.0, zy + zh - 22, lw - 16, 0.42 if lat_ok else 0.36, cap_c,
             bold=False)
    if lat_ok:
        buttons.append((lx, zy, lw, zh, "toggle", "lateral"))

    # Window mode, a pair of radio options like GE's NORM / MAG, and Stop.
    y += mh + 10
    ch = h - y - m
    _ge_card(img, m, y, w - 2 * m, ch)
    for i, (text, name, on) in enumerate((("FULLSCREEN", "fullscreen", p["fullscreen"]),
                                          ("WINDOWED", "windowed", not p["fullscreen"]))):
        ry = y + ch // 2 - 17 + i * 34
        cv.circle(img, (m + 28, ry), 9, GE_WHITE, 2, cv.LINE_AA)
        if on:
            cv.circle(img, (m + 28, ry), 5, GE_WHITE, -1, cv.LINE_AA)
        ui_put_text(img, text, (m + 46, ry + 6), 0.5, GE_WHITE if on else GE_MUTED, 2 if on else 1)
        buttons.append((m + 12, ry - 16, 170, 32, "action", name))
    _ge_stop(img, m + 196, y + (ch - 50) // 2, w - 2 * m - 196 - 14, 50, 0.5, buttons)

    # Fit the design-size panel into the window.
    k = min(width / float(w), height / float(h))
    if abs(k - 1.0) > 1e-3:
        img = cv.resize(img, (max(1, int(w * k)), max(1, int(h * k))),
                        interpolation=cv.INTER_AREA if k < 1 else cv.INTER_LINEAR)
        buttons = [(int(x * k), int(y * k), int(bw * k), int(bh * k), kind, name)
                   for (x, y, bw, bh, kind, name) in buttons]
    return img, buttons


def render_ge_image(src, width, height, s, logo, split, live, rec_elapsed):
    '''The picture at width x height, GE style: no HUD text on it - the logo
    and the HUD's notes sit in its corners instead. A frame wider than the
    space is cut to its centre (the field is square), so windowed mode's
    square shows the field alone; otherwise - and always for the split view -
    it is letterboxed whole, with the notes in the black beside it.'''
    if src is None:
        return np.zeros((height, width, 3), np.uint8)
    bgr = cv.cvtColor(src, cv.COLOR_GRAY2BGR) if src.ndim == 2 else src
    h, w = bgr.shape[:2]
    if not split and w * height > h * width:
        cw = int(round(h * width / float(height)))
        x0 = (w - cw) // 2
        img = cv.resize(bgr[:, x0:x0 + cw], (width, height), interpolation=cv.INTER_LINEAR)
    else:
        img = fit_frame(bgr, width, height)
    if logo is not None:
        paste_rgba(img, logo, 14, 14, min(120, max(60, width // 6)))
    cv.circle(img, (width - 20, 20), 6, C_DOT_ON if live else C_DOT_OFF, -1, cv.LINE_AA)
    if rec_elapsed is not None:
        txt = "REC %s" % rec_elapsed
        (tw, th), _ = ui_text_size(txt, 0.5)
        cv.circle(img, (width - 54 - tw, 20), 6, C_DOT_OFF, -1, cv.LINE_AA)
        ui_put_text(img, txt, (width - 40 - tw, 20 + th // 2), 0.5, C_QUIT_TX)
    if s["hud"]:
        # A steady note: the pedal reaches the loop as a stream of 'b' presses,
        # so a note that followed them would flicker while it is held.
        if s["pedal_mode"]:
            ui_put_text(img, "PEDAL MODE", (16, height - 16), 0.5, GE_YELLOW)
        zoom = ZOOM_REF_CM / max(0.05 * ZOOM_REF_CM, ZOOM_REF_CM + FLIP_Z * s["pos_z_cm"])
        mag = "%.2fx" % zoom
        (zw, _), _ = ui_text_size(mag, 0.5)
        ui_put_text(img, mag, (width - zw - 16, height - 16), 0.5, GE_MUTED)
    return img


def render_tab_view(view, s, width, height):
    '''The Remote Access or Library view at width x height, with the on-screen
    keyboard over its bottom while a text field has focus. -> (image, buttons)'''
    if view == "remote":
        canvas, vbtns = render_remote_view(s, width, height)
    else:
        canvas, vbtns = render_library_view(width, height)
    if ui["focus"] and view == "remote":
        # A focused text field pops the on-screen keyboard over the bottom of
        # the view (its echo strip keeps the typed text visible even when the
        # field itself is covered).
        kb, kbtns = render_keyboard(width, ui[ui["focus"]],
                                    ui["focus"] == "password" and not ui["show_pw"])
        kh = kb.shape[0]
        if kh < canvas.shape[0]:
            canvas[-kh:] = kb
            vbtns = list(vbtns) + [(x, y + canvas.shape[0] - kh, w, h, k, n)
                                   for (x, y, w, h, k, n) in kbtns]
    return canvas, vbtns


def _ge_fit(img, buttons, width):
    '''Scale a design-width strip, and its hit-boxes, down to `width`.'''
    k = width / float(img.shape[1])
    img = cv.resize(img, (width, max(1, int(round(img.shape[0] * k)))), interpolation=cv.INTER_AREA)
    return img, [(int(x * k), int(y * k), int(w * k), int(h * k), kind, name)
                 for (x, y, w, h, kind, name) in buttons]


@cache_last_render
def render_ge_header(p, width, rec_elapsed):
    '''Fullscreen's top strip, TAB_H tall: the overlapping tabs on the left;
    the Fullscreen | Windowed switch and Stop on the right, with the
    recording's time between while one runs. -> (image, buttons)'''
    W = max(width, GE_FS_W)
    img = np.full((TAB_H, W, 3), GE_BG, np.uint8)
    buttons = []
    _ge_tabs(img, p["ui_view"], 8, 6, TAB_H - 3, 150, 24, 0.5, buttons)
    sw, sh = 168, 30
    sx, sy = W - 8 - sw, 6
    _ge_stop(img, sx, sy, sw, sh, 0.42, buttons)
    segw = 116                                  # the two halves of the window switch
    gx = sx - 14 - 2 * segw
    _blend(img, np.clip(0.5 - _box_sdf(2 * segw, sh, sh // 2), 0, 1), (40, 36, 32), gx, sy)
    hx = gx if p["fullscreen"] else gx + segw
    _blend(img, np.clip(0.5 - _box_sdf(segw, sh, sh // 2), 0, 1),
           _shade(segw, sh, GE_CARD, False), hx, sy)
    for k, (text, name) in enumerate((("FULLSCREEN", "fullscreen"), ("WINDOWED", "windowed"))):
        sel = (k == 0) == bool(p["fullscreen"])
        _ge_text(img, text, gx + segw * (k + 0.5), sy + sh / 2.0, segw - 20, 0.4,
                 GE_WHITE if sel else GE_DIM, bold=sel)
        buttons.append((gx + k * segw, sy, segw, sh, "action", name))
    if rec_elapsed is not None:
        txt = "REC %s" % rec_elapsed
        (rw, rh), _ = ui_text_size(txt, 0.5)
        rx = gx - 24 - rw
        cv.circle(img, (rx - 14, TAB_H // 2), 6, C_DOT_OFF, -1, cv.LINE_AA)
        ui_put_text(img, txt, (rx, TAB_H // 2 + rh // 2), 0.5, C_QUIT_TX)
    return _ge_fit(img, buttons, width) if W != width else (img, buttons)


@cache_last_render
def render_ge_bar(p, width):
    '''Fullscreen's control bar, BAR_H tall: slate cards for the round
    toggles, the readout, the direction pad with the Z rocker, and the
    Lateral view switch. Laid out for GE_FS_W: centred on a wider screen and
    scaled down on a narrower one. -> (image, buttons)'''
    W = max(width, GE_FS_W)
    img = np.full((BAR_H, W, 3), GE_BG, np.uint8)
    buttons = []
    m, gap, y = 8, 8, 8
    ch = BAR_H - 2 * y
    x = (W - GE_FS_W) // 2 + m

    aw = 374                                    # toggles
    _ge_card(img, x, y, aw, ch)
    _ge_toggles(img, p, x, y + 10, aw, 62, 87, 0.45, 0.34, buttons)
    x += aw + gap

    rw = 296                                    # readout
    _ge_card(img, x, y, rw, ch)
    _ge_readout(img, p, x + 8, y + 8, rw - 16, ch - 16, 46, 0.8, buttons)
    x += rw + gap

    mw = 176                                    # direction pad + Z rocker
    _ge_card(img, x, y, mw, ch)
    R = 50
    _ge_dpad(img, x + 10 + R, y + ch // 2, R, buttons)
    _ge_zrocker(img, x + 10 + 2 * R + 14, y + 7, 38, ch - 14, 7, 0.34, buttons)
    x += mw + gap

    lw = (W - GE_FS_W) // 2 + GE_FS_W - m - x  # Lateral view switch
    lat_ok = bool(p["lateral_available"])
    _blend(img, np.clip(0.5 - _box_sdf(lw, ch, 14), 0, 1),
           _shade(lw, ch, GE_GREY if lat_ok else GE_GREY_OFF, False), x, y)
    _ge_text(img, "LATERAL VIEW", x + lw / 2.0, y + 20, lw - 16, 0.42,
             GE_WHITE if lat_ok else GE_DIM)
    _ge_switch(img, x + (lw - 70) // 2, y + 38, 70, 34, bool(p["lateral"]), lat_ok)
    cap, cap_c = _ge_lateral_caption(p)
    _ge_text(img, cap, x + lw / 2.0, y + ch - 13, lw - 14, 0.34 if lat_ok else 0.3, cap_c,
             bold=False)
    if lat_ok:
        buttons.append((x, y, lw, ch, "toggle", "lateral"))
    return _ge_fit(img, buttons, width) if W != width else (img, buttons)


def fit_frame(vid, W, H):
    '''Letterbox a BGR frame onto a W×H canvas (aspect kept, black bars).'''
    scale = min(W / float(vid.shape[1]), H / float(vid.shape[0]))
    tw = max(1, int(vid.shape[1] * scale))
    th = max(1, int(vid.shape[0] * scale))
    canvas = np.zeros((H, W, 3), np.uint8)
    x0, y0 = (W - tw) // 2, (H - th) // 2
    canvas[y0:y0 + th, x0:x0 + tw] = cv.resize(vid, (tw, th))
    return canvas


# ── On-screen touch keyboard ──────────────────────────────────────────────────
# Drawn across the bottom of the Remote Access view whenever a text field has
# focus (i.e. it pops up the moment a field is tapped), so a touch display
# needs no physical keyboard. Key taps feed the same ui[focus] buffer the
# physical keyboard types into — both input paths work at once. An echo strip
# above the keys always shows what's been typed, because the keyboard may
# cover the focused field itself.

KB_ROWS_ABC = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
KB_ROWS_SYM = ["1234567890", "@#$%&*-_+=", "!?/:;()'\""]


def render_keyboard(W, value, masked):
    '''Render the touch keyboard + echo strip. Returns (image, buttons).'''
    gap = 6
    row_h = max(40, min(56, W // 18))
    echo_h = 36
    img = np.full((echo_h + 4 * (row_h + gap) + gap, W, 3), C_BG, np.uint8)
    cv.line(img, (0, 0), (W, 0), C_BTN_BD, 1, cv.LINE_AA)
    buttons = []

    shown = ("*" * len(value)) if masked else value
    shown += "_"
    while len(shown) > 1 and ui_text_size(shown, 0.6)[0][0] > W - 28:
        shown = shown[1:]  # keep the tail visible
    ui_put_text(img, shown, (14, 26), 0.6, C_ON_TX)

    def key_row(keys, y):
        '''keys = [(label, name, weight, on)]; weights share the row's width.'''
        kw = (W - gap * (len(keys) + 1)) / float(sum(k[2] for k in keys))
        x = float(gap)
        for label, name, weight, on in keys:
            w = int(kw * weight)
            draw_button(img, int(x), y, w, row_h, label, None, on)
            buttons.append((int(x), y, w, row_h, "key", name))
            x += w + gap

    rows = KB_ROWS_SYM if ui["kb_sym"] else KB_ROWS_ABC
    y = echo_h + gap
    for r, chars in enumerate(rows):
        keys = [("Shift", "shift", 1.4, ui["kb_shift"])] if r == 2 else []
        for ch in chars:
            label = ch.upper() if ui["kb_shift"] and not ui["kb_sym"] else ch
            keys.append((label, "c:" + ch, 1.0, False))
        if r == 2:
            keys.append(("<--", "bksp", 1.4, False))
        key_row(keys, y)
        y += row_h + gap
    key_row([("abc" if ui["kb_sym"] else "?123", "sym", 1.4, ui["kb_sym"]),
             ("@", "c:@", 1.0, False), ("space", "space", 3.6, False),
             (".", "c:.", 1.0, False), ("Done", "done", 1.6, True),
             ("Hide", "hide", 1.2, False)], y)
    return img, buttons


def handle_key_button(name):
    '''Apply an on-screen keyboard tap to the focused text field.'''
    focus = ui["focus"]
    if name == "hide":
        ui["focus"] = None
        return
    if focus is None:
        return
    if name == "shift":
        ui["kb_shift"] = not ui["kb_shift"]
    elif name == "sym":
        ui["kb_sym"] = not ui["kb_sym"]
    elif name == "bksp":
        ui[focus] = ui[focus][:-1]
    elif name == "done":
        ui["focus"] = None
        submit_field(focus)
    elif name == "space":
        ui[focus] += " "
    elif name.startswith("c:"):
        ch = name[2:]
        if ui["kb_shift"]:
            ch = ch.upper()
            ui["kb_shift"] = False  # one-shot, like a phone keyboard
        ui[focus] += ch


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

    def text(txt, x, yy, scale=0.55, color=C_TEXT):
        ui_put_text(img, txt, (x, yy), scale, color)

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
            (cw_, _), _ = ui_text_size(chip[0], 0.5)
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
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))


def render_library_view(W, H):
    '''Render the Library view (list or playback). Returns (canvas, buttons).'''
    library.tick()
    img = np.full((H, W, 3), C_BG, np.uint8)
    buttons = []
    m, gap, bh = 24, 8, 38

    def text(txt, x, yy, scale=0.55, color=C_TEXT):
        ui_put_text(img, txt, (x, yy), scale, color)

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
                (pw, ph), _ = ui_text_size(pct, 0.3)
                ui_put_text(img, pct, (cx - pw // 2, cy + ph // 2), 0.3, C_TEXT)
        elif st == "failed":
            cv.line(img, (cx - 5, cy - 5), (cx + 5, cy + 5), C_DOT_OFF, 2, cv.LINE_AA)
            cv.line(img, (cx - 5, cy + 5), (cx + 5, cy - 5), C_DOT_OFF, 2, cv.LINE_AA)
        elif st in ("queued", "retrying"):
            cv.circle(img, (cx, cy), 5, C_DOT_WAIT, -1, cv.LINE_AA)
        else:
            cv.circle(img, (cx, cy), 5, C_BTN_BD, -1, cv.LINE_AA)
        # Columns are laid out right-to-left from the buttons' edge, each
        # placed by its measured text width, so the duration can't overwrite
        # the timestamp and neither can slide underneath Play/Delete;
        # whatever width is left belongs to the (truncated) file name.
        bx = W - m - 200
        when = _fmt_when(e["mtime"])
        dur = library.duration(e["path"])
        dur_x = bx - ui_text_size(dur, 0.45)[0][0] - 12
        date_x = dur_x - ui_text_size(when, 0.45)[0][0] - 16
        text(when, date_x, y + 25, 0.45, C_SUBTEXT)
        text(dur, dur_x, y + 25, 0.45, C_SUBTEXT)
        name = e["name"]
        name_w = date_x - (m + 28) - 10
        while len(name) > 4 and ui_text_size(name, 0.5)[0][0] > name_w:
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


# The direction pad and Z rocker: button -> (axis, direction) for pan_step.
PAN_BUTTONS = {"xm": (0, -1), "xp": (0, 1), "ym": (1, -1), "yp": (1, 1),
               "zm": (2, -1), "zp": (2, 1)}


def _pan_locked(axis, direction):
    '''One pan/zoom step of the shared position (the caller holds state_lock).'''
    state["pos_x_cm"], state["pos_y_cm"], state["pos_z_cm"] = pan_step(
        (state["pos_x_cm"], state["pos_y_cm"], state["pos_z_cm"]), axis, direction)


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
    if kind == "key":
        handle_key_button(name)
        return
    if kind in ("setup", "lib"):
        # Remote Access / Library view buttons — handled outside state_lock
        # (they talk to their own managers, which have their own locks).
        handle_view_button(kind, name)
        return
    if name == "lateral":
        with state_lock:
            want = not state["lateral"]
        set_lateral(want)
        return
    with state_lock:
        if kind == "toggle":
            state[name] = not state[name]
        elif kind == "pan" and name in PAN_BUTTONS:
            _pan_locked(*PAN_BUTTONS[name])
        elif name == "fullscreen":
            state["fullscreen"] = True
        elif name == "windowed":
            state["fullscreen"] = False
        elif name in ("stop", "quit"):
            # Standby, not process exit: the web server stays up so the panel's
            # Start button (or the desktop shortcut) can bring the sim back.
            state["running"] = False
        elif name == "fluoro_reset":
            state["fluoro_sec"] = 0.0


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


class HeldKey(object):
    '''Whether a key is down right now, asked of the X server.

    cv.waitKey reports key presses only, so a held foot pedal - a keyboard that
    types 'b' - arrives as one 'b' and then, after the key-repeat delay (0.6 s
    by default), as the key's repeats. The X server (Xwayland on the Pi) knows
    the key's real state: XQueryKeymap on a connection of our own, through
    libX11. held() is None where there is no X server to ask.
    '''

    def __init__(self, keysym):
        self._x = None
        try:
            lib = ctypes.util.find_library("X11")
            if not lib or not os.environ.get("DISPLAY"):
                return
            x = ctypes.CDLL(lib)
            x.XOpenDisplay.restype = ctypes.c_void_p
            x.XOpenDisplay.argtypes = [ctypes.c_char_p]
            x.XKeysymToKeycode.restype = ctypes.c_ubyte
            x.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            x.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char * 32]
            dpy = x.XOpenDisplay(None)
            code = x.XKeysymToKeycode(dpy, keysym) if dpy else 0
            if code:
                self._x, self._dpy, self._code = x, dpy, code
        except Exception:      # no libX11 / no display: held() says it can't tell
            self._x = None

    def held(self):
        if self._x is None:
            return None
        keys = (ctypes.c_char * 32)()
        self._x.XQueryKeymap(self._dpy, keys)
        return bool(bytearray(keys.raw)[self._code >> 3] & (1 << (self._code & 7)))


def fluoro_tick(beam_t, now, beam):
    '''One step of the fluoro clock -> (beam_t, seconds to add).

    beam_t is when the beam was last seen on (None: it has been off since).
    beam is True while it is on - a frame taken, or the pedal held - False
    when it is off, and None when that is unknown: a held pedal between two of
    its key repeats where X cannot be asked, in which case the gap is counted
    if the next repeat comes within FLUORO_GAP_SEC.
    '''
    if beam:
        gap = None if beam_t is None else now - beam_t
        return now, (gap if gap is not None and gap <= FLUORO_GAP_SEC else 0.0)
    return (beam_t if beam is None else None), 0.0


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
    the HUD, then both show it in the FLUORO window and hand it to
    publish_preview() for the web preview. The loop
    reads the shared toggle state once per frame via get_state_snapshot(), so the
    web buttons and the keyboard shortcuts drive exactly the same behaviour.

    cam_index   : int | None — V4L2 device index named on the command line;
                  None means "discover the cameras" (the normal case)
    show_window : bool — open the on-screen FLUORO window (False = web preview only)
    '''
    def open_camera(cam):
        '''Open one camera record, or None if it will not open.

        Opens by V4L2 device index, which is what the backend has always been
        given here; the port -> index lookup already happened in the scan.
        Returning None rather than a dead VideoCapture lets the loop fall
        through to the "no camera" placeholder and retry the open later.
        '''
        c = cv.VideoCapture(cam["index"], cv.CAP_V4L2)
        if not c.isOpened():
            c.release()
            return None
        return LatestFrameCapture(c)

    def publish_camera(ok):
        '''Mirror camera presence into the shared state (read by the web panel).'''
        with state_lock:
            if state["camera"] != ok:
                state["camera"] = ok

    def rescan_cameras():
        '''Re-inventory the attached cameras and re-resolve the two roles.

        Publishes the count and whether a lateral view is on offer, and drops
        back to the main view if the lateral camera has been unplugged.
        -> (main_spec, lateral_spec)
        '''
        cams = list_cameras()
        main_cam, lateral_cam = resolve_cameras(cams)
        if main_cam is None and (cam_index is not None or not discovery_works()):
            # Either the operator named a device index on the command line, or
            # this machine cannot enumerate its cameras - open one blindly, as
            # the simulator always did. Where discovery does work, finding no
            # camera is the answer: probing anyway just fills the log.
            blind = 0 if cam_index is None else cam_index
            main_cam = {"index": blind, "device": "video source %s" % blind,
                        "port": None, "name": "unidentified"}
        with state_lock:
            state["cameras"] = len(cams)
            state["lateral_available"] = lateral_cam is not None
            if state["lateral"] and lateral_cam is None:
                state["lateral"] = False
        return main_cam, lateral_cam

    # Placeholder frame published to the preview while the sim is in standby.
    stopped_img = np.full((480, 640), 16, np.uint8)
    draw_str(stopped_img, (200, 230), "SIMULATION STOPPED")
    draw_str(stopped_img, (150, 260), "press Start simulator on the control panel")
    ok, _stopped_jpg = cv.imencode(".jpg", stopped_img, [cv.IMWRITE_JPEG_QUALITY, 80])
    stopped_buf = _stopped_jpg.tobytes() if ok else None

    # Frame shown in place of the video whenever the camera for the current view
    # is not attached: a plain black screen with the message. It is the same
    # shape as an ordinary processed frame, so the tab bar, the control bar, the
    # Library tab and the web preview all lay out exactly as with a live feed.
    # One per view, so a missing lateral camera says so.
    def build_placeholder(head):
        img = np.zeros((480, 640), np.uint8)
        for text, size, thick, grey, ty in (
                (head, 0.9, 2, 235, 232),
                ("connect a camera - it is picked up automatically", 0.45, 1, 130, 262)):
            (tw, _), _ = ui_text_size(text, size, thick)
            ui_put_text(img, text, ((640 - tw) // 2, ty), size, grey, thick)
        enc_ok, jpg = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 80])
        return img, (jpg.tobytes() if enc_ok else None)

    no_camera_img, no_camera_buf = {}, {}
    for _role, _head in (("main", "NO CAMERA CONNECTED"),
                         ("lateral", "NO LATERAL CAMERA CONNECTED")):
        no_camera_img[_role], no_camera_buf[_role] = build_placeholder(_head)
    # Both halves missing, in split screen: composed once here rather than
    # every frame while the simulator sits waiting for cameras.
    no_camera_img["split"] = label_split(
        compose_split(no_camera_img["main"], no_camera_img["lateral"]),
        "FRONTAL", "LATERAL")
    _ok, _split_jpg = cv.imencode(".jpg", no_camera_img["split"],
                                  [cv.IMWRITE_JPEG_QUALITY, 80])
    no_camera_buf["split"] = _split_jpg.tobytes() if _ok else None

    # Full-resolution "master" anatomy background, kept untouched at full res so we
    # can crop a fresh viewport out of it every frame (see compute_viewport). Falls
    # back to the static skel.jpg (shown whole, no panning) if the master is absent.
    master = load_master(MASTER_IMAGE)
    if master is None:
        print("Master image not found at %s — falling back to %s (no panning)"
              % (MASTER_IMAGE, OVERLAY_IMAGE))
        # skel.jpg is a film-convention radiograph (white bone, dark air).
        master = load_master(OVERLAY_IMAGE, invert=True)
        if master is None:
            raise FileNotFoundError("Cannot load image: %s" % OVERLAY_IMAGE)

    # The lateral view's background. If the lateral master is missing, an
    # obvious placeholder field stands in so the split view still works.
    lateral_master = load_master(LATERAL_MASTER_IMAGE)
    if lateral_master is None:
        print("No lateral master at %s — using a placeholder background for "
              "the lateral view." % LATERAL_MASTER_IMAGE)
        lateral_master = build_filler_master(master.shape)
    elif lateral_master.shape != master.shape:
        print("Note: lateral master is %s but the frontal master is %s — the "
              "pan/zoom calibration is expressed in the frontal master's "
              "pixels, so the two views will not pan alike."
              % (lateral_master.shape, master.shape))
    masters = {"main": master, "lateral": lateral_master}
    set_master_size(master)     # the anatomy's edges, for clamp_position
    with state_lock:            # (a position posted before now is brought inside them)
        state["pos_x_cm"], state["pos_y_cm"], state["pos_z_cm"] = clamp_position(
            state["pos_x_cm"], state["pos_y_cm"], state["pos_z_cm"])

    # On-screen control panel: the logo image and a helper to (re)open the
    # separate CONTROLS window used in windowed mode.
    logo = cv.imread(LOGO_IMAGE, cv.IMREAD_UNCHANGED) if show_window else None

    # Both windows go without OpenCV's Qt toolbar and status bar
    # (WINDOW_GUI_NORMAL): off fullscreen they only took room and added zoom
    # buttons, and they made the CONTROLS panel taller than a touch panel.
    def show_controls_window():
        cv.namedWindow("CONTROLS", cv.WINDOW_AUTOSIZE | cv.WINDOW_GUI_NORMAL)
        cv.setMouseCallback("CONTROLS", on_mouse_controls)
        tiles = windowed_layout()
        cv.moveWindow("CONTROLS", *(tiles[1][:2] if tiles else (20, 20)))

    def show_fluoro_window():
        cv.namedWindow("FLUORO", cv.WINDOW_NORMAL | cv.WINDOW_GUI_NORMAL)
        cv.setWindowProperty("FLUORO", cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)
        if SCREEN_POS:
            cv.moveWindow("FLUORO", *SCREEN_POS)
        # Clicks on the FLUORO window only matter in fullscreen, where the control
        # bar is stacked below the video (see overlay_buttons).
        cv.setMouseCallback("FLUORO", on_mouse_fluoro)

    if show_window and logo is None:
        print("Warning: logo not found at", LOGO_IMAGE)

    # The camera and the windows are opened lazily on the first running
    # iteration (and reopened after a standby stop) — see the loop below.
    # Both cameras can be open at once (split screen); each carries its own
    # retry deadline and failure count so one going away never disturbs the
    # other. The lateral camera is only held open while the split view is on.
    caps = {"main": None, "lateral": None}
    opened_as = {"main": None, "lateral": None}   # the camera record each handle holds
    cam_fails = {"main": 0, "lateral": 0}
    next_open = {"main": 0.0, "lateral": 0.0}
    missing_logged = {"main": False, "lateral": False}
    windows_open = False       # FLUORO window currently created (independent of the camera)
    next_cam_scan = 0.0        # monotonic deadline for the next camera inventory
    main_cam = lateral_cam = None     # camera records the two roles resolve to
    split = False              # showing both views side by side this iteration
    placeholder_key = None     # which prebuilt placeholder `res` currently is
    rec_shape = None           # frame shape the running recording was opened at
    applied_fullscreen = None  # last fullscreen state pushed to the window (avoids redundant calls)
    retile_until = {}          # window name -> monotonic deadline for putting it back in its tile
    res = None                 # last processed frame (shown + streamed)
    res_plain = None           # that frame before the HUD text went on (for the screen)
    live = False               # True once a frame has been read (drives the status dot)
    beam_t = None              # when the beam was last seen on (fluoro time; see fluoro_tick)
    pedal_key = HeldKey(0x62) if show_window else None   # is the pedal's 'b' down right now?
    held_frames = {}           # the camera frames the picture was last made from, per role
    drawn_with = None          # the view settings it was drawn with (see `redraw` below)

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
            if any(c is not None for c in caps.values()):
                for _role in caps:
                    if caps[_role] is not None:
                        caps[_role].release()
                        caps[_role] = None
                if show_window:
                    library.stop_playback()
                    cv.destroyAllWindows()
                    cv.waitKey(1)  # let the GUI process the window teardown
                applied_fullscreen = None
                res, res_plain, live, beam_t = None, None, False, None
                held_frames, drawn_with = {}, None
            windows_open = False
            next_cam_scan = 0.0
            for _role in caps:
                opened_as[_role] = None
                cam_fails[_role] = 0
                next_open[_role] = 0.0
                missing_logged[_role] = False
            placeholder_key = None
            publish_camera(False)
            if stopped_buf is not None:
                publish_preview(jpeg=stopped_buf)
            time.sleep(0.2)
            continue

        # (Re)open the windows on the first running iteration and when resuming
        # from standby. This no longer waits on the camera: the simulator runs,
        # and shows its tabs, with or without one.
        if show_window and not windows_open:
            show_fluoro_window()
            windows_open = True

        # Re-inventory the attached cameras on a timer, so a second one plugged
        # in mid-session offers the lateral view (and one pulled out withdraws
        # it) without a restart.
        if time.monotonic() >= next_cam_scan:
            next_cam_scan = time.monotonic() + CAMERA_SCAN_SEC
            main_cam, lateral_cam = rescan_cameras()
            s = get_state_snapshot()   # the rescan may have cleared "lateral"

        # Selecting the lateral view puts both projections on screen side by
        # side, so both cameras stay open together; the frontal-only view
        # closes the lateral camera rather than hold USB bandwidth it is not
        # using. The frontal camera is never dropped for a switch, so toggling
        # the split view never interrupts the main image.
        split = bool(s["lateral"])
        specs = {"main": main_cam, "lateral": lateral_cam}
        wanted = ("main", "lateral") if split else ("main",)
        for _role in ("main", "lateral"):
            if _role not in wanted and caps[_role] is not None:
                caps[_role].release()
                caps[_role] = None
                opened_as[_role] = None
                cam_fails[_role] = 0

        # Reconcile every open camera against what its role resolves to NOW.
        # USB renumbering moves a camera to a different /dev node - a replug, or
        # simply a second camera appearing and the kernel handing out nodes in a
        # different order - and a handle left on the old node is both the wrong
        # camera and holds the node the other role now wants. That is how a
        # lateral camera which is plugged in, discovered, and offered on the
        # button still fails to open, while the frontal view quietly shows the
        # lateral camera's picture.
        for _role in ("main", "lateral"):
            if caps[_role] is None:
                continue
            now_at, was = specs[_role], opened_as[_role]
            if now_at is not None and was is not None and (
                    (now_at["port"], now_at["index"]) == (was["port"], was["index"])):
                continue
            print("%s camera moved from %s to %s - reopening."
                  % (_role.capitalize(),
                     was["device"] if was else "?",
                     now_at["device"] if now_at else "nothing"))
            caps[_role].release()
            caps[_role] = None
            opened_as[_role] = None
            cam_fails[_role] = 0
            next_open[_role] = 0.0     # reopen immediately, not on the next tick

        # (Re)open what this view needs. A missing camera is not fatal - the
        # capture section below stands a "NO CAMERA CONNECTED" frame in for
        # that half - so we keep retrying on a timer and pick up a camera
        # plugged in later.
        now = time.monotonic()
        for _role in wanted:
            if caps[_role] is not None or specs[_role] is None:
                continue
            if now < next_open[_role]:
                continue
            next_open[_role] = now + CAMERA_RETRY_SEC
            caps[_role] = open_camera(specs[_role])
            cam_fails[_role] = 0
            opened_as[_role] = specs[_role] if caps[_role] is not None else None
            if caps[_role] is not None:
                print("%s camera connected: %s (%s)"
                      % (_role.capitalize(), specs[_role]["device"],
                         specs[_role]["name"]))
                missing_logged[_role] = False
            elif not missing_logged[_role]:
                missing_logged[_role] = True
                print("No %s camera at %s - running without it "
                      "(retrying every %.0fs)."
                      % (_role, specs[_role]["device"], CAMERA_RETRY_SEC))
        publish_camera(any(caps[r] is not None for r in wanted))

        # Keep the OpenCV window's fullscreen state in sync with the toggle, and
        # move the on-screen controls between the separate CONTROLS window
        # (windowed) and a bar stacked below the video (fullscreen).
        if show_window and s["fullscreen"] != applied_fullscreen:
            if s["fullscreen"] and SCREEN_POS and applied_fullscreen is not None:
                # Back onto the touch panel first: fullscreen fills whichever
                # output the window is on, and windowed it may have been
                # dragged onto the other monitor.
                cv.moveWindow("FLUORO", *SCREEN_POS)
            cv.setWindowProperty(
                "FLUORO", cv.WND_PROP_FULLSCREEN,
                cv.WINDOW_FULLSCREEN if s["fullscreen"] else cv.WINDOW_NORMAL)
            if s["fullscreen"]:
                if applied_fullscreen is not None:   # close the window if it was open
                    cv.destroyWindow("CONTROLS")
            else:
                show_controls_window()
                overlay_buttons[:] = []
                tiles = windowed_layout()
                if tiles:
                    # Off fullscreen the window gets back whatever size it last
                    # had (the whole panel); give it its tile beside CONTROLS.
                    fx, fy, fw, fh = tiles[0]
                    cv.resizeWindow("FLUORO", fw, fh)
                    cv.moveWindow("FLUORO", fx, fy)
            applied_fullscreen = s["fullscreen"]

        # Keyboard: the tab UI (focused text fields, ESC-back) gets first look;
        # unconsumed keys keep their existing shortcut meaning on the Fluoro view.
        key = (cv.waitKey(1) & 0xFF) if show_window else 0xFF
        consumed = handle_ui_key(key) if key != 0xFF else False
        # The foot pedal types 'b' (a keyboard 'b' does the same).
        pedal_down = not consumed and key == ord('b')
        if key != 0xFF and not consumed:
            _handle_key(key)
        if key == 27 and not consumed:  # ESC (on the Fluoro view) exits
            with state_lock:
                state["quit"] = True
            break

        # Which tab is showing (read after waitKey so a click this iteration
        # counts). While a Library recording is actually playing, skip the
        # live processing so the CPU goes to decoding — each camera's reader
        # thread keeps it drained, so switching back to Fluoro is instant.
        with state_lock:
            view = state["ui_view"]
        lib_playing = show_window and view == "library" and library.playing

        # In pedal mode, only grab a frame while the pedal/'b' is held; otherwise
        # capture continuously.
        any_open = any(caps[r] is not None for r in wanted)
        capture_now = ((pedal_down or not s["pedal_mode"])
                       and not lib_playing and any_open)

        # Is the beam on, for the fluoro time? With Pedal mode off, while frames
        # are taken; in Pedal mode, while the pedal is held - from the X
        # server's key state, as the 'b' presses alone come with gaps (None:
        # unknown until the next 'b', where X cannot be asked).
        if not any_open or lib_playing:
            beam = False
        elif not s["pedal_mode"]:
            beam = True
        else:
            held = pedal_key.held() if pedal_key is not None else None
            if view != "fluoro" and ui["focus"] is not None:
                held = False      # a 'b' typed into a text field is not the pedal
            beam = True if pedal_down else held

        # Between pedal presses the picture is held. A pan, a zoom or a toggle
        # made meanwhile is drawn on the held frames straight away - the
        # anatomy moves under the still camera image, as the readout says it
        # has - instead of waiting for the next press and then jumping by all
        # of it at once.
        view_now = (split, s["overlay"], s["equalize"], s["hud"], s["pedal_mode"],
                    s["pos_x_cm"], s["pos_y_cm"], s["pos_z_cm"])
        redraw = (not capture_now and any_open and not lib_playing
                  and view_now != drawn_with
                  and all(r in held_frames for r in wanted))
        new_frame = False          # a picture was made this iteration (captured or redrawn)

        if not any_open:
            # Nothing to show: publish the prebuilt placeholder so the window,
            # the tab bar and the Library keep rendering, and idle a little so
            # an unattended simulator does not spin a core.
            live = False
            placeholder_key = "split" if split else "main"
            res, res_plain = no_camera_img[placeholder_key], None
            time.sleep(0.03)
        elif capture_now or redraw:
            # Read every camera this view needs - or, redrawing, take the held
            # frames again - and turn each frame into a finished view. In split
            # screen the halves are composited at a reduced working width
            # (SPLIT_WORK_MAX_W): the pipeline is the expensive part of the
            # frame and it runs twice here.
            frames = {} if capture_now else held_frames
            for _role in (wanted if capture_now else ()):
                camera = caps[_role]
                if camera is None:
                    continue
                ret, frame = camera.read()
                if not ret:
                    # A few dropped frames are normal; a steady stream of them
                    # means that camera went away, so release it and let the
                    # retry above find it again.
                    cam_fails[_role] += 1
                    if cam_fails[_role] >= CAMERA_READ_FAIL_LIMIT:
                        print("%s camera stopped delivering frames - released."
                              % _role.capitalize())
                        camera.release()
                        caps[_role] = None
                        opened_as[_role] = None
                        cam_fails[_role] = 0
                        next_open[_role] = time.monotonic() + CAMERA_RETRY_SEC
                    continue
                cam_fails[_role] = 0
                frames[_role] = frame
            work_w = SPLIT_WORK_MAX_W if split else None
            views = {r: render_fluoro_view(frames[r], masters[r], s, work_w)
                     for r in wanted if r in frames}

            if not views:
                time.sleep(0.01)
                continue     # transient: keep the previous frame on screen
            held_frames, drawn_with, new_frame = frames, view_now, True
            live = True
            placeholder_key = None

            if split:
                # A half with no camera shows that half's placeholder, so one
                # camera dropping out never takes the working view with it.
                halves = []
                for _role in ("main", "lateral"):
                    half = views.get(_role)
                    if half is None:
                        half = no_camera_img[_role]
                    halves.append(half)
                res = label_split(compose_split(halves[0], halves[1]),
                                  "FRONTAL", "LATERAL")
            else:
                res = views["main"]

            # On screen the field gets GE-style notes in its corners instead
            # of the HUD text, so the screen takes the frame from before it;
            # recordings and the web preview keep the HUD text.
            res_plain = res.copy()
            if s["hud"]:
                draw_str(res, (20, 20), "Overlay:%s  Equalize:%s" %
                         (s["overlay"], s["equalize"]))
                draw_str(res, (20, 40), "Pedal mode:%s  HUD:%s" % (s["pedal_mode"], s["hud"]))
                draw_str(res, (20, 60), "Cam: x=%.1f  y=%.1f  z=%.1f cm" %
                         (s["pos_x_cm"], s["pos_y_cm"], s["pos_z_cm"]))
            if pedal_down:
                draw_str(res, (20, 80), "PEDAL ACTIVE")

        # Fluoro time. (A camera read that failed has skipped this: the gap
        # until its next frame counts if that comes within FLUORO_GAP_SEC.)
        beam_t, beam_sec = fluoro_tick(beam_t, time.monotonic(), beam)
        if beam_sec:
            with state_lock:
                state["fluoro_sec"] += beam_sec

        # Recording needs a live camera: if it disappears mid-session, close the
        # file cleanly and flip the toggle off instead of filling the recording
        # with the placeholder frame. The hold window covers a view switch, so
        # toggling Lateral view mid-recording does not end the recording.
        if s["recording"] and not any_open:
            if recorder.active:
                recorder.stop()
                uploads.poke()
                rec_shape = None
            library.error = "Recording needs a camera - none is connected."
            with state_lock:
                state["recording"] = False
            s["recording"] = False

        # Session recording: feed the processed frame to the recorder while
        # the toggle is on; a writer that fails to open flips the toggle back
        # off and surfaces its error on the Library tab.
        if s["recording"] and res is not None:
            if recorder.ensure_started(res.shape):
                if rec_shape is None:
                    rec_shape = res.shape
                if new_frame:
                    # The writer is fixed at the size it opened with, so a view
                    # switch to a camera of a different resolution is scaled to
                    # match rather than silently corrupting the file.
                    frame_out = res
                    if res.shape[:2] != rec_shape[:2]:
                        frame_out = cv.resize(res, (rec_shape[1], rec_shape[0]))
                    recorder.submit(frame_out)
            else:
                with state_lock:
                    state["recording"] = False
                library.error = recorder.error
                rec_shape = None
        elif not s["recording"] and recorder.active:
            recorder.stop()
            uploads.poke()
            rec_shape = None

        rec_elapsed = recorder.elapsed_str() if recorder.active else None
        # Windowed on a touch panel: the boxes the two windows are tiled into.
        tiles = None if s["fullscreen"] else windowed_layout()

        # A window closed with its title-bar X comes back with the next
        # cv.imshow below, but labwc places it anew - lower down, partly off
        # the panel, and the resizable FLUORO window shrunk to fit there - a
        # moment after it reappears. Keep putting it back in its tile for a
        # second after that; a window the user drags is left be.
        if show_window and tiles and applied_fullscreen is False:
            now = time.monotonic()
            for name, box in (("FLUORO", tiles[0]), ("CONTROLS", tiles[1])):
                if not window_open(name):
                    retile_until[name] = now + 1.0
                elif retile_until.get(name, 0.0) > now:
                    if name == "FLUORO":
                        cv.resizeWindow(name, box[2], box[3])
                    cv.moveWindow(name, box[0], box[1])

        if show_window and not s["fullscreen"]:
            # Windowed: the GE-style panel, tabs and all, in CONTROLS; FLUORO
            # shows the image alone - or the Remote Access / Library view -
            # filling its square. A failure in tab code must never take down
            # the sim: fall back to the Fluoro view.
            try:
                if tiles:
                    side, (pw, ph) = tiles[0][2], tiles[1][2:]
                else:
                    side, (pw, ph) = (res.shape[0] if res is not None else 480), (GE_W, GE_H)
                if view == "fluoro":
                    canvas, vbtns = render_ge_image(
                        res if res_plain is None else res_plain, side, side, s, logo, split,
                        live, rec_elapsed), []
                else:
                    canvas, vbtns = render_tab_view(view, s, side, side)
                tab_buttons[:] = []
                view_buttons[:] = vbtns
                panel, btns = render_ge_panel(ge_panel_state(s), pw, ph)
                ctrl_buttons[:] = btns
                cv.imshow("CONTROLS", panel)
                cv.imshow("FLUORO", canvas)
            except Exception:
                traceback.print_exc()
                with state_lock:
                    state["ui_view"] = "fluoro"
        elif show_window and view != "fluoro":
            # Remote Access / Library replace the image and the control bar
            # under the header strip; the capture pipeline above keeps running
            # so the preview stays live and switching back is instant. A
            # failure in tab code must never take down the sim: fall back to
            # the Fluoro view.
            try:
                fh, fw = res.shape[:2] if res is not None else (480, 640)
                vw = SCREEN_SIZE[0] if SCREEN_SIZE else fw
                head, hbtns = render_ge_header(ge_panel_state(s), vw, rec_elapsed)
                if SCREEN_SIZE:
                    canvas_h = SCREEN_SIZE[1] - head.shape[0]
                else:   # as tall as the fluoro view: its frame plus the bar
                    canvas_h = fh + render_ge_bar(ge_panel_state(s), vw)[0].shape[0]
                canvas, vbtns = render_tab_view(view, s, vw, canvas_h)
                tab_buttons[:] = hbtns
                view_buttons[:] = [(x, y + head.shape[0], w, h, k, n)
                                   for (x, y, w, h, k, n) in vbtns]
                cv.imshow("FLUORO", np.vstack([head, canvas]))
            except Exception:
                traceback.print_exc()
                with state_lock:
                    state["ui_view"] = "fluoro"
        elif res is not None and show_window:
            # The image between the header strip and the control bar. With
            # --screen it is composed natively at the display resolution -
            # letterboxed, not stretched - so every pixel and tap maps 1:1.
            p = ge_panel_state(s)
            width = SCREEN_SIZE[0] if SCREEN_SIZE else res.shape[1]
            head, hbtns = render_ge_header(p, width, rec_elapsed)
            bar, bbtns = render_ge_bar(p, width)
            vid_h = (SCREEN_SIZE[1] - head.shape[0] - bar.shape[0]) if SCREEN_SIZE else res.shape[0]
            vid = render_ge_image(res if res_plain is None else res_plain, width, vid_h, s, logo,
                                  split, live, None)
            tab_buttons[:] = hbtns
            overlay_buttons[:] = [(x, y + head.shape[0] + vid_h, w, h, k, n)
                                  for (x, y, w, h, k, n) in bbtns]
            cv.imshow("FLUORO", np.vstack([head, vid, bar]))
        if placeholder_key and no_camera_buf.get(placeholder_key) is not None:
            # Static placeholder: publish the pre-encoded copy rather than
            # re-encoding the same image on every iteration.
            publish_preview(jpeg=no_camera_buf[placeholder_key])
        elif res is not None and new_frame:
            # The web MJPEG preview streams the picture as the screen shows it -
            # without the HUD text, whose notes the page draws over it the way
            # the screen does (recordings keep the text) - and not the composited
            # view, as the page has its own controls. Only a newly made frame is
            # published: it is never drawn on again after this, so the web
            # thread can encode it unlocked.
            publish_preview(frame=res if res_plain is None else res_plain)

    recorder.stop(wait=True)  # finalize the .avi before os._exit
    library.stop_playback()
    for _role in caps:
        if caps[_role] is not None:
            caps[_role].release()
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
            _pan_locked(0, -1)
        elif key == ord('d'):            # X+
            _pan_locked(0, 1)
        elif key == ord('w'):            # Y−
            _pan_locked(1, -1)
        elif key == ord('s'):            # Y+
            _pan_locked(1, 1)
        elif key == ord('q'):            # Z− (lower the camera = zoom in)
            _pan_locked(2, -1)
        elif key == ord('e'):            # Z+ (raise the camera = zoom out)
            _pan_locked(2, 1)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(__doc__)

    # Line-buffered, so launch.log shows what the simulator is doing while it
    # is doing it: under block buffering the camera diagnostics sat unwritten
    # in a 4KB buffer for hours, which is precisely when they are wanted.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    args = sys.argv[1:]
    port = 5000
    host = "0.0.0.0"
    show_window = True
    cam_index = None     # None = discover; an explicit index opens that device
    force_http = False
    role_args = {}       # --main / --lateral: port, /dev path or index to save

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
        if a == "--screen":        # compose fullscreen natively at WxH[+X+Y] (touch displays)
            m = re.match(r"^(\d+)x(\d+)(?:\+(\d+)\+(\d+))?$", args[i + 1].lower())
            if not m:
                sys.exit("--screen wants WxH or WxH+X+Y, e.g. 1024x600+0+0")
            SCREEN_SIZE = (int(m.group(1)), int(m.group(2)))
            if m.group(3) is not None:
                SCREEN_POS = (int(m.group(3)), int(m.group(4)))
            i += 2; continue
        if a == "--list-cameras":  # show what is attached, and which port is which
            print(describe_cameras()); sys.exit(0)
        if a in ("--main", "--lateral"):
            role_args[a[2:]] = args[i + 1]; i += 2; continue
        cam_index = int(a); i += 1

    # --main/--lateral bind a view to the USB PORT a camera is plugged into and
    # save it, so the assignment survives replugs, reboots and /dev/videoN
    # renumbering. Ports come from --list-cameras.
    if role_args:
        attached = list_cameras()
        roles = read_camera_roles()
        for role, spec in role_args.items():
            cam = match_camera(attached, spec)
            if cam is None:
                print("No camera attached matching %r - saving it for the %s view "
                      "anyway (plug it in and it will be picked up)." % (spec, role))
                roles[role] = spec
            else:
                roles[role] = cam["port"]
                print("%s view: %s  (%s, %s)"
                      % (role.capitalize(), cam["port"], cam["device"], cam["name"]))
        if roles.get("main") and roles.get("main") == roles.get("lateral"):
            print("Both views point at the same port - refusing to save.")
            sys.exit(1)
        if write_camera_roles(roles):
            print("Saved to %s" % CAMERA_CONFIG)

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
