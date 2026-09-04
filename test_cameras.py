# -*- coding: utf-8 -*-
'''Tests for camera discovery and the main / lateral views in fluoro_web.py.

Run it on the device, from this directory:

    python3 test_cameras.py          # exits non-zero if anything fails

Section 1 runs the discovery filter against this machine's real /dev, so it is
worth running on a Pi with cameras plugged in: it prints what was found and
which port each camera sits on. Everything after that stubs the hardware -
these tests must pass on a bench with no cameras attached at all, which is
exactly when the no-camera and lateral-availability paths matter most.

The capture-loop section (8) drives the real run_simulation() with a fake
VideoCapture, so it takes ~20s. It calls os._exit at the end because
run_simulation owns the process by design.
'''
import glob
import os
import sys
import tempfile
import threading
import time

import numpy as np

import fluoro_web as f

PASSED, FAILED = [], []
REAL_READ_ROLES = f.read_camera_roles


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print("  %-58s %s%s" % (name, "PASS" if ok else "FAIL",
                            "" if ok else "   <- " + str(detail)))


def cam(index, port, name="Fake Cam"):
    '''A camera record shaped like the ones list_cameras() returns.'''
    return {"device": "/dev/video%d" % index, "index": index,
            "port": port, "name": name}


# Never touch the real ~/.config/fluorosim/cameras.json.
f.CAMERA_CONFIG = os.path.join(tempfile.gettempdir(), "fluorosim_test_cameras.json")
if os.path.exists(f.CAMERA_CONFIG):
    os.remove(f.CAMERA_CONFIG)

A, B, C = cam(0, "usb-port-A"), cam(2, "usb-port-B"), cam(4, "usb-port-C")


print("\n1. Discovery filter against this machine's real /dev")
real = f.list_cameras()
print("     %d video node(s) present; list_cameras() -> %s"
      % (len(glob.glob("/dev/video*")),
         [(c["device"], c["port"], c["name"]) for c in real] or "[]"))
# A Pi exposes a dozen video nodes and four of its bcm2835 ISP ones report
# video-capture + streaming, so without the bus filter this returns phantoms.
check("no phantom cameras from platform codec/ISP nodes",
      all(not c["port"].startswith("platform:bcm2835") for c in real))
check("one entry per port (a camera's extra nodes are collapsed)",
      len({c["port"] for c in real}) == len(real))


print("\n2. Role resolution")
f.read_camera_roles = lambda: {}
check("no cameras -> no views", f.resolve_cameras([]) == (None, None))
main, lat = f.resolve_cameras([A])
check("one camera -> main only, no lateral view", main is A and lat is None)
main, lat = f.resolve_cameras([A, B])
check("two cameras, nothing saved -> roles by port order", main is A and lat is B)
main, lat = f.resolve_cameras([B, A])
check("order comes from the port, not the scan order", main is A and lat is B)

f.read_camera_roles = lambda: {"main": "usb-port-B", "lateral": "usb-port-A"}
main, lat = f.resolve_cameras([A, B])
check("saved ports override port order", main is B and lat is A)

f.read_camera_roles = lambda: {"main": "usb-port-GONE", "lateral": "usb-port-B"}
main, lat = f.resolve_cameras([A, B])
check("a saved port that is unplugged falls back to a spare", main is A and lat is B)

f.read_camera_roles = lambda: {"main": "usb-port-A", "lateral": "usb-port-A"}
main, lat = f.resolve_cameras([A, B])
check("both roles on one port -> the other camera takes lateral",
      main is A and lat is B)

f.read_camera_roles = lambda: {"main": "usb-port-GONE", "lateral": "usb-port-A"}
main, lat = f.resolve_cameras([A])
check("only the lateral camera plugged in -> it drives main, no lateral",
      main is A and lat is None)

f.read_camera_roles = lambda: {"main": "usb-port-A", "lateral": "usb-port-B"}
main, lat = f.resolve_cameras([A, B, C])
check("a third camera is left unassigned", main is A and lat is B)


print("\n3. Saved assignment round-trips through the config file")
f.read_camera_roles = REAL_READ_ROLES
check("no config file -> no saved roles", f.read_camera_roles() == {})
f.write_camera_roles({"main": "usb-port-B", "lateral": "usb-port-A"})
check("write then read gives the ports back",
      f.read_camera_roles() == {"main": "usb-port-B", "lateral": "usb-port-A"})
check("match_camera by port", f.match_camera([A, B], "usb-port-B") is B)
check("match_camera by /dev path", f.match_camera([A, B], "/dev/video0") is A)
check("match_camera by index", f.match_camera([A, B], "2") is B)
check("match_camera with no match -> None", f.match_camera([A, B], "nope") is None)
os.remove(f.CAMERA_CONFIG)


print("\n4. The Lateral view gate")
with f.state_lock:
    f.state["lateral_available"] = False
    f.state["lateral"] = False
f.set_lateral(True)
check("cannot select lateral with one camera", f.state["lateral"] is False)
with f.state_lock:
    f.state["lateral_available"] = True
f.set_lateral(True)
check("can select lateral once two are attached", f.state["lateral"] is True)
f.set_lateral(False)
check("can always return to the main view", f.state["lateral"] is False)


print("\n5. On-screen buttons")
with f.state_lock:
    f.state["lateral_available"] = False
snap = f.get_state_snapshot()
bar_off, bar_btns_off = f.render_control_bar(snap, 1920, True)
check("control bar: no Lateral hit-box while unavailable",
      not any(n == "lateral" for (_x, _y, _w, _h, _k, n) in bar_btns_off))
_panel_off, panel_btns_off = f.render_controls(snap, None, True)
check("controls panel: no Lateral hit-box while unavailable",
      not any(n == "lateral" for (_x, _y, _w, _h, _k, n) in panel_btns_off))

with f.state_lock:
    f.state["lateral_available"] = True
snap = f.get_state_snapshot()
bar_on, bar_btns_on = f.render_control_bar(snap, 1920, True)
check("control bar: Lateral is pressable with two cameras",
      any(n == "lateral" for (_x, _y, _w, _h, _k, n) in bar_btns_on))
_panel_on, panel_btns_on = f.render_controls(snap, None, True)
check("controls panel: Lateral is pressable with two cameras",
      any(n == "lateral" for (_x, _y, _w, _h, _k, n) in panel_btns_on))
check("control bar keeps its height (the video must not be resized)",
      bar_on.shape[0] == f.BAR_H)

# Drive a real click through the real handler, at the button's own hit-box.
f.overlay_buttons[:] = list(bar_btns_on)
f.tab_buttons[:] = []
with f.state_lock:
    f.state["ui_view"] = "fluoro"
    f.state["lateral"] = False
for (x, y, w, h, _kind, name) in bar_btns_on:
    if name == "lateral":
        f.on_mouse_fluoro(f.cv.EVENT_LBUTTONDOWN, x + w // 2, y + h // 2, 0, None)
check("clicking Lateral view on the bar switches the view",
      f.state["lateral"] is True)


print("\n6. Web panel API")
client = f.app.test_client()
with f.state_lock:
    f.state["lateral"] = False
    f.state["lateral_available"] = False
    f.state["cameras"] = 1
body = client.get("/api/state").get_json()
check("/api/state exposes the camera fields",
      all(k in body for k in ("lateral", "cameras", "lateral_available")), body)
body = client.post("/api/toggle/lateral").get_json()
check("POST /api/toggle/lateral refused with one camera", body["lateral"] is False)
with f.state_lock:
    f.state["lateral_available"] = True
    f.state["cameras"] = 2
body = client.post("/api/toggle/lateral").get_json()
check("POST /api/toggle/lateral accepted with two", body["lateral"] is True)
body = client.post("/api/toggle/lateral").get_json()
check("POST again returns to the main view", body["lateral"] is False)
check("the panel page carries the Lateral view button",
      'data-toggle="lateral"' in client.get("/").get_data(as_text=True))


print("\n7. Masters and the attenuation composite")
check("a missing master file loads as None",
      f.load_master(os.path.join(tempfile.gettempdir(), "nope-no-such.png")) is None)
frontal = f.load_master(f.MASTER_IMAGE)
check("the frontal master loads", frontal is not None and frontal.ndim == 2)
filler = f.build_filler_master(frontal.shape)
check("the lateral filler matches the frontal master's pixel dimensions",
      filler.shape == frontal.shape[:2],
      "%s vs %s" % (filler.shape, frontal.shape))
check("the filler is a light field (it attenuates little)", filler.mean() > 150,
      filler.mean())
# Any viewport crop must land on a caption, so the filler can never be mistaken
# for real anatomy however the operator pans.
crop = f.compute_viewport(filler, 0, 0, 0)
check("a viewport crop of the filler contains its caption",
      crop.min() < 175, "darkest pixel in crop: %d" % crop.min())

# composite_overlay runs its transmission curve through a LUT rather than a
# float power. Guard that against the reference maths it replaced.
_g = np.full((240, 320), 170, np.uint8)
cv_ = f.cv
cv_.circle(_g, (150, 120), 60, 60, -1)
_g = cv_.GaussianBlur(_g, (0, 0), 3)
_over = cv_.resize(f.compute_viewport(frontal, 0, 0, 0), (320, 240))
_ref = (_over.astype(np.float32) * np.clip(
    _g.astype(np.float32) / f.estimate_illumination(_g), 0.0, 1.0) ** f.ATTEN_GAMMA
        ).astype(np.uint8)
_got = f.composite_overlay(_g, _over, False)
_diff = np.abs(_ref.astype(np.int16) - _got.astype(np.int16))
check("the LUT composite matches the float reference within 1 level",
      _diff.max() <= 1, "max difference %d" % _diff.max())


print("\n8. Split screen in the real capture loop")
STUB = {"cams": [A], "dead": set()}
LEVEL = {0: 100, 2: 200}      # the two cameras differ only in brightness


class FakeCap(object):
    '''Stands in for cv.VideoCapture: a uniform frame per device index.'''

    def __init__(self, index):
        self.index = index

    def isOpened(self):
        return self.index not in STUB["dead"]

    def read(self):
        if self.index in STUB["dead"]:
            return False, None
        return True, np.full((480, 640, 3), LEVEL[self.index], np.uint8)

    def grab(self):
        return True

    def release(self):
        pass


f.list_cameras = lambda: list(STUB["cams"])
f.read_camera_roles = lambda: {}
f.cv.VideoCapture = lambda index, *a, **kw: FakeCap(index)
with f.state_lock:
    f.state.update({"lateral": False, "overlay": False, "hud": False,
                    "running": True, "quit": False})


def preview_image():
    '''The frame currently on the web preview, decoded.'''
    with f._latest_lock:
        buf = f._latest_jpeg
    if not buf:
        return None
    return f.cv.imdecode(np.frombuffer(buf, np.uint8), f.cv.IMREAD_GRAYSCALE)


def preview_mean():
    '''Mean brightness of the frame currently on the web preview.'''
    img = preview_image()
    return None if img is None else round(float(img.mean()), 1)


threading.Thread(target=lambda: f.run_simulation(None, False), daemon=True).start()

time.sleep(4)
single = preview_image()
main_level = preview_mean()
# The circular C-arm aperture blacks out the corners, so a uniform frame of
# level L reads back at roughly 0.5*L. What matters is that each view tracks
# its own camera, so the later checks compare against this reading.
check("one camera: the main view is live",
      main_level is not None and 35 < main_level < 65, main_level)
check("one camera: no lateral view on offer",
      f.state["lateral_available"] is False)

STUB["cams"] = [A, B]                      # a second camera is plugged in
time.sleep(4)
check("a second camera appears -> lateral becomes available",
      f.state["lateral_available"] is True and f.state["cameras"] == 2)

f.set_lateral(True)
time.sleep(4)
split = preview_image()
single_aspect = single.shape[1] / float(single.shape[0])
split_aspect = split.shape[1] / float(split.shape[0])
check("selecting Lateral view puts two views on screen (frame ~twice as wide)",
      split_aspect > 1.8 * single_aspect,
      "aspect %.2f -> %.2f" % (single_aspect, split_aspect))

# The two stub cameras differ only in brightness, so each half must track its
# own camera: left the frontal one, right the lateral one.
mid = split.shape[1] // 2
left_half = split[:, :mid - 6].mean()
right_half = split[:, mid + 6:].mean()
check("the left half is the frontal camera",
      abs(left_half - main_level) < 8,
      "single view %.1f, left half %.1f" % (main_level, left_half))
check("the right half is the lateral camera (2x the brightness)",
      right_half > 1.7 * left_half,
      "left %.1f right %.1f" % (left_half, right_half))
check("the frontal camera was never dropped for the switch",
      f.state["camera"] is True)

STUB["cams"] = [A]                         # the lateral camera is pulled out
STUB["dead"].add(2)
time.sleep(6)
back_level = preview_mean()
check("unplugging the lateral camera falls back to the main view",
      f.state["lateral"] is False and f.state["lateral_available"] is False)
check("the main view is live again after the fallback",
      back_level is not None and main_level is not None
      and abs(back_level - main_level) < 5,
      "was %s, now %s" % (main_level, back_level))
check("the frame is back to a single view",
      abs(preview_image().shape[1] / float(preview_image().shape[0])
          - single_aspect) < 0.2)


print("\n%d passed, %d failed" % (len(PASSED), len(FAILED)))
for failure in FAILED:
    print("  FAILED: %s" % failure)
sys.stdout.flush()
os._exit(1 if FAILED else 0)   # run_simulation owns the process
