# -*- coding: utf-8 -*-
'''Tests for camera discovery and the main / lateral views in fluoro_web.py.

Run it on the device, from this directory:

    python3 test_cameras.py          # exits non-zero if anything fails

Section 1 runs the discovery filter against this machine's real /dev, so it is
worth running on a Pi with cameras plugged in: it prints what was found and
which port each camera sits on. Everything after that stubs the hardware -
these tests must pass on a bench with no cameras attached at all, which is
exactly when the no-camera and lateral-availability paths matter most.

The capture-loop section (7) drives the real run_simulation() with a fake
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


print("\n7. Live switching inside the real capture loop")
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


def preview_mean():
    '''Mean brightness of the frame currently on the web preview.'''
    with f._latest_lock:
        buf = f._latest_jpeg
    if not buf:
        return None
    img = f.cv.imdecode(np.frombuffer(buf, np.uint8), f.cv.IMREAD_GRAYSCALE)
    return None if img is None else round(float(img.mean()), 1)


threading.Thread(target=lambda: f.run_simulation(None, False), daemon=True).start()

time.sleep(4)
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
lateral_level = preview_mean()
check("switching to lateral shows the OTHER camera",
      lateral_level is not None and main_level is not None
      and lateral_level > 1.7 * main_level,
      "main=%s lateral=%s" % (main_level, lateral_level))

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


print("\n%d passed, %d failed" % (len(PASSED), len(FAILED)))
for failure in FAILED:
    print("  FAILED: %s" % failure)
sys.stdout.flush()
os._exit(1 if FAILED else 0)   # run_simulation owns the process
