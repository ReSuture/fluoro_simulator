#!/usr/bin/env python3
'''Which display is the touch panel, and mirroring it onto a second monitor.

The fullscreen UI is laid out for the touch panel (fluoro_web.py --screen), so
with two monitors attached it has to know which one that is. A Pi 5 has two
HDMI ports: plug a monitor into the one the panel is not using and it shows
the same picture as the panel, via wl-mirror; unplug it and nothing changes.
The monitor is only ever a picture of the panel: touch, the pointer and the
keyboard (so the foot pedal) all stay with the panel.

    python3 displays.py screen-arg   # "1024x600+0+0", for launch_fluoro.sh
    python3 displays.py status       # what is attached, and which is the panel
    python3 displays.py mirror       # the fluorosim-mirror service's loop

Settings, in ~/.config/fluorosim/display.conf (all optional):

    TOUCH_DISPLAY=WIMAXIT   # matched against each output's name, make and model
    MIRROR=on               # off = a second monitor stays an ordinary desktop

The panel is found by its EDID name rather than its port, so it can go in
either HDMI socket. If nothing matches TOUCH_DISPLAY, the first output is
taken to be the panel - with one monitor, that is simply the monitor.
'''

import json
import os
import re
import subprocess
import sys
import time

CONF = os.path.expanduser("~/.config/fluorosim/display.conf")
DEFAULTS = {"TOUCH_DISPLAY": "WIMAXIT", "MIRROR": "on"}

# labwc's per-user config. labwc runs with -m (merge), so this only needs the
# lines FluoroSim adds; the system defaults in /etc/xdg/labwc still apply.
LABWC_RC = os.path.expanduser("~/.config/labwc/rc.xml")
TOUCH_MARK = "<!-- FluoroSim: touch input follows the touch panel (displays.py) -->"
END_MARK = "<!-- FluoroSim: end of displays.py settings -->"

POLL_SEC = 1.0              # how quickly a plugged/unplugged monitor is noticed
MIRROR_RETRY_SEC = 5.0      # wait before restarting a wl-mirror that died
MIRROR_TITLE = "FluoroSim mirror"   # wl-mirror's window title; labwc's focus rule matches it

# While mirroring, the monitor sits this far right of the panel in the desktop
# layout. The compositor keeps the pointer on the nearest output, so a gap wider
# than any single mouse movement keeps the pointer on the panel - on the monitor
# it would only be over wl-mirror's picture, where clicks reach nothing.
MIRROR_GAP_PX = 4096


def read_conf(path=CONF):
    '''display.conf as a dict over DEFAULTS. Missing file = defaults.'''
    conf = dict(DEFAULTS)
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    conf[key.strip().upper()] = value.strip()
    except OSError:
        pass
    return conf


def _session_env():
    '''This process's environment, pointed at the desktop session.

    A user service can start before the desktop has published WAYLAND_DISPLAY
    and DISPLAY to systemd, and its environment never updates after that, so
    fall back to the names the Pi's labwc session always uses.
    '''
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", "/run/user/%d" % os.getuid())
    env.setdefault("WAYLAND_DISPLAY", "wayland-0")
    env.setdefault("DISPLAY", ":0")
    return env


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5,
                              env=_session_env()).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def wayland_outputs():
    '''Enabled outputs, as wlr-randr reports them: [{name, label, x, y, w, h}].

    label is "make model" from the EDID; x, y, w, h are the output's box in the
    desktop layout, in logical pixels. labwc's NOOP-1 stand-in, which exists
    only until a real display is up, is left out. Empty when wlr-randr is
    missing or the desktop is not up yet.
    '''
    try:
        data = json.loads(_run(["wlr-randr", "--json"]) or "[]")
    except ValueError:
        return []
    outputs = []
    for o in data:
        name = o.get("name", "")
        if not o.get("enabled") or name.startswith(("NOOP-", "HEADLESS-")):
            continue
        mode = next((m for m in o.get("modes", []) if m.get("current")), {})
        w, h = mode.get("width", 0), mode.get("height", 0)
        if o.get("transform", "normal").endswith(("90", "270")):
            w, h = h, w
        scale = o.get("scale") or 1.0
        pos = o.get("position") or {}
        outputs.append({"name": name,
                        "label": "%s %s" % (o.get("make", ""), o.get("model", "")),
                        "x": pos.get("x", 0), "y": pos.get("y", 0),
                        "w": int(round(w / scale)), "h": int(round(h / scale))})
    return outputs


_XRANDR_OUTPUT = re.compile(r"^(\S+) connected (?:primary )?(\d+)x(\d+)\+(\d+)\+(\d+)")


def x_geometry():
    '''{output name: "WxH+X+Y"} for each active output, from xrandr.

    In X coordinates, because the FLUORO window is an Xwayland client and is
    positioned in those. Ordered as xrandr lists them.
    '''
    geo = {}
    for line in _run(["xrandr", "--current"]).splitlines():
        m = _XRANDR_OUTPUT.match(line)
        if m:
            geo[m.group(1)] = "%sx%s+%s+%s" % m.groups()[1:]
    return geo


def pick_touch(outputs, pattern):
    '''Name of the output that is the touch panel, or None if there are none.'''
    if not outputs:
        return None
    want = (pattern or "").lower()
    if want:
        for out in outputs:
            if want in out["name"].lower() or want in out["label"].lower():
                return out["name"]
    return outputs[0]["name"]


def screen_arg(conf=None):
    '''The touch panel's "WxH+X+Y", or "" when it cannot be worked out.'''
    conf = conf or read_conf()
    geo = x_geometry()
    if not geo:
        return ""
    outputs = wayland_outputs() or [{"name": n, "label": ""} for n in geo]
    touch = pick_touch(outputs, conf["TOUCH_DISPLAY"])
    return geo.get(touch) or next(iter(geo.values()))


def map_input(output, path=LABWC_RC):
    '''Point touch input and the keyboard at the panel in labwc's config. -> changed?

    Without the touch line, a second monitor stretches the touch area across
    both screens and every tap lands in the wrong place. The line names no
    device, which makes it labwc's default for every touchscreen; an empty
    deviceName="" would match none of them. labwc focuses every new window, so
    the window rule hands the keyboard - and the foot pedal, which is one -
    back to the panel's window whenever the mirror opens. Only the marked
    block is ever written; anything else in the file is left exactly as it
    was. labwc is told to reload when the file changes.
    '''
    entry = ('%s\n'
             '  <touch mapToOutput="%s" />\n'
             '  <windowRules>\n'
             '    <windowRule title="%s">\n'
             '      <action name="FocusOutput" output="%s" />\n'
             '    </windowRule>\n'
             '  </windowRules>\n'
             '  %s') % (TOUCH_MARK, output, MIRROR_TITLE, output, END_MARK)
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError:
        text = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<openbox_config xmlns="http://openbox.org/3.4/rc">\n'
                '</openbox_config>\n')
    # The marked block, or the lone touch line that earlier versions wrote.
    ours = re.compile(re.escape(TOUCH_MARK) + r"\s*<touch [^>]*/>(?:.*?"
                      + re.escape(END_MARK) + r")?", re.S)
    if ours.search(text):
        new = ours.sub(lambda _m: entry, text, count=1)
    else:
        close = text.rfind("</")           # the root element's closing tag
        if close < 0:
            print("Not touching %s: no closing tag found" % path)
            return False
        new = text[:close] + "  " + entry + "\n" + text[close:]
    if new == text:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".fluorosim-tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(new)
    os.replace(tmp, path)
    subprocess.run(["pkill", "-HUP", "-x", "labwc"], check=False)
    print("Touch input and keyboard focus mapped to %s (%s)" % (output, path))
    return True


def park_monitor(panel, monitor, mirroring):
    '''Keep a mirroring monitor out of the pointer's reach. -> moved?

    While it mirrors, the monitor goes MIRROR_GAP_PX right of the panel. If
    mirroring is switched off while it is parked there, it goes back beside
    the panel so the pointer can reach it as an ordinary desktop; any other
    arrangement is left alone.
    '''
    right = panel["x"] + panel["w"]
    parked = (right + MIRROR_GAP_PX, panel["y"])
    now = (monitor["x"], monitor["y"])
    if mirroring and now != parked:
        dest = parked
    elif not mirroring and now == parked:
        dest = (right, panel["y"])
    else:
        return False
    _run(["wlr-randr", "--output", monitor["name"], "--pos", "%d,%d" % dest])
    print("Moved %s to %d,%d" % ((monitor["name"],) + dest))
    return True


def _mirror_on(conf):
    return conf["MIRROR"].strip().lower() in ("on", "1", "yes", "true")


def _stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def mirror_loop():
    '''Keep wl-mirror showing the panel on any second monitor, and only then.'''
    proc, showing = None, None   # running wl-mirror and its (panel, monitor)
    started = 0.0
    mapped = None
    while True:
        conf = read_conf()
        outputs = wayland_outputs()
        touch = pick_touch(outputs, conf["TOUCH_DISPLAY"])
        if touch and touch != mapped:
            map_input(touch)
            mapped = touch
        others = [o for o in outputs if o["name"] != touch]
        want = (touch, others[0]["name"]) if (touch and others and _mirror_on(conf)) else None
        if touch and others:
            panel = next(o for o in outputs if o["name"] == touch)
            park_monitor(panel, others[0], mirroring=bool(want))

        if proc is not None and (proc.poll() is not None or showing != want):
            if proc.poll() is None:
                _stop(proc)
                print("Mirror stopped")
            else:
                print("wl-mirror exited (code %s)" % proc.returncode)
            proc, showing = None, None
        if want and proc is None and time.monotonic() - started >= MIRROR_RETRY_SEC:
            panel, monitor = want
            print("Mirroring %s onto %s" % (panel, monitor))
            try:
                proc = subprocess.Popen(
                    ["wl-mirror", "--fullscreen-output", monitor,
                     "--title", MIRROR_TITLE, panel],
                    env=_session_env())
                showing = want
            except OSError as exc:
                print("Cannot start wl-mirror: %s (sudo apt install wl-mirror)" % exc)
            started = time.monotonic()
        time.sleep(POLL_SEC)


def status():
    conf = read_conf()
    outputs = wayland_outputs()
    geo = x_geometry()
    touch = pick_touch(outputs, conf["TOUCH_DISPLAY"])
    print("Settings (%s): %s" % (CONF, conf))
    if not outputs:
        print("No outputs reported by wlr-randr (desktop not up, or not installed)")
    for out in outputs:
        print("  %-10s %-12s %-40s %s" % (
            out["name"], "touch panel" if out["name"] == touch else "",
            out["label"].strip(), geo.get(out["name"], "")))
    print("--screen %s" % (screen_arg(conf) or "(none)"))
    print("Mirror: %s" % ("on" if _mirror_on(conf) else "off"))


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "screen-arg":
        print(screen_arg())
    elif cmd == "mirror":
        mirror_loop()
    elif cmd == "status":
        status()
    else:
        sys.exit("usage: displays.py [screen-arg|status|mirror]")
