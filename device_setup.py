'''FluoroSim Remote Access backend: WiFi (NetworkManager) + portal registration.

Everything here is called from the Remote Access tab in fluoro_web.py. Two
rules keep the sim safe:

1. Nothing in this module ever runs on the OpenCV main thread — SetupManager
   spawns a daemon worker per operation and the render loop only ever calls
   the non-blocking snapshot()/start_*() methods.
2. Nothing here can stop the sim from launching — importing the module reads
   at most three small local files, and every nmcli/portal failure is caught
   and turned into a user-facing message instead of an exception.

Device identity comes from the files provisioning writes (see
pi_setup/provision_pi.py); the FLUOROSIM_* env vars override them so the flow
can be exercised on a dev machine against a PORTAL_DEV portal without sudo.
'''

from __future__ import print_function

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request

DEVICE_ID_FILE = "/etc/fluorosim/device-id"
PORTAL_URL_FILE = "/etc/fluorosim/portal-url"
CLAIM_SECRET_FILE = "/etc/fluorosim/claim-secret"
CACHE_DIR = os.path.expanduser("~/.config/fluorosim")
EMAIL_CACHE = os.path.join(CACHE_DIR, "registered-email")

# Cloudflare's Browser Integrity Check rejects Python-urllib's default
# signature with error 1010, so every portal request identifies itself.
USER_AGENT = "resuture-fluorosim/1.0"


def _read_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def read_identity():
    '''The device's portal identity; any value may be None (unprovisioned).'''
    return {
        "device_id": os.environ.get("FLUOROSIM_DEVICE_ID") or _read_file(DEVICE_ID_FILE),
        "portal_url": (os.environ.get("FLUOROSIM_PORTAL_URL")
                       or _read_file(PORTAL_URL_FILE) or "").rstrip("/") or None,
        "claim_secret": (os.environ.get("FLUOROSIM_CLAIM_SECRET")
                         or _read_file(CLAIM_SECRET_FILE)),
    }


def read_cached_email():
    return _read_file(EMAIL_CACHE)


def write_cached_email(email):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(EMAIL_CACHE, "w", encoding="utf-8") as f:
            f.write((email or "") + "\n")
    except OSError:
        pass  # cache only — registration itself lives on the portal


# ── nmcli wrappers ────────────────────────────────────────────────────────────
# All return (ok, value-or-error-message) and never raise; nmcli output is the
# terse -t format, where fields are ':'-separated and ':' in values is escaped.

def _nmcli(args, timeout):
    try:
        p = subprocess.run(["nmcli"] + args, capture_output=True, text=True,
                           timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, "Network tool unavailable: %s" % exc
    if p.returncode != 0:
        return None, (p.stderr or p.stdout or "nmcli failed").strip()
    return p.stdout, None


def _split_terse(line):
    '''Split one `nmcli -t` line on unescaped ':' and unescape the fields.'''
    fields, cur, esc = [], "", False
    for ch in line:
        if esc:
            cur += ch
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == ":":
            fields.append(cur)
            cur = ""
        else:
            cur += ch
    fields.append(cur)
    return fields


def wifi_status():
    '''-> (ok, {"connected": bool, "ssid": str|None} | error message)'''
    out, err = _nmcli(["-t", "-f", "DEVICE,TYPE,STATE,CONNECTION",
                       "device", "status"], timeout=10)
    if err:
        return False, err
    for line in out.splitlines():
        f = _split_terse(line)
        if len(f) >= 4 and f[1] == "wifi" and f[2] == "connected":
            return True, {"connected": True, "ssid": f[3] or None}
    return True, {"connected": False, "ssid": None}


def wifi_scan():
    '''-> (ok, [{"ssid", "signal", "secured"}] strongest-first | error message)'''
    out, err = _nmcli(["-t", "-f", "SSID,SIGNAL,SECURITY",
                       "device", "wifi", "list", "--rescan", "yes"], timeout=25)
    if err:
        return False, err
    best = {}
    for line in out.splitlines():
        f = _split_terse(line)
        if len(f) < 3 or not f[0]:
            continue  # hidden networks
        try:
            signal = int(f[1])
        except ValueError:
            signal = 0
        secured = f[2] not in ("", "--")
        if f[0] not in best or signal > best[f[0]]["signal"]:
            best[f[0]] = {"ssid": f[0], "signal": signal, "secured": secured}
    return True, sorted(best.values(), key=lambda n: -n["signal"])


def wifi_connect(ssid, password):
    '''-> (ok, None | error message). Blocks up to ~40 s.'''
    # The password is briefly visible in the process list — acceptable on a
    # single-user appliance; nmcli stores it in the connection profile after.
    args = ["--wait", "30", "device", "wifi", "connect", ssid]
    if password:
        args += ["password", password]
    _, err = _nmcli(args, timeout=45)
    if err:
        if "Secrets were required" in err or "secrets" in err.lower():
            return False, "Wrong password for %s" % ssid
        return False, err
    return True, None


def connectivity():
    '''-> "full" | "limited" | "portal" | "none" | "unknown"'''
    out, _err = _nmcli(["networking", "connectivity", "check"], timeout=15)
    return (out or "unknown").strip() or "unknown"


# ── Portal client ─────────────────────────────────────────────────────────────

def _post(identity, path, extra=None):
    '''POST to the portal's device API. -> (ok, json-dict | error message)'''
    if not (identity["device_id"] and identity["portal_url"]
            and identity["claim_secret"]):
        return False, "Device not provisioned"
    body = {"device_id": identity["device_id"],
            "claim_secret": identity["claim_secret"]}
    body.update(extra or {})
    req = urllib.request.Request(
        identity["portal_url"] + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return True, json.load(resp)
    except urllib.error.HTTPError as exc:
        try:
            message = json.load(exc).get("error", "")
        except Exception:
            message = ""
        return False, message or "Portal error (HTTP %d)" % exc.code
    except Exception as exc:
        return False, "Cannot reach the portal: %s" % getattr(exc, "reason", exc)


# ── State machine ─────────────────────────────────────────────────────────────

class SetupManager:
    '''Owns the Remote Access tab's state; all work happens on daemon threads.

    The render loop calls snapshot() every frame (a cheap dict copy) and the
    start_scan/connect/refresh_status/claim methods from button presses; each
    spawns one worker, and a busy flag makes extra clicks no-ops.
    '''

    STATUS_TTL = 30.0  # auto-refresh when the tab is shown and this stale

    def __init__(self):
        self._lock = threading.Lock()
        self._busy = False
        identity = read_identity()
        self._s = {
            "provisioned": all(identity.values()),
            "wifi": "idle",          # idle | scanning | connecting
            "wifi_connected": False,
            "wifi_ssid": None,
            "connectivity": "unknown",
            "networks": [],
            "portal": "idle",        # idle | checking | claiming
            "online": False,         # the portal answered our last status call
            "registered_email": read_cached_email(),
            "error": None,
            "notice": None,
            "last_status_check": 0.0,
        }

    def snapshot(self):
        with self._lock:
            return dict(self._s)

    def set_error(self, message):
        '''Surface a locally detected problem (e.g. malformed email) in the UI.'''
        self._set(error=message)

    def _set(self, **kv):
        with self._lock:
            self._s.update(kv)

    def _start(self, target):
        '''Run target on a daemon worker unless one is already running.'''
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        def wrapper():
            try:
                target()
            except Exception as exc:  # belt and braces: never kill the app
                self._set(error="Unexpected error: %s" % exc)
            finally:
                with self._lock:
                    self._busy = False
        threading.Thread(target=wrapper, daemon=True).start()
        return True

    # ── operations (each runs on a worker) ──────────────────────────────────

    def start_scan(self):
        def work():
            self._set(wifi="scanning", error=None, notice=None)
            ok, result = wifi_scan()
            if ok:
                self._set(wifi="idle", networks=result)
            else:
                self._set(wifi="idle", error=result)
        self._start(work)

    def connect(self, ssid, password):
        def work():
            self._set(wifi="connecting", error=None, notice=None)
            ok, err = wifi_connect(ssid, password)
            if not ok:
                self._set(wifi="idle", error=err)
                return
            self._set(wifi="idle")
            self._refresh_status_now()
        self._start(work)

    def refresh_status(self, force=False):
        with self._lock:
            stale = time.monotonic() - self._s["last_status_check"] > self.STATUS_TTL
        if force or stale:
            self._start(self._refresh_status_now)

    def _refresh_status_now(self):
        self._set(portal="checking", last_status_check=time.monotonic())
        ok, wifi = wifi_status()
        if ok:
            self._set(wifi_connected=wifi["connected"], wifi_ssid=wifi["ssid"])
        with self._lock:
            provisioned = self._s["provisioned"]
        if not provisioned:
            self._set(portal="idle")
            return
        ok, result = _post(read_identity(), "/api/device/status")
        if ok:
            email = result.get("claimed_email")
            self._set(portal="idle", online=True, registered_email=email)
            write_cached_email(email or "")
        else:
            self._set(portal="idle", online=False,
                      connectivity=connectivity())

    def claim(self, email):
        def work():
            self._set(portal="claiming", error=None, notice=None)
            ok, result = _post(read_identity(), "/api/device/claim",
                               {"email": email})
            if ok:
                claimed = result.get("claimed_email")
                self._set(portal="idle", online=True, registered_email=claimed,
                          notice="Registered! Sign in at the portal with this "
                                 "email to open your simulator from anywhere.")
                write_cached_email(claimed or "")
            else:
                self._set(portal="idle", error=result)
        self._start(work)
