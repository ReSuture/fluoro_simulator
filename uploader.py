'''Auto-upload of finished session recordings to the portal library.

A single daemon worker watches ~/fluorosim_recordings: when a recording has
been finished for a few seconds (mtime settled, and it isn't the file the
Recorder is still writing), it is transcoded to a browser-playable H.264 MP4
with ffmpeg and POSTed to the portal's /api/device/videos, authenticated with
the same per-device claim secret the Remote Access tab uses (device_setup).

Rules, mirroring device_setup.py:

1. Nothing here runs on the OpenCV main thread — the render loop only calls
   the non-blocking snapshot()/status_for()/poke() methods.
2. Nothing here can break recording or the sim: every ffmpeg/portal failure
   becomes a per-file status shown on the Library tab, and a device that was
   never provisioned just leaves the uploader idle.

Progress is persisted in RECORDINGS_DIR/.uploads.json so already-uploaded
recordings aren't re-sent after a restart; the portal additionally dedupes by
file hash, so losing that state costs bandwidth, not duplicates.
'''

from __future__ import print_function

import hashlib
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid

import cv2 as cv

import device_setup
import recording

STATE_FILE = os.path.join(recording.RECORDINGS_DIR, ".uploads.json")
TMP_DIR = os.path.join(recording.RECORDINGS_DIR, ".upload_tmp")

SETTLE_SECONDS = 10     # a recording must be this quiet before it's "finished"
POLL_SECONDS = 60       # background rescan interval (poke() forces one sooner)
UPLOAD_TIMEOUT = 600    # generous: sessions upload over home/hospital uplinks
MAX_UPLOAD_BYTES = 95 * 1024 * 1024  # keep under Cloudflare's 100 MB request cap
BACKOFF_SECONDS = (60, 300, 900, 3600)  # transient-failure retry schedule

# H.264 + yuv420p + faststart is the portable HTML5 <video> combination. The
# encode is niced so a long transcode never competes with the live sim loop.
FFMPEG_CMD = ["nice", "-n", "10", "ffmpeg", "-y", "-nostdin",
              "-loglevel", "error", "-i", "{src}",
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
              "-pix_fmt", "yuv420p", "-movflags", "+faststart", "{dst}"]


def _load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            state = json.load(f)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(state):
    try:
        os.makedirs(recording.RECORDINGS_DIR, exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp, STATE_FILE)
    except OSError:
        pass  # worst case we re-upload and the portal dedupes


def _probe_duration_s(path):
    cap = cv.VideoCapture(path)
    frames = cap.get(cv.CAP_PROP_FRAME_COUNT) or 0
    fps = cap.get(cv.CAP_PROP_FPS) or recording.FPS
    cap.release()
    return int(frames / fps) if fps > 0 else 0


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _multipart(fields, file_name, file_path):
    '''Encode fields + one file part. Returns (body_bytes, content_type).

    The MP4 is read into memory — it is capped at MAX_UPLOAD_BYTES, well
    within a Pi's RAM, and it keeps this stdlib-only (no requests dependency).
    '''
    boundary = "----fluorosim-%s" % uuid.uuid4().hex
    lines = []
    for key, value in fields.items():
        lines += [b"--" + boundary.encode(),
                  ('Content-Disposition: form-data; name="%s"' % key).encode(),
                  b"", str(value).encode()]
    lines += [b"--" + boundary.encode(),
              ('Content-Disposition: form-data; name="video"; filename="%s"'
               % file_name).encode(),
              b"Content-Type: video/mp4", b""]
    with open(file_path, "rb") as f:
        lines.append(f.read())
    lines += [b"--" + boundary.encode() + b"--", b""]
    return b"\r\n".join(lines), "multipart/form-data; boundary=%s" % boundary


class Uploader:
    '''Owns upload state; the render loop reads it via snapshot()/status_for().

    Per-file status codes:
      uploaded | queued | working | retrying | failed | (None = untracked)
    '''

    def __init__(self, active_path_fn=lambda: None):
        self._active_path_fn = active_path_fn
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._state = _load_state()      # name -> {"status","video_id","error",...}
        self._live = {}                  # name -> "queued"|"working"|"retrying"
        self._retry_at = {}              # name -> (monotonic deadline, tries)
        self._summary = None             # one-line notice for the Library tab
        self._thread = None

    # ── main-thread API ──────────────────────────────────────────────────────

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def poke(self):
        '''Nudge the worker (called when a recording stops); never blocks.'''
        self._wake.set()

    def status_for(self, name):
        with self._lock:
            entry = self._state.get(name)
            if entry and entry.get("status") == "uploaded":
                return "uploaded"
            if entry and entry.get("permanent"):
                return "failed"
            return self._live.get(name)

    def snapshot(self):
        with self._lock:
            return {"summary": self._summary, "live": dict(self._live)}

    # ── worker ───────────────────────────────────────────────────────────────

    def _set_summary(self, text):
        with self._lock:
            self._summary = text

    def _set_live(self, name, status):
        with self._lock:
            if status is None:
                self._live.pop(name, None)
            else:
                self._live[name] = status

    def _record(self, name, **entry):
        with self._lock:
            self._state[name] = entry
            _save_state(self._state)

    def _worker(self):
        while True:
            self._wake.wait(POLL_SECONDS)
            self._wake.clear()
            try:
                self._scan_once()
            except Exception as exc:  # never kill the thread
                self._set_summary("Upload worker error: %s" % exc)

    def _candidates(self):
        '''Finished, not-yet-uploaded recordings, oldest first.'''
        active = self._active_path_fn()
        now = time.time()
        entries = []
        for e in recording.list_recordings():
            entry = self._state.get(e["name"], {})
            if entry.get("status") == "uploaded" or entry.get("permanent"):
                continue
            if e["path"] == active or now - e["mtime"] < SETTLE_SECONDS:
                continue  # still being written (or just closed) — next pass
            deadline, _tries = self._retry_at.get(e["name"], (0, 0))
            if time.monotonic() < deadline:
                continue
            entries.append(e)
        return sorted(entries, key=lambda e: e["mtime"])

    def _scan_once(self):
        identity = device_setup.read_identity()
        if not all(identity.values()):
            self._set_summary(None)  # unprovisioned: quietly do nothing
            return
        # Forget state for recordings deleted from the Library tab.
        with self._lock:
            names = {e["name"] for e in recording.list_recordings()}
            stale = [n for n in self._state if n not in names]
            for n in stale:
                del self._state[n]
            if stale:
                _save_state(self._state)

        todo = self._candidates()
        for e in todo:
            self._set_live(e["name"], "queued")
        for e in todo:
            self._process(identity, e)
        with self._lock:
            done = sum(1 for v in self._state.values()
                       if v.get("status") == "uploaded")
            pending = len(self._live)
            failed = sum(1 for v in self._state.values() if v.get("permanent"))
        if pending or failed:
            self._set_summary("Uploads: %d done, %d pending, %d failed"
                              % (done, pending, failed))
        elif done:
            self._set_summary("All recordings uploaded to the portal library")

    def _process(self, identity, e):
        name = e["name"]
        self._set_live(name, "working")
        mp4 = None
        try:
            mp4 = self._transcode(e["path"])
            if mp4 is None:
                return  # ffmpeg missing — summary set, retry next scan
            if os.path.getsize(mp4) > MAX_UPLOAD_BYTES:
                self._record(name, status="failed", permanent=True,
                             error="Too large to upload — split the session")
                self._set_live(name, None)
                return
            ok, result, retryable = self._upload(identity, e, mp4)
            if ok:
                self._record(name, status="uploaded",
                             video_id=result.get("video_id"))
                self._set_live(name, None)
                self._retry_at.pop(name, None)
            elif retryable:
                _deadline, tries = self._retry_at.get(name, (0, 0))
                delay = BACKOFF_SECONDS[min(tries, len(BACKOFF_SECONDS) - 1)]
                self._retry_at[name] = (time.monotonic() + delay, tries + 1)
                self._set_live(name, "retrying")
                self._set_summary("Upload of %s failed (%s) — will retry"
                                  % (name, result))
            else:
                self._record(name, status="failed", permanent=True, error=result)
                self._set_live(name, None)
                self._set_summary("Upload of %s failed: %s" % (name, result))
        finally:
            if mp4 is not None:
                try:
                    os.remove(mp4)
                except OSError:
                    pass

    def _transcode(self, src):
        '''AVI → temporary MP4. Returns the mp4 path, or None on failure.'''
        os.makedirs(TMP_DIR, exist_ok=True)
        dst = os.path.join(
            TMP_DIR, os.path.basename(src)[:-len(recording.EXT)] + ".mp4")
        cmd = [a.format(src=src, dst=dst) for a in FFMPEG_CMD]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            try:  # no `nice` (unlikely) — try ffmpeg bare before giving up
                p = subprocess.run(cmd[3:], capture_output=True, text=True)
            except FileNotFoundError:
                self._set_summary("ffmpeg is not installed — run: "
                                  "sudo apt install -y ffmpeg")
                return None
        if p.returncode != 0 or not os.path.exists(dst):
            tail = (p.stderr or "").strip().splitlines()
            self._set_summary("Transcode failed: %s"
                              % (tail[-1] if tail else "ffmpeg error"))
            return None
        return dst

    def _upload(self, identity, e, mp4):
        '''-> (ok, json-or-error-message, retryable)'''
        self._set_live(e["name"], "working")
        fields = {
            "device_id": identity["device_id"],
            "claim_secret": identity["claim_secret"],
            "sha256": _sha256_file(mp4),
            "duration_s": _probe_duration_s(e["path"]),
            "recorded_at": e["mtime"],
        }
        body, content_type = _multipart(
            fields, os.path.basename(mp4), mp4)
        req = urllib.request.Request(
            identity["portal_url"] + "/api/device/videos", data=body,
            headers={"Content-Type": content_type,
                     "User-Agent": device_setup.USER_AGENT},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=UPLOAD_TIMEOUT) as resp:
                return True, json.load(resp), False
        except urllib.error.HTTPError as exc:
            try:
                message = json.load(exc).get("error", "")
            except Exception:
                message = ""
            message = message or "Portal error (HTTP %d)" % exc.code
            # 4xx (bar rate limiting) won't fix itself by resending the same
            # bytes; everything else — 429, 5xx, quota-full — is worth retrying.
            permanent = 400 <= exc.code < 500 and exc.code != 429
            return False, message, not permanent
        except Exception as exc:
            return False, "Cannot reach the portal: %s" % getattr(exc, "reason", exc), True
