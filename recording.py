'''FluoroSim session recording + the Library tab's backend.

Recorder writes the processed fluoro frames to MJPG .avi files (OpenCV's
built-in encoder — no external codecs needed on the Pi) in
~/fluorosim_recordings. Frames are handed off through a bounded queue to a
daemon writer thread, so a slow SD card drops frames instead of stalling the
live render loop.
'''

from __future__ import print_function

import os
import queue
import threading
import time

import cv2 as cv

RECORDINGS_DIR = os.path.expanduser("~/fluorosim_recordings")
FOURCC = "MJPG"
EXT = ".avi"
FPS = 20.0

_STOP = object()  # queue sentinel


class Recorder:
    '''Feed processed frames in with submit(); everything else is automatic.'''

    def __init__(self):
        self._queue = None
        self._thread = None
        self._started_at = None
        self._last_submit = 0.0
        self.error = None
        self.dropped = 0

    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive()

    def ensure_started(self, frame_shape):
        '''Open the writer for this session if not already open. -> ok bool.'''
        if self.active:
            return True
        self.error = None
        self.dropped = 0
        h, w = frame_shape[:2]
        try:
            os.makedirs(RECORDINGS_DIR, exist_ok=True)
        except OSError as exc:
            self.error = "Cannot create %s: %s" % (RECORDINGS_DIR, exc)
            return False
        path = os.path.join(RECORDINGS_DIR,
                            time.strftime("fluoro_%Y%m%d_%H%M%S") + EXT)
        writer = cv.VideoWriter(path, cv.VideoWriter_fourcc(*FOURCC), FPS, (w, h))
        if not writer.isOpened():
            self.error = "Cannot open video writer for %s" % path
            return False
        self._queue = queue.Queue(maxsize=90)
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._worker,
                                        args=(writer, self._queue), daemon=True)
        self._thread.start()
        return True

    def submit(self, frame):
        '''Queue one frame for writing; never blocks the render loop.

        Each frame carries its capture time; the writer duplicates frames as
        needed so the file's timeline tracks wall time whatever rate the
        camera actually delivers (Pi cameras are often well under FPS).
        '''
        if not self.active:
            return
        now = time.monotonic()
        if now - self._last_submit < 1.0 / FPS - 0.002:
            return  # camera faster than FPS: thin the stream
        self._last_submit = now
        try:
            self._queue.put_nowait((frame.copy(), now))
        except queue.Full:
            self.dropped += 1

    def stop(self, wait=False):
        '''Finish the current file (idempotent). Non-blocking by default; pass
        wait=True on process exit so the writer can finalize the .avi before
        os._exit kills its daemon thread.'''
        thread = self._thread
        if self._queue is not None:
            try:
                self._queue.put_nowait(_STOP)
            except queue.Full:
                # Make room for the sentinel; losing one tail frame is fine.
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(_STOP)
        self._queue = None
        self._thread = None
        self._started_at = None
        if wait and thread is not None:
            thread.join(timeout=10)

    def elapsed_str(self):
        if self._started_at is None:
            return "0:00"
        t = int(time.monotonic() - self._started_at)
        return "%d:%02d" % (t // 60, t % 60)

    def _worker(self, writer, q):
        # The queue is passed in (not read from self) so stop() detaching the
        # attribute can't race us — we drain until the sentinel, then close.
        # A slow camera would make a fixed-FPS file play back fast, so each
        # frame is written as many times as its timestamp says the timeline
        # has advanced (a paused/slow feed becomes a frozen image, not a jump).
        t0 = None
        written = 0
        while True:
            item = q.get()
            if item is _STOP:
                break
            frame, ts = item
            if frame.ndim == 2:
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
            if t0 is None:
                t0 = ts
            target = max(written + 1, int(round((ts - t0) * FPS)) + 1)
            while written < target:
                writer.write(frame)
                written += 1
        writer.release()


def list_recordings():
    '''Recordings newest-first: [{"path","name","mtime","size"}].'''
    try:
        names = [n for n in os.listdir(RECORDINGS_DIR) if n.endswith(EXT)]
    except OSError:
        return []
    entries = []
    for n in names:
        path = os.path.join(RECORDINGS_DIR, n)
        try:
            st = os.stat(path)
        except OSError:
            continue
        entries.append({"path": path, "name": n,
                        "mtime": st.st_mtime, "size": st.st_size})
    return sorted(entries, key=lambda e: -e["mtime"])


def probe_durations(paths, cache, lock):
    '''Worker-thread helper: fill cache[path] = "m:ss" for each unknown path.'''
    for path in paths:
        with lock:
            if path in cache:
                continue
        cap = cv.VideoCapture(path)
        frames = cap.get(cv.CAP_PROP_FRAME_COUNT) or 0
        fps = cap.get(cv.CAP_PROP_FPS) or FPS
        cap.release()
        seconds = int(frames / fps) if fps > 0 else 0
        with lock:
            cache[path] = "%d:%02d" % (seconds // 60, seconds % 60)


def delete_recording(path):
    '''Remove a recording; refuses paths outside RECORDINGS_DIR. -> ok bool.'''
    real = os.path.realpath(path)
    if not real.startswith(os.path.realpath(RECORDINGS_DIR) + os.sep):
        return False
    try:
        os.remove(real)
        return True
    except OSError:
        return False
