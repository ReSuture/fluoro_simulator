#!/usr/bin/env python

'''
FluoroSim
Code to run a fluoroscopy simulation as presented at SIR 2018

Based on opencv video_threaded.py sample (Multithreaded video processing sample):
https://github.com/opencv/opencv/tree/master/samples/python

Usage:
   fluoro_sim.py {<video device number>}

A second, dark-themed CONTROLS window (logo + clickable buttons) mirrors the web
control panel (fluoro_web.py): the same toggles and Fullscreen / Windowed / Quit
actions, driven by mouse clicks. The keyboard shortcuts below still work and stay
in sync with the on-screen buttons.

Keyboard shortcuts:
   ESC - exit
   Space - Toggle Peddle
   1 - Toggle Overlay
   2 - Toggle Subtraction
   3 - Fullscreen
   4 - Windowed mode
   5 - Toggle background overlay off/on (off = full raw video)
   6 - Equalize histogram
   7 - Toggle HUD (On screen text display)
'''

# Python 2/3 compatibility — ensures print() works the same in both versions
from __future__ import print_function

import numpy as np                          # Array operations used for image data (frames are numpy arrays)
import cv2 as cv                            # OpenCV — all camera capture, image processing, and display
import os                                   # Used to build the overlay image path relative to this script
from multiprocessing.pool import ThreadPool # Thread pool for parallel frame processing across CPU cores
from collections import deque              # Double-ended queue used as a FIFO buffer of pending frame tasks
import threading                            # Imported for thread-safety (not directly used but available)

# Build an absolute path to skel.jpg using the directory this script lives in.
# This ensures the overlay image is found regardless of where the script is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OVERLAY_IMAGE = os.path.join(BASE_DIR, "skel.jpg")
LOGO_IMAGE = os.path.join(BASE_DIR, "static", "logosign_white.png")
print("Overlay path:", OVERLAY_IMAGE)

# Linux input device path for the USB foot switch (PCsensor FootSwitch).
# This path is used to detect physical pedal presses in real deployments.
PEDAL_PATH = "/dev/input/by-id/usb-PCsensor-FootSwitch-event-kbd"


# ── Shared control state ──────────────────────────────────────────────────────
# Single source of truth for the toggles/actions, shared by the keyboard handler,
# the main loop, and the CONTROLS-window mouse callback. Mirrors fluoro_web.py's
# `state` dict so both front-ends behave identically.
state = {
    "subtract": True,        # (2/1) background subtraction + inversion
    "overlay": True,         # (1/5) anatomy overlay; off => full raw video
    "equalize": True,        # (6) CLAHE histogram equalisation
    "hud": True,             # (7) on-screen text HUD on the FLUORO view
    "pedal_mode": False,     # (Space) only capture while the pedal is pressed
    "pedal_pressed": False,  # latched pedal stand-in (also driven momentarily by 'b')
    "fullscreen": False,     # (3/4) FLUORO window fullscreen vs. windowed
    "quit": False,           # set by the Quit button / ESC to stop the loop
}

# Button hit-boxes, rebuilt every render as (x, y, w, h, kind, name) tuples
# (kind is "toggle" or "action"). `ctrl_buttons` hold positions inside the
# separate CONTROLS window (windowed mode); `overlay_buttons` hold positions on
# the FLUORO frame when the panel is composited as a fullscreen overlay.
ctrl_buttons = []
overlay_buttons = []


# ── CONTROLS-window theme (BGR — matches the web panel's hex colours) ─────────
C_BG       = (20, 15, 11)     # #0b0f14  page background
C_BTN_BG   = (34, 26, 19)     # #131a22  button background
C_BTN_BD   = (63, 50, 38)     # #26323f  button border
C_ON_BG    = (42, 59, 16)     # #103b2a  active button background
C_ON_BD    = (106, 183, 18)   # #12b76a  active button border
C_ON_TX    = (182, 240, 122)  # #7af0b6  active button text
C_TEXT     = (243, 237, 231)  # #e7edf3  normal text
C_SUBTEXT  = (153, 138, 125)  # #7d8a99  ON/OFF sub-label
C_QUIT_BG  = (22, 20, 42)     # #2a1416  quit button background
C_QUIT_BD  = (39, 35, 91)     # #5b2327  quit button border
C_QUIT_TX  = (154, 154, 255)  # #ff9a9a  quit button text
C_DOT_OFF  = (56, 68, 240)    # #f04438  status dot (no live feed)
C_DOT_ON   = (106, 183, 18)   # #12b76a  status dot (live)


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


def render_controls(logo, live):
    '''Render the dark control panel (logo + button grid).

    Returns (image, buttons) where buttons is a list of (x, y, w, h, kind, name)
    hit-boxes relative to the panel's top-left. The caller places the panel —
    its own CONTROLS window when windowed, or composited onto the FLUORO frame
    when fullscreen — and offsets the hit-boxes to match.
    '''
    W, m, gap, top_pad = 380, 16, 10, 18
    bt, ba = 56, 52  # toggle / action button heights

    lw = W - 2 * m
    lh = int(logo.shape[0] * (lw / float(logo.shape[1]))) if logo is not None else 0
    H = top_pad + lh + 14 + 26 + 14 + bt * 3 + gap * 3 + ba + gap + ba + 16

    img = np.full((H, W, 3), C_BG, np.uint8)
    buttons = []
    y = top_pad

    # Logo, centred across the full content width.
    if logo is not None:
        paste_rgba(img, logo, m, y, lw)
    y += lh + 14

    # Live-status dot + title, centred as a row.
    title = "FluoroSim Controls"
    font = cv.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv.getTextSize(title, font, 0.6, 1)
    row_w = tw + 18
    tx0 = (W - row_w) // 2
    cv.circle(img, (tx0 + 5, y + 9), 5, C_DOT_ON if live else C_DOT_OFF, -1, cv.LINE_AA)
    cv.putText(img, title, (tx0 + 18, y + 9 + th // 2), font, 0.6, C_TEXT, 1, cv.LINE_AA)
    y += 26 + 14

    # Toggle grid (2 columns), mirroring the web panel's toggles.
    cw = (W - 2 * m - gap) // 2
    toggles = [("Subtraction", "subtract"), ("Overlay", "overlay"),
               ("Equalize", "equalize"), ("HUD", "hud"),
               ("Pedal mode", "pedal_mode"), ("Pedal press", "pedal_pressed")]
    for i, (label, key) in enumerate(toggles):
        col, row = i % 2, i // 2
        bx = m + col * (cw + gap)
        by = y + row * (bt + gap)
        on = bool(state[key])
        draw_button(img, bx, by, cw, bt, label, "ON" if on else "OFF", on)
        buttons.append((bx, by, cw, bt, "toggle", key))
    y += 3 * (bt + gap)

    # Fullscreen / Windowed actions (highlighted to reflect the current mode).
    draw_button(img, m, y, cw, ba, "Fullscreen", None, state["fullscreen"])
    buttons.append((m, y, cw, ba, "action", "fullscreen"))
    draw_button(img, m + cw + gap, y, cw, ba, "Windowed", None, not state["fullscreen"])
    buttons.append((m + cw + gap, y, cw, ba, "action", "windowed"))
    y += ba + gap

    # Quit, full width.
    draw_button(img, m, y, W - 2 * m, ba, "Quit simulator", None, False, "quit")
    buttons.append((m, y, W - 2 * m, ba, "action", "quit"))
    return img, buttons


def render_control_bar(width, live):
    '''Render a short, full-width control strip for fullscreen mode.

    The buttons are laid out horizontally (6 toggles on top, the 3 actions
    below) so the bar stays thin — it is stacked *below* the fluoro video rather
    than covering it. Returns (image, buttons) with hit-boxes relative to the
    bar's top-left.
    '''
    pad, gap, bh = 6, 6, 34
    bar_h = pad + bh + gap + bh + pad
    img = np.full((bar_h, width, 3), C_BG, np.uint8)
    cv.line(img, (0, 0), (width, 0), C_BTN_BD, 1, cv.LINE_AA)  # top divider
    buttons = []

    # Row 1 — the six toggles. Single-line labels; the green highlight (not an
    # ON/OFF caption) signals state, which keeps the bar thin.
    toggles = [("Subtraction", "subtract"), ("Overlay", "overlay"),
               ("Equalize", "equalize"), ("HUD", "hud"),
               ("Pedal mode", "pedal_mode"), ("Pedal press", "pedal_pressed")]
    cw = (width - 2 * pad - 5 * gap) // 6
    y = pad
    for i, (label, key) in enumerate(toggles):
        bx = pad + i * (cw + gap)
        on = bool(state[key])
        draw_button(img, bx, y, cw, bh, label, None, on)
        buttons.append((bx, y, cw, bh, "toggle", key))

    # Row 2 — Fullscreen / Windowed / Quit.
    y += bh + gap
    aw = (width - 2 * pad - 2 * gap) // 3
    draw_button(img, pad, y, aw, bh, "Fullscreen", None, state["fullscreen"])
    buttons.append((pad, y, aw, bh, "action", "fullscreen"))
    draw_button(img, pad + aw + gap, y, aw, bh, "Windowed", None, not state["fullscreen"])
    buttons.append((pad + aw + gap, y, aw, bh, "action", "windowed"))
    draw_button(img, pad + 2 * (aw + gap), y, aw, bh, "Quit simulator", None, False, "quit")
    buttons.append((pad + 2 * (aw + gap), y, aw, bh, "action", "quit"))
    return img, buttons


def apply_button(kind, name):
    '''Apply a button press to the shared `state` dict (toggles flip, actions set).'''
    if kind == "toggle":
        state[name] = not state[name]
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
    '''Click handler for buttons overlaid on the FLUORO frame (fullscreen mode).'''
    if event == cv.EVENT_LBUTTONDOWN:
        hit = _hit(overlay_buttons, x, y)
        if hit:
            apply_button(*hit)


def create_capture(source=0):
    '''Open a video capture device or file.

    source: <int> or '<int>|<filename>|synth [:<param_name>=<value> [:...]]'

    Parses the source string to separate the device/filename from optional
    parameters (e.g. size=640x480), opens the camera using the V4L2 backend
    (Linux), and optionally sets the capture resolution.
    '''
    source = str(source).strip()

    # Split on ':' to separate the device identifier from key=value parameters.
    chunks = source.split(':')

    # Edge case: Windows drive letters like 'C:' look like a parameter separator.
    # Re-join the drive letter with the rest of the path so it isn't lost.
    if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha():
        chunks[1] = chunks[0] + ':' + chunks[1]
        del chunks[0]

    source = chunks[0]

    # Try to convert the source to an integer (camera index).
    # If it can't be converted, leave it as a string (file path).
    try:
        source = int(source)
    except ValueError:
        pass

    # Parse any remaining 'key=value' pairs after the device identifier.
    params = dict(s.split('=') for s in chunks[1:])

    # Open the capture using the V4L2 backend — required for Linux webcams.
    cap = cv.VideoCapture(source, cv.CAP_V4L2)

    # If a 'size' parameter was provided (e.g. size=1280x720), apply it.
    if 'size' in params:
        w, h = map(int, params['size'].split('x'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)

    return cap


def clock():
    '''Return the current time in seconds using OpenCV's high-resolution tick counter.

    Used instead of time.time() because cv.getTickCount() is backed by the same
    clock OpenCV uses internally, so latency measurements stay consistent.
    '''
    return cv.getTickCount() / cv.getTickFrequency()


def draw_str(dst, target, s):
    '''Draw a white string with a black drop-shadow on image dst at position target.

    Draws the text twice: once offset by 1px in black (shadow) and once at the
    exact position in white. This keeps the text legible over both bright and
    dark backgrounds without needing a separate background rectangle.
    '''
    x, y = target
    # Black shadow — drawn slightly offset (+1, +1) and thicker so it bleeds around the white text
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    # White foreground text on top of the shadow
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


class StatValue:
    '''Exponential moving average for smoothing noisy scalar measurements.

    Used to smooth the latency and frame-interval readings so the HUD displays
    stable numbers rather than jumping around every frame.

    smooth_coef controls how much weight the historical average gets vs. the
    newest sample: higher = smoother (slower to react), lower = more reactive.
    '''

    def __init__(self, smooth_coef=0.5):
        self.value = None              # None until the first sample arrives
        self.smooth_coef = smooth_coef # Blending weight for the historical value

    def update(self, v):
        if self.value is None:
            # First sample — seed the average with the raw value
            self.value = v
        else:
            c = self.smooth_coef
            # Exponential moving average: blend previous average with new sample
            self.value = c * self.value + (1.0 - c) * v


class DummyTask:
    '''Wraps a synchronously computed result to look like an AsyncResult.

    When threaded_mode is False, process_frame() is called directly instead of
    being submitted to the thread pool. DummyTask makes the result compatible
    with the same ready()/get() interface used for real async tasks, so the
    main loop doesn't need a separate code path.
    '''

    def __init__(self, data):
        self.data = data

    def ready(self):
        # Always ready — the result was computed synchronously
        return True

    def get(self):
        return self.data


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    # Print the module docstring (usage + keyboard shortcuts) to the terminal
    print(__doc__)

    # Accept an optional command-line argument for the camera device index.
    # Defaults to 0 (first webcam) if not provided.
    try:
        fn = sys.argv[1]
    except:
        fn = 0

    cap = create_capture(fn)

    # Create the display window.  WND_PROP_FULLSCREEN makes it capable of going
    # fullscreen; WINDOW_KEEPRATIO preserves the camera's aspect ratio on resize.
    cv.namedWindow("FLUORO", cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("FLUORO", cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)
    # Clicks on the FLUORO window only do anything in fullscreen, where the
    # control buttons are composited onto the frame (see overlay_buttons).
    cv.setMouseCallback("FLUORO", on_mouse_fluoro)

    # ── Control panel window ──────────────────────────────────────────────────
    # A separate, auto-sized window holding the logo + clickable buttons, shown
    # in windowed mode. In fullscreen it is closed and the buttons move onto the
    # FLUORO frame instead. Clicks route through on_mouse_controls into `state`.
    def show_controls_window():
        cv.namedWindow("CONTROLS", cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback("CONTROLS", on_mouse_controls)
        cv.moveWindow("CONTROLS", 20, 20)

    show_controls_window()
    logo = cv.imread(LOGO_IMAGE, cv.IMREAD_UNCHANGED)
    if logo is None:
        print("Warning: logo not found at", LOGO_IMAGE)

    # ── Load the anatomy overlay image ────────────────────────────────────────
    # skel.jpg is a static skeletal/anatomy image that is composited onto the
    # live feed to mimic how a fluoroscope shows anatomy underneath the subject.
    overlay = cv.imread(OVERLAY_IMAGE)
    if overlay is None:
        raise FileNotFoundError(f"Cannot load image: {OVERLAY_IMAGE}")

    # Convert the overlay to grayscale to match the grayscale live feed.
    # The simulator works entirely in single-channel (grayscale) images.
    overlay = cv.cvtColor(overlay, cv.COLOR_RGB2GRAY)
    # Flip options (disabled) — uncomment to mirror the overlay if needed:
    #overlay = cv.flip(overlay, 1)  # horizontal flip
    #overlay = cv.flip(overlay, 0)  # vertical flip

    # ── Per-frame processing function (runs inside the thread pool) ───────────
    def process_frame(frame, raw, t0, subtract_mode, overlay_mode, equalize_mode):
        '''Apply fluoroscopy-style image processing to a single grayscale frame.

        Parameters
        ----------
        frame         : uint8 grayscale numpy array — the (possibly subtracted) camera frame
        raw           : uint8 grayscale numpy array — the untouched camera frame (no subtraction/inversion)
        t0            : float — timestamp when the frame was captured (passed through for latency calc)
        subtract_mode : bool  — whether background subtraction has already been applied upstream
        overlay_mode  : bool  — whether to composite the anatomy overlay onto the frame
        equalize_mode : bool  — whether to apply histogram equalisation for contrast enhancement

        Returns
        -------
        (processed_frame, t0)
        '''

        # Brightness threshold above which a pixel is considered "white background".
        # Pixels at or above this value are the bright surface the vasculature rests on.
        # Lowering this value catches more of the surface; raising it is more conservative.
        # Overlay off — show the full raw video. `raw` is the untouched grayscale camera
        # frame (no subtraction, inversion, masking, or CLAHE), so nothing is removed and
        # the brightest/whitest areas stay white.
        if not overlay_mode:
            return raw, t0

        mask_threshold = 220

        gray = frame  # Work on a local reference; frame is already grayscale from the main loop

        # Build a binary mask that identifies non-white (vasculature/tissue) pixels.
        # THRESH_BINARY_INV: pixels > mask_threshold → 0, pixels <= mask_threshold → 255
        # Result: bg_mask is 255 where the frame is NOT white, and 0 where it IS white.
        _, bg_mask = cv.threshold(gray, mask_threshold, 255, cv.THRESH_BINARY_INV)

        # Median blur removes small noise speckles from the mask boundary.
        # Kernel size 5 smooths without significantly blurring the vasculature edges.
        bg_mask = cv.medianBlur(bg_mask, 5)

        # ── Overlay compositing ───────────────────────────────────────────────
        if overlay_mode:
            # White areas (bg_mask == 0): 30% video, 70% overlay (background shows through)
            # Non-white areas (vasculature, bg_mask > 0): 60% video, 40% overlay
            ov = overlay.astype(np.float32)
            fr = gray.astype(np.float32)
            result = ov.copy()
            white = bg_mask == 0
            result[white]        = 0.30 * fr[white]        + 0.70 * ov[white]
            result[~white]       = 0.60 * fr[~white]       + 0.40 * ov[~white]
            gray = np.clip(result, 0, 255).astype(np.uint8)

        # ── Histogram equalisation ────────────────────────────────────────────
        if equalize_mode:
            # CLAHE boosts contrast locally so vasculature in low-contrast regions
            # is enhanced without blowing out areas that are already high-contrast.
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        return gray, t0

    # ── Thread pool setup ─────────────────────────────────────────────────────
    # One worker thread per CPU core. Each thread processes a single frame
    # independently; finished frames are collected in arrival order via `pending`.
    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes=threadn)

    # FIFO queue of AsyncResult / DummyTask objects for frames currently being processed.
    # The main loop submits frames to the pool and drains finished results from this queue.
    pending = deque()

    # ── Mode flags ────────────────────────────────────────────────────────────
    # The user-facing toggles live in the shared `state` dict (driven by both the
    # keyboard and the CONTROLS buttons). threaded_mode stays a private local.
    threaded_mode = True   # Use the thread pool (False = process synchronously)

    # ── Background model ──────────────────────────────────────────────────────
    # `background` is a float32 accumulator updated each frame via accumulateWeighted.
    # Initialised to None; seeded from the first captured frame.
    background = None
    applied_fullscreen = None  # last fullscreen state pushed to the FLUORO window
    live = False               # True once a frame has been read (drives the status dot)
    res = None                 # latest processed frame to display

    # Learning rate for cv.accumulateWeighted: background = (1-alpha)*background + alpha*frame
    # 0.05 means the background adapts slowly — a new object takes ~20 frames to be absorbed.
    alpha = 0.05

    # ── Timing stats ─────────────────────────────────────────────────────────
    latency        = StatValue()  # Time from frame capture to display (smoothed)
    frame_interval = StatValue()  # Time between consecutive frame captures (smoothed)
    last_frame_time = clock()

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        # Poll for a keypress without blocking (1 ms wait).
        # Mask to 8 bits to avoid issues with extended key codes on some platforms.
        key = cv.waitKey(1) & 0xFF

        # 'b' is a momentary keyboard stand-in for the physical foot pedal; the
        # "Pedal press" button latches state["pedal_pressed"]. Either one counts.
        key_pedal = (key == ord('b'))
        pedal_down = state["pedal_pressed"] or key_pedal

        # Keep the FLUORO window's fullscreen state in sync with the toggle, and
        # move the controls between the separate CONTROLS window (windowed) and a
        # fullscreen overlay on the FLUORO frame.
        if state["fullscreen"] != applied_fullscreen:
            cv.setWindowProperty(
                "FLUORO", cv.WND_PROP_FULLSCREEN,
                cv.WINDOW_FULLSCREEN if state["fullscreen"] else cv.WINDOW_NORMAL)
            if state["fullscreen"]:
                # Buttons overlay the fluoro image — drop the separate window.
                cv.destroyWindow("CONTROLS")
            else:
                # Back to windowed — restore the separate control panel window.
                show_controls_window()
                overlay_buttons[:] = []
            applied_fullscreen = state["fullscreen"]

        # ── Drain completed frames from the thread pool ───────────────────────
        # Check the front of the queue; pop and display frames that are done.
        while len(pending) > 0 and pending[0].ready():
            res, t0 = pending.popleft().get()

            # Update the smoothed latency stat (used by the commented-out HUD lines)
            latency.update(clock() - t0)

            # ── HUD overlay ───────────────────────────────────────────────────
            if state["hud"]:
                # Draw current mode states in the top-left corner of the frame.
                # Lines are spaced 20px apart so they don't overlap.
                draw_str(res, (20, 20), "(1)Toggle Subtraction : " + str(state["subtract"]) + "  (3)Fullscreen (4)Windowed")
                draw_str(res, (20, 40), "(2)Toggle Overlay     : " + str(state["overlay"])  + "   (5)Toggle Background (6)EqualizeHist")
                draw_str(res, (20, 60), "(Space) Toggle Peddle : " + str(state["pedal_mode"]) + "  (7)Toggle HUD")
                # Uncomment below to show timing diagnostics in the HUD:
                #draw_str(res, (20, 80), "latency        :  %.1f ms" % (latency.value*1000))
                #draw_str(res, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value*1000))
                #draw_str(res, (20, 80), "threaded      :  " + str(threaded_mode))

            # Show "PEDAL ACTIVE" near the bottom when the pedal is pressed
            if pedal_down:
                draw_str(res, (20, 450), "PEDAL ACTIVE")
            # `res` is displayed below (clean, or with the fullscreen overlay).

        # ── Capture and submit a new frame ────────────────────────────────────
        # Only submit a new frame if the pool has room (queue shorter than thread count).
        # This prevents unbounded accumulation of pending tasks if processing is slow.
        if len(pending) < threadn:

            # In pedal mode, only grab a frame when the pedal is pressed.
            # When pedal mode is off, capture continuously.
            if pedal_down or state["pedal_mode"] == False:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to read from camera")
                live = True

                # Convert the colour frame to grayscale.
                # All subsequent processing is single-channel for simplicity and speed.
                frame_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

                # Keep an untouched copy of the raw grayscale frame BEFORE any subtraction
                # or inversion. This is what gets displayed when the overlay is toggled off,
                # so the full video shows with its bright/white areas intact.
                frame_raw = frame_gray.copy()

                # Resize the overlay to exactly match the current frame dimensions.
                # Done every frame because the first frame establishes the true resolution.
                overlay = cv.resize(overlay, (frame_gray.shape[1], frame_gray.shape[0]))

                # Seed the background accumulator from the very first frame.
                # float32 is required by cv.accumulateWeighted.
                if background is None:
                    background = frame_gray.astype("float")
                # Disabled: continuous background adaptation while capturing
                # (left off so the background only updates via key '5')
                # else:
                #     cv.accumulateWeighted(frame_gray, background, alpha)

                # Convert the float32 background accumulator to uint8 for subtraction.
                bg_uint8 = cv.convertScaleAbs(background)

                if state["subtract"]:
                    # absdiff produces a difference image: pixels that match the
                    # background are near 0; pixels that differ (vasculature) are brighter.
                    frame_gray = cv.absdiff(frame_gray, bg_uint8)
                    # Stretch small vasculature differences to the full 0-255 range so
                    # they remain visibly dark after inversion instead of washing out.
                    cv.normalize(frame_gray, frame_gray, 0, 255, cv.NORM_MINMAX)
                    # Invert: vasculature (bright diff) → dark, background (zero diff) → white.
                    frame_gray = cv.bitwise_not(frame_gray)

                # Slowly update the background model so it adapts to gradual lighting
                # changes. alpha=0.05 means each new frame contributes 5% of the background.
                cv.accumulateWeighted(frame_gray, background, alpha)

                # Record timing for frame-interval stat
                t = clock()
                frame_interval.update(t - last_frame_time)
                last_frame_time = t

                # Submit frame for processing — threaded or synchronous depending on mode
                if threaded_mode:
                    # Send a copy of the frame to avoid a race condition: the main loop
                    # may modify frame_gray before the thread reads it without the copy.
                    task = pool.apply_async(process_frame, (frame_gray.copy(), frame_raw.copy(), t, state["subtract"], state["overlay"], state["equalize"]))
                else:
                    # Wrap the synchronous result in DummyTask so the drain loop above
                    # can call .ready() and .get() on it without any special casing.
                    task = DummyTask(process_frame(frame_gray, frame_raw, t, state["subtract"], state["overlay"], state["equalize"]))

                pending.append(task)

        # ── Display: fluoro view + controls ────────────────────────────────────
        # Windowed: the fluoro view stays clean and the vertical panel lives in
        # its own CONTROLS window. Fullscreen: stack the video on top and a thin
        # control bar below it (so the controls never cover the video), shown as
        # one composited image in the FLUORO window.
        if state["fullscreen"]:
            if res is not None:
                vid = cv.cvtColor(res, cv.COLOR_GRAY2BGR) if res.ndim == 2 else res
                bar, bbtns = render_control_bar(vid.shape[1], live)
                composite = np.vstack([vid, bar])
                # Offset the bar's hit-boxes by the video height so clicks on the
                # fullscreen window land on the right buttons.
                overlay_buttons[:] = [(x, y + vid.shape[0], w, h, k, n)
                                      for (x, y, w, h, k, n) in bbtns]
                cv.imshow("FLUORO", composite)
        else:
            panel, btns = render_controls(logo, live)
            ctrl_buttons[:] = btns
            cv.imshow("CONTROLS", panel)
            if res is not None:
                cv.imshow("FLUORO", res)

        # ── Keyboard shortcut handling ─────────────────────────────────────────
        # Second waitKey call at the bottom of the loop catches presses that
        # arrive during the frame processing work above. Shortcuts mutate the same
        # `state` dict the buttons do, so the two stay in lock-step.
        ch = cv.waitKey(1)

        # Space — toggle pedal mode (continuous capture vs. pedal-gated capture)
        if ch == ord(' '):
            state["pedal_mode"] = not state["pedal_mode"]

        if ch == 49:   # '1' — toggle background subtraction on/off
            state["subtract"] = not state["subtract"]

        if ch == 50:   # '2' — toggle anatomy overlay compositing on/off
            state["overlay"] = not state["overlay"]

        if ch == 51:   # '3' — switch display window to fullscreen
            state["fullscreen"] = True

        if ch == 52:   # '4' — return display window to normal (windowed) mode
            state["fullscreen"] = False

        if ch == 53:   # '5' — toggle the background overlay off/on (off = full raw video)
            state["overlay"] = not state["overlay"]

        if ch == 54:   # '6' — toggle histogram equalisation on/off
            state["equalize"] = not state["equalize"]

        if ch == 55:   # '7' — toggle the on-screen HUD on/off
            state["hud"] = not state["hud"]

        if ch == 27:   # ESC — exit the main loop
            state["quit"] = True

        # Quit requested by ESC or the Quit button.
        if state["quit"]:
            break

# Release all OpenCV windows after the loop exits
cv.destroyAllWindows()
