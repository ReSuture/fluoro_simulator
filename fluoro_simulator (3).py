#!/usr/bin/env python

'''
FluoroSim
Code to run a fluoroscopy simulation as presented at SIR 2018

Based on opencv video_threaded.py sample (Multithreaded video processing sample):
https://github.com/opencv/opencv/tree/master/samples/python

Usage:
   fluoro_sim.py {<video device number>}

Keyboard shortcuts:
   ESC - exit
   Space - Toggle Peddle
   1 - Toggle Overlay
   2 - Toggle Subtraction
   3 - Fullscreen
   4 - Windowed mode
   5 - Retake background image used in subtraction
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
print("Overlay path:", OVERLAY_IMAGE)

# Linux input device path for the USB foot switch (PCsensor FootSwitch).
# This path is used to detect physical pedal presses in real deployments.
PEDAL_PATH = "/dev/input/by-id/usb-PCsensor-FootSwitch-event-kbd"

# Global flag that tracks whether the foot pedal is currently pressed.
# Set to True when the 'b' key is held (keyboard stand-in for the real pedal).
pedal_pressed = False


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
    def process_frame(frame, t0, subtract_mode, overlay_mode, equalize_mode):
        '''Apply fluoroscopy-style image processing to a single grayscale frame.

        Parameters
        ----------
        frame         : uint8 grayscale numpy array — the (possibly subtracted) camera frame
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
            # Strategy: use the overlay as the base image (background), then paste
            # the live video feed on top only where pixels are NOT white.
            # This makes very white areas (the bright surface) "transparent" —
            # the overlay shows through in those regions instead.
            result = overlay.copy()                   # Start with pure overlay everywhere
            result[bg_mask > 0] = gray[bg_mask > 0]  # Paste vasculature/dark pixels from video feed
            gray = result

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

    # ── Mode flags (toggled by keyboard shortcuts) ────────────────────────────
    subtract_mode = True   # Subtract accumulated background from each frame and invert
    threaded_mode = True   # Use the thread pool (False = process synchronously)
    overlay_mode  = True   # Composite the anatomy overlay onto the frame
    equalize_mode = True   # Apply histogram equalisation
    peddle_mode   = False  # When True, only capture frames while the pedal is pressed
    hud_mode      = True   # Show the on-screen text HUD

    # ── Background model ──────────────────────────────────────────────────────
    # `background` is a float32 accumulator updated each frame via accumulateWeighted.
    # Initialised to None; seeded from the first captured frame.
    background = None

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

        # 'b' acts as a keyboard stand-in for the physical foot pedal.
        # The real pedal would set pedal_pressed via a separate input thread reading PEDAL_PATH.
        if key == ord('b'):
            pedal_pressed = True
        else:
            pedal_pressed = False

        # ── Drain completed frames from the thread pool ───────────────────────
        # Check the front of the queue; pop and display frames that are done.
        while len(pending) > 0 and pending[0].ready():
            res, t0 = pending.popleft().get()

            # Update the smoothed latency stat (used by the commented-out HUD lines)
            latency.update(clock() - t0)

            # ── HUD overlay ───────────────────────────────────────────────────
            if hud_mode:
                # Draw current mode states in the top-left corner of the frame.
                # Lines are spaced 20px apart so they don't overlap.
                draw_str(res, (20, 20), "(1)Toggle Subtraction : " + str(subtract_mode) + "  (3)Fullscreen (4)Windowed")
                draw_str(res, (20, 40), "(2)Toggle Overlay     : " + str(overlay_mode)  + "   (5)Take Background (6)EqualizeHist")
                draw_str(res, (20, 60), "(Space) Toggle Peddle : " + str(peddle_mode)   + "  (7)Toggle HUD")
                # Uncomment below to show timing diagnostics in the HUD:
                #draw_str(res, (20, 80), "latency        :  %.1f ms" % (latency.value*1000))
                #draw_str(res, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value*1000))
                #draw_str(res, (20, 80), "threaded      :  " + str(threaded_mode))

            # Show "PEDAL ACTIVE" near the bottom when the pedal is pressed
            if pedal_pressed:
                draw_str(res, (20, 450), "PEDAL ACTIVE")

            # Push the finished frame to the display window
            cv.imshow("FLUORO", res)

        # ── Capture and submit a new frame ────────────────────────────────────
        # Only submit a new frame if the pool has room (queue shorter than thread count).
        # This prevents unbounded accumulation of pending tasks if processing is slow.
        if len(pending) < threadn:

            # In pedal mode, only grab a frame when the pedal is pressed.
            # When pedal mode is off, capture continuously (peddle_mode == False).
            if pedal_pressed or peddle_mode == False:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to read from camera")

                # Convert the colour frame to grayscale.
                # All subsequent processing is single-channel for simplicity and speed.
                frame_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

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

                if subtract_mode:
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
                    task = pool.apply_async(process_frame, (frame_gray.copy(), t, subtract_mode, overlay_mode, equalize_mode))
                else:
                    # Wrap the synchronous result in DummyTask so the drain loop above
                    # can call .ready() and .get() on it without any special casing.
                    task = DummyTask(process_frame(frame_gray, t, subtract_mode, overlay_mode, equalize_mode))

                pending.append(task)

        # ── Keyboard shortcut handling ─────────────────────────────────────────
        # Second waitKey call at the bottom of the loop catches presses that
        # arrive during the frame processing work above.
        ch = cv.waitKey(1)

        # Space — toggle pedal mode (continuous capture vs. pedal-gated capture)
        # Uncomment the threaded_mode line if you want Space to toggle threading instead:
        #if ch == ord(' '):
        #    threaded_mode = not threaded_mode
        if ch == ord(' '):
            peddle_mode = not peddle_mode

        if ch == 49:   # '1' — toggle background subtraction on/off
            subtract_mode = not subtract_mode

        if ch == 50:   # '2' — toggle anatomy overlay compositing on/off
            overlay_mode = not overlay_mode

        if ch == 51:   # '3' — switch display window to fullscreen
            cv.setWindowProperty("FLUORO", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

        if ch == 52:   # '4' — return display window to normal (windowed) mode
            cv.setWindowProperty("FLUORO", cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)

        if ch == 53:   # '5' — retake the background from the last displayed frame
            # Uses `res` (the most recently dequeued result) as the new background.
            # Converts back to grayscale in case it was colourised by a previous step.
            background = res
            background = cv.cvtColor(background, cv.COLOR_RGB2GRAY)

        if ch == 54:   # '6' — toggle histogram equalisation on/off
            equalize_mode = not equalize_mode

        if ch == 55:   # '7' — toggle the on-screen HUD on/off
            hud_mode = not hud_mode

        if ch == 27:   # ESC — exit the main loop
            break

# Release all OpenCV windows after the loop exits
cv.destroyAllWindows()
