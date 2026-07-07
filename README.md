# FluoroSim

A real-time **fluoroscopy simulator** built on OpenCV. It takes a live camera feed and processes it to mimic the look of an X-ray fluoroscopy image — inverted grayscale with background subtraction, an anatomical overlay, and an optional foot-pedal trigger — for training and demonstration purposes.

Originally presented at **SIR 2018** (Society of Interventional Radiology).

Based on the OpenCV [`video_threaded.py`](https://github.com/opencv/opencv/tree/master/samples/python) multithreaded video-processing sample.

## What it does

The simulator captures frames from a webcam and, in real time:

- **Converts to grayscale** and applies an inverted background subtraction (`absdiff` + `bitwise_not`) so that bright/empty areas read as the dark "fluoro" background and objects appear as the X-ray would render them.
- **Blends an anatomical overlay** image (`skel.jpg`) into the masked background region, simulating skeletal/anatomy underneath the live scene.
- **Equalizes the histogram** to boost contrast.
- **Gates capture on a foot pedal** (USB foot switch) so frames are only processed while the pedal is held — mimicking a fluoroscopy unit that only images while the pedal is pressed.
- **Runs frame processing across a thread pool** (one task per CPU) to keep the display responsive.
- Displays everything full-screen in a window named `FLUORO`, with an optional on-screen HUD listing the active modes and key bindings.

## Requirements

- **Python 3** (the code keeps Python 2/3 compatibility imports but targets Python 3)
- [OpenCV](https://pypi.org/project/opencv-python/) (`cv2`)
- [NumPy](https://pypi.org/project/numpy/)
- A **webcam / video capture device**
- **Linux** is assumed for full functionality:
  - Camera capture uses the V4L2 backend (`cv.CAP_V4L2`).
  - The foot pedal path is a Linux input device (`/dev/input/by-id/usb-PCsensor-FootSwitch-event-kbd`).
- An overlay image named **`skel.jpg`** placed in the same directory as the script. This file is included in the repository root (`fluoro_simulator/skel.jpg`) — an AP lumbar spine X-ray used as the anatomical background overlay.

Install dependencies:

```bash
pip install opencv-python numpy
```

## Usage

```bash
python "fluoro_simulator (3).py" [<video device number>]
```

- `<video device number>` is optional and defaults to `0` (the first camera). Pass `1`, `2`, etc. to select a different device.

Make sure `skel.jpg` is in the same folder as the script before running, or it will raise a `FileNotFoundError`.

## Keyboard shortcuts

| Key       | Action                                              |
|-----------|-----------------------------------------------------|
| `ESC`     | Exit                                                |
| `Space`   | Toggle pedal mode (require pedal to capture frames) |
| `1`       | Toggle background subtraction                       |
| `2`       | Toggle anatomy overlay                              |
| `3`       | Fullscreen                                          |
| `4`       | Windowed mode                                       |
| `5`       | Toggle the anatomy overlay (off = full raw video)   |
| `6`       | Toggle histogram equalization                       |
| `7`       | Toggle the HUD (on-screen text display)             |
| `b`       | Acts as the pedal press (keyboard stand-in)         |

> When **pedal mode** is on, frames are only captured while the pedal is held (or the `b` key is pressed). When it's off, the simulator captures continuously.
>
> Keys `2` and `5` both toggle the overlay. With the overlay **off**, the simulator shows the full raw video with the bright/white areas intact (nothing removed).

## Web control panel (`fluoro_web.py`)

`fluoro_web.py` runs the same simulation (the fullscreen `FLUORO` window and the keyboard shortcuts above) **and** serves a small web page so the simulation can be controlled from a phone, tablet, or any browser on the same network — handy for operating the demo from a tablet while the monitor shows the fluoro view.

### What it provides

- A **live preview** (MJPEG stream) of the processed feed.
- On/off **toggle buttons** mirroring the keyboard shortcuts: Subtraction, Overlay, Equalize, Pedal mode, Pedal press, and HUD. Toggles stay in sync whether you use the web buttons or the keyboard.
- **Fullscreen / Windowed** buttons that control both the `FLUORO` popup window on the computer **and** the live preview in your browser, plus a **Quit** button.

### Requirements

In addition to the base requirements, install Flask:

```bash
pip install flask
```

### Usage

```bash
python fluoro_web.py [<video device number>] [--port 5000] [--no-window] [--http]
```

- `<video device number>` — camera index (default `0`).
- `--port` — web server port (default `5000`).
- `--no-window` — run web-only, without the on-screen `FLUORO` window.
- `--http` — force plain HTTP even if a TLS cert is present.

Then open `https://<this-machine-ip>:<port>/` in a browser (or `http://…` if running without a cert).

### HTTPS (self-signed cert)

Some browsers force `https://`. If `cert.pem` and `key.pem` are present next to the script, the panel is served over HTTPS automatically. Generate a self-signed cert (valid ~2 years) with:

```bash
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem \
    -days 825 -subj "/CN=FluoroSim" \
    -addext "subjectAltName=IP:<your-lan-ip>,DNS:localhost,IP:127.0.0.1"
```

The browser will show a one-time "not private" warning for the self-signed cert — click through to proceed. `cert.pem` and `key.pem` are git-ignored, so the private key is never committed; each machine generates its own.

> **Fullscreen note:** the panel forces OpenCV's X11/XWayland Qt backend (`QT_QPA_PLATFORM=xcb`) so the Fullscreen/Windowed controls can actually toggle the `FLUORO` window — the native Wayland backend ignores those calls.

## Running as a service (auto-start, crash recovery)

`fluorosim.service` is a systemd **user** unit that runs `launch_fluoro.sh` under
supervision: the simulator starts automatically at boot (once the desktop session
is up) and restarts itself within a few seconds if it crashes. A deliberate quit
(the web panel's **Quit** button or `ESC` on the `FLUORO` window) stays stopped —
only crashes trigger a restart.

Install on a new machine:

```bash
mkdir -p ~/.config/systemd/user
cp fluorosim.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now fluorosim.service
```

Manage it with `systemctl --user {start,stop,restart,status} fluorosim`. The
app's output goes to `launch.log` in this directory (via `launch_fluoro.sh`),
not the journal. Auto-start at boot assumes the machine logs the user into the
desktop automatically (standard Raspberry Pi OS autologin).

The `FluoroSim.desktop` shortcut runs `systemctl --user restart fluorosim.service`,
so the desktop button (re)launches the supervised service in any state — stopped,
running, or wedged.

## Notes

- Tuning knobs live near the top of the processing code: the background mask cutoff (`MASK_THRESHOLD` / `mask_threshold`), the running-background accumulation weight (`ALPHA` / `alpha`), and the overlay blend weights (bright areas 30% video / 70% overlay; vasculature 60% video / 40% overlay).
- The `FLUORO` window opens full-screen by default; use `4` to drop into a normal window.
- This is a demonstration/training tool and is **not** a medical device.
