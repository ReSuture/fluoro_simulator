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

- `<video device number>` — camera index (default `0`), used only as a fallback
  when no camera can be discovered (see **Two cameras** below).
- `--port` — web server port (default `5000`).
- `--no-window` — run web-only, without the on-screen `FLUORO` window.
- `--http` — force plain HTTP even if a TLS cert is present.
- `--list-cameras` — print the attached cameras, their USB ports and which view
  each one drives, then exit.
- `--main <port>` / `--lateral <port>` — bind a view to a USB port and save it.

Then open `https://<this-machine-ip>:<port>/` in a browser (or `http://…` if running without a cert).

### Two cameras: main and lateral view

The simulator drives two views, mirroring a C-arm's frontal and lateral
projections. With two cameras attached, a **Lateral view** button appears on the
web panel, the on-screen control bar and the `CONTROLS` panel; it is dimmed and
unpressable whenever fewer than two are attached. Selecting it releases the main
camera and opens the lateral one — two USB cameras rarely have the bandwidth to
stream at once, so the views take turns. The last frame stays up during the
switch, and the image is marked `LATERAL` in the black rim outside the aperture
so the two projections can never be confused.

Which camera plays which role is decided by the **USB port it is plugged into**,
not by its `/dev/videoN` number — Linux hands those out in probe order, so they
shuffle between boots and replugs, while the port is a property of the bench.
Ports come from V4L2's `bus_info`. To see what is attached:

```bash
python fluoro_web.py --list-cameras
```

```
Cameras attached: 2
  main     usb-3f980000.usb-1.4           /dev/video0   HD Pro Webcam C920
  lateral  usb-3f980000.usb-1.2           /dev/video2   HD Pro Webcam C920
```

With nothing saved, the roles follow port order: first port = main, second =
lateral. To pin them to specific ports (saved to
`~/.config/fluorosim/cameras.json`, so it survives replugs and reboots):

```bash
python fluoro_web.py --main usb-3f980000.usb-1.2 --lateral usb-3f980000.usb-1.4
```

Cameras are re-inventoried every few seconds (`CAMERA_SCAN_SEC`), so plugging a
second camera in offers the lateral view without a restart, and unplugging it
returns to the main view on its own. A single camera always drives the main
view, even if it sits in the port assigned to the lateral role.

> **Note:** both views currently composite against the same frontal anatomy
> master. A true lateral projection needs its own master image — that is a
> separate change.

### No camera attached

A missing camera never stops the app. The simulator starts as usual — web panel,
`FLUORO` window, tab bar, Remote Access and Library all work — and shows a black
**NO CAMERA CONNECTED** screen where the video would be; the web panel adds a
banner under the preview. The camera index is retried every few seconds
(`CAMERA_RETRY_SEC`), so plugging a camera in picks it up automatically with no
restart, and a camera that is unplugged while running drops back to the same
placeholder instead of freezing. With the lateral view selected the screen reads
**NO LATERAL CAMERA CONNECTED** instead. Recording is unavailable without a
camera: an in-progress recording is closed cleanly and the Record toggle
switches itself off (switching views mid-recording does not end it — the
recording continues, scaled to the size the file was opened at).

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

`pi_setup/fluorosim.service` is a systemd **user** unit that runs `launch_fluoro.sh`
under supervision: the simulator starts automatically at boot (once the desktop
session is up) and restarts itself within a few seconds if it crashes. A deliberate
quit (the web panel's **Quit** button or `ESC` on the `FLUORO` window) stays stopped —
only crashes trigger a restart.

Install on a new machine:

```bash
mkdir -p ~/.config/systemd/user
cp pi_setup/fluorosim.service ~/.config/systemd/user/
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

## Remote access: customer portal + Cloudflare Tunnel

Customer devices are reachable from anywhere through a per-device Cloudflare
Tunnel and a sign-in portal:

- **`portal/`** — the multi-tenant portal (Flask). Customers sign in through
  Cloudflare Access and open their own Pi's panel at
  `https://sim-<device-id>.<device-domain>/`. Admins assign devices to
  customer emails at `/admin`. Deployment runbook: `portal/README.md`.
- **`pi_setup/`** — bench provisioning for new Pis (`provision_pi.py`) and the
  canonical `fluorosim.service`. Manufacturing checklist: `pi_setup/README.md`.

On tunneled customer devices, `launch_fluoro.sh` starts the panel with
`--host 127.0.0.1 --http`: port 5000 is unreachable from the LAN, the local
`cloudflared` is the only way in, and Cloudflare Access enforces the sign-in
at the edge (the panel itself has no auth). For a LAN-only demo machine, run
`python3 fluoro_web.py` directly — it still binds `0.0.0.0:5000` by default.

## Notes

- Tuning knobs live near the top of the processing code: the background mask cutoff (`MASK_THRESHOLD` / `mask_threshold`), the running-background accumulation weight (`ALPHA` / `alpha`), and the overlay blend weights (bright areas 30% video / 70% overlay; vasculature 60% video / 40% overlay).
- The `FLUORO` window opens full-screen by default; use `4` to drop into a normal window.
- This is a demonstration/training tool and is **not** a medical device.
