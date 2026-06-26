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
- An overlay image named **`skel.jpg`** placed in the same directory as the script.

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
| `5`       | Retake the background image used in subtraction     |
| `6`       | Toggle histogram equalization                       |
| `7`       | Toggle the HUD (on-screen text display)             |
| `b`       | Acts as the pedal press (keyboard stand-in)         |

> When **pedal mode** is on, frames are only captured while the pedal is held (or the `b` key is pressed). When it's off, the simulator captures continuously.

## Notes

- Tuning knobs live near the top of the main loop: `mask_threshold` (background mask cutoff) and `alpha` (running-background accumulation weight).
- The `FLUORO` window opens full-screen by default; use `4` to drop into a normal window.
- This is a demonstration/training tool and is **not** a medical device.
