# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup on a new machine

```bash
pip install -r requirements.txt        # needs Python 3.10+; pyinstaller only needed for exe builds
python -m beam_profiler --sim          # verify everything works without hardware
```

For real cameras, the Basler pylon SDK must be installed system-wide and the
camera model must have an entry in `camera_config.yaml` (keyed by exact model
name, e.g. `a2A5060-15umBAS`). With no camera connected, the app automatically
falls back to the simulator. `run.sh` contains a machine-specific interpreter
path — adjust or ignore it on other PCs.

## Test signal / debugging without hardware

- `python -m beam_profiler --sim` — full GUI on a simulated camera: a fixed
  grid of Gaussian spots with noise/jitter. Options: `--sim-grid 80x80`,
  `--sim-size 4096x4096`, `--sim-jitter 0` (static spots), `--sim-sigma <px>`
  (default pitch/10). Auto-exposure, ROI, fitting and the HTTP API all work
  against it.
- `python tools/make_test_images.py` — writes fixed spot pictures with
  ground-truth JSON to `test_images/`.
- `beam_profiler/synthetic.py` renders ground-truth frames programmatically —
  use it to benchmark or debug CV changes headlessly (see tests for examples).
- The HTTP API (press `w` in the GUI, port 5000) and the `pylon_camera.json`
  status mirror are the hooks for scripting against a running instance.

## Commands

```bash
python -m pytest tests/                              # run tests
python -m pytest tests/test_processing.py -k grid    # single test
python -m beam_profiler                              # run (hardware, sim fallback)
pyinstaller --onefile pylon_camera.py                # Windows exe (see `compile`)
```

Work happens on the `dev` branch. CI (`.github/workflows/build.yml`) runs only
on push to `main` and every main push builds a Windows exe **and publishes a
GitHub release** — don't merge to main casually.

## Architecture

All code lives in the `beam_profiler/` package; `pylon_camera.py` is only a
compatibility shim for run.sh/PyInstaller.

- `cameras/` — `base.Camera` defines the interface (grab_image, ROI,
  ExposureTime, Gain properties) plus shared PID auto-exposure;
  `basler.py` (pypylon) and `simulated.py` implement it. `open_cameras()`
  does discovery + sim fallback. Anything testable without hardware must not
  import pypylon at module level.
- `processing/` — pure CV, no camera/UI dependencies: `blobs.py` (detection:
  blob search on a ≤1024 px downscale, then sub-pixel Gaussian *moment* fits
  on full-resolution crops with local background subtraction), `grid.py`
  (row/column classification), `stats.py` (rectangle pixel stats — the single
  source used by both UI and HTTP server).
- `ui/viewer.py` — single-threaded main loop (grab → process → draw → keys);
  owns all mutable state, which `server.py` (Flask, daemon thread) and
  `status.py` read. `ui/overlays.py` draws HUD/stats/spots.

### Conventions that bite

- **Three coordinate systems**: *sensor* coords (full chip; rectangles are
  stored in these), *ROI* coords (grabbed frame; spot fits are in these;
  sensor = ROI + offset), and *display* coords (1100 px window;
  `roi.ViewTransform` maps between them). All overlay drawing happens in
  display space so text/lines don't scale with zoom.
- Reported `sigma_0/sigma_1` are **2×** the Gaussian σ (≈ beam waist per
  axis); tests encode this (`sigma_0 ≈ 2 * true_sigma`).
- "16Bit" mode is Mono12 wrapped MSB-aligned in uint16 (saturation 65535).
- Keyboard codes in `viewer.KEY_FACTORS` must cover both Windows and Linux
  `waitKeyEx` values.
- CV changes must keep `tests/` passing — they check detection count,
  sub-pixel position error, widths, orientation and grid classification
  against synthetic ground truth.
