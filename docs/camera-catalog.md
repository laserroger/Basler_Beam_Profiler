# Camera calibration catalog

`camera_config.yaml` supplies the model name, sensor rectangle and physical pixel
pitch used to convert fitted beam widths from pixels to meters. The catalog was
checked against manufacturer specifications on 2026-09-22. Each configuration
includes its `source` URL; the app does not need internet access to use it.
The snapshot contains **175 FLIR models and 414 Basler models**, with **888 lookup
names** including aliases.

FLIR coverage follows the [official online camera references](https://www.teledynevisionsolutions.com/support/support-center/technical-guidance/iis/online-camera-references/):
Blackfly S USB3, GigE and board-level, Dragonfly S, Firefly, Forge and Oryx.
Both monochrome and color model metadata are included. Where documented, short
model names, family-prefixed names and lens-mount variants share one configuration.
Legacy FlyCapture-only cameras and thermal camera families are outside this catalog.

Basler coverage follows the area-camera specification pages in the
[official documentation](https://docs.baslerweb.com/): ace, ace 2, boost, dart and
pulse (including dart M). Evaluation-kit pages are represented by their camera
model rather than the kit part number. Line-scan and 3D cameras are outside the 2D beam profiler's scope.

These entries describe sensors, not additional drivers. The connected camera must
still support the app's Pylon or Spinnaker backend and monochrome acquisition.
A color or polarization sensor's optical response also differs from a monochrome
sensor even when its pixel pitch is the same.

## Values and overrides

- `default_roi`: `[width, height, offset_x, offset_y]`, in sensor pixels. New
  catalog entries use the documented full resolution with zero offsets.
- `pixel_size`: square pixel pitch in meters, before any external magnification.
- `source`: the manufacturer's specification used for these values.
- `sync`: optional, model-specific GPIO assignments. The existing BFS-U3-31S4M
  mapping is retained; no output wiring is inferred for other cameras.

For example, `Blackfly S BFS-U3-122S6M`, `BFS-U3-122S6M` and `BFS-U3-122S6M-C`
use 4096 × 3000 pixels at 3.45 µm, a 14.1312 × 10.35 mm sensor.

An installed app loads its bundled catalog and then applies the writable user
configuration by model name. This makes new models available after an upgrade
without overwriting saved calibration values or custom ROIs. The writable file is
in `~/Library/Application Support/BeamProfiler` on macOS and
`%LOCALAPPDATA%\BeamProfiler` on Windows. Running from source uses the repository's
YAML directly.

To update an entry, verify both the exact model name and the resolution/pixel-pitch
rows at its source URL. Keep aliases consistent and run
`python -m pytest tests/test_camera_catalog.py tests/test_packaged_config.py -q`.
