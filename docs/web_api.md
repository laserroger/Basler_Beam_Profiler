# HTTP API

Press `w` in the profiler window to start the server on port `5000`
(`http://localhost:5000` serves an index page listing the endpoints). Pressing
`w` again hides the on-screen rectangle statistics; the Flask thread is a daemon
and shuts down with the application.

All rectangle coordinates are **sensor** coordinates `[x1, y1, x2, y2]`, not
display coordinates. The white rectangle (`rect`) and the green fitting
rectangle (`fit_rect`) are handled by separate endpoints.

Each request reads one complete frame snapshot. Image geometry, fitting results
and statistics cannot come from different frames within that response.
The `timestamp` in measurement responses identifies the processing time of the
published frame, rather than the HTTP request time.

Rectangle and fitting-setting writes are queued for the acquisition thread and
applied before its next frame. After a write, wait for a new measurement timestamp
before reading its effects. Settings PUT returns the accepted configuration;
settings GET returns the currently applied configuration. These endpoints do
not access native windows or camera drivers from the HTTP thread.

`/api/spots` also includes a `saturated` Boolean array, aligned with `spots`, and
`profiler_message`. Saturated fits are unreliable; reduce exposure before using
their sizes for optimization. W can hide rectangle statistics without stopping
HTTP updates.

`/api/spots` includes `stats.grid_shear`, calculated from that frame even when
the H overlay is hidden. When `valid` is true it contains `shear_deg` (signed
`90 - angle_deg`, zero is the target), `angle_deg` (between rightward row and
downward column directions), `rotation_deg` (best-fit orthogonal grid rotation,
clockwise in image coordinates), `orthogonal_rms_px`, `line_rms_px` (remaining
perpendicular line residual with shear allowed), `spot_count`, `num_rows`,
`num_columns`, and unit `row_direction`/`column_direction` vectors. Both fits
allow independent line offsets, so nonuniform spacing is not counted as shear.
Saturated, degenerate, or insufficiently classified grids return `valid: false`
and a `reason` instead of numerical fit values. The existing classifier expects
approximately horizontal rows and vertical columns; large rotations may prevent
correct grouping. H toggles the desktop grid and shear display together; G is
not required for the shear readout.

## Read

| Endpoint | Description |
| --- | --- |
| `GET /api/status` | Camera model/serial, exposure, ROI, auto-exposure and fitting flags, both rectangles |
| `GET /api/rect_stats` | Pixel statistics inside the white rectangle |
| `GET /api/fit_rect_stats` | Pixel statistics inside the green fitting rectangle |
| `GET /api/spots` | Detected spots plus the aggregated row/column statistics |
| `GET /api/fit_config` | Current spot-fitting parameters, plus each setting's range and help text |
| `PUT /api/fit_config` | Update spot-fitting parameters; JSON body of `{name: value}`, unknown names are rejected |
| `GET /api/image` | Current frame as JPEG |
| `GET /api/image_within_rect` | White-rectangle crop of the current frame as JPEG |

## Write

| Endpoint | Body | Description |
| --- | --- | --- |
| `POST /api/set_rect` | `{"coords": [x1, y1, x2, y2]}` | Set the white rectangle |
| `POST /api/set_fit_rect` | `{"coords": [x1, y1, x2, y2]}` | Set the green fitting rectangle |
| `POST /api/clear_rect` | - | Clear the white rectangle |
| `POST /api/clear_fit_rect` | - | Clear the green fitting rectangle |

## Examples

```bash
# camera state
curl http://localhost:5000/api/status

# define the region of interest to integrate over
curl -X POST -H "Content-Type: application/json" \
     -d '{"coords": [100, 100, 300, 300]}' \
     http://localhost:5000/api/set_rect

# read back its pixel statistics
curl http://localhost:5000/api/rect_stats

# grab the current frame
curl http://localhost:5000/api/image -o current_image.jpg
```

`/api/status` responds with:

```json
{
  "camera_model": "a2A5060-15umBAS",
  "camera_serial": "40615333",
  "exposure_time": 2866.0,
  "roi": [252, 252, 2502, 1774],
  "auto_exposure": false,
  "fitting_enabled": false,
  "rect_sensor": [2476, 1740, 2837, 2099],
  "fit_rect_sensor": null
}
```

`/api/rect_stats` responds with:

```json
{
  "rectangle_bounds": {
    "sensor_coords": [100, 100, 300, 300],
    "roi_coords": [100, 100, 300, 300],
    "width_pixels": 200,
    "height_pixels": 200,
    "width_um": 500.0,
    "height_um": 500.0
  },
  "pixel_stats": {
    "sum": 2560000,
    "mean": 64.0,
    "std": 12.5,
    "min": 32,
    "max": 255,
    "median": 63.0,
    "total_pixels": 40000
  },
  "timestamp": 1692960000.0
}
```

Statistics are computed on the **raw** frame (12-bit data in a 16-bit array when
the camera runs in `16Bit` mode), while `/api/image` returns an 8-bit JPEG
rendering of it.

## Notes

- Errors are returned as `{"error": "..."}` with HTTP 400 - e.g. when no frame
  has been grabbed yet, or when no rectangle has been defined.
- The server binds `0.0.0.0`, so it is reachable from other machines on the
  network. There is no authentication; keep it on a trusted network.
- The same status payload as `/api/status` is written to `pylon_camera.json`
  next to the application on every change, and does not require the server.
