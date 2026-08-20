# Basler Beam Profiler

The Basler Beam Profiler is a Python application designed to interface with Basler cameras for beam profiling tasks.

![Demo](docs/demo.png)


## Usage
### Running the Application
```bash
pip install -r requirements.txt
python pylon_camera.py
```


### Configuring the camera

`camera_config.yaml`

```yaml
cameras:
  a2A5060-15umBAS:                     # the name should match the camera name shown in pylon_camera
    default_roi: [5060, 5060, 4, 4]    # [width, height, x_pad, y_pad]
    pixel_size: 2.5e-6                 # pixel size in meters, i.e. 2.5um
```
### Using the compiled application
First clone this repo, setup `camera_config.yaml`

Check release to download the latest version: [Releases](https://github.com/tim4431/Basler_Beam_Profiler/releases)

Put the `pylon_camera.exe` executable in the same directory as `camera_config.yaml`.
Saved frames (`data/`) and the live status file (`pylon_camera.json`) are written
next to the executable.

### Keyboard Shortcuts
- `Esc`: Quit the application
- `arrow keys`: up/down=change exposure by 10 times, left/right=change exposure by 10%
- `a`: toggle auto-exposure (auto exposure adjusts the exposure time to keep the max intensity to be near-saturated)
- `f`: toggle blob fitting
- `g`: toggle statistics showing (statistics of fitted blobs)
- `h`: toggle row/col statistics (works only for beam arrays)
- `mouse drag`: create a white rectangle in the canvas
- `c`: clear the white rectangle
- `ctrl + mouse drag`: create a green rectangle in the canvas, only blobs inside the rectangle will be fitted
- `v`: clear the green rectangle
- `s`: quick save the current frame
- `d`: save the current frame with dialogue box, allowing the user to choose the file location and name
- `t`: switch to the next camera (if multiple cameras are connected)
- `w`: toggle the HTTP server (port 5000), together with the live pixel statistics of the white rectangle drawn on the canvas
- `y`: toggle the user-defined line output (Line3) used to sync external hardware
- `mouse wheel`: zoom in/out the canvas

## Remote control / readout

Pressing `w` starts a small HTTP server on port 5000 that exposes the camera
status, the current frame, and the pixel statistics of the white / green
rectangles. Open <http://localhost:5000> for the endpoint list, or see
[docs/web_api.md](docs/web_api.md) for the full reference.

The current state (camera, exposure, ROI, rectangles) is also mirrored to
`pylon_camera.json` whenever it changes, so another process can read it without
starting the server.

## Hardware sync

The camera is armed with a software trigger, and drives two GPIO lines:

- **Line2** - inverted `ExposureActive`, i.e. it mirrors the exposure window.
- **Line3** - user-controlled output, toggled with `y`. While enabled it is
  pulsed low for the duration of each acquisition.

Cameras that do not expose these lines (or the software trigger) log a warning at
startup and keep running with that feature disabled.
