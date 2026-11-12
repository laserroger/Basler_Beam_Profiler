"""Entry point: python -m beam_profiler [--sim] [options]."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="beam_profiler", description="Basler / FLIR Beam Profiler"
    )
    parser.add_argument("--camera", choices=["auto", "flir", "basler", "sim"],
                        default="auto", help="camera driver; explicit hardware selection never simulates")
    parser.add_argument(
        "--sim", action="store_true",
        help="use the simulated camera (fixed spot grid) instead of hardware",
    )
    parser.add_argument(
        "--mode", choices=["8Bit", "16Bit"], default="16Bit", help="pixel depth"
    )
    parser.add_argument(
        "--sim-grid", metavar="NxM", default="4x4",
        help="simulated spot grid, e.g. 4x4 (default) or 8x8",
    )
    parser.add_argument(
        "--sim-size", metavar="WxH", default="1200x1200", help="simulated sensor size"
    )
    parser.add_argument(
        "--sim-jitter", type=float, default=0.3,
        help="per-frame spot position jitter in px (0 = perfectly static)",
    )
    parser.add_argument(
        "--sim-sigma", type=float, default=None,
        help="spot sigma in px (default: pitch/10, i.e. waist = pitch/5)",
    )
    parser.add_argument("--frames", type=int, default=None,
                        help="exit after this many frames (for package smoke checks)")
    parser.add_argument("--check-drivers", action="store_true",
                        help="verify bundled Basler and FLIR libraries and exit")
    parser.add_argument('--calibration', metavar='PROFILE',
                        help='flat-field profile JSON or a calibration output directory')
    parser.add_argument('--exposure-us', type=float, help='explicit initial exposure in microseconds')
    parser.add_argument('--save-dir', help='directory for saved raw frames, display JPEGs and settings')
    parser.add_argument('--adc-bits', type=int, choices=[8, 10, 12], help='explicit FLIR ADC digitization depth')
    parser.add_argument('--web', action='store_true', help='start the HTTP API with the viewer')
    parser.add_argument('--spot-sizes', action='store_true', help='start fitting, live D4sigma diameter plots, and HTTP server')
    args = parser.parse_args(argv)
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be positive")
    if args.exposure_us is not None and (not math.isfinite(args.exposure_us) or args.exposure_us <= 0):
        parser.error('--exposure-us must be positive and finite')
    if args.save_dir:
        from . import config
        config.DATA_DIR = os.path.abspath(os.path.expanduser(args.save_dir))
        os.makedirs(config.DATA_DIR, exist_ok=True)
    from .calibration import FlatField
    calibration = FlatField.load(args.calibration) if args.calibration else None

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.check_drivers:
        import PySpin
        from pypylon import pylon
        system = PySpin.System.GetInstance()
        try:
            version = system.GetLibraryVersion()
            print(f"FLIR runtime: {version.major}.{version.minor}.{version.type}.{version.build}")
            print(f"Basler runtime: {pylon.GetPylonVersionString()}")
        finally:
            system.ReleaseInstance()
        return

    from .cameras import open_cameras

    grid = tuple(int(v) for v in args.sim_grid.lower().split("x"))
    width, height = (int(v) for v in args.sim_size.lower().split("x"))
    sim_opts = dict(
        grid=grid, width=width, height=height,
        jitter=args.sim_jitter, sigma=args.sim_sigma,
    )
    try:
        cameras = open_cameras(mode=args.mode, simulate=args.sim, backend=args.camera, adc_bits=args.adc_bits, **sim_opts)
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")

    from .ui import Viewer

    viewer = None
    try:
        if calibration is not None and calibration.metadata.get('camera_serial'):
            serial = str(calibration.metadata['camera_serial'])
            cameras.sort(key=lambda camera: str(camera.serial) != serial)
        if args.exposure_us is not None:
            cameras[0].ExposureTime = args.exposure_us
        if calibration is not None:
            reason = calibration.mismatch(cameras[0])
            if reason:
                raise ValueError('Cannot apply calibration: ' + reason)
            calibration.check_adc(cameras[0])
            logging.info('Loaded calibration %s', calibration.path)
        viewer = Viewer(cameras, calibration=calibration)
        viewer.exposure_us = cameras[0].ExposureTime
        if args.spot_sizes:
            viewer._toggle_sizes()
        if args.web or args.spot_sizes:
            viewer._toggle_web_server()
        viewer.run(max_frames=args.frames)
    finally:
        close_cameras(cameras, failed=viewer is not None and viewer.camera_failed)


def exit_failed_camera_process():
    """Exit the CLI/app without re-entering a broken vendor SDK destructor.

    Observed on FLIR unplug: CameraPtr destruction waits forever in Spinnaker's
    EventProcessor thread join, even after EndAcquisition raised -1014. Python
    sys.exit/exception unwinding would run that destructor again. This is only
    the fatal camera path, after the viewer has attempted GUI cleanup. The OS
    reclaims process resources; normal exits still explicitly close cameras.
    """
    logging.error('Camera connection failed; terminating the camera process.')
    try:
        for stream in (sys.stdout, sys.stderr):
            if stream is not None:
                stream.flush()
    finally:
        os._exit(1)


def close_cameras(cameras, *, failed=False):
    if failed:
        exit_failed_camera_process()
        return
    for camera in cameras:
        try:
            camera.close()
        except Exception:
            # Exit inside this handler, before disposing of the traceback's
            # SWIG CameraPtr references can invoke the stuck C++ destructor.
            logging.exception('Camera shutdown failed')
            exit_failed_camera_process()
            return


if __name__ == "__main__":
    main()
