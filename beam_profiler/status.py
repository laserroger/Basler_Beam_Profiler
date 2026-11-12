"""Mirror the live viewer state to pylon_camera.json so other processes can
read it without starting the HTTP server."""

from __future__ import annotations

import json
import logging

from .config import STATUS_JSON_PATH


def viewer_status(viewer) -> dict:
    return {
        "camera_model": viewer.camera.model_name,
        "camera_serial": viewer.camera.serial,
        "exposure_time": viewer.camera.ExposureTime,
        "gain_db": viewer.camera.Gain,
        "pixel_format": 'Mono16' if viewer.camera.mode == '16Bit' else 'Mono8',
        "adc_bit_depth": getattr(viewer.camera, 'adc_bit_depth', None),
        "calibration": viewer.calibration_status(),
        "roi": viewer.camera.ROI,
        "auto_exposure": viewer.auto_exp,
        "fitting_enabled": viewer.do_fitting,
        "profiler_enabled": getattr(viewer, 'profiler_enabled', False),
        "rect_sensor": viewer.rect_sensor,
        "fit_rect_sensor": viewer.fit_rect_sensor,
    }


def write_status(viewer):
    try:
        with open(STATUS_JSON_PATH, "w") as f:
            json.dump(viewer_status(viewer), f, indent=4)
    except Exception as e:
        logging.error(f"Error saving status to {STATUS_JSON_PATH}: {e}")
