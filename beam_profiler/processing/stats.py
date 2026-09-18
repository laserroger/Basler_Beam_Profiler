"""Pixel statistics of a sensor-coordinate rectangle within a grabbed frame."""

from __future__ import annotations

import numpy as np


def crop_rect(frame, roi, rect_sensor):
    """Crop `rect_sensor` (x1, y1, x2, y2 in sensor coords) out of a frame that
    was grabbed with `roi` = (w, h, ox, oy).  Returns (region, roi_coords) or
    None if the rectangle misses the ROI."""
    if frame is None or rect_sensor is None:
        return None
    w, h, ox, oy = roi
    sx1, sy1, sx2, sy2 = rect_sensor
    x1, y1 = max(0, int(sx1 - ox)), max(0, int(sy1 - oy))
    x2, y2 = min(w, int(sx2 - ox)), min(h, int(sy2 - oy))
    if x1 >= x2 or y1 >= y2:
        return None
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def region_stats(frame, roi, rect_sensor, pixel_size: float) -> dict | None:
    cropped = crop_rect(frame, roi, rect_sensor)
    if cropped is None:
        return None
    region, (x1, y1, x2, y2) = cropped
    um = pixel_size * 1e6
    return {
        "sum": int(region.sum()),
        "mean": float(region.mean()),
        "std": float(region.std()),
        "min": int(region.min()),
        "max": int(region.max()),
        "median": float(np.median(region)),
        "total_pixels": int(region.size),
        "width_pixels": x2 - x1,
        "height_pixels": y2 - y1,
        "width_um": float((x2 - x1) * um),
        "height_um": float((y2 - y1) * um),
        "sensor_coords": [int(c) for c in rect_sensor],
        "roi_coords": [x1, y1, x2, y2],
    }
