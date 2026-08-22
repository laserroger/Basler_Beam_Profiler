"""Spot detection and rendering.

Candidate blobs are found on a contrast-normalised downscale of the frame
(cheap even for 25 MP sensors), then every spot is refined with a sub-pixel
Gaussian moment fit on a full-resolution crop - batched over all candidates,
see `fit.py`.
"""

from __future__ import annotations

import cv2
import numpy as np

from .fit import fit_spot, fit_spots  # noqa: F401  (fit_spot re-exported)
from .spots import SpotArray

DETECT_MAX_SIDE = 1024  # longest edge used for blob search
MIN_DYNAMIC_RANGE = 0.02  # of full scale; below this the frame is considered empty


def _make_detector() -> cv2.SimpleBlobDetector:
    p = cv2.SimpleBlobDetector_Params()
    # the input is a binary mask, so one threshold pass suffices; the default
    # multi-threshold sweep costs ~200x more on dense spot arrays
    p.minThreshold, p.maxThreshold, p.thresholdStep = 127, 129, 2
    p.minRepeatability = 1
    p.filterByArea, p.minArea, p.maxArea = True, 5, 1e8
    p.filterByCircularity, p.minCircularity = True, 0.4
    p.filterByConvexity, p.minConvexity = True, 0.1
    p.filterByInertia, p.minInertiaRatio = True, 0.01
    p.filterByColor, p.blobColor = True, 255
    return cv2.SimpleBlobDetector_create(p)


_DETECTOR = _make_detector()


def detect_spots(img: np.ndarray, max_side: int = DETECT_MAX_SIDE) -> SpotArray:
    """Detect Gaussian-like spots in a mono frame (uint8 or uint16).

    Returns a SpotArray with columns x, y (ROI pixel coords, sub-pixel),
    sigma_0/sigma_1 (major/minor widths), vec_0/vec_1 (principal axes),
    I0 (peak) and I0_weighted.  Indexing it yields the per-spot dicts."""
    mask, scale = candidate_mask(img, max_side)
    if mask is None:
        return SpotArray.empty()
    keypoints = _DETECTOR.detect(mask)
    if not keypoints:
        return SpotArray.empty()
    pts = np.array([(kp.pt[0], kp.pt[1], kp.size) for kp in keypoints], dtype=np.float64)
    x, y = pts[:, 0] / scale, pts[:, 1] / scale
    radius = np.maximum(pts[:, 2] / (2 * scale), 2.0)
    return fit_spots(img, x, y, radius)


def candidate_mask(img: np.ndarray, max_side: int = DETECT_MAX_SIDE):
    """Binary spot mask on a downscale of the frame, plus the downscale factor.

    Returns (None, scale) when the frame carries no usable contrast."""
    h, w = img.shape
    scale = min(1.0, max_side / max(h, w))
    small = (
        cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else img
    )
    # blur first so the empty-frame check sees noise-suppressed contrast;
    # otherwise the min-max normalisation would stretch pure noise to full scale
    small = cv2.GaussianBlur(small, (5, 5), 0)
    full_scale = 255 if img.dtype == np.uint8 else 65535
    if int(small.max()) - int(small.min()) < MIN_DYNAMIC_RANGE * full_scale:
        return None, scale
    small = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, mask = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, scale


def render_spots(
    img: np.ndarray,
    spots: list[dict],
    pixel_size: float | None = None,
    render_axis: bool = True,
    render_xy: bool = True,
    render_sigma: bool = True,
) -> np.ndarray:
    """Draw fitted ellipses, principal axes and labels onto an RGB image."""
    RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for s in spots:
        x, y = int(round(s["x"])), int(round(s["y"]))
        s0, s1 = s["sigma_0"], s["sigma_1"]
        angle = np.degrees(np.arctan2(s["vec_0"][1], s["vec_0"][0]))
        cv2.ellipse(img, (x, y), (int(round(s0)), int(round(s1))), angle, 0, 360, GREEN, 2)
        if render_axis:
            a = np.radians(angle)
            p_major = (int(round(x + s0 * np.cos(a))), int(round(y + s0 * np.sin(a))))
            p_minor = (
                int(round(x + s1 * np.cos(a + np.pi / 2))),
                int(round(y + s1 * np.sin(a + np.pi / 2))),
            )
            cv2.line(img, (x, y), p_major, RED, 2)
            cv2.line(img, (x, y), p_minor, BLUE, 2)
            cv2.putText(img, "0", p_major, font, 0.7, RED, 1)
            cv2.putText(img, "1", p_minor, font, 0.7, BLUE, 1)
        dy = 20
        ty = y + int(np.sqrt(s0 * s1) + 10 + dy)
        if render_xy:
            cv2.putText(img, f"({x}, {y})", (x, ty), font, 0.7, GREEN, 1)
        if render_sigma and pixel_size is not None:
            um = pixel_size * 1e6
            cv2.putText(img, f"s0={s0 * um:.1f} um", (x, ty + dy), font, 0.7, GREEN, 1)
            cv2.putText(img, f"s1={s1 * um:.1f} um", (x, ty + 2 * dy), font, 0.7, GREEN, 1)
    return img
