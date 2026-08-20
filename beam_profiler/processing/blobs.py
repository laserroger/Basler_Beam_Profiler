"""Spot detection and rendering.

Candidate blobs are found on a contrast-normalised downscale of the frame
(cheap even for 25 MP sensors), then each spot is refined with a sub-pixel
Gaussian moment fit on a full-resolution crop.
"""

from __future__ import annotations

import cv2
import numpy as np

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


def detect_spots(img: np.ndarray, max_side: int = DETECT_MAX_SIDE) -> list[dict]:
    """Detect Gaussian-like spots in a mono frame (uint8 or uint16).

    Returns a list of dicts with keys x, y (ROI pixel coords, sub-pixel),
    sigma_0/sigma_1 (major/minor widths), sigma, vec_0/vec_1 (principal axes),
    I0 (peak) and I0_weighted."""
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
        return []
    small = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, mask = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    spots = []
    for kp in _DETECTOR.detect(mask):
        x, y = kp.pt[0] / scale, kp.pt[1] / scale
        radius = max(kp.size / (2 * scale), 2.0)
        spot = fit_spot(img, x, y, radius)
        if spot is not None:
            spots.append(spot)
    return spots


def fit_spot(
    img: np.ndarray, x: float, y: float, radius: float, window: float = 2.0
) -> dict | None:
    """Sub-pixel Gaussian moment fit on a crop around (x, y)."""
    h, w = img.shape
    x0, x1 = int(max(0, x - window * radius)), int(min(w, x + window * radius))
    y0, y1 = int(max(0, y - window * radius)), int(min(h, y + window * radius))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None

    # subtract the local background (median of the crop border) so the constant
    # offset does not inflate the second moments
    crop = img[y0:y1, x0:x1].astype(np.float32)
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    crop -= np.median(border)
    np.clip(crop, 0, None, out=crop)

    # coarse-grain large crops so the moment sums stay O(1) regardless of spot size
    N = 40
    if max(crop.shape) > 2 * N:
        Z = cv2.resize(crop, (N, N), interpolation=cv2.INTER_AREA)
        xs = np.linspace(x0, x1, N)
        ys = np.linspace(y0, y1, N)
    else:
        Z = crop
        xs = np.arange(x0, x1, dtype=np.float64)
        ys = np.arange(y0, y1, dtype=np.float64)

    # moments from the marginals (identical maths to a full 2D sum, ~3x faster)
    col = Z.sum(axis=0).astype(np.float64)
    row = Z.sum(axis=1).astype(np.float64)
    total = col.sum()
    if total <= 0:
        return None
    mx, my = col @ xs / total, row @ ys / total
    dx, dy = xs - mx, ys - my
    cxx = col @ (dx * dx) / total
    cyy = row @ (dy * dy) / total
    cxy = dy @ Z @ dx / total
    mu = np.array([mx, my])
    cov = np.array([[cxx, cxy], [cxy, cyy]])
    try:
        eigval, eigvec = np.linalg.eigh(cov)  # ascending
    except np.linalg.LinAlgError:
        return None
    sigma_1, sigma_0 = np.sqrt(np.clip(4 * eigval, 0, None))  # minor, major
    vec_1, vec_0 = eigvec[:, 0], eigvec[:, 1]

    # peak intensity: raw value at the fitted centre, plus a weighted estimate
    # averaged over the 1-sigma ellipse (robust against single-pixel noise)
    I0_peak = float(Z[int(np.abs(dy).argmin()), int(np.abs(dx).argmin())])
    I0_weighted = I0_peak
    det = cxx * cyy - cxy * cxy
    if det > 0:
        a, b, c = cyy / det, cxy / det, cxx / det  # inverse covariance
        quad = (a * dx * dx)[None, :] + (c * dy * dy)[:, None] - 2 * b * np.outer(dy, dx)
        inside = quad <= 1.0
        n_inside = int(inside.sum())
        if n_inside:
            I0_weighted = float(Z[inside].sum() / (n_inside * 2 * (1 - np.exp(-0.5))))

    return {
        "x": float(mu[0]),
        "y": float(mu[1]),
        "sigma_0": float(sigma_0),
        "sigma_1": float(sigma_1),
        "sigma": float(np.sqrt(sigma_0 * sigma_1)),
        "vec_0": vec_0,
        "vec_1": vec_1,
        "I0": I0_peak,
        "I0_weighted": I0_weighted,
    }


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
