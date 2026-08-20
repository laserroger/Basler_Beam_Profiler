"""Synthetic beam images: fixed Gaussian spot patterns with realistic noise.

Shared by the simulated camera (real-time frames), the test suite (ground
truth) and tools/make_test_images.py (fixed pictures on disk).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Spot:
    """Ground-truth spot: centre (px), 1/e^2-ish widths (px), tilt (rad),
    amplitude as a fraction of saturation."""

    x: float
    y: float
    sigma_x: float
    sigma_y: float | None = None
    theta: float = 0.0
    amplitude: float = 0.85

    @property
    def sy(self) -> float:
        return self.sigma_x if self.sigma_y is None else self.sigma_y


def spot_grid(
    width: int,
    height: int,
    nx: int = 4,
    ny: int = 4,
    sigma: float = 8.0,
    pitch: float | None = None,
    amplitude: float = 0.85,
) -> list[Spot]:
    """A centred nx x ny grid of round spots."""
    if pitch is None:
        pitch = min(width, height) / (max(nx, ny) + 1)
    x0 = width / 2 - (nx - 1) * pitch / 2
    y0 = height / 2 - (ny - 1) * pitch / 2
    return [
        Spot(x0 + i * pitch, y0 + j * pitch, sigma, amplitude=amplitude)
        for j in range(ny)
        for i in range(nx)
    ]


def render(
    width: int,
    height: int,
    spots: list[Spot],
    saturation: int = 65535,
    background: float = 0.01,
    noise: float = 0.003,
    gain: float = 1.0,
    rng: np.random.Generator | None = None,
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Render spots into a (height, width) frame.

    `gain` scales spot intensity (used to emulate exposure time), `offset`
    shifts spot coordinates (used to emulate a sensor ROI).  Spots are drawn
    patch-wise so large frames stay cheap.
    """
    img = np.full((height, width), background * saturation, dtype=np.float32)
    ox, oy = offset
    for s in spots:
        cx, cy = s.x - ox, s.y - oy
        r = 4.0 * max(s.sigma_x, s.sy)
        x0, x1 = int(max(0, cx - r)), int(min(width, cx + r + 1))
        y0, y1 = int(max(0, cy - r)), int(min(height, cy + r + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1, dtype=np.float32) - cx
        ys = np.arange(y0, y1, dtype=np.float32) - cy
        X, Y = np.meshgrid(xs, ys)
        ct, st = np.cos(s.theta), np.sin(s.theta)
        u, v = ct * X + st * Y, -st * X + ct * Y
        patch = np.exp(-0.5 * ((u / s.sigma_x) ** 2 + (v / s.sy) ** 2))
        img[y0:y1, x0:x1] += (s.amplitude * gain * saturation) * patch
    if noise > 0:
        rng = rng or np.random.default_rng()
        img += rng.normal(0.0, noise * saturation, img.shape).astype(np.float32)
    dtype = np.uint8 if saturation <= 255 else np.uint16
    return np.clip(img, 0, saturation).astype(dtype)
