"""Raw-pixel saturation warnings, independent of display scaling or heatmaps."""
import numpy as np


def saturated_spots(frame, spots, full_scale):
    """Flag spots with pixels at >=99.9% full scale inside their fitted ellipse.

    The tolerance includes MSB-aligned 10/12-bit samples in a 16-bit container.
    Coordinates are camera-ROI pixels, not sensor or display coordinates.
    A warning is conservative: even one clipped/hot pixel is worth inspecting.
    """
    flags = np.zeros(len(spots), dtype=bool)
    if frame is None or not len(spots):
        return flags
    ys, xs = np.nonzero(frame >= full_scale * .999)
    if not len(xs):
        return flags
    for i in range(len(spots)):
        a, b = max(1., spots.sigma_0[i]), max(1., spots.sigma_1[i])
        radius = max(a, b)
        nearby = (abs(xs-spots.x[i]) <= radius) & (abs(ys-spots.y[i]) <= radius)
        dx, dy = xs[nearby]-spots.x[i], ys[nearby]-spots.y[i]
        theta = np.deg2rad(spots.angle[i])
        c, s = np.cos(theta), np.sin(theta)
        flags[i] = np.any(((c*dx+s*dy)/a)**2 + ((-s*dx+c*dy)/b)**2 <= 1)
    return flags
