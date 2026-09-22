"""Single elliptical Gaussian plus constant background, without blob detection.

Fit an area-averaged image for interactive CPU performance. Coordinates and
1/e^2 radii are returned in the original frame's pixels. Unlike spot moments,
the background is fitted jointly: an illuminated border is not treated as dark.
This is a Gaussian model estimate, not an ISO second-moment measurement.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from .spots import SpotArray


@dataclass
class BeamFit:
    spots: SpotArray
    message: str


def fit_single_beam(frame, max_side=128):
    def empty(message):
        return BeamFit(SpotArray.empty(), message)

    h, w = frame.shape
    if min(h, w) < 8:
        return empty('Select a larger fitting region')
    full_scale = np.iinfo(frame.dtype).max if np.issubdtype(frame.dtype, np.integer) else 1.0
    if np.count_nonzero(frame >= full_scale * 0.999) / frame.size > 0.001:
        return empty('Saturated: reduce exposure')
    scale = min(1.0, max_side / max(h, w))
    sw, sh = max(8, round(w * scale)), max(8, round(h * scale))
    image = cv2.resize(frame.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    low, high = np.percentile(image, [1, 99])
    span = high - low
    if not np.isfinite(image).all() or span < full_scale * 0.002:
        return empty('Insufficient beam contrast')
    z = ((image - low) / span).ravel().astype(np.float64)
    size = max(h, w)
    xs = ((np.arange(sw) + 0.5) * w / sw - 0.5 - (w - 1) / 2) / size
    ys = ((np.arange(sh) + 0.5) * h / sh - 0.5 - (h - 1) / 2) / size
    x, y = (v.ravel() for v in np.meshgrid(xs, ys))
    # Positive weights only initialize the optimizer, not the final measurement.
    weights = np.maximum(z, 0)
    total = weights.sum()
    cx, cy = np.dot(weights, x) / total, np.dot(weights, y) / total
    dx, dy = x - cx, y - cy
    cov = np.array([[np.dot(weights, dx * dx), np.dot(weights, dx * dy)],
                    [np.dot(weights, dx * dy), np.dot(weights, dy * dy)]]) / total
    values, vectors = np.linalg.eigh(cov)
    initial_sigma = np.sqrt(np.maximum(values[::-1], 0.01 ** 2))
    direction = vectors[:, -1]
    theta = np.arctan2(direction[1], direction[0])
    initial = [0, max(float(z.max()), 0.1), cx, cy, *np.log(initial_sigma), theta]
    minimum_sigma = 1 / min(sw, sh)
    # Normalization subtracts `low`; zero physical background is -low/span.
    # An unconstrained negative pedestal can cancel an arbitrarily wide
    # Gaussian and turn weak curvature into a spurious enormous radius.
    lower = [-float(low / span), 0, xs[0], ys[0], np.log(minimum_sigma), np.log(minimum_sigma), -2*np.pi]
    upper = [1, 10, xs[-1], ys[-1], np.log(2), np.log(2), 2*np.pi]
    initial = np.clip(initial, np.array(lower) + 1e-6, np.array(upper) - 1e-6)

    evaluator = _GaussianEvaluator(x, y, z)

    # Fit intensity residuals without suppressing the bright core as an outlier.
    # A robust loss scaled to the full-frame percentiles can instead fit a
    # faint halo when the core occupies only a small fraction of the frame.
    result = least_squares(evaluator.residual, initial, jac=evaluator.jacobian, bounds=(lower, upper),
                           loss='linear', max_nfev=60, ftol=1e-5)
    if not result.success or not np.isfinite(result.x).all():
        return empty('Gaussian fit did not converge')
    bg, amplitude, cx, cy, log_sx, log_sy, angle = result.x
    residual = evaluator.residual(result.x)
    if amplitude < 3 * np.sqrt(np.mean(residual**2)) or np.mean(residual**2) > 0.5 * np.var(z):
        return empty('No clear single Gaussian beam')
    # Zero background is a valid solution, not a failed/edge-clipped fit.
    if np.any(result.active_mask[1:6]):
        return empty('Beam is not sufficiently contained for a stable fit')
    sigmas = np.exp([log_sx, log_sy]) * size
    axes = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    order = np.argsort(sigmas)[::-1]
    sigmas, axes = sigmas[order], axes[order]
    cx, cy = cx * size + (w - 1) / 2, cy * size + (h - 1) / 2
    extents = 2 * np.sqrt(np.sum((sigmas[:, None] * axes)**2, axis=0))
    clipped = cx-extents[0] < 0 or cx+extents[0] >= w or cy-extents[1] < 0 or cy+extents[1] >= h
    # Normalize by beam signal energy, not the number of dark pixels. This
    # is a model mismatch diagnostic, not a statistical uncertainty estimate.
    mismatch = np.linalg.norm(residual) / max(np.linalg.norm(z - bg), 1e-12)
    if mismatch > 0.20:
        message = f'Poor Gaussian match ({mismatch:.0%} residual); model estimate'
        if clipped:
            message += '; contour outside region'
    else:
        message = 'Gaussian model: 1/e^2 contour outside region' if clipped else 'Gaussian fit'
    peak = (bg + amplitude) * span + low
    mean_core = (bg + amplitude * 2 * (1 - np.exp(-0.5))) * span + low
    spots = SpotArray([cx], [cy], [2*sigmas[0]], [2*sigmas[1]],
                      axes[0:1], axes[1:2], [peak], [mean_core])
    return BeamFit(spots, message)


class _GaussianEvaluator:
    """Exact derivatives, sharing Gaussian terms with the residual evaluation.

    SciPy otherwise evaluates the whole image seven extra times per step to
    approximate the derivatives. Keep this cache local to one fit so changing
    frames/ROIs and concurrent fits cannot reuse stale image data.
    """

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self._parameters = None

    def _evaluate(self, p):
        if self._parameters is not None and np.array_equal(p, self._parameters):
            return
        bg, amplitude, mx, my, log_sx, log_sy, angle = p
        c, s = np.cos(angle), np.sin(angle)
        u = c * (self.x - mx) + s * (self.y - my)
        v = -s * (self.x - mx) + c * (self.y - my)
        ix, iy = np.exp(-2 * log_sx), np.exp(-2 * log_sy)
        e = np.exp(-0.5 * (u*u*ix + v*v*iy))
        signal = amplitude * e
        self._residual = bg + signal - self.z
        self._terms = (c, s, u, v, ix, iy, e, signal)
        self._jacobian = None
        self._parameters = np.array(p, copy=True)

    def residual(self, p):
        self._evaluate(p)
        return self._residual

    def jacobian(self, p):
        self._evaluate(p)
        if self._jacobian is None:
            c, s, u, v, ix, iy, e, signal = self._terms
            jac = np.empty((self.z.size, 7), dtype=np.float64, order="F")
            jac[:, 0] = 1
            jac[:, 1] = e
            jac[:, 2] = signal * (c*u*ix - s*v*iy)
            jac[:, 3] = signal * (s*u*ix + c*v*iy)
            jac[:, 4] = signal * u*u*ix
            jac[:, 5] = signal * v*v*iy
            jac[:, 6] = signal * u*v*(iy-ix)
            self._jacobian = jac
        return self._jacobian
