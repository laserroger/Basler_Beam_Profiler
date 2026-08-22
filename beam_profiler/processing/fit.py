"""Sub-pixel Gaussian *moment* fits on spot crops.

Two code paths, identical maths:

* `fit_spot` - one spot at a time, and the only path that handles very large
  crops (they are coarse-grained to `COARSE_N` first so the moment sums stay
  O(1) regardless of spot size).
* `fit_spots` - all spots at once.  Per-spot numpy calls dominate the frame
  time on dense arrays (~70 us/spot, i.e. 0.45 s for an 80x80 grid), and
  almost none of it is arithmetic; gathering the crops into one (n, h, w)
  tensor and running the moments vectorised removes that overhead.  With CuPy
  available the same maths runs on the GPU (see `gpu.py`).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import gpu
from .spots import SpotArray

COARSE_N = 40  # crops larger than 2*COARSE_N are downsampled to COARSE_N^2
COARSE_MAX = 2 * COARSE_N
DEFAULT_WINDOW = 2.0  # crop half-width in units of the detected blob radius
_I0W_NORM = 2 * (1 - np.exp(-0.5))  # mean/peak ratio inside the 1-sigma ellipse


def crop_bounds(img_shape, x, y, r, window: float = DEFAULT_WINDOW):
    """Integer crop rectangles around each (x, y); mirrors int() truncation."""
    h, w = img_shape
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    x0 = np.maximum(0.0, x - window * r).astype(np.int64)
    x1 = np.minimum(float(w), x + window * r).astype(np.int64)
    y0 = np.maximum(0.0, y - window * r).astype(np.int64)
    y1 = np.minimum(float(h), y + window * r).astype(np.int64)
    return x0, x1, y0, y1


# --------------------------------------------------------------------------- #
#  single spot (legacy path, handles arbitrarily large crops)
# --------------------------------------------------------------------------- #
def fit_spot(img, x: float, y: float, radius: float, window: float = DEFAULT_WINDOW):
    """Sub-pixel Gaussian moment fit on a crop around (x, y).  None if unusable."""
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
    if max(crop.shape) > COARSE_MAX:
        Z = cv2.resize(crop, (COARSE_N, COARSE_N), interpolation=cv2.INTER_AREA)
        xs = np.linspace(x0, x1, COARSE_N)
        ys = np.linspace(y0, y1, COARSE_N)
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
    sigma_0, sigma_1, vec_0, vec_1 = _principal_axes(cxx, cyy, cxy)

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
            I0_weighted = float(Z[inside].sum() / (n_inside * _I0W_NORM))

    return {
        "x": float(mx), "y": float(my),
        "sigma_0": float(sigma_0), "sigma_1": float(sigma_1),
        "sigma": float(np.sqrt(sigma_0 * sigma_1)),
        "vec_0": vec_0, "vec_1": vec_1,
        "I0": I0_peak, "I0_weighted": I0_weighted,
    }


def _principal_axes(cxx, cyy, cxy):
    """Eigen-decomposition of the 2x2 covariance; widths are 2x the Gaussian sigma."""
    cov = np.array([[cxx, cxy], [cxy, cyy]])
    try:
        eigval, eigvec = np.linalg.eigh(cov)  # ascending
    except np.linalg.LinAlgError:
        return 0.0, 0.0, np.array([1.0, 0.0]), np.array([0.0, 1.0])
    sigma_1, sigma_0 = np.sqrt(np.clip(4 * eigval, 0, None))  # minor, major
    return sigma_0, sigma_1, eigvec[:, 1], eigvec[:, 0]


# --------------------------------------------------------------------------- #
#  all spots at once
# --------------------------------------------------------------------------- #
def fit_spots(img, x, y, r, window: float = DEFAULT_WINDOW) -> SpotArray:
    """Fit every candidate; returns a SpotArray in candidate order (drops failures)."""
    if len(x) == 0:
        return SpotArray.empty()
    x0, x1, y0, y1 = crop_bounds(img.shape, x, y, r, window)
    cw, ch = x1 - x0, y1 - y0
    usable = (cw >= 3) & (ch >= 3)
    # oversized crops need the coarse-graining resize, which only the per-spot
    # path does; they are rare (few, fat spots) so the loop cost is irrelevant
    batched = usable & (np.maximum(cw, ch) <= COARSE_MAX)
    legacy = np.nonzero(usable & ~batched)[0]

    idx, parts = [], []
    sel = np.nonzero(batched)[0]
    if sel.size:
        keep, arr = _fit_batched(img, x0[sel], y0[sel], cw[sel], ch[sel])
        idx.append(sel[keep])
        parts.append(arr)
    if legacy.size:
        fits = [(i, fit_spot(img, x[i], y[i], r[i], window)) for i in legacy]
        fits = [(i, s) for i, s in fits if s is not None]
        if fits:
            idx.append(np.array([i for i, _ in fits]))
            parts.append(SpotArray.from_dicts([s for _, s in fits]))
    if not parts:
        return SpotArray.empty()
    return SpotArray.concat(parts)[np.argsort(np.concatenate(idx), kind="stable")]


def _fit_batched(img, x0, y0, cw, ch):
    """Returns (indices into the given candidates that produced a fit, SpotArray)."""
    on_gpu = gpu.fit_batch(img, x0, y0, cw, ch)
    if on_gpu is not None:
        return on_gpu
    # the CPU path needs one uniformly sized tensor per distinct crop size
    keep, parts = [], []
    key = ch.astype(np.int64) * (img.shape[1] + 1) + cw
    for k in np.unique(key):
        g = np.nonzero(key == k)[0]
        good, cols = fit_crops(img, x0[g], y0[g], int(ch[g[0]]), int(cw[g[0]]), np)
        keep.append(g[good])
        parts.append(SpotArray(*cols)[good])
    return np.concatenate(keep), SpotArray.concat(parts)


def fit_crops(img, x0, y0, ch: int, cw: int, xp):
    """Vectorised moment fit of n equally sized crops; xp is numpy or cupy.

    Returns (good mask, columns), both already on the host."""
    _, W = img.shape
    n = x0.shape[0]
    ar_h, ar_w = xp.arange(ch), xp.arange(cw)
    x0d, y0d = xp.asarray(x0), xp.asarray(y0)
    flat = (y0d[:, None, None] + ar_h[None, :, None]) * W + (
        x0d[:, None, None] + ar_w[None, None, :]
    )
    Z = img.ravel()[flat.ravel()].reshape(n, ch, cw).astype(xp.float32)

    border = xp.concatenate([Z[:, 0, :], Z[:, -1, :], Z[:, :, 0], Z[:, :, -1]], axis=1)
    Z -= xp.median(border, axis=1)[:, None, None]
    xp.clip(Z, 0, None, out=Z)

    xs = (x0d[:, None] + ar_w[None, :]).astype(xp.float64)
    ys = (y0d[:, None] + ar_h[None, :]).astype(xp.float64)
    col = Z.sum(axis=1, dtype=xp.float64)  # marginal over y -> (n, cw)
    row = Z.sum(axis=2, dtype=xp.float64)  # marginal over x -> (n, ch)
    total = col.sum(axis=1)
    good = total > 0
    tot = xp.where(good, total, 1.0)
    mx = (col * xs).sum(axis=1) / tot
    my = (row * ys).sum(axis=1) / tot
    dx, dy = xs - mx[:, None], ys - my[:, None]
    cxx = (col * dx * dx).sum(axis=1) / tot
    cyy = (row * dy * dy).sum(axis=1) / tot
    cxy = xp.einsum("ni,nij,nj->n", dy, Z.astype(xp.float64), dx) / tot

    sigma_0, sigma_1, vx, vy = principal_axes_batch(cxx, cyy, cxy, xp)

    rows_ = xp.arange(n)
    I0 = Z[rows_, xp.abs(dy).argmin(axis=1), xp.abs(dx).argmin(axis=1)].astype(xp.float64)

    det = cxx * cyy - cxy * cxy
    ok = det > 0
    safe_det = xp.where(ok, det, 1.0)
    a, b, c = cyy / safe_det, cxy / safe_det, cxx / safe_det  # inverse covariance
    quad = (
        (a[:, None] * dx * dx)[:, None, :]
        + (c[:, None] * dy * dy)[:, :, None]
        - 2 * b[:, None, None] * dy[:, :, None] * dx[:, None, :]
    )
    inside = quad <= 1.0
    n_inside = inside.sum(axis=(1, 2))
    I0w = xp.where(
        ok & (n_inside > 0),
        xp.where(inside, Z, 0).sum(axis=(1, 2), dtype=xp.float64)
        / xp.maximum(n_inside, 1)
        / _I0W_NORM,
        I0,
    )
    cols = (
        mx, my, sigma_0, sigma_1,
        xp.stack([vx, vy], axis=1), xp.stack([-vy, vx], axis=1),
        I0, I0w,
    )
    return to_host(good), tuple(to_host(v) for v in cols)


def principal_axes_batch(cxx, cyy, cxy, xp):
    """Analytic symmetric-2x2 eigendecomposition, ascending (matches eigh)."""
    half_tr = 0.5 * (cxx + cyy)
    disc = xp.sqrt(xp.maximum(0.25 * (cxx - cyy) ** 2 + cxy * cxy, 0.0))
    sigma_1 = xp.sqrt(xp.clip(4 * (half_tr - disc), 0, None))
    sigma_0 = xp.sqrt(xp.clip(4 * (half_tr + disc), 0, None))
    vx, vy = cxy, half_tr + disc - cxx  # eigenvector of the larger eigenvalue
    norm = xp.sqrt(vx * vx + vy * vy)
    degenerate = norm < 1e-12
    safe = xp.where(degenerate, 1.0, norm)
    vx = xp.where(degenerate, 1.0, vx / safe)
    vy = xp.where(degenerate, 0.0, vy / safe)
    return sigma_0, sigma_1, vx, vy


def to_host(v) -> np.ndarray:
    return np.asarray(v.get() if hasattr(v, "get") else v)
