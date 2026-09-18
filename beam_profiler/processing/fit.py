"""Sub-pixel spot fitting by second moments, with a self-consistent crop.

The estimator, in order:

1. Crop around the blob-detector candidate (`seed_window` x its radius).
2. Subtract the median of the crop border - the local background.
3. Second moments of the (un-clipped) crop -> centre and covariance.
4. Re-crop at `crop_sigma` x the fitted sigma *per axis* and repeat, so the
   window is set by the spot itself rather than by the detector's guess.
5. Optionally (`adaptive_iters` > 0) iterate a Gaussian weight matched to the
   current covariance.  For a Gaussian source measured through a Gaussian
   weight the measured covariance is (C^-1 + W^-1)^-1, so at the fixed point
   W = C it is exactly C/2 - hence `C = 2M`, with no calibration constant.

Two things that look like details and are not:

* **Nothing is clipped at zero.**  Clipping rectifies zero-mean noise into a
  positive pedestal (E[max(z,0)] = 0.4 sigma_noise) which the second moment
  then multiplies by distance^2.  The error grows like L^4 with crop size:
  measured +160% (bright) and +364% (faint) at a 12-sigma crop, converging on
  the width of the *crop* rather than the beam.  A Gaussian weight suppresses
  the same tail noise without touching its expectation, which is why weighting
  is safe where clipping is not.
* **Crop half-widths are per axis** (`sqrt(cxx)`, `sqrt(cyy)`), not one radius
  from the major axis.  On an 18x5 elliptical spot that is 3.6x fewer pixels
  *and* takes the minor width from +6.25% error to +0.01%.

All the numbers live in `beam_profiler.fitconfig` and are editable from the
settings window; the defaults are the measured optima.
"""

from __future__ import annotations

import numpy as np

from ..fitconfig import FitConfig
from ..fitconfig import active as active_config
from . import gpu
from .spots import SpotArray

_I0W_NORM = 2 * (1 - np.exp(-0.5))  # mean/peak ratio inside the 1-sigma ellipse
_QUAD_MAX = 200.0  # exp(-q/2) underflows well before this; guards inf*0
_EPS = 1e-12


# --------------------------------------------------------------------------- #
#  crop geometry
# --------------------------------------------------------------------------- #
def neighbour_limit(x, y, cfg: FitConfig) -> np.ndarray:
    """Half-width cap per spot: a fraction of the distance to the nearest
    other candidate, so no crop can swallow its neighbour."""
    n = len(x)
    if n < 2:
        return np.full(n, np.inf)
    from scipy.spatial import cKDTree  # noqa: PLC0415  (deferred: ~0.1 s import)

    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    dist, _ = cKDTree(pts).query(pts, k=2)
    return cfg.neighbour_fraction * dist[:, 1]


def crop_grid(shape, cx, cy, half_x, half_y, cfg: FitConfig):
    """Integer crops of quantised size centred on (cx, cy).

    Sizes are rounded up to `crop_quantum` so the CPU path gets few distinct
    shapes to batch; crops that would fall off the frame are shifted inwards
    rather than truncated, which keeps them symmetric about their own centre."""
    H, W = shape
    q = max(1, int(cfg.crop_quantum))
    hi_w = int(min(cfg.max_crop, W))
    hi_h = int(min(cfg.max_crop, H))
    cw = np.clip((np.ceil(2 * np.asarray(half_x) / q) * q).astype(np.int64), cfg.min_crop, hi_w)
    ch = np.clip((np.ceil(2 * np.asarray(half_y) / q) * q).astype(np.int64), cfg.min_crop, hi_h)
    x0 = np.clip(np.round(cx).astype(np.int64) - cw // 2, 0, W - cw)
    y0 = np.clip(np.round(cy).astype(np.int64) - ch // 2, 0, H - ch)
    return x0, y0, cw, ch


def next_half(cxx, cyy, cfg: FitConfig, cap):
    """Crop half-widths for the next stage, per axis, capped by the neighbours."""
    hx = cfg.crop_sigma * np.sqrt(np.maximum(cxx, _EPS))
    hy = cfg.crop_sigma * np.sqrt(np.maximum(cyy, _EPS))
    return np.minimum(hx, cap), np.minimum(hy, cap)


# --------------------------------------------------------------------------- #
#  the moment core (numpy or cupy)
# --------------------------------------------------------------------------- #
def moment_core(Z, xs, ys, cfg: FitConfig, xp):
    """Centre and covariance of n crops.  `xs`/`ys` are the (n, cw)/(n, ch)
    absolute pixel coordinates; `Z` is background-subtracted."""
    # plain moments first, from the marginals (same maths as the full 2D sum)
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

    for _ in range(int(cfg.adaptive_iters)):
        det = cxx * cyy - cxy * cxy
        ok = good & (det > 0)
        safe = xp.where(ok, det, 1.0)
        ia, ib, ic = cyy / safe, cxy / safe, cxx / safe  # inverse covariance
        dx, dy = xs - mx[:, None], ys - my[:, None]
        quad = (
            (ia[:, None] * dx * dx)[:, None, :]
            + (ic[:, None] * dy * dy)[:, :, None]
            - 2 * ib[:, None, None] * dy[:, :, None] * dx[:, None, :]
        )
        w = xp.exp(-0.5 * xp.clip(quad, 0.0, _QUAD_MAX))
        wz = w * Z
        s = wz.sum(axis=(1, 2), dtype=xp.float64)
        ok = ok & (s > 0)
        s_safe = xp.where(ok, s, 1.0)
        mx_n = (wz * xs[:, None, :]).sum(axis=(1, 2), dtype=xp.float64) / s_safe
        my_n = (wz * ys[:, :, None]).sum(axis=(1, 2), dtype=xp.float64) / s_safe
        ddx, ddy = xs - mx_n[:, None], ys - my_n[:, None]
        # C = 2M: exact for a Gaussian source measured through a matched weight
        cxx_n = 2 * (wz * (ddx * ddx)[:, None, :]).sum(axis=(1, 2), dtype=xp.float64) / s_safe
        cyy_n = 2 * (wz * (ddy * ddy)[:, :, None]).sum(axis=(1, 2), dtype=xp.float64) / s_safe
        cxy_n = 2 * xp.einsum("nij,ni,nj->n", wz, ddy, ddx) / s_safe
        mx = xp.where(ok, mx_n, mx)
        my = xp.where(ok, my_n, my)
        cxx = xp.where(ok, cxx_n, cxx)
        cyy = xp.where(ok, cyy_n, cyy)
        cxy = xp.where(ok, cxy_n, cxy)

    return good, mx, my, cxx, cyy, cxy


def principal_axes(cxx, cyy, cxy, xp):
    """Analytic symmetric-2x2 eigendecomposition; widths are 2x the Gaussian sigma."""
    half_tr = 0.5 * (cxx + cyy)
    disc = xp.sqrt(xp.maximum(0.25 * (cxx - cyy) ** 2 + cxy * cxy, 0.0))
    sigma_0 = xp.sqrt(xp.clip(4 * (half_tr + disc), 0, None))  # major
    sigma_1 = xp.sqrt(xp.clip(4 * (half_tr - disc), 0, None))  # minor
    vx, vy = cxy, half_tr + disc - cxx
    norm = xp.sqrt(vx * vx + vy * vy)
    degenerate = norm < _EPS
    safe = xp.where(degenerate, 1.0, norm)
    vx = xp.where(degenerate, 1.0, vx / safe)
    vy = xp.where(degenerate, 0.0, vy / safe)
    return sigma_0, sigma_1, vx, vy


def gather(img, x0, y0, ch: int, cw: int, cfg: FitConfig, xp):
    """Pull n equally sized crops into one (n, ch, cw) tensor, background removed."""
    W = img.shape[1]
    ar_h, ar_w = xp.arange(ch), xp.arange(cw)
    x0d, y0d = xp.asarray(x0), xp.asarray(y0)
    flat = (y0d[:, None, None] + ar_h[None, :, None]) * W + (
        x0d[:, None, None] + ar_w[None, None, :]
    )
    Z = img.ravel()[flat.ravel()].reshape(-1, ch, cw).astype(xp.float32)
    if cfg.subtract_background:
        border = xp.concatenate([Z[:, 0, :], Z[:, -1, :], Z[:, :, 0], Z[:, :, -1]], axis=1)
        Z -= xp.median(border, axis=1)[:, None, None]
    if cfg.clip_negative:
        xp.clip(Z, 0, None, out=Z)
    xs = (x0d[:, None] + ar_w[None, :]).astype(xp.float64)
    ys = (y0d[:, None] + ar_h[None, :]).astype(xp.float64)
    return Z, xs, ys


def fit_crops(img, x0, y0, ch: int, cw: int, cfg: FitConfig, xp, full: bool = True):
    """One stage over n equally sized crops.

    Returns (good, mx, my, cxx, cyy, cxy, I0, I0_weighted); the last two are
    None unless `full`, since intermediate stages never use them."""
    Z, xs, ys = gather(img, x0, y0, ch, cw, cfg, xp)
    good, mx, my, cxx, cyy, cxy = moment_core(Z, xs, ys, cfg, xp)
    if not full:
        return good, mx, my, cxx, cyy, cxy, None, None

    dx, dy = xs - mx[:, None], ys - my[:, None]
    rows_ = xp.arange(Z.shape[0])
    I0 = Z[rows_, xp.abs(dy).argmin(axis=1), xp.abs(dx).argmin(axis=1)].astype(xp.float64)
    det = cxx * cyy - cxy * cxy
    ok = det > 0
    safe = xp.where(ok, det, 1.0)
    ia, ib, ic = cyy / safe, cxy / safe, cxx / safe
    quad = (
        (ia[:, None] * dx * dx)[:, None, :]
        + (ic[:, None] * dy * dy)[:, :, None]
        - 2 * ib[:, None, None] * dy[:, :, None] * dx[:, None, :]
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
    return good, mx, my, cxx, cyy, cxy, I0, I0w


# --------------------------------------------------------------------------- #
#  all spots at once
# --------------------------------------------------------------------------- #
def fit_spots(img, x, y, r, cfg: FitConfig | None = None) -> SpotArray:
    """Fit every candidate; returns a SpotArray in candidate order.

    Candidates whose crop carries no positive signal are dropped."""
    cfg = cfg or active_config()
    n = len(x)
    if n == 0:
        return SpotArray.empty()
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    cap = neighbour_limit(x, y, cfg)

    batched = gpu.fit_batch(img, x, y, r, cap, cfg)
    if batched is not None:
        keep, spots = batched
        return spots[np.argsort(keep, kind="stable")]

    alive = np.ones(n, bool)
    cx, cy = x.copy(), y.copy()
    half = np.minimum(cfg.seed_window * r, cap)
    half_x, half_y = half.copy(), half.copy()
    cxx = np.zeros(n)
    cyy = np.zeros(n)
    cxy = np.zeros(n)
    I0 = np.zeros(n)
    I0w = np.zeros(n)

    stages = int(cfg.crop_stages)
    previous = None
    for stage in range(stages + 1):
        last = stage == stages
        grid = crop_grid(img.shape, cx, cy, half_x, half_y, cfg)
        if previous is None or last:
            todo = alive.copy()
        else:
            # the fit depends only on the crop, so a spot whose crop did not
            # move would return exactly what it returned last stage.  Most
            # spots settle after one re-crop, which halves the work.
            moved = np.zeros(n, bool)
            for now, before in zip(grid, previous):
                moved |= now != before
            todo = alive & moved
        previous = grid

        sel = np.nonzero(todo)[0]
        if sel.size:
            good, mx, my, axx, ayy, axy, i0, i0w = _fit_stage(
                img, [g[sel] for g in grid], cfg, full=last
            )
            alive[sel] = good
            ok = sel[good]
            cx[ok], cy[ok] = mx[good], my[good]
            cxx[ok], cyy[ok], cxy[ok] = axx[good], ayy[good], axy[good]
            if last:
                I0[ok], I0w[ok] = i0[good], i0w[good]
        if not last:
            half_x[alive], half_y[alive] = next_half(cxx[alive], cyy[alive], cfg, cap[alive])

    if not alive.any():
        return SpotArray.empty()
    sigma_0, sigma_1, vx, vy = principal_axes(cxx[alive], cyy[alive], cxy[alive], np)
    return SpotArray(
        cx[alive], cy[alive], sigma_0, sigma_1,
        np.stack([vx, vy], axis=1), np.stack([-vy, vx], axis=1),
        I0[alive], I0w[alive],
    )


def _fit_stage(img, grid, cfg: FitConfig, full: bool):
    """Run one crop/fit stage over the given crops, grouped by crop size."""
    x0, y0, cw, ch = grid
    n = len(x0)
    good = np.zeros(n, bool)
    cols = [np.zeros(n) for _ in range(7)]
    key = ch * (img.shape[1] + 1) + cw
    for k in np.unique(key):
        g = np.nonzero(key == k)[0]
        res = fit_crops(img, x0[g], y0[g], int(ch[g[0]]), int(cw[g[0]]), cfg, np, full=full)
        good[g] = np.asarray(res[0])
        for slot, value in enumerate(res[1:]):
            if value is not None:
                cols[slot][g] = np.asarray(value)
    return (good, *cols)


# --------------------------------------------------------------------------- #
#  single spot: independent scalar reference for the batched paths
# --------------------------------------------------------------------------- #
def fit_spot(img, x: float, y: float, radius: float, cfg: FitConfig | None = None,
             cap: float = np.inf):
    """Same estimator, one spot, written without any batching.  Returns the
    legacy dict, or None if the crop carries no signal.

    `cap` is the neighbour half-width limit `fit_spots` would apply; leave it
    infinite for a genuinely isolated spot."""
    cfg = cfg or active_config()
    H, W = img.shape
    cx, cy = float(x), float(y)
    half_x = half_y = min(cfg.seed_window * radius, cap)
    for stage in range(int(cfg.crop_stages) + 1):
        q = max(1, int(cfg.crop_quantum))
        cw = int(np.clip(np.ceil(2 * half_x / q) * q, cfg.min_crop, min(cfg.max_crop, W)))
        ch = int(np.clip(np.ceil(2 * half_y / q) * q, cfg.min_crop, min(cfg.max_crop, H)))
        x0 = int(np.clip(round(cx) - cw // 2, 0, W - cw))
        y0 = int(np.clip(round(cy) - ch // 2, 0, H - ch))

        crop = img[y0 : y0 + ch, x0 : x0 + cw].astype(np.float32)
        if cfg.subtract_background:
            border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
            crop = crop - np.median(border)
        if cfg.clip_negative:
            crop = np.clip(crop, 0, None)
        xs = np.arange(x0, x0 + cw, dtype=np.float64)
        ys = np.arange(y0, y0 + ch, dtype=np.float64)

        # accumulate in float64 so this stays bit-comparable to the batched paths
        col = crop.sum(axis=0, dtype=np.float64)
        row = crop.sum(axis=1, dtype=np.float64)
        total = col.sum()
        if total <= 0:
            return None
        cx, cy = col @ xs / total, row @ ys / total
        dx, dy = xs - cx, ys - cy
        cxx = col @ (dx * dx) / total
        cyy = row @ (dy * dy) / total
        cxy = dy @ crop.astype(np.float64) @ dx / total

        for _ in range(int(cfg.adaptive_iters)):
            det = cxx * cyy - cxy * cxy
            if det <= 0:
                break
            ia, ib, ic = cyy / det, cxy / det, cxx / det
            dx, dy = xs - cx, ys - cy
            quad = (ia * dx * dx)[None, :] + (ic * dy * dy)[:, None] - 2 * ib * np.outer(dy, dx)
            wz = np.exp(-0.5 * np.clip(quad, 0.0, _QUAD_MAX)) * crop
            s = wz.sum()
            if s <= 0:
                break
            cx, cy = (wz.sum(axis=0) @ xs) / s, (wz.sum(axis=1) @ ys) / s
            ddx, ddy = xs - cx, ys - cy
            cxx = 2 * (wz.sum(axis=0) @ (ddx * ddx)) / s
            cyy = 2 * (wz.sum(axis=1) @ (ddy * ddy)) / s
            cxy = 2 * (ddy @ wz @ ddx) / s

        if stage < int(cfg.crop_stages):
            half_x = min(cfg.crop_sigma * np.sqrt(max(cxx, _EPS)), cap)
            half_y = min(cfg.crop_sigma * np.sqrt(max(cyy, _EPS)), cap)

    sigma_0, sigma_1, vx, vy = principal_axes(
        np.array([cxx]), np.array([cyy]), np.array([cxy]), np
    )
    dx, dy = xs - cx, ys - cy
    I0 = float(crop[int(np.abs(dy).argmin()), int(np.abs(dx).argmin())])
    I0_weighted = I0
    det = cxx * cyy - cxy * cxy
    if det > 0:
        ia, ib, ic = cyy / det, cxy / det, cxx / det
        quad = (ia * dx * dx)[None, :] + (ic * dy * dy)[:, None] - 2 * ib * np.outer(dy, dx)
        inside = quad <= 1.0
        n_inside = int(inside.sum())
        if n_inside:
            I0_weighted = float(crop[inside].sum(dtype=np.float64) / (n_inside * _I0W_NORM))

    return {
        "x": float(cx), "y": float(cy),
        "sigma_0": float(sigma_0[0]), "sigma_1": float(sigma_1[0]),
        "sigma": float(np.sqrt(sigma_0[0] * sigma_1[0])),
        "vec_0": np.array([vx[0], vy[0]]), "vec_1": np.array([-vy[0], vx[0]]),
        "I0": I0, "I0_weighted": I0_weighted,
    }
