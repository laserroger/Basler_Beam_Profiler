"""Per-spot nonlinear displacement relative to a freely spaced grid."""
import numpy as np
from collections import deque


class DisplacementAverage:
    """Rolling five-frame mean, matched by grid cell rather than detection order."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=5)
        self.context = None
        self.frame_id = None
        self.latest = None

    def update(self, displacement, context, frame_id):
        if not displacement['valid']:
            self.reset()
            return displacement
        order = np.lexsort((displacement['columns'], displacement['rows']))
        current = {**displacement, **{k: displacement[k][order].copy()
                    for k in ('reference', 'residual', 'rows', 'columns')}}
        changed = context != self.context
        if self.frames:
            previous = self.frames[-1]
            changed |= (not np.array_equal(current['rows'], previous['rows'])
                        or not np.array_equal(current['columns'], previous['columns']))
        if changed:
            self.reset()
        self.context = context
        if frame_id is not None and frame_id == self.frame_id:
            return self.latest
        self.frames.append(current)
        self.frame_id = frame_id
        if len(self.frames) < 5:
            self.latest = {'valid': False, 'reason': f'Collecting {len(self.frames)}/5 frames'}
        else:
            residual = np.mean([f['residual'] for f in self.frames], axis=0)
            self.latest = {**current,
                           'reference': np.mean([f['reference'] for f in self.frames], axis=0),
                           'residual': residual, 'averaged_frames': 5,
                           'rms_px': float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))}
        return self.latest


def grid_displacement(rows, columns, shear):
    """Return ROI-coordinate reference points and residual vectors for E.

    Intersect the fitted row/column lines, allowing their measured rotation
    and shear, then remove the residual affine field. No equal pitch is assumed.
    Arrays are local to this display calculation; published results stay intact.
    """
    if not shear['valid']:
        return {'valid': False, 'reason': shear['reason']}
    rxy = np.concatenate([np.column_stack((r.x, r.y)) for r in rows])
    cxy = np.concatenate([np.column_stack((c.x, c.y)) for c in columns])
    rid = np.repeat(np.arange(len(rows)), [len(r) for r in rows])
    cid = np.repeat(np.arange(len(columns)), [len(c) for c in columns])
    ro = np.lexsort((rxy[:, 1], rxy[:, 0]))
    co = np.lexsort((cxy[:, 1], cxy[:, 0]))
    xy, rid, cid = rxy[ro], rid[ro], cid[co]
    center = xy.mean(axis=0)
    ru = np.asarray(shear['row_direction'])
    cv = np.asarray(shear['column_direction'])
    normals = np.array([[cv[1], -cv[0]], [-ru[1], ru[0]]])
    uv = (xy - center) @ normals.T
    xs = np.bincount(cid, weights=uv[:, 0]) / np.bincount(cid)
    ys = np.bincount(rid, weights=uv[:, 1]) / np.bincount(rid)
    reference = np.linalg.solve(normals, np.column_stack((xs[cid], ys[rid])).T).T + center
    delta = xy - reference
    extent = np.ptp(reference, axis=0).max()
    if extent <= 0:
        return {'valid': False, 'reason': 'Grid has no spatial extent'}
    design = np.column_stack((np.ones(len(xy)), (reference - center) / extent))
    coefficients, _, rank, _ = np.linalg.lstsq(design, delta, rcond=None)
    if rank < 3:
        return {'valid': False, 'reason': 'Grid does not span two dimensions'}
    linear = design @ coefficients
    residual = delta - linear
    return {'valid': True, 'reference': reference + linear, 'residual': residual,
            'rows': rid, 'columns': cid,
            'rms_px': float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))}


class ImageGridAverage:
    """Non-overlapping 20-image batches: average pixels, then fit once."""
    def __init__(self, frames=20):
        self.frames = frames
        self.reset()

    def reset(self):
        self.total = None
        self.count = 0
        self.context = None
        self.frame_id = None
        self.spots = None
        self.latest = {'valid': False, 'reason': f'Collecting 0/{self.frames} images'}

    def update(self, frame, *, roi, camera, exposure, gain, full_scale, config,
               fit_rect=None, frame_id=None):
        context = (tuple(roi), camera, exposure, gain, full_scale, config,
                   tuple(fit_rect) if fit_rect is not None else None, frame.shape)
        if context != self.context:
            self.reset()
            self.context = context
        if frame_id is not None and frame_id == self.frame_id:
            return self.latest
        self.frame_id = frame_id
        if self.total is None:
            self.total = np.array(frame, dtype=np.float64, copy=True)
        else:
            self.total += frame
        self.count += 1
        if self.count < self.frames:
            if not self.latest['valid']:
                self.latest = {'valid': False,
                               'reason': f'Collecting {self.count}/{self.frames} images'}
            return self.latest
        mean = self.total / self.frames
        self.total, self.count = None, 0
        self.latest, self.spots = fit_averaged_grid(mean, roi, full_scale, config, fit_rect)
        self.latest['averaged_frames'] = self.frames
        return self.latest


def fit_averaged_grid(mean, roi, full_scale, config, fit_rect=None):
    """Fit beam centers and a quadratic residual field after pixel averaging."""
    from scipy.ndimage import gaussian_filter, maximum_filter
    from .fit import fit_spots
    from .measurements import SpotStatistics
    from .spots import SpotArray
    from .pipeline import saturated_spots

    # Detection threshold is in equivalent 8-bit units; centroid fitting retains
    # the floating-point full-depth average and uses untruncated image crops.
    preview = mean * (255. / full_scale)
    blurred = gaussian_filter(preview, 1.2)
    contrast = blurred - gaussian_filter(preview, 12)
    y, x = np.where((blurred == maximum_filter(blurred, 25)) & (contrast > 2.))
    if fit_rect is not None:
        x1, y1, x2, y2 = fit_rect
        keep = ((x + roi[2] >= x1) & (x + roi[2] <= x2)
                & (y + roi[3] >= y1) & (y + roi[3] <= y2))
        x, y = x[keep], y[keep]
    if len(x) < 4:
        return {'valid': False, 'reason': 'Too few spots in averaged image'}, SpotArray.empty()
    spots = fit_spots(mean, x, y, np.full(len(x), 6.), config.replace(gpu_enabled=False))
    stats = SpotStatistics()
    stats.update(spots, saturated=saturated_spots(mean, spots, full_scale))
    result = grid_displacement(stats.rows, stats.columns, stats.grid_shear)
    if not result['valid']:
        return result, spots
    ref, residual = result['reference'], result['residual']
    scale = np.ptp(ref, axis=0) / 2
    if np.any(scale <= 0):
        return {'valid': False, 'reason': 'Grid has no spatial extent'}, spots
    x, y = ((ref - ref.mean(axis=0)) / scale).T
    design = np.column_stack((np.ones(len(x)), x, y, x*x, x*y, y*y))
    coef, _, rank, _ = np.linalg.lstsq(design, residual, rcond=None)
    if rank < 6:
        return {'valid': False, 'reason': 'Need at least three rows and columns'}, spots
    smooth = design @ coef
    result.update(measured_residual=residual, residual=smooth,
                  fit_error_rms_px=float(np.sqrt(np.mean(np.sum((residual-smooth)**2, axis=1)))))
    return result, spots
