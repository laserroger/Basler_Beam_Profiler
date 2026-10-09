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
