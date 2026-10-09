"""Rotation-independent shear of classified rows/columns, in image coordinates."""
from __future__ import annotations

import numpy as np


def unavailable(reason):
    return {"valid": False, "reason": reason}


def _axis_data(groups):
    points = [np.column_stack((g.x, g.y)) for g in groups]
    xy = np.concatenate(points)
    labels = np.repeat(np.arange(len(points)), [len(p) for p in points])
    centered = np.concatenate([p - p.mean(axis=0) for p in points])
    return xy, labels, centered.T @ centered


def fit_grid_shear(rows, columns):
    """Fit a shared direction per axis, with an independent intercept per line.

    Pooled total least squares weights spots equally and does not assume equal
    spacing (nor use grid indices as spatial coordinates). Directions point
    right/down; signed shear is 90 minus their included angle. Positive image
    rotation is clockwise. The orthogonal reference independently optimizes
    rotation and every row/column offset; its residual includes shear and bow.

    Classification is supplied by the existing approximately axis-aligned grid
    grouper. Incomplete grids are allowed, but every supplied spot must belong
    to both a row and a column, with at most one spot per intersection.
    """
    if len(rows) < 2 or len(columns) < 2:
        return unavailable("Need at least two classified rows and columns")
    if any(len(g) < 2 for g in (*rows, *columns)):
        return unavailable("Each grid line needs at least two spots")
    rxy, rid, sr = _axis_data(rows)
    cxy, cid, sc = _axis_data(columns)
    if not (np.isfinite(rxy).all() and np.isfinite(cxy).all()):
        return unavailable("Grid coordinates are not finite")
    ro = np.lexsort((rxy[:, 1], rxy[:, 0]))
    co = np.lexsort((cxy[:, 1], cxy[:, 0]))
    if rxy.shape != cxy.shape or not np.array_equal(rxy[ro], cxy[co]):
        return unavailable("Row/column grouping is incomplete or ambiguous")
    cells = rid[ro] * len(columns) + cid[co]
    if len(np.unique(cells)) != len(cells):
        return unavailable("Multiple spots assigned to one grid intersection")

    er, vr = np.linalg.eigh(sr)
    ec, vc = np.linalg.eigh(sc)
    if min(er[-1], ec[-1]) <= 0 or er[-1] <= er[0] or ec[-1] <= ec[0]:
        return unavailable("Grid directions are degenerate")
    row_direction, column_direction = vr[:, -1], vc[:, -1]
    if row_direction[0] < 0:
        row_direction = -row_direction
    if column_direction[1] < 0:
        column_direction = -column_direction
    cross = (row_direction[0] * column_direction[1]
             - row_direction[1] * column_direction[0])
    if cross <= 1e-8:
        return unavailable("Grid directions are parallel or reversed")
    angle = np.degrees(np.arctan2(cross, row_direction @ column_direction))

    # Minimize v.T Sr v + u.T Sc u, where v is perpendicular to u.
    # All line intercepts were eliminated by centering each group above.
    _, vectors = np.linalg.eigh(sc - sr)
    u = vectors[:, 0]
    if u[0] < 0:
        u = -u
    v = np.array([-u[1], u[0]])
    orthogonal_sse = float(v @ sr @ v + u @ sc @ u)
    return {
        "valid": True,
        "spot_count": len(rxy),
        "num_rows": len(rows),
        "num_columns": len(columns),
        "angle_deg": float(angle),
        "shear_deg": float(90.0 - angle),
        "rotation_deg": float(np.degrees(np.arctan2(u[1], u[0]))),
        "orthogonal_rms_px": float(np.sqrt(max(0., orthogonal_sse) / len(rxy))),
        "line_rms_px": float(np.sqrt(max(0., er[0] + ec[0]) / len(rxy))),
        "row_direction": row_direction.tolist(),
        "column_direction": column_direction.tolist(),
    }
