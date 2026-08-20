"""Classify detected spots into rows and columns of a beam array."""

from __future__ import annotations

import numpy as np


def _cluster_1d(values: np.ndarray, eps: float | None, min_samples: int) -> list[np.ndarray]:
    """Group indices of `values` by splitting sorted values at large gaps.
    With eps=None the split threshold adapts to the gap distribution (Tukey-style
    outlier fence, floored at 2x the median gap)."""
    order = np.argsort(values)
    gaps = np.diff(values[order])
    if eps is not None:
        threshold = eps
    elif len(gaps):
        q75, q25 = np.percentile(gaps, [85, 15])
        threshold = max(q75 + (70 / 15) * (q75 - q25), 2.0 * np.median(gaps))
    else:
        threshold = 1.0

    groups, start = [], 0
    for b in np.where(gaps > threshold)[0]:
        if b + 1 - start >= min_samples:
            groups.append(order[start : b + 1])
        start = b + 1
    if len(values) - start >= min_samples:
        groups.append(order[start:])
    return groups


def classify_grid(spots: list[dict], eps: float | None = None, min_samples: int = 2):
    """Split spots into rows (clustered by y) and columns (clustered by x).

    Returns (rows, columns, stats) where stats holds per-row spacing/straightness
    figures (mean_dx, std_x, std_y for rows; mean_dy, std_y, std_x for columns)
    plus their averages under avg_* keys.  All distances are in pixels."""
    if not spots:
        return [], [], {}

    xs = np.array([s["x"] for s in spots])
    ys = np.array([s["y"] for s in spots])
    columns = [
        sorted((spots[i] for i in g), key=lambda s: s["y"])
        for g in _cluster_1d(xs, eps, min_samples)
    ]
    rows = [
        sorted((spots[i] for i in g), key=lambda s: s["x"])
        for g in _cluster_1d(ys, eps, min_samples)
    ]

    stats = {
        "rows": {"mean_dx": [], "std_x": [], "std_y": []},
        "columns": {"mean_dy": [], "std_y": [], "std_x": []},
    }
    for row in rows:
        if len(row) >= 2:
            rx = np.sort([s["x"] for s in row])
            dxs = np.diff(rx)
            stats["rows"]["mean_dx"].append(float(dxs.mean()))
            stats["rows"]["std_x"].append(float(dxs.std()))
            stats["rows"]["std_y"].append(float(np.std([s["y"] for s in row])))
    for col in columns:
        if len(col) >= 2:
            cy = np.sort([s["y"] for s in col])
            dys = np.diff(cy)
            stats["columns"]["mean_dy"].append(float(dys.mean()))
            stats["columns"]["std_y"].append(float(dys.std()))
            stats["columns"]["std_x"].append(float(np.std([s["x"] for s in col])))

    for axis, keys in (
        ("rows", ("mean_dx", "std_x", "std_y")),
        ("columns", ("mean_dy", "std_y", "std_x")),
    ):
        for key in keys:
            if stats[axis][key]:
                stats[axis][f"avg_{key}"] = float(np.mean(stats[axis][key]))

    return rows, columns, stats
