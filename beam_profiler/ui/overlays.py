"""HUD, statistics bars and row/column visualisation drawn over the frame."""

from __future__ import annotations

import cv2
import numpy as np

from ..processing.spots import SpotArray
from ..roi import ViewTransform

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, YELLOW, CYAN, GREEN, RED = (
    (255, 255, 255),
    (255, 255, 0),
    (0, 255, 255),
    (0, 255, 0),
    (0, 0, 255),
)
LINE_H = 26


def text(img, s, xy, color=YELLOW, scale=0.7, thickness=2):
    cv2.putText(img, s, xy, FONT, scale, color, thickness, cv2.LINE_AA)


def hbar(img, x, y, value, factor, color=CYAN, max_len=100, height=20):
    """Filled horizontal bar with a white outline; length = value * factor."""
    length = int(min(value * factor, max_len))
    cv2.rectangle(img, (x, y), (x + length, y + height), color, -1)
    cv2.rectangle(img, (x, y), (x + max_len, y + height), WHITE, 2)


def centered_bar(img, x, y, value, factor, max_len=100, height=20):
    """Bar growing from the centre: green for positive, red for negative."""
    half = max_len // 2
    length = int(min(abs(value) * factor, half))
    cv2.rectangle(img, (x, y), (x + max_len, y + height), WHITE, 2)
    if value < 0:
        cv2.rectangle(img, (x + half - length, y), (x + half, y + height), RED, -1)
    else:
        cv2.rectangle(img, (x + half, y), (x + half + length, y + height), GREEN, -1)


def draw_hud(img, lines: list[str]) -> int:
    """Top-left HUD text block; returns the y coordinate after the last line."""
    y = 5
    for line in lines:
        y += LINE_H
        text(img, line, (10, y), WHITE)
    return y


def draw_rect_stats(img, stats: dict | None, y: int) -> int:
    """Live pixel statistics of the white rectangle; returns the next free y."""
    if stats is None:
        return y
    y += 24
    text(img, "=== Rect Stats ===", (10, y), GREEN)
    lines = [
        f"Sum: {stats['sum']:,}",
        f"Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}",
        f"Min: {stats['min']}, Max: {stats['max']}, Median: {stats['median']:.1f}",
        f"Size: {stats['width_pixels']}x{stats['height_pixels']} px "
        f"({stats['width_um']:.1f}x{stats['height_um']:.1f} um)",
        f"Total pixels: {stats['total_pixels']:,}",
    ]
    for line in lines:
        y += 24
        text(img, line, (10, y), CYAN, scale=0.6)
    return y + 24


def draw_spot_stats(
    img,
    y: int,
    std_dx: float,
    std_dy: float,
    mean_dx_um: float,
    mean_dy_um: float,
    curv_x: float,
    curv_y: float,
    sigma_std: float,
    show_curvature: bool,
) -> int:
    """Spot-spacing statistics: std bars, mean spacings, curvature and sigma."""
    bar_y = y + LINE_H
    text(img, f"std(x): {std_dx:.2f}", (10, bar_y + 15), scale=0.8)
    hbar(img, 170, bar_y, std_dx, factor=5)
    text(img, f"std(y): {std_dy:.2f}", (320, bar_y + 15), scale=0.8)
    hbar(img, 480, bar_y, std_dy, factor=5)
    text(img, f"Dx: {mean_dx_um:.1f} um", (630, bar_y + 15), scale=0.8)
    text(img, f"Dy: {mean_dy_um:.1f} um", (830, bar_y + 15), scale=0.8)

    if show_curvature:
        cy = bar_y + 30
        text(img, f"CurX: {curv_x:.2f}", (10, cy + 15), scale=0.8)
        centered_bar(img, 170, cy, curv_x, factor=2.5)
        text(img, f"CurY: {curv_y:.2f}", (320, cy + 15), scale=0.8)
        centered_bar(img, 480, cy, curv_y, factor=2.5)
        text(img, f"std(s): {sigma_std:.3f}", (800, cy + 15), scale=0.8)
        hbar(img, 960, cy, sigma_std, factor=20)
    return bar_y


def draw_grid_stats(img, bar_y: int, rows, columns, grid_stats: dict, pixel_to_um: float):
    """Row/column counts and average spacings (beam-array mode)."""
    if not grid_stats:
        return
    y = bar_y + 40
    text(img, f"NRow: {len(rows)}", (10, y), scale=0.8)
    r = grid_stats["rows"]
    if "avg_mean_dx" in r:
        text(
            img,
            f"Dx: {r['avg_mean_dx'] * pixel_to_um:.1f} um, "
            f"std(x): {r['avg_std_x'] * pixel_to_um:.1f} um, "
            f"std(y): {r['avg_std_y'] * pixel_to_um:.1f} um",
            (200, y),
            scale=0.8,
        )
    y += 30
    text(img, f"NCol: {len(columns)}", (10, y), scale=0.8)
    c = grid_stats["columns"]
    if "avg_mean_dy" in c:
        text(
            img,
            f"Dy: {c['avg_mean_dy'] * pixel_to_um:.1f} um, "
            f"std(y): {c['avg_std_y'] * pixel_to_um:.1f} um, "
            f"std(x): {c['avg_std_x'] * pixel_to_um:.1f} um",
            (200, y),
            scale=0.8,
        )


def draw_spots(img, spots, view: ViewTransform, pixel_size: float, min_label_radius: int = 15):
    """Draw fitted ellipses and labels in display space, so line widths and
    text stay the same size at every zoom level.  Labels are skipped for spots
    too small on screen to keep dense arrays readable."""
    BGR_GREEN, BGR_RED, BGR_BLUE = (0, 255, 0), (255, 0, 0), (0, 0, 255)
    if not isinstance(spots, SpotArray):
        spots = SpotArray.from_dicts(spots)
    # the per-spot geometry is computed for the whole array up front; only the
    # cv2 draw calls have to happen one spot at a time
    cx, cy = view.roi_to_display_many(spots.x, spots.y)
    a0, a1 = spots.sigma_0 * view.scale, spots.sigma_1 * view.scale
    angle = spots.angle
    axis0 = np.maximum(1, np.round(a0).astype(np.int32))
    axis1 = np.maximum(1, np.round(a1).astype(np.int32))
    labelled = a0 >= min_label_radius
    rad = np.radians(angle)
    major = (np.round(cx + a0 * np.cos(rad)).astype(np.int32),
             np.round(cy + a0 * np.sin(rad)).astype(np.int32))
    minor = (np.round(cx + a1 * np.cos(rad + np.pi / 2)).astype(np.int32),
             np.round(cy + a1 * np.sin(rad + np.pi / 2)).astype(np.int32))
    ty = cy + np.sqrt(a0 * a1).astype(np.int32) + 24
    um = pixel_size * 1e6
    for i in range(len(spots)):
        centre = (int(cx[i]), int(cy[i]))
        cv2.ellipse(
            img, centre, (int(axis0[i]), int(axis1[i])),
            float(angle[i]), 0, 360, BGR_GREEN, 2,
        )
        if not labelled[i]:
            continue
        cv2.line(img, centre, (int(major[0][i]), int(major[1][i])), BGR_RED, 2)
        cv2.line(img, centre, (int(minor[0][i]), int(minor[1][i])), BGR_BLUE, 2)
        label_y = int(ty[i])
        text(img, f"({spots.x[i]:.0f}, {spots.y[i]:.0f})", (centre[0], label_y),
             BGR_GREEN, scale=0.5, thickness=1)
        text(img, f"s0={spots.sigma_0[i] * um:.1f} um", (centre[0], label_y + 18),
             BGR_GREEN, scale=0.5, thickness=1)
        text(img, f"s1={spots.sigma_1[i] * um:.1f} um", (centre[0], label_y + 36),
             BGR_GREEN, scale=0.5, thickness=1)


def _hue_color(i: int, saturation: int) -> list:
    hsv = np.uint8([[[(i * 30) % 180, saturation, 255]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()


def _dashed_line(img, p1, p2, color, dash=8, gap=4):
    dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    if dist == 0:
        return
    ux, uy = (p2[0] - p1[0]) / dist, (p2[1] - p1[1]) / dist
    pos = 0.0
    while pos < dist:
        end = min(pos + dash, dist)
        cv2.line(
            img,
            (int(p1[0] + ux * pos), int(p1[1] + uy * pos)),
            (int(p1[0] + ux * end), int(p1[1] + uy * end)),
            color,
            2,
            cv2.LINE_AA,
        )
        pos = end + gap


def draw_row_col(img, view: ViewTransform, rows, columns):
    """Hue-coded row lines (solid, circles) and column lines (dashed, squares)."""
    for i, row in enumerate(rows):
        if len(row) < 2:
            continue
        color = _hue_color(i, saturation=255)
        mean_y = float(np.mean(row.y))
        p1 = view.roi_to_display(float(row.x.min()), mean_y)
        p2 = view.roi_to_display(float(row.x.max()), mean_y)
        cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
        for px, py in zip(*view.roi_to_display_many(row.x, row.y)):
            cv2.circle(img, (int(px), int(py)), 5, color, 2)

    for i, col in enumerate(columns):
        if len(col) < 2:
            continue
        color = _hue_color(i, saturation=180)
        mean_x = float(np.mean(col.x))
        p1 = view.roi_to_display(mean_x, float(col.y.min()))
        p2 = view.roi_to_display(mean_x, float(col.y.max()))
        _dashed_line(img, p1, p2, color)
        for px, py in zip(*view.roi_to_display_many(col.x, col.y)):
            cv2.drawMarker(img, (int(px), int(py)), color, cv2.MARKER_SQUARE, 10, 2)
